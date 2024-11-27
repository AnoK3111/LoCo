import numpy as np
import logging
import os
import cv2
import torch
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import torch.nn as nn
from torchvision import transforms

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):

    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss
    

class FocalDiceloss(nn.Module):

    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        """
        pred: [B, C, H, W]
        mask: [B, C, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        return loss1

class SAM_semi_supervised_loss(nn.Module):
    def __init__(self, conf_thresh, classes=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.conf_thresh = conf_thresh
        self.classes = classes
        self.eps = 1e-8

    def forward(self, cnn_pred, sam_pred, threshold):
        mask = sam_pred.argmax(dim=1).detach()
        cla = cnn_pred.argmax(dim=1).reshape(cnn_pred.size(0), -1).detach()
        class_num = []
        for c in range(1, self.classes+1):
            class_num.append((cla == c).sum(dim=1).unsqueeze(1))
        mask_class = (torch.concat(class_num, dim=1).argmax(dim=1) + 1).reshape(cnn_pred.size(0), 1, 1)
        loss = self.ce_loss(cnn_pred, mask_class * mask)
        # cnn_conf = cnn_pred.softmax(dim=1).max(1)[0].detach()
        sam_conf = sam_pred.softmax(dim=1).max(1)[0].detach()
        # conf_mask = torch.minimum(cnn_conf, sam_conf)
        # conf_mask *= (((mask_class * mask) == cnn_pred.argmax(dim=1).detach()) | (sam_conf > cnn_conf))
        loss *= (sam_conf >= threshold)
        return loss
    

# class Adaptive_threshold:
#     def __init__(self, classes, device, lam1=0.99,lam2=0.99):
#         self.lam1 = lam1
#         self.lam2 = lam2
#         self.classes = classes
#         self.global_threshold = torch.full((3,), 1.0 / classes).to(device)
#         self.classes_conf_list = [torch.full((1,), 1.0 / classes).to(device) for _ in range(classes)]

#     def g_threshold(self):
#         return self.global_threshold.mean()


#     def update(self, x):
#         # global
#         all_conf = x.detach().clone().softmax(dim=1).max(dim=1)[0] # (b, c ,)
#         all_mask = x.detach().argmax(dim=1)
#         global_conf = all_conf.reshape(x.size(0),-1)
#         global_conf = global_conf.mean(dim=1)
#         global_conf  = global_conf.mean(dim=0)
#         self.global_threshold = self.global_threshold * self.lam1 + (1 - self.lam1) * global_conf

#         # local
#         for c in range(self.classes):
#             if (all_mask == c).sum() == 0:
#                 label_confidences = 1 / self.classes
#             else:
#                 label_confidences = all_conf[all_mask == c].mean()
#             self.classes_conf_list[c] = self.classes_conf_list[c] * self.lam2 + (1 - self.lam2) * label_confidences

#     def get_threshold(self, class_mask,epoch):
#         conf = torch.concatenate(self.classes_conf_list, dim=0)
#         conf =  conf / conf.max() * self.global_threshold
#         res = torch.ones_like(class_mask).float()
#         res[class_mask == c] = conf[c]

#         return res

class Adaptive_threshold:
    def __init__(self, classes, device='cpu', lam=0.999):
        self.classes = classes
        self.device = device
        self.lam = lam
        self.global_threshold = (1 / classes)

    def set_threshold(self, t):
        self.global_threshold = t

    def update(self, x, ignore_mask=None):
        all_conf = x.detach().clone().softmax(dim=1).max(dim=1)[0]
        all_mask = x.detach().clone().argmax(dim=1)
        cls_num =self.classes
        global_t = 0
        for c in range(self.classes):
            cls_map = (all_mask == c)
            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            cls_t = all_conf[cls_map].mean()
            global_t += cls_t

        global_t /= cls_num
        self.global_threshold = self.global_threshold * self.lam + (1 - self.lam) * global_t



            

            

        



    
class light_transform:
    def __init__(self, device="cpu") -> None:
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.wl_mean = norm(torch.tensor([203.55483451, 129.55401181, 125.14028332]).reshape(3, 1, 1).to(device) / 255.0)
        self.nbi_mean = norm(torch.tensor([109.74046222, 119.49505484, 108.43316262]).reshape(3, 1, 1).to(device) / 255.0)
        self.device = device

    def transform(self, x, l_label):
        x = x.clone()
        u = x.mean(dim=(2, 3), keepdim=True)
        target_u = torch.zeros((x.size(0), 3, 1, 1)).to(self.device)
        target_u[l_label == 0] = self.nbi_mean
        target_u[l_label == 1] = self.wl_mean
        x = x - u + target_u
        return x
    

def kl_loss(branch1, branch2):
    classes = branch1.size(1)
    branch1 = branch1.transpose(1, 3).contiguous().view(-1, classes)
    branch2 = branch2.transpose(1, 3).contiguous().view(-1, classes)
    b1_loss = F.kl_div(F.log_softmax(branch1, dim=1), F.softmax(branch2, dim=1), reduction='none')
    b2_loss = F.kl_div(F.log_softmax(branch2, dim=1), F.softmax(branch1, dim=1), reduction='none')

    b1_loss = b1_loss.mean()
    b2_loss = b2_loss.mean()
    loss = (b1_loss + b2_loss) / 2
    return loss


if __name__=="__main__":
    at = Adaptive_threshold(3)
    x = torch.rand(2, 3, 256, 256)
    print(x)
    at.update(x)

    

    




