import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(proj, label, conf=None):
    proj = proj.permute(0, 2, 3, 1)
    proj = proj.contiguous().view(-1, proj.size(-1))
    label = label.contiguous().view(-1)
    if conf is not None:
        conf = conf.contiguous().view(-1)
        return proj, label, conf
    return proj, label

def k_nearest(k):
    queue = []
    k_sqrt  =  math.ceil(math.sqrt(k))
    for x in range(-k_sqrt, k_sqrt + 1):
        for y in range(-k_sqrt, k_sqrt + 1):
            queue.append((x, y))

    return sorted(queue, key=lambda x: x[0] * x[0] + x[1] * x[1])[1: k + 1]

def inter_class_selection(proj, label, prototype_x, prototype_y, sampling_rate=0.1):
    normed_proj = F.normalize(proj, dim=1).detach()
    x_list = []
    y_list = []
    total_sample = label.size(0)
    select_sample = int(total_sample * sampling_rate / prototype_y.size(0))
    for i, c in enumerate(prototype_y):
        proj_dis = (normed_proj[label == c] @ F.normalize(prototype_x[i].detach(), dim=0).unsqueeze(-1)).reshape(-1)
        _, index = torch.sort(proj_dis)
        x_list.append(proj[label == c][index[:select_sample]])
        y_list.append(torch.ones(index[:select_sample].size(0), dtype=torch.long, device=label.device) * c)
    
    proj = torch.cat(x_list)
    label = torch.cat(y_list)
    return proj, label

def is_boundary(mask, neighbors=[(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]):
    B, H, W = mask.size()
    mask = mask.clone()
    padding_size = max([max(a, b) for a, b in neighbors])
    padded_feature_map = F.pad(mask, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
    boundary = torch.ones(B, H, W, device=mask.device)

    for i, neighbor in enumerate(neighbors):
        top, left = neighbor[0] + padding_size, neighbor[1] + padding_size
        neighbor_label = padded_feature_map[:, top: top + H, left: left + W]
        boundary *= (neighbor_label == mask)
    return 1 - boundary

def boundary_selection(proj_l, proj_u, label_l, pred_u, threshold, sampling_rate=0.25, neighbors=k_nearest(64)):
    batch_size = proj_l.size(0)
    device = proj_l.device
    size = label_l.size(-1)
    cls_num = pred_u.size(1)
    x_list = []
    y_list = []
    label_u = pred_u.argmax(1)
    conf_u = pred_u.softmax(dim=1).max(dim=1)[0]
    for c in range(cls_num):
        label_u[(label_u == c) * (conf_u <= threshold[c])] = 255
    # if neighborhood == "8":
    #     conv_kernel = torch.tensor([
    #                         [1, 1, 1],
    #                         [1, -8, 1],
    #                         [1, 1, 1]], dtype=torch.float, device=device).view(1, 1, 3, 3)
    # else:
    #     conv_kernel = torch.tensor([
    #                         [0, 1, 0],
    #                         [1, -4, 1],
    #                         [0, 1, 0]], dtype=torch.float, device=device).view(1, 1, 3, 3)
    # bs_l = F.conv2d(label_l.unsqueeze(1).float(), conv_kernel, padding=1).view(batch_size, size, size).long() != 0
    # bs_u = F.conv2d(label_u.unsqueeze(1).float(), conv_kernel, padding=1).view(batch_size, size, size).long() != 0
    bs_l = is_boundary(label_l, neighbors).bool()
    bs_u = is_boundary(label_u, neighbors).bool()
    dis_l = neighborhood_similarity(proj_l, neighbors).permute(0, 2, 3, 1).min(dim=-1)[0][bs_l]
    dis_u = neighborhood_similarity(proj_u, neighbors).permute(0, 2, 3, 1).min(dim=-1)[0][bs_u]
    proj_l = proj_l.clone().permute(0, 2, 3, 1)[bs_l]
    proj_u = proj_u.clone().permute(0, 2, 3, 1)[bs_u]
    label_l = label_l[bs_l]
    label_u = label_u[bs_u]
    for c in range(1, cls_num):
        proj_c_l = proj_l[label_l == c]
        dis_c_l = dis_l[label_l == c]
        proj_c_u = proj_u[label_u == c]
        dis_c_u = dis_u[label_u == c]
        num_l = int(dis_c_l.size(0) * sampling_rate)
        num_u = int(dis_c_u.size(0) * sampling_rate)
        aaa, index_l = torch.sort(dis_c_l)
        bbb, index_u = torch.sort(dis_c_u)
        x_list.append(proj_c_l[index_l][:num_l])
        x_list.append(proj_c_u[index_u][:num_u])
        y_list.append(torch.ones(index_l[:num_l].size(0) + index_u[:num_u].size(0), dtype=torch.long, device=device) * c)

    proj = torch.cat(x_list)
    label = torch.cat(y_list)
    return proj, label
    

def unlabeled_filter(proj, label, conf, threshold):
    threshold_tensor = torch.ones_like(label, device=label.device, dtype=torch.float32)
    for c in label.unique():
        threshold_tensor[label == c] = threshold[c]
    indices = torch.nonzero(conf > threshold_tensor).view(-1)
    proj = proj[indices, :]
    label = label[indices]

    return proj, label

# def neighborhood_similarity(x):
#     neighbor = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
#     B, C, H, W = x.size()
#     x = x.clone()
#     x = F.normalize(x, dim=1)
#     padded_feature_map = F.pad(x, (1, 1, 1, 1), mode='replicate')
#     cosine_similarities = torch.zeros(B, 8, H, W, device=x.device)

#     for i, neighbor in enumerate(neighbor):
#         top, left = neighbor[0] + 1, neighbor[1] + 1
#         neighbor_feature = padded_feature_map[:, :, top: top + H, left: left + W]
#         similarity = (x * neighbor_feature).sum(dim=1)
#         cosine_similarities[:, i, :, :] = similarity
#     return cosine_similarities

def neighborhood_similarity(x, neighbors=[(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]):
    B, C, H, W = x.size()
    x = x.clone()
    x = F.normalize(x, dim=1)
    padding_size = max([max(a, b) for a, b in neighbors])
    padded_feature_map = F.pad(x, (padding_size, padding_size, padding_size, padding_size), mode='replicate')
    cosine_similarities = torch.zeros(B, len(neighbors), H, W, device=x.device)

    for i, neighbor in enumerate(neighbors):
        top, left = neighbor[0] + padding_size, neighbor[1] + padding_size
        neighbor_feature = padded_feature_map[:, :, top: top + H, left: left + W]
        similarity = (x * neighbor_feature).sum(dim=1)
        cosine_similarities[:, i, :, :] = similarity
    return cosine_similarities

def get_index_matrix(size):
    matrix = torch.zeros(2, size, size)
    x_indices, y_indices = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='xy')
    matrix[0] = y_indices
    matrix[1] = x_indices
    return matrix


class ContrastLoss(nn.Module):
    def __init__(self, cfg, bank=None, temperature=0.1, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.bank = bank
        self.temperature = temperature
        self.base_temperature = temperature
        self.eps = eps

    def contrast_loss(self, x, y, x_anchor, y_anchor):
        anchor_num = y_anchor.size(0)
        sample_num = y.size(0)
        device = x.device
        x = F.normalize(x, dim=1)
        x_anchor = F.normalize(x_anchor, dim=1)
        y = y.long().contiguous().view(-1, 1)
        y_anchor = y_anchor.long().contiguous().view(-1, 1)

        mask = torch.eq(y_anchor, y.T).float().to(device)
        
        anchor_dot_contrast = torch.matmul(x_anchor, x.T)
        anchor_dot_contrast_t = torch.div(anchor_dot_contrast, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast_t, dim=1, keepdim=True)
        logits = anchor_dot_contrast_t - logits_max.detach()
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask)
        logits_mask[1 - anchor_dot_contrast <= self.eps] = 0

        mask = mask * logits_mask
        
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
    
    def forward(self, proj_l, label_l, proj_u, pred_u, ignore_mask_u, threshold, rs=True, bs=True):
        device = label_l.device
        label_l = label_l.clone()
        pred_u = pred_u.clone()
        h, w = proj_l.shape[-2:]
        
        unique_cls = torch.unique(label_l)
        unique_cls = unique_cls[unique_cls != 255]
        x_prototype = torch.zeros(len(unique_cls), proj_l.size(1), dtype=torch.float32).to(device)
        y_prototype = torch.zeros(len(unique_cls), dtype=torch.long).to(device)
        conf_proj = proj_l.clone().permute(0, 2, 3, 1).reshape(-1, proj_l.size(1))
        for i, c in enumerate(unique_cls):
            x_prototype[i] = conf_proj[label_l.reshape(-1) == c].mean(dim=0)
            y_prototype[i] = c
        # label_u = pred_u.argmax(dim=1)
        # label_all = torch.cat([label_l, label_u])
        # proj_all = torch.cat([proj_l, proj_u])
        # unique_cls = torch.unique(label_all)
        # unique_cls = unique_cls[unique_cls != 255]
        # x_prototype = torch.zeros(len(unique_cls), proj_all.size(1), dtype=torch.float32).to(device)
        # y_prototype = torch.zeros(len(unique_cls), dtype=torch.long).to(device)
        # conf_proj = proj_all.clone().permute(0, 2, 3, 1).reshape(-1, proj_all.size(1))
        
        # for i, c in enumerate(unique_cls):
        #     x_prototype[i] = conf_proj[label_all.reshape(-1) == c].mean(dim=0)
        #     y_prototype[i] = c


        ignore_mask_l = torch.zeros_like(ignore_mask_u)
        ignore_mask_l[label_l == 255] = 255

        conf_u = pred_u.softmax(dim=1).max(dim=1)[0]
        label_u = pred_u.argmax(dim=1)
        conf_u[ignore_mask_u == 255] = 0.

        if isinstance(threshold, float):
            threshold = torch.ones((pred_u.size(1), ), device=device, dtype=torch.float32) * threshold
        
        proj_l_f, label_l_f = flatten(proj_l, label_l)
        proj_u_f, label_u_f, conf_u_f = flatten(proj_u, label_u, conf_u)
        x = torch.zeros((0, proj_l_f.size(1)), device=device, dtype=torch.float)
        y = torch.zeros((0, ), device=device, dtype=torch.long)
        proj_u_f, label_u_f = unlabeled_filter(proj_u_f, label_u_f, conf_u_f, threshold)
        if rs:
            rs_proj_l, rs_label_l = inter_class_selection(proj_l_f, label_l_f, x_prototype, y_prototype, self.cfg["rs_sampling_rate"])
            rs_proj_u, rs_label_u = inter_class_selection(proj_u_f, label_u_f, x_prototype, y_prototype, self.cfg["rs_sampling_rate"])
            x = torch.cat([x, rs_proj_l, rs_proj_u], dim=0).contiguous()
            y = torch.cat([y, rs_label_l, rs_label_u], dim=0).contiguous()
        if bs:
            bs_proj, bs_label = boundary_selection(proj_l, proj_u, label_l, pred_u, threshold, self.cfg["bs_sampling_rate"])
            x = torch.cat([x, bs_proj], dim=0).contiguous()
            y = torch.cat([y, bs_label], dim=0).contiguous()
        if not rs and not bs:
            x = torch.cat([proj_l_f, proj_u_f], dim=0).contiguous()
            y = torch.cat([label_l_f, label_u_f], dim=0).contiguous()
        if x.size(0) == 0:
            return 0 * proj_l.mean()

        # if self.bank:
        #     self.bank.update(x, y)
        #     x_prototype, y_prototype = self.bank.get_prototype()
        # else:
        #     unique_cls = torch.unique(y)
        #     x_prototype = torch.zeros(len(unique_cls), x.size(-1)).to(device)
        #     y_prototype = torch.zeros(len(unique_cls),).to(device)
        #     for i, c in enumerate(unique_cls):
        #         # cla_x = x[y == c]
        #         # cla_conf = conf[y == c]
        #         # cla_conf /= cla_conf.sum()
        #         # cla_conf = cla_conf.unsqueeze(-1).expand(cla_x.shape)
        #         # x_prototype[i] = (cla_x * cla_conf).sum(dim=0)
        #         print(x.shape, y==c)
        #         x_prototype[i] = x[y == c].mean(dim=0)
        #         y_prototype[i] = c

        return self.contrast_loss(x, y, x_prototype, y_prototype)
    
class Feature_Consistency_Loss(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    @staticmethod
    def cos_similarity(x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        return (x1 * x2).sum(dim=1)

    def forward(self, x, target):
        target = target.detach()
        size = self.cfg["crop_size"] // self.cfg["patch_size"]
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)
        target = F.interpolate(target, size=(size, size), mode="bilinear", align_corners=True)
        target = target.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        channel_size = x.size(-1)
        target = target.view(-1, channel_size)
        x = x.view(-1, channel_size)
        loss = self.cos_similarity(x, target)
        return 1 - loss.mean()

        

    

if __name__=="__main__":
    cfg = {}
    cfg['conf_thresh'] = 0.95
    cfg['nclass'] = 3
    loss = ContrastLoss(cfg, None)
    proj_l = torch.rand(8, 64, 40, 40, requires_grad=True).cuda()
    proj_u = torch.rand(8, 64, 40, 40, requires_grad=True).cuda()
    label_l = torch.randint(0, 3, (8, 321, 321)).cuda()
    label_l[:, 0:100, 0:100] = 255
    pred_u = torch.rand(8, 3, 321, 321).cuda()
    print(loss(proj_l, label_l, proj_u, pred_u, torch.zeros_like(label_l), torch.tensor([0.1, 0.3, 0.5])))