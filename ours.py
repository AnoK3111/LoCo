import argparse
import logging
import os
import pprint
import time

import torch
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
import yaml


from ours.dataset.semi import SemiDataset
from ours.model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from classes import CLASSES
from ours.util.utils import count_params, init_log, AverageMeter, Adaptive_threshold, Balanced_adaptive_threshold
from ours.util.dist_helper import setup_distributed, AllGather
from ours.util.ohem import ProbOhemCrossEntropy2d
from ours.util.losses import ContrastLoss, Feature_Consistency_Loss
from ours.util.bank import Bank



import warnings 
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    allgather = AllGather.apply
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False)
    
    # Teacher model -- freeze training
    model_teacher = DeepLabV3Plus(cfg)
    model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    for p in model_teacher.parameters():
        p.requires_grad = False
    # initialize teacher model -- not neccesary if using warmup
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    ada_thresh = Balanced_adaptive_threshold(cfg['nclass'], 'cuda', 0.999, init_threshold=cfg["init_threshold"])

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    criterion_u = nn.CrossEntropyLoss(reduction='none', ignore_index=255).cuda(local_rank)
    criterion_c = ContrastLoss(cfg).cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=6, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=6, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
            
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_f = AverageMeter()
        total_threshold = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        for step, ((img_x, mask_x),
            (img_u_w, img_u_s, ignore_mask, cutmix_box)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            
            i_iter = epoch * len(trainloader_l) + step
            # start the training
            # 1. generate pseudo labels
            with torch.no_grad():
                model_teacher.eval()
                logit_u_w_t, proj_u_w_t = model_teacher(img_u_w.detach(), need_projection=True)
                logit_u_w_t = logit_u_w_t.detach()
                conf_u_w_t = logit_u_w_t.softmax(dim=1).max(dim=1)[0]
                mask_u_w_t = logit_u_w_t.argmax(dim=1)

            model.train()
            # 2. apply cutmix
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_x[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = \
                mask_u_w_t.clone(), conf_u_w_t.clone(), ignore_mask.clone()
            mask_u_w_cutmixed[cutmix_box == 1] = mask_x[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = 1.0
            ignore_mask_cutmixed[(mask_x == 255) & (cutmix_box == 1)] = 255


            # 3. forward concate labeled + unlabeld into student networks
            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]

            logits, projections = model(torch.cat((img_x, img_u_w, img_u_s)), need_projection=True)
            logit_x, logit_u_w, logit_u_s = logits.chunk(3)
            proj_x, proj_u_w, proj_u_s = projections.chunk(3)


            # 4. update threshold
            all_logit_u_w_t = allgather(logit_u_w_t)
            all_ignore_mask = allgather(ignore_mask)
            ada_thresh.update(all_logit_u_w_t, all_ignore_mask)
            threshold = ada_thresh.get_thres()

            # 5. supervised loss
            loss_x = criterion_l(logit_x, mask_x)

            # 6. unsupervised loss
            loss_u_s = criterion_u(logit_u_s, mask_u_w_cutmixed)
            threshold_tensor = torch.ones_like(conf_u_w_t)
            for c in mask_u_w_cutmixed.unique():
                if c == 255:
                    continue
                threshold_tensor[mask_u_w_cutmixed == c] = threshold[c]
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= threshold_tensor) & (ignore_mask_cutmixed != 255))
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()

            # 7. contrastive loss
            all_proj_x = allgather(proj_x)
            all_mask_x = allgather(mask_x)
            all_proj_u_w = allgather(proj_u_w)
            loss_c = criterion_c(all_proj_x, 
                                 all_mask_x, 
                                 all_proj_u_w, 
                                 all_logit_u_w_t, 
                                 all_ignore_mask, 
                                 threshold if cfg["use_ada_threshold"] else cfg["conf_thresh"], True, True)
            

            loss = (loss_x + 0.5 * loss_u_s + cfg['contrastive_weight'] * loss_c) / (1.5 + cfg['contrastive_weight'])

            # update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update teacher model with EMA
            with torch.no_grad():
                ema_decay = min(1 - 1 / (i_iter + 1), cfg["ema"])

                # update weight
                for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                    param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
                # update bn
                for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                    buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                    # buffer_eval.data = buffer_train.data

            lr = cfg['lr'] * (1 - i_iter / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_c.update(loss_c.item())
            total_threshold.update(threshold.cpu().numpy())
            for c in mask_u_w_t.unique():
                threshold_tensor[mask_u_w_t == c] = threshold[c]
            mask_ratio = ((conf_u_w_t >= threshold_tensor) & (ignore_mask != 255)).sum().item() / \
                         (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            if (step % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss c: {:.3f}, Threshold: {}, Mask ratio: '
                    '{:.3f}'.format(step, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_c.avg,
                                    total_threshold.avg, total_mask_ratio.avg))
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))





if __name__ == "__main__":
    main()