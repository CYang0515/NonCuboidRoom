import copy
import os
import random

import cv2
import numpy as np
import torch
import yaml
import argparse
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

from datasets import SUNRGBD, Structured3D
from models import (AverageMeter, Detector, Loss, evaluate, get_optimizer,
                    gt_check, printfs, post_process)

best_acc = 0
best_info = ''


def train(model, criterion, dataloader, dataloader_val, optimizer, scheduler, cfg, device):
    writer_train, writer_val, logger = printfs(cfg)
    accumulators_train = [AverageMeter() for _ in range(11)]
    accumulators_val = [AverageMeter() for _ in range(11)]
    for epoch in range(cfg.epochs):
        # run one epoch train
        run_train(model, criterion, optimizer, dataloader,
                  accumulators_train, logger, writer_train, epoch, device, cfg)
        # run one validation
        run_val(model, criterion, dataloader_val,
                accumulators_val, logger, writer_val, epoch, device, cfg)
        # adjust lr
        scheduler.step()


def run_train(model, criterion, optimizer, dataloader, accumulators, logger, writer, epoch, device, cfg):
    for accumulator in accumulators:
        accumulator.reset()
    model.train()
    for iters, inputs in enumerate(dataloader):
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        # forward
        x = model(inputs['img'])
        loss, loss_stats = criterion(x, **inputs)
        # optmizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logger records
        printf = f'{iters}/{len(dataloader)}:{epoch}/{cfg.epochs}::' + \
                 '__'.join(f'{key}:{value.data:.4f}' for key,
                           value in loss_stats.items())
        logger.info(printf)

        # write tensrboardx for steps
        for i, (key, value) in enumerate(loss_stats.items()):
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            writer.add_scalar(f'iter/{key}', value.data,
                              epoch * len(dataloader) + iters)
            accumulators[i].update(key, value.data)
    for accumulator in accumulators:
        writer.add_scalar('epoch/' + accumulator.name, accumulator.avg, epoch)


def run_val(model, criterion, dataloader, accumulators, logger, writer, epoch, device, cfg):
    global best_acc
    global best_info
    dts_planes = []
    dts_lines = []
    gts_planes = []
    gts_lines = []
    for accumulator in accumulators:
        accumulator.reset()
    model.eval()
    for iters, inputs in enumerate(dataloader):
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        # forward
        x = model(inputs['img'])
        loss, loss_stats = criterion(x, **inputs)

        # post process
        # parse predict plane and line results
        dt_planes, dt_lines, dt_params3d, _ = post_process(x)
        # parse gt plane and line results to evaluate model roughly.
        gt_planes, gt_lines, gt_params3d = gt_check(inputs)
        # collect results
        dts_planes.extend(dt_planes)  # each img topk dt planes
        gts_planes.extend(gt_planes)
        dts_lines.extend([dt[dt[:, 3] == 1] for dt in dt_lines])  # each img has variable number of dt lines
        gts_lines.extend([gt[gt[:, 3] == 1] for gt in gt_lines])

        for i, (key, value) in enumerate(loss_stats.items()):
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            accumulators[i].update(key, value.data)

    for accumulator in accumulators:
        writer.add_scalar('epoch/' + accumulator.name, accumulator.avg, epoch)

    # evaluate
    mAR_p, mAP_p, mAR_l, mAP_l = evaluate(dts_planes, dts_lines, gts_planes, gts_lines)
    writer.add_scalar('epoch/mAR_p', mAR_p, epoch)
    writer.add_scalar('epoch/mAP_p', mAP_p, epoch)
    writer.add_scalar('epoch/mAR_l', mAR_l, epoch)
    writer.add_scalar('epoch/mAP_l', mAP_l, epoch)

    # save model
    if epoch % 10 == 0:
        if not os.path.isdir(f'./checkpoints/checkpoints_{cfg.model_name}'):
            os.makedirs(f'./checkpoints/checkpoints_{cfg.model_name}')
        if cfg.num_gpus > 1:
            torch.save(model.module.state_dict(),
                    f'./checkpoints/checkpoints_{cfg.model_name}/{epoch}.pt')
        else:
            torch.save(model.state_dict(),
                    f'./checkpoints/checkpoints_{cfg.model_name}/{epoch}.pt')

    # save best model
    if (mAP_p + mAP_l) > best_acc:
        best_acc = mAP_p + mAP_l
        best_info = f'mAR_p:{mAR_p},mAP_p:{mAP_p},mAR_l:{mAR_l},mAP_l:{mAP_l},epoch:{epoch},best_acc:{best_acc}'
        if not os.path.isdir(f'./checkpoints/checkpoints_{cfg.model_name}'):
            os.makedirs(f'./checkpoints/checkpoints_{cfg.model_name}')
        if cfg.num_gpus > 1:
            torch.save(model.module.state_dict(),
                   f'./checkpoints/checkpoints_{cfg.model_name}/best.pt')
        else:
            torch.save(model.state_dict(),
                    f'./checkpoints/checkpoints_{cfg.model_name}/best.pt')
        logger.info(f'best_acc:{best_acc}, info:{best_info}')

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='s3d', required=True, help='the model name')
    parser.add_argument('--data', type=str, default='Structured3D', choices=['Structured3D', 'SUNRGBD'])
    parser.add_argument('--pretrained', type=str, default=None, help='the pretrained model')

    parser.add_argument('--split', type=str, default='all', choices=['all', 'nyu'], help='the training set for SUNRGBD')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=51)
    parser.add_argument('--lr_step', type=list, default=[30, 40])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        config = yaml.load(f)
        cfg = EasyDict(config)
    args = parse()
    cfg.update(vars(args))
    random.seed(123)
    torch.manual_seed(123)
    np.random.seed(123)
    torch.cuda.manual_seed(123)

    #  dataset
    if cfg.data == 'Structured3D':
        dataset = Structured3D(cfg.Dataset.Structured3D, 'training')
        dataset_val = Structured3D(cfg.Dataset.Structured3D, 'validation')
    elif cfg.data == 'SUNRGBD':
        dataset = SUNRGBD(cfg.Dataset.SUNRGBD, 'train', cfg.split)
        dataset_val = SUNRGBD(cfg.Dataset.SUNRGBD, 'test', cfg.split)
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4)

    # create network
    model = Detector()
    # compute loss
    criterion = Loss(cfg.Weights)

    # resume checkpoints
    if cfg.pretrained is not None:
        state_dict = torch.load(cfg.pretrained,
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    # set data parallel
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    # optimizer
    optimizer = get_optimizer(model.parameters(), cfg.Solver)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.lr_step)

    train(model, criterion, dataloader, dataloader_val,
          optimizer, scheduler, cfg, device)
