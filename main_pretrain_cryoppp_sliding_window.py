# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

# cryoppp用

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch
from cryoppp_pretrain_sliding_window import CryoPPPDataset

import wandb  # wandbをインポート


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--root_path',default='../../../ssd2/riku/cryoppp/', type=str,
                        help='Root path for datasets')
    parser.add_argument('--data_ids', type=str, nargs='+', default=10005,
                        help='List of dataset IDs to use (e.g., 10005 10075). Use "all" to include all datasets.')

    parser.add_argument('--output_dir', default='./output_cryoppp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_cryoppp',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    # wandbの初期化
    if misc.is_main_process():
        wandb.init(project="2dmae_cryoppp-sliding-window", config=vars(args))
        wandb.run.name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform = transforms.Compose([
            # transforms.Resize(224), # Resize to 224 for MAE
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]) # CIFAR-10 mean and std
        ])

    if args.data_ids == ['all']:
        args.data_ids = [
            '10005', '10075', '10184', '10387', '10532', '10737', '11051',
            '10017', '10077', '10240', '10576', '10760', '11056',
            '10028', '10081', '10289', '10406', '10590', '10816', '11057',
            '10059', '10093', '10291', '10444', '10669', '10852', '11183',
            '10061', '10096', '10345', '10526', '10671', '10947'
        ] # 10389は構造が違って面倒なので一旦除外
    else:
        if isinstance(args.data_ids, int):
            args.data_ids = [str(args.data_ids)]
        else:
            args.data_ids = [str(data_id) for data_id in args.data_ids]

    dataset_paths = [os.path.join(args.root_path, data_id) for data_id in args.data_ids]
    datasets = []
    for path in dataset_paths:
        datasets.append(CryoPPPDataset(os.path.join(path, 'micrographs'), transform=transform))
    dataset = torch.utils.data.ConcatDataset(datasets)
    print(dataset)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()  # global_rankを初期化
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("sampler = %s" % str(sampler))
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, weight_decay=args.weight_decay)
    # 手動でパラメーターグループを作成
    param_groups = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "bias" not in n and "norm" not in n], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "bias" in n or "norm" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    log_stats = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, 
            dataloader,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats.append({**{f'train_{k}': v for k, v in train_stats.items()},
                          'epoch': epoch,})

        # wandbにログを記録
        if misc.is_main_process():
            wandb.log(log_stats[-1])

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats[-1]) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # wandbの終了
    if misc.is_main_process():
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
