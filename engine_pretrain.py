# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb

# cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
# cifar10_std = np.array([0.2470, 0.2435, 0.2616])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3

    # NumPy配列かPyTorchテンソルかを判定
    if isinstance(image, np.ndarray):
        image = np.clip(image * 255, 0, 255).astype(np.int32)
    elif isinstance(image, torch.Tensor):
        image = torch.clip(image * 255, 0, 255).int()
    else:
        raise TypeError("Unsupported image type. Expected numpy.ndarray or torch.Tensor.")

    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize_reconstruction(samples, targets, y, mask, model, epoch, data_iter_step, output_dir, sample_index, sample_image_lists):
    # 入力画像（GT）
    img = samples[0].detach().cpu()
    x = torch.tensor(img)
    x = torch.einsum('chw->hwc', x)  # [C, H, W] -> [H, W, C]

    # ターゲット画像
    target_vis = targets[0].detach().cpu()
    target_vis = torch.einsum('chw->hwc', target_vis)  # [C, H, W] -> [H, W, C]

    # 再構成画像
    y_vis = y[0].detach().cpu()
    y_vis = model.unpatchify(y_vis.unsqueeze(0))  # 再構成
    y_vis = torch.einsum('nchw->nhwc', y_vis)[0]  # [N, C, H, W] -> [H, W, C]

    # マスク画像
    mask_vis = mask[0].detach().cpu()
    mask_vis = mask_vis.unsqueeze(-1).repeat(1, model.patch_embed.patch_size[0]**2 * 3)  # (H*W, p*p*3)
    mask_vis = model.unpatchify(mask_vis.unsqueeze(0))  # マスクを画像形式に変換
    mask_vis = torch.einsum('nchw->nhwc', mask_vis)[0]  # [N, C, H, W] -> [H, W, C]

    # マスク適用後の画像
    im_masked = x * (1 - mask_vis)

    # 再構成画像とマスク適用画像を合成
    im_paste = x * (1 - mask_vis) + y_vis * mask_vis

    plt.rcParams['figure.figsize'] = [25, 10]
    plt.subplot(1, 5, 1)
    show_image(x.numpy(), "Original (Input)")
    plt.subplot(1, 5, 2)
    show_image(im_masked.numpy(), "Masked")
    plt.subplot(1, 5, 3)
    show_image(y_vis.numpy(), "Reconstruction")
    plt.subplot(1, 5, 4)
    show_image(im_paste.numpy(), "Reconstruction + Visible")
    plt.subplot(1, 5, 5)
    show_image(target_vis.numpy(), "Target (GT)")

    # 保存
    vis_dir = os.path.join(output_dir, f"visualizations/step_{data_iter_step}/sample_{sample_index}")  # 保存先ディレクトリ
    os.makedirs(vis_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
    save_path = os.path.join(vis_dir, f"epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()  # メモリを解放

    # 画像をリストに追加
    if sample_index not in sample_image_lists:
        sample_image_lists[sample_index] = []
    sample_image_lists[sample_index].append(wandb.Image(save_path, caption=f"Epoch {epoch}"))

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad() 

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 各 sample_index ごとの画像リストを格納する辞書
    sample_image_lists = {}

    for data_iter_step, (inputs, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(inputs, targets, mask_ratio=args.mask_ratio)
            
            if data_iter_step == 0 and epoch % 10 == 0:
                dataset_size = len(data_loader.dataset) 
                for i in range(0, min(dataset_size, 2000), 100): # 20枚の画像を取得
                    fixed_sample, fixed_target, _ = data_loader.dataset[i]  # データセットから直接取得
                    fixed_sample = fixed_sample.unsqueeze(0).to(device)  # バッチ次元を追加してGPUに送る
                    fixed_target = fixed_target.unsqueeze(0).to(device)

                    # モデルに通して再構成画像とマスクを取得
                    with torch.no_grad():
                        _, y_fixed, mask_fixed = model(fixed_sample, fixed_target, mask_ratio=args.mask_ratio)

                    # 可視化とリストへの追加
                    visualize_reconstruction(fixed_sample, fixed_target, y_fixed, mask_fixed, model, epoch, data_iter_step, args.output_dir, i, sample_image_lists)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 勾配の逆伝搬
        loss /= accum_iter  # 勾配をaccum_iter回分に分ける
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  # 内部でloss.backward()を呼び出す

        # 勾配の更新
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  # 勾配を初期化

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # wandbに損失と学習率を記録
        if misc.is_main_process():
            wandb.log({"train_loss": loss_value, "lr": lr, "epoch": epoch})

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Wandbに各 sample_index ごとの画像リストをアップロード
    if misc.is_main_process():
        for sample_index, image_list in sample_image_lists.items():
            wandb.log({f"visualizations_sample_{sample_index}": image_list})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}