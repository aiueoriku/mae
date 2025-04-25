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

def visualize_reconstruction(samples, y, mask, model, epoch, data_iter_step, output_dir, sample_index):
    # 入力画像（GT）
    img = samples[0].detach().cpu()
    x = torch.tensor(img)
    x = torch.einsum('chw->hwc', x)  # [C, H, W] -> [H, W, C]

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

    # 可視化
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(1, 4, 1)
    show_image(x.numpy(), "Original (GT)")
    plt.subplot(1, 4, 2)
    show_image(im_masked.numpy(), "Masked")
    plt.subplot(1, 4, 3)
    show_image(y_vis.numpy(), "Reconstruction")
    plt.subplot(1, 4, 4)
    show_image(im_paste.numpy(), "Reconstruction + Visible")

    # 保存
    vis_dir = os.path.join(output_dir, f"visualizations/step_{data_iter_step}/sample_{sample_index}")  # 保存先ディレクトリ
    os.makedirs(vis_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
    save_path = os.path.join(vis_dir, f"epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()  # メモリを解放

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

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, mask_ratio=args.mask_ratio)
            
            if data_iter_step == 0 and epoch % 10 == 0:
                # 特定のインデックスのサンプルを複数取得 (例: インデックス0〜4)
                num_visualizations = 10
                for i in range(0, len(data_loader.dataset), len(data_loader.dataset) // num_visualizations):
                    fixed_sample, _ = data_loader.dataset[i]  # データセットから直接取得
                    fixed_sample = fixed_sample.unsqueeze(0).to(device)  # バッチ次元を追加してGPUに送る

                    # モデルに通して再構成画像とマスクを取得
                    with torch.no_grad():
                        _, y_fixed, mask_fixed = model(fixed_sample, mask_ratio=args.mask_ratio)

                    # 可視化
                    visualize_reconstruction(fixed_sample, y_fixed, mask_fixed, model, epoch, data_iter_step, args.output_dir, i)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 勾配の逆伝搬
        loss /= accum_iter # 勾配をaccum_iter回分に分ける
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0) # 内部でloss.backward()を呼び出す

        # 勾配の更新
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad() # 勾配を初期化

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}