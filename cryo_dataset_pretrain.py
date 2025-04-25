import os
import torch
from torch.utils.data import Dataset
import mrcfile
import numpy as np
from torchvision import transforms
import copy

class MRC2DDataset(Dataset):
    def __init__(self, mrc_path, slice_axis=0, transform=None, normalize=True):
        """
        2Dスライスを取得するためのMRCデータセットクラス。

        Args:
            mrc_path (str): MRCファイルのパス。
            slice_axis (int): スライスする軸 (0: z, 1: y, 2: x)。
            transform (callable, optional): データ変換処理。
            normalize (bool): 標準化を行うかどうか。
        """
        self.mrc_path = copy.copy(mrc_path)
        self.slice_axis = slice_axis
        self.transform = transform
        self.normalize = normalize

        # MRCファイルを読み込む
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            self.data = mrc.data  # 3Dデータ (z, y, x)
            print(f"[DEBUG __init__] Loaded MRC file: shape={self.data.shape}, "
                  f"min={self.data.min()}, max={self.data.max()}, mean={self.data.mean():.4f}, std={self.data.std():.4f}")

        # 標準化
        if self.normalize:
            self.data = (self.data - self.data.mean()) / self.data.std()
            print(f"[DEBUG __init__] Normalized data: mean={self.data.mean():.4f}, std={self.data.std():.4f}")


        # 指定された軸でスライス
        self.slices = np.moveaxis(self.data, self.slice_axis, 0)
        print(f"[DEBUG __init__] Sliced data along axis {self.slice_axis}: shape={self.slices.shape}")

    def __len__(self):
        return len(self.slices) * 4

    def __getitem__(self, idx):
        """
        指定されたインデックスの2Dスライスを取得。

        Args:
            idx (int): スライスのインデックス。

        Returns:
            tuple: (torch.Tensor, int) 分割されたスライスの1つとダミーラベル。
        """
        slice_idx = idx // 4
        sub_idx = idx % 4
        slice_2d = self.slices[slice_idx]

        # 縦横を半分に分割
        h, w = slice_2d.shape
        h_half, w_half = h // 2, w // 2
        sub_slices = [
            slice_2d[:h_half, :w_half],  # 左上
            slice_2d[:h_half, w_half:],  # 右上
            slice_2d[h_half:, :w_half],  # 左下
            slice_2d[h_half:, w_half:]   # 右下
        ]

        sub_slice = sub_slices[sub_idx]

        # チャンネル次元を追加 (H, W) -> (1, H, W)
        sub_slice = np.expand_dims(sub_slice, axis=0)

        # 1チャンネルを3チャンネルに変換 (1, H, W) -> (3, H, W)
        sub_slice = np.repeat(sub_slice, 3, axis=0)

        # データ変換を適用
        if self.transform:
            sub_slice = self.transform(torch.tensor(sub_slice, dtype=torch.float32))

        # ダミーラベルとして0を返す
        return sub_slice, 0

    def get_all_slices(self):
        """
        全スライスを取得。

        Returns:
            list[torch.Tensor]: 分割された全スライスのリスト。
        """
        all_slices = []
        for idx in range(len(self.slices)):
            slice_2d = self.slices[idx]

            # 縦横を半分に分割
            h, w = slice_2d.shape
            h_half, w_half = h // 2, w // 2
            sub_slices = [
                slice_2d[:h_half, :w_half],  # 左上
                slice_2d[:h_half, w_half:],  # 右上
                slice_2d[h_half:, :w_half],  # 左下
                slice_2d[h_half:, w_half:]   # 右下
            ]

            for sub_slice in sub_slices:
                sub_slice = np.expand_dims(sub_slice, axis=0)  # チャンネル次元を追加 (H, W) -> (1, H, W)
                if self.transform:
                    sub_slice = self.transform(torch.tensor(sub_slice, dtype=torch.float32))
                all_slices.append(sub_slice)

        return all_slices

# if __name__ == "__main__":
#     # ../dataset/model_0/grandmodel.mrc を読み込む
#     mrc_path = "../dataset/model_0/grandmodel.mrc"
#     dataset = MRC2DDataset(mrc_path, slice_axis=0, normalize=True)

#     # データセットの長さを確認
#     print(f"Dataset length: {len(dataset)}")

#     # 最初のスライスを取得して形状を確認
#     first_slice = dataset[0]
#     print(f"First slice shape: {first_slice.shape}")

#     # 全スライスを取得して確認
#     all_slices = dataset.get_all_slices()
#     print(f"Total slices retrieved: {len(all_slices)}")
#     print(f"Shape of a single slice: {all_slices[0].shape}")