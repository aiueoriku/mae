# TODO particle用にする．4分割はしない


import os
import torch
from torch.utils.data import Dataset
import mrcfile
import numpy as np
from torchvision import transforms
import copy

from torchvision.transforms import InterpolationMode

class MRC2DDataset(Dataset):
    # def __init__(self, mrc_path, slice_axis=0, transform=None, normalize=True):
    def __init__(self, input_mrc_path, target_mrc_path, slice_axis=0, transform=None, normalize=True):
        """
        2Dスライスを取得するためのMRCデータセットクラス。

        Args:
            mrc_path (str): MRCファイルのパス。
            slice_axis (int): スライスする軸 (0: z, 1: y, 2: x)。
            transform (callable, optional): データ変換処理。
            normalize (bool): 標準化を行うかどうか。
        """
        # self.mrc_path = copy.copy(mrc_path)
        self.input_mrc_path = input_mrc_path
        self.target_mrc_path = target_mrc_path
        
        self.slice_axis = slice_axis
        self.transform = transform
        self.normalize = normalize

        # 入力データを読み込む
        with mrcfile.open(input_mrc_path, permissive=True) as mrc:
            self.input_data = mrc.data
        if self.normalize:
            self.input_data = (self.input_data - self.input_data.mean()) / self.input_data.std()

        # ターゲットデータを読み込む
        with mrcfile.open(target_mrc_path, permissive=True) as mrc:
            self.target_data = mrc.data
        if self.normalize:
            self.target_data = (self.target_data - self.target_data.mean()) / self.target_data.std()
             
        # # MRCファイルを読み込む
        # with mrcfile.open(mrc_path, permissive=True) as mrc:
        #     self.data = mrc.data  # 3Dデータ (z, y, x)
        #     print(f"[DEBUG __init__] Loaded MRC file: shape={self.data.shape}, "
        #           f"min={self.data.min()}, max={self.data.max()}, mean={self.data.mean():.4f}, std={self.data.std():.4f}")

        # # 標準化
        # if self.normalize:
        #     self.data = (self.data - self.data.mean()) / self.data.std()
        #     print(f"[DEBUG __init__] Normalized data: mean={self.data.mean():.4f}, std={self.data.std():.4f}")


        # # 指定された軸でスライス
        # self.input_slices = np.moveaxis(self.data, self.slice_axis, 0)
        # print(f"[DEBUG __init__] Sliced data along axis {self.slice_axis}: shape={self.input_slices.shape}")

        # スライス
        self.input_slices = np.moveaxis(self.input_data, self.slice_axis, 0)
        self.target_slices = np.moveaxis(self.target_data, self.slice_axis, 0)

    def __len__(self):
        return len(self.input_slices)

    def __getitem__(self, idx):
        """
        指定されたインデックスの2Dスライスを取得。

        Args:
            idx (int): スライスのインデックス。

        Returns:
            tuple: (入力スライス, ターゲットスライス, ダミーラベル)
        """
        slice_idx = idx
        # 入力スライスとターゲットスライスを取得
        input_slice_2d = self.input_slices[slice_idx]
        target_slice_2d = self.target_slices[slice_idx]

        # チャンネル次元を追加 (H, W) -> (1, H, W)
        input_slice_2d = np.expand_dims(input_slice_2d, axis=0)
        target_slice_2d = np.expand_dims(target_slice_2d, axis=0)

        # 1チャンネルを3チャンネルに変換 (1, H, W) -> (3, H, W)
        input_slice_2d = np.repeat(input_slice_2d, 3, axis=0)
        target_slice_2d = np.repeat(target_slice_2d, 3, axis=0)

        # データ変換を適用
        if self.transform:
            # 同じ transform を input と target に適用
            seed = torch.seed()  # ランダムシードを固定
            torch.manual_seed(seed)

            input_slice_2d = self.transform(torch.tensor(input_slice_2d, dtype=torch.float32))
            torch.manual_seed(seed)
            target_slice_2d = self.transform(torch.tensor(target_slice_2d, dtype=torch.float32))

        # ダミーラベルとして0を返す
        dummy_label = 0

        return input_slice_2d, target_slice_2d, dummy_label
    def get_all_slices(self):
        """
        全スライスを取得。

        Returns:
            list[torch.Tensor]: 分割された全スライスのリスト。
        """
        all_slices = []
        for idx in range(len(self.input_slices)):
            slice_2d = self.input_slices[idx]

            sub_slice = np.expand_dims(slice_2d, axis=0)  # チャンネル次元を追加 (H, W) -> (1, H, W)
            if self.transform:
                sub_slice = self.transform(torch.tensor(sub_slice, dtype=torch.float32))
            all_slices.append(sub_slice)

        return all_slices

# if __name__ == "__main__":
#     # ../dataset/model_0/input/reconstruction.mrc と ../dataset/model_0/grandmodel.mrc を読み込む
#     input_mrc_path = "../dataset/model_0/reconstruction.mrc"
#     target_mrc_path = "../dataset/model_0/grandmodel.mrc"
    
#     # データセットを初期化
#     dataset = MRC2DDataset(input_mrc_path=input_mrc_path, 
#                            target_mrc_path=target_mrc_path, 
#                            slice_axis=0, 
#                            normalize=True)

#     # データセットの長さを確認
#     print(f"Dataset length: {len(dataset)}")

#     # 最初のスライスを取得して形状を確認
#     input_slice, target_slice, dummy_label = dataset[0]
#     print(f"Input slice shape: {input_slice.shape}")
#     print(f"Target slice shape: {target_slice.shape}")
#     print(f"Dummy label: {dummy_label}")

#     # 全スライスを取得して確認
#     all_slices = dataset.get_all_slices()
#     print(f"Total slices retrieved: {len(all_slices)}")
#     print(f"Shape of a single slice: {all_slices[0].shape}")