import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class CryoPPPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg', '.mrc'))
        ]
        # mean, stdをキャッシュ
        self.mean_std_cache = []
        for img_path in self.image_paths:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            mean = image_np.mean(axis=(0, 1))
            std = image_np.std(axis=(0, 1)) + 1e-6
            self.mean_std_cache.append((mean, std))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mean, std = self.mean_std_cache[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            tensor = self.transform(image)
        else:
            image_np = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        return tensor, tensor, 0  # input, target, dummy label