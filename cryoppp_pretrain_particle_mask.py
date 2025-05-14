# choose particle area from cryo-EM images

import os
import csv
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import resize
from tqdm import tqdm

class CryoPPPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_patches = []

        # micrographsディレクトリと対応するground_truth/particle_coordinatesディレクトリを取得
        micrographs_dir = root_dir
        csv_dir = os.path.join(os.path.dirname(root_dir), 'ground_truth', 'particle_coordinates')
        # micrographs内の画像ファイル一覧
        image_files = sorted([f for f in os.listdir(micrographs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # csvファイル一覧
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')])
        # 画像とcsvを対応付けてループ
        for img_fname, csv_fname in zip(image_files, csv_files):
            img_path = os.path.join(micrographs_dir, img_fname)
            csv_path = os.path.join(csv_dir, csv_fname)
            image = Image.open(img_path).convert("RGB")
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for particle in reader:
                    x = int(float(particle["X-Coordinate"]))
                    y = int(float(particle["Y-Coordinate"]))
                    d = int(float(particle["Diameter"]))
                    left = x - d // 2
                    upper = y - d // 2
                    right = x + d // 2
                    lower = y + d // 2
                    # 画像範囲外をcropしないようにクリップ
                    left = max(left, 0)
                    upper = max(upper, 0)
                    right = min(right, image.width)
                    lower = min(lower, image.height)
                    patch = image.crop((left, upper, right, lower))
                    self.image_patches.append(patch)
                    import pdb; p = pdb.Pdb(); p.set_trace()
        print(f"Loaded {len(self.image_patches)} particles from {len(image_files)} images.")
        
    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self)} items.")
        patch = self.image_patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, patch, 0  # input, target, and dummy label