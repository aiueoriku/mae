import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import resize
from tqdm import tqdm

class CryoPPPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_patches = []

        for fname in tqdm(os.listdir(root_dir), desc="Processing images"):
            if fname.endswith(('.png', '.jpg', '.jpeg', '.mrc')):
                img_path = os.path.join(root_dir, fname)
                image = Image.open(img_path).convert("RGB")

                # Resize image to the nearest multiple of 224
                width, height = image.size
                new_width = (width // 224) * 224
                new_height = (height // 224) * 224
                image = resize(image, (new_height, new_width))

                # Sliding window crop to 224x224 patches
                for top in range(0, new_height, 224):
                    for left in range(0, new_width, 224):
                        patch = image.crop((left, top, left + 224, top + 224))
                        self.image_patches.append(patch)

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self)} items.")
        patch = self.image_patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, patch, 0  # input, target, and dummy label