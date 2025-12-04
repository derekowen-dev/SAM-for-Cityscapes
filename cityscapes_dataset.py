import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CityscapesFineDataset(Dataset):
    def __init__(self, root, split="val", transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root, "leftImg8bit", split)
        self.gt_dir = os.path.join(root, "gtFine", split)

        self.samples = []
        cities = sorted(os.listdir(self.img_dir))
        for city in cities:
            img_city_dir = os.path.join(self.img_dir, city)
            gt_city_dir = os.path.join(self.gt_dir, city)
            for fname in os.listdir(img_city_dir):
                if not fname.endswith("leftImg8bit.png"):
                    continue
                img_path = os.path.join(img_city_dir, fname)
                gt_name = fname.replace("leftImg8bit", "gtFine_labelIds")
                gt_path = os.path.join(gt_city_dir, gt_name)
                if os.path.exists(gt_path):
                    self.samples.append((img_path, gt_path))

        print(f"[CityscapesFineDataset] Found {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path)

        img_np = np.array(img)
        gt_np = np.array(gt).astype(np.int64)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_np)

        return img_tensor, gt_tensor, img_path