from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224, augment=False):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentation(self, image, mask):
        # Horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Brightness variation
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)

        # Small rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        return image, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
