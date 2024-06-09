import os
import random

import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, subset, image_transform=None, mask_transform=None):
        super().__init__()
        self.image_paths = sorted(
            [os.path.join(root_dir, 'images', subset, filename) for filename in
             os.listdir(str(os.path.join(root_dir, 'images', subset)))]
        )
        self.mask_paths = sorted(
            [os.path.join(root_dir, 'masks', subset, filename) for filename in
             os.listdir(str(os.path.join(root_dir, 'masks', subset)))]
        )
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = np.load(self.mask_paths[idx])
        mask = torch.from_numpy(mask).long()

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


if __name__ == "__main__":
    image_transform = None
    mask_transform = None

    train_dataset = SegmentationDataset(root_dir='../data/dataset', subset='test',
                                        image_transform=image_transform, mask_transform=mask_transform)

    print("Number of training samples:", len(train_dataset))

    rand_idx = random.randint(0, len(train_dataset) -1)

    img, mask = train_dataset[rand_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.show()