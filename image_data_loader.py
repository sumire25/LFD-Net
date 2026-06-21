import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class Haze1kDataset(data.Dataset):
    def __init__(self, input_dir, target_dir, resize=(480, 640)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.resize = resize
        self.filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])

    def __getitem__(self, index):
        filename = self.filenames[index]
        hazy_path = os.path.join(self.input_dir, filename)
        clear_path = os.path.join(self.target_dir, filename)

        hazy_image = Image.open(hazy_path).convert('RGB')
        clear_image = Image.open(clear_path).convert('RGB')

        if self.resize:
            hazy_image = hazy_image.resize(self.resize, Image.LANCZOS)
            clear_image = clear_image.resize(self.resize, Image.LANCZOS)

        hazy_np = np.asarray(hazy_image).astype(np.float32) / 255.0
        clear_np = np.asarray(clear_image).astype(np.float32) / 255.0

        hazy_tensor = torch.from_numpy(hazy_np).permute(2, 0, 1)
        clear_tensor = torch.from_numpy(clear_np).permute(2, 0, 1)

        return clear_tensor, hazy_tensor

    def __len__(self):
        return len(self.filenames)
