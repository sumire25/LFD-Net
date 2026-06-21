import os
import re
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


def _extract_id(filename):
    match = re.match(r'(\d+)', filename)
    return match.group(1) if match else filename


def _build_id_map(directory):
    return {_extract_id(f): f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))}


class Haze1kDataset(data.Dataset):
    def __init__(self, input_dir, target_dir, resize=(480, 640)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.resize = resize

        input_map = _build_id_map(input_dir)
        target_map = _build_id_map(target_dir)

        self.common_ids = sorted(set(input_map.keys()) & set(target_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        self.input_map = input_map
        self.target_map = target_map

    def __getitem__(self, index):
        img_id = self.common_ids[index]
        hazy_path = os.path.join(self.input_dir, self.input_map[img_id])
        clear_path = os.path.join(self.target_dir, self.target_map[img_id])

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
        return len(self.common_ids)
