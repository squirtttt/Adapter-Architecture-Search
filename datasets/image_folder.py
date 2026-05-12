import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_image_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMG_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return sorted(files)


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, first_k=None, repeat=1, cache='none', **kwargs):
        self.root_path_1 = root_path_1
        self.root_path_2 = root_path_2
        self.repeat = repeat
        self.cache = cache

        image_files = list_image_files(root_path_1)
        mask_files = list_image_files(root_path_2)
        mask_by_stem = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in mask_files
        }

        self.files = []
        missing = []
        for image_path in image_files:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = mask_by_stem.get(stem)
            if mask_path is None:
                missing.append(image_path)
                continue
            self.files.append((image_path, mask_path))

        if first_k is not None:
            self.files = self.files[:first_k]

        if not self.files:
            raise RuntimeError(
                f'No paired images found between {root_path_1} and {root_path_2}. '
                f'Found {len(image_files)} images and {len(mask_files)} masks.'
            )

        if missing:
            print(
                f'[PairedImageFolders] skipped {len(missing)} images without matching masks. '
                f'Example: {missing[0]}'
            )

        if self.cache == 'in_memory':
            self.files = [self._load_pair(image_path, mask_path) for image_path, mask_path in self.files]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        pair = self.files[idx % len(self.files)]
        if self.cache == 'in_memory':
            return pair
        image_path, mask_path = pair
        return self._load_pair(image_path, mask_path)

    def _load_pair(self, image_path, mask_path):
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return image, mask
