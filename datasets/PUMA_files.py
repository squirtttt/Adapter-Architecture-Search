"""
PUMA Dataset loader for tissue segmentation.
Handles TIF images and GeoJSON masks.
"""
import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register
import cv2


# Tissue class mapping (background = 0)
TISSUE_CLASSES = {
    'background': 0,
    'tissue_tumor': 1,
    'tissue_stroma': 2,
    'tissue_necrosis': 3,
    'tissue_blood_vessel': 4,
    'tissue_epidermis': 5,
}

# For binary segmentation (all tissue vs background)
BINARY_MAPPING = {
    'background': 0,
    'tissue_tumor': 1,
    'tissue_stroma': 1,
    'tissue_necrosis': 1,
    'tissue_blood_vessel': 1,
    'tissue_epidermis': 1,
}


def geojson_to_mask(geojson_path, img_size=(1024, 1024), binary=True):
    """
    Convert GeoJSON annotation to binary or multi-class mask image.

    Args:
        geojson_path: Path to GeoJSON file
        img_size: Output mask size (height, width)
        binary: If True, create binary mask (tissue vs background)
                If False, create multi-class mask

    Returns:
        PIL Image (mode 'L') containing the mask
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(img_size, dtype=np.uint8)
    class_map = BINARY_MAPPING if binary else TISSUE_CLASSES

    for feature in data.get('features', []):
        geometry = feature.get('geometry', {})
        properties = feature.get('properties', {})

        # Get class name
        classification = properties.get('classification', {})
        class_name = classification.get('name', 'background')
        class_id = class_map.get(class_name, 0)

        if geometry.get('type') == 'Polygon':
            coords = geometry.get('coordinates', [])
            for polygon in coords:
                # Convert coordinates to numpy array
                pts = np.array(polygon, dtype=np.int32)
                # Fill polygon on mask
                cv2.fillPoly(mask, [pts], class_id)

        elif geometry.get('type') == 'MultiPolygon':
            coords = geometry.get('coordinates', [])
            for multi_poly in coords:
                for polygon in multi_poly:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], class_id)

    return Image.fromarray(mask)


@register('puma-tissue')
class PUMATissueDataset(Dataset):
    """
    PUMA Tissue Segmentation Dataset.
    Loads TIF images and converts GeoJSON masks on-the-fly.
    """

    def __init__(self, image_dir, mask_dir, split_key=None, cache='none', binary=True):
        """
        Args:
            image_dir: Directory containing TIF images
            mask_dir: Directory containing GeoJSON mask files
            split_key: Not used, for compatibility
            cache: 'none' or 'in_memory'
            binary: If True, binary segmentation (tissue vs background)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cache = cache
        self.binary = binary

        # Find matching image-mask pairs
        self.samples = []

        image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir)
                       if f.endswith(('.tif', '.tiff', '.png', '.jpg'))}

        for img_basename, img_filename in image_files.items():
            # Look for corresponding mask
            mask_filename = img_basename + '_tissue.geojson'
            mask_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                self.samples.append({
                    'image': os.path.join(image_dir, img_filename),
                    'mask': mask_path
                })

        print(f"[PUMA Dataset] Found {len(self.samples)} samples")

        # Cache if requested
        if cache == 'in_memory':
            self.cached_data = [self._load_sample(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        sample = self.samples[idx]

        # Load image (convert RGBA to RGB)
        img = Image.open(sample['image']).convert('RGB')

        # Convert GeoJSON to mask
        mask = geojson_to_mask(sample['mask'], img_size=img.size[::-1], binary=self.binary)

        return img, mask

    def __getitem__(self, idx):
        if self.cache == 'in_memory':
            return self.cached_data[idx]
        return self._load_sample(idx)


@register('puma-tissue-png')
class PUMATissuePNGDataset(Dataset):
    """
    PUMA Tissue Segmentation Dataset with pre-converted PNG masks.
    Use this after running the conversion script for faster loading.
    """

    def __init__(self, image_dir, mask_dir, split_key=None, cache='none'):
        """
        Args:
            image_dir: Directory containing images (TIF/PNG)
            mask_dir: Directory containing PNG mask files
            split_key: Not used, for compatibility
            cache: 'none' or 'in_memory'
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cache = cache

        # Find matching image-mask pairs
        self.samples = []

        image_files = {}
        for f in os.listdir(image_dir):
            if f.endswith(('.tif', '.tiff', '.png', '.jpg')):
                basename = f.rsplit('.', 1)[0]
                image_files[basename] = f

        mask_files = {}
        for f in os.listdir(mask_dir):
            if f.endswith('.png'):
                # Remove _mask suffix if present
                basename = f.replace('_mask.png', '').replace('.png', '')
                mask_files[basename] = f

        # Match by basename
        for basename, img_file in image_files.items():
            if basename in mask_files:
                self.samples.append({
                    'image': os.path.join(image_dir, img_file),
                    'mask': os.path.join(mask_dir, mask_files[basename])
                })

        print(f"[PUMA PNG Dataset] Found {len(self.samples)} samples")

        if cache == 'in_memory':
            self.cached_data = [self._load_sample(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')
        return img, mask

    def __getitem__(self, idx):
        if self.cache == 'in_memory':
            return self.cached_data[idx]
        return self._load_sample(idx)
