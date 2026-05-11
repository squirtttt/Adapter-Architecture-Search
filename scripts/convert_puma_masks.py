"""
Convert PUMA GeoJSON masks to PNG format for faster training.
Usage:
    # All tissue as foreground
    python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation

    # Only tumor as foreground
    python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation --target_class tumor

    # Multi-class segmentation
    python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation --multiclass
"""
import os
import argparse
import json
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


# Tissue class mapping (for multi-class)
TISSUE_CLASSES = {
    'background': 0,
    'tissue_tumor': 1,
    'tissue_stroma': 2,
    'tissue_necrosis': 3,
    'tissue_blood_vessel': 4,
    'tissue_epidermis': 5,
}

# All tissue as foreground (default binary)
BINARY_MAPPING = {
    'background': 0,
    'tissue_tumor': 1,
    'tissue_stroma': 1,
    'tissue_necrosis': 1,
    'tissue_blood_vessel': 1,
    'tissue_epidermis': 1,
}

# Available target classes for single-class binary segmentation
AVAILABLE_TARGETS = ['tumor', 'stroma', 'necrosis', 'blood_vessel', 'epidermis', 'all']


def get_target_class_mapping(target_class):
    """
    Create binary mapping for a specific target class.
    Only the target class becomes foreground (1), others become background (0).
    """
    if target_class == 'all':
        return BINARY_MAPPING

    target_key = f'tissue_{target_class}'
    mapping = {
        'background': 0,
        'tissue_tumor': 0,
        'tissue_stroma': 0,
        'tissue_necrosis': 0,
        'tissue_blood_vessel': 0,
        'tissue_epidermis': 0,
    }

    if target_key in mapping:
        mapping[target_key] = 1
    else:
        raise ValueError(f"Unknown target class: {target_class}. Available: {AVAILABLE_TARGETS}")

    return mapping


def geojson_to_mask(geojson_path, img_size=(1024, 1024), class_map=None):
    """
    Convert GeoJSON annotation to mask image.

    Args:
        geojson_path: Path to GeoJSON file
        img_size: (height, width) of output mask
        class_map: Dictionary mapping class names to integer values
    """
    if class_map is None:
        class_map = BINARY_MAPPING

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(img_size, dtype=np.uint8)

    for feature in data.get('features', []):
        geometry = feature.get('geometry', {})
        properties = feature.get('properties', {})

        classification = properties.get('classification', {})
        class_name = classification.get('name', 'background')
        class_id = class_map.get(class_name, 0)

        if geometry.get('type') == 'Polygon':
            coords = geometry.get('coordinates', [])
            for polygon in coords:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], class_id)

        elif geometry.get('type') == 'MultiPolygon':
            coords = geometry.get('coordinates', [])
            for multi_poly in coords:
                for polygon in multi_poly:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], class_id)

    return mask


def convert_split(input_dir, split, class_map, output_suffix='', is_binary=True):
    """
    Convert all GeoJSON masks in a split to PNG.

    Args:
        input_dir: Base directory containing splits
        split: 'train', 'val', or 'test'
        class_map: Dictionary mapping class names to integer values
        output_suffix: Suffix for output directory (e.g., '_tumor')
        is_binary: Whether this is binary segmentation (affects value scaling)
    """
    mask_dir = os.path.join(input_dir, split, 'masks')
    image_dir = os.path.join(input_dir, split, 'images')
    output_dir = os.path.join(input_dir, split, f'masks_png{output_suffix}')

    os.makedirs(output_dir, exist_ok=True)

    geojson_files = [f for f in os.listdir(mask_dir) if f.endswith('.geojson')]

    print(f"\nConverting {split} split: {len(geojson_files)} files")

    for geojson_file in tqdm(geojson_files, desc=split):
        # Get corresponding image to determine size
        basename = geojson_file.replace('_tissue.geojson', '')

        # Find image file
        img_path = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            candidate = os.path.join(image_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Warning: No image found for {geojson_file}")
            continue

        # Get image size
        with Image.open(img_path) as img:
            img_size = (img.size[1], img.size[0])  # (height, width)

        # Convert GeoJSON to mask
        geojson_path = os.path.join(mask_dir, geojson_file)
        mask = geojson_to_mask(geojson_path, img_size=img_size, class_map=class_map)

        # For binary mask, scale to 0-255 for visibility
        if is_binary:
            mask_save = (mask * 255).astype(np.uint8)
        else:
            mask_save = mask

        # Save as PNG
        output_file = basename + '_mask.png'
        output_path = os.path.join(output_dir, output_file)
        Image.fromarray(mask_save).save(output_path)

    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PUMA GeoJSON masks to PNG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All tissue as foreground (default)
  python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation

  # Only tumor as foreground
  python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation --target_class tumor

  # Multi-class segmentation
  python scripts/convert_puma_masks.py --input_dir load/Patholgy_PUMA/tissue_segmentation --multiclass

Available target classes: tumor, stroma, necrosis, blood_vessel, epidermis, all
        """
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing train/val/test splits')
    parser.add_argument('--target_class', type=str, default='all',
                        choices=AVAILABLE_TARGETS,
                        help='Target class for binary segmentation (default: all)')
    parser.add_argument('--multiclass', action='store_true',
                        help='Create multi-class masks (overrides --target_class)')
    args = parser.parse_args()

    # Determine class mapping and output suffix
    if args.multiclass:
        class_map = TISSUE_CLASSES
        output_suffix = '_multiclass'
        is_binary = False
        print(f"Mode: Multi-class segmentation")
        print(f"Classes: {list(TISSUE_CLASSES.keys())}")
    else:
        class_map = get_target_class_mapping(args.target_class)
        output_suffix = f'_{args.target_class}' if args.target_class != 'all' else ''
        is_binary = True
        if args.target_class == 'all':
            print(f"Mode: Binary segmentation (all tissue as foreground)")
        else:
            print(f"Mode: Binary segmentation (only '{args.target_class}' as foreground)")

    print(f"Input directory: {args.input_dir}")
    print(f"Output suffix: masks_png{output_suffix}")

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.input_dir, split)
        if os.path.exists(split_dir):
            convert_split(args.input_dir, split, class_map=class_map,
                          output_suffix=output_suffix, is_binary=is_binary)
        else:
            print(f"Warning: {split} directory not found")

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
