import os
import shutil

test_dir = '/home/yn-jang/aas/load/CAMO/Images/Train'
gt_dir = '/home/yn-jang/aas/load/CAMO/GT'
target_dir = '/home/yn-jang/aas/load/CAMO/Train_gt'

test_files = os.listdir(test_dir)

for filename in test_files:
    base_name, _ = os.path.splitext(filename)
    png_filename = base_name+'.png'

    gt_file_path = os.path.join(gt_dir, png_filename)
    
    if os.path.exists(gt_file_path):
        target_path = os.path.join(target_dir, png_filename)
        shutil.move(gt_file_path, target_path)
        print(f"move: {filename}")
    else:
        print(f"not found in gt: {filename}")