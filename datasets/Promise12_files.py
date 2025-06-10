import numpy as np
from PIL import Image
import argparse
import os



def main(ori_path, save_path):
    imgs = np.load(ori_path)
    os.makedirs(save_path, exist_ok = True)
    
    # for i, img in enumerate(imgs):
            
    #     if img.dtype != np.uint8:
    #         img = (255*(img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
    #     image = Image.fromarray(img)

    
    for i, img in enumerate(imgs):
        print(f'{i}/{len(imgs)}')
        print(f'{np.shape(img)}')
        # img = (img*255).astype(np.uint8)
        # image = Image.fromarray(img)

        # image.save(f'{save_path}/img_{i:04d}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', default='../load/PROMISE2012/npy_image/X_test.npy')
    parser.add_argument('--target', default = '../load/PROMISE2012/Train_img')
    args = parser.parse_args()

    ori_path = os.path.join('../load/PROMISE2012/npy_image_2', args.origin)
    save_path = os.path.join('../load/PROMISE2012/', args.target)

    main(ori_path, save_path)