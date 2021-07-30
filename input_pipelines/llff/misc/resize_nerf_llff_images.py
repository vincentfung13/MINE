import os
import cv2
import numpy as np
import shutil


def resize_nerf_llff(dataset_path, downsample_ratio):
    scene_list = []
    for scene in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, scene)):
            continue
        scene_list.append(scene)

        images_dir = os.path.join(dataset_path, scene, 'images')
        images_down_dir = os.path.join(dataset_path, scene, "images_" + str(downsample_ratio))
        if os.path.exists(images_down_dir):
            shutil.rmtree(images_down_dir)
        os.makedirs(images_down_dir)

        for img_name in os.listdir(images_dir):
            img_np = cv2.imread(os.path.join(images_dir, img_name), cv2.IMREAD_COLOR)  # HxWx3
            H, W = img_np.shape[0:2]
            H_down = int(round(H / downsample_ratio))
            W_down = int(round(W / downsample_ratio))
            img_down_np = cv2.resize(img_np, (W_down, H_down))
            cv2.imwrite(os.path.join(images_down_dir, img_name), img_down_np)

    print(scene_list)


if __name__ == '__main__':
    dataset_path = '/data00/home/jiaxinli/data/nerf_llff_data'
    downsample_ratio = 7.875
    resize_nerf_llff(dataset_path, downsample_ratio)