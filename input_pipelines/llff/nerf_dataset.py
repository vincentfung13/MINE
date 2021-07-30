import random
import os
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
import torch.utils.data as data

from input_pipelines import colmap_utils


def _collate_fn(batch):
    _src_items, _tgt_items = zip(*batch)

    # Gather and stack tgt infos
    tgt_items = defaultdict(list)
    for si in _tgt_items:
        for k, v in si.items():
            tgt_items[k].append(default_collate(v))

    for k in tgt_items.keys():
        tgt_items[k] = torch.stack(tgt_items[k], axis=0)

    src_items = default_collate(_src_items)
    src_items = {k: v for k, v in src_items.items()
                 if k != "G_cam_world"}
    return src_items, tgt_items


class NeRFDataset(data.Dataset):
    def __init__(self, config, logger, root, is_validation, img_size,
                 supervision_count=5, visible_points_count=8, img_pre_downsample_ratio=7.875):
        self.logger = logger
        self.config = config
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.is_validation = is_validation
        self.visible_points_count = visible_points_count
        self.supervision_count = supervision_count
        self.collate_fn = _collate_fn
        self._init_img_transforms()

        # use pre_downsampled image data
        if img_pre_downsample_ratio is None or img_pre_downsample_ratio <= 1:
            image_folder = "images"
        else:
            image_folder = "images_" + str(img_pre_downsample_ratio)

        if is_validation:
            image_folder += "_val"

        # Read data
        self.dataset_infos = defaultdict(dict)
        self.scene_to_indices = defaultdict(set)
        self.keys = []
        self.imgs = []
        index = 0
        for i, scene_name in enumerate(sorted([p for p in os.listdir(root)])):
            # Load colmap results
            scene_dir = os.path.join(root, scene_name)
            colmap_db = os.path.join(scene_dir, "sparse/0")
            cameras, images, points3D = colmap_utils.read_model(colmap_db, ext=".bin")
            assert len(cameras) == 1

            # Parse colmap results
            for img_id, img_item in images.items():
                img_path = os.path.join(scene_dir, image_folder, img_item.name)

                if not os.path.exists(img_path):
                    continue

                qvec = img_item.qvec
                tvec = img_item.tvec

                # Read image from disk and put it in RAM (for small dataset)
                img = Image.open(img_path)
                w, h = img.size
                img = self.img_transforms(img)

                # Gather info globally for random access via index
                self.scene_to_indices[scene_name].add(index)
                self.keys.append((scene_name, img_path))

                # Construct each data object
                xyzs = [(point3D_id, xy, points3D[point3D_id].xyz)
                        for xy, point3D_id in zip(img_item.xys, img_item.point3D_ids)
                        if point3D_id != -1]
                self.dataset_infos[scene_name][img_path] = self._info_transform(
                    {"img": img, "qvec": qvec, "tvec": tvec, "xyzs": xyzs,
                     "camera_params": cameras[img_item.camera_id].params},
                    (w * img_pre_downsample_ratio / self.img_w,
                     h * img_pre_downsample_ratio / self.img_h)
                )
                assert len(xyzs) >= visible_points_count
                index += 1

        self.length = len(self.keys)
        if self.logger:
            self.logger.info("Dataset root: {}, is_validation: {}, number of images: {}"
                             .format(root, self.is_validation, self.length))

    def __getitem__(self, index):
        # Read src item
        scene_name, img_path = self.keys[index]
        _src_item = self.dataset_infos[scene_name][img_path]

        # Copy new src_item
        src_item = {k: v for k, v in _src_item.items()}

        # Read tgt items:
        tgt_items = self._sample_tgt_items(index, src_item)

        # Sample 3D points in src items
        # TODO: deterministic behavior in val
        sampled_indices = random.sample(range(len(_src_item["xyzs_ids"])),
                                        self.visible_points_count)
        # sampled_indices = random.sample(range(len(_src_item["xyzs_ids"])),
        #                                 self.visible_points_count) \
        #     if not self.is_validation \
        #     else sorted(range(len(_src_item["xyzs_ids"])))[:256]
        src_item["xyzs"] = src_item["xyzs"][:, sampled_indices]
        src_item["xyzs_ids"] = src_item["xyzs_ids"][sampled_indices]
        src_item["depths"] = src_item["depths"][sampled_indices]
        return src_item, tgt_items

    def __len__(self):
        return self.length

    def _init_img_transforms(self):
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def _info_transform(self, info, downsample_ratio):
        downsample_ratio_x, downsample_ratio_y = downsample_ratio
        _info = {"img": info["img"]}

        # Compute homogenous transfomration matrix
        R_cam_world = colmap_utils.qvec2rotmat(info["qvec"]).astype(np.float32)
        T_cam_world = np.array(info["tvec"], dtype=np.float32)
        _info["G_cam_world"] = np.vstack([
            np.hstack((R_cam_world, np.expand_dims(T_cam_world, axis=1))),
            np.array([0, 0, 0, 1])
        ]).astype(np.float32)

        # Assuming simple_radial camera model
        # Compute K matrix
        f_x = info["camera_params"][0] / downsample_ratio_x
        f_y = info["camera_params"][0] / downsample_ratio_y
        p_x = info["camera_params"][1] / downsample_ratio_x
        p_y = info["camera_params"][2] / downsample_ratio_y
        _info["K"] = np.array([
            [f_x, 0, p_x],
            [0, f_y, p_y],
            [0, 0, 1]
        ], dtype=np.float32)
        _info["K_inv"] = np.linalg.inv(_info["K"])

        # Scale coordinates of tracked points, then compute and normalize the depth for each point
        I_Zero = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32)
        P = _info["K"] @ I_Zero @ _info["G_cam_world"]
        m_det_sign = np.sign(np.linalg.det(P[:, :-1]))
        m3_norm = np.linalg.norm(P[2][:-1])

        # Convert xyzs_world to homogeneous coordinates
        xyzs_ids, xys_cam, xyzs_world = zip(*info["xyzs"])
        xys_cam = np.array(xys_cam).T.astype(np.float32)
        xys_cam[0] /= downsample_ratio_x
        xys_cam[1] /= downsample_ratio_y

        xyzs_world = np.array(xyzs_world)
        xyzs_world_homo = np.hstack((xyzs_world,
                                     np.ones((len(xyzs_world), 1)))).T.astype(np.float32)

        # Transform xyzs to camera coordiantes
        xyzs_cam_homo = _info["G_cam_world"] @ xyzs_world_homo
        xyzs_cam_homo /= xyzs_cam_homo[-1]

        # Reproject to image plane to obtain depths
        xys_cam_reproj = _info["K"] @ I_Zero @ xyzs_cam_homo
        depths = (m_det_sign * xys_cam_reproj[-1]) / m3_norm
        xys_cam_reproj /= xys_cam_reproj[-1]

        _info["xyzs"] = xyzs_cam_homo[:-1]
        _info["xyzs_ids"] = np.array(xyzs_ids)
        _info["depths"] = depths
        return _info

    def _sample_tgt_items(self, src_idx, src_item):
        G_src_world = src_item["G_cam_world"]
        scene_name, _ = self.keys[src_idx]

        # randomly sample K items for supervision, excluding the src_idx
        scene_indices = [i for i in self.scene_to_indices[scene_name] if i != src_idx]
        if not self.is_validation:
            sampled_indices = random.sample(scene_indices, self.supervision_count)
        else:
            sampled_indices = [scene_indices[(src_idx + 1) % (len(scene_indices)) - 1]]

        # Generate sampled_items and calculate the relative rotation matrix and translation vector
        # accordingly.
        sampled_items = defaultdict(list)
        for index in sampled_indices:
            _, img_path = self.keys[index]
            img_info = self.dataset_infos[scene_name][img_path]

            G_tgt_world = img_info["G_cam_world"]
            G_src_tgt = G_src_world @ np.linalg.inv(G_tgt_world)

            sampled_items["img"].append(img_info["img"])
            sampled_items["K"].append(img_info["K"])
            sampled_items["K_inv"].append(img_info["K_inv"])
            sampled_items["G_src_tgt"].append(G_src_tgt)

            # Sample xyz points
            # TODO: deterministic behavior in val
            # sampled_xyzs_indices = random.sample(range(len(img_info["xyzs_ids"])),
            #                                      self.visible_points_count) \
            #     if not self.is_validation \
            #     else sorted(range(len(img_info["xyzs_ids"]))[:256])
            sampled_xyzs_indices = random.sample(range(len(img_info["xyzs_ids"])),
                                                 self.visible_points_count)
            sampled_items["xyzs"].append(img_info["xyzs"][:, sampled_xyzs_indices])
            sampled_items["xyzs_ids"].append(img_info["xyzs_ids"][sampled_xyzs_indices])
            sampled_items["depths"].append(img_info["depths"][sampled_xyzs_indices])
        return sampled_items


if __name__ == "__main__":
    import logging
    dataset = NeRFDataset({}, logging,
                          root="/data00/home/vincentfeng/datasets/nerf_llff_data",
                          is_validation=True,
                          img_size=(384, 256),
                          supervision_count=1,
                          img_pre_downsample_ratio=7.875)
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=4, shuffle=False,
                    drop_last=True, num_workers=0,
                    collate_fn=_collate_fn)

    for batch in dl:
        src_item, supervision_items = batch

        for k, v in src_item.items():
            print(k, v.size())

        print("********")

        for k, v in supervision_items.items():
            print(k, v.size())

        break
