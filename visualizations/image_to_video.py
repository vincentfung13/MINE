import numpy as np
import cv2
import time
import yaml
import sys
from scipy.interpolate import interp1d
import os
import argparse
import logging
import math
import torch
import json
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from torch.utils.tensorboard import SummaryWriter

from synthesis_task import SynthesisTask
from operations import mpi_rendering
from torch.utils.data import DataLoader


def path_planning(num_frames, x, y, z, path_type='', s=0.3):
    if path_type == 'straight-line':
        corner_points = np.array([[0, 0, 0], [(0 + x) * 0.5, (0 + y) * 0.5, (0 + z) * 0.5], [x, y, z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, num_frames)
        cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
    elif path_type == 'double-straight-line':
        corner_points = np.array([[s*x, s*y, s*z], [-x, -y, -z]])
        corner_t = np.linspace(0, 1, len(corner_points))
        t = np.linspace(0, 1, int(num_frames*0.5))
        cs = interp1d(corner_t, corner_points, axis=0, kind='linear')
        spline = cs(t)
        xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
        xs = np.concatenate((xs, np.flip(xs)))
        ys = np.concatenate((ys, np.flip(ys)))
        zs = np.concatenate((zs, np.flip(zs)))
    elif path_type == 'circle':
        xs, ys, zs = [], [], []
        for frame_id, bs_shift_val in enumerate(np.arange(-2.0, 2.0, (4./num_frames))):
            xs += [np.cos(bs_shift_val * np.pi) * 1 * x]
            ys += [np.sin(bs_shift_val * np.pi) * 1 * y]
            zs += [np.cos(bs_shift_val * np.pi/2.) * 1 * z - s*z]
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    return xs, ys, zs


def disparity_normalization_vis(disparity):
    """
    :param disparity: Bx1xHxW, pytorch tensor of float32
    :return:
    """
    assert len(disparity.size()) == 4 and disparity.size(1) == 1
    disp_min = torch.amin(disparity, (1, 2, 3), keepdim=True)
    disp_max = torch.amax(disparity, (1, 2, 3), keepdim=True)
    disparity_syn_scaled = (disparity - disp_min) / (disp_max - disp_min)
    disparity_syn_scaled = torch.clip(disparity_syn_scaled, 0.0, 1.0)
    return disparity_syn_scaled


def img_tensor_to_np(img_tensor):
    B, C, H, W = img_tensor.size()
    assert B == 1
    assert C == 1 or C == 3
    img_np_HWC = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]
    img_np_HWC_255 = np.clip(np.round(img_np_HWC * 255), a_min=0, a_max=255).astype(np.uint8)
    if C == 1:
        img_np_HWC_255 = cv2.applyColorMap(img_np_HWC_255, cv2.COLORMAP_HOT)
        img_np_HWC_255 = cv2.cvtColor(img_np_HWC_255, cv2.COLOR_BGR2RGB)
    return img_np_HWC_255


def write_img_to_disk(img_tensor, step, postfix, output_dir):
    B, C, H, W = img_tensor.size()
    assert C==1 or C==3
    img_np_BHWC = img_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    img_np_BHWC_255 = np.clip(np.round(img_np_BHWC * 255), a_min=0, a_max=255)
    for b in range(B):
        if C == 3:
            cv2.imwrite(os.path.join(output_dir, '%d_%d_%s.png'%(step, b, postfix)),
                        cv2.cvtColor(img_np_BHWC_255[b], cv2.COLOR_RGB2BGR))
        elif C == 1:
            cv2.imwrite(os.path.join(output_dir, '%d_%d_%s.png'%(step, b, postfix)),
                        img_np_BHWC_255[b, :, :, 0])


class VideoGenerator:
    def __init__(self, synthesis_task, config, logger, img, output_dir):
        self.synthesis_task = synthesis_task
        self.config = config
        self.logger = logger

        self.synthesis_task.global_step = config["training.eval_interval"]
        self.synthesis_task.logger.info("Start running evaluation on validation set:")
        self.synthesis_task.backbone.eval()
        self.synthesis_task.decoder.eval()

        if isinstance(img, np.ndarray):
            img = cv2.resize(img, (config["data.img_w"], config["data.img_h"]), cv2.INTER_LINEAR)
            self.img = torch.from_numpy(img).cuda().permute(2, 0, 1).contiguous().unsqueeze(0) / 255.0
        else:
            self.img = img

        self.output_dir = output_dir
        self.tgts_poses, self.traj_config = self.traj_generation()

        self.infer_network()

    def infer_network(self):
        B, _, H, W = self.img.size()
        self.K = torch.from_numpy(self.compute_camera_intrinsic(H, W).astype(np.float32)).unsqueeze(0).to(self.img.device)
        self.K_inv = torch.inverse(self.K).to(self.img.device)
        N_pt = 128

        src_items = {
            "img": self.img,
            "K": self.K,
            "K_inv": self.K_inv,
            "xyzs": torch.ones((B, 3, N_pt), dtype=torch.float32)
        }
        tgt_items = {
            "img": self.img.unsqueeze(1),
            "K": self.K.unsqueeze(1),
            "K_inv": self.K_inv.unsqueeze(1),
            "xyzs": torch.ones((B, 1, 3, N_pt), dtype=torch.float32),
            "G_src_tgt": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        }
        self.synthesis_task.set_data((src_items, tgt_items))

        # self.xyz_src_BS3HW
        endpoints = self.synthesis_task.network_forward()
        self.disparity_all_src = endpoints["disparity_all_src"]
        mpi_all_src = endpoints["mpi_all_src_list"][0]

        # Do RGB blending
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            self.synthesis_task.homography_sampler_list[0].meshgrid,
            self.disparity_all_src,
            self.K_inv.to(self.img.device)
        )
        self.mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
        self.mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render(
            self.mpi_all_rgb_src,
            self.mpi_all_sigma_src,
            xyz_src_BS3HW,
            use_alpha=self.config.get("mpi.use_alpha", False),
            is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        self.mpi_all_rgb_src = blend_weights * self.img.unsqueeze(1) + (1 - blend_weights) * self.mpi_all_rgb_src


    def traj_generation(self):
        traj_config = {}
        if self.config["data.name"] == "kitti_raw":
            traj_config["fps"] = 30
            traj_config["num_frames"] = 90
            traj_config["x_shift_range"] = [0.0, -0.8]
            traj_config["y_shift_range"] = [0.0, -0.0]
            traj_config["z_shift_range"] = [-1.5, -1.0]
            traj_config["traj_types"] = ['double-straight-line', 'circle']
            traj_config["name"] = ['zoom-in', 'swing']
        elif self.config["data.name"] in ["nyu", "ibims", "realestate10k"]:
            traj_config["fps"] = 30
            traj_config["num_frames"] = 90
            traj_config["x_shift_range"] = [0.0, -0.16]
            traj_config["y_shift_range"] = [0.0, -0.0]
            traj_config["z_shift_range"] = [-0.30, -0.2]
            traj_config["traj_types"] = ['double-straight-line', 'circle']
            traj_config["name"] = ['zoom-in', 'swing']
        else:
            raise RuntimeError("Unsupported dataset.")

        tgts_poses = []
        generic_pose = np.eye(4)
        for traj_idx in range(len(traj_config['traj_types'])):
            tgt_poses = []
            sx, sy, sz = path_planning(traj_config['num_frames'],
                                       traj_config['x_shift_range'][traj_idx],
                                       traj_config['y_shift_range'][traj_idx],
                                       traj_config['z_shift_range'][traj_idx],
                                       path_type=traj_config['traj_types'][traj_idx])
            for xx, yy, zz in zip(sx, sy, sz):
                tgt_poses.append(generic_pose * 1.)
                tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
            tgts_poses += [tgt_poses]
        return tgts_poses, traj_config

    @staticmethod
    def compute_camera_intrinsic(H, W, fov=90):
        fov = fov * math.pi / 180
        fx = W * 0.5 / math.tan(fov * 0.5)
        fy = fx
        cx = W * 0.5
        cy = H * 0.5
        K = np.asarray([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float)
        return K

    def render_pose(self, G_tgt_src_np):
        G_tgt_src = torch.from_numpy(G_tgt_src_np.astype(np.float32)).unsqueeze(0)
        render_results = self.synthesis_task.render_novel_view(
            self.mpi_all_src,
            self.disparity_all_src,
            G_tgt_src, self.K_inv, self.K,
            scale_factor=1.0
        )
        tgt_imgs_syn = render_results["tgt_imgs_syn"]
        tgt_disparity_syn = render_results["tgt_disparity_syn"]
        tgt_disparity_syn = disparity_normalization_vis(tgt_disparity_syn)

        write_img_to_disk(tgt_imgs_syn, 0, "tgt_rgb", self.output_dir)
        write_img_to_disk(tgt_disparity_syn, 0, "tgt_disp", self.output_dir)

    def render_video(self, output_name):
        for i, name in enumerate(self.traj_config["name"]):
            poses = self.tgts_poses[i]
            tgt_img_np_list = []
            tgt_disp_np_list = []
            self.logger.info("Processing trajectory %s ..." % name)
            for pose in tqdm(poses):
                G_tgt_src = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(self.img.device)

                render_results = self.synthesis_task.render_novel_view(
                    self.mpi_all_rgb_src,
                    self.mpi_all_sigma_src,
                    self.disparity_all_src, G_tgt_src,
                    self.K_inv, self.K,
                    scale=0,
                    scale_factor=torch.tensor([1.0]).to(G_tgt_src.device)
                )
                tgt_imgs_syn = render_results["tgt_imgs_syn"]
                tgt_disparity_syn = render_results["tgt_disparity_syn"]
                tgt_disparity_syn = disparity_normalization_vis(tgt_disparity_syn)

                tgt_img_np = img_tensor_to_np(tgt_imgs_syn)
                tgt_disp_np = img_tensor_to_np(tgt_disparity_syn)
                tgt_img_np_list.append(tgt_img_np)
                tgt_disp_np_list.append(tgt_disp_np)

            # write to video
            rgb_clip = ImageSequenceClip(tgt_img_np_list, fps=self.traj_config["fps"])
            rgb_clip.write_videofile(os.path.join(self.output_dir, output_name+"_"+name+"_rgb.mp4"),
                                     fps=self.traj_config["fps"],
                                     verbose=False,
                                     logger=None)
            disp_clip = ImageSequenceClip(tgt_disp_np_list, fps=self.traj_config["fps"])
            disp_clip.write_videofile(os.path.join(self.output_dir, output_name+"_"+name + "_disp.mp4"),
                                      fps=self.traj_config["fps"],
                                      verbose=False,
                                      logger=None)


def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--extra_config", type=str, default="{}", required=False)
    args = parser.parse_args()

    # Enable cudnn benchmark for speed optimization
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True

    # Load config yaml file
    extra_config = json.loads(args.extra_config)
    config_path = os.path.join(os.path.dirname(args.checkpoint_path), "params.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f)
        for k in extra_config.keys():
            assert k in config, k
        config.update(extra_config)

    # preprocess config
    config["current_epoch"] = 0
    config["global_rank"] = 0
    config["training.pretrained_checkpoint_path"] = args.checkpoint_path

    # pre-process params
    config["mpi.disparity_list"] = np.zeros((1), dtype=np.float32)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        config["local_workspace"] = args.output_dir

    # logging to file and stdout
    config["log_file"] = os.path.join(args.output_dir, "inference.log")
    logger = logging.getLogger("graph_view_synthesis_inference")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.handlers = [stream_handler]
    logger.propagate = False

    config["logger"] = logger
    config["tb_writer"] = None  # SummaryWriter(args.output_dir)
    config["data.val_set_path"] = args.data_path
    config["data.per_gpu_batch_size"] = 1

    synthesis_task = SynthesisTask(config=config, logger=logger, is_val=True)

    img_np = cv2.imread(args.data_path, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    video_generator = VideoGenerator(synthesis_task, config, config["logger"], img_np, args.output_dir)
    with torch.no_grad():
        video_generator.render_video(os.path.basename(args.data_path)[0:-4])


if __name__ == '__main__':
    main()
