import os
import sys
import argparse
import yaml
import json
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from synthesis_task import SynthesisTask
from utils import run_shell_cmd


parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./params.yaml", type=str)
parser.add_argument("--workspace", type=str, required=True)
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--extra_config", type=str, default="{}", required=False)
parser.add_argument("--local_rank", default=0, type=int,
                    help="node rank for distributed training")
args = parser.parse_args()


local_rank = int(args.local_rank)

# Load config yaml file and pre-process params
default_config_path = os.path.join(os.path.dirname(args.config_path), "params_default.yaml")
with open(default_config_path, "r") as f:
    config = yaml.safe_load(f)

extra_config = json.loads(args.extra_config)
with open(args.config_path, "r") as f:
    dataset_specific_config = yaml.safe_load(f)
    for k in dataset_specific_config.keys():
        assert k in config, k
    config.update(dataset_specific_config)

    for k in extra_config.keys():
        assert k in config, k
    config.update(extra_config)

# Dump tmp config file
tmp_config_path = os.path.join(os.path.dirname(args.config_path), "params_tmp.yaml")
if local_rank == 0:
    with open(tmp_config_path, "w") as f:
        print("Dumping extra config file...")
        yaml.dump(config, f)

# pre-process params
config["training.gpus"] = [int(s) for s in str(config["training.gpus"]).split(",")]
config["lr.decay_steps"] = [int(s) for s in str(config["lr.decay_steps"]).split(",")]
config["current_epoch"] = 0

# Config gpu
gpus = config["training.gpus"][local_rank]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

# dist env
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
global_rank = dist.get_rank()
config["global_rank"] = global_rank


def get_dataset(config, logger):
    # Init data loader
    assert config["data.name"] in ["llff", "realestate10k", "flowers", "kitti_raw", "dtu"]

    if config["data.name"] == "llff":
        from input_pipelines.llff.nerf_dataset import NeRFDataset
        train_dataset = NeRFDataset(config,
                                    logger,
                                    root=config["data.training_set_path"],
                                    is_validation=False,
                                    img_size=(config["data.img_w"], config["data.img_h"]),
                                    supervision_count=config["data.num_tgt_views"],
                                    visible_points_count=config["data.visible_point_count"],
                                    img_pre_downsample_ratio=config["data.img_pre_downsample_ratio"])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=config["data.per_gpu_batch_size"],
                                       drop_last=True, num_workers=0,
                                       sampler=train_sampler,
                                       collate_fn=train_dataset.collate_fn)

        val_dataset = NeRFDataset(config,
                                  logger,
                                  root=config["data.training_set_path"],
                                  is_validation=True,
                                  img_size=(config["data.img_w"], config["data.img_h"]),
                                  supervision_count=config["data.num_tgt_views"],
                                  visible_points_count=config["data.visible_point_count"],
                                  img_pre_downsample_ratio=config["data.img_pre_downsample_ratio"])
        val_data_loader = DataLoader(val_dataset, batch_size=config["data.per_gpu_batch_size"],
                                     shuffle=False, drop_last=False, num_workers=0,
                                     collate_fn=val_dataset.collate_fn)
    else:
        raise NotImplementedError

    return train_data_loader, val_data_loader


def train():
    config["local_rank"] = local_rank
    config["world_size"] = world_size

    # Enable cudnn benchmark for speed optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Config logging and tb writer
    logger = None
    if global_rank == 0:
        import logging
        # logging to file and stdout
        config["log_file"] = "./training.log" \
            if args.workspace.startswith("hdfs") \
            else os.path.join(args.workspace, args.version, "training.log")
        logger = logging.getLogger("mine")
        file_handler = logging.FileHandler(config["log_file"])
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.handlers = [file_handler, stream_handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("Training config: {}".format(config))

        # tensorboard summary_writer
        config["tb_writer"] = SummaryWriter(log_dir=config["local_workspace"])
    config["logger"] = logger

    # Init data loader
    train_data_loader, val_data_loader = get_dataset(config, logger)
    synthesis_task = SynthesisTask(config=config, logger=logger)
    synthesis_task.train(train_data_loader, val_data_loader)


def main():
    if config["global_rank"] == 0:
        # Create sub working dir
        workspace = os.path.join(args.workspace, args.version)
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        config["local_workspace"] = workspace
        shutil.copy(tmp_config_path, os.path.join(workspace, "params.yaml"))
    dist.barrier()

    # Start training
    train()


if __name__ == "__main__":
    main()
