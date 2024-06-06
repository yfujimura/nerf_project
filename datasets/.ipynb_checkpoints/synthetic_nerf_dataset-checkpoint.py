# partially borrowd from K-Planes (https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/synthetic_nerf_dataset.py)

import json
import os
from typing import Tuple, Optional, Any
from loguru import logger

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch

from .ray_utils import get_ray_directions, get_rays
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class SyntheticNerfDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 max_frames: Optional[int] = None):
        self.max_frames = max_frames
        self.near_far = [2.0, 6.0]

        frames, transform = load_frames(datadir, split, self.max_frames)
        imgs, poses = load_images(frames, datadir, split)
        intrinsics = load_intrinsics(transform, img_h=imgs[0].shape[0], img_w=imgs[0].shape[1])
        rays_o, rays_d, imgs = create_rays(imgs, poses, merge_all=split == 'train', intrinsics=intrinsics)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            imgs=imgs,
            rays_o=rays_o,
            rays_d=rays_d,
            intrinsics=intrinsics,
        )
        

    def __getitem__(self, index):
        out = super().__getitem__(index)
        pixels = out["imgs"]
        if self.split == 'train':
            bg_color = torch.rand((1, 3), dtype=pixels.dtype, device=pixels.device)
            #bg_color = torch.ones((1, 3), dtype=pixels.dtype, device=pixels.device)
        else:
            if pixels is None:
                bg_color = torch.ones((1, 3), dtype=torch.float32, device='cuda:0')
            else:
                bg_color = torch.ones((1, 3), dtype=pixels.dtype, device=pixels.device)
        # Alpha compositing
        if pixels is not None:
            pixels = pixels[:, :3] * pixels[:, 3:] + bg_color * (1.0 - pixels[:, 3:])
        out["imgs"] = pixels
        out["bg_color"] = bg_color
        out["near_fars"] = torch.tensor([[2.0, 6.0]])
        return out


def create_rays(
              imgs: Optional[torch.Tensor],
              poses: torch.Tensor,
              merge_all: bool,
              intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True)  # [H, W, 3]
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        rays_o, rays_d = get_rays(directions, poses[i], ndc=False, normalize_rd=True)  # h*w, 3
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        if imgs is not None:
            imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
        all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
    return all_rays_o, all_rays_d, imgs


def load_frames(datadir, split, max_frames: int) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as f:
        meta = json.load(f)
        frames = meta['frames']

        # Subsample frames
        tot_frames = len(frames)
        num_frames = min(tot_frames, max_frames or tot_frames)
        if split == 'train' or split == 'test':
            subsample = int(round(tot_frames / num_frames))
            frame_ids = np.arange(tot_frames)[::subsample]
            if subsample > 1:
                log.info(f"Subsampling {split} set to 1 every {subsample} images.")
        else:
            frame_ids = np.arange(num_frames)
        frames = np.take(frames, frame_ids).tolist()
    return frames, meta


def load_images(frames, datadir, split) -> Tuple[torch.Tensor, torch.Tensor]:
    with ThreadPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), 32)
    ) as executor:
        imgs = list(
            executor.map(
                lambda f: torch.tensor(
                    np.array(Image.open(os.path.join(datadir, split, os.path.basename(f["file_path"]) + ".png"))).astype(np.float32) / 255.
                ),
                frames,
            ))
        
    logger.info(f"==> {len(imgs)} images were loaded for {split}.")
    poses = [torch.tensor(np.array(frame["transform_matrix"]).astype(np.float32)) for frame in frames]

    imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
    poses = torch.stack(poses, 0)  # [N, ????]
    return imgs, poses


def load_intrinsics(transform, img_h, img_w) -> Intrinsics:
    height = img_h
    width = img_w
    fl_x = fl_y = width / (2 * np.tan(transform['camera_angle_x'] / 2))
    cx = cy = width / 2
    return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)
