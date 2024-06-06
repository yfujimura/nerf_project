# borrowd from K-Planes (https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/ray_utils.py)

from typing import Tuple, Optional

import numpy as np
import torch

from .intrinsics import Intrinsics

def create_meshgrid(height: int,
                    width: int,
                    dev: str = 'cpu',
                    add_half: bool = True,
                    flat: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(width, dtype=torch.float32, device=dev)
    ys = torch.arange(height, dtype=torch.float32, device=dev)
    if add_half:
        xs += 0.5
        ys += 0.5
    # generate grid by stacking coordinates
    yy, xx = torch.meshgrid([ys, xs], indexing="ij")  # both HxW
    if flat:
        return xx.flatten(), yy.flatten()
    return xx, yy


def stack_camera_dirs(x: torch.Tensor, y: torch.Tensor, intrinsics: Intrinsics, opengl_camera: bool):
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    x = x.float()
    y = y.float()
    return torch.stack([
        (x - intrinsics.center_x) / intrinsics.focal_x,
        (y - intrinsics.center_y) / intrinsics.focal_y
        * (-1.0 if opengl_camera else 1.0),
        torch.full_like(x, fill_value=-1.0 if opengl_camera else 1.0)
    ], -1)  # (H, W, 3)


def get_ray_directions(intrinsics: Intrinsics, opengl_camera: bool, add_half: bool = True) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width, add_half=add_half)

    return stack_camera_dirs(xx, yy, intrinsics, opengl_camera)


def get_rays(directions: torch.Tensor,
             c2w: torch.Tensor,
             ndc: bool,
             ndc_near: float = 1.0,
             intrinsics: Optional[Intrinsics] = None,
             normalize_rd: bool = True):
    """Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Args:
        directions:
        c2w:
        ndc:
        ndc_near:
        intrinsics:
        normalize_rd:

    Returns:

    """
    directions = directions.view(-1, 3)  # [n_rays, 3]
    if len(c2w.shape) == 2:
        c2w = c2w[None, ...]
    rd = (directions[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    ro = torch.broadcast_to(c2w[:, :3, 3], directions.shape)
    if ndc:
        assert intrinsics is not None, "intrinsics must not be None when NDC active."
        ro, rd = ndc_rays_blender(
            intrinsics=intrinsics, near=ndc_near, rays_o=ro, rays_d=rd)
    if normalize_rd:
        rd /= torch.linalg.norm(rd, dim=-1, keepdim=True)
    return ro, rd

