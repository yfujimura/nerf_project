# partially borrowed from K-Planes (https://github.com/sarafridov/K-Planes/blob/main/plenoxels/runners/base_trainer.py)
# and nerfacc (https://github.com/nerfstudio-project/nerfacc/blob/master/examples/train_ngp_nerf_occ.py) 

import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from loguru import logger
from skimage.metrics import structural_similarity

import torch
import torch.optim
import torch.nn.functional as F

from nerfacc.estimators.occ_grid import OccGridEstimator
from .render_utils import render_image_with_occgrid

class Trainer():
    
    def __init__(
        self,
        model,
        train_loader,
        train_dataset,
        test_dataset,
        **kwargs):
        
        self.model = model
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.extra_args = kwargs
        
        self.log_dir = os.path.join(self.extra_args["log_dir"], self.extra_args["exp_name"])
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.num_steps = self.extra_args["num_steps"]
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.extra_args["lr"],
            eps=1e-15,
            weight_decay=self.extra_args["weight_decay"],
        )
        self.scheduler = self.init_scheduler(self.optimizer, self.num_steps)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        
        aabb = [-1 * self.extra_args["radius"]] * 3 + [self.extra_args["radius"]] * 3
        self.estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=self.extra_args["grid_resolution"], levels=self.extra_args["grid_nlvl"]
        ).to("cuda")
        self.render_step_size = self.extra_args["render_step_size"]
        self.near_plane = self.extra_args["near_plane"]
        
        self.global_step = None
        self.loss_info = {"Loss": 0.}
        self.save_every = self.extra_args["save_every"]
        self.valid_every = self.extra_args["valid_every"]
        
        
        
    def init_scheduler(self, optimizer, max_steps):
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        return scheduler
    
    def train(self):
        self.global_step = 0
        self.pre_epoch()
        batch_iter = iter(self.train_loader)
        
        logger.info("==> Start training ...")
        pbar = tqdm(total=self.num_steps, dynamic_ncols=True)
        while self.global_step < self.num_steps:
            try:
                data = next(batch_iter)
            except StopIteration:
                self.pre_epoch()
                batch_iter = iter(self.train_loader)

            self.before_step()
            self.train_step(data)
            self.post_step(pbar)
                
            
    def train_step(self, data):
        data = self._move_data_to_device(data)
        
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            self.model,
            self.estimator,
            data["rays_o"],
            data["rays_d"],
            # rendering options
            near_plane=self.near_plane,
            render_step_size=self.render_step_size,
            render_bkgd=data["bg_color"],
        )
        
        loss = F.smooth_l1_loss(rgb, data["imgs"])
        
        # https://github.com/nerfstudio-project/nerfacc/issues/100#issuecomment-1303886174
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        
        self.loss_info.update(Loss=loss.item())
        
        
    def before_step(self):
        self.model.train()
        self.estimator.train()
        
        def occ_eval_fn(x):
            density = self.model.query_density(x)
            return density * self.render_step_size
        
        self.estimator.update_every_n_steps(
            step=self.global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )
        
        
    def post_step(self, progress_bar):
        self.global_step += 1
        progress_bar.set_postfix(self.loss_info)
        progress_bar.update(1)
        
        if self.global_step % self.save_every == 0:
            self.save_model()
        
        if self.global_step % self.valid_every == 0:
            self.validate()
        
        if self.global_step == self.num_steps:
            progress_bar.close()
        
        
    def pre_epoch(self):
        self.train_dataset.reset_iter()
        
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.estimator.eval()
        height = self.test_dataset.img_h
        width = self.test_dataset.img_w
        psnrs = []
        ssims = []
        lpipss = []
        logger.info(f"==> Start evaluation ...")
        
        out_dir = os.path.join(self.log_dir, f"{self.global_step}")
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"==> Outputs are saved to {out_dir}")
        
        with tqdm(total=len(self.test_dataset), dynamic_ncols=True) as pbar:
            for img_idx, data in enumerate(self.test_dataset):
                data = self._move_data_to_device(data)
                rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                    self.model,
                    self.estimator,
                    data["rays_o"],
                    data["rays_d"],
                    # rendering options
                    near_plane=self.near_plane,
                    render_step_size=self.render_step_size,
                    render_bkgd=data["bg_color"],
                )
                rgb = torch.clamp(rgb, 0, 1)
                
                rgb_np = rgb.cpu().numpy().reshape(height, width, 3)
                depth_np = depth.cpu().numpy().reshape(height, width)
                target_np = data["imgs"].cpu().numpy().reshape(height, width, 3)
                
                mse = F.mse_loss(rgb, data["imgs"])
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                
                ssim = structural_similarity(rgb_np, target_np, data_range=1., channel_axis=-1)
                ssims.append(ssim)
                
                # save output
                out_img_path = os.path.join(out_dir, f"color-{img_idx}.png")
                out_depth_path = os.path.join(out_dir, f"depth-{img_idx}.png")
                Image.fromarray((rgb_np * 255).astype(np.uint8)).save(out_img_path)
                Image.fromarray((depth_np * 1000).astype(np.uint16)).save(out_depth_path)
                
                pbar.set_postfix(PSNR=psnrs[-1])
                pbar.update(1)
        psnr_avg = sum(psnrs) / len(psnrs)
        ssim_avg = sum(ssims) / len(ssims)
        logger.info(f"==> PSNR={psnr_avg}, SSIM={ssim_avg} ")
        
        
    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "estimator": self.estimator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }
        
    def save_model(self):
        out_dir = os.path.join(self.log_dir, f"{self.global_step}")
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f'model.pth')
        torch.save(self.get_save_dict(), model_path)
        
    def load_model(self):
        ckpt = torch.load(os.path.join(self.log_dir, f"{self.extra_args['num_steps']}/model.pth"))
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.estimator.load_state_dict(ckpt["estimator"], strict=False)
        self.global_step = ckpt["global_step"]
        
        
    def _move_data_to_device(self, data, device="cuda"):
        data["rays_o"] = data["rays_o"].to(device)
        data["rays_d"] = data["rays_d"].to(device)
        data["imgs"] = data["imgs"].to(device)
        data["near_fars"] = data["near_fars"].to(device)
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(device)
        data["bg_color"] = bg_color
        return data
 
        