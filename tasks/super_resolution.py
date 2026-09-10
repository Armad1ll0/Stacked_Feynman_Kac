from dataclasses import dataclass
import torch
import torch.nn.functional as F

from tasks.base import Task, TaskConfig


@dataclass
class SuperResConfig(TaskConfig):
    scale_factor: int = 4    

class SuperResTask(Task):

    def __init__(self, cfg: SuperResConfig):
        super().__init__(cfg)
        self.cfg: SuperResConfig = cfg

    def degrade(self, x_clean, rng=None):
        was_3d = x_clean.dim() == 3
        x = x_clean.unsqueeze(0) if was_3d else x_clean
        y_lr = self._downsample(x)
        y_lr = y_lr.squeeze(0) if was_3d else y_lr

        metadata = {
            "y_lr": y_lr,
            "scale_factor": self.cfg.scale_factor,
            "sigma_y": self.sigma_y,
            "hr_size": tuple(x.shape[-2:]),
        }
        return y_lr, metadata

    def log_potential(self, x0_hat, observation, metadata):
        y = observation if observation.dim() == 4 else observation.unsqueeze(0)
        sigma_sq = metadata["sigma_sq"]
        x0_lr = self._downsample(x0_hat)      
        diff = x0_lr - y                    
        return -0.5 / sigma_sq * (diff ** 2).sum(dim=(1, 2, 3))

    def _downsample(self, x):
        was_3d = x.dim() == 3
        if was_3d:
            x = x.unsqueeze(0)
        r = self.cfg.scale_factor
        H, W = x.shape[-2:]
        if H % r or W % r:
            raise ValueError(
                f"scale_factor {r} does not divide image size {(H, W)}; "
                "avg_pool2d would silently drop the trailing rows/columns "
                "and change the effective operator."
            )
        out = F.avg_pool2d(x, kernel_size=r, stride=r)
        return out.squeeze(0) if was_3d else out

    def _upsample_for_display(self, y_lr, target_size):
        was_3d = y_lr.dim() == 3
        if was_3d:
            y_lr = y_lr.unsqueeze(0)
        out = F.interpolate(y_lr, size=tuple(target_size), mode="nearest")
        return out.squeeze(0) if was_3d else out