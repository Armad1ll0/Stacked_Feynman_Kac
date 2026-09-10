from dataclasses import dataclass, field
from typing import Tuple
import torch

from tasks.base import Task, TaskConfig


@dataclass
class InpaintingConfig(TaskConfig):
    mask_type: str = "center_box"
    hole_fraction: float = 0.4  


class InpaintingTask(Task):

    def __init__(self, cfg: InpaintingConfig):
        super().__init__(cfg)
        self.cfg: InpaintingConfig = cfg

    def log_potential(self, x0_hat, observation, metadata):
        mask = metadata["mask"]          
        sigma_sq = metadata["sigma_sq"]
        diff = mask * (x0_hat - observation)
        return -0.5 / sigma_sq * (diff ** 2).sum(dim=(1, 2, 3))

    @property
    def sigma_y(self) -> float:
        return float(getattr(self.cfg, "sigma_y", 0.0))

    def make_sigma_sq(self, a_bar_t: torch.Tensor) -> torch.Tensor:
        return self.sigma_y ** 2 + self.cfg.sigma_sq_scale * (1.0 - a_bar_t) / a_bar_t

    def degrade(self, x_clean, rng=None):
        C, H, W = x_clean.shape
        mask = self._make_mask(C, H, W, rng, device=x_clean.device)
        y = mask * x_clean
        return y, {"mask": mask, "sigma_y": self.sigma_y, "hr_size": (H, W)}

    def _make_mask(self, C, H, W, rng, device):
        mask = torch.ones(C, H, W, device=device)
        t = self.cfg.mask_type
        f = self.cfg.hole_fraction

        if t == "center_box":
            h0 = int(H * (1 - f) / 2)
            h1 = int(H * (1 + f) / 2)
            w0 = int(W * (1 - f) / 2)
            w1 = int(W * (1 + f) / 2)
            mask[:, h0:h1, w0:w1] = 0.0

        elif t == "random_box":
            side_h = int(H * f)
            side_w = int(W * f)
            h0 = torch.randint(0, H - side_h + 1, (1,), generator=rng).item()
            w0 = torch.randint(0, W - side_w + 1, (1,), generator=rng).item()
            mask[:, h0:h0 + side_h, w0:w0 + side_w] = 0.0

        elif t == "random_pixels":
            drop = torch.rand(1, H, W, generator=rng, device=device) < f
            mask = mask * (~drop).float()

        elif t == "half":
            mask[:, :, W // 2:] = 0.0

        else:
            raise ValueError(f"Unknown mask_type: {t!r}")

        return mask