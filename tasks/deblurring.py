from dataclasses import dataclass
import torch
import torch.nn.functional as F

from tasks.base import Task, TaskConfig


@dataclass
class DeblurringConfig(TaskConfig):
    kernel_size: int = 61      
    blur_sigma: float = 3.0    


class DeblurringTask(Task):

    def __init__(self, cfg: DeblurringConfig):
        super().__init__(cfg)
        self.cfg: DeblurringConfig = cfg
        if cfg.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {cfg.kernel_size}")
        self._kernel: torch.Tensor | None = None  # lazily built per device

    def log_potential(self, x0_hat, observation, metadata):
        y = observation if observation.dim() == 4 else observation.unsqueeze(0)
        sigma_sq = metadata["sigma_sq"]
        blurred = self._blur(x0_hat)
        diff = blurred - y
        return -0.5 / sigma_sq * (diff ** 2).sum(dim=(1, 2, 3))

    def degrade(self, x_clean, rng=None):
        was_3d = x_clean.dim() == 3
        x = x_clean.unsqueeze(0) if was_3d else x_clean
        y = self._blur(x)
        return (y.squeeze(0) if was_3d else y), {
            "sigma_y": self.sigma_y,
            "hr_size": tuple(x.shape[-2:]),
        }

    @property
    def sigma_y(self) -> float:
        return float(getattr(self.cfg, "sigma_y", 0.0))

    def make_sigma_sq(self, a_bar_t):
        return self.sigma_y ** 2 + self.cfg.sigma_sq_scale * (1.0 - a_bar_t) / a_bar_t

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        was_3d = x.dim() == 3
        if was_3d:
            x = x.unsqueeze(0)

        kernel = self._get_kernel(x.device, x.dtype)
        C = x.shape[1]
        k = kernel.expand(C, 1, -1, -1)
        pad = self.cfg.kernel_size // 2
        out = F.conv2d(x, k, padding=pad, groups=C)

        return out.squeeze(0) if was_3d else out

    def _get_kernel(self, device, dtype) -> torch.Tensor:
        device = torch.device(device)
        if self._kernel is not None and self._kernel.device == device:
            return self._kernel.to(dtype)

        K = self.cfg.kernel_size
        sigma = self.cfg.blur_sigma
        coords = torch.arange(K, dtype=torch.float32) - K // 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d /= g1d.sum()
        g2d = g1d[:, None] * g1d[None, :]   # outer product
        self._kernel = g2d.unsqueeze(0).unsqueeze(0).to(device)
        return self._kernel.to(dtype)