from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TaskConfig:
    sigma_sq_scale: float = 1.0  
    clamp_x0: bool = False        
    sigma_y: float = 0.0


class Task(ABC):

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg

    @property
    def sigma_y(self) -> float:
        return float(getattr(self.cfg, "sigma_y", 0.0))

    def make_sigma_sq(self, a_bar_t: torch.Tensor) -> torch.Tensor:
        return self.sigma_y ** 2 + self.cfg.sigma_sq_scale * (1.0 - a_bar_t) / a_bar_t

    @abstractmethod
    def log_potential(
        self,
        x0_hat: torch.Tensor,      
        observation: torch.Tensor,  
        metadata: dict,             
    ) -> torch.Tensor:             
        ...

    @abstractmethod
    def degrade(
        self,
        x_clean: torch.Tensor,  
        rng: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        ...

    def compute_twisted_x0(
        self,
        x_t: torch.Tensor,            # (P, C, H, W) current noisy sample
        x0_hat: torch.Tensor,         # (P, C, H, W) predicted clean image
        observation: torch.Tensor,    # observed signal
        metadata: dict,               # passed through to log_potential
        a_bar_t: torch.Tensor,        # scalar, alpha_bar at timestep t
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # --- log-potential and its gradient w.r.t. x0_hat ---
        x0_hat_g = x0_hat.detach().requires_grad_(True)
        log_pot = self.log_potential(x0_hat_g, observation, metadata)

        (grad_x0,) = torch.autograd.grad(
            log_pot.sum(), x0_hat_g, allow_unused=True
        )
        if grad_x0 is None:
            grad_x0 = torch.zeros_like(x0_hat)
        grad_x0 = grad_x0.detach()

        grad_xt = grad_x0 / a_bar_t.sqrt()

        shift = (1.0 - a_bar_t) / a_bar_t.sqrt() * grad_xt
        twisted_x0 = x0_hat.detach() + shift

        if self.cfg.clamp_x0:
            twisted_x0 = twisted_x0.clamp(-1, 1)

        return twisted_x0, log_pot.detach()

    def postprocess_particles(self, x: torch.Tensor) -> torch.Tensor:
        return x