from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from diffusers import DDPMScheduler

from sampler.schedule import (
    get_schedule_tensors,
    predict_x0,
    q_posterior_mean,
    posterior_variance,
)
from sampler.smc import ess, get_resampler
from tasks.base import Task

log = logging.getLogger(__name__)


@dataclass
class SamplerConfig:
    num_particles: int = 8
    resampler: str = "systematic"  
    resample_ess_threshold: float = 0.5   
    verbose_every: int = 100             
    paste_observation: bool = False        


@dataclass
class SamplerResult:
    particles: torch.Tensor      # (P, C, H, W)  at t=0
    log_weights: torch.Tensor    # (P,)  final weights
    ess_trace: list[float] = field(default_factory=list)

    @property
    def best(self) -> torch.Tensor:
        return self.particles[self.log_weights.argmax()]

    @property
    def weighted_mean(self) -> torch.Tensor:
        w = torch.softmax(self.log_weights, dim=0)  # (P,)
        return (w[:, None, None, None] * self.particles).sum(dim=0)


def run_tds(
    model,
    scheduler: DDPMScheduler,
    task: Task,
    observation: torch.Tensor,   # (C, H, W)
    metadata: dict,              # task-specific: mask, y_lr_up, etc.
    cfg: SamplerConfig,
    device: torch.device | str = "cuda",
    progress_callback: Callable[[int, float], None] | None = None,
) -> SamplerResult:

    device = torch.device(device)
    alphas_cumprod, betas = get_schedule_tensors(scheduler, device)
    T = len(betas)
    P = cfg.num_particles
    C = observation.shape[-3]
    H, W = metadata.get("hr_size", observation.shape[-2:])

    observation = observation.to(device)
    metadata = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in metadata.items()}

    resample_fn = get_resampler(cfg.resampler)
    ess_threshold = cfg.resample_ess_threshold * P
    ess_trace = []

    # ── Step 1: initialize particles from N(0, I) ──────────────────────
    x_t = torch.randn(P, C, H, W, device=device)
    log_weights = torch.zeros(P, device=device)  # uniform weights initially

    # ── Step 2: reverse loop ───────────────────────────────────────────
    for t in reversed(range(T)):
        a_bar_t = alphas_cumprod[t]

        # Per-step observation noise (schedule-dependent)
        sigma_sq = task.make_sigma_sq(a_bar_t)
        step_metadata = {**metadata, "sigma_sq": sigma_sq}

        # (a) Predict x_0 from current noisy particles
        x0_hat = predict_x0(model, x_t, t, alphas_cumprod, device)

        if t > 0:
            post_var = posterior_variance(t, alphas_cumprod, betas)

            # (b) Twist: compute shifted x0 and log-potential
            twisted_x0, log_pot = task.compute_twisted_x0(
                x_t, x0_hat, observation, step_metadata, a_bar_t
            )

            # (c) Reverse means under twisted and untwisted proposals
            twisted_mean   = q_posterior_mean(twisted_x0, x_t, t, alphas_cumprod, betas)
            untwisted_mean = q_posterior_mean(x0_hat,     x_t, t, alphas_cumprod, betas)

            # (d) Sample x_{t-1}
            noise = torch.randn_like(x_t)
            x_tm1 = twisted_mean + post_var.sqrt() * noise

            # (e) Incremental importance weights (log space)
            log_p_untwisted = -0.5 / post_var * ((x_tm1 - untwisted_mean) ** 2).sum(dim=(1, 2, 3))
            log_p_twisted   = -0.5 / post_var * ((x_tm1 - twisted_mean)   ** 2).sum(dim=(1, 2, 3))
            log_w_increment = log_p_untwisted + log_pot - log_p_twisted - log_weights
            log_weights     = log_pot   # carry potential forward

            current_ess = ess(log_w_increment)
            ess_trace.append(current_ess)

            # (f) Conditionally resample
            if cfg.resample_ess_threshold == 0 or current_ess < ess_threshold:
                idx = resample_fn(log_w_increment)
                x_t = x_tm1[idx]
                log_weights = log_weights[idx]
            else:
                x_t = x_tm1

        else:
            # t=0: deterministic final step
            x_t = q_posterior_mean(x0_hat, x_t, t=0,
                                   alphas_cumprod=alphas_cumprod, betas=betas)

            if cfg.paste_observation and "mask" in metadata:
                mask = metadata["mask"]
                x_t = observation * mask + x_t * (1 - mask)

            log_weights = task.log_potential(x0_hat, observation, step_metadata)

        # Logging / callback
        if cfg.verbose_every > 0 and t % cfg.verbose_every == 0:
            e = ess_trace[-1] if ess_trace else P
            log.info(f"  t={t:4d}  ESS={e:.1f}/{P}")
            if progress_callback:
                progress_callback(t, e)

    return SamplerResult(particles=x_t, log_weights=log_weights, ess_trace=ess_trace)