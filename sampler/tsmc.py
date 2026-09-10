from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.distributions import Normal
from diffusers import DDPMScheduler
from tqdm import tqdm
import math

from tasks.base import Task

log = logging.getLogger(__name__)

@dataclass
class SamplerConfig:
    num_particles: int = 8
    num_steps: int = 1000                 
    resample_ess_threshold: float = 0.5   
    verbose_every: int = 100              
    paste_observation: bool = False        
    hmc_iters: int = 1                   
    L: int = 3                      
    lambda_like: float = 4.0         
    hmc_step_scale: float = 0.1  
    stop_at_step: int = 5


@dataclass
class SamplerResult:
    """Holds all particles and diagnostics after a run."""
    particles: torch.Tensor      # (P, C, H, W) at t=0
    log_weights: torch.Tensor    # (P,) final (pre-resample) log weights
    ess_trace: list[float] = field(default_factory=list)

    @property
    def best(self) -> torch.Tensor:
        """MAP-ish: highest-weight particle."""
        return self.particles[self.log_weights.argmax()]

    @property
    def weighted_mean(self) -> torch.Tensor:
        w = torch.softmax(self.log_weights, dim=0)
        return (w[:, None, None, None] * self.particles).sum(dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (inlined to avoid new util modules)
# ──────────────────────────────────────────────────────────────────────────────

def _abar_to_alpha_sigma(a_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha_bar -> (alpha = sqrt(a_bar), sigma = sqrt(1 - a_bar))."""
    return a_bar.sqrt(), (1.0 - a_bar).sqrt()


def _ddpm_posterior(
    x_t: torch.Tensor,
    x_hat: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    alpha_s: torch.Tensor,
    sigma_s: torch.Tensor,
) -> Normal:

    a_bar_t = alpha_t ** 2
    a_bar_s = alpha_s ** 2
    coef_x0 = alpha_s * (1.0 - a_bar_t / a_bar_s) / (1.0 - a_bar_t)
    coef_xt = (a_bar_t / a_bar_s).sqrt() * (1.0 - a_bar_s) / (1.0 - a_bar_t)
    mean = coef_x0 * x_hat + coef_xt * x_t
    var = (1.0 - a_bar_s) * (1.0 - a_bar_t / a_bar_s) / (1.0 - a_bar_t)
    std = var.clamp_min(1e-20).sqrt()
    return Normal(mean, std.expand_as(mean))


def _ess(log_w: torch.Tensor) -> float:
    w = torch.softmax(log_w, dim=0)
    return (1.0 / (w.pow(2).sum() + 1e-12)).item()


def run_tds(
    model,
    scheduler: DDPMScheduler,
    task: Task,
    observation: torch.Tensor,   # (C, H, W) in [-1, 1]
    metadata: dict,              # task-specific: mask, y_lr_up, etc.
    cfg: SamplerConfig,
    device: torch.device | str = "cuda",
    progress_callback: Callable[[int, float], None] | None = None,
    stop_at_step: int = 0,       # NEW: number of final steps to skip (0 = run to end)
) -> SamplerResult:
    """
    Run the Twisted Diffusion Sampler for a single observation.

    Args:
        model:       pretrained DDPM UNet (predicts epsilon).
        scheduler:   DDPMScheduler with noise schedule.
        task:        a Task subclass defining log_potential and make_sigma_sq.
        observation: observed signal (C, H, W) in [-1, 1].
        metadata:    task-specific dict forwarded to task.log_potential.
        cfg:         SamplerConfig.
        device:      torch device.
        progress_callback: optional fn(t_int, ess).
        stop_at_step: number of trailing diffusion steps to skip. With the
            default of 0, the loop runs to t=0 as before. With stop_at_step=1,
            the loop exits one iteration early and the returned particles are
            still noisy at scheduler.timesteps[-2]. The paste-observation block
            is also skipped in this case (it would corrupt noisy particles).

    Returns:
        SamplerResult with particles, final log weights, and ESS trace.
    """
    device = torch.device(device)

    # ── Scheduler timesteps ─────────────────────────────────────────────
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps  # e.g. tensor([T-1, ..., 0])

    P = cfg.num_particles
    C = observation.shape[-3]
    H, W = metadata.get("hr_size", observation.shape[-2:])

    # Broadcast observation / metadata across P particles
    observation = observation.to(device).unsqueeze(0).expand(P, *observation.shape).contiguous()
    metadata = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in metadata.items()
    }

    ess_trace: list[float] = []
    ess_threshold_abs = cfg.resample_ess_threshold * P

    # ── Init particles from N(0, I) ─────────────────────────────────────
    x = torch.randn(P, C, H, W, device=device)
    carry_log_w: torch.Tensor | None = None
    log_w = torch.zeros(P, device=device)  # uniform initial weights
    last_log_w = log_w.clone()

    # ── Reverse loop over (t, s) pairs ──────────────────────────────────
    n_loop = max(0, len(timesteps) - 1 - stop_at_step)
    pbar = tqdm(range(n_loop), desc="TDS", leave=False)
    for step_i in pbar:
        t_ = timesteps[step_i]
        s_ = timesteps[step_i + 1]
        t_int = int(t_.item())
        s_int = int(s_.item())

        a_bar_t = scheduler.alphas_cumprod[t_int].to(device)
        a_bar_s = scheduler.alphas_cumprod[s_int].to(device)
        alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
        alpha_s, sigma_s = _abar_to_alpha_sigma(a_bar_s)

        # (a-b) x_hat prediction + autograd twist -----------------------
        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            out = model(x_t, t_)
            eps = out.sample if hasattr(out, "sample") else out

            x_hat = (x_t - sigma_t * eps) / alpha_t
            # print(x_hat.abs().mean())

            sigma_sq = task.make_sigma_sq(a_bar_t)
            step_metadata = {**metadata, "sigma_sq": sigma_sq}

            log_p_y = task.log_potential(x_hat, observation, step_metadata)  # (P,)
            score_y = torch.autograd.grad(log_p_y.sum(), x_t)[0]             # (P,C,H,W)
            # score_y = cfg.lambda_like*score_y

        log_p_y = log_p_y.detach()
        x_t = x_t.detach()
        x_hat = x_hat.detach()
        score_y = score_y.detach()

        # (d) accumulate weights -----------------------------------------
        if carry_log_w is None:
            log_w = log_p_y
        else:
            log_w = log_w + log_p_y + carry_log_w

        last_log_w = log_w
        current_ess = _ess(log_w)
        ess_trace.append(current_ess)
        pbar.set_postfix(ESS=f"{current_ess:.1f}/{P}")

        # Conditional multinomial resampling -----------------------------
        if current_ess < ess_threshold_abs:
            w = torch.softmax(log_w, dim=0)
            idx = torch.multinomial(w, P, replacement=True)
            x_t = x_t[idx]
            x_hat = x_hat[idx]
            log_p_y = log_p_y[idx]
            score_y = score_y[idx]
            log_w = torch.full((P,), -torch.log(torch.tensor(float(P))), device=device)

        # (c) build proposals & sample x_s -------------------------------
        q_s = _ddpm_posterior(x_t, x_hat, alpha_t, sigma_t, alpha_s, sigma_s)
        x_hat_shift = x_hat + (sigma_t ** 2 / alpha_t) * score_y
        q_s_y = _ddpm_posterior(x_t, x_hat_shift, alpha_t, sigma_t, alpha_s, sigma_s)

        x_s = q_s_y.sample()

        # (e) carry weight update ----------------------------------------
        log_q_xs   = q_s.log_prob(x_s).sum(dim=(1, 2, 3))
        log_q_xs_y = q_s_y.log_prob(x_s).sum(dim=(1, 2, 3))
        carry_log_w = log_q_xs - log_q_xs_y - log_p_y

        x = x_s.detach()

        # Logging --------------------------------------------------------
        if cfg.verbose_every > 0 and step_i % cfg.verbose_every == 0:
            log.info(f"  step={step_i:4d}  t={t_int:4d}  ESS={current_ess:.1f}/{P}")
        if progress_callback is not None:
            progress_callback(t_int, current_ess)

    # ── Final paste-observation (only when running to t=0) ─────────────
    if stop_at_step == 0 and cfg.paste_observation and "mask" in metadata:
        mask = metadata["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        x = observation * mask + x * (1.0 - mask)
    # print(x.abs().mean())

    return SamplerResult(
        particles=x,
        log_weights=last_log_w,
        ess_trace=ess_trace,
    )


# This is never used in the experiments as it is expensive to run. The refined version below is used instead. 
# Keeping it here for full reference. 
def run_tds_hmc(
    model,
    scheduler: DDPMScheduler,
    task: Task,
    observation: torch.Tensor,   # (C, H, W) in [-1, 1]
    metadata: dict,
    cfg: SamplerConfig,
    device: torch.device | str = "cuda",
    progress_callback: Callable[[int, float], None] | None = None,
) -> SamplerResult:
    """
    Twisted Diffusion Sampler with an HMC corrector block per diffusion step.

    Identical to `run_tds` up through the carry-weight update. After each
    DDPM proposal, we run `cfg.hmc_iters` rounds of `cfg.L` leapfrog steps
    using a guided score (prior + lambda_like * likelihood) and accumulate
    a momentum-based correction term `carry_log_hmc` that is added to
    `log_w` on the next iteration.

    See `run_tds` for the non-HMC variant.
    """
    import math

    device = torch.device(device)

    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

    P = cfg.num_particles
    C = observation.shape[-3]
    H, W = metadata.get("hr_size", observation.shape[-2:])

    observation = observation.to(device).unsqueeze(0).expand(P, *observation.shape).contiguous()
    metadata = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in metadata.items()
    }

    ess_trace: list[float] = []
    ess_threshold_abs = cfg.resample_ess_threshold * P

    x = torch.randn(P, C, H, W, device=device)
    carry_log_w: torch.Tensor | None = None
    carry_log_hmc: torch.Tensor | float = 0.0
    log_w = torch.zeros(P, device=device)
    last_log_w = log_w.clone()

    # ── Leapfrog step with guided score ────────────────────────────────
    def guided_score(x_in: torch.Tensor, t_tensor, sigma_t, alpha_t, step_metadata):
        """Score of prior + lambda_like * score of likelihood."""
        x_g = x_in.detach().requires_grad_(cfg.lambda_like != 0.0)

        with torch.no_grad():
            out = model(x_g, t_tensor)
            eps = out.sample if hasattr(out, "sample") else out
        score_prior = -eps / sigma_t

        if cfg.lambda_like == 0.0:
            return score_prior.detach()

        out2 = model(x_g, t_tensor)
        eps2 = out2.sample if hasattr(out2, "sample") else out2
        x_hat_g = (x_g - sigma_t * eps2) / alpha_t
        log_p_y = task.log_potential(x_hat_g, observation, step_metadata)
        score_like = torch.autograd.grad(log_p_y.sum(), x_g)[0]

        return (score_prior + cfg.lambda_like * score_like).detach()

    def leapfrog_step(x_in, p_in, h, M, t_tensor, sigma_t, alpha_t, step_metadata):
        p_in = p_in + 0.5 * h * guided_score(x_in, t_tensor, sigma_t, alpha_t, step_metadata)
        x_in = x_in + h * p_in / M
        p_in = p_in + 0.5 * h * guided_score(x_in, t_tensor, sigma_t, alpha_t, step_metadata)
        return x_in, p_in

    # ── Reverse loop ────────────────────────────────────────────────────
    pbar = tqdm(range(len(timesteps) - 1), desc="TDS+HMC", leave=False)
    for step_i in pbar:
        t_ = timesteps[step_i]
        s_ = timesteps[step_i + 1]
        t_int = int(t_.item())
        s_int = int(s_.item())

        a_bar_t = scheduler.alphas_cumprod[t_int].to(device)
        a_bar_s = scheduler.alphas_cumprod[s_int].to(device)
        alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
        alpha_s, sigma_s = _abar_to_alpha_sigma(a_bar_s)

        sigma_sq = task.make_sigma_sq(a_bar_t)
        step_metadata = {**metadata, "sigma_sq": sigma_sq}

        # (a-b) x_hat + autograd twist ----------------------------------
        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            out = model(x_t, t_)
            eps = out.sample if hasattr(out, "sample") else out
            x_hat = (x_t - sigma_t * eps) / alpha_t
            log_p_y = task.log_potential(x_hat, observation, step_metadata)
            score_y = torch.autograd.grad(log_p_y.sum(), x_t)[0]

        log_p_y = log_p_y.detach()
        x_t = x_t.detach()
        x_hat = x_hat.detach()
        score_y = score_y.detach()

        # (d) accumulate weights (with HMC carry) -----------------------
        if carry_log_w is None:
            log_w = log_p_y
        else:
            log_w = log_w + log_p_y + carry_log_w + carry_log_hmc

        last_log_w = log_w
        current_ess = _ess(log_w)
        ess_trace.append(current_ess)
        pbar.set_postfix(ESS=f"{current_ess:.1f}/{P}")

        if current_ess < ess_threshold_abs:
            w = torch.softmax(log_w, dim=0)
            idx = torch.multinomial(w, P, replacement=True)
            x_t = x_t[idx]
            x_hat = x_hat[idx]
            log_p_y = log_p_y[idx]
            score_y = score_y[idx]
            log_w = torch.full((P,), -torch.log(torch.tensor(float(P))), device=device)

        # (c) build proposals & sample x_s ------------------------------
        q_s = _ddpm_posterior(x_t, x_hat, alpha_t, sigma_t, alpha_s, sigma_s)
        x_hat_shift = x_hat + (sigma_t ** 2 / alpha_t) * score_y
        q_s_y = _ddpm_posterior(x_t, x_hat_shift, alpha_t, sigma_t, alpha_s, sigma_s)
        x_s = q_s_y.sample()

        log_q_xs   = q_s.log_prob(x_s).sum(dim=(1, 2, 3))
        log_q_xs_y = q_s_y.log_prob(x_s).sum(dim=(1, 2, 3))
        carry_log_w = log_q_xs - log_q_xs_y - log_p_y

        x = x_s.detach()
        x_hmc = x.detach()

        # ── HMC corrector block ───────────────────────────────────────
        beta_t = scheduler.betas[t_int].to(device)
        h = cfg.hmc_step_scale * beta_t
        M = beta_t.item()
        M_dist = Normal(0.0, math.sqrt(M))
        carry_log_hmc = torch.zeros(P, device=device)

        for _ in range(cfg.hmc_iters):
            mom = M_dist.sample(x_hmc.shape).to(device)
            mom_start = mom.detach().clone()

            for _ in range(cfg.L):
                x_hmc, mom = leapfrog_step(
                    x_hmc, mom, h, M, t_, sigma_t, alpha_t, step_metadata
                )

            # Momentum log-prob correction, averaged over per-particle dims
            reduce_dims = tuple(range(1, mom.ndim))
            log_q_M_new = M_dist.log_prob(-mom).mean(dim=reduce_dims)
            log_q_M_old = M_dist.log_prob(mom_start).mean(dim=reduce_dims)
            carry_log_hmc = carry_log_hmc + (log_q_M_new - log_q_M_old)

        x = x_hmc.detach()

        if cfg.verbose_every > 0 and step_i % cfg.verbose_every == 0:
            log.info(f"  step={step_i:4d}  t={t_int:4d}  ESS={current_ess:.1f}/{P}")
        if progress_callback is not None:
            progress_callback(t_int, current_ess)

    if cfg.paste_observation and "mask" in metadata:
        mask = metadata["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        x = observation * mask + x * (1.0 - mask)

    # ── Final projection into the task's valid range (no-op unless defined) ──
    x = task.postprocess_particles(x)

    return SamplerResult(
        particles=x,
        log_weights=last_log_w,
        ess_trace=ess_trace,
    )

@dataclass
class TDSThenHMCResult:
    particles_tds: torch.Tensor      # (P, C, H, W) post-resample, pre-HMC
    particles_hmc: torch.Tensor      # (P, C, H, W) after HMC corrector
    resample_idx: torch.Tensor       # (P,) indices used for the forced resample
    log_weights: torch.Tensor        # final HMC weights
    ess_trace: list[float]
 
    @property
    def best(self) -> torch.Tensor:
        return self.particles_hmc[self.log_weights.argmax()]


def run_tds_then_hmc_denoised(
    model,
    scheduler,
    task,
    observation,
    metadata,
    cfg,
    device="cuda",
    progress_callback=None,
    stop_at_step: int = 1,    # how many trailing diffusion steps to skip
):
    """
    Run TDS up to (but not including) the final `stop_at_step` diffusion
    iterations, force-resample, run HMC at that noise level using the
    denoiser's score as the prior gradient (plus the autograd likelihood
    score), then do one final denoising step from t_final to t=0.
    """
 
    device = torch.device(device)
    # print('Here', stop_at_step, cfg.hmc_iters, cfg.lambda_like, cfg.L)
 
    # ── Phase 1: TDS, stopping `stop_at_step` iterations early ──
    result_tds = run_tds(
        model=model, scheduler=scheduler, task=task,
        observation=observation, metadata=metadata,
        cfg=cfg, device=device, progress_callback=progress_callback,
        stop_at_step=cfg.stop_at_step,
    )
 
    x = result_tds.particles                # noisy particles at t_final
    log_w = result_tds.log_weights
    ess_trace = list(result_tds.ess_trace)
    P = cfg.num_particles
 
    C = observation.shape[-3]
    H, W = metadata.get("hr_size", observation.shape[-2:])
    obs_dev = observation.to(device).unsqueeze(0).expand(P, *observation.shape).contiguous()
    meta_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in metadata.items()
    }
 
    # ── Identify t_final and the noise level we are at ──
    scheduler.set_timesteps(cfg.num_steps, device=device)
    final_step_idx = len(scheduler.timesteps) - 1 - cfg.stop_at_step
    t_final = scheduler.timesteps[final_step_idx]
    t_final_int = int(t_final.item())
    a_bar_t = scheduler.alphas_cumprod[t_final_int].to(device)
    alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
    sigma_sq = task.make_sigma_sq(a_bar_t)
    step_metadata = {**meta_dev, "sigma_sq": sigma_sq}
 
    # ── Force-resample at t_final ──
    w = torch.softmax(log_w, dim=0)
    resample_idx = torch.multinomial(w, P, replacement=True)
    x = x[resample_idx]
    particles_tds_noisy = x.detach().clone()  # snapshot in noisy space
    log_w = torch.full((P,), -math.log(P), device=device)
    ess_trace.append(_ess(log_w))
 
    # ── HMC corrector with denoiser prior + likelihood ──
    beta_t = scheduler.betas[t_final_int].to(device)
    h = cfg.hmc_step_scale * beta_t#**(1.5)*6
    M = beta_t.item()
    M_dist = Normal(0.0, math.sqrt(max(M, 1e-12)))
    ess_threshold_abs = cfg.resample_ess_threshold * P
    #ess_threshold_abs = P*1.1
 
    def guided_score_and_logp(x_in):
            """Returns (score_prior + lambda_like * score_like, log_p_y)."""
            with torch.enable_grad():
                x_g = x_in.detach().requires_grad_(True)
                out = model(x_g, t_final)
                eps = out.sample if hasattr(out, "sample") else out
                x_hat_g = (x_g - sigma_t * eps) / alpha_t
                log_p = task.log_potential(x_hat_g, obs_dev, step_metadata)
                score_like = torch.autograd.grad(log_p.sum(), x_g)[0].detach()
            score_prior = (-eps / sigma_t).detach()
            return (score_prior + cfg.lambda_like * score_like), log_p.detach()
 
    def leapfrog(x_in, p_in):
        g, _ = guided_score_and_logp(x_in)
        p_in = p_in + 0.5 * h * g
        x_in = x_in + h * p_in / M
        g, _ = guided_score_and_logp(x_in)
        p_in = p_in + 0.5 * h * g
        return x_in, p_in
 
    # baseline log p(y | x_hat) using current particles
    _, log_p_y_prev = guided_score_and_logp(x)
    log_w = log_p_y_prev.clone()
 
    pbar = tqdm(range(cfg.hmc_iters), desc=f"HMC@t={t_final_int}", leave=False)
    for it in pbar:
        mom = M_dist.sample(x.shape).to(device)
        mom_start = mom.detach().clone()
 
        x_new = x.clone()
        for _ in range(cfg.L):
            x_new, mom = leapfrog(x_new, mom)
 
        _, log_p_y_new = guided_score_and_logp(x_new)
        reduce_dims = tuple(range(1, mom.ndim))
        log_q_M_new = M_dist.log_prob(-mom).sum(dim=reduce_dims)
        log_q_M_old = M_dist.log_prob(mom_start).sum(dim=reduce_dims)
        carry_log_hmc = log_q_M_new - log_q_M_old
 
        x = x_new
        # print('Hereeeee', cfg.lambda_like)
        # print(x.numel(), log_w + (log_p_y_new - log_p_y_prev) + carry_log_hmc)
        log_w = log_w + ((log_p_y_new - log_p_y_prev) + carry_log_hmc)/x.numel()
        log_p_y_prev = log_p_y_new

        current_ess = _ess(log_w)
        ess_trace.append(current_ess)
        pbar.set_postfix(ESS=f"{current_ess:.1f}/{P}")

        if current_ess < ess_threshold_abs:
            w = torch.softmax(log_w, dim=0)
            idx = torch.multinomial(w, P, replacement=True)
            x = x[idx]
            log_p_y_prev = log_p_y_prev[idx]
            log_w = torch.full((P,), -math.log(P), device=device)
 
    # ── Final denoise step (Tweedie) for both snapshots ──
    def final_denoise(x_noisy):
        with torch.no_grad():
            out = model(x_noisy, t_final)
            eps = out.sample if hasattr(out, "sample") else out
        return (x_noisy - sigma_t * eps) / alpha_t
 
    particles_tds = final_denoise(particles_tds_noisy)
    x = final_denoise(x)
 
    # ── Paste-observation on the now-clean particles ──
    if cfg.paste_observation and "mask" in meta_dev:
        mask = meta_dev["mask"]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        x = obs_dev * mask + x * (1.0 - mask)
        particles_tds = obs_dev * mask + particles_tds * (1.0 - mask)

    # ── Final projection into the task's valid range (no-op unless defined) ──
    x = task.postprocess_particles(x)
    particles_tds = task.postprocess_particles(particles_tds)

    return TDSThenHMCResult(
        particles_tds=particles_tds.detach(),
        particles_hmc=x.detach(),
        resample_idx=resample_idx.detach().cpu(),
        log_weights=log_w.detach(),
        ess_trace=ess_trace,
    )