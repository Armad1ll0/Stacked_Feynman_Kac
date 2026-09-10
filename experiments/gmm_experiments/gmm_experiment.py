from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Normal

from experiments.gmm_experiments.gmm_analytic import (
    GMM, LinearObs, exact_posterior, make_default_problem, AnalyticEpsModel,
)
from experiments.gmm_experiments.gmm_metrics import all_metrics, mode_coverage
from tqdm import trange, tqdm

class SimpleScheduler:
    """Linear-beta DDPM scheduler exposing the fields the samplers read."""

    def __init__(self, T: int = 1000, beta_start=1e-4, beta_end=0.02):
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self._T = T
        self.timesteps = torch.arange(T - 1, -1, -1)

    def set_timesteps(self, n: int, device=None):
        step = max(1, self._T // n)
        ts = torch.arange(self._T - 1, -1, -step)[:n]
        self.timesteps = ts.to(device) if device is not None else ts
        return self.timesteps


@dataclass
class GMMTaskConfig:
    sigma_sq_scale: float = 1.0
    clamp_x0: bool = False
    use_fixed_sigma: bool = True


class GMMTask:

    def __init__(self, obs: LinearObs, cfg: GMMTaskConfig | None = None):
        self.obs = obs
        self.cfg = cfg or GMMTaskConfig()

    def log_potential(self, x0_hat, observation, metadata):
        sigma_sq = metadata["sigma_sq"]
        P = x0_hat.shape[0]
        x = x0_hat.reshape(P, -1)                       # (P, 2)
        y = observation.reshape(-1)[: self.obs.A.shape[0]]  # (m,)
        diff = x @ self.obs.A.T.to(x.device) - y[None, :].to(x.device)
        return -0.5 / sigma_sq * (diff ** 2).sum(dim=1)

    def make_sigma_sq(self, a_bar_t):
        if self.cfg.use_fixed_sigma:
            return torch.tensor(self.obs.sigma_y ** 2)
        return self.cfg.sigma_sq_scale * (1.0 - a_bar_t) / a_bar_t

    def degrade(self, x_clean, rng=None):
        x = x_clean.reshape(1, -1)
        y = self.obs.sample_y(x, generator=rng).reshape(-1)
        return y, {}

    @property
    def sigma_y(self):
        return float(self.obs.sigma_y)

    def A(self, x):
        P = x.shape[0]
        return (x.reshape(P, -1) @ self.obs.A.T.to(x.device)).reshape(P, 1, 1, -1)

    def At(self, y):
        P = y.shape[0]
        return (y.reshape(P, -1) @ self.obs.A.to(y.device)).reshape(P, 1, 1, 2)


def _abar_to_alpha_sigma(a_bar):
    return a_bar.sqrt(), (1.0 - a_bar).sqrt()


def _ddpm_posterior_mean_std(x_t, x_hat, alpha_t, alpha_s):
    a_bar_t, a_bar_s = alpha_t ** 2, alpha_s ** 2
    coef_x0 = alpha_s * (1.0 - a_bar_t / a_bar_s) / (1.0 - a_bar_t)
    coef_xt = (a_bar_t / a_bar_s).sqrt() * (1.0 - a_bar_s) / (1.0 - a_bar_t)
    mean = coef_x0 * x_hat + coef_xt * x_t
    var = (1.0 - a_bar_s) * (1.0 - a_bar_t / a_bar_s) / (1.0 - a_bar_t)
    return mean, var.clamp_min(1e-20).sqrt()


def _ess(log_w):
    w = torch.softmax(log_w, dim=0)
    return float(1.0 / (w.pow(2).sum() + 1e-12))

def run_exact_posterior(posterior: GMM, n: int, **kw):
    return posterior.sample(n)


def run_point_estimate(posterior: GMM, n: int, **kw):
    mean = (posterior.weights[:, None] * posterior.means).sum(0, keepdim=True)
    return mean.repeat(n, 1) + 1e-3 * torch.randn(n, 2)


def run_tds_2d(model, scheduler, task, y, n_particles, num_steps,
               ess_thresh=0.5, device="cpu", seed=0):

    torch.manual_seed(seed)
    scheduler.set_timesteps(num_steps)
    ts = scheduler.timesteps
    P = n_particles

    obs = y.reshape(1, 1, 1, -1).expand(P, 1, 1, y.numel()).contiguous()
    x = torch.randn(P, 1, 1, 2)
    log_w = torch.zeros(P)
    carry = None

    pbar = trange(len(ts) - 1, desc="TDS", leave=True)
    for i in pbar:
        t_, s_ = ts[i], ts[i + 1]
        a_bar_t = scheduler.alphas_cumprod[int(t_)]
        a_bar_s = scheduler.alphas_cumprod[int(s_)]
        alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
        alpha_s, sigma_s = _abar_to_alpha_sigma(a_bar_s)

        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            eps = model(x_t, t_)
            x_hat = (x_t - sigma_t * eps) / alpha_t
            sigma_sq = task.make_sigma_sq(a_bar_t)
            lp = task.log_potential(x_hat, obs, {"sigma_sq": sigma_sq})
            score_y = torch.autograd.grad(lp.sum(), x_t)[0]

        lp = lp.detach(); x_t = x_t.detach()
        x_hat = x_hat.detach(); score_y = score_y.detach()

        log_w = lp if carry is None else log_w + lp + carry

        current_ess = _ess(log_w)
        pbar.set_postfix(ess=f"{current_ess:.1f}/{P}")

        if _ess(log_w) < ess_thresh * P:
            w = torch.softmax(log_w, 0)
            idx = torch.multinomial(w, P, replacement=True)
            x_t, x_hat, lp, score_y = x_t[idx], x_hat[idx], lp[idx], score_y[idx]
            log_w = torch.full((P,), -math.log(P))

        mean, std = _ddpm_posterior_mean_std(x_t, x_hat, alpha_t, alpha_s)
        x_hat_sh = x_hat + (sigma_t ** 2 / alpha_t) * score_y
        mean_y, std_y = _ddpm_posterior_mean_std(x_t, x_hat_sh, alpha_t, alpha_s)

        noise = torch.randn_like(x_t)
        x_s = mean_y + std_y * noise

        lq = (-0.5 * ((x_s - mean) / std) ** 2 - torch.log(std)).sum((1, 2, 3))
        lqy = (-0.5 * ((x_s - mean_y) / std_y) ** 2 - torch.log(std_y)).sum((1, 2, 3))
        carry = lq - lqy - lp
        x = x_s.detach()

    return x.reshape(P, 2), log_w


def run_tds_hmc_2d(model, scheduler, task, y, n_particles, num_steps,
                   stop_at_step=5, hmc_iters=20, L=3, lambda_like=1.0,
                   hmc_step_scale=0.05, ess_thresh=0.5, seed=0):

    torch.manual_seed(seed)
    scheduler.set_timesteps(num_steps)
    ts = scheduler.timesteps
    P = n_particles
    obs = y.reshape(1, 1, 1, -1).expand(P, 1, 1, y.numel()).contiguous()

    x = torch.randn(P, 1, 1, 2)
    log_w = torch.zeros(P)
    carry = None
    n_loop = max(0, len(ts) - 1 - stop_at_step)
    pbar = trange(len(ts) - 1, desc="TDS", leave=True)
    for i in pbar:
        t_, s_ = ts[i], ts[i + 1]
        a_bar_t = scheduler.alphas_cumprod[int(t_)]
        a_bar_s = scheduler.alphas_cumprod[int(s_)]
        alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
        alpha_s, sigma_s = _abar_to_alpha_sigma(a_bar_s)

        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            eps = model(x_t, t_)
            x_hat = (x_t - sigma_t * eps) / alpha_t
            sigma_sq = task.make_sigma_sq(a_bar_t)
            lp = task.log_potential(x_hat, obs, {"sigma_sq": sigma_sq})
            score_y = torch.autograd.grad(lp.sum(), x_t)[0]

        lp = lp.detach(); x_t = x_t.detach()
        x_hat = x_hat.detach(); score_y = score_y.detach()
        log_w = lp if carry is None else log_w + lp + carry

        current_ess = _ess(log_w)
        pbar.set_postfix(ess=f"{current_ess:.1f}/{P}")

        if _ess(log_w) < ess_thresh * P:
            w = torch.softmax(log_w, 0)
            idx = torch.multinomial(w, P, replacement=True)
            x_t, x_hat, lp, score_y = x_t[idx], x_hat[idx], lp[idx], score_y[idx]
            log_w = torch.full((P,), -math.log(P))

        mean, std = _ddpm_posterior_mean_std(x_t, x_hat, alpha_t, alpha_s)
        x_hat_sh = x_hat + (sigma_t ** 2 / alpha_t) * score_y
        mean_y, std_y = _ddpm_posterior_mean_std(x_t, x_hat_sh, alpha_t, alpha_s)
        x_s = mean_y + std_y * torch.randn_like(x_t)
        lq = (-0.5 * ((x_s - mean) / std) ** 2 - torch.log(std)).sum((1, 2, 3))
        lqy = (-0.5 * ((x_s - mean_y) / std_y) ** 2 - torch.log(std_y)).sum((1, 2, 3))
        carry = lq - lqy - lp
        x = x_s.detach()

    # -- phase 2: HMC corrector at t_final --
    fi = len(ts) - 1 - stop_at_step
    t_final = ts[fi]
    a_bar_t = scheduler.alphas_cumprod[int(t_final)]
    alpha_t, sigma_t = _abar_to_alpha_sigma(a_bar_t)
    sigma_sq = task.make_sigma_sq(a_bar_t)
    beta_t = scheduler.betas[int(t_final)]
    h = hmc_step_scale * beta_t
    M = float(beta_t)
    M_dist = Normal(0.0, math.sqrt(max(M, 1e-12)))
    ess_threshold_abs = ess_thresh * P
    ess_trace = []

    def guided_score_and_logp(x_in):
        x_g = x_in.detach().requires_grad_(True)
        eps = model(x_g, t_final)
        x_hat_g = (x_g - sigma_t * eps) / alpha_t
        lp = task.log_potential(x_hat_g, obs, {"sigma_sq": sigma_sq})
        sl = torch.autograd.grad(lp.sum(), x_g)[0].detach()
        score = (-eps / sigma_t).detach() + lambda_like * sl
        return score, lp.detach()

    def leapfrog(x_in, p_in):
        g, _ = guided_score_and_logp(x_in)
        p_in = p_in + 0.5 * h * g
        x_in = x_in + h * p_in / M
        g, _ = guided_score_and_logp(x_in)
        p_in = p_in + 0.5 * h * g
        return x_in, p_in

    log_p_y_prev = None

    pbar2 = trange(hmc_iters, desc="HMC (phase 2)", leave=True)
    for _ in pbar2:
        if log_p_y_prev is None:
            _, log_p_y_prev = guided_score_and_logp(x)

        mom = M_dist.sample(x.shape)
        mom_start = mom.detach().clone()

        x_new = x.clone()
        for _ in range(L):
            x_new, mom = leapfrog(x_new, mom)

        _, log_p_y_new = guided_score_and_logp(x_new)
        reduce_dims = tuple(range(1, mom.ndim))
        log_q_M_new = M_dist.log_prob(-mom).sum(dim=reduce_dims)
        log_q_M_old = M_dist.log_prob(mom_start).sum(dim=reduce_dims)
        carry_log_hmc = log_q_M_new - log_q_M_old

        x = x_new
        log_w = log_w + (log_p_y_new - log_p_y_prev) + (carry_log_hmc) #/ x.numel()
        log_p_y_prev = log_p_y_new

        current_ess = _ess(log_w)
        ess_trace.append(current_ess)
        pbar2.set_postfix(ess=f"{current_ess:.1f}/{P}")

        if current_ess < ess_threshold_abs:
            w = torch.softmax(log_w, dim=0)
            idx = torch.multinomial(w, P, replacement=True)
            x = x[idx]
            log_p_y_prev = log_p_y_prev[idx]
            log_w = torch.full((P,), -math.log(P))

    return x.reshape(P, 2), log_w

