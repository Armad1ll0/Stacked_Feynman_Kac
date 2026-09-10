from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GMM:
    weights: torch.Tensor     # (K,)   mixture weights, sum to 1
    means: torch.Tensor       # (K, 2)
    covs: torch.Tensor        # (K, 2, 2)

    def __post_init__(self):
        self.weights = self.weights / self.weights.sum()
        self.K = self.means.shape[0]
        self.d = self.means.shape[1]

    def sample(self, n: int, generator=None) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True,
                                generator=generator)
        L = torch.linalg.cholesky(self.covs)            # (K,2,2)
        z = torch.randn(n, self.d, generator=generator)
        return self.means[idx] + torch.einsum("nij,nj->ni", L[idx], z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 2) -> (N,)"""
        diff = x[:, None, :] - self.means[None, :, :]          # (N,K,2)
        prec = torch.linalg.inv(self.covs)                     # (K,2,2)
        maha = torch.einsum("nki,kij,nkj->nk", diff, prec, diff)
        logdet = torch.logdet(self.covs)                       # (K,)
        logn = -0.5 * (maha + logdet + self.d * np.log(2 * np.pi))
        return torch.logsumexp(torch.log(self.weights)[None, :] + logn, dim=1)

    def score_noised(self, x: torch.Tensor, alpha_t, sigma_t) -> torch.Tensor:
        a = float(alpha_t)
        s2 = float(sigma_t) ** 2
        means_t = a * self.means                                    # (K,2)
        covs_t = (a ** 2) * self.covs + s2 * torch.eye(self.d)[None]  # (K,2,2)

        diff = x[:, None, :] - means_t[None, :, :]                  # (N,K,2)
        prec = torch.linalg.inv(covs_t)                             # (K,2,2)
        maha = torch.einsum("nki,kij,nkj->nk", diff, prec, diff)
        logdet = torch.logdet(covs_t)
        logn = -0.5 * (maha + logdet + self.d * np.log(2 * np.pi))
        logw = torch.log(self.weights)[None, :] + logn              # (N,K)
        resp = torch.softmax(logw, dim=1)                           # (N,K)

        # grad log N_k = -prec_k (x - mean_k)
        g = -torch.einsum("kij,nkj->nki", prec, diff)               # (N,K,2)
        return (resp[:, :, None] * g).sum(dim=1)                    # (N,2)


@dataclass
class LinearObs:
    A: torch.Tensor           # (m, 2)
    sigma_y: float

    def sample_y(self, x: torch.Tensor, generator=None) -> torch.Tensor:
        """x: (N,2) -> y: (N,m)"""
        mean = x @ self.A.T
        return mean + self.sigma_y * torch.randn(mean.shape, generator=generator)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log p(y | x) for x: (N,2), y: (m,) -> (N,)"""
        diff = x @ self.A.T - y[None, :]
        return -0.5 / self.sigma_y ** 2 * (diff ** 2).sum(dim=1)

    def grad_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """grad_x log p(y|x): (N,2)"""
        diff = x @ self.A.T - y[None, :]                 # (N,m)
        return -(diff @ self.A) / self.sigma_y ** 2      # (N,2)


def exact_posterior(prior: GMM, obs: LinearObs, y: torch.Tensor) -> GMM:
    A = obs.A
    sy2 = obs.sigma_y ** 2
    m = A.shape[0]
    d = prior.d

    S_inv = torch.linalg.inv(prior.covs)                     # (K,2,2)
    AtA = (A.T @ A) / sy2                                    # (2,2)
    S_post = torch.linalg.inv(S_inv + AtA[None])             # (K,2,2)

    Aty = (A.T @ y) / sy2                                    # (2,)
    rhs = torch.einsum("kij,kj->ki", S_inv, prior.means) + Aty[None]  # (K,2)
    mu_post = torch.einsum("kij,kj->ki", S_post, rhs)        # (K,2)

    # marginal likelihood of y under each component
    mean_y = prior.means @ A.T                               # (K,m)
    cov_y = torch.einsum("ij,kjl,ml->kim", A, prior.covs, A) \
            + sy2 * torch.eye(m)[None]                       # (K,m,m)
    diff = (y[None, :] - mean_y)                             # (K,m)
    prec_y = torch.linalg.inv(cov_y)
    maha = torch.einsum("ki,kij,kj->k", diff, prec_y, diff)
    logdet = torch.logdet(cov_y)
    log_marg = -0.5 * (maha + logdet + m * np.log(2 * np.pi))

    logw = torch.log(prior.weights) + log_marg
    w_post = torch.softmax(logw, dim=0)

    return GMM(weights=w_post, means=mu_post, covs=S_post)


class AnalyticEpsModel:

    def __init__(self, prior: GMM, scheduler):
        self.prior = prior
        self.scheduler = scheduler
        self.nfe = 0   # count function evaluations for fair-budget reporting

    def _alpha_sigma(self, t_int: int):
        a_bar = self.scheduler.alphas_cumprod[t_int]
        return a_bar.sqrt(), (1.0 - a_bar).sqrt()

    def __call__(self, x, t):
        self.nfe += 1
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        alpha_t, sigma_t = self._alpha_sigma(t_int)

        shape = x.shape
        x_flat = x.reshape(shape[0], -1)                  # (P, 2)
        score = self.prior.score_noised(x_flat, alpha_t, sigma_t)
        eps = -float(sigma_t) * score
        return eps.reshape(shape)

    def eval(self):
        return self

    def to(self, *a, **k):
        return self


def make_default_problem(seed: int = 0, center_scale: float = 3.0,
                         spacing: float = 1.0, base_var: float = 0.25,
                         grid_side: int = 3):

    g = torch.Generator().manual_seed(seed)

    K = grid_side ** 2
    coords = (torch.arange(grid_side, dtype=torch.float32)
              - (grid_side - 1) / 2) * spacing
    means = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1)
    means = means.reshape(-1, 2)

    covs = base_var * torch.eye(2)[None].repeat(K, 1, 1)
    if grid_side % 2 == 1:                       # exact centre exists
        centre_idx = (K - 1) // 2
        covs[centre_idx] = center_scale * base_var * torch.eye(2)

    weights = torch.ones(K) / K
    prior = GMM(weights=weights, means=means, covs=covs)

    A = torch.tensor([[1.0, 1.0]])
    obs = LinearObs(A=A, sigma_y=0.1)

    return prior, obs, g