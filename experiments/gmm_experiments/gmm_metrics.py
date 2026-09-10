from __future__ import annotations

import numpy as np
import torch


def sliced_wasserstein(X: torch.Tensor, Y: torch.Tensor,
                       n_proj: int = 512, p: int = 2,
                       generator=None) -> float:

    d = X.shape[1]
    theta = torch.randn(n_proj, d, generator=generator)
    theta = theta / theta.norm(dim=1, keepdim=True)

    Xp = X @ theta.T          # (N, n_proj)
    Yp = Y @ theta.T          # (M, n_proj)

    Xs, _ = torch.sort(Xp, dim=0)
    Ys, _ = torch.sort(Yp, dim=0)

    n, m = Xs.shape[0], Ys.shape[0]
    if n == m:
        diff = (Xs - Ys).abs()
    else:
        # quantile-match onto a common grid
        q = torch.linspace(0, 1, min(n, m))
        xi = (q * (n - 1)).round().long()
        yi = (q * (m - 1)).round().long()
        diff = (Xs[xi] - Ys[yi]).abs()

    return (diff ** p).mean().pow(1.0 / p).item()


def _pdist2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.cdist(A, B, p=2) ** 2


def median_bandwidth(X: torch.Tensor, Y: torch.Tensor, max_n: int = 2000) -> float:
    Z = torch.cat([X[:max_n], Y[:max_n]], dim=0)
    d2 = _pdist2(Z, Z)
    iu = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
    med = d2[iu[0], iu[1]].median()
    return float(med.clamp_min(1e-12).sqrt())


def mmd_rbf(X: torch.Tensor, Y: torch.Tensor,
            bandwidth: float | None = None,
            max_n: int = 4000, unbiased: bool = True) -> float:
    X = X[:max_n]
    Y = Y[:max_n]
    if bandwidth is None:
        bandwidth = median_bandwidth(X, Y)
    h2 = 2.0 * bandwidth ** 2

    Kxx = torch.exp(-_pdist2(X, X) / h2)
    Kyy = torch.exp(-_pdist2(Y, Y) / h2)
    Kxy = torch.exp(-_pdist2(X, Y) / h2)

    n, m = X.shape[0], Y.shape[0]
    if unbiased:
        Kxx = Kxx - torch.diag(torch.diag(Kxx))
        Kyy = Kyy - torch.diag(torch.diag(Kyy))
        mmd2 = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) \
               - 2.0 * Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(mmd2.clamp_min(0.0).sqrt())


def energy_distance(X: torch.Tensor, Y: torch.Tensor, max_n: int = 4000) -> float:
    X, Y = X[:max_n], Y[:max_n]
    dxy = torch.cdist(X, Y).mean()
    dxx = torch.cdist(X, X).mean()
    dyy = torch.cdist(Y, Y).mean()
    return float((2 * dxy - dxx - dyy).clamp_min(0.0))


def mode_coverage(X: torch.Tensor, posterior) -> np.ndarray:
    diff = X[:, None, :] - posterior.means[None, :, :]          # (N,K,d)
    prec = torch.linalg.inv(posterior.covs)                     # (K,d,d)
    maha = torch.einsum("nki,kij,nkj->nk", diff, prec, diff)
    logdet = torch.logdet(posterior.covs)                       # (K,)
    logn = -0.5 * (maha + logdet)
    logw = torch.log(posterior.weights.clamp_min(1e-30))[None, :] + logn
    assign = logw.argmax(dim=1)
    K = posterior.means.shape[0]
    counts = torch.bincount(assign, minlength=K).float()
    return (counts / counts.sum()).numpy()


def mode_coverage_error(X: torch.Tensor, posterior) -> float:
    emp = mode_coverage(X, posterior)
    true = posterior.weights.numpy()
    return float(0.5 * np.abs(emp - true).sum())


def all_metrics(X: torch.Tensor, Y_true: torch.Tensor, posterior,
                generator=None) -> dict:
    return {
        "sw": sliced_wasserstein(X, Y_true, generator=generator),
        "mmd": mmd_rbf(X, Y_true),
        "energy": energy_distance(X, Y_true),
        "mode_cov_err": mode_coverage_error(X, posterior),
    }