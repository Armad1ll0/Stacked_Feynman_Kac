from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

_EPS = 1e-12


def reverse_kernel_mean_var(x, x0_est, alpha_t, sigma_t, alpha_s, sigma_s):
    alpha_ts = alpha_t / alpha_s
    var_ts = (sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2).clamp_min(0.0)
    st2 = (sigma_t ** 2).clamp_min(_EPS)
    mean = (alpha_ts * sigma_s ** 2 / st2) * x + (alpha_s * var_ts / st2) * x0_est
    var = var_ts * sigma_s ** 2 / st2
    return mean, var


def sample_coord_noise(mean, var, dof):
    var = torch.as_tensor(var, device=mean.device).clamp_min(0.0)
    if mean.is_complex():
        dof_t = torch.as_tensor(dof, device=mean.device, dtype=mean.real.dtype)
        zr = torch.randn_like(mean.real)
        zi = torch.randn_like(mean.real) * (dof_t == 2).to(mean.real.dtype)
        return mean + (var / dof_t).sqrt() * torch.complex(zr, zi)
    return mean + var.sqrt() * torch.randn_like(mean)


def gaussian_energy(d, var, dof):
    var = torch.as_tensor(var, device=d.device).clamp_min(_EPS)
    real_dtype = d.real.dtype if d.is_complex() else d.dtype
    dof_t = torch.as_tensor(dof, device=d.device, dtype=real_dtype)
    return -(dof_t / 2.0) * (d.abs() ** 2) / var


def systematic_resample(log_w: torch.Tensor) -> torch.Tensor:
    P = log_w.shape[0]
    w = torch.softmax(log_w, dim=0)
    u = (torch.rand(1, device=log_w.device) +
         torch.arange(P, device=log_w.device, dtype=w.dtype)) / P
    return torch.searchsorted(torch.cumsum(w, dim=0), u).clamp_max(P - 1)


class DiagConditioner:
    def __init__(self, svd, y, sigma_y, floor, device):
        s = svd.singulars.to(device)
        if s.dim() == 3:
            s = s.unsqueeze(0)
        self.meas = s > floor                       # (1,C,H,W)
        self.tau = torch.where(self.meas, y, torch.zeros_like(y))
        self.var_obs = torch.full_like(self.tau, sigma_y ** 2)
        self.dof = torch.ones_like(self.tau)

    def coords(self, x):
        return x

    def embed(self, x, new, where):
        return torch.where(where & self.meas, new, x)


class PoolConditioner:

    def __init__(self, svd, y_LR, sigma_y, device):
        self.r = int(svd.extra["r"])
        r = self.r
        self.tau = r * y_LR                        
        self.var_obs = torch.full_like(self.tau, (r * sigma_y) ** 2)
        self.dof = torch.ones_like(self.tau)
        self.meas = torch.ones_like(self.tau, dtype=torch.bool)

    def coords(self, x):
        return self.r * F.avg_pool2d(x, self.r, self.r)

    def _up(self, c):
        return F.interpolate(c, scale_factor=self.r, mode="nearest")

    def embed(self, x, new, where):
        cur = self.coords(x)
        c = torch.where(where, new, cur)
        return x - self._up(cur / self.r) + self._up(c / self.r)


class SeparableConditioner:

    def __init__(self, svd, y_img, sigma_y, floor, device):
        e = svd.extra
        self.U_h = e["U_h"].to(device)
        self.V_h = e["V_h"].to(device)
        self.U_w = e["U_w"].to(device)
        self.V_w = e["V_w"].to(device)

        S = svd.singulars.to(device)[None, None]      
        self.meas = S > floor
        S_safe = torch.where(self.meas, S, torch.ones_like(S))

        y = y_img if y_img.dim() == 4 else y_img.unsqueeze(0)
        y = y.to(device=device, dtype=self.U_h.dtype)
        B = self.U_h.transpose(-2, -1) @ y @ self.U_w   

        self.tau = torch.where(self.meas, B / S_safe, torch.zeros_like(B))
        self.var_obs = torch.where(self.meas,
                                   (sigma_y ** 2) / (S_safe ** 2),
                                   torch.zeros_like(S))
        self.dof = torch.ones_like(S)

    def coords(self, x):
        return self.V_h.transpose(-2, -1) @ x @ self.V_w

    def embed(self, x, new, where):
        c = self.coords(x)
        c = torch.where(where & self.meas, new, c)
        return self.V_h @ c @ self.V_w.transpose(-2, -1)


class FourierConditioner:

    def __init__(self, svd, y_img, sigma_y, floor, device, H, W):
        self.H, self.W = H, W
        Wf = W // 2 + 1

        K = None
        if getattr(svd, "extra", None):
            K = svd.extra.get("transfer", None)      # complex (H, Wf) preferred
        if K is None:
            log.warning(
                "FourierConditioner: SVDRep provides only |K| (no complex "
                "'transfer' in svd.extra). Falling back to a real transfer -- "
                "this is EXACT only for symmetric (zero-phase) kernels. "
                "Expose the complex kernel FFT for general blurs.")
            K = svd.singulars.to(device).to(torch.complex64)
        else:
            K = K.to(device)
            if not K.is_complex():
                K = K.to(torch.complex64)
        self.K = K[None, None]                        # (1,1,H,Wf)
        Kmag2 = (self.K.abs() ** 2)

        cols = torch.arange(Wf, device=device)
        edge_col = (cols == 0)
        if W % 2 == 0:
            edge_col = edge_col | (cols == Wf - 1)
        rows = torch.arange(H, device=device)[:, None]
        dead = edge_col[None, :] & (rows > H // 2)    # redundant mirrors
        selfconj = edge_col[None, :] & ((rows == 0) |
                                        ((H % 2 == 0) & (rows == H // 2)))
        self._edge_col = edge_col
        self._dead = dead[None, None]
        self._selfconj = selfconj[None, None]

        keep = (self.K.abs() > floor)
        self.meas = keep & ~self._dead
        self.dof = torch.where(self._selfconj,
                               torch.ones_like(Kmag2), 2 * torch.ones_like(Kmag2))

        B = torch.fft.rfft2(y_img, norm="ortho")      # (1,C,H,Wf) complex
        K_safe = self.K + (self.K.abs() < _EPS).to(self.K.dtype)
        self.tau = B / K_safe                         # complex division: phase kept
        self.var_obs = (sigma_y ** 2) / Kmag2.clamp_min(floor ** 2)

    def coords(self, x):
        return torch.fft.rfft2(x, norm="ortho")

    def _hermitize(self, Xf):
        ec = self._edge_col
        sub = Xf[..., :, ec]                          # (P,C,H,n_edge)
        mirror = torch.roll(torch.flip(sub, dims=[-2]), 1, dims=-2).conj()
        rows = torch.arange(self.H, device=Xf.device)[:, None]
        sub = torch.where((rows > self.H // 2), mirror, sub)
        sc = self._selfconj[..., ec][0, 0]            # (H, n_edge)
        sub = torch.where(sc, sub.real.to(sub.dtype), sub)
        Xf[..., :, ec] = sub
        return Xf

    def embed(self, x, new, where):
        Xf = self.coords(x)
        Xf = torch.where(where & self.meas, new, Xf)
        Xf = self._hermitize(Xf)
        return torch.fft.irfft2(Xf, s=(self.H, self.W), norm="ortho")


def make_conditioner(svd, observation, sigma_y, floor, hr_shape, device):
    C, H, W = hr_shape
    y = observation.to(device)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    if svd.mode == "diag":
        return DiagConditioner(svd, y, sigma_y, floor, device)
    if svd.mode == "pool":
        return PoolConditioner(svd, y, sigma_y, device)
    if svd.mode == "separable":                       # NEW: deblur
        return SeparableConditioner(svd, y, sigma_y, floor, device)
    if svd.mode == "fourier":
        return FourierConditioner(svd, y, sigma_y, floor, device, H, W)
    raise ValueError(f"unknown SVDRep mode {svd.mode!r}")


def masked_sum(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum t over all non-particle dims where mask is True. (P,...) -> (P,)."""
    out = t * mask.to(t.dtype)
    return out.sum(dim=tuple(range(1, out.ndim)))