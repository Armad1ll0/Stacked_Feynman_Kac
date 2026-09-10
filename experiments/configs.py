from __future__ import annotations

from sampler.tsmc import SamplerConfig, run_tds, run_tds_hmc, run_tds_then_hmc_denoised
from tasks.inpainting import InpaintingTask, InpaintingConfig
from tasks.super_resolution import SuperResTask, SuperResConfig
from tasks.deblurring import DeblurringTask, DeblurringConfig

DATASETS: dict[str, dict] = {
    "flowers": {
        "model_id": "anton-l/ddpm-ema-flowers-64",
        "resolution": 64,
        "channels": 3,
    },
    "celeba": {
        "model_id": "google/ddpm-ema-celebahq-256",
        "resolution": 256,
        "channels": 3,
    },
    "celeba_real": {
        "model_id": "google/ddpm-ema-celebahq-256",
        "resolution": 256,
        "channels": 3,
    },
    "mnist": {
        "model_id": "1aurent/ddpm-mnist",
        "resolution": 28,
        "channels": 1,
    },
    "butterflies": {
        "model_id": "anton-l/ddpm-butterflies-128",
        "resolution": 128,
        "channels": 3,
    },
}

_BASE_HMC = dict(
    resample_ess_threshold=0.5,
    verbose_every=0,
    paste_observation=False,
    hmc_iters=20,
    L=3,
    hmc_step_scale=0.3,
)

SAMPLER_CFG: dict[str, SamplerConfig] = {
    "mnist":       SamplerConfig(num_particles=16, num_steps=100, lambda_like=100, stop_at_step=5, **_BASE_HMC),
    "butterflies": SamplerConfig(num_particles=8, num_steps=500, lambda_like=100, stop_at_step=25,  **_BASE_HMC),
    "flowers":     SamplerConfig(num_particles=8, num_steps=500, lambda_like=100, stop_at_step=25,  **_BASE_HMC),
    "celeba":      SamplerConfig(num_particles=4,  num_steps=1000, lambda_like=50, stop_at_step=50,  **_BASE_HMC),
    "celeba_real":      SamplerConfig(num_particles=4,  num_steps=1000, lambda_like=50, stop_at_step=50,  **_BASE_HMC),
}

def _inpaint_for(res: int, sigma_y: float = 0.0):
    return InpaintingTask(InpaintingConfig(
        mask_type="center_box",
        hole_fraction=0.5,
        sigma_sq_scale=1.0,
        clamp_x0=True,
        sigma_y=sigma_y,
    ))


def _inpaint_random_for(res: int, sigma_y: float = 0.0):
    return InpaintingTask(InpaintingConfig(
        mask_type="random_pixels",
        hole_fraction=0.7,
        sigma_sq_scale=1.0,
        clamp_x0=True,
        sigma_y=sigma_y,
    ))


def _superres_for(res: int, sigma_y: float = 0.0):
    scale = 2 if res <= 64 else 4
    return SuperResTask(SuperResConfig(
        scale_factor=scale,
        sigma_sq_scale=1.0 / scale ** 2,
        clamp_x0=True,
        sigma_y=sigma_y,
    ))


def _deblur_for(res: int, sigma_y: float = 0.0):
    blur_sigma = max(1.0, round(res / 21.0, 1))
    kernel_size = int(6 * blur_sigma) | 1        # ~6 sigma, forced odd
    return DeblurringTask(DeblurringConfig(
        kernel_size=kernel_size,
        blur_sigma=blur_sigma,
        sigma_sq_scale=1.0,
        clamp_x0=True,
        sigma_y=sigma_y,
    ))


TASKS: dict[str, callable] = {
    "inpaint":        _inpaint_for,
    "inpaint_random": _inpaint_random_for,
    "superres":       _superres_for,
    "deblur":         _deblur_for,
}

METHODS: dict[str, callable] = {
    "tds":     run_tds,
    "tds_hmc_refined": run_tds_then_hmc_denoised,
}

NUM_IMAGES_PER_DATASET = 100
GENERATION_SEED = 42
CACHE_DIR = "experiments/cache"
RESULTS_DIR = "experiments/results"