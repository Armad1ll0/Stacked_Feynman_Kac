import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMPipeline

from experiments.configs import (
    DATASETS, TASKS, METHODS, SAMPLER_CFG,
    NUM_IMAGES_PER_DATASET, CACHE_DIR, RESULTS_DIR,
    SamplerConfig,
)
from sampler.tsmc import run_tds_then_hmc_denoised
from tasks.inpainting import InpaintingTask
from tasks.super_resolution import SuperResTask
from tasks.deblurring import DeblurringTask
from experiments.noise import make_observation, apply_forward


WALLCLOCK_DATASETS = ["mnist", "flowers"]
WALLCLOCK_TASKS    = ["inpaint_random", "superres"]

_BASE_HMC = dict(hmc_iters=10, L=3) 

WALLCLOCK_CFG: dict[str, dict[str, SamplerConfig]] = {
    "tds": {
        "mnist":   SamplerConfig(num_particles=16, num_steps=170, lambda_like=1, stop_at_step=0),
        "flowers": SamplerConfig(num_particles=8, num_steps=575, lambda_like=1,  stop_at_step=0),
        "butterflies": SamplerConfig(num_particles=8, num_steps=575, lambda_like=1,  stop_at_step=0),
    },
}


STOP_AT_STEP_DATASETS = ["mnist", "flowers"]
STOP_AT_STEP_TASKS    = ["inpaint_random", "superres"]
STOP_AT_STEP_DEFAULT  = [60, 70, 80, 90, 100, 120, 150, 200]   


class _TdsHmcResult:
    def __init__(self, raw):
        self.log_weights = raw.log_weights
        self.ess_trace   = raw.ess_trace
        self.particles   = raw.particles_hmc
        self.best        = self.particles[raw.log_weights.argmax()]


LAMBDA_LIKE: dict[str, float] = {
    "mnist":       1000.0,
    "flowers":     100.0,
    "buttflies":     100.0,
}
_LAMBDA_LIKE_DEFAULT = 1


def _run_tds_hmc(model, scheduler, task, observation, metadata, cfg, device,
                 dataset_name: str = "", stop_at_step: int = 5):
    lambda_like = LAMBDA_LIKE.get(dataset_name, _LAMBDA_LIKE_DEFAULT)
    hmc_cfg = replace(cfg, lambda_like=lambda_like, stop_at_step=stop_at_step)
    raw = run_tds_then_hmc_denoised(
        model=model, scheduler=scheduler, task=task,
        observation=observation, metadata=metadata,
        cfg=hmc_cfg, device=device,
        stop_at_step=stop_at_step,
    )
    return _TdsHmcResult(raw)


def psnr(x, y, data_range=2.0):
    mse = (x - y).pow(2).mean().item()
    if mse < 1e-20:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(data_range ** 2 / mse)).item()


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim(x, y, window_size=11, sigma=1.5, data_range=2.0):
    if x.dim() == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    C = x.shape[1]
    window = _gaussian_window(window_size, sigma, x.device, x.dtype).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu_x  = F.conv2d(x, window, padding=pad, groups=C)
    mu_y  = F.conv2d(y, window, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    s_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    s_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    return (((2 * mu_xy + C1) * (2 * s_xy + C2))
            / ((mu_x2 + mu_y2 + C1) * (s_x2 + s_y2 + C2))).mean().item()


def observation_consistency(task, x_recon, x_clean, metadata) -> float:
    """RMSE(A(x_recon), A(x_clean)). Measurement-space fidelity with a 0
    floor, defined identically for every method and independent of which
    noise realization the method conditioned on."""
    with torch.no_grad():
        Ax  = apply_forward(task, x_recon.to(x_clean.device), metadata)
        Axc = apply_forward(task, x_clean, metadata)
    # print('Herrreee', Ax.shape, Axc.shape)
    return (Ax - Axc).pow(2).mean().sqrt().item()


def observation_residual(task, x_recon, y, metadata) -> float:
    """RMSE(A(x_recon), y) against the actual noisy measurement.
    Floor ~ sigma_y."""
    y = y.unsqueeze(0) if y.dim() == 3 else y
    with torch.no_grad():
        Ax = apply_forward(task, x_recon.to(y.device), metadata)
    return (Ax - y).pow(2).mean().sqrt().item()


def diversity_masked(particles, mask):
    P = particles.shape[0]
    if P < 2:
        return 0.0
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    inpaint = (1.0 - mask).bool().expand_as(particles[0:1])
    flat = particles[:, inpaint[0]].view(P, -1)
    return float("nan") if flat.shape[1] == 0 else torch.pdist(flat, p=2).mean().item()


def variance_diversity_masked(particles, mask, log_w=None):
    P = particles.shape[0]
    if P < 2:
        return 0.0
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    inpaint = (1.0 - mask).bool().expand_as(particles[0:1])
    flat = particles[:, inpaint[0]].view(P, -1)
    if flat.shape[1] == 0:
        return float("nan")
    if log_w is None:
        return flat.var(dim=0, unbiased=False).mean().item()
    w = torch.softmax(log_w, dim=0)
    mean = (w[:, None] * flat).sum(0)
    return (w[:, None] * (flat - mean) ** 2).sum(0).mean().item()


def compute_diversity(task, result, metadata):
    if not isinstance(task, InpaintingTask):
        return float("nan"), float("nan")
    particles = getattr(result, "particles", None)
    if particles is None:
        return float("nan"), float("nan")
    mask = metadata["mask"]
    return (
        diversity_masked(particles, mask),
        variance_diversity_masked(particles, mask, result.log_weights),
    )


def _fmt(v, decimals):
    return round(v, decimals) if not math.isnan(v) else "nan"


def tensor_to_pil(t):
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    if arr.shape[0] == 1:
        return Image.fromarray(arr[0], mode="L")
    return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")


def save_comparison(path, orig, obs, recon):
    orig_pil, obs_pil, recon_pil = map(tensor_to_pil, (orig, obs, recon))
    W, H = orig_pil.size
    gap, mode = 10, orig_pil.mode
    bg = 255 if mode == "L" else (255, 255, 255)
    strip = Image.new(mode, (W * 3 + gap * 2, H), color=bg)
    for img, x in zip([orig_pil, obs_pil, recon_pil], [0, W + gap, 2 * W + 2 * gap]):
        strip.paste(img, (x, 0))
    strip.save(path)


def load_cached_image(dataset, idx):
    path = Path(CACHE_DIR) / dataset / f"{idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sample {path}. Run generate_samples.py first.")
    return torch.load(path, weights_only=True)


def load_done_rows(csv_path):
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            done.add(tuple(row[k] for k in FIELDNAMES[:5]))  # first 5 cols form the key
    return done


def append_row(csv_path, row, fieldnames):
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


FIELDNAMES = [
    "dataset", "task", "method", "image_idx", "ablation_param",
    "psnr", "ssim", "obs_consistency", "runtime_s",
    "ess_mean", "ess_min", "best_log_w",
    "diversity", "var_diversity",
]


def run_one(
    *,
    model, scheduler, task, task_name,
    dataset_name, method_name, ablation_param,
    cfg, device, cell_dir, csv_path, done, num_images,
):
    for idx in range(num_images):
        key = (dataset_name, task_name, method_name, str(idx), str(ablation_param))
        if key in done:
            continue

        x_clean = load_cached_image(dataset_name, idx)
        observation, metadata = task.degrade(x_clean)

        torch.manual_seed(1000 + idx)
        t0 = time.perf_counter()

        if method_name.startswith("tds_hmc"):
            result = _run_tds_hmc(
                model=model, scheduler=scheduler, task=task,
                observation=observation, metadata=metadata,
                cfg=cfg, device=device,
                dataset_name=dataset_name,
                stop_at_step=cfg.stop_at_step,
            )
        else:
            result = METHODS[method_name](
                model=model, scheduler=scheduler, task=task,
                observation=observation, metadata=metadata,
                cfg=cfg, device=device,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_s = time.perf_counter() - t0

        x_recon      = result.best.detach().cpu()
        div, var_div = compute_diversity(task, result, metadata)

        row = {
            "dataset":         dataset_name,
            "task":            task_name,
            "method":          method_name,
            "image_idx":       idx,
            "ablation_param":  ablation_param,
            "psnr":            _fmt(psnr(x_recon, x_clean), 4),
            "ssim":            _fmt(ssim(x_recon, x_clean), 4),
            "obs_consistency": _fmt(observation_consistency(task, x_recon, x_clean, metadata), 6),
            "runtime_s":       round(runtime_s, 3),
            "ess_mean":        round(sum(result.ess_trace) / len(result.ess_trace), 3),
            "ess_min":         round(min(result.ess_trace), 3),
            "best_log_w":      round(result.log_weights.max().item(), 4),
            "diversity":       _fmt(div, 4),
            "var_diversity":   _fmt(var_div, 6),
        }

        save_comparison(cell_dir / f"{idx:02d}_compare.png", x_clean, observation, x_recon)
        with open(cell_dir / f"{idx:02d}_metrics.json", "w") as f:
            json.dump(row, f, indent=2)

        append_row(csv_path, row, FIELDNAMES)
        done.add(key)

        print(
            f"      img {idx:02d}  "
            f"PSNR={row['psnr']:.2f}  "
            f"SSIM={row['ssim']:.3f}  "
            f"div={row['diversity']}  "
            f"t={row['runtime_s']:.1f}s"
        )


def run_wallclock_ablation(args, device, results_root, csv_path, done, num_images):

    datasets_to_run = args.only_dataset or WALLCLOCK_DATASETS
    tasks_to_run    = args.only_task    or WALLCLOCK_TASKS
    methods_to_run  = args.only_method  or list(WALLCLOCK_CFG.keys())

    print(f"  Datasets : {datasets_to_run}")
    print(f"  Tasks    : {tasks_to_run}")
    print(f"  Methods  : {methods_to_run}")
    print(f"  Images   : {num_images}")

    for dataset_name in datasets_to_run:
        info = DATASETS[dataset_name]
        print(f"\n=== Dataset: {dataset_name} ===")
        pipe  = DDPMPipeline.from_pretrained(info["model_id"])
        model = pipe.unet.to(device).eval()
        scheduler = pipe.scheduler

        for task_name in tasks_to_run:
            task = TASKS[task_name](info["resolution"])
            print(f"\n  Task: {task_name}")

            for method_name in methods_to_run:
                if method_name not in WALLCLOCK_CFG:
                    print(f"    [skip] {method_name} – not in WALLCLOCK_CFG")
                    continue
                if dataset_name not in WALLCLOCK_CFG[method_name]:
                    print(f"    [skip] {method_name}/{dataset_name} – not in WALLCLOCK_CFG")
                    continue

                cfg = WALLCLOCK_CFG[method_name][dataset_name]
                if cfg.num_steps is None:
                    cfg = replace(cfg, num_steps=scheduler.config.num_train_timesteps)

                cell_label = f"wallclock_{dataset_name}_{task_name}_{method_name}"
                cell_dir   = results_root / cell_label
                cell_dir.mkdir(parents=True, exist_ok=True)

                print(f"    Method: {method_name}  "
                      f"(P={cfg.num_particles}, steps={cfg.num_steps})")

                run_one(
                    model=model, scheduler=scheduler,
                    task=task, task_name=task_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    ablation_param=f"P{cfg.num_particles}_S{cfg.num_steps}",
                    cfg=cfg, device=device,
                    cell_dir=cell_dir, csv_path=csv_path, done=done,
                    num_images=num_images,
                )

        del pipe, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_stop_at_step_ablation(args, device, results_root, csv_path, done, stop_values, num_images):

    datasets_to_run = args.only_dataset or STOP_AT_STEP_DATASETS
    tasks_to_run    = args.only_task    or STOP_AT_STEP_TASKS

    print(f"  Datasets    : {datasets_to_run}")
    print(f"  Tasks       : {tasks_to_run}")
    print(f"  stop_values : {stop_values}")
    print(f"  Images      : {num_images}")

    for dataset_name in datasets_to_run:
        info = DATASETS[dataset_name]
        print(f"\n=== Dataset: {dataset_name} ===")
        pipe  = DDPMPipeline.from_pretrained(info["model_id"])
        model = pipe.unet.to(device).eval()
        scheduler = pipe.scheduler

        base_cfg = SAMPLER_CFG[dataset_name]
        if base_cfg.num_steps is None:
            base_cfg = replace(base_cfg, num_steps=scheduler.config.num_train_timesteps)

        for task_name in tasks_to_run:
            task = TASKS[task_name](info["resolution"])
            print(f"\n  Task: {task_name}")

            for stop in stop_values:
                method_name = f"tds_hmc_stop{stop}"
                cfg = replace(base_cfg, stop_at_step=stop)

                cell_label = f"stop_at_step_{dataset_name}_{task_name}_{method_name}"
                cell_dir   = results_root / cell_label
                cell_dir.mkdir(parents=True, exist_ok=True)

                print(f"    stop_at_step={stop}  (P={cfg.num_particles}, steps={cfg.num_steps})")

                run_one(
                    model=model, scheduler=scheduler,
                    task=task, task_name=task_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    ablation_param=stop,
                    cfg=cfg, device=device,
                    cell_dir=cell_dir, csv_path=csv_path, done=done,
                    num_images=num_images,
                )

        del pipe, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation",      required=True, choices=["wallclock", "stop_at_step"],
                   help="Which ablation study to run")
    p.add_argument("--device",        default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_images",    type=int, default=None,
                   help="Images per dataset (default: NUM_IMAGES_PER_DATASET from configs.py)")
    p.add_argument("--only_dataset",  nargs="*", default=None,
                   help="Override default datasets for this ablation")
    p.add_argument("--only_task",     nargs="*", default=None,
                   help="Override default tasks for this ablation")
    p.add_argument("--only_method",   nargs="*", default=None,
                   help="Restrict to these methods (wallclock ablation only)")
    p.add_argument("--stop_values",   nargs="*", type=int,
                   default=STOP_AT_STEP_DEFAULT,
                   help="stop_at_step values to sweep (stop_at_step ablation only)")
    args = p.parse_args()

    num_images = args.num_images if args.num_images is not None else NUM_IMAGES_PER_DATASET

    device       = torch.device(args.device)
    results_root = Path(RESULTS_DIR) / f"ablation_{args.ablation}"
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path     = results_root / "summary.csv"
    done         = load_done_rows(csv_path)

    print(f"Ablation  : {args.ablation}")
    print(f"Results   : {results_root}")
    print(f"Skipping  : {len(done)} already-completed rows")

    if args.ablation == "wallclock":
        run_wallclock_ablation(args, device, results_root, csv_path, done, num_images)
    else:
        run_stop_at_step_ablation(args, device, results_root, csv_path, done, args.stop_values, num_images)

    print(f"\nDone. Summary: {csv_path}")


if __name__ == "__main__":
    main()
