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
    DATASETS, SAMPLER_CFG, TASKS, METHODS,
    NUM_IMAGES_PER_DATASET, CACHE_DIR, RESULTS_DIR,
)
from experiments.noise import make_observation, apply_forward
from tasks.inpainting import InpaintingTask


def psnr(x, y, data_range=2.0):
    """x, y in [-1, 1]. data_range=2.0 because range is [-1, 1]."""
    mse = (x - y).pow(2).mean().item()
    if mse < 1e-20:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(data_range ** 2 / mse)).item()


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


def ssim(x, y, window_size=11, sigma=1.5, data_range=2.0):
    """Simple SSIM for (C,H,W) tensors in [-1,1]. Averaged over channels."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    C = x.shape[1]
    window = _gaussian_window(window_size, sigma, x.device, x.dtype).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu_x  = F.conv2d(x, window, padding=pad, groups=C)
    mu_y  = F.conv2d(y, window, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x2  = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    s_y2  = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    s_xy  = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    return (((2 * mu_xy + C1) * (2 * s_xy + C2))
            / ((mu_x2 + mu_y2 + C1) * (s_x2 + s_y2 + C2))).mean().item()


def observation_consistency(task, x_recon, x_clean, metadata) -> float:
    with torch.no_grad():
        Ax  = apply_forward(task, x_recon.to(x_clean.device), metadata)
        Axc = apply_forward(task, x_clean, metadata)
    return (Ax - Axc).pow(2).mean().sqrt().item()


def observation_residual(task, x_recon, y, metadata) -> float:
    y = y.unsqueeze(0) if y.dim() == 3 else y
    with torch.no_grad():
        Ax = apply_forward(task, x_recon.to(y.device), metadata)
    return (Ax - y).pow(2).mean().sqrt().item()


def diversity_masked(particles: torch.Tensor, mask: torch.Tensor) -> float:
    P = particles.shape[0]
    if P < 2:
        return 0.0
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    inpaint = (1.0 - mask).bool().expand_as(particles[0:1])
    flat = particles[:, inpaint[0]].view(P, -1)
    if flat.shape[1] == 0:
        return float("nan")
    return torch.pdist(flat, p=2).mean().item()


def variance_diversity_masked(
    particles: torch.Tensor,
    mask: torch.Tensor,
    log_w: torch.Tensor = None,
) -> float:
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
    var  = (w[:, None] * (flat - mean) ** 2).sum(0)
    return var.mean().item()


def compute_diversity(task, result, metadata) -> tuple[float, float]:
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


def filter_csv(csv_path, only_dataset, only_task, only_method):
    if not csv_path.exists():
        return 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    def matches_filters(row):
        if only_dataset and row["dataset"] not in only_dataset:
            return False
        if only_task and row["task"] not in only_task:
            return False
        if only_method and row["method"] not in only_method:
            return False
        return True

    kept = [r for r in rows if not matches_filters(r)]
    removed = len(rows) - len(kept)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    return removed


def _fmt(v: float, decimals: int):
    return round(v, decimals) if not math.isnan(v) else "nan"


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    if arr.shape[0] == 1:
        return Image.fromarray(arr[0], mode="L")
    return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")


def _display_obs(observation, target_hw):
    o = observation.unsqueeze(0) if observation.dim() == 3 else observation
    if o.shape[-2:] != tuple(target_hw):
        o = F.interpolate(o, size=tuple(target_hw), mode="nearest")
    return o.squeeze(0)


def save_comparison(path: Path, orig, obs, recon):
    H, W = orig.shape[-2:]
    orig_pil  = tensor_to_pil(orig)
    obs_pil   = tensor_to_pil(_display_obs(obs, (H, W)))
    recon_pil = tensor_to_pil(recon)
    Wp, Hp = orig_pil.size
    gap  = 10
    mode = orig_pil.mode
    bg   = 255 if mode == "L" else (255, 255, 255)
    strip = Image.new(mode, (Wp * 3 + gap * 2, Hp), color=bg)
    strip.paste(orig_pil,  (0, 0))
    strip.paste(obs_pil,   (Wp + gap, 0))
    strip.paste(recon_pil, (2 * Wp + 2 * gap, 0))
    strip.save(path)


def load_cached_image(dataset: str, idx: int) -> torch.Tensor:
    path = Path(CACHE_DIR) / dataset / f"{idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached sample {path}. Run generate_samples.py first."
        )
    return torch.load(path, weights_only=True)


def load_done_rows(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["task"], row["method"], int(row["image_idx"])))
    return done


def append_row(csv_path: Path, row: dict, fieldnames: list[str]):
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


FIELDNAMES = [
    "dataset", "task", "method", "image_idx",
    "psnr", "ssim", "obs_consistency", "obs_residual", "runtime_s",
    "ess_mean", "ess_min", "ess_final",
    "diversity", "var_diversity",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",       default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only_dataset", nargs="*", default=None)
    p.add_argument("--only_task",    nargs="*", default=None)
    p.add_argument("--only_method",  nargs="*", default=None)
    p.add_argument("--force",        action="store_true",
                   help="Re-run cells even if already completed. "
                        "Combine with --only_* to redo only specific cells.")
    p.add_argument("--sigma_y",      type=float, default=0.05,
                   help="Measurement-noise std, applied to all methods.")
    p.add_argument("--csv_name",     default="summary_noisy.csv",
                   help="Summary file name inside RESULTS_DIR.")
    args = p.parse_args()

    device = torch.device(args.device)
    results_root = Path(RESULTS_DIR)
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / args.csv_name

    if args.force:
        removed = filter_csv(
            csv_path, args.only_dataset, args.only_task, args.only_method)
        print(f"--force given: removed {removed} matching rows from {csv_path}")

    done = load_done_rows(csv_path)
    print(f"Found {len(done)} completed runs in {csv_path}")
    print(f"sigma_y = {args.sigma_y}")

    datasets_to_run = args.only_dataset or list(DATASETS.keys())
    tasks_to_run    = args.only_task    or list(TASKS.keys())
    methods_to_run  = args.only_method  or list(METHODS.keys())

    for dataset_name in datasets_to_run:
        info = DATASETS[dataset_name]
        print(f"\n=== Dataset: {dataset_name} ({info['model_id']}) ===")
        print("Loading model...")
        pipe  = DDPMPipeline.from_pretrained(info["model_id"])
        model = pipe.unet.to(device).eval()
        scheduler = pipe.scheduler

        s_cfg = SAMPLER_CFG[dataset_name]
        if s_cfg.num_steps is None:
            s_cfg = replace(s_cfg, num_steps=scheduler.config.num_train_timesteps)
        print(f"Sampler: P={s_cfg.num_particles}, steps={s_cfg.num_steps}")

        for task_name in tasks_to_run:
            task = TASKS[task_name](info["resolution"], sigma_y=args.sigma_y)
            print(f"\n  Task: {task_name} ({task.__class__.__name__})  "
                  f"sigma_y={task.sigma_y}")

            for method_name in methods_to_run:
                sampler_fn = METHODS[method_name]
                cell     = f"{dataset_name}_{task_name}_{method_name}"
                cell_dir = results_root / cell
                cell_dir.mkdir(parents=True, exist_ok=True)
                print(f"    Method: {method_name}")

                for idx in range(NUM_IMAGES_PER_DATASET):
                    key = (dataset_name, task_name, method_name, idx)
                    if key in done:
                        continue

                    x_clean = load_cached_image(dataset_name, idx)

                    observation, metadata = make_observation(
                        task, x_clean, task_name, idx, args.sigma_y)

                    torch.manual_seed(1000 + idx)
                    t0 = time.perf_counter()
                    result = sampler_fn(
                        model=model, scheduler=scheduler, task=task,
                        observation=observation, metadata=metadata,
                        cfg=s_cfg, device=device,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    runtime_s = time.perf_counter() - t0

                    x_recon      = result.best.detach().cpu()
                    div, var_div = compute_diversity(task, result, metadata)
                    ess_trace    = list(result.ess_trace)

                    row = {
                        "dataset":         dataset_name,
                        "task":            task_name,
                        "method":          method_name,
                        "image_idx":       idx,
                        "psnr":            _fmt(psnr(x_recon, x_clean), 4),
                        "ssim":            _fmt(ssim(x_recon, x_clean), 4),
                        "obs_consistency": _fmt(
                            observation_consistency(task, x_recon, x_clean, metadata), 6),
                        "obs_residual":    _fmt(
                            observation_residual(task, x_recon, observation, metadata), 6),
                        "runtime_s":       round(runtime_s, 3),
                        "ess_mean":        round(sum(ess_trace) / len(ess_trace), 3),
                        "ess_min":         round(min(ess_trace), 3),
                        "ess_final":       round(ess_trace[-1], 3),
                        "diversity":       _fmt(div, 4),
                        "var_diversity":   _fmt(var_div, 6),
                    }

                    save_comparison(
                        cell_dir / f"{idx:02d}_compare.png",
                        x_clean, observation, x_recon,
                    )
                    with open(cell_dir / f"{idx:02d}_metrics.json", "w") as f:
                        json.dump({**row, "sigma_y": args.sigma_y,
                                   "ess_trace": ess_trace}, f, indent=2)

                    append_row(csv_path, row, FIELDNAMES)
                    done.add(key)

                    print(f"      img {idx:02d}  PSNR={row['psnr']}  "
                          f"SSIM={row['ssim']}  cons={row['obs_consistency']}  "
                          f"resid={row['obs_residual']}  "
                          f"t={row['runtime_s']:.1f}s")

        del pipe, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Summary: {csv_path}")


if __name__ == "__main__":
    main()
