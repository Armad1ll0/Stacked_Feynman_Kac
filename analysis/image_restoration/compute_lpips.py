import argparse
import csv
import importlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def resolve_config(which: str):
    mod = importlib.import_module("experiments.configs")
    return Path(mod.CACHE_DIR), Path(mod.RESULTS_DIR), mod.DATASETS


def pil_to_tensor(img: Image.Image, channels: int) -> torch.Tensor:
    if channels == 1:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr)[None, None]       # (1,1,H,W)
        t = t.expand(1, 3, -1, -1).contiguous()     # (1,3,H,W)
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1)[None]  # (1,3,H,W)
    return t


def extract_recon_from_compare(compare_png: Path, orig_size: tuple[int, int]) -> Image.Image:
    strip = Image.open(compare_png)
    W_panel, H_panel = orig_size
    gap = 10
    left = 2 * (W_panel + gap)
    recon = strip.crop((left, 0, left + W_panel, H_panel))
    return recon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["tds"], default="tds",
                    help="Which result matrix to compute LPIPS for. Determines "
                         "which summary.csv is read/written and where cached "
                         "clean images and comparison PNGs are looked up.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"],
                   help="LPIPS backbone. 'alex' is the standard default.")
    args = p.parse_args()

    try:
        import lpips
    except ImportError:
        raise SystemExit("Install lpips first:  pip install lpips")

    CACHE_DIR, RESULTS_DIR, DATASETS = resolve_config(args.which)

    device = torch.device(args.device)
    print(f"Loading LPIPS-{args.net} on {device} (matrix: {args.which})")
    loss_fn = lpips.LPIPS(net=args.net).to(device).eval()

    csv_path = RESULTS_DIR / "summary_noisy.csv"
    if not csv_path.exists():
        raise SystemExit(f"No summary.csv at {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "lpips" not in fieldnames:
        fieldnames.append("lpips")

    n_done = 0
    n_new = 0
    n_failed = 0

    for row in rows:
        if row.get("lpips"):
            n_done += 1
            continue

        dataset = row["dataset"]
        task = row["task"]
        method = row["method"]
        idx = int(row["image_idx"])

        cell_dir = RESULTS_DIR / f"{dataset}_{task}_{method}"
        compare_png = cell_dir / f"{idx:02d}_compare.png"
        clean_pt = CACHE_DIR / dataset / f"{idx:02d}.pt"

        if not compare_png.exists() or not clean_pt.exists():
            print(f"  MISSING: {compare_png.name} or {clean_pt.name}")
            n_failed += 1
            continue

        try:
            x_clean = torch.load(clean_pt, weights_only=True) 
            C, H, W = x_clean.shape

            recon_pil = extract_recon_from_compare(compare_png, orig_size=(W, H))

            t_clean = pil_to_tensor(
                Image.fromarray(
                    ((x_clean.clamp(-1, 1) + 1) * 127.5).byte().numpy().squeeze()
                    if C == 1 else
                    ((x_clean.clamp(-1, 1) + 1) * 127.5).byte().numpy().transpose(1, 2, 0),
                    mode="L" if C == 1 else "RGB",
                ),
                channels=C,
            ).to(device)
            t_recon = pil_to_tensor(recon_pil, channels=C).to(device)

            if t_clean.shape[-1] < 64:
                t_clean = torch.nn.functional.interpolate(
                    t_clean, size=64, mode="bilinear", align_corners=False)
                t_recon = torch.nn.functional.interpolate(
                    t_recon, size=64, mode="bilinear", align_corners=False)

            with torch.no_grad():
                d = loss_fn(t_clean, t_recon).item()

            row["lpips"] = round(d, 6)
            n_new += 1
            if n_new % 25 == 0:
                print(f"  ...computed {n_new} new LPIPS values")
        except Exception as e:
            print(f"  FAILED {dataset}/{task}/{method}/{idx}: {e}")
            n_failed += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {n_done} already had LPIPS, {n_new} new, {n_failed} failed.")
    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()
