import argparse
import json
from pathlib import Path

import torch
from PIL import Image

try:
    from experiments.configs import CACHE_DIR as DEFAULT_CACHE_DIR
except Exception: 
    DEFAULT_CACHE_DIR = "cache"

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect_files(src: Path, exts) -> list[Path]:
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_image(path: Path, resolution: int, channels: int, crop: str) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L" if channels == 1 else "RGB")
    if crop == "center":
        img = center_crop_square(img)
    img = img.resize((resolution, resolution), Image.LANCZOS)

    t = torch.from_numpy(
        __import__("numpy").asarray(img).copy()
    ).float().div_(255.0)
    if t.dim() == 2:                 # grayscale -> (1, H, W)
        t = t.unsqueeze(0)
    else:                            # (H, W, C) -> (C, H, W)
        t = t.permute(2, 0, 1)
    return t.mul_(2.0).sub_(1.0).contiguous()   # [0,1] -> [-1,1]


def describe(t: torch.Tensor) -> str:
    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="Folder of source images.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name; becomes the cache subdirectory and the "
                        "key you register in DATASETS.")
    p.add_argument("--resolution", required=True, type=int,
                   help="Must equal DATASETS[dataset]['resolution'] and the "
                        "UNet's sample_size.")
    p.add_argument("--num", required=True, type=int,
                   help="How many images to cache. Should equal "
                        "NUM_IMAGES_PER_DATASET (see note below).")
    p.add_argument("--start", type=int, default=0,
                   help="Skip this many files before taking --num.")
    p.add_argument("--channels", type=int, default=3, choices=[1, 3])
    p.add_argument("--crop", choices=["center", "stretch"], default="center",
                   help="'center' crops to square before resizing (correct for "
                        "non-square CelebA); 'stretch' resizes directly.")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .pt files.")
    p.add_argument("--compare", type=Path, default=None,
                   help="Existing cached .pt to print alongside for format "
                        "comparison.")
    args = p.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"--src is not a directory: {args.src}")

    files = collect_files(args.src, EXTS)[args.start:]
    if len(files) < args.num:
        raise SystemExit(
            f"Need {args.num} images but found only {len(files)} in {args.src} "
            f"(after --start {args.start})."
        )
    files = files[:args.num]

    out_dir = Path(args.cache_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.pt"))
    if existing and not args.force:
        raise SystemExit(
            f"{out_dir} already contains {len(existing)} .pt files. "
            f"Pass --force to overwrite (this will change the idx -> image "
            f"mapping and invalidate any rows already in the summary CSV)."
        )

    manifest = {
        "dataset": args.dataset,
        "source_dir": str(args.src.resolve()),
        "resolution": args.resolution,
        "channels": args.channels,
        "crop": args.crop,
        "range": [-1.0, 1.0],
        "images": {},
    }

    for idx, path in enumerate(files):
        t = load_image(path, args.resolution, args.channels, args.crop)
        assert t.shape == (args.channels, args.resolution, args.resolution), t.shape
        torch.save(t, out_dir / f"{idx:02d}.pt")
        manifest["images"][f"{idx:02d}"] = path.name
        print(f"  {idx:02d}  {path.name:40s}  {describe(t)}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(files)} tensors to {out_dir}")

    if args.compare is not None:
        ref = torch.load(args.compare, weights_only=True)
        new = torch.load(out_dir / "00.pt", weights_only=True)
        print(f"\nFormat check:")
        print(f"  reference {args.compare}\n    {describe(ref)}")
        print(f"  new       {out_dir / '00.pt'}\n    {describe(new)}")
        if ref.dtype != new.dtype:
            print("  WARNING: dtype mismatch")
        if ref.shape[0] != new.shape[0]:
            print("  WARNING: channel-count mismatch")


if __name__ == "__main__":
    main()
