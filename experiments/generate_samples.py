import argparse
from pathlib import Path

import torch
from diffusers import DDPMPipeline
import numpy as np 

from experiments.configs import (
    DATASETS, NUM_IMAGES_PER_DATASET, GENERATION_SEED, CACHE_DIR,
)


def cache_is_full(cache_dir: Path, n: int) -> bool:
    return all((cache_dir / f"{i:02d}.pt").exists() for i in range(n))


def generate_for_dataset(name: str, info: dict, device: torch.device, force: bool):
    cache_dir = Path(CACHE_DIR) / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force and cache_is_full(cache_dir, NUM_IMAGES_PER_DATASET):
        print(f"[{name}] cache complete, skipping")
        return

    print(f"[{name}] loading {info['model_id']}")
    pipe = DDPMPipeline.from_pretrained(info["model_id"]).to(device)

    gen = torch.Generator(device=device).manual_seed(GENERATION_SEED)

    print(f"[{name}] sampling {NUM_IMAGES_PER_DATASET} images at {info['resolution']}x{info['resolution']}")
    with torch.no_grad():
        out = pipe(
            batch_size=NUM_IMAGES_PER_DATASET,
            generator=gen,
            output_type="numpy",   
        )

    imgs = out.images
    if imgs.ndim == 3:  
        imgs = imgs[..., None]
    arr = torch.from_numpy(np.asarray(imgs)).permute(0, 3, 1, 2).float()  
    arr = arr * 2.0 - 1.0  

    if info["channels"] == 1 and arr.shape[1] == 3:
        arr = arr.mean(dim=1, keepdim=True)

    from PIL import Image
    for i in range(arr.shape[0]):
        t = arr[i].cpu()
        torch.save(t, cache_dir / f"{i:02d}.pt")
        png = ((t.clamp(-1, 1) + 1) * 127.5).byte().numpy()
        if png.shape[0] == 1:
            Image.fromarray(png[0], mode="L").save(cache_dir / f"{i:02d}.png")
        else:
            Image.fromarray(png.transpose(1, 2, 0), mode="RGB").save(cache_dir / f"{i:02d}.png")

    print(f"[{name}] saved {arr.shape[0]} -> {cache_dir}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only generate for these datasets (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if cache is full")
    args = p.parse_args()

    device = torch.device(args.device)
    targets = args.only or list(DATASETS.keys())

    for name in targets:
        if name not in DATASETS:
            print(f"Unknown dataset {name!r}, skipping")
            continue
        generate_for_dataset(name, DATASETS[name], device, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()