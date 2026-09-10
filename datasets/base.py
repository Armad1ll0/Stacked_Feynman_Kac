from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tds.tasks.base import Task


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: str | Path, size: int) -> torch.Tensor:
    """Load, resize, and convert an image to a (C, H, W) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # -> [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)         # (C, H, W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(C, H, W) tensor in [-1, 1] -> PIL Image."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


class BaseDataset(Dataset):

    def __init__(
        self,
        paths: Sequence[str | Path],
        task: Task,
        size: int = 256,
        seed: int | None = None,
    ):
        self.paths = list(paths)
        self.task = task
        self.size = size
        self._rng_seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        x_clean = load_image(self.paths[idx], self.size)
        rng = None
        if self._rng_seed is not None:
            rng = torch.Generator()
            rng.manual_seed(self._rng_seed + idx)   # reproducible per image
        observation, metadata = self.task.degrade(x_clean, rng=rng)
        return x_clean, observation, metadata


class FolderDataset(BaseDataset):

    def __init__(
        self,
        root: str | Path,
        task: Task,
        size: int = 256,
        limit: int | None = None,
        seed: int | None = None,
    ):
        root = Path(root)
        paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in EXTS
        )
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found in {root}")
        super().__init__(paths, task=task, size=size, seed=seed)


class CelebAHQDataset(FolderDataset):
    """
    Thin wrapper for CelebA-HQ 256.

    Expects the standard layout:
        <root>/
            celeba-256/
                00000.jpg
                00001.jpg
                ...
    """

    def __init__(self, root: str | Path, task: Task, split: str = "val",
                 limit: int | None = None, seed: int | None = None):
        img_dir = Path(root) / "celeba-256"
        if not img_dir.exists():
            img_dir = Path(root)   # fallback: images directly in root
        super().__init__(img_dir, task=task, size=256, limit=limit, seed=seed)



_DATASET_REGISTRY = {
    "folder": FolderDataset,
    "celeba_hq": CelebAHQDataset,
}


def build_dataset(cfg_dict: dict, task: Task) -> BaseDataset:

    cfg_dict = dict(cfg_dict)
    name = cfg_dict.pop("name")
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(_DATASET_REGISTRY)}")
    cls = _DATASET_REGISTRY[name]
    return cls(task=task, **cfg_dict)