import hashlib
import torch

from tasks.inpainting import InpaintingTask
from tasks.super_resolution import SuperResTask
from tasks.deblurring import DeblurringTask

def degradation_seed(task_name: str, idx: int) -> int:
    h = hashlib.sha256(f"degrade:{task_name}:{idx}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def make_observation(task, x_clean, task_name: str, idx: int, sigma_y: float):
    torch.manual_seed(degradation_seed(task_name, idx))
    y, metadata = task.degrade(x_clean)

    if sigma_y > 0:
        g = torch.Generator(device="cpu").manual_seed(
            degradation_seed(f"noise:{task_name}", idx))
        noise = torch.randn(y.shape, generator=g, dtype=torch.float32)
        noise = noise.to(y.device, y.dtype)
        mask = metadata.get("mask")
        if mask is not None:                     # inpainting: observed only
            m = mask.to(noise.device, noise.dtype)
            if m.dim() == 4 and noise.dim() == 3:
                m = m.squeeze(0)
            noise = noise * m
        y = y + sigma_y * noise

    metadata = {**metadata, "sigma_y": sigma_y}
    return y, metadata


def apply_forward(task, x, metadata):

    x = x.unsqueeze(0) if x.dim() == 3 else x
    if isinstance(task, InpaintingTask):
        m = metadata["mask"]
        m = m.unsqueeze(0) if m.dim() == 3 else m
        return m.to(x.device, x.dtype) * x
    if isinstance(task, SuperResTask):
        return task._downsample(x)
    if isinstance(task, DeblurringTask):
        return task._blur(x)
    raise TypeError(f"apply_forward: unknown task {type(task).__name__}")