from tasks.inpainting import InpaintingTask, InpaintingConfig
from tasks.super_resolution import SuperResTask, SuperResConfig
from tasks.deblurring import DeblurringTask, DeblurringConfig
from tasks.base import Task

_REGISTRY = {
    "inpainting": (InpaintingTask, InpaintingConfig),
    "super_resolution": (SuperResTask, SuperResConfig),
    "deblurring": (DeblurringTask, DeblurringConfig),
}


def build_task(cfg_dict: dict) -> Task:
    name = cfg_dict.pop("name")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown task {name!r}. Available: {list(_REGISTRY)}")
    task_cls, config_cls = _REGISTRY[name]
    config = config_cls(**cfg_dict)
    return task_cls(config)