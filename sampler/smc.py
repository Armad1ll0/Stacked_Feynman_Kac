
import torch


def ess(log_weights: torch.Tensor) -> float:
    w = torch.softmax(log_weights, dim=0)
    return (1.0 / (w ** 2).sum()).item()


def systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    P = log_weights.shape[0]
    device = log_weights.device
    weights = torch.softmax(log_weights, dim=0)
    cumsum = torch.cumsum(weights, dim=0)
    u = (torch.arange(P, device=device, dtype=torch.float32) +
         torch.rand(1, device=device)) / P
    indices = torch.searchsorted(cumsum.contiguous(), u.contiguous()).clamp(max=P - 1)
    return indices


def multinomial_resample(log_weights: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(log_weights, dim=0)
    indices = torch.multinomial(weights, num_samples=weights.shape[0], replacement=True)
    return indices


def stratified_resample(log_weights: torch.Tensor) -> torch.Tensor:
    P = log_weights.shape[0]
    device = log_weights.device
    weights = torch.softmax(log_weights, dim=0)
    cumsum = torch.cumsum(weights, dim=0)
    u = (torch.arange(P, device=device, dtype=torch.float32) +
         torch.rand(P, device=device)) / P
    indices = torch.searchsorted(cumsum.contiguous(), u.contiguous()).clamp(max=P - 1)
    return indices


RESAMPLE_FNS = {
    "systematic": systematic_resample,
    "multinomial": multinomial_resample,
    "stratified": stratified_resample,
}


def get_resampler(name: str):
    if name not in RESAMPLE_FNS:
        raise ValueError(f"Unknown resampler {name!r}. Options: {list(RESAMPLE_FNS)}")
    return RESAMPLE_FNS[name]