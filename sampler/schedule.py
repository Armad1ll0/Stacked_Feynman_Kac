import torch
from diffusers import DDPMScheduler


def get_schedule_tensors(scheduler: DDPMScheduler, device: torch.device):
    """
    Returns:
        alphas_cumprod : (T,) -- alpha_bar_t for each timestep
        betas          : (T,) -- beta_t for each timestep
    """
    alphas_cumprod = torch.tensor(
        scheduler.alphas_cumprod, dtype=torch.float32, device=device
    )
    betas = torch.tensor(
        scheduler.betas, dtype=torch.float32, device=device
    )
    return alphas_cumprod, betas


def q_posterior_mean(
    x_start: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean of q(x_{t-1} | x_t, x_0):

        mu = coeff_x0 * x_0 + coeff_xt * x_t
    """
    a_bar_t   = alphas_cumprod[t]
    a_bar_tm1 = alphas_cumprod[t - 1] if t > 0 else x_t.new_tensor(1.0)
    beta_t    = betas[t]

    coeff_x0 = (a_bar_tm1.sqrt() * beta_t) / (1.0 - a_bar_t)
    coeff_xt  = ((1.0 - a_bar_tm1) * (1.0 - beta_t).sqrt()) / (1.0 - a_bar_t)
    return coeff_x0 * x_start + coeff_xt * x_t


def posterior_variance(
    t: int,
    alphas_cumprod: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Variance of q(x_{t-1} | x_t, x_0):

        sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    """
    a_bar_t   = alphas_cumprod[t]
    a_bar_tm1 = alphas_cumprod[t - 1] if t > 0 else alphas_cumprod.new_tensor(1.0)
    beta_t    = betas[t]
    return beta_t * (1.0 - a_bar_tm1) / (1.0 - a_bar_t)


@torch.no_grad()
def predict_x0(
    model,
    x_t: torch.Tensor,
    t_index: int,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the UNet to get an epsilon prediction and recover x_0:

        x_0_hat = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)

    """
    a_bar = alphas_cumprod[t_index]
    t_tensor = torch.full(
        (x_t.shape[0],), t_index, device=device, dtype=torch.long
    )
    eps_pred = model(x_t, t_tensor).sample
    x0_hat = (x_t - (1.0 - a_bar).sqrt() * eps_pred) / a_bar.sqrt()
    return x0_hat