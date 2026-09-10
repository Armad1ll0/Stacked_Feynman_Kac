import argparse
import os
import time

import torch
from diffusers import DDPMPipeline

from sampler.tsmc import (
    SamplerConfig,
    run_tds,
    run_tds_then_hmc_denoised,
)
from experiments.class_conditional_experiments.class_task import ClassConditionalTask, ClassConditionalConfig


def _run_sampler(method, model, scheduler, task, observation, metadata, cfg,
                 device):
    if method == "tds":
        res = run_tds(
            model=model, scheduler=scheduler, task=task,
            observation=observation, metadata=metadata,
            cfg=cfg, device=device,
        )
        return res.particles, res.log_weights
    elif method == "tds_hmc":
        res = run_tds_then_hmc_denoised(
            model=model, scheduler=scheduler, task=task,
            observation=observation, metadata=metadata,
            cfg=cfg, device=device,
        )
        return res.particles_hmc, res.log_weights
    else:
        raise ValueError(f"unknown method {method!r}, expected 'tds' or 'tds_hmc'")
 
 
def _draw_independent(particles, log_weights, n):
    w = torch.softmax(log_weights, dim=0)
    idx = torch.multinomial(w, n, replacement=True)
    return particles[idx]
 

def generate_for_class(method, model, scheduler, task, label, *,
                       samples_per_class, cfg, device, image_shape,
                       samples_per_run=1):

    C, H, W = image_shape

    observation = torch.zeros(C, H, W, device=device)
    metadata = {"label": int(label)}
 
    collected = []
    n_have = 0
    runs = 0
    t0 = time.time()
    while n_have < samples_per_class:
        particles, log_w = _run_sampler(
            method, model, scheduler, task, observation, metadata, cfg, device
        )
        draws = _draw_independent(particles, log_w, samples_per_run)
        collected.append(draws.detach().cpu())
        n_have += draws.shape[0]
        runs += 1
 
    samples = torch.cat(collected, dim=0)[:samples_per_class] 
    dt = time.time() - t0
    print(f"  class {label}: {samples.shape[0]} samples "
          f"({runs} runs x {samples_per_run}/run) in {dt:.1f}s")
    return samples
 

def main():
    p = argparse.ArgumentParser(description="Class-conditional MNIST sampling")
    p.add_argument("--method", choices=["tds", "tds_hmc"], default="tds")
    p.add_argument("--model-id", type=str, default="1aurent/ddpm-mnist",
                   help="HF model id or local path of the unconditional MNIST DDPM")
    p.add_argument("--classifier-ckpt", type=str, default="clf_guide.pt")
    p.add_argument("--out-dir", type=str, default="samples")
    p.add_argument("--samples-per-class", type=int, default=1000)
    p.add_argument("--samples-per-run", type=int, default=1,
                   help="weight-sampled particles drawn per run; 1 = fully "
                        "independent samples (one per restart)")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    # sampler hyperparameters
    p.add_argument("--num-particles", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--ess-threshold", type=float, default=0.5)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    # tds_hmc-only knobs (ignored for method=tds)
    p.add_argument("--hmc-iters", type=int, default=20)
    p.add_argument("--leapfrog-L", type=int, default=3)
    p.add_argument("--lambda-like", type=float, default=1.0)
    p.add_argument("--hmc-step-scale", type=float, default=0.1)
    p.add_argument("--stop-at-step", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
 
    print(f"loading pipeline {args.model_id} ...")
    pipe = DDPMPipeline.from_pretrained(args.model_id)
    model = pipe.unet.to(device).eval()
    scheduler = pipe.scheduler
 
    C = model.config.in_channels
    H = W = model.config.sample_size
    print(f"model: in_channels={C}, sample_size={H}, "
          f"prediction_type={getattr(scheduler.config, 'prediction_type', 'epsilon')}")
 
    task_cfg = ClassConditionalConfig(
        classifier_ckpt=args.classifier_ckpt,
        guidance_scale=args.guidance_scale,
        device=str(device),
    )
    task = ClassConditionalTask(task_cfg)
 
    cfg = SamplerConfig(
        num_particles=args.num_particles,
        num_steps=args.num_steps,
        resample_ess_threshold=args.ess_threshold,
        paste_observation=False,   
        hmc_iters=args.hmc_iters,
        L=args.leapfrog_L,
        lambda_like=args.lambda_like,
        hmc_step_scale=args.hmc_step_scale,
        stop_at_step=args.stop_at_step,
        verbose_every=0,
    )
 
    print(f"generating with method={args.method}, "
          f"{args.samples_per_class} samples/class ...")
    for label in args.classes:
        samples = generate_for_class(
            args.method, model, scheduler, task, label,
            samples_per_class=args.samples_per_class,
            cfg=cfg, device=device, image_shape=(C, H, W),
            samples_per_run=args.samples_per_run,
        )
        out_path = os.path.join(args.out_dir, f"class_{label}.pt")
        torch.save({"label": label, "samples": samples,
                    "method": args.method}, out_path)
        print(f"    saved -> {out_path}")
 
    print("done.")
 
 
if __name__ == "__main__":
    main()
