import argparse
import json
import os

import torch
from diffusers import DDPMPipeline

from experiments.class_conditional_experiments.train_classifiers import train_one
from experiments.class_conditional_experiments.class_task import ClassConditionalTask, ClassConditionalConfig
from sampler.tsmc import SamplerConfig
from experiments.class_conditional_experiments.sample_mnist import generate_for_class
import experiments.class_conditional_experiments.metrics as metrics_mod
from experiments.class_conditional_experiments.plot_samples import plot_grid


def ensure_classifiers(guide_ckpt, eval_ckpt, *, epochs, data_root, device):
    common = dict(epochs=epochs, lr=1e-3, data_root=data_root,
                  batch_size=128, num_workers=2, device=device)
    if os.path.exists(guide_ckpt):
        print(f"guidance classifier present: {guide_ckpt}")
    else:
        train_one("guidance", seed=0, out_path=guide_ckpt, **common)
    if os.path.exists(eval_ckpt):
        print(f"eval classifier present: {eval_ckpt}")
    else:
        train_one("eval", seed=1, out_path=eval_ckpt, **common)
 
 
def generate_method(method, model, scheduler, task, cfg, *,
                    classes, samples_per_class, device, image_shape, out_dir,
                    samples_per_run=1):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- generating: method={method} -> {out_dir} ---")
    for label in classes:
        samples = generate_for_class(
            method, model, scheduler, task, label,
            samples_per_class=samples_per_class,
            cfg=cfg, device=device, image_shape=image_shape,
            samples_per_run=samples_per_run,
        )
        torch.save({"label": label, "samples": samples, "method": method},
                   os.path.join(out_dir, f"class_{label}.pt"))


def print_comparison(all_results):
    methods = list(all_results.keys())
    preferred = ["accuracy", "precision", "recall",
                 "fid_mnist", "kid_mnist_mean",
                 "fid_inception", "kid_inception_mean"]
    keys = [k for k in preferred if any(k in all_results[m]["aggregate"]
                                        for m in methods)]
 
    col_w = max(14, max(len(m) for m in methods) + 2)
    header = "metric".ljust(22) + "".join(m.ljust(col_w) for m in methods)
    print("\n" + "=" * len(header))
    print("COMPARISON (macro-average over classes)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in keys:
        row = k.ljust(22)
        for m in methods:
            v = all_results[m]["aggregate"].get(k)
            row += (f"{v:.4f}" if v is not None else "-").ljust(col_w)
        print(row)
    print("=" * len(header))
 
    print("\nReading: high accuracy = right class; high recall with high "
          "precision = full within-class coverage (not mode collapse); "
          "lower FID/KID = closer to the true class distribution.")
 
 
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Class-conditional MNIST experiment")
    p.add_argument("--model-id", type=str, default='1aurent/ddpm-mnist',
                   help="HF id or path of the unconditional MNIST DDPM")
    p.add_argument("--methods", nargs="+", default=["tds", "tds_hmc"],
                   choices=["tds", "tds_hmc"])
    p.add_argument("--out-dir", type=str, default="runs/exp/0p1")
    p.add_argument("--guide-ckpt", type=str, default="clf_guide.pt")
    p.add_argument("--eval-ckpt", type=str, default="clf_eval.pt")
    p.add_argument("--classifier-epochs", type=int, default=3)
    p.add_argument("--samples-per-class", type=int, default=1000)
    p.add_argument("--samples-per-run", type=int, default=100,
                   help="weight-sampled particles per run; 1 = independent")
    p.add_argument("--plot-samples-per-cell", type=int, default=1,
                   help="stack k samples per cell in the grid to show diversity")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--feature-spaces", nargs="+", default=["mnist"],
                   choices=["mnist", "inception"])
    # sampler hyperparameters
    p.add_argument("--num-particles", type=int, default=100)
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--ess-threshold", type=float, default=0.5)
    p.add_argument("--guidance-scale", type=float, default=0.1)
    # tds_hmc-only knobs
    p.add_argument("--hmc-iters", type=int, default=20)
    p.add_argument("--leapfrog-L", type=int, default=3)
    p.add_argument("--lambda-like", type=float, default=10.0)
    p.add_argument("--hmc-step-scale", type=float, default=0.1)
    p.add_argument("--stop-at-step", type=int, default=5)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
 
    ensure_classifiers(
        args.guide_ckpt, args.eval_ckpt,
        epochs=args.classifier_epochs, data_root=args.data_root,
        device=str(device),
    )
 
    print(f"\nloading pipeline {args.model_id} ...")
    pipe = DDPMPipeline.from_pretrained(args.model_id)
    model = pipe.unet.to(device).eval()
    scheduler = pipe.scheduler
    C = model.config.in_channels
    H = W = model.config.sample_size
    print(f"model: in_channels={C}, sample_size={H}")
 
    task = ClassConditionalTask(ClassConditionalConfig(
        classifier_ckpt=args.guide_ckpt,
        guidance_scale=args.guidance_scale,
        device=str(device),
    ))
 
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
 
    all_results = {}
    method_dirs = {}
    for method in args.methods:
        samples_dir = os.path.join(args.out_dir, method)
        method_dirs[method] = samples_dir
        generate_method(
            method, model, scheduler, task, cfg,
            classes=args.classes, samples_per_class=args.samples_per_class,
            device=device, image_shape=(C, H, W), out_dir=samples_dir,
            samples_per_run=args.samples_per_run,
        )
        print(f"\n--- scoring: method={method} ---")
        res = metrics_mod.evaluate(
            samples_dir=samples_dir, eval_ckpt=args.eval_ckpt,
            feature_spaces=args.feature_spaces, data_root=args.data_root,
            device=str(device),
        )
        all_results[method] = res
        with open(os.path.join(args.out_dir, f"results_{method}.json"), "w") as f:
            json.dump(res, f, indent=2)
 
    print_comparison(all_results)
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump({m: all_results[m]["aggregate"] for m in all_results},
                  f, indent=2)
 
    grid_methods = [m for m in ("tds", "tds_hmc") if m in method_dirs]
    if grid_methods:
        row_label = {"tds": "TDS", "tds_hmc": "TDS+HMC"}
        try:
            plot_grid(
                method_dirs=[(row_label[m], method_dirs[m]) for m in grid_methods],
                out_path=os.path.join(args.out_dir, "samples_grid.png"),
                classes=args.classes,
                row_labels=[row_label[m] for m in grid_methods],
                samples_per_cell=args.plot_samples_per_cell,
            )
        except Exception as e: 
            print(f"warning: sample grid plotting failed ({e})")
 
    print(f"\nall outputs under: {args.out_dir}")
 
 
if __name__ == "__main__":
    main()