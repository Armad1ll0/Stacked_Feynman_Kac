from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.gmm_experiments.gmm_analytic import (
    GMM, LinearObs, exact_posterior, make_default_problem, AnalyticEpsModel,
)
from experiments.gmm_experiments.gmm_experiment import (
    SimpleScheduler, GMMTask, GMMTaskConfig,
    run_point_estimate, run_tds_2d, run_tds_hmc_2d,
)
from experiments.gmm_experiments.gmm_metrics import all_metrics, mode_coverage
from analysis.gmm.plot_results_gmm import build_figure


def build_method_table(args):
    tds_hmc_kw = dict(
        stop_at_step=args.stop_at_step,
        hmc_iters=args.hmc_iters,
        L=args.L,
        lambda_like=args.lambda_like,
    )
    table = {
        "tds":     (run_tds_2d,     {}),
        "tds_hmc": (run_tds_hmc_2d, tds_hmc_kw),
    }
    return table


DEFAULT_METHODS = ["tds", "tds_hmc"]
BASELINE_ROWS = ["exact (ref)", "point estimate"]
DEFAULT_SIGMA_YS = [0.1, 0.5, 1.0, 3.0]


def _aggregate_metrics(metric_dicts: list[dict]) -> dict:
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([float(d[k]) for d in metric_dicts])
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def _pooled_for_seed(fn, kw, model, sched, task, y, args, base_seed: int):
    outs = []
    for r in range(args.reps):
        X, lw = fn(model, sched, task, y, args.P, args.steps,
                   seed=base_seed * 1000 + r, **kw)
        if lw is not None and lw.numel() == X.shape[0] \
                and not torch.allclose(lw, lw[0].expand_as(lw)):
            idx = torch.multinomial(
                torch.softmax(lw, dim=0), X.shape[0], replacement=True)
            X = X[idx]
        outs.append(X)
    return torch.cat(outs, dim=0)


def save_results(results, posts, keeps, sigma_ys, methods, seeds, args,
                  save_dir="gmm_results"):
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag else "run"

    raw_path = out_dir / f"{tag}_raw.pt"
    torch.save(
        dict(
            results=results,       
            posts=posts,       
            keeps=keeps,        
            sigma_ys=sigma_ys,
            methods=methods,
            seeds=seeds,
            args=vars(args),
        ),
        raw_path,
    )
    print(f"Wrote {raw_path}  (full results incl. sample tensors, for replotting)")

    summary = {}
    for sigma_y in sigma_ys:
        summary[str(sigma_y)] = {}
        for name, r in results[sigma_y].items():
            summary[str(sigma_y)][name] = {
                k: {"mean": v[0], "std": v[1]} for k, v in r["agg"].items()
            } | {"elapsed_s": r["elapsed"]}

    json_path = out_dir / f"{tag}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}  (mean/std metrics only, easy to diff/load)")


def main():
    ap = argparse.ArgumentParser()
    # problem difficulty knobs
    ap.add_argument("--spacing", type=float, default=3.0,
                    help="grid spacing between modes (smaller = closer/easier)")
    ap.add_argument("--center_scale", type=float, default=3.0,
                    help="how much wider the centre mode is")
    ap.add_argument("--base_var", type=float, default=0.25)
    ap.add_argument("--sigma_ys", nargs="+", type=float,
                    default=DEFAULT_SIGMA_YS,
                    help="observation-noise values to sweep (larger = softer "
                         "likelihood, more live modes). Pass one value for a "
                         "single-setting run.")
    ap.add_argument("--y", type=float, default=0.0,
                    help="observed value of x1 + x2")
    ap.add_argument("--grid_side", type=int, default=5,
                    help="modes per dimension (3 -> 9 modes, 5 -> 25)")
    # sampler budget
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--P", type=int, default=10000, help="particles per run")
    ap.add_argument("--reps", type=int, default=1,
                    help="independent runs pooled WITHIN one seed "
                         "(samples per seed = reps * P)")
    # TDS+HMC / PC knobs
    ap.add_argument("--stop_at_step", type=int, default=5)
    ap.add_argument("--hmc_iters", type=int, default=20)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--lambda_like", type=float, default=10.0)
    # DPS / PC knobs
    ap.add_argument("--dps_step_size", type=float, default=0.01)
    ap.add_argument("--pc_nfe_factor", type=float, default=1.0,
                    help="PC corrector budget = factor * hmc_iters * L. "
                         "1.0 matches Langevin updates to leapfrog STEPS; "
                         "2.0 matches MODEL CALLS of the uncached leapfrog. "
                         "Use the same convention as the image experiments.")
    # method selection
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"subset to run (default: {' '.join(DEFAULT_METHODS)})")
    ap.add_argument("--include_guidance", action="store_true",
                    help="(deprecated, no effect: DPS/PC are now included by "
                         "default; use --methods to run a subset)")
    # output
    ap.add_argument("--tag", default="", help="suffix for the figure filename")
    ap.add_argument("--no_fig", action="store_true")
    ap.add_argument("--fig_max_points", type=int, default=4000,
                    help="subsample cap per scatter panel (readability)")
    ap.add_argument("--contour_style", choices=["lines", "filled", "both"],
                    default="both",
                    help="how to draw the posterior density contour: "
                         "outline only, filled only, or both (default).")
    ap.add_argument("--contour_cmap", default="Greys",
                    help="matplotlib colormap for the filled contour "
                         "(ignored if --contour_style=lines), e.g. "
                         "viridis, Blues, magma, coolwarm.")
    ap.add_argument("--contour_levels", type=int, default=6,
                    help="number of contour levels/bands.")
    ap.add_argument("--contour_line_color", default="k",
                    help="color of the contour outline (ignored if "
                         "--contour_style=filled).")
    ap.add_argument("--suptitle_fontsize", type=float, default=13,
                    help="figure-level title font size.")
    ap.add_argument("--col_label_fontsize", type=float, default=13,
                    help="method-name column headers / 'Ground truth' "
                         "label font size.")
    ap.add_argument("--row_label_fontsize", type=float, default=12,
                    help="sigma_y row label font size.")
    ap.add_argument("--title_fontsize", type=float, default=9,
                    help="per-panel SW/modeErr metric title font size.")
    ap.add_argument("--failed_label_fontsize", type=float, default=11,
                    help="'failed' placeholder text font size.")
    ap.add_argument("--seed", type=int, default=0,
                    help="problem-geometry seed, and the single sampling "
                         "seed used if --seeds is not given")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="list of sampling seeds to average metrics over, "
                         "e.g. --seeds 0 1 2 3 4. Overrides --seed for the "
                         "sampling (not the problem geometry) if given.")
    ap.add_argument("--save_dir", default="gmm_results",
                    help="directory to write the raw results (.pt) and "
                         "summary (.json) into, for later replotting.")
    ap.add_argument("--no_save", action="store_true",
                    help="skip writing the raw/summary result files.")
    args = ap.parse_args()

    methods = args.methods if args.methods is not None else list(DEFAULT_METHODS)
    table = build_method_table(args)
    unknown = [m for m in methods if m not in table]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. "
                         f"Available: {list(table.keys())}")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    sigma_ys = list(args.sigma_ys)
    sched = SimpleScheduler(1000)
    N = args.reps * args.P

    print(f"Sweeping sigma_y over: {sigma_ys}")
    print(f"Seeds: {seeds}")
    print(f"Geometry: grid_side={args.grid_side} "
          f"({args.grid_side ** 2} prior modes) spacing={args.spacing} "
          f"center_scale={args.center_scale} y={args.y}")
    print(f"Budget per seed: {args.reps} runs x P={args.P} = {N} samples, "
          f"{args.steps} diffusion steps")
    print(f"Methods: {', '.join(methods)}\n")

    results = {}
    posts = {}
    keeps = {}

    for sigma_y in sigma_ys:
        torch.manual_seed(args.seed)

        prior, obs, _ = make_default_problem(
            seed=args.seed,
            center_scale=args.center_scale,
            spacing=args.spacing,
            base_var=args.base_var,
            grid_side=args.grid_side,
        )
        obs.sigma_y = sigma_y
        y = torch.tensor([args.y])
        post = exact_posterior(prior, obs, y)
        posts[sigma_y] = post

        model = AnalyticEpsModel(prior, sched)
        task = GMMTask(obs, GMMTaskConfig(use_fixed_sigma=True))

        keep = [i for i in range(post.means.shape[0])
                if post.weights[i] > 1e-4]
        keeps[sigma_y] = keep

        print("=" * 68)
        print(f"sigma_y = {sigma_y}")
        print(f"Surviving posterior modes: {len(keep)} of "
              f"{post.means.shape[0]}")
        for i in keep:
            print(f"   mode {i}: w={float(post.weights[i]):.4f} "
                  f"mean={post.means[i].numpy().round(2)}")
        print(f"   true weights@modes = {post.weights[keep].numpy().round(3)}")
        print()

        ref = post.sample(20000)
        rows = {}

        for name in BASELINE_ROWS + methods:
            seed_metrics, seed_mc, plot_samples = [], [], []
            t0 = time.time()
            ok = True
            for seed in seeds:
                torch.manual_seed(seed)
                try:
                    if name == "exact (ref)":
                        X = post.sample(N)
                    elif name == "point estimate":
                        X = run_point_estimate(post, N)
                    else:
                        fn, kw = table[name]
                        X = _pooled_for_seed(fn, kw, model, sched, task, y,
                                              args, seed)
                except Exception as e:
                    print(f"{name:16s} seed={seed} FAILED: "
                          f"{type(e).__name__}: {e}")
                    ok = False
                    break
                seed_metrics.append(all_metrics(X, ref, post))
                seed_mc.append(mode_coverage(X, post))
                plot_samples.append(X)

            if not ok or not seed_metrics:
                continue

            elapsed = time.time() - t0
            agg = _aggregate_metrics(seed_metrics)
            mc_mean = np.mean(np.stack(seed_mc), axis=0)
            pooled_plot_samples = torch.cat(plot_samples, dim=0)

            rows[name] = dict(agg=agg, mc_mean=mc_mean,
                               plot_samples=pooled_plot_samples,
                               elapsed=elapsed)

            sw_m, sw_s = agg["sw"]
            mmd_m, mmd_s = agg["mmd"]
            e_m, e_s = agg["energy"]
            me_m, me_s = agg["mode_cov_err"]
            print(f"{name:16s} SW={sw_m:7.3f}+-{sw_s:.3f} "
                  f"MMD={mmd_m:.4f}+-{mmd_s:.4f} "
                  f"E={e_m:8.4f}+-{e_s:.4f} "
                  f"modeErr={me_m:.3f}+-{me_s:.3f} "
                  f"({elapsed:.0f}s, {len(seeds)} seed(s))")
            print(f"{'':16s} mass@modes(mean over seeds)="
                  f"{mc_mean[keep].round(3)}")

        results[sigma_y] = rows
        print()

    for sigma_y in sigma_ys:
        print(f"\n### sigma_y = {sigma_y}")
        print("| method | SW | MMD | energy | mode err |")
        print("|---|---|---|---|---|")
        for name, r in results[sigma_y].items():
            agg = r["agg"]
            sw_m, sw_s = agg["sw"]
            mmd_m, mmd_s = agg["mmd"]
            e_m, e_s = agg["energy"]
            me_m, me_s = agg["mode_cov_err"]
            print(f"| {name} | {sw_m:.3f}+-{sw_s:.3f} | "
                  f"{mmd_m:.4f}+-{mmd_s:.4f} | {e_m:.4f}+-{e_s:.4f} | "
                  f"{me_m:.3f}+-{me_s:.3f} |")

    all_names = []
    for sigma_y in sigma_ys:
        for name in results[sigma_y]:
            if name not in all_names:
                all_names.append(name)

    print("\n### SW across sigma_y (mean +/- std over seeds)")
    print("| method | " + " | ".join(f"sigma_y={s}" for s in sigma_ys) + " |")
    print("|---" * (len(sigma_ys) + 1) + "|")
    for name in all_names:
        cells = []
        for sigma_y in sigma_ys:
            r = results[sigma_y].get(name)
            if r is None:
                cells.append("--")
            else:
                sw_m, sw_s = r["agg"]["sw"]
                cells.append(f"{sw_m:.3f}+-{sw_s:.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    if not args.no_save:
        save_results(results, posts, keeps, sigma_ys, methods, seeds, args,
                     save_dir=args.save_dir)

    if args.no_fig:
        return

    plot_methods = [
        m for m in methods
        if any(m in results[s] for s in sigma_ys)
    ]
    fname = f"gmm_overlay{('_' + args.tag) if args.tag else ''}.png"
    build_figure(results, posts, sigma_ys, plot_methods, seeds, args, fname)


if __name__ == "__main__":
    main()
