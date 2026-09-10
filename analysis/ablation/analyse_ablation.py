import argparse
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from experiments.configs import RESULTS_DIR


METRICS = ["psnr", "ssim", "obs_consistency", "runtime_s", "ess_mean", "ess_min", "diversity", "var_diversity"]
LOWER_IS_BETTER = {"obs_consistency", "runtime_s"}

WALLCLOCK_TABLE_METRICS = ["runtime_s", "psnr", "ssim", "obs_consistency", "ess_mean", "ess_min", "diversity", "var_diversity"]

GRID_METRICS = ["psnr", "ssim"]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


TMLR_COLOUR = "#0072B2"

TMLR_SINGLE_W = 3.25   
TMLR_GRID_COL = 2.5    
TMLR_GRID_ROW = 2.0    

TMLR_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    # Lines / markers
    "lines.linewidth":    1.5,
    "lines.markersize":   4,
    "patch.linewidth":    0.5,
    # Axes
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "grid.color":         "#bbbbbb",
    # Ticks
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # Output
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,   # TrueType embedding
    "ps.fonttype":        42,
}

FIGURE_SIZE = (5, 4)  


@contextmanager
def tmlr_style():
    with plt.rc_context(TMLR_RC):
        yield


def _metric_label(metric: str) -> str:
    labels = {
        "psnr":            r"PSNR (dB) $\uparrow$",
        "ssim":            r"SSIM $\uparrow$",
        "obs_consistency": r"Obs. Consistency (RMSE) $\downarrow$",
        "runtime_s":       r"Runtime (s) $\downarrow$",
        "ess_mean":        r"ESS mean $\uparrow$",
        "ess_min":         r"ESS min $\uparrow$",
        "diversity":       r"Diversity (L2) $\uparrow$",
        "var_diversity":   r"Var. Diversity $\uparrow$",
    }
    return labels.get(metric, metric)


def _task_label(task: str) -> str:
    return {
        "inpaint_random": "Random Mask",
        "inpaint":        "Inpainting",
        "superres":       "Super-Res.",
        "deblur":         "Deblurring",
    }.get(task, task)


def _dataset_label(dataset: str) -> str:
    return {
        "mnist":       "MNIST",
        "flowers":     "Flowers",
        "butterflies": "Butterflies",
        "celeba":      "CelebA",
    }.get(dataset, dataset.capitalize())


def _metric_header(metric: str) -> str:
    labels = {
        "psnr":            "PSNR (dB) ↑",
        "ssim":            "SSIM ↑",
        "obs_consistency": "Obs. Cons. ↓",
        "runtime_s":       "Runtime (s) ↓",
        "ess_mean":        "ESS mean ↑",
        "ess_min":         "ESS min ↑",
        "diversity":       "Diversity ↑",
        "var_diversity":   "Var. Div. ↑",
    }
    return labels.get(metric, metric)



def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {path}\n"
            "Run run_ablations.py first to generate results."
        )
    df = pd.read_csv(path)
    df.replace("nan", np.nan, inplace=True)
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def _save_pdf(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _set_tight_ylim(ax, means, stds, pad_frac=0.12):
    y_min = (means - stds).min()
    y_max = (means + stds).max()
    pad   = (y_max - y_min) * pad_frac or 0.01
    ax.set_ylim(y_min - pad, y_max + pad)


def compute_stats(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = {}
    for m in metrics:
        if m not in df.columns:
            continue
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"]  = (m, "std")
    return df.groupby(group_cols).agg(**agg).reset_index()


def make_wallclock_markdown(df: pd.DataFrame, eval_dir: Path, metrics: list[str]):
    table_metrics = [m for m in WALLCLOCK_TABLE_METRICS if m in metrics and m in df.columns]
    datasets = sorted(df["dataset"].unique())
    tasks    = sorted(df["task"].unique())

    lines = ["# Wallclock Ablation Results\n"]

    for dataset in datasets:
        for task in tasks:
            sub = df[(df["dataset"] == dataset) & (df["task"] == task)]
            if sub.empty:
                continue

            methods = sorted(sub["method"].unique())
            lines.append(f"## {dataset} / {task}\n")

            headers = ["Method"] + [_metric_header(m) for m in table_metrics]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for method in methods:
                msub = sub[sub["method"] == method]
                row  = [method]
                for m in table_metrics:
                    vals = msub[m].dropna()
                    if vals.empty:
                        row.append("—")
                    else:
                        mean = vals.mean()
                        std  = vals.std() if len(vals) > 1 else 0.0
                        row.append(f"{mean:.3f} ± {std:.3f}")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

    md_path = eval_dir / "stats_table.md"
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")


def plot_wallclock(df: pd.DataFrame, plot_dir: Path, metrics: list[str]):
    datasets = df["dataset"].unique()
    tasks    = df["task"].unique()
    methods  = df["method"].unique()
    colours  = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}

    for dataset in datasets:
        for task in tasks:
            sub = df[(df["dataset"] == dataset) & (df["task"] == task)]
            if sub.empty:
                continue

            for metric in metrics:
                if metric not in sub.columns or sub[metric].isna().all():
                    continue

                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                x     = np.arange(len(methods))
                width = 0.6
                means, stds = [], []

                for method in methods:
                    vals = sub[sub["method"] == method][metric].dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    stds.append(vals.std()   if len(vals) > 1 else 0.0)

                bars = ax.bar(
                    x, means, width,
                    yerr=stds, capsize=4,
                    color=[colours[m] for m in methods],
                    edgecolor="white", linewidth=0.5,
                )

                for bar, mean in zip(bars, means):
                    if np.isnan(mean):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(stds) * 0.05 if any(s > 0 for s in stds) else 0.01),
                        f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
                ax.set_ylabel(_metric_label(metric), fontsize=9)
                ax.set_title(f"{dataset} / {task}", fontsize=10, fontweight="bold")
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()

                _save_pdf(fig, plot_dir / f"{dataset}_{task}_{metric}.pdf")


def plot_stop_at_step_grid(df: pd.DataFrame, plot_dir: Path, metric: str):

    DATASET_ORDER = ["mnist", "flowers", "butterflies", "celeba"]
    all_datasets  = set(df["dataset"].unique())
    datasets = [d for d in DATASET_ORDER if d in all_datasets] + \
               sorted(all_datasets - set(DATASET_ORDER))

    tasks  = sorted(df["task"].unique())
    n_rows = len(tasks)
    n_cols = len(datasets)

    with tmlr_style():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(TMLR_GRID_COL * n_cols, TMLR_GRID_ROW * n_rows),
            squeeze=False,
        )

        for r, task in enumerate(tasks):
            for c, dataset in enumerate(datasets):
                ax  = axes[r][c]
                sub = df[(df["dataset"] == dataset) & (df["task"] == task)].copy()

                if sub.empty or metric not in sub.columns or sub[metric].isna().all():
                    ax.set_visible(False)
                    continue

                sub["ablation_param"] = sub["ablation_param"].astype(int)
                stop_values = sorted(sub["ablation_param"].dropna().unique())
                grp   = sub.groupby("ablation_param")[metric]
                means = grp.mean().reindex(stop_values)
                stds  = grp.std().reindex(stop_values).fillna(0)
                xs    = np.array(stop_values)

                ax.plot(xs, means.values, marker="o", color=TMLR_COLOUR)
                ax.fill_between(
                    xs,
                    means.values - stds.values,
                    means.values + stds.values,
                    alpha=0.15, color=TMLR_COLOUR,
                )

                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
                ax.set_xticks([xs[i] for i in np.linspace(0, len(xs)-1, min(5, len(xs)), dtype=int)])
                ax.set_xticklabels(ax.get_xticks().astype(int), rotation=45, ha="right")

                _set_tight_ylim(ax, means, stds)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

                if r == 0:
                    ax.set_title(_dataset_label(dataset), fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{_task_label(task)}\n{_metric_label(metric)}")
                else:
                    ax.set_ylabel("")
                if r == n_rows - 1:
                    ax.set_xlabel(r"$t_{\mathrm{stop}}$")
                else:
                    ax.set_xlabel("")

        fig.tight_layout(h_pad=0.8, w_pad=0.6)
        _save_pdf(fig, plot_dir / f"{metric}_grid.pdf")


def plot_stop_at_step_individual(df: pd.DataFrame, plot_dir: Path, metrics: list[str]):
    datasets = df["dataset"].unique()
    tasks    = df["task"].unique()

    for dataset in datasets:
        for task in tasks:
            sub = df[(df["dataset"] == dataset) & (df["task"] == task)].copy()
            if sub.empty:
                continue
            sub["ablation_param"] = sub["ablation_param"].astype(int)
            stop_values = sorted(sub["ablation_param"].dropna().unique())

            for metric in metrics:
                if metric not in sub.columns or sub[metric].isna().all():
                    continue

                with tmlr_style():
                    fig, ax = plt.subplots(
                        figsize=(TMLR_SINGLE_W, TMLR_SINGLE_W * 0.75)
                    )

                    grp   = sub.groupby("ablation_param")[metric]
                    means = grp.mean().reindex(stop_values)
                    stds  = grp.std().reindex(stop_values).fillna(0)
                    xs    = np.array(stop_values)

                    ax.plot(xs, means.values, marker="o", color=TMLR_COLOUR,
                            label="Mean")
                    ax.fill_between(
                        xs,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=TMLR_COLOUR,
                        label=r"$\pm1$ std",
                    )

                    ax.set_xticks(xs)
                    if len(xs) > 5:
                        ax.set_xticklabels(xs, rotation=45, ha="right")

                    _set_tight_ylim(ax, means, stds)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

                    ax.set_xlabel(r"$t_{\mathrm{stop}}$")
                    ax.set_ylabel(_metric_label(metric))
                    ax.set_title(
                        f"{_dataset_label(dataset)} — {_task_label(task)}",
                        fontweight="bold",
                    )
                    ax.legend(frameon=False, loc="best")
                    fig.tight_layout()

                    _save_pdf(fig, plot_dir / f"{dataset}_{task}_{metric}.pdf")


def evaluate_ablation(ablation: str, metrics: list[str]):
    root     = Path(RESULTS_DIR) / f"ablation_{ablation}"
    csv_path = root / "summary.csv"
    eval_dir = root / "eval"
    plot_dir = eval_dir / "plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Ablation: {ablation}")
    print(f"Reading : {csv_path}")

    df = _load_csv(csv_path)
    print(f"Rows    : {len(df)}")

    if ablation == "wallclock":
        group_cols = ["dataset", "task", "method"]
    else:
        group_cols = ["dataset", "task", "ablation_param"]

    stats = compute_stats(df, group_cols, metrics)
    stats_path = eval_dir / "stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"\nStats CSV → {stats_path}")
    print(stats.to_string(index=False))

    if ablation == "wallclock":
        print(f"\nGenerating markdown table → {eval_dir}/")
        make_wallclock_markdown(df, eval_dir, metrics)
        print(f"\nGenerating bar plots → {plot_dir}/")
        plot_wallclock(df, plot_dir, metrics)

    else:
        grid_metrics       = [m for m in GRID_METRICS if m in metrics]
        individual_metrics = [m for m in metrics if m not in GRID_METRICS]

        if grid_metrics:
            print(f"\nGenerating TMLR grid plots ({grid_metrics}) → {plot_dir}/")
            for metric in grid_metrics:
                plot_stop_at_step_grid(df, plot_dir, metric)

        if individual_metrics:
            print(f"\nGenerating TMLR individual plots ({individual_metrics}) → {plot_dir}/")
            plot_stop_at_step_individual(df, plot_dir, individual_metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation", nargs="*",
        default=["wallclock", "stop_at_step"],
        choices=["wallclock", "stop_at_step"],
        help="Which ablation(s) to evaluate (default: both)",
    )
    p.add_argument(
        "--metrics", nargs="*",
        default=METRICS,
        help="Metrics to include in plots and stats table",
    )
    args = p.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for ablation in args.ablation:
            evaluate_ablation(ablation, args.metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
