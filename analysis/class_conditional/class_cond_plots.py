import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

RESULT_FILE_RE = re.compile(r"results_(?P<method>.+)\.json$")
CLASS_FILE_RE = re.compile(r"class_(?P<label>\d+)\.pt$")

METHODS = ["tds", "tds_hmc"]
METHOD_LABELS = {"tds": "TDS", "tds_hmc": "rSFK-uHMC"}
METHOD_COLORS = {"tds": "#4C72B0", "tds_hmc": "#DD8452"}
METHOD_MARKERS = {"tds": "o", "tds_hmc": "s"}

DEFAULT_SWEEP_METRICS = ["fid_mnist", "kid_mnist_mean", "accuracy", "precision", "recall"]
DEFAULT_PANEL_METRICS = ["fid_mnist"]

METRIC_LABELS = {
    "fid_mnist": "FID",
    "kid_mnist_mean": "KID",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
}
LOWER_IS_BETTER = {"fid_mnist", "kid_mnist_mean"}


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   
        "ps.fonttype": 42,
    })


def parse_guidance_strength(folder_name: str) -> float:
    try:
        return float(folder_name)
    except ValueError:
        pass
    if "p" in folder_name:
        try:
            return float(folder_name.replace("p", "."))
        except ValueError:
            pass
    raise ValueError(
        f"Could not parse a guidance strength from folder name '{folder_name}'. "
        "Expected a plain number ('5', '0.1') or 'p'-decimal ('0p1')."
    )


def strength_dirs_sorted(root: Path):
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: parse_guidance_strength(d.name),
    )
    if not dirs:
        raise FileNotFoundError(f"No subdirectories found under {root}")
    return dirs


def load_results(root: Path):
    per_class_rows, aggregate_rows = [], []

    for strength_dir in strength_dirs_sorted(root):
        strength = parse_guidance_strength(strength_dir.name)
        result_files = sorted(strength_dir.glob("results_*.json"))
        if not result_files:
            print(f"warning: no results_*.json found in {strength_dir}, skipping")
            continue

        for result_file in result_files:
            m = RESULT_FILE_RE.match(result_file.name)
            if not m:
                continue
            method = m.group("method")

            with open(result_file) as f:
                data = json.load(f)

            agg = data.get("aggregate", {})
            aggregate_rows.append({"guidance_strength": strength, "method": method, **agg})

            for class_label, metrics in data.get("per_class", {}).items():
                per_class_rows.append({
                    "guidance_strength": strength,
                    "method": method,
                    "class": int(class_label),
                    **metrics,
                })

    per_class_df = pd.DataFrame(per_class_rows).sort_values(
        ["guidance_strength", "method", "class"]).reset_index(drop=True)
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["guidance_strength", "method"]).reset_index(drop=True)

    return per_class_df, aggregate_df


def write_summary_latex(aggregate_df: pd.DataFrame, out_path: Path,
                        metrics=("accuracy", "precision", "recall",
                                 "fid_mnist", "kid_mnist_mean")):
    methods = sorted(aggregate_df["method"].unique())
    strengths = sorted(aggregate_df["guidance_strength"].unique())
    metrics = [m for m in metrics if m in aggregate_df.columns]

    col_spec = "l" + "".join("c" * len(metrics) for _ in methods)
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{" + col_spec + "}", r"\toprule"]

    header_top = ["Guidance strength"]
    for method in methods:
        header_top.append(r"\multicolumn{%d}{c}{%s}" % (len(metrics), METHOD_LABELS.get(method, method)))
    lines.append(" & ".join(header_top) + r" \\")

    cmidrules, col = [], 2
    for _ in methods:
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(metrics) - 1))
        col += len(metrics)
    lines.append(" ".join(cmidrules))

    metric_label = {"accuracy": "Acc.", "precision": "Prec.", "recall": "Rec.",
                    "fid_mnist": "FID", "kid_mnist_mean": "KID"}
    header_bottom = [""]
    for _ in methods:
        header_bottom.extend(metric_label.get(m, m) for m in metrics)
    lines.append(" & ".join(header_bottom) + r" \\")
    lines.append(r"\midrule")

    for strength in strengths:
        row = [f"{strength:g}"]
        for method in methods:
            sub = aggregate_df[(aggregate_df["guidance_strength"] == strength)
                               & (aggregate_df["method"] == method)]
            if sub.empty:
                row.extend(["--"] * len(metrics))
                continue
            sub = sub.iloc[0]
            for m in metrics:
                val = sub.get(m)
                row.append(f"{val:.3f}" if pd.notna(val) else "--")
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Aggregate metrics across classifier guidance strengths.}",
              r"\label{tab:guidance-sweep}", r"\end{table}"]

    out_path.write_text("\n".join(lines))


def plot_metric_lines(per_class_df: pd.DataFrame, aggregate_df: pd.DataFrame,
                      metric: str, out_path_base: Path, methods=METHODS,
                      log_scale=False, share_y=True):
    if metric not in per_class_df.columns:
        print(f"warning: metric '{metric}' not in per-class data, skipping line panel")
        return

    avail_methods = [m for m in methods if m in per_class_df["method"].unique()]
    if not avail_methods:
        avail_methods = sorted(per_class_df["method"].unique())

    classes = sorted(per_class_df["class"].unique())
    metric_label = METRIC_LABELS.get(metric, metric)

    ncol = 5
    nrow_classes = int(np.ceil(len(classes) / ncol))
    nrow = nrow_classes + 1  

    fig = plt.figure(figsize=(7.0, 1.55 * nrow + 0.6), constrained_layout=True)
    gs = gridspec.GridSpec(
        nrow, ncol, figure=fig,
        height_ratios=[1.0] * nrow_classes + [1.25],
    )

    vals_all = per_class_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if share_y and len(vals_all):
        lo = float(np.nanpercentile(vals_all, 1))
        hi = float(np.nanpercentile(vals_all, 99))
        pad = 0.06 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        shared_ylim = (lo - pad, hi + pad)
    else:
        shared_ylim = None

    for idx, cls in enumerate(classes):
        r, c = divmod(idx, ncol)
        ax = fig.add_subplot(gs[r, c])
        for method in avail_methods:
            sub = per_class_df[
                (per_class_df["method"] == method)
                & (per_class_df["class"] == cls)
            ].sort_values("guidance_strength")
            if sub.empty:
                continue
            ax.plot(
                sub["guidance_strength"], sub[metric],
                marker=METHOD_MARKERS.get(method, "o"), markersize=3.5,
                linewidth=1.3, color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_title(f"Digit: {cls}", fontsize=9, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
        if log_scale:
            ax.set_yscale("log")
        if shared_ylim and not log_scale:
            ax.set_ylim(*shared_ylim)
        if c != 0:
            ax.set_yticklabels([])
        if r != nrow_classes - 1:
            ax.set_xticklabels([])

    ax_overall = fig.add_subplot(gs[nrow_classes, :])
    for method in avail_methods:
        sub = aggregate_df[aggregate_df["method"] == method].sort_values("guidance_strength")
        if sub.empty or metric not in sub.columns:
            continue
        ax_overall.plot(
            sub["guidance_strength"], sub[metric],
            marker=METHOD_MARKERS.get(method, "o"), markersize=4.5,
            linewidth=1.7, color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    ax_overall.set_title("Overall", fontsize=9.5, pad=3)
    ax_overall.spines[["top", "right"]].set_visible(False)
    ax_overall.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_overall.set_axisbelow(True)
    ax_overall.tick_params(labelsize=8)
    if log_scale:
        ax_overall.set_yscale("log")

    ylabel = metric_label + (r"  ($\downarrow$)" if metric in LOWER_IS_BETTER else "")
    fig.supxlabel("$g$", fontsize=10)
    fig.supylabel(ylabel, fontsize=10)

    handles, labels = ax_overall.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(avail_methods),
               frameon=False, fontsize=9)

    fig.savefig(str(out_path_base) + "_lines.pdf")
    fig.savefig(str(out_path_base) + "_lines.png")
    plt.close(fig)


def _macro_aggregate(per_class_df, metric, strength, method, exclude=()):
    sub = per_class_df[
        (per_class_df["guidance_strength"] == strength)
        & (per_class_df["method"] == method)
    ]
    if exclude:
        sub = sub[~sub["class"].isin(list(exclude))]
    vals = sub[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan, 0
    return float(vals.mean()), int(sub["class"].nunique())

 
def plot_metric_panel(per_class_df: pd.DataFrame, aggregate_df: pd.DataFrame,
                      metric: str, out_path_base, methods=None,
                      log_scale=False, exclude_classes=(1,),
                      overall_mode="macro"):

    if methods is None:
        methods = METHODS
 
    if metric not in per_class_df.columns:
        print(f"warning: metric '{metric}' not in per-class data, skipping panel")
        return
 
    strengths = sorted(per_class_df["guidance_strength"].unique())
    avail_methods = [m for m in methods if m in per_class_df["method"].unique()]
    if not avail_methods:
        avail_methods = sorted(per_class_df["method"].unique())
 
    classes = sorted(per_class_df["class"].unique())
    exclude = [c for c in exclude_classes if c in classes]
    if exclude_classes and not exclude:
        print(f"warning: none of exclude_classes={list(exclude_classes)} are "
              f"present; falling back to a single aggregate bar")
        overall_mode = "pooled"
 
    excl_str = ", ".join(str(c) for c in sorted(exclude))
 
    if overall_mode == "pooled" or not exclude:
        groups = [("pooled", "Overall")]
    elif overall_mode == "both":
        groups = [("pooled", "Pooled\n(ref.)"),
                  ("macro_all", "All"),
                  ("macro_excl", f"Excl. {excl_str}")]
    else:  # 'macro'
        groups = [("macro_all", "All"),
                  ("macro_excl", f"Excl. {excl_str}")]
 
    x = np.arange(len(classes))
    width = 0.8 / len(avail_methods)
    metric_label = METRIC_LABELS.get(metric, metric)
 
    n_rows = len(strengths)
    fig = plt.figure(figsize=(7.0 + 0.4 * (len(groups) - 1),
                              0.7 * n_rows + 0.3),
                     constrained_layout=True)
    outer = gridspec.GridSpec(
        n_rows, 2, figure=fig,
        width_ratios=[len(classes), 1.4 * len(groups)], wspace=0.06,
    )
 
    per_class_vals = per_class_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(per_class_vals):
        ymax = float(np.nanpercentile(per_class_vals, 99)) * 1.08
        ymin = 0 if per_class_vals.min() >= 0 else float(np.nanpercentile(per_class_vals, 1)) * 1.08
    else:
        ymin, ymax = 0, 1
 
    gx = np.arange(len(groups))
    n_class_counts = {}          
    first_ax = None
 
    for row_i, strength in enumerate(strengths):
        ax = fig.add_subplot(outer[row_i, 0])
        ax_o = fig.add_subplot(outer[row_i, 1])
        if first_ax is None:
            first_ax = ax
 
        for i, method in enumerate(avail_methods):
            sub = per_class_df[
                (per_class_df["guidance_strength"] == strength)
                & (per_class_df["method"] == method)
            ].set_index("class")
            vals = [sub.loc[c, metric] if c in sub.index else np.nan for c in classes]
 
            offset = (i - (len(avail_methods) - 1) / 2) * width
            color = METHOD_COLORS.get(method)
            ax.bar(x + offset, vals, width=width,
                   label=METHOD_LABELS.get(method, method),
                   color=color, edgecolor="black", linewidth=0.4)
 
            for gi, (kind, _) in enumerate(groups):
                if kind == "pooled":
                    agg_sub = aggregate_df[
                        (aggregate_df["guidance_strength"] == strength)
                        & (aggregate_df["method"] == method)
                    ]
                    val = (agg_sub.iloc[0][metric]
                           if not agg_sub.empty and metric in agg_sub else np.nan)
                    hatch, alpha = "///", 0.55
                elif kind == "macro_all":
                    val, n_c = _macro_aggregate(
                        per_class_df, metric, strength, method)
                    n_class_counts.setdefault(kind, set()).add(n_c)
                    hatch, alpha = None, 1.0
                else:  # macro_excl
                    val, n_c = _macro_aggregate(
                        per_class_df, metric, strength, method, exclude=exclude)
                    n_class_counts.setdefault(kind, set()).add(n_c)
                    hatch, alpha = "..", 1.0
 
                ax_o.bar(gx[gi] + offset, val, width=width,
                         color=color, edgecolor="black", linewidth=0.4,
                         hatch=hatch, alpha=alpha)

        for a in (ax, ax_o):
            if log_scale:
                a.set_yscale("log")
                a.yaxis.set_major_locator(LogLocator(base=10))
                #a.yaxis.set_major_formatter(ScalarFormatter())
                a.yaxis.set_minor_formatter(NullFormatter())
                #a.ticklabel_format(axis="y", style="plain")
            a.spines[["top", "right"]].set_visible(False)
            a.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
            a.set_axisbelow(True)
            a.tick_params(labelsize=6)
        if not log_scale:
            ax.set_ylim(ymin, ymax)
            ax_o.set_ylim(ymin, ymax)
 
        ax.set_ylabel(f"$g = {strength:g}$", fontsize=7.5)
        ax_o.set_yticklabels([])
        ax_o.set_xticks(gx)
        ax_o.set_xticklabels([lbl for _, lbl in groups] if row_i == n_rows - 1 else [],
                             fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes] if row_i == n_rows - 1 else [])

        if len(groups) > 1:
            ax_o.axvline(-0.5, color="0.8", linewidth=0.8)
 
    ylabel = metric_label + (r"  ($\downarrow$)" if metric in LOWER_IS_BETTER else "")
    fig.supylabel(ylabel, fontsize=10)
 
    for kind, counts in n_class_counts.items():
        if len(counts) > 1:
            print(f"warning: '{kind}' macro-averages for '{metric}' cover "
                  f"differing class counts {sorted(counts)} across cells -- "
                  f"some per-class results are missing and the aggregate bars "
                  f"are not comparable")
 
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=len(avail_methods), frameon=False, fontsize=7.5)
 
    suffix = "" if overall_mode == "pooled" or not exclude else \
        "_excl" + "".join(str(c) for c in sorted(exclude))
    fig.savefig(str(out_path_base) + suffix + ".pdf")
    fig.savefig(str(out_path_base) + suffix + ".png")
    plt.close(fig)


def plot_per_class_heatmap(per_class_df: pd.DataFrame, method: str, metric: str, out_path: Path):
    sub = per_class_df[per_class_df["method"] == method]
    if sub.empty or metric not in sub.columns:
        print(f"warning: no data for method='{method}', metric='{metric}', skipping heatmap")
        return

    pivot = sub.pivot(index="class", columns="guidance_strength", values=metric).sort_index()

    fig, ax = plt.subplots(figsize=(0.6 * len(pivot.columns) + 2, 4.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("classifier guidance strength")
    ax.set_ylabel("digit class")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {METHOD_LABELS.get(method, method)}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _to_display_image(img: np.ndarray) -> np.ndarray:
    if img.min() < -0.05:
        img = (img + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    if img.shape[0] == 1:
        return img[0]
    return np.transpose(img, (1, 2, 0))


def load_class_samples(pt_path: Path, n_samples: int):
    if not _HAS_TORCH:
        return None
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"warning: failed to load {pt_path} ({e})")
        return None

    samples = data.get("samples") if isinstance(data, dict) else data
    if samples is None:
        return None

    samples = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    n = min(n_samples, samples.shape[0])
    return [_to_display_image(samples[i]) for i in range(n)]


def find_class_files(method_dir: Path):
    out = {}
    if not method_dir.is_dir():
        return out
    for f in method_dir.glob("class_*.pt"):
        m = CLASS_FILE_RE.match(f.name)
        if m:
            out[int(m.group("label"))] = f
    return out


def plot_sample_grid_for_strength(strength_dir: Path, out_path: Path,
                                  classes, n_samples: int, methods=METHODS):

    if not _HAS_TORCH:
        print("warning: torch not available, skipping sample grids")
        return False

    available_methods = [m for m in methods if find_class_files(strength_dir / m)]
    if not available_methods:
        print(f"warning: no method subfolders with class_*.pt found under {strength_dir}, skipping")
        return False

    n_rows = len(classes)
    n_meth = len(available_methods)

    gutter = 0.5 
    width_ratios = []
    for mi in range(n_meth):
        width_ratios += [1.0] * n_samples
        if mi < n_meth - 1:
            width_ratios.append(gutter)
    n_cols = len(width_ratios)

    fig = plt.figure(figsize=(max(0.62 * sum(width_ratios), 4),
                              max(0.62 * n_rows + 0.4, 3)))
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.06, hspace=0.06, width_ratios=width_ratios,
    )

    col_index, c = [], 0
    for mi in range(n_meth):
        col_index.append((mi, c))
        c += n_samples
        if mi < n_meth - 1:
            c += 1 

    for row_i, cls in enumerate(classes):
        for (method_i, start_col) in col_index:
            method = available_methods[method_i]
            files = find_class_files(strength_dir / method)
            imgs = load_class_samples(files[cls], n_samples) if cls in files else None
            for k in range(n_samples):
                ax = fig.add_subplot(gs[row_i, start_col + k])
                if imgs is not None and k < len(imgs):
                    ax.imshow(imgs[k], cmap="gray" if imgs[k].ndim == 2 else None,
                              interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.4)
                    spine.set_color("0.75")
                #if method_i == 0 and k == 0:
                #    ax.set_ylabel(str(cls), rotation=0, labelpad=12,
                #                  va="center", ha="right", fontsize=9)
                if row_i == 0 and k == 0:
                    ax.annotate(
                        METHOD_LABELS.get(method, method),
                        xy=(0.5 + (n_samples - 1) / 2.0, 1.15),
                        xycoords=ax.transAxes, ha="center", va="bottom",
                        fontsize=10, annotation_clip=False,
                    )

    #fig.suptitle(f"$g = {parse_guidance_strength(strength_dir.name):g}$",
    #             fontsize=11, y=0.9)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, required=True,
                   help="Root directory containing one subfolder per guidance strength, e.g. runs/exp")
    p.add_argument("--out-dir", type=str, default="analysis_out")
    p.add_argument("--panel-metrics", nargs="+", default=DEFAULT_PANEL_METRICS,
                   help="Metrics to render as guidance-strength panels")
    p.add_argument("--panel-style", choices=["lines", "bars", "both"], default="lines",
                   help="Line panels (recommended), bar panels, or both")
    p.add_argument("--panel-log", action="store_true",
                   help="Use log y-scale on panels (off by default; log bars distort length)")
    p.add_argument("--heatmap-metric", type=str, default="fid_mnist",
                   help="Per-class metric to render as a class x guidance-strength heatmap")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(10)),
                   help="Classes to include in the qualitative sample grids")
    p.add_argument("--samples-per-class-plot", type=int, default=5,
                   help="How many sample images to show per class per method in the grids")
    p.add_argument("--skip-samples", action="store_true",
                   help="Skip qualitative sample grids (e.g. if class_*.pt files aren't available)")
    p.add_argument("--exclude-classes", type=int, nargs="+", default=[1],
                   help="Classes dropped from the second aggregate bar "
                        "group in the bar panels. Pass none to disable.")
    p.add_argument("--overall-mode", choices=["macro", "both", "pooled"],
                   default="macro",
                   help="'macro': two comparable macro-average aggregate "
                        "bars (all vs excluded). 'both': also show the "
                        "pooled JSON aggregate, hatched, as a reference. "
                        "'pooled': original single-bar behaviour.")

    args = p.parse_args()

    set_paper_style()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_df, aggregate_df = load_results(root)

    per_class_df.to_csv(out_dir / "tidy_per_class_results.csv", index=False)
    aggregate_df.to_csv(out_dir / "tidy_aggregate_results.csv", index=False)
    print(f"wrote tidy CSVs to {out_dir}")

    write_summary_latex(aggregate_df, out_dir / "summary_table.tex")
    print(f"wrote {out_dir / 'summary_table.tex'}")

    for metric in args.panel_metrics:
        if args.panel_style in ("lines", "both"):
            plot_metric_lines(per_class_df, aggregate_df, metric,
                              out_dir / f"panel_{metric}", log_scale=args.panel_log)
        if args.panel_style in ("bars", "both"):
            plot_metric_panel(per_class_df, aggregate_df, metric,
                out_dir / f"panel_{metric}", log_scale=True,
                exclude_classes=tuple(args.exclude_classes),
                overall_mode=args.overall_mode)
    print(f"wrote panel figures for: {args.panel_metrics} (style={args.panel_style})")

    for method in sorted(per_class_df["method"].unique()):
        plot_per_class_heatmap(
            per_class_df, method, args.heatmap_metric,
            out_dir / f"per_class_heatmap_{args.heatmap_metric}_{method}.png",
        )
    print(f"wrote per-class heatmaps for metric: {args.heatmap_metric}")

    if not args.skip_samples:
        if not _HAS_TORCH:
            print("warning: torch is not installed; skipping sample grids "
                  "(install torch or pass --skip-samples to silence this)")
        else:
            for strength_dir in strength_dirs_sorted(root):
                strength = parse_guidance_strength(strength_dir.name)
                out_path = out_dir / f"samples_gs{strength:g}.pdf"
                ok = plot_sample_grid_for_strength(
                    strength_dir, out_path, args.classes,
                    args.samples_per_class_plot,
                )
                if ok:
                    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
