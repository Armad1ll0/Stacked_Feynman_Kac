"""
replot_gmm.py
-------------
Reload results saved by run_gmm_comparison.py ({tag}_raw.pt) and rebuild
the summary table and/or the figure -- optionally with a different
sigma_y order, a different method order/subset, or a different
--fig_max_points -- WITHOUT rerunning any sampling.

Usage:
    # exactly reproduce the original figure/table from saved data
    python replot_gmm.py gmm_results/main_raw.pt

    # re-order sigma_y rows (e.g. descending instead of ascending)
    python replot_gmm.py gmm_results/main_raw.pt --sigma_ys 3.0 1.0 0.5 0.1

    # only show a subset of methods, in a specific order
    python replot_gmm.py gmm_results/main_raw.pt --methods tds_hmc tds pc

    # denser scatter points, different output filename
    python replot_gmm.py gmm_results/main_raw.pt --fig_max_points 8000 \
        --output figs/gmm_final.png

    # just reprint the markdown table, skip the figure
    python replot_gmm.py gmm_results/main_raw.pt --table_only
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_publication_style(usetex: bool = True) -> None:
    """Apply the same NeurIPS/TMLR-style font setup as make_comparison.py."""
    plt.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{mathptmx}",  # Times-style text + math
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

set_publication_style(usetex=True)

# reuse the exact plotting logic from the main driver so the figure is
# pixel-for-pixel the same as a fresh run would produce

DISPLAY_NAMES = {
    "tds": "TDS",
    "tds_hmc": "rSFK-uHMC",
    "dps": "DPS",
    "pc": "PC",
    "fps": "FPS",
    "mcgdiff": "MCGDiff",
}

def _draw_contour(ax, GX, GY, dens, args, alpha_lines=0.6, alpha_fill=0.25):
    """Draw the posterior density contour on ax, honoring the configurable
    style/cmap/levels on args. Falls back to the original look (black
    lines + light grey fill) if these attributes aren't set.
 
    Options (all read off `args`):
      contour_style       "lines" | "filled" | "both"   (default "both")
      contour_cmap        matplotlib colormap name       (default "Greys")
      contour_levels      number of contour bands        (default 6)
      contour_line_color  outline color                  (default "k")
    """
    style = getattr(args, "contour_style", "both")
    cmap = getattr(args, "contour_cmap", "viridis")
    levels = getattr(args, "contour_levels", 15)
    line_color = getattr(args, "contour_line_color", "k")
 
    if style in ("lines", "both"):
        ax.contour(GX, GY, dens, levels=levels, colors=line_color,
                   alpha=alpha_lines, linewidths=.8)
    if style in ("filled", "both"):
        ax.contourf(GX, GY, dens, levels=levels, cmap=cmap, alpha=alpha_fill)
 
 
def build_figure(results, posts, sigma_ys, plot_methods, seeds, args, fname):
    """One ROW per sigma_y, ground-truth column + one COLUMN per method.
 
    Font sizes and contour styling are all pulled from `args` via
    getattr(..., default) -- see the CLI flags in run_gmm_comparison.py
    and replot_gmm.py for the full list:
 
      suptitle_fontsize      figure-level title            (default 13)
      col_label_fontsize     method name / "Ground truth"  (default 13)
      row_label_fontsize     sigma_y row label              (default 12)
      title_fontsize         per-panel SW/modeErr title     (default 9)
      failed_label_fontsize  "failed" placeholder text      (default 11)
      fig_max_points         subsample cap per scatter panel
 
    To change the overall figure LAYOUT (panel size, spacing, dpi,
    scatter point size/color/alpha, etc.), edit the constants directly
    in this function.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    if not plot_methods:
        print("\nNo sampler rows to plot; skipping figure.")
        return
 
    nrow = len(sigma_ys)
    ncol = 1 + len(plot_methods)   # +1 for the ground-truth-only column
    lim = args.spacing * (args.grid_side - 1) / 2 * 1.3
    xs = torch.linspace(-lim, lim, 200)
    GX, GY = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([GX.reshape(-1), GY.reshape(-1)], 1)
 
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.6 * ncol, 3.9 * nrow),
                             squeeze=False)
 
    for i, sigma_y in enumerate(sigma_ys):
        post = posts[sigma_y]
        dens = post.log_prob(pts).exp().reshape(200, 200)
        rows = results[sigma_y]
 
        # ── column 0: ground-truth posterior contour, no samples ──
        ax_gt = axes[i][0]
        _draw_contour(ax_gt, GX, GY, dens, args, alpha_lines=.6, alpha_fill=.25)
        ax_gt.set_xlim(-lim, lim)
        ax_gt.set_ylim(-lim, lim)
        ax_gt.set_aspect("equal")
        ax_gt.grid(alpha=.2)
        ax_gt.set_xticklabels([])
        ax_gt.set_yticklabels([])
        ax_gt.set_ylabel(f"$\\sigma_y$ = {sigma_y}",
                         fontsize=getattr(args, "row_label_fontsize", 16))
        if i == 0:
            ax_gt.annotate("GT", xy=(0.5, 1.05),
                           xycoords="axes fraction",
                           ha="center", va="bottom",
                           fontsize=getattr(args, "col_label_fontsize", 20))
 
        # ── remaining columns: one per method ──
        for j, name in enumerate(plot_methods, start=1):
            ax = axes[i][j]
            _draw_contour(ax, GX, GY, dens, args, alpha_lines=.25, alpha_fill=.12)
 
            entry = rows.get(name)
            if entry is None:
                ax.text(0.5, 0.5, "failed", ha="center", va="center",
                        transform=ax.transAxes, color="tab:red",
                        fontsize=getattr(args, "failed_label_fontsize", 11))
            else:
                S = entry["plot_samples"]
                S = S.detach().cpu() if torch.is_tensor(S) else torch.as_tensor(S)
                if S.shape[0] > args.fig_max_points:
                    sel = torch.randperm(S.shape[0])[:args.fig_max_points]
                    S = S[sel]
                agg = entry["agg"]
                sw_m, sw_s = agg["sw"]
                me_m, me_s = agg["mode_cov_err"]
                ax.scatter(S[:, 0], S[:, 1], s=4, alpha=.35, c="tab:blue")
                #ax.set_title(
                #    f"SW={sw_m:.3f}$\\pm${sw_s:.3f}\n"
                #    f"modeErr={me_m:.3f}$\\pm${me_s:.3f}",
                #    fontsize=getattr(args, "title_fontsize", 9))
 
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(alpha=.2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
 
            # method name across the top
            if i == 0:
                ax.annotate(DISPLAY_NAMES.get(name, name), xy=(0.5, 1.05), xycoords="axes fraction",
                            ha="center", va="bottom",
                            fontsize=getattr(args, "col_label_fontsize", 20))
                
    n_modes = args.grid_side ** 2
    #fig.suptitle(
    #    f"Posterior sampling, analytic GMM ({n_modes} modes, "
    #    f"spacing={args.spacing}, y={args.y})\n"
    #    f"averaged over {len(seeds)} seed(s): {seeds}",
    #    fontsize=getattr(args, "suptitle_fontsize", 13))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname, dpi=100)
    print(f"\nWrote {fname}")


def print_summary_table(results, sigma_ys):
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
            cells.append(f"{r['agg']['sw'][0]:.3f}+-{r['agg']['sw'][1]:.3f}"
                         if r else "--")
        print(f"| {name} | " + " | ".join(cells) + " |")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_path", help="path to a {tag}_raw.pt file written "
                                     "by run_gmm_comparison.py")
    ap.add_argument("--sigma_ys", type=float, nargs="*", default=None,
                    help="reorder/subset the sigma_y rows shown (default: "
                         "the order they were saved in)")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="reorder/subset the method columns shown "
                         "(default: the order they were saved in)")
    ap.add_argument("--fig_max_points", type=int, default=None,
                    help="override the saved subsample cap per panel")
    ap.add_argument("--contour_style", choices=["lines", "filled", "both"],
                    default="filled",
                    help="override the saved contour style: outline only, "
                         "filled only, or both.")
    ap.add_argument("--contour_cmap", default=None,
                    help="override the saved colormap for the filled "
                         "contour, e.g. viridis, Blues, magma, coolwarm.")
    ap.add_argument("--contour_levels", type=int, default=None,
                    help="override the saved number of contour levels.")
    ap.add_argument("--contour_line_color", default=None,
                    help="override the saved contour outline color.")
    ap.add_argument("--suptitle_fontsize", type=float, default=None,
                    help="override the saved figure-level title font size.")
    ap.add_argument("--col_label_fontsize", type=float, default=None,
                    help="override the saved method-name/'Ground truth' "
                         "column header font size.")
    ap.add_argument("--row_label_fontsize", type=float, default=None,
                    help="override the saved sigma_y row label font size.")
    ap.add_argument("--title_fontsize", type=float, default=None,
                    help="override the saved per-panel metric title font size.")
    ap.add_argument("--failed_label_fontsize", type=float, default=None,
                    help="override the saved 'failed' placeholder font size.")
    ap.add_argument("--output", default=None,
                    help="output figure path (default: derived from the "
                         "saved tag, same convention as the original run)")
    ap.add_argument("--table_only", action="store_true",
                    help="print the markdown tables and skip the figure")
    args = ap.parse_args()
 
    blob = torch.load(args.raw_path, weights_only=False)
    results = blob["results"]
    posts = blob["posts"]
    saved_sigma_ys = blob["sigma_ys"]
    saved_methods = blob["methods"]
    seeds = blob["seeds"]
    saved_args = SimpleNamespace(**blob["args"])
 
    sigma_ys = args.sigma_ys if args.sigma_ys is not None else saved_sigma_ys
    missing_sigma = [s for s in sigma_ys if s not in results]
    if missing_sigma:
        raise SystemExit(f"sigma_y value(s) not found in saved results: "
                         f"{missing_sigma}. Available: {list(results.keys())}")
 
    methods = args.methods if args.methods is not None else saved_methods
    plot_methods = [m for m in methods if any(m in results[s] for s in sigma_ys)]
    unknown = [m for m in methods if m not in plot_methods]
    if unknown:
        print(f"Note: method(s) with no data in the selected sigma_y "
              f"range, skipped: {unknown}")
 
    print(f"Loaded {args.raw_path}")
    print(f"sigma_y rows: {sigma_ys}")
    print(f"methods (columns): {plot_methods}")
    print(f"seeds averaged in saved data: {seeds}")
 
    print_summary_table(results, sigma_ys)
 
    if args.table_only:
        return
 
    if args.fig_max_points is not None:
        saved_args.fig_max_points = args.fig_max_points
    if args.contour_style is not None:
        saved_args.contour_style = args.contour_style
    if args.contour_cmap is not None:
        saved_args.contour_cmap = args.contour_cmap
    if args.contour_levels is not None:
        saved_args.contour_levels = args.contour_levels
    if args.contour_line_color is not None:
        saved_args.contour_line_color = args.contour_line_color
    if args.suptitle_fontsize is not None:
        saved_args.suptitle_fontsize = args.suptitle_fontsize
    if args.col_label_fontsize is not None:
        saved_args.col_label_fontsize = args.col_label_fontsize
    if args.row_label_fontsize is not None:
        saved_args.row_label_fontsize = args.row_label_fontsize
    if args.title_fontsize is not None:
        saved_args.title_fontsize = args.title_fontsize
    if args.failed_label_fontsize is not None:
        saved_args.failed_label_fontsize = args.failed_label_fontsize
 
    fname = args.output or (
        f"gmm_overlay{('_' + saved_args.tag) if saved_args.tag else ''}_replot.pdf"
    )
    build_figure(results, posts, sigma_ys, plot_methods, seeds, saved_args, fname)
 
 
if __name__ == "__main__":
    main()

# python -m analysis.gmm.plot_results_gmm gmm_results/main_raw.pt