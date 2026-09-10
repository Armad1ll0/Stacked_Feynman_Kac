"""
experiments/plot_ess.py
-----------------------
Plot effective-sample-size trajectories for the SMC-based samplers.

Reads the per-image metrics JSONs written alongside each run:

    experiments/results/{dataset}_{task}_{method}/00_metrics.json
    experiments/results/{dataset}_{task}_{method}/01_metrics.json
    ...

Several results directories are scanned, since the TDS matrix and the
baselines are written separately -- FPS and MCGDiff live under
experiments/results_baseline. Both defaults are searched; override with
--results-dir DIR [DIR ...].

Each file is expected to contain an "ess_trace" entry: a list of ESS
values, one per diffusion step. DPS and PC are not particle methods and
have no weights, so they are skipped automatically (this matches the NaN
ess_mean/ess_min/ess_final columns for those rows in summary_noisy.csv).

Output is one panel per (dataset, task), each showing the MEDIAN ESS
across images with an interquartile band, one line per method. The median
plus IQR is used rather than per-image spaghetti or a mean: ESS
trajectories are strongly non-Gaussian near collapse (a floor at 1 and a
ceiling at N), so a mean is pulled around by a few images and hides the
degeneracy point.

Usage:
    # every (dataset, task) found across both results folders
    python -m experiments.plot_ess

    # non-default folder layout
    python -m experiments.plot_ess --results-dir path/to/tds path/to/baselines

    # single panel, specific methods, saved as pdf
    python -m experiments.plot_ess --dataset butterflies --task deblur \\
        --methods tds tds_hmc_refined --output figs/ess_butterflies_deblur.pdf

    # one multi-panel grid instead of separate files
    python -m experiments.plot_ess --grid --output figs/ess_all.pdf

    # raw ESS on the y axis instead of ESS/N
    python -m experiments.plot_ess --raw

NORMALISATION
    ESS is only comparable across methods once divided by the particle
    count, since different samplers may have been run with different N.
    By default the script normalises to ESS/N and infers N from, in order:
      1. an explicit --n-particles value
      2. an "n_particles" / "num_particles" / "n_samples" key in the JSON
      3. the maximum value observed in that run's traces, rounded up
    A warning is printed whenever it falls back to (3), because that is a
    guess: it is only right if ESS reaches (near) N at some point, which
    is usually true at t=T where the weights start uniform.

TIME AXIS
    The x axis is the index into ess_trace. Whether index 0 is t=T (start
    of sampling, weights uniform) or t=0 (end) depends on the sampler
    loop, and getting it backwards silently mirrors every curve. The
    script guesses from the data -- ESS almost always starts high and
    decays -- and prints what it decided. Override with --time-order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Particle-based samplers only. DPS and PC carry no importance weights.
SMC_METHODS = ["tds", "tds_hmc_refined", "fps", "mcgdiff"]
NON_SMC_METHODS = ["dps", "pc", "pinv"]

# Used to split "{dataset}_{task}_{method}" unambiguously -- both
# "inpaint_random" and "tds_hmc_refined" contain underscores, so the
# directory name cannot simply be split on "_".
KNOWN_DATASETS = ["flowers", "celeba", "mnist", "butterflies"]
KNOWN_METHODS = SMC_METHODS + NON_SMC_METHODS

# Canonical presentation order. This drives panel layout, so changing it
# reorders every figure -- keep it in sync with the table order in the paper.
DATASET_ORDER = ["mnist", "flowers", "butterflies", "celeba"]
TASK_ORDER = ["inpaint", "inpaint_random", "superres", "deblur"]

# Baselines first, ours last, so our curve is drawn on top of the others
# and sits at the end of the legend.
METHOD_ORDER = ["tds", "fps", "mcgdiff", "tds_hmc_refined"]

# Directory-name matching uses the same set; one list, so adding a dataset
# to DATASET_ORDER is enough to make it both parseable and ordered.
KNOWN_DATASETS = list(DATASET_ORDER)

# Pretty names for figure titles and legends. Edit these freely -- keys
# are the raw folder tokens, values are what appears in the figure.
# Anything missing falls back to prettify(): underscores become spaces and
# the first word is capitalised, so a new dataset or task still renders
# sensibly without an entry here.
DATASET_NAMES = {
    "celeba": "CelebA",
    "mnist": "MNIST",
    "flowers": "Flowers",
    "butterflies": "Butterflies",
}

TASK_NAMES = {
    "inpaint": "Center Mask",
    "inpaint_random": "Random Mask",
    "superres": "Super-resolution",
    "deblur": "Deblurring",
}

DISPLAY_NAMES = {
    "tds": "TDS",
    "tds_hmc_refined": "rSFK-uHMC",
    "fps": "FPS",
    "mcgdiff": "MCGDiff",
}

COLORS = {
    "tds": "tab:blue",
    "tds_hmc_refined": "tab:red",
    "fps": "tab:green",
    "mcgdiff": "tab:orange",
}

N_PARTICLE_KEYS = ["n_particles", "num_particles", "n_samples",
                   "num_samples", "n_particle", "K"]

ESS_KEYS = ["ess_trace", "ess_history", "ess", "ess_per_step"]


def prettify(name, mapping=None):
    """Look up a display name, else de-underscore and capitalise."""
    if mapping and name in mapping:
        return mapping[name]
    return name.replace("_", " ").capitalize()


def panel_title(dataset, task, shared_n=None):
    t = f"{prettify(dataset, DATASET_NAMES)} / {prettify(task, TASK_NAMES)}"
    return t


def set_publication_style(usetex: bool = False) -> None:
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

def _rank(value, order):
    """Sort key: listed items in the given order, unknown ones after,
    alphabetically. Means a new dataset or task still plots -- it just
    lands at the end until it is added to the order list."""
    if value in order:
        return (0, order.index(value), "")
    return (1, 0, value)


def order_panels(panels, dataset_order, task_order):
    """Sort (dataset, task) panels; task varies fastest."""
    return OrderedDict(
        sorted(panels.items(),
               key=lambda kv: (_rank(kv[0][0], dataset_order),
                               _rank(kv[0][1], task_order))))


# ─────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────

def parse_run_dir(name: str):
    """'butterflies_deblur_tds_hmc_refined' -> ('butterflies','deblur','tds_hmc_refined').

    Matched against known dataset prefixes and method suffixes so that
    multi-word tasks and methods survive. Returns None if unrecognised.
    """
    ds = next((d for d in KNOWN_DATASETS if name.startswith(d + "_")), None)
    if ds is None:
        return None
    rest = name[len(ds) + 1:]
    # longest method suffix first, so "tds_hmc_refined" beats "tds"
    for m in sorted(KNOWN_METHODS, key=len, reverse=True):
        if rest.endswith("_" + m):
            return ds, rest[:-(len(m) + 1)], m
    return None


def find_runs(results_dirs):
    """-> OrderedDict[(dataset, task, method)] = directory Path.

    Accepts several directories, since the TDS matrix and the baselines
    (FPS / MCGDiff) are written to separate results folders. Later
    directories do not overwrite earlier ones: if the same
    (dataset, task, method) appears twice, the first is kept and a warning
    is printed, because silently picking one of two runs is exactly the
    kind of thing that produces an unreproducible figure.
    """
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]

    runs = OrderedDict()
    origin = {}
    missing = []
    for rd in results_dirs:
        rd = Path(rd)
        if not rd.is_dir():
            missing.append(rd)
            continue
        for d in sorted(rd.iterdir()):
            if not d.is_dir():
                continue
            parsed = parse_run_dir(d.name)
            if parsed is None:
                print(f"  ? skipping unrecognised directory: {d}", file=sys.stderr)
                continue
            if parsed in runs:
                print(f"  ! duplicate run {'/'.join(parsed)}: keeping "
                      f"{origin[parsed]}, ignoring {d}", file=sys.stderr)
                continue
            runs[parsed] = d
            origin[parsed] = d

    if missing:
        print("  ! results directory not found: "
              + ", ".join(str(m) for m in missing), file=sys.stderr)
    if not runs:
        raise SystemExit(
            "No run directories found in: "
            + ", ".join(str(Path(r)) for r in results_dirs))
    return runs


def _first_key(blob, keys):
    for k in keys:
        if k in blob:
            return blob[k]
    return None


def load_traces(run_dir: Path):
    """Read every NN_metrics.json in run_dir.

    Returns (traces, n_particles_declared) where traces is a list of lists.
    """
    files = sorted(run_dir.glob("*_metrics.json"))
    traces, declared = [], None
    for f in files:
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  ! {f} is not valid JSON ({e}); skipping.", file=sys.stderr)
            continue

        tr = _first_key(blob, ESS_KEYS)
        if tr is None:
            continue
        # tolerate {"ess_trace": {"values": [...]}} and nested singletons
        if isinstance(tr, dict):
            tr = _first_key(tr, ["values", "trace", "data"]) or []
        if not isinstance(tr, list) or not tr:
            continue

        vals = []
        for v in tr:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            vals.append(fv)
        if all(math.isnan(v) for v in vals):
            continue
        traces.append(vals)

        if declared is None:
            n = _first_key(blob, N_PARTICLE_KEYS)
            if n is not None:
                try:
                    declared = int(n)
                except (TypeError, ValueError):
                    pass

    return traces, declared


# ─────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────

def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _quantile(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarise(traces):
    """Per-step median and interquartile band across images.

    Traces of unequal length are truncated to the shortest, with a warning
    -- interpolating would invent structure at exactly the steps where the
    curves are most informative.
    """
    if not traces:
        return None
    lens = {len(t) for t in traces}
    L = min(lens)
    if len(lens) > 1:
        print(f"  ! ess_trace lengths differ {sorted(lens)}; "
              f"truncating all to {L}.", file=sys.stderr)

    med, lo, hi = [], [], []
    for i in range(L):
        col = [t[i] for t in traces if not math.isnan(t[i])]
        if not col:
            med.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
            continue
        med.append(_median(col))
        lo.append(_quantile(col, 0.25))
        hi.append(_quantile(col, 0.75))
    return dict(median=med, q25=lo, q75=hi, n_images=len(traces), n_steps=L)


def parse_n_particles_spec(tokens):
    """Parse --n-particles into a lookup usable per run.

    Accepts a bare integer (applies to everything) or KEY=N pairs keyed by
    method, 'dataset/task', or 'dataset/task/method'. Returns a dict; look
    it up with resolve_n_override().
    """
    if not tokens:
        return {}
    spec = {}
    for tok in tokens:
        if "=" not in tok:
            try:
                spec["*"] = int(tok)
            except ValueError:
                raise SystemExit(
                    f"--n-particles: '{tok}' is neither an integer nor KEY=N")
            continue
        key, _, val = tok.partition("=")
        try:
            spec[key.strip()] = int(val)
        except ValueError:
            raise SystemExit(f"--n-particles: '{tok}' has a non-integer count")
    return spec


def resolve_n_override(spec, dataset, task, method):
    """Most specific match wins: full run > setting > method > global."""
    if not spec:
        return None
    for key in (f"{dataset}/{task}/{method}", f"{dataset}/{task}",
                dataset, method, "*"):
        if key in spec:
            return spec[key]
    return None


def infer_n_particles(traces, declared, override):
    """-> (N, source) where source is 'flag' | 'json' | 'guess'."""
    if override:
        return float(override), "flag"
    if declared:
        return float(declared), "json"
    peak = max((v for t in traces for v in t if not math.isnan(v)), default=float("nan"))
    if math.isnan(peak) or peak <= 0:
        return float("nan"), "guess"
    # already normalised?
    if peak <= 1.0 + 1e-4:
        return 1.0, "json"
    # ESS = 1/sum(w^2) hits exactly N when the weights are uniform, which
    # is where the peak sits. In float32 that lands slightly ABOVE N
    # (relative eps ~1.2e-7), so a naive ceil() rounds 8.0000012 up to 9.
    # Snap to the nearest integer when the peak is within tolerance of it,
    # and only round up when the peak genuinely falls between integers.
    nearest = round(peak)
    tol = max(1e-4, 1e-4 * peak)
    if nearest >= 1 and abs(peak - nearest) <= tol:
        return float(nearest), "guess"
    return float(math.ceil(peak)), "guess"


def guess_time_order(summary):
    """Return 'forward' if index 0 looks like the start of sampling.

    ESS starts at N with uniform weights and decays as the particles
    degenerate, so a trace whose first value exceeds its last is stored
    forward in sampling order.
    """
    med = [v for v in summary["median"] if not math.isnan(v)]
    if len(med) < 2:
        return "forward"
    head = _median(med[:max(1, len(med) // 10)])
    tail = _median(med[-max(1, len(med) // 10):])
    return "forward" if head >= tail else "reverse"


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def draw_panel(ax, data, raw, show_band=True, logy=False,
               show_counts=True, xlabel=True, ylabel=True, x_frac=False):
    Ns = {d["N"] for d in data.values()
          if d["N"] and not math.isnan(d["N"]) and d["N"] > 1}
    mixed_N = len(Ns) > 1

    for method, d in data.items():
        s, N = d["summary"], d["N"]
        scale = 1.0 if raw else (N if N and not math.isnan(N) else 1.0)
        # With x_frac the axis is progress through the trajectory rather
        # than the step index, so datasets with different step counts
        # become directly comparable (and can share an x axis).
        T = s["n_steps"]
        x = ([i / (T - 1) for i in range(T)] if (x_frac and T > 1)
             else list(range(T)))
        med = [v / scale for v in s["median"]]
        c = COLORS.get(method, None)
        # Always show N when the panel mixes particle counts: ESS/N = 0.5
        # means 4 effective particles at N=8 but 16 at N=32, and the
        # normalised curves overlap while the samplers do not.
        lab = f"{DISPLAY_NAMES.get(method, method)}"
        if show_counts:
            lab += f" ({s['n_images']} img)"
        ax.plot(x, med, label=lab, color=c, lw=1.6)
        if show_band:
            ax.fill_between(x,
                            [v / scale for v in s["q25"]],
                            [v / scale for v in s["q75"]],
                            color=c, alpha=0.18, linewidth=0)

    if xlabel:
        ax.set_xlabel("progress through trajectory" if x_frac
                      else "diffusion step (start $\\rightarrow$ end)")
    if x_frac:
        ax.set_xlim(0, 1)
    if ylabel:
        ax.set_ylabel("ESS" if raw else "ESS / N")
    if not raw:
        ax.set_ylim(0, 1.02)
        ax.axhline(1.0, color="k", lw=.6, alpha=.3)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=.25)
    # When every method in the panel used the same N -- the usual case,
    # since the particle count is set per dataset -- report it once in the
    # title rather than repeating it in every legend entry.
    return (mixed_N, (Ns.pop() if len(Ns) == 1 else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", nargs="+",
                    default=["experiments/results",
                             "experiments/results_baselines"],
                    help="one or more directories holding the "
                         "{dataset}_{task}_{method} folders. The TDS matrix "
                         "and the baselines (FPS / MCGDiff) live in separate "
                         "folders, so both are scanned by default.")
    ap.add_argument("--dataset", nargs="*", default=None,
                    help="restrict to these datasets")
    ap.add_argument("--task", nargs="*", default=None,
                    help="restrict to these tasks")
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"methods to plot (default: {' '.join(SMC_METHODS)}). "
                         "DPS/PC carry no weights and have no ESS.")
    ap.add_argument("--n-particles", nargs="+", default=None,
                    metavar="SPEC",
                    help="particle count override. Either a bare integer "
                         "applied to every run, or one or more KEY=N pairs "
                         "where KEY is a dataset ('celeba=8'), a method "
                         "('mcgdiff=32'), a setting ('celeba/superres=16'), "
                         "or a full run ('celeba/superres/tds=16'). The "
                         "particle count is typically per-dataset, so "
                         "'--n-particles celeba=8 mnist=32' is the usual "
                         "form. More specific keys "
                         "win. Only needed when n_particles is absent from "
                         "the JSONs; otherwise the value there is used.")
    ap.add_argument("--raw", action="store_true",
                    help="plot raw ESS instead of ESS/N")
    ap.add_argument("--no-band", action="store_true",
                    help="median line only, no interquartile band")
    ap.add_argument("--logy", action="store_true",
                    help="log-scale the y axis (useful when ESS collapses to ~1)")
    ap.add_argument("--time-order", choices=["auto", "forward", "reverse"],
                    default="auto",
                    help="whether ess_trace index 0 is the START of sampling "
                         "(default: auto-detect from the shape of the curve)")
    ap.add_argument("--x-frac", action="store_true",
                    help="plot progress through the trajectory (0-1) "
                         "instead of the raw step index. Use this when the "
                         "number of diffusion steps differs by dataset and "
                         "you want the panels directly comparable.")
    ap.add_argument("--share-x", action="store_true",
                    help="force a shared x axis across grid panels. Off by "
                         "default because step counts differ by dataset; "
                         "implied by --x-frac.")
    ap.add_argument("--ncol", type=int, default=4,
                    help="panels per row in --grid mode (default: 4)")
    ap.add_argument("--legend-counts", action="store_true",
                    help="include the per-method image count in the legend. "
                         "Off by default in --grid mode, where one shared "
                         "legend cannot show per-panel counts anyway.")
    ap.add_argument("--grid", action="store_true",
                    help="one multi-panel figure instead of one file per setting")
    ap.add_argument("--output", default=None,
                    help="output path. With --grid this is the single file; "
                         "otherwise it is treated as a directory.")
    ap.add_argument("--dpi", type=int, default=200,
                    help="raster dpi; ignored for vector formats (pdf/svg)")
    ap.add_argument("--usetex", action="store_true",
                    help="render text with LaTeX")
    args = ap.parse_args()

    set_publication_style(args.usetex)

    methods = args.methods or SMC_METHODS
    bad = [m for m in methods if m in NON_SMC_METHODS]
    if bad:
        print(f"  ! {', '.join(bad)} are not particle methods and have no "
              f"ESS; dropping them.", file=sys.stderr)
        methods = [m for m in methods if m not in NON_SMC_METHODS]
    if not methods:
        raise SystemExit("No SMC methods left to plot.")

    n_spec = parse_n_particles_spec(args.n_particles)
    runs = find_runs(args.results_dir)

    # (dataset, task) -> {method: {summary, N}}
    panels = OrderedDict()
    for (ds, task, method), d in runs.items():
        if method not in methods:
            continue
        if args.dataset and ds not in args.dataset:
            continue
        if args.task and task not in args.task:
            continue

        traces, declared = load_traces(d)
        if not traces:
            print(f"  ! no usable ess_trace in {d}", file=sys.stderr)
            continue

        override = resolve_n_override(n_spec, ds, task, method)
        N, src = infer_n_particles(traces, declared, override)
        if src == "guess" and not args.raw:
            print(f"  ? {ds}/{task}/{method}: particle count not recorded; "
                  f"assuming N={N:g} from the peak ESS. Pass --n-particles "
                  f"if that is wrong.", file=sys.stderr)

        s = summarise(traces)
        order = args.time_order
        if order == "auto":
            order = guess_time_order(s)
        if order == "reverse":
            for k in ("median", "q25", "q75"):
                s[k] = list(reversed(s[k]))

        panels.setdefault((ds, task), OrderedDict())
        panels[(ds, task)][method] = dict(summary=s, N=N, order=order)

    if not panels:
        raise SystemExit("Nothing to plot -- no matching runs with ess_trace.")
    
    # CLI order wins when the user names datasets/tasks explicitly:
    # `--dataset celeba mnist` should plot celeba first.
    ds_order = args.dataset or DATASET_ORDER
    tk_order = args.task or TASK_ORDER
    panels = order_panels(panels, ds_order, tk_order)

    # Method order within each panel controls both draw order (later =
    # on top) and legend order.
    m_order = args.methods or METHOD_ORDER
    for key, data in panels.items():
        panels[key] = OrderedDict(
            sorted(data.items(), key=lambda kv: _rank(kv[0], m_order)))

    orders = {d["order"] for p in panels.values() for d in p.values()}
    print(f"\nTime axis: {'/'.join(sorted(orders))}"
          + (" (MIXED -- check --time-order)" if len(orders) > 1 else ""))
    print(f"Plotting {len(panels)} setting(s): "
          + ", ".join(f"{a}/{b}" for a, b in panels))

    if args.grid:
        n = len(panels)
        ncol = min(args.ncol, n)
        nrow = math.ceil(n / ncol)
        # Step counts differ by dataset, so a shared x axis would stretch
        # the shorter runs across the longest range and leave dead space.
        # Sharing is only safe when the axis is a 0-1 progress fraction,
        # or when explicitly asked for.
        share_x = bool(args.x_frac or args.share_x)
        # ESS/N is always in [0, 1]; raw ESS tops out at N, which varies
        # by dataset, so only share y in the normalised case.
        share_y = not args.raw
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(3.7 * ncol, 3.1 * nrow),
                                 squeeze=False,
                                 sharex=share_x, sharey=share_y)
        for ax in axes.flat[n:]:
            ax.axis("off")

        handles = labels = None
        for i, (ax, ((ds, task), data)) in enumerate(
                zip(axes.flat, panels.items())):
            row, col = divmod(i, ncol)
            # Label only the outer edge: with shared axes the inner ticks
            # are identical, and repeating "ESS / N" on every panel eats
            # width that the curves could use.
            is_bottom = (i + ncol) >= n          # nothing below this panel
            # Without a shared x axis every panel carries its own ticks;
            # the axis LABEL still only goes on the bottom row, since the
            # text is identical everywhere.
            _mixed, sharedN = draw_panel(
                ax, data, args.raw, not args.no_band, args.logy,
                show_counts=args.legend_counts,
                xlabel=is_bottom, ylabel=(col == 0),
                x_frac=args.x_frac)
            ax.set_title(panel_title(ds, task, sharedN), fontsize=11)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

        # One legend for the whole figure, centred beneath the panels.
        fig.tight_layout()
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(len(labels), 4), frameon=False,
                       fontsize=9, bbox_to_anchor=(0.5, -0.02))
        out = Path(args.output or "ess_all.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {out}")
        return

    outdir = Path(args.output or "figs/ess")
    outdir.mkdir(parents=True, exist_ok=True)
    for (ds, task), data in panels.items():
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        _mixed, sharedN = draw_panel(ax, data, args.raw,
                                    not args.no_band, args.logy,
                                    x_frac=args.x_frac)
        ax.set_title(panel_title(ds, task, sharedN), fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        out = outdir / f"ess_{ds}_{task}.pdf"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()


# python -m analysis.image_restoration.plot_ess