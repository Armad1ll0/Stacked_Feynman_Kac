"""
experiments/combine_summaries.py
---------------------------------
Combine results from ALL result matrices -- TDS/TDS+HMC, the baselines
(DPS/PC/FPS/MCGdiff), and the pseudo-inverse deblurring anchor -- into
markdown tables so every method can be compared side by side.

Three tables are produced:

  1. MEANS            -- one row per (dataset, task, method), the metric
                         means. This is the original table, produced by
                         experiments.summarize.print_markdown_table, and
                         is unchanged.

  2. PAIRED DELTA     -- for one metric, the mean per-image improvement of
                         a reference method over each competitor, with a
                         95% CI and (optionally) a Wilcoxon signed-rank
                         p-value.

  3. WIN RATE         -- for the same metric, the fraction of images on
                         which the reference method beats each competitor.

Why tables 2 and 3 exist
------------------------
The marginal std of PSNR across images is dominated by IMAGE DIFFICULTY,
not by method quality: an image on which every method scores 20 dB and one
on which every method scores 30 dB inflate the std of all methods equally,
so marginal error bars overlap even when one method is uniformly better.

Because every method is evaluated on the SAME images, the design is paired
(a randomised block design, blocking on image). Forming the per-image
difference d_i = m_ref(i) - m_other(i) cancels the shared image-difficulty
term, and the SE of mean(d) is typically several times smaller than the
marginal SEs. The win rate is the distribution-free version of the same
idea, and is robust to a handful of catastrophic-failure images.

Tables 2 and 3 need PER-IMAGE metrics. summary_noisy.csv already IS
per-image -- one row per (dataset, task, method, image_idx) -- and
summarize.load_summary aggregates it on the way in, so this script simply
re-reads the same file in raw form. No extra input is required.
--tds-per-image / --baselines-per-image are there only if you keep the
per-image records somewhere else.

Note on the pseudo-inverse: run_pinv.py appends its rows to the SAME
summary file as the TDS matrix (method="pinv", task="deblur" only), so the
rows load automatically. But "pinv" is not in configs.METHODS, so without
--include-methods it would be dropped from the column list and silently
vanish from the table. Hence the extra-methods handling below.

The pseudo-inverse is an anchor, not a competitor: it uses only the
observation and the operator, with no learned prior, and its CG iteration
count is chosen by the PSNR turnover (see run_pinv.py). It belongs at the
END of the method ordering, and should be captioned as a reference row.
It is excluded from the paired comparisons by default.

Usage:
    python -m experiments.combine_summaries
    python -m experiments.combine_summaries --out results_combined.md
    python -m experiments.combine_summaries --no-pinv

    # paired tables for a different metric / reference method
    python -m experiments.combine_summaries --metric lpips
    python -m experiments.combine_summaries --ref tds --metric psnr

    # paired tables for every metric at once
    python -m experiments.combine_summaries --metric all --out results.md

    # explicit paths
    python -m experiments.combine_summaries \
        --tds-csv path/to/tds_summary.csv \
        --baselines-csv path/to/baselines_summary.csv \
        --tds-per-image path/to/tds_per_image.csv \
        --baselines-per-image path/to/baselines_per_image.csv
"""

import argparse
import contextlib
import io
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

from experiments.summarize import (
    resolve_config,
    load_summary,
    print_markdown_table,
)


# Methods that live in a summary file but are not listed in any METHODS
# dict. Appended to the end of the column order if present in the rows.
EXTRA_METHODS = ["pinv"]

# Methods excluded from the paired comparisons: not competing samplers.
NON_COMPETITORS = {"pinv"}

# metric name -> higher_is_better. Used to orient the paired delta so that
# POSITIVE ALWAYS MEANS "reference method is better", whichever metric.
METRIC_DIRECTION = OrderedDict([
    ("psnr", True),
    ("ssim", True),
    ("lpips", False),
    ("obs_cons", False),
])

# Column-name aliases tolerated in the per-image csv.
IMAGE_ID_ALIASES = ["image_id", "image", "image_idx", "idx", "index",
                    "img", "img_id", "sample_id", "i"]
METRIC_ALIASES = {
    "psnr": ["psnr"],
    "ssim": ["ssim"],
    "lpips": ["lpips"],
    "obs_cons": ["obs_cons", "obs_consistency", "obs_err", "obs"],
}


# ─────────────────────────────────────────────────────────────────────────
# Loading (means) -- unchanged from the original script
# ─────────────────────────────────────────────────────────────────────────

def load_matrix(which: str, csv_override: str | None):
    """Load (rows, datasets, tasks, methods) for one matrix ('tds' or 'baselines')."""
    datasets, tasks, methods, results_dir = resolve_config(which)
    csv_path = Path(csv_override) if csv_override else results_dir / "summary_noisy.csv"

    if not csv_path.exists():
        print(f"  ! No results found at {csv_path} -- skipping {which}.", file=sys.stderr)
        return [], datasets, tasks, methods

    rows = load_summary(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path} (matrix: {which})")
    return rows, datasets, tasks, methods


def merge_ordered(*lists):
    """Union of several lists, preserving first-seen order, no duplicates."""
    seen = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.append(item)
    return seen


def _row_get(row, key):
    """Read a field from a row that may be a dict or an object."""
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def methods_present(rows):
    """Method names actually appearing in the loaded rows, in first-seen order."""
    seen = []
    for r in rows:
        m = _row_get(r, "method")
        if m is not None and m not in seen:
            seen.append(m)
    return seen


# ─────────────────────────────────────────────────────────────────────────
# Loading (per-image)
# ─────────────────────────────────────────────────────────────────────────

def _resolve_column(fieldnames, aliases, what, path):
    lowered = {c.lower().strip(): c for c in fieldnames}
    for a in aliases:
        if a in lowered:
            return lowered[a]
    raise SystemExit(
        f"\nCould not find a '{what}' column in {path}.\n"
        f"  Looked for any of: {', '.join(aliases)}\n"
        f"  Columns present:   {', '.join(fieldnames)}\n"
        f"Rename the column, or extend the alias list at the top of this file."
    )


def load_per_image(path: Path, metrics):
    """Read a per-image csv into {(dataset, task, method): {image_id: {metric: val}}}.

    Required columns: dataset, task, method, an image identifier (see
    IMAGE_ID_ALIASES), and one column per requested metric.
    """
    import csv

    store = defaultdict(dict)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} appears to be empty.")

        fn = reader.fieldnames
        c_ds = _resolve_column(fn, ["dataset"], "dataset", path)
        c_tk = _resolve_column(fn, ["task"], "task", path)
        c_me = _resolve_column(fn, ["method"], "method", path)
        c_id = _resolve_column(fn, IMAGE_ID_ALIASES, "image id", path)
        c_metric = {m: _resolve_column(fn, METRIC_ALIASES[m], m, path)
                    for m in metrics}

        n = 0
        for row in reader:
            key = (row[c_ds].strip(), row[c_tk].strip(), row[c_me].strip())
            img = row[c_id].strip()
            vals = {}
            for m, col in c_metric.items():
                raw = (row[col] or "").strip()
                if raw == "":
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if not math.isnan(v):
                    vals[m] = v
            if vals:
                store[key][img] = vals
                n += 1

    print(f"Loaded {n} per-image records from {path}")
    return store


def merge_per_image(*stores):
    out = defaultdict(dict)
    for s in stores:
        for k, v in s.items():
            out[k].update(v)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Paired statistics
# ─────────────────────────────────────────────────────────────────────────

def _mean(xs):
    return sum(xs) / len(xs)


def _stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_stats(ref_vals, other_vals, higher_is_better, wilcoxon=False):
    """Paired comparison over the images both methods were evaluated on.

    Returns a dict, or None if there is no overlap. `delta` is oriented so
    that POSITIVE means the reference method is better, regardless of
    whether the metric is higher- or lower-is-better.
    """
    shared = sorted(set(ref_vals) & set(other_vals))
    if not shared:
        return None

    sign = 1.0 if higher_is_better else -1.0
    d = [sign * (ref_vals[i] - other_vals[i]) for i in shared]

    n = len(d)
    mean = _mean(d)
    sd = _stdev(d)
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    # Normal approximation; with n < ~15 prefer the win rate / Wilcoxon.
    ci = 1.96 * se if n > 1 else float("nan")

    wins = sum(1 for x in d if x > 0)
    ties = sum(1 for x in d if x == 0)

    p = None
    if wilcoxon and n >= 6:
        try:
            from scipy.stats import wilcoxon as _w
            if any(x != 0 for x in d):
                p = float(_w(d).pvalue)
        except ImportError:
            p = None

    return dict(n=n, mean=mean, sd=sd, se=se, ci=ci,
                wins=wins, ties=ties, win_rate=wins / n, p=p)


# ─────────────────────────────────────────────────────────────────────────
# Table rendering
# ─────────────────────────────────────────────────────────────────────────

def _settings_in_order(per_image, datasets, tasks):
    out = []
    for ds in datasets:
        for tk in tasks:
            if any(k[0] == ds and k[1] == tk for k in per_image):
                out.append((ds, tk))
    # anything the configs did not know about
    for ds, tk, _m in per_image:
        if (ds, tk) not in out:
            out.append((ds, tk))
    return out


def render_paired_tables(per_image, datasets, tasks, methods, ref, metric,
                         wilcoxon=False, min_images=1):
    """Return (delta_table_md, winrate_table_md) for one metric."""
    higher = METRIC_DIRECTION[metric]
    competitors = [m for m in methods
                   if m != ref and m not in NON_COMPETITORS]
    settings = _settings_in_order(per_image, datasets, tasks)

    arrow = "higher is better" if higher else "lower is better"
    head = "| dataset | task | n | " + " | ".join(f"vs {m}" for m in competitors) + " |"
    sep = "|---" * (len(competitors) + 3) + "|"

    d_lines = [
        f"### Paired mean difference -- {metric} ({arrow})",
        "",
        f"Per-image improvement of `{ref}` over each competitor, "
        f"mean +/- 95% CI. **Positive means `{ref}` is better.** "
        f"Computed on the images both methods were run on, so the "
        f"between-image difficulty variance cancels."
        + (" Bold marks a CI excluding zero." if True else ""),
        "",
        head, sep,
    ]
    w_lines = [
        f"### Per-image win rate -- {metric} ({arrow})",
        "",
        f"Fraction of images on which `{ref}` beats each competitor "
        f"(ties counted as losses). Distribution-free, and robust to a "
        f"small number of catastrophic-failure images.",
        "",
        head, sep,
    ]

    any_data = False
    for ds, tk in settings:
        ref_vals = {i: v[metric]
                    for i, v in per_image.get((ds, tk, ref), {}).items()
                    if metric in v}
        d_cells, w_cells, ns = [], [], []

        for m in competitors:
            other = {i: v[metric]
                     for i, v in per_image.get((ds, tk, m), {}).items()
                     if metric in v}
            st = paired_stats(ref_vals, other, higher, wilcoxon=wilcoxon)
            if st is None or st["n"] < min_images:
                d_cells.append("--")
                w_cells.append("--")
                continue

            any_data = True
            ns.append(st["n"])
            sig = (not math.isnan(st["ci"])) and abs(st["mean"]) > st["ci"]
            cell = f"{st['mean']:+.3f} ± {st['ci']:.3f}"
            if sig:
                cell = f"**{cell}**"
            if st["p"] is not None:
                cell += f"<br><sub>p={st['p']:.1e}</sub>"
            d_cells.append(cell)

            w_cells.append(f"{100 * st['win_rate']:.0f}% "
                           f"<sub>({st['wins']}/{st['n']})</sub>")

        n_str = str(max(ns)) if ns else "--"
        d_lines.append(f"| {ds} | {tk} | {n_str} | " + " | ".join(d_cells) + " |")
        w_lines.append(f"| {ds} | {tk} | {n_str} | " + " | ".join(w_cells) + " |")

    if not any_data:
        note = (f"\n_No paired data for `{ref}` on metric `{metric}` -- "
                f"check that the per-image csv contains rows for this method._\n")
        return note, note

    return "\n".join(d_lines) + "\n", "\n".join(w_lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tds-csv", default=None,
                    help="Override path to the TDS summary csv "
                         "(defaults to experiments/configs' RESULTS_DIR/summary_noisy.csv).")
    ap.add_argument("--baselines-csv", default=None,
                    help="Override path to the baselines summary csv "
                         "(defaults to configs_baselines' RESULTS_DIR/summary_noisy.csv).")
    ap.add_argument("--tds-per-image", default=None,
                    help="Path to the TDS PER-IMAGE csv. Defaults to the "
                         "TDS summary csv itself, which is already stored "
                         "one row per (dataset, task, method, image_idx).")
    ap.add_argument("--baselines-per-image", default=None,
                    help="Path to the baselines PER-IMAGE csv. Defaults to "
                         "the baselines summary csv itself.")
    ap.add_argument("--ref", default="tds_hmc_refined",
                    help="Reference method for the paired tables (default: "
                         "tds_hmc_refined). Positive deltas mean this method wins.")
    ap.add_argument("--metric", default="psnr",
                    choices=list(METRIC_DIRECTION) + ["all"],
                    help="Metric for the paired tables, or 'all' for one "
                         "pair of tables per metric (default: psnr).")
    ap.add_argument("--wilcoxon", action="store_true",
                    help="Also report a Wilcoxon signed-rank p-value per cell "
                         "(requires scipy).")
    ap.add_argument("--min-images", type=int, default=1,
                    help="Skip a paired cell with fewer than this many shared "
                         "images (default: 1).")
    ap.add_argument("--means-only", action="store_true",
                    help="Produce only the original means table, as before.")
    ap.add_argument("--no-pinv", action="store_true",
                    help="Exclude the pseudo-inverse anchor rows from the table.")
    ap.add_argument("--include-methods", nargs="*", default=None,
                    help="Extra method names to keep even if absent from the "
                         "METHODS dicts. Defaults to: " + ", ".join(EXTRA_METHODS))
    ap.add_argument("--out", default=None,
                    help="If given, also write the combined markdown to this file "
                         "(in addition to printing it).")
    args = ap.parse_args()

    # ── Table 1: means (unchanged) ─────────────────────────────────────
    tds_rows, tds_datasets, tds_tasks, tds_methods = load_matrix("tds", args.tds_csv)
    base_rows, base_datasets, base_tasks, base_methods = load_matrix("baselines", args.baselines_csv)

    all_rows = tds_rows + base_rows
    if not all_rows:
        print("No rows loaded from either results folder -- nothing to combine.")
        return

    extras = EXTRA_METHODS if args.include_methods is None else args.include_methods
    if args.no_pinv:
        extras = [m for m in extras if m != "pinv"]
        all_rows = [r for r in all_rows if _row_get(r, "method") != "pinv"]

    present = methods_present(all_rows)
    extras_found = [m for m in extras if m in present]
    extras_missing = [m for m in extras if m not in present]

    for m in extras_missing:
        print(f"  ! '{m}' rows not found in any summary -- "
              f"run experiments.run_pinv first if you want that column.",
              file=sys.stderr)

    datasets = merge_ordered(tds_datasets, base_datasets)
    tasks = merge_ordered(tds_tasks, base_tasks)
    methods = merge_ordered(tds_methods, base_methods, extras_found)

    unknown = [m for m in present if m not in methods]
    if unknown:
        print(f"  ! Methods present in the data but not in the column order "
              f"(dropped): {', '.join(unknown)}. Add them via --include-methods.",
              file=sys.stderr)

    n_pinv = sum(1 for r in all_rows if _row_get(r, "method") == "pinv")
    print(f"\nCombined {len(all_rows)} rows across {len(methods)} methods "
          f"({len(tds_rows)} TDS + {len(base_rows)} baseline"
          + (f", incl. {n_pinv} pseudo-inverse" if n_pinv else "") + ").")
    if n_pinv:
        print("Note: pseudo-inverse rows exist for the deblurring task only; "
              "other tasks will show blanks in that column.")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_markdown_table(all_rows, datasets, tasks, methods)
    means_md = buf.getvalue().strip() + "\n"

    sections = ["## 1. Metric means\n", means_md]

    # ── Tables 2 and 3: paired delta + win rate ────────────────────────
    if not args.means_only:
        metrics = list(METRIC_DIRECTION) if args.metric == "all" else [args.metric]

        def default_per_image(which, override, summary_override):
            # The summary csvs are ALREADY per-image: one row per
            # (dataset, task, method, image_idx). summarize.load_summary
            # aggregates them on the way in, but the raw rows are what we
            # want here, so by default we re-read the same file.
            if override:
                return Path(override)
            if summary_override:
                return Path(summary_override)
            _d, _t, _m, results_dir = resolve_config(which)
            return results_dir / "summary_noisy.csv"

        p_tds = default_per_image("tds", args.tds_per_image, args.tds_csv)
        p_base = default_per_image("baselines", args.baselines_per_image,
                                   args.baselines_csv)

        stores, found = [], []
        for p in (p_tds, p_base):
            if p.exists():
                stores.append(load_per_image(p, metrics))
                found.append(str(p))
            else:
                print(f"  ! No per-image csv at {p}.", file=sys.stderr)

        if not stores:
            print(
                "\n" + "=" * 70 + "\n"
                "Tables 2 and 3 need PER-IMAGE metrics and none were found.\n"
                f"  Looked in: {p_tds}\n"
                f"             {p_base}\n"
                "Pass --tds-per-image / --baselines-per-image, or re-run with\n"
                "--means-only to suppress this message.\n" + "=" * 70,
                file=sys.stderr)
        else:
            per_image = merge_per_image(*stores)
            if args.no_pinv:
                per_image = {k: v for k, v in per_image.items() if k[2] != "pinv"}

            refs_present = {k[2] for k in per_image}
            if args.ref not in refs_present:
                print(f"  ! Reference method '{args.ref}' has no per-image rows. "
                      f"Present: {', '.join(sorted(refs_present))}", file=sys.stderr)

            for i, metric in enumerate(metrics):
                d_md, w_md = render_paired_tables(
                    per_image, datasets, tasks, methods, args.ref, metric,
                    wilcoxon=args.wilcoxon, min_images=args.min_images)
                sections.append(f"\n## 2.{i + 1} Paired mean difference ({metric})\n")
                sections.append(d_md)
                sections.append(f"\n## 3.{i + 1} Per-image win rate ({metric})\n")
                sections.append(w_md)

    doc = "\n".join(sections)
    print(doc)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(doc.strip() + "\n")
        print(f"\nWrote combined tables to {out_path}")


if __name__ == "__main__":
    main()