"""
experiments/combine_summaries.py
---------------------------------
Combine results from ALL result matrices -- TDS/TDS+HMC, the baselines
(DPS/PC/FPS/MCGdiff), and the pseudo-inverse deblurring anchor -- into a
single markdown table so every method can be compared side by side.

This reuses the loading/formatting logic from experiments/summarize.py --
it does not duplicate it -- so any changes you make there (new metric
columns, formatting, etc.) are automatically picked up here too.

Note on the pseudo-inverse: run_pinv.py appends its rows to the SAME
summary file as the TDS matrix (method="pinv", task="deblur" only), so the
rows load automatically. But "pinv" is not in configs.METHODS, so without
--include-methods it would be dropped from the column list and silently
vanish from the table. Hence the extra-methods handling below.

The pseudo-inverse is an anchor, not a competitor: it uses only the
observation and the operator, with no learned prior, and its CG iteration
count is chosen by the PSNR turnover (see run_pinv.py). It belongs at the
END of the method ordering, and should be captioned as a reference row.

Usage:
    python -m experiments.combine_summaries
    python -m experiments.combine_summaries --out results_combined.md
    python -m experiments.combine_summaries --no-pinv
    python -m experiments.combine_summaries \
        --tds-csv path/to/tds_summary.csv \
        --baselines-csv path/to/baselines_summary.csv
"""

import argparse
import sys
from pathlib import Path

from experiments.summarize import (
    resolve_config,
    load_summary,
    print_markdown_table,
)


# Methods that live in a summary file but are not listed in any METHODS
# dict. Appended to the end of the column order if present in the rows.
EXTRA_METHODS = ["pinv"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tds-csv", default=None,
                     help="Override path to the TDS summary csv "
                          "(defaults to experiments/configs' RESULTS_DIR/summary_noisy.csv).")
    ap.add_argument("--baselines-csv", default=None,
                     help="Override path to the baselines summary csv "
                          "(defaults to configs_baselines' RESULTS_DIR/summary_noisy.csv).")
    ap.add_argument("--no-pinv", action="store_true",
                     help="Exclude the pseudo-inverse anchor rows from the table.")
    ap.add_argument("--include-methods", nargs="*", default=None,
                     help="Extra method names to keep even if absent from the "
                          "METHODS dicts. Defaults to: " + ", ".join(EXTRA_METHODS))
    ap.add_argument("--out", default=None,
                     help="If given, also write the combined markdown table to this file "
                          "(in addition to printing it).")
    args = ap.parse_args()

    tds_rows, tds_datasets, tds_tasks, tds_methods = load_matrix("tds", args.tds_csv)
    base_rows, base_datasets, base_tasks, base_methods = load_matrix("baselines", args.baselines_csv)

    all_rows = tds_rows + base_rows
    if not all_rows:
        print("No rows loaded from either results folder -- nothing to combine.")
        return

    # ── Extra methods (pinv) ───────────────────────────────────────────
    # These are already in all_rows -- run_pinv writes to the TDS summary --
    # but are not in any METHODS dict, so they must be added to the column
    # order explicitly or the formatter will drop them.
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
    # Extras go LAST: the pseudo-inverse is a reference anchor, not a
    # competing sampler, and reads better at the end of the row.
    methods = merge_ordered(tds_methods, base_methods, extras_found)

    # Warn about anything in the data that no config knows about, rather
    # than silently dropping it.
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

    if args.out:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_markdown_table(all_rows, datasets, tasks, methods)
        table_text = buf.getvalue()

        print(table_text)  # still show it in the terminal

        out_path = Path(args.out)
        out_path.write_text(table_text.strip() + "\n")
        print(f"\nWrote combined table to {out_path}")
    else:
        print_markdown_table(all_rows, datasets, tasks, methods)


if __name__ == "__main__":
    main()