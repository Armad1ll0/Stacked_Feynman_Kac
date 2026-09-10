import argparse
import csv
from collections import defaultdict
from pathlib import Path


CONS = "obs_consistency"   # RMSE(A(x_hat), A(x_clean))
RESID = "obs_residual"     # RMSE(A(x_hat), y)


def load(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _f(row, field):
    v = row.get(field, "")
    if v == "" or v is None:
        return None
    try:
        x = float(v)
    except (ValueError, TypeError):
        return None
    return x if x == x else None  # drop NaN


def collect(rows):
    """Per (dataset, task, method): list of (cons, resid) pairs, one per sample."""
    out = defaultdict(list)
    skipped = 0
    for r in rows:
        c, d = _f(r, CONS), _f(r, RESID)
        if c is None or d is None:
            skipped += 1
            continue
        out[(r["dataset"], r["task"], r["method"])].append((c, d))
    return out, skipped


def summarize(pairs, sigma):
    n = len(pairs)
    n_pass = sum(1 for c, d in pairs if c < d)
    mean_cons = sum(c for c, _ in pairs) / n
    mean_resid = sum(d for _, d in pairs) / n
    return {
        "n": n,
        "n_pass": n_pass,
        "pass_rate": n_pass / n,
        "mean_cons": mean_cons,
        "mean_resid": mean_resid,
        "ratio": mean_resid / sigma if sigma else float("nan"),
        # worst individual violation, for spotting a single bad image
        "worst_margin": min(d - c for c, d in pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sigma", type=float, required=True,
                    help="Measurement noise std, in the same units as the "
                         "pixel data (e.g. 0.05 on [0,1]).")
    ap.add_argument("--method", default=None,
                    help="Restrict to one method, e.g. tds_hmc_refined.")
    ap.add_argument("--markdown", action="store_true",
                    help="Emit a markdown table instead of plain text.")
    args = ap.parse_args()

    rows = load(Path(args.csv))
    grouped, skipped = collect(rows)
    if skipped:
        print(f"# skipped {skipped} rows missing {CONS} or {RESID}\n")

    keys = sorted(grouped)
    if args.method:
        keys = [k for k in keys if k[2] == args.method]

    if args.markdown:
        print(f"sigma_y = {args.sigma}\n")
        print("| dataset | task | method | mean cons | mean resid | "
              "resid/sigma | cons < resid |")
        print("|---|---|---|---|---|---|---|")

    total_n = total_pass = 0
    for k in keys:
        s = summarize(grouped[k], args.sigma)
        total_n += s["n"]
        total_pass += s["n_pass"]
        if args.markdown:
            print(f"| {k[0]} | {k[1]} | {k[2]} | "
                  f"{s['mean_cons']:.4f} | {s['mean_resid']:.4f} | "
                  f"{s['ratio']:.2f} | {s['n_pass']}/{s['n']} |")
        else:
            flag = "" if s["n_pass"] == s["n"] else "   <-- VIOLATION"
            print(f"{k[0]:>12} {k[1]:>15} {k[2]:>18}  "
                  f"cons={s['mean_cons']:.4f}  resid={s['mean_resid']:.4f}  "
                  f"resid/sigma={s['ratio']:.2f}  "
                  f"pass={s['n_pass']}/{s['n']}  "
                  f"worst_margin={s['worst_margin']:+.4f}{flag}")

    if total_n:
        print(f"\nOverall: {total_pass}/{total_n} samples satisfy "
              f"cons < resid ({100 * total_pass / total_n:.1f}%)")


if __name__ == "__main__":
    main()

# python -m analysis.ablation.check_consistency --csv experiments/results/summary_noisy.csv --sigma 0.05 --method tds_hmc_refined --markdown