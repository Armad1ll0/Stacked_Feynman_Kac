import argparse
import glob
import os

import torch
import matplotlib
import matplotlib.pyplot as plt


def _load_one_per_class(samples_dir, classes):
    out = {}
    for c in classes:
        path = os.path.join(samples_dir, f"class_{c}.pt")
        if not os.path.exists(path):
            out[c] = None
            continue
        blob = torch.load(path, map_location="cpu")
        s = blob["samples"]
        if s.shape[0] == 0:
            out[c] = None
            continue
        img = s[0].float()                       # (C,H,W), first sample, in [-1,1]
        img = (img + 1.0) / 2.0                  # -> [0,1] for display
        img = img.clamp(0, 1)
        out[c] = img
    return out


def plot_grid(method_dirs, out_path, classes=range(10),
              row_labels=None, samples_per_cell=1):
    matplotlib.use("Agg")        
    classes = list(classes)
    n_rows = len(method_dirs)
    n_cols = len(classes)
    if row_labels is None:
        row_labels = [lbl for lbl, _ in method_dirs]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.1, n_rows * 1.1 * samples_per_cell),
        squeeze=False,
    )

    for r, (_, sdir) in enumerate(method_dirs):
        per_class = _load_one_per_class(sdir, classes) if samples_per_cell == 1 \
            else _load_k_per_class(sdir, classes, samples_per_cell)
        for ci, c in enumerate(classes):
            ax = axes[r][ci]
            ax.set_xticks([]); ax.set_yticks([])
            img = per_class[c]
            if img is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                # img is (C,H,W) for k=1, or (C, k*H, W) tiled for k>1.
                arr = img.squeeze(0).numpy() if img.shape[0] == 1 \
                    else img.permute(1, 2, 0).numpy()
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(str(c), fontsize=11)
            if ci == 0:
                ax.set_ylabel(row_labels[r], fontsize=11, rotation=90,
                              labelpad=8, va="center")

    fig.suptitle("Class-conditional samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved sample grid -> {out_path}")
    return out_path


def _load_k_per_class(samples_dir, classes, k):
    out = {}
    for c in classes:
        path = os.path.join(samples_dir, f"class_{c}.pt")
        if not os.path.exists(path):
            out[c] = None
            continue
        blob = torch.load(path, map_location="cpu")
        s = blob["samples"].float()
        if s.shape[0] == 0:
            out[c] = None
            continue
        k_use = min(k, s.shape[0])
        imgs = (s[:k_use] + 1.0) / 2.0
        imgs = imgs.clamp(0, 1)                       # (k,C,H,W)
        C, H, W = imgs.shape[1:]
        tiled = imgs.permute(1, 0, 3, 2).reshape(C, k_use * W, H).transpose(1, 2)
        out[c] = tiled                                # (C, k*H, W)
    return out


def main():
    p = argparse.ArgumentParser(description="Two-row sample grid: TDS vs TDS+HMC")
    p.add_argument("--tds-dir", type=str, required=True)
    p.add_argument("--hmc-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="samples_grid.png")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--samples-per-cell", type=int, default=1,
                   help="stack k samples vertically per cell to show diversity")
    args = p.parse_args()

    plot_grid(
        method_dirs=[("TDS", args.tds_dir), ("TDS+HMC", args.hmc_dir)],
        out_path=args.out,
        classes=args.classes,
        row_labels=["TDS", "TDS+HMC"],
        samples_per_cell=args.samples_per_cell,
    )


if __name__ == "__main__":
    main()