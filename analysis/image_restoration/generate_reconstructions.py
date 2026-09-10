import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

RESULTS_DIR = Path("experiments/results")

DATASET_TASKS = {
    "mnist":       ["inpaint", "inpaint_random", "superres", "deblur"],
    "flowers":     ["inpaint", "inpaint_random", "superres", "deblur"],
    "butterflies": ["inpaint", "inpaint_random", "superres", "deblur"],
    "celeba":      ["inpaint", "inpaint_random", "superres", "deblur"],
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "flowers": "Flowers",
    "butterflies": "Butterflies",
    "celeba": "CelebA",
}

TASK_LABELS = {
    "inpaint":        "Center Mask",
    "inpaint_random": "Random Mask",
    "superres":       "Super-Resolution",
    "deblur":         "Deblur",
}

METHOD_DIRS = {
    "tds":             RESULTS_DIR,
    "tds_hmc_refined": RESULTS_DIR,
}

METHODS = ["tds", "tds_hmc_refined"]

METHOD_LABELS = {
    "tds":             "TDS",
    "tds_hmc_refined": "rSFK-uHMC",
}


GT_KEY = "__gt__"
OBS_KEY = "__obs__"
EXTRA_LABELS = {GT_KEY: "Ground truth", OBS_KEY: "Observation"}

GT_FILES = ["{index}_gt.png", "{index}_ground_truth.png", "{index}_target.png",
            "{index}_original.png", "{index}_clean.png"]
OBS_FILES = ["{index}_observed.png", "{index}_observation.png",
             "{index}_y.png", "{index}_measurement.png",
             "{index}_degraded.png", "{index}_masked.png"]



def set_publication_style(usetex: bool = True, font_size: float = 1.0) -> None:
    """NeurIPS / TMLR-compatible rcParams, all sizes derived from font_size."""
    plt.rcParams.update({
        "text.usetex":        usetex,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "Computer Modern Roman",
                               "DejaVu Serif"],
        "font.size":          font_size,
        "axes.titlesize":     font_size,
        "axes.labelsize":     font_size,
        "xtick.labelsize":    font_size,
        "ytick.labelsize":    font_size,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth":     0.5,
    })
    if usetex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"


def load_image(path: Path, quiet: bool = False):
    if not path.exists():
        if not quiet:
            print(f"  [missing] {path}")
        return None
    return mpimg.imread(str(path))


def result_path(dataset: str, task: str, method: str, index: str) -> Path:
    base = METHOD_DIRS[method]
    return base / f"{dataset}_{task}_{method}" / f"{index}_compare.png"


def find_aux(dataset: str, task: str, index: str, candidates, methods) -> Path | None:
    for method in methods:
        folder = METHOD_DIRS[method] / f"{dataset}_{task}_{method}"
        for pattern in candidates:
            p = folder / pattern.format(index=index)
            if p.exists():
                return p
    return None


def pick_interpolation(img, cell_px: float) -> str:
    if img is None:
        return "nearest"
    h = img.shape[0]
    return "nearest" if cell_px >= h * 1.5 else "lanczos"


def normalise_for_show(img):
    if img is None:
        return None, None
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    cmap = "gray" if a.ndim == 2 else None
    if a.dtype.kind == "f" and a.size and float(np.nanmax(a)) > 1.0 + 1e-6:
        a = np.clip(a / 255.0, 0, 1)
    return a, cmap


def make_figure(
    dataset: str,
    index: str,
    output: str | None = None,
    dpi: int = 300,
    fig_width: float = 5.5,
    usetex: bool = True,
    show_title: bool = False,
    methods: list[str] | None = None,
    transpose: bool = False,
    font_size: float = 8.0,
    show_gt: bool = False,
    show_observed: bool = False,
    highlight: str | None = None,
    pad: float = 0.02,
    method_gap: float | None = None,
    task_gap: float | None = None,
    bold_labels: bool = True,
) -> None:

    set_publication_style(usetex=usetex, font_size=font_size)

    methods = methods or METHODS
    tasks = DATASET_TASKS[dataset]

    extras = []
    if show_gt:
        extras.append(GT_KEY)
    if show_observed:
        extras.append(OBS_KEY)
    method_cols = extras + methods

    imgs: dict[tuple, np.ndarray | None] = {}
    for task in tasks:
        for m in methods:
            imgs[(task, m)] = load_image(result_path(dataset, task, m, index))
        if show_gt:
            p = find_aux(dataset, task, index, GT_FILES, methods)
            imgs[(task, GT_KEY)] = load_image(p, quiet=True) if p else None
        if show_observed:
            p = find_aux(dataset, task, index, OBS_FILES, methods)
            imgs[(task, OBS_KEY)] = load_image(p, quiet=True) if p else None

    if show_gt and all(imgs.get((t, GT_KEY)) is None for t in tasks):
        print("  [warn] --gt requested but no ground-truth images found "
              f"(looked for {', '.join(GT_FILES)})")
    if show_observed and all(imgs.get((t, OBS_KEY)) is None for t in tasks):
        print("  [warn] --observed requested but no observation images found "
              f"(looked for {', '.join(OBS_FILES)})")

    found = [v for v in imgs.values() if v is not None]
    if not found:
        print(f"  [skip] no images at all for {dataset} index {index}")
        return

    if transpose:
        row_keys, col_keys = method_cols, tasks
        def row_label_of(k):
            return EXTRA_LABELS.get(k) or METHOD_LABELS.get(k, k)
        def col_label_of(k):
            return TASK_LABELS.get(k, k)
    else:
        row_keys, col_keys = tasks, method_cols
        def row_label_of(k):
            return TASK_LABELS.get(k, k)
        def col_label_of(k):
            return EXTRA_LABELS.get(k) or METHOD_LABELS.get(k, k)

    n_rows, n_cols = len(row_keys), len(col_keys)

    h0, w0 = found[0].shape[:2]
    aspect = h0 / w0

    pt = 1.0 / 72.0
    label_w_in = font_size * pt * 2.4 if any(row_label_of(k) for k in row_keys) else 0.0
    header_h_in = font_size * pt * 2.0

    _m_gap = pad if method_gap is None else method_gap
    _t_gap = pad if task_gap is None else task_gap
    _wsp, _hsp = (_t_gap, _m_gap) if transpose else (_m_gap, _t_gap)

    img_w_in = (fig_width - label_w_in) / (n_cols + (n_cols - 1) * _wsp)
    img_h_in = img_w_in * aspect
    fig_h = header_h_in + img_h_in * (n_rows + (n_rows - 1) * _hsp)

    m_gap = pad if method_gap is None else method_gap
    t_gap = pad if task_gap is None else task_gap
    wspace, hspace = (t_gap, m_gap) if transpose else (m_gap, t_gap)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_h),
        squeeze=False,
        gridspec_kw={
            "left":   label_w_in / fig_width,
            "right":  0.998,
            "top":    1.0 - header_h_in / fig_h,
            "bottom": 0.002,
            "wspace": wspace,
            "hspace": hspace,
        },
    )

    cell_px = img_w_in * dpi

    for r, row_key in enumerate(row_keys):
        for c, col_key in enumerate(col_keys):
            task, mkey = (col_key, row_key) if transpose else (row_key, col_key)
            ax = axes[r, c]
            arr, cmap = normalise_for_show(imgs.get((task, mkey)))

            if arr is not None:
                ax.imshow(arr, cmap=cmap,
                          interpolation=pick_interpolation(arr, cell_px))
            else:
                ax.set_facecolor("#f2f2f2")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="#9a9a9a",
                        fontsize=font_size, style="italic")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if highlight and mkey == highlight:
                for side, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(1.0)
                    spine.set_color("#c0392b")

            if r == 0:
                ax.set_title(col_label_of(col_key), pad=3,
                             fontweight=("bold" if bold_labels and not
                                         plt.rcParams["text.usetex"] else "normal"))

    for r, row_key in enumerate(row_keys):
        box = axes[r, 0].get_position()
        fig.text(
            (label_w_in / fig_width) * 0.42,
            box.y0 + box.height / 2.0,
            row_label_of(row_key),
            ha="center", va="center", rotation=90,
            fontweight=("bold" if bold_labels and not
                        plt.rcParams["text.usetex"] else "normal"),
        )

    if show_title:
        fig.suptitle(
            f"{DATASET_LABELS.get(dataset, dataset)} — image {index}",
            fontsize=font_size, y=1.005,
        )

    if output is None:
        output = f"{dataset}_{index}_comparison.png"
    out_path = Path(output)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved → {out_path}  ({fig_width:.2f} x {fig_h:.2f} in, "
          f"{n_rows}x{n_cols})")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reconstruction comparison figures across all methods."
    )
    parser.add_argument("--dataset", choices=list(DATASET_TASKS.keys()),
                        help="Dataset to plot (ignored when --all is set).")
    parser.add_argument("--index", default="01",
                        help="Zero-padded image index, e.g. 01, 07, 23.")
    parser.add_argument("--output", default=None,
                        help="Output path (PNG or PDF). Prefer .pdf for "
                             "vector text in the paper.")
    parser.add_argument("--all", action="store_true",
                        help="Generate figures for all datasets at --index.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for raster outputs (default: 300).")
    parser.add_argument("--width", type=float, default=5.5,
                        help="Figure width in inches (default: 5.5).")
    parser.add_argument("--font-size", type=float, default=5.0,
                        help="Base font size in points; every other size is "
                             "derived from it (default: 8).")
    parser.add_argument("--pad", type=float, default=0.02,
                        help="Default gap between image cells, as a fraction "
                             "of cell size (default: 0.02; 0 = seamless). "
                             "Overridden by --method-gap / --task-gap.")
    parser.add_argument("--method-gap", type=float, default=None,
                        help="Gap between different METHODS. Separates the "
                             "samplers from each other; raise this to make "
                             "the per-method columns read as distinct.")
    parser.add_argument("--task-gap", type=float, default=None,
                        help="Gap between different TASKS (masks / "
                             "super-resolution / deblur). Lower this to pull "
                             "the rows for one method tightly together.")
    parser.add_argument("--gt", dest="show_gt", action="store_true",
                        help="Prepend a ground-truth column.")
    parser.add_argument("--observed", dest="show_observed", action="store_true",
                        help="Prepend the degraded observation column.")
    parser.add_argument("--highlight", default=None,
                        help="Outline this method's panels, e.g. tds_hmc_refined.")
    parser.add_argument("--no-bold", dest="bold_labels", action="store_false",
                        default=True, help="Do not embolden row/column labels.")
    parser.add_argument("--no-usetex", dest="usetex", action="store_false",
                        default=True, help="Disable LaTeX rendering.")
    parser.add_argument("--title", dest="show_title", action="store_true",
                        default=False, help="Add a draft title.")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset/order of methods, e.g. "
                             "'tds,tds_hmc_refined'. Default: "
                             + ",".join(METHODS))
    parser.add_argument("--transpose", action="store_true", default=False,
                        help="Rows = methods, columns = tasks.")
    args = parser.parse_args()

    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        unknown = [m for m in methods if m not in METHOD_DIRS]
        if unknown:
            parser.error(f"Unknown method(s): {', '.join(unknown)}. "
                         f"Available: {', '.join(METHOD_DIRS)}")

    if args.highlight and args.highlight not in METHOD_DIRS:
        parser.error(f"--highlight must be one of: {', '.join(METHOD_DIRS)}")

    kwargs = dict(
        dpi=args.dpi, fig_width=args.width, usetex=args.usetex,
        show_title=args.show_title, methods=methods, transpose=args.transpose,
        font_size=args.font_size, show_gt=args.show_gt,
        show_observed=args.show_observed, highlight=args.highlight,
        pad=args.pad, method_gap=args.method_gap,
        task_gap=args.task_gap, bold_labels=args.bold_labels,
    )

    if args.all:
        for dataset in DATASET_TASKS:
            out = args.output
            if out:
                stem, suffix = os.path.splitext(out)
                out = f"{stem}_{dataset}{suffix}"
            out = out or f"{dataset}_{args.index}_comparison.png"
            make_figure(dataset, args.index, out, **kwargs)
    else:
        if args.dataset is None:
            parser.error("Please supply --dataset or use --all.")
        make_figure(args.dataset, args.index, args.output, **kwargs)


if __name__ == "__main__":
    main()
