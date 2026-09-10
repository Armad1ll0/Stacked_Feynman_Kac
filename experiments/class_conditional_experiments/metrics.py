import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from experiments.class_conditional_experiments.train_classifiers import load_classifier, MNIST_TRANSFORM



def load_generated(samples_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(samples_dir, "class_*.pt"))):
        blob = torch.load(path, map_location="cpu")
        out[int(blob["label"])] = blob["samples"].float()
    if not out:
        raise FileNotFoundError(f"no class_*.pt files found in {samples_dir}")
    return out


def load_real_by_class(data_root="./data"):
    ds = datasets.MNIST(data_root, train=False, download=True,
                        transform=MNIST_TRANSFORM)
    xs = torch.stack([ds[i][0] for i in range(len(ds))])  
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    return {c: xs[ys == c] for c in range(10)}


@torch.no_grad()
def class_accuracy(samples, label, eval_clf, device, batch_size=512):
    correct = total = 0
    for i in range(0, samples.shape[0], batch_size):
        x = samples[i:i + batch_size].to(device)
        pred = eval_clf(x).argmax(dim=1)
        correct += (pred == label).sum().item()
        total += x.shape[0]
    return correct / max(total, 1)


class MNISTFeatures:

    def __init__(self, eval_ckpt, device):
        self.net = load_classifier(eval_ckpt, device=device)
        self.device = device
        if not hasattr(self.net, "features_only"):
            raise AttributeError(
                "eval classifier has no features_only(); expected an EvalNet "
                "checkpoint (see models.py)."
            )

    @torch.no_grad()
    def __call__(self, x, batch_size=512):
        feats = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size].to(self.device)  # (B,1,28,28) [-1,1]
            feats.append(self.net.features_only(xb).cpu())
        return torch.cat(feats, dim=0).numpy()


class InceptionFeatures:

    def __init__(self, device):
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        net = inception_v3(weights=weights, aux_logits=True)
        net.fc = torch.nn.Identity()   # expose 2048-d pool features
        self.net = net.to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x, batch_size=128):
        feats = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i + batch_size].to(self.device)          # (B,1,28,28) [-1,1]
            xb = (xb + 1.0) / 2.0                              # -> [0,1]
            xb = xb.repeat(1, 3, 1, 1)                         # -> 3 channels
            xb = F.interpolate(xb, size=(299, 299),
                               mode="bilinear", align_corners=False)
            xb = (xb - self.mean) / self.std
            feats.append(self.net(xb).cpu())
        return torch.cat(feats, dim=0).numpy()


def _sqrtm_psd(mat):
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def fid_from_features(feat_real, feat_fake):
    mu_r, mu_f = feat_real.mean(0), feat_fake.mean(0)
    cov_r = np.cov(feat_real, rowvar=False)
    cov_f = np.cov(feat_fake, rowvar=False)
    diff = mu_r - mu_f
    covmean = _sqrtm_psd(cov_r @ cov_f)
    fid = diff @ diff + np.trace(cov_r + cov_f - 2.0 * covmean)
    return float(fid)


def kid_from_features(feat_real, feat_fake, n_subsets=100, subset_size=1000,
                      rng=None):
    rng = rng or np.random.default_rng(0)
    n_r, n_f = len(feat_real), len(feat_fake)
    m = min(subset_size, n_r, n_f)
    d = feat_real.shape[1]

    def poly_kernel(a, b):
        return (a @ b.T / d + 1.0) ** 3

    ests = []
    for _ in range(n_subsets):
        r = feat_real[rng.choice(n_r, m, replace=False)]
        f = feat_fake[rng.choice(n_f, m, replace=False)]
        krr, kff, krf = poly_kernel(r, r), poly_kernel(f, f), poly_kernel(r, f)
        np.fill_diagonal(krr, 0.0)
        np.fill_diagonal(kff, 0.0)
        mmd2 = (krr.sum() / (m * (m - 1))
                + kff.sum() / (m * (m - 1))
                - 2.0 * krf.mean())
        ests.append(mmd2)
    ests = np.asarray(ests)
    return float(ests.mean()), float(ests.std())


def _pairwise_dist(a, b):
    a2 = (a ** 2).sum(1, keepdims=True)
    b2 = (b ** 2).sum(1, keepdims=True).T
    d2 = np.clip(a2 + b2 - 2.0 * a @ b.T, 0.0, None)
    return np.sqrt(d2)


def _knn_radii(feat, k):
    d = _pairwise_dist(feat, feat)
    np.fill_diagonal(d, np.inf)          # exclude self
    d.sort(axis=1)
    return d[:, k - 1]                    # k-th neighbour (0-indexed k-1)


def precision_recall(feat_real, feat_fake, k=3):
    real_radii = _knn_radii(feat_real, k)
    fake_radii = _knn_radii(feat_fake, k)
    d_rf = _pairwise_dist(feat_real, feat_fake)   # (n_real, n_fake)

    precision = (d_rf < real_radii[:, None]).any(axis=0).mean()
    recall = (d_rf < fake_radii[None, :]).any(axis=1).mean()
    return float(precision), float(recall)


def evaluate(samples_dir, eval_ckpt, feature_spaces, data_root, device,
             pr_k=3, kid_subset=1000):
    device = torch.device(device)
    gen = load_generated(samples_dir)
    real = load_real_by_class(data_root)
    eval_clf = load_classifier(eval_ckpt, device=device)

    extractors = {}
    if "mnist" in feature_spaces:
        extractors["mnist"] = MNISTFeatures(eval_ckpt, device)
    if "inception" in feature_spaces:
        extractors["inception"] = InceptionFeatures(device)

    per_class = {}
    for c in sorted(gen.keys()):
        g, r = gen[c], real[c]
        entry = {"n_gen": int(g.shape[0]), "n_real": int(r.shape[0])}

        # 1. accuracy
        entry["accuracy"] = class_accuracy(g, c, eval_clf, device)

        # 2 & 3. per feature space
        for space, extract in extractors.items():
            fr, ff = extract(r), extract(g)
            entry[f"fid_{space}"] = fid_from_features(fr, ff)
            kid_m, kid_s = kid_from_features(
                fr, ff, subset_size=min(kid_subset, len(fr), len(ff)))
            entry[f"kid_{space}_mean"] = kid_m
            entry[f"kid_{space}_std"] = kid_s
            if space == "mnist":
                if min(len(fr), len(ff)) >= pr_k + 1:
                    prec, rec = precision_recall(fr, ff, k=pr_k)
                    entry["precision"] = prec
                    entry["recall"] = rec
                else:
                    entry["precision"] = None
                    entry["recall"] = None

        per_class[c] = entry
        _print_class_row(c, entry)

    agg = {}
    keys = [k for k in next(iter(per_class.values()))
            if k not in ("n_gen", "n_real")]
    for k in keys:
        vals = [per_class[c][k] for c in per_class if per_class[c][k] is not None]
        agg[k] = float(np.mean(vals)) if vals else None

    print("\n=== macro-average over classes ===")
    for k, v in agg.items():
        print(f"  {k:22s} {v:.4f}" if v is not None else f"  {k:22s} n/a")

    return {"per_class": per_class, "aggregate": agg,
            "samples_dir": samples_dir, "feature_spaces": feature_spaces}


def _print_class_row(c, e):
    bits = [f"class {c}", f"acc {e['accuracy']:.3f}"]
    if "fid_mnist" in e:
        bits.append(f"FID(mnist) {e['fid_mnist']:.2f}")
    if e.get("recall") is not None:
        bits.append(f"P/R {e['precision']:.2f}/{e['recall']:.2f}")
    if "fid_inception" in e:
        bits.append(f"FID(incep) {e['fid_inception']:.2f}")
    print("  " + "  ".join(bits))


def main():
    p = argparse.ArgumentParser(description="Metrics for class-conditional MNIST")
    p.add_argument("--samples-dir", type=str, default='runs/exp/tds',
                   help="dir of class_*.pt from sample_mnist.py")
    p.add_argument("--eval-ckpt", type=str, default="clf_eval.pt")
    p.add_argument("--feature-spaces", nargs="+", default=["mnist"],
                   choices=["mnist", "inception"],
                   help="feature space(s) for FID/KID; 'mnist' is domain-matched")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--pr-k", type=int, default=3,
                   help="k for precision/recall k-NN manifolds")
    p.add_argument("--kid-subset", type=int, default=1000)
    p.add_argument("--out", type=str, default=None, help="write results JSON here")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    results = evaluate(
        args.samples_dir, args.eval_ckpt, args.feature_spaces,
        args.data_root, args.device, pr_k=args.pr_k, kid_subset=args.kid_subset,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()