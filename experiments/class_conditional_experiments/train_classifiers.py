import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.class_conditional_experiments.models import build_model

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                       # -> [0, 1]
    transforms.Normalize((0.5,), (0.5,)),        # -> [-1, 1]
])


def ensure_mnist(data_root: str) -> None:

    raw_dir = os.path.join(data_root, "MNIST", "raw")
    already = os.path.isdir(raw_dir) and any(
        os.path.exists(os.path.join(raw_dir, f))
        for f in ("train-images-idx3-ubyte", "train-images-idx3-ubyte.gz")
    )
    if already:
        print(f"MNIST already present in {data_root}, skipping download.")
        return

    print(f"MNIST not found in {data_root}; downloading...")
    datasets.MNIST(data_root, train=True,  download=True)
    datasets.MNIST(data_root, train=False, download=True)
    print("  download complete.")


def get_loaders(data_root: str, batch_size: int, num_workers: int):
    ensure_mnist(data_root)
    train = datasets.MNIST(data_root, train=True,  download=True,
                           transform=MNIST_TRANSFORM)
    test = datasets.MNIST(data_root, train=False, download=True,
                          transform=MNIST_TRANSFORM)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: str) -> float:
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = net(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train_one(which: str, seed: int, out_path: str, *,
              epochs: int, lr: float, data_root: str, batch_size: int,
              num_workers: int, device: str) -> nn.Module:
    torch.manual_seed(seed)

    train_loader, test_loader = get_loaders(data_root, batch_size, num_workers)
    net = build_model(which).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n=== training '{which}' (seed={seed}) -> {out_path} ===")
    for epoch in range(epochs):
        net.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        sched.step()
        train_loss = running / len(train_loader.dataset)
        test_acc = evaluate(net, test_loader, device)
        print(f"  epoch {epoch + 1}/{epochs}  "
              f"train_loss {train_loss:.4f}  test_acc {test_acc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"which": which, "seed": seed,
                "state_dict": net.state_dict()}, out_path)
    print(f"  saved -> {out_path}")
    return net


def load_classifier(path: str, device: str = "cpu") -> nn.Module:
    """Rebuild a trained classifier from a checkpoint saved by this script."""
    ckpt = torch.load(path, map_location=device)
    net = build_model(ckpt["which"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    p = argparse.ArgumentParser(description="Train MNIST guidance/eval classifiers")
    p.add_argument("--which", choices=["guidance", "eval", "both"],
                   default="both")
    p.add_argument("--seed", type=int, default=None,
                   help="override seed (default: 0 for guidance, 1 for eval)")
    p.add_argument("--out", type=str, default=None,
                   help="override output path (only valid with a single --which)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    defaults = {
        "guidance": {"seed": 0, "out": "clf_guide.pt"},
        "eval":     {"seed": 1, "out": "clf_eval.pt"},
    }

    if args.which == "both":
        if args.out is not None:
            p.error("--out cannot be used with --which both")
        for role in ("guidance", "eval"):
            train_one(
                role,
                seed=defaults[role]["seed"] if args.seed is None else args.seed,
                out_path=defaults[role]["out"],
                epochs=args.epochs, lr=args.lr, data_root=args.data_root,
                batch_size=args.batch_size, num_workers=args.num_workers,
                device=args.device,
            )
    else:
        role = args.which
        train_one(
            role,
            seed=defaults[role]["seed"] if args.seed is None else args.seed,
            out_path=args.out or defaults[role]["out"],
            epochs=args.epochs, lr=args.lr, data_root=args.data_root,
            batch_size=args.batch_size, num_workers=args.num_workers,
            device=args.device,
        )


if __name__ == "__main__":
    main()