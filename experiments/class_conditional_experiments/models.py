import torch
import torch.nn as nn


class GuidanceNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) in [-1, 1] -> logits (B, num_classes)."""
        return self.classifier(self.features(x))


class EvalNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 7x7
            nn.AdaptiveAvgPool2d(1),                      # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) in [-1, 1] -> logits (B, num_classes)."""
        return self.classifier(self.features(x))

    def features_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


ARCHITECTURES = {
    "guidance": GuidanceNet,
    "eval": EvalNet,
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Factory: build_model('guidance') or build_model('eval')."""
    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown model '{name}', expected one of {list(ARCHITECTURES)}"
        )
    return ARCHITECTURES[name](num_classes=num_classes)


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)  # stand-in for x_hat in [-1, 1]
    for role in ARCHITECTURES:
        net = build_model(role)
        out = net(x)
        assert out.shape == (4, 10), (role, out.shape)
        print(f"{role:8s} -> logits {tuple(out.shape)}  params "
              f"{sum(p.numel() for p in net.parameters()):,}")