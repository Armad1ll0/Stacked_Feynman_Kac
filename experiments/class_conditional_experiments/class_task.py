from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from tasks.base import Task, TaskConfig


@dataclass
class ClassConditionalConfig(TaskConfig):
    classifier_ckpt: str = "clf_guide.pt"
    num_classes: int = 10
    guidance_scale: float = 1.0
    clamp_classifier_input: bool = True
    device: str = "cpu"


class ClassConditionalTask(Task):

    def __init__(self, cfg: ClassConditionalConfig):
        super().__init__(cfg)
        self.cfg: ClassConditionalConfig = cfg
        self._clf: Optional[torch.nn.Module] = None  # lazy-loaded

    @property
    def classifier(self) -> torch.nn.Module:
        if self._clf is None:
            # Local import to avoid a hard dependency when the task is only
            # being introspected (e.g. config validation).
            from experiments.class_conditional_experiments.train_classifiers import load_classifier
            clf = load_classifier(self.cfg.classifier_ckpt, device=self.cfg.device)
            for p in clf.parameters():
                p.requires_grad_(False)
            clf.eval()
            self._clf = clf
        return self._clf

    def log_potential(self, x0_hat, observation, metadata):
        if "label" not in metadata:
            raise KeyError(
                "ClassConditionalTask.log_potential expects the target class in "
                "metadata['label']. Pass metadata={'label': c} to the sampler "
                "(the observation tensor is used only for its (C,H,W) shape)."
            )

        clf_in = x0_hat
        if self.cfg.clamp_classifier_input:
            clf_in = clf_in.clamp(-1, 1)

        logits = self.classifier(clf_in)                 # (P, num_classes)
        logp = F.log_softmax(logits, dim=1)              # (P, num_classes)

        P = x0_hat.shape[0]
        lbl = metadata["label"]
        if not torch.is_tensor(lbl):
            lbl = torch.as_tensor(lbl, device=x0_hat.device)
        lbl = lbl.to(device=x0_hat.device, dtype=torch.long).reshape(-1)
        if lbl.numel() == 1:
            lbl = lbl.expand(P)                          # same class for all
        elif lbl.numel() != P:
            raise ValueError(
                f"metadata['label'] has {lbl.numel()} entries but there are "
                f"{P} particles"
            )

        idx = torch.arange(P, device=x0_hat.device)
        lp = logp[idx, lbl]                              # (P,)
        return self.cfg.guidance_scale * lp

    def degrade(self, x_clean, rng=None):
        with torch.no_grad():
            x = x_clean.unsqueeze(0) if x_clean.dim() == 3 else x_clean
            if self.cfg.clamp_classifier_input:
                x = x.clamp(-1, 1)
            pred = self.classifier(x).argmax(dim=1)      # (1,)
        y = pred.reshape(()).long()
        return y, {}

    def make_sigma_sq(self, a_bar_t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(a_bar_t) if torch.is_tensor(a_bar_t) \
            else torch.tensor(1.0)

    @staticmethod
    def class_from_label(label: int, device: str = "cpu") -> Tuple[torch.Tensor, dict]:
        return torch.tensor(int(label), device=device, dtype=torch.long), {}
    
    def postprocess_particles(self, x):
        return x.clamp(-1.0, 1.0)