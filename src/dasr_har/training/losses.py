from __future__ import annotations

import torch
from torch.nn import functional as F


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).to(torch.float64).clamp(min=1)
    inverse = counts.reciprocal()
    return (inverse / inverse.mean()).to(torch.float32)


def classification_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: torch.Tensor,
    auxiliary_weight: float = 0.1,
) -> torch.Tensor:
    loss = F.cross_entropy(output["logits"], target, weight=weights)
    if "aux_loss" in output:
        loss = loss + auxiliary_weight * output["aux_loss"]
    return loss

