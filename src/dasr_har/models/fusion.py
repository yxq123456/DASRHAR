from __future__ import annotations

import torch
from torch import nn


class AvailabilityAwareFusion(nn.Module):
    def __init__(self, hidden_dim: int, duration_dim: int = 0):
        super().__init__()
        self.state_map = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.duration_map = (
            nn.Linear(duration_dim, hidden_dim, bias=False) if duration_dim > 0 else None
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.prior = nn.Parameter(torch.zeros(hidden_dim))

    def forward_step(
        self,
        states: torch.Tensor,
        available: torch.Tensor,
        previous: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.state_map(states)
        if self.duration_map is not None and durations is not None:
            scores = scores + self.duration_map(durations)
        scores = self.score(torch.tanh(scores + self.bias)).squeeze(-1)
        mask = available.bool()
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=-1)
        any_available = mask.any(dim=-1)
        weights = torch.where(any_available.unsqueeze(-1), weights, torch.zeros_like(weights))
        fused = torch.sum(weights.unsqueeze(-1) * states, dim=1)
        fused = torch.where(any_available.unsqueeze(-1), fused, previous)
        return fused, weights


class TemporalHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        step_logits = self.classifier(states)
        logits = self.classifier(states.mean(dim=1))
        return logits, step_logits

