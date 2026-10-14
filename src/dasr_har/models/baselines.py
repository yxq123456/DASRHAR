from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import CausalStreamEncoder, RecurrentStack
from .fusion import AvailabilityAwareFusion, TemporalHead


class MultiBranchGRU(nn.Module):
    def __init__(
        self,
        stream_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        sample_rate: float = 1.0,
        strategy: str = "hold",
        encoder_kernels: tuple[int, int] = (5, 3),
        learned_fusion: bool = True,
    ):
        super().__init__()
        self.streams = list(stream_dims)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.delta_t = 1.0 / sample_rate
        self.strategy = strategy
        self.learned_fusion = learned_fusion
        self.encoders = nn.ModuleDict(
            {
                name: CausalStreamEncoder(dimension, hidden_dim, encoder_kernels, dropout)
                for name, dimension in stream_dims.items()
            }
        )
        self.recurrent = nn.ModuleDict(
            {
                name: RecurrentStack(hidden_dim, num_layers, dropout)
                for name in self.streams
            }
        )
        self.state_priors = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(num_layers, hidden_dim))
                for name in self.streams
            }
        )
        self.decay_logits = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(num_layers, hidden_dim))
                for name in self.streams
            }
        )
        self.fusion = AvailabilityAwareFusion(hidden_dim)
        self.head = TemporalHead(hidden_dim, num_classes)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        reentry: dict[str, torch.Tensor] | None = None,
        return_diagnostics: bool = False,
        initial_state: dict[str, torch.Tensor] | None = None,
        initial_fused: torch.Tensor | None = None,
        encoder_context: dict[str, dict[str, torch.Tensor]] | None = None,
        return_runtime_state: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        first = x[self.streams[0]]
        batch_size, steps = first.shape[:2]
        if reentry is None:
            reentry = {}
            for name in self.streams:
                mask = available[name].bool()
                events = torch.zeros_like(mask)
                events[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
                reentry[name] = events
        else:
            reentry = {
                name: reentry[name].bool() & available[name].bool() for name in self.streams
            }
        encoded = {}
        next_context = {}
        encoder_context = encoder_context or {}
        for name in self.streams:
            previous = encoder_context.get(name)
            if previous is None:
                combined_x = x[name]
                combined_available = available[name]
            else:
                combined_x = torch.cat((previous["x"], x[name]), dim=1)
                combined_available = torch.cat(
                    (previous["available"], available[name]), dim=1
                )
            encoded[name] = self.encoders[name](combined_x, combined_available)[:, -steps:]
            context_length = int(getattr(self.encoders[name], "context_length", 0))
            next_context[name] = {
                "x": combined_x[:, -context_length:] if context_length else combined_x[:, :0],
                "available": (
                    combined_available[:, -context_length:]
                    if context_length
                    else combined_available[:, :0]
                ),
            }
        states = (
            {name: initial_state[name] for name in self.streams}
            if initial_state is not None
            else {
                name: self.state_priors[name]
                .unsqueeze(1)
                .expand(-1, batch_size, -1)
                .clone()
                for name in self.streams
            }
        )
        fused_previous = (
            initial_fused
            if initial_fused is not None
            else self.fusion.prior.unsqueeze(0).expand(batch_size, -1)
        )
        fused_sequence = []
        state_sequences = {name: [] for name in self.streams}
        weight_sequence = []
        for time in range(steps):
            for name in self.streams:
                observed = available[name][:, time].bool()
                event = reentry[name][:, time].bool()
                predecessor = states[name]
                if self.strategy == "reset":
                    zero = torch.zeros_like(predecessor)
                    predecessor = torch.where(event.view(1, -1, 1), zero, predecessor)
                updated = self.recurrent[name].forward_step(
                    predecessor, encoded[name][:, time]
                )
                if self.strategy == "update":
                    missing = updated
                elif self.strategy == "decay":
                    rate = F.softplus(self.decay_logits[name]).unsqueeze(1)
                    factor = torch.exp(-rate * self.delta_t)
                    prior = self.state_priors[name].unsqueeze(1)
                    missing = factor * states[name] + (1.0 - factor) * prior
                else:
                    missing = states[name]
                states[name] = torch.where(observed.view(1, -1, 1), updated, missing)
            top_states = torch.stack([states[name][-1] for name in self.streams], dim=1)
            mask = torch.stack(
                [available[name][:, time].bool() for name in self.streams], dim=1
            )
            if self.learned_fusion:
                fused_previous, weights = self.fusion.forward_step(
                    top_states, mask, fused_previous
                )
            else:
                weights = mask.to(top_states.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
                candidate_fused = (weights.unsqueeze(-1) * top_states).sum(dim=1)
                fused_previous = torch.where(
                    mask.any(dim=1, keepdim=True), candidate_fused, fused_previous
                )
            fused_sequence.append(fused_previous)
            weight_sequence.append(weights)
            if return_diagnostics:
                for name in self.streams:
                    state_sequences[name].append(states[name][-1])
        fused = torch.stack(fused_sequence, dim=1)
        logits, step_logits = self.head(fused)
        output: dict[str, Any] = {"logits": logits, "step_logits": step_logits}
        if return_diagnostics:
            output["states"] = {
                name: torch.stack(values, dim=1) for name, values in state_sequences.items()
            }
            output["fusion_weights"] = torch.stack(weight_sequence, dim=1)
        if return_runtime_state:
            output["runtime_state"] = {
                "states": states,
                "fused": fused_previous,
                "encoder_context": next_context,
            }
        return output


class MaskedEarlyFusion(nn.Module):
    def __init__(
        self,
        stream_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        encoder_kernels: tuple[int, int] = (5, 3),
    ):
        super().__init__()
        self.streams = list(stream_dims)
        input_dim = sum(stream_dims.values()) + len(stream_dims) - 1
        self.encoder = CausalStreamEncoder(input_dim, hidden_dim, encoder_kernels, dropout)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = TemporalHead(hidden_dim, num_classes)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        values = []
        masks = []
        for name in self.streams:
            mask = available[name].to(x[name].dtype).unsqueeze(-1)
            values.append(x[name] * mask)
            masks.append(mask)
        concatenated = torch.cat(values + masks, dim=-1)
        any_available = torch.stack([available[name] for name in self.streams], dim=-1).any(-1)
        encoded = self.encoder(concatenated, any_available)
        states, _ = self.gru(encoded)
        logits, step_logits = self.head(states)
        return {"logits": logits, "step_logits": step_logits}


class BidirectionalImputer(nn.Module):
    def __init__(self, total_dim: int, hidden_dim: int):
        super().__init__()
        self.forward_cell = nn.GRUCell(total_dim * 2, hidden_dim)
        self.backward_cell = nn.GRUCell(total_dim * 2, hidden_dim)
        self.forward_readout = nn.Linear(hidden_dim, total_dim)
        self.backward_readout = nn.Linear(hidden_dim, total_dim)

    def _direction(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        cell: nn.GRUCell,
        readout: nn.Linear,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = values.shape
        state = values.new_zeros(batch, cell.hidden_size)
        completed = [values.new_zeros(batch, values.shape[-1]) for _ in range(steps)]
        losses = []
        indices = range(steps - 1, -1, -1) if reverse else range(steps)
        for time in indices:
            prediction = readout(state)
            observed = mask[:, time]
            complete = torch.where(observed, values[:, time], prediction)
            losses.append((prediction - values[:, time]).abs() * observed.to(values.dtype))
            state = cell(torch.cat((complete, observed.to(values.dtype)), dim=-1), state)
            completed[time] = complete
        return torch.stack(completed, dim=1), torch.stack(losses, dim=1).mean()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward, forward_loss = self._direction(
            values, mask, self.forward_cell, self.forward_readout, False
        )
        backward, backward_loss = self._direction(
            values, mask, self.backward_cell, self.backward_readout, True
        )
        imputed = torch.where(mask, values, 0.5 * (forward + backward))
        consistency = ((forward - backward).abs() * (~mask).to(values.dtype)).mean()
        return imputed, forward_loss + backward_loss + consistency


class BRITSClassifier(nn.Module):
    def __init__(
        self,
        stream_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        sample_rate: float,
        encoder_kernels: tuple[int, int],
    ):
        super().__init__()
        self.streams = list(stream_dims)
        self.stream_dims = stream_dims
        self.imputer = BidirectionalImputer(sum(stream_dims.values()), hidden_dim)
        self.classifier = MultiBranchGRU(
            stream_dims,
            num_classes,
            hidden_dim,
            num_layers,
            dropout,
            sample_rate,
            "hold",
            encoder_kernels,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = torch.cat([x[name] for name in self.streams], dim=-1)
        channel_masks = torch.cat(
            [
                available[name].unsqueeze(-1).expand(-1, -1, self.stream_dims[name])
                for name in self.streams
            ],
            dim=-1,
        ).bool()
        imputed, auxiliary = self.imputer(values, channel_masks)
        split = torch.split(imputed, [self.stream_dims[name] for name in self.streams], dim=-1)
        imputed_streams = dict(zip(self.streams, split))
        full = {
            name: torch.ones_like(available[name], dtype=torch.bool) for name in self.streams
        }
        output = self.classifier(imputed_streams, full, **kwargs)
        output["aux_loss"] = auxiliary
        return output


class CrossSensorReconstructionClassifier(nn.Module):
    def __init__(
        self,
        stream_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        sample_rate: float,
        encoder_kernels: tuple[int, int],
    ):
        super().__init__()
        self.streams = list(stream_dims)
        self.stream_dims = stream_dims
        total = sum(stream_dims.values())
        self.reconstructor = nn.Sequential(
            nn.Linear(total + len(self.streams), hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, total),
        )
        self.classifier = MultiBranchGRU(
            stream_dims,
            num_classes,
            hidden_dim,
            num_layers,
            dropout,
            sample_rate,
            "hold",
            encoder_kernels,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, Any]:
        masked = []
        masks = []
        channel_masks = []
        for name in self.streams:
            mask = available[name].to(x[name].dtype).unsqueeze(-1)
            masked.append(x[name] * mask)
            masks.append(mask)
            channel_masks.append(mask.expand(-1, -1, self.stream_dims[name]))
        values = torch.cat([x[name] for name in self.streams], dim=-1)
        prediction = self.reconstructor(torch.cat(masked + masks, dim=-1))
        channel_mask = torch.cat(channel_masks, dim=-1).bool()
        completed = torch.where(channel_mask, values, prediction)
        missing = (~channel_mask).to(values.dtype)
        auxiliary = (((prediction - values) ** 2) * missing).sum() / missing.sum().clamp(min=1.0)
        split = torch.split(completed, [self.stream_dims[name] for name in self.streams], dim=-1)
        streams = dict(zip(self.streams, split))
        full = {
            name: torch.ones_like(available[name], dtype=torch.bool) for name in self.streams
        }
        output = self.classifier(streams, full, **kwargs)
        output["aux_loss"] = auxiliary
        return output
