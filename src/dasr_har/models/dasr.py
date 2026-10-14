from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import CausalStreamEncoder, RecurrentStack
from .fusion import AvailabilityAwareFusion, TemporalHead


class DASR(nn.Module):
    def __init__(
        self,
        stream_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int,
        duration_dim: int,
        repair_width: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        sample_rate: float = 1.0,
        encoder_kernels: tuple[int, int] = (5, 3),
        use_decay: bool = True,
        use_duration: bool = True,
        use_context: bool = True,
        adaptive_gate: bool = True,
    ):
        super().__init__()
        self.streams = list(stream_dims)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.delta_t = 1.0 / sample_rate
        self.use_decay = use_decay
        self.use_duration = use_duration
        self.use_context = use_context
        self.adaptive_gate = adaptive_gate
        self.encoders = nn.ModuleDict(
            {
                name: CausalStreamEncoder(
                    dimension, hidden_dim, encoder_kernels, dropout
                )
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
        self.context_priors = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden_dim)) for name in self.streams}
        )
        self.decay_logits = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(num_layers, hidden_dim))
                for name in self.streams
            }
        )
        self.duration_encoder = nn.Sequential(
            nn.Linear(2, duration_dim),
            nn.GELU(),
            nn.Linear(duration_dim, duration_dim),
        )
        self.duration_projection = nn.Linear(duration_dim, hidden_dim)
        self.repair_core = nn.Sequential(
            nn.Linear(4, repair_width),
            nn.GELU(),
            nn.Linear(repair_width, repair_width),
            nn.GELU(),
        )
        self.candidate_head = nn.Linear(repair_width, 1)
        self.gate_head = nn.Linear(repair_width, 1)
        self.alpha_candidate = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(hidden_dim)) for name in self.streams}
        )
        self.eta_candidate = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden_dim)) for name in self.streams}
        )
        self.alpha_gate = nn.ParameterDict(
            {name: nn.Parameter(torch.ones(hidden_dim)) for name in self.streams}
        )
        self.eta_gate = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden_dim)) for name in self.streams}
        )
        self.fusion = AvailabilityAwareFusion(hidden_dim, duration_dim)
        self.head = TemporalHead(hidden_dim, num_classes)

    def _duration_codes(
        self,
        duration: dict[str, torch.Tensor],
        reentry: dict[str, torch.Tensor],
        window_duration: float,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.streams:
            ratio = duration[name].to(torch.float32) / max(window_duration, 1e-8)
            features = torch.stack((ratio, torch.log1p(ratio)), dim=-1)
            code = self.duration_encoder(features)
            event = reentry[name].to(code.dtype).unsqueeze(-1)
            result[name] = code * event if self.use_duration else torch.zeros_like(code)
        return result

    def _repair(
        self,
        name: str,
        provisional: torch.Tensor,
        observation: torch.Tensor,
        context: torch.Tensor,
        duration_code: torch.Tensor,
        event: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        duration_hidden = self.duration_projection(duration_code)
        descriptor = torch.stack((provisional, observation, context, duration_hidden), dim=-1)
        embedded = self.repair_core(descriptor)
        candidate_logit = self.candidate_head(embedded).squeeze(-1)
        gate_logit = self.gate_head(embedded).squeeze(-1)
        candidate = torch.tanh(
            self.alpha_candidate[name] * candidate_logit + self.eta_candidate[name]
        )
        if self.adaptive_gate:
            gate = torch.sigmoid(self.alpha_gate[name] * gate_logit + self.eta_gate[name])
        else:
            gate = torch.full_like(candidate, 0.5)
        gate = gate * event.to(gate.dtype).unsqueeze(-1)
        repaired = (1.0 - gate) * provisional + gate * candidate
        return repaired, gate

    def forward(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        gap_age: dict[str, torch.Tensor] | None = None,
        reentry: dict[str, torch.Tensor] | None = None,
        duration: dict[str, torch.Tensor] | None = None,
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
                values = available[name].bool()
                events = torch.zeros_like(values)
                events[:, 1:] = values[:, 1:] & ~values[:, :-1]
                reentry[name] = events
        else:
            reentry = {
                name: reentry[name].bool() & available[name].bool() for name in self.streams
            }
        if gap_age is None:
            gap_age = {
                name: torch.zeros_like(available[name], dtype=first.dtype)
                for name in self.streams
            }
        if duration is None:
            duration = {}
            for name in self.streams:
                values = torch.zeros_like(gap_age[name], dtype=first.dtype)
                values[:, 1:] = reentry[name][:, 1:] * gap_age[name][:, :-1]
                duration[name] = values
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
        duration_codes = self._duration_codes(
            duration, reentry, steps * self.delta_t
        )
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
        gate_sequences = {name: [] for name in self.streams}
        weight_sequence = []
        for time in range(steps):
            provisional = {}
            for name in self.streams:
                observed = available[name][:, time].bool()
                event = reentry[name][:, time].bool()
                updated = self.recurrent[name].forward_step(
                    states[name], encoded[name][:, time]
                )
                if self.use_decay:
                    rate = F.softplus(self.decay_logits[name]).unsqueeze(1)
                    factor = torch.exp(-rate * self.delta_t)
                    prior = self.state_priors[name].unsqueeze(1)
                    missing_state = factor * states[name] + (1.0 - factor) * prior
                else:
                    missing_state = states[name]
                candidate = torch.where(observed.view(1, -1, 1), updated, missing_state)
                candidate = torch.where(event.view(1, -1, 1), states[name], candidate)
                provisional[name] = candidate
            committed = {}
            gates = {}
            for name in self.streams:
                event = reentry[name][:, time].bool()
                anchors = []
                anchor_masks = []
                for other in self.streams:
                    if other == name:
                        continue
                    anchor = available[other][:, time].bool() & ~reentry[other][:, time].bool()
                    anchors.append(provisional[other][-1])
                    anchor_masks.append(anchor)
                if anchors:
                    anchor_states = torch.stack(anchors, dim=1)
                    anchor_mask = torch.stack(anchor_masks, dim=1)
                    count = anchor_mask.sum(dim=1, keepdim=True)
                    context = (
                        anchor_states * anchor_mask.to(anchor_states.dtype).unsqueeze(-1)
                    ).sum(dim=1) / count.clamp(min=1).to(anchor_states.dtype)
                    fallback = self.context_priors[name].unsqueeze(0).expand(batch_size, -1)
                    context = torch.where(count > 0, context, fallback)
                else:
                    context = self.context_priors[name].unsqueeze(0).expand(batch_size, -1)
                if not self.use_context:
                    context = torch.zeros_like(context)
                repaired, gate = self._repair(
                    name,
                    provisional[name][-1],
                    encoded[name][:, time],
                    context,
                    duration_codes[name][:, time],
                    event,
                )
                returning_update = self.recurrent[name].forward_step(
                    states[name], encoded[name][:, time], top_predecessor=repaired
                )
                committed[name] = torch.where(
                    event.view(1, -1, 1), returning_update, provisional[name]
                )
                gates[name] = gate
            states = committed
            top_states = torch.stack([states[name][-1] for name in self.streams], dim=1)
            availability = torch.stack(
                [available[name][:, time].bool() for name in self.streams], dim=1
            )
            step_durations = torch.stack(
                [duration_codes[name][:, time] for name in self.streams], dim=1
            )
            fused_previous, weights = self.fusion.forward_step(
                top_states, availability, fused_previous, step_durations
            )
            fused_sequence.append(fused_previous)
            weight_sequence.append(weights)
            if return_diagnostics:
                for name in self.streams:
                    state_sequences[name].append(states[name][-1])
                    gate_sequences[name].append(gates[name])
        fused = torch.stack(fused_sequence, dim=1)
        logits, step_logits = self.head(fused)
        output: dict[str, Any] = {"logits": logits, "step_logits": step_logits}
        if return_diagnostics:
            output["states"] = {
                name: torch.stack(values, dim=1) for name, values in state_sequences.items()
            }
            output["repair_gates"] = {
                name: torch.stack(values, dim=1) for name, values in gate_sequences.items()
            }
            output["fusion_weights"] = torch.stack(weight_sequence, dim=1)
        if return_runtime_state:
            output["runtime_state"] = {
                "states": states,
                "fused": fused_previous,
                "encoder_context": next_context,
            }
        return output
