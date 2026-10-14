from __future__ import annotations

from typing import Any

import torch


class EventTracker:
    def __init__(self, streams: list[str], delta_t: float):
        self.streams = streams
        self.delta_t = delta_t
        self.previous: dict[str, torch.Tensor] = {}
        self.age: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.previous = {}
        self.age = {}

    def update(
        self, masks: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ages = {}
        reentries = {}
        durations = {}
        for name in self.streams:
            mask = masks[name].bool()
            batch, steps = mask.shape
            previous = self.previous.get(
                name, torch.ones(batch, dtype=torch.bool, device=mask.device)
            )
            age = self.age.get(
                name, torch.zeros(batch, dtype=torch.float32, device=mask.device)
            )
            age_values = []
            event_values = []
            duration_values = []
            for time in range(steps):
                current = mask[:, time]
                event = current & ~previous
                duration_values.append(torch.where(event, age, torch.zeros_like(age)))
                age = torch.where(current, torch.zeros_like(age), age + self.delta_t)
                age_values.append(age)
                event_values.append(event)
                previous = current
            ages[name] = torch.stack(age_values, dim=1)
            reentries[name] = torch.stack(event_values, dim=1)
            durations[name] = torch.stack(duration_values, dim=1)
            self.previous[name] = previous
            self.age[name] = age
        return ages, reentries, durations


class CausalRuntime:
    def __init__(self, model: torch.nn.Module, streams: list[str], sample_rate: float):
        self.model = model
        self.streams = streams
        self.tracker = EventTracker(streams, 1.0 / sample_rate)
        self.runtime: dict[str, Any] | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.runtime = None

    def step(
        self,
        x: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        event_mask: dict[str, torch.Tensor] | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        gap_age, reentry, duration = self.tracker.update(event_mask or available)
        for name in self.streams:
            reentry[name] = reentry[name] & available[name].bool()
            duration[name] = duration[name] * reentry[name].to(duration[name].dtype)
        runtime = self.runtime or {}
        output = self.model(
            x=x,
            available=available,
            gap_age=gap_age,
            reentry=reentry,
            duration=duration,
            return_diagnostics=return_diagnostics,
            initial_state=runtime.get("states"),
            initial_fused=runtime.get("fused"),
            encoder_context=runtime.get("encoder_context"),
            return_runtime_state=True,
        )
        self.runtime = _detach(output.pop("runtime_state"))
        return output


def _detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value
