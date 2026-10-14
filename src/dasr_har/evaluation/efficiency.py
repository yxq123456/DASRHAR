from __future__ import annotations

import statistics
import time
from typing import Any

import torch


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    ) + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())


def profile_flops(model: torch.nn.Module, batch: dict[str, Any]) -> int:
    with torch.profiler.profile(with_flops=True) as profiler:
        model(**batch)
    return int(sum(event.flops for event in profiler.key_averages() if event.flops))


def latency(
    model: torch.nn.Module,
    batch: dict[str, Any],
    warmup: int = 50,
    runs: int = 200,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values = []
        for _ in range(runs):
            started = time.perf_counter()
            model(**batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "q1_ms": ordered[int(0.25 * (len(ordered) - 1))],
        "q3_ms": ordered[int(0.75 * (len(ordered) - 1))],
    }
