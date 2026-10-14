from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from dasr_har.evaluation.metrics import macro_f1

from .checkpoint import save_checkpoint
from .losses import classification_loss


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        weights: torch.Tensor,
        device: torch.device,
        gradient_clip: float = 1.0,
        auxiliary_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weights = weights.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.auxiliary_weight = auxiliary_weight

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(
                x=batch["x"],
                available=batch["available"],
                gap_age=batch["gap_age"],
                reentry=batch["reentry"],
                duration=batch["duration"],
            )
            loss = classification_loss(
                output, batch["label"], self.weights, self.auxiliary_weight
            )
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        labels = []
        predictions = []
        losses = []
        for batch in loader:
            batch = move_to_device(batch, self.device)
            output = self.model(
                x=batch["x"],
                available=batch["available"],
                gap_age=batch["gap_age"],
                reentry=batch["reentry"],
                duration=batch["duration"],
            )
            loss = classification_loss(
                output, batch["label"], self.weights, self.auxiliary_weight
            )
            labels.extend(batch["label"].cpu().tolist())
            predictions.extend(output["logits"].argmax(dim=-1).cpu().tolist())
            losses.append(float(loss.cpu()))
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "macro_f1": macro_f1(np.asarray(labels), np.asarray(predictions)),
        }

    def fit(
        self,
        train_loader,
        validation_loaders: dict[str, Any],
        epochs: int,
        patience: int,
        checkpoint_path: str | Path,
        log_path: str | Path,
        config: dict[str, Any],
        normalizer: dict[str, Any],
        clean_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        history = []
        best_score = -float("inf")
        stale = 0
        output_log = Path(log_path)
        output_log.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_loss = self.train_epoch(train_loader)
            validation = {
                name: self.evaluate(loader) for name, loader in validation_loaders.items()
            }
            clean = validation.get("clean", next(iter(validation.values())))["macro_f1"]
            outage_values = [
                value["macro_f1"] for name, value in validation.items() if name != "clean"
            ]
            robust = float(np.mean(outage_values)) if outage_values else clean
            score = clean_weight * clean + (1.0 - clean_weight) * robust
            elapsed = time.perf_counter() - started
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation,
                "score": score,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            }
            history.append(record)
            with output_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if self.scheduler is not None:
                try:
                    self.scheduler.step(score)
                except TypeError:
                    self.scheduler.step()
            if score > best_score:
                best_score = score
                stale = 0
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    score,
                    config,
                    normalizer,
                )
            else:
                stale += 1
                if stale >= patience:
                    break
        return history

