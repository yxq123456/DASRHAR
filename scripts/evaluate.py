from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dasr_har.config import load_config
from dasr_har.data.dataset import WindowDataset, collate_windows, load_split
from dasr_har.evaluation.metrics import accuracy, macro_f1
from dasr_har.models.factory import build_model
from dasr_har.outages.manifest import Manifest
from dasr_har.training.checkpoint import load_checkpoint
from dasr_har.training.trainer import move_to_device
from dasr_har.utils.configuration import TRAINING_DEFAULTS, resolved_dataset_config
from dasr_har.utils.io import resolve_project_path
from dasr_har.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    dataset_config = resolved_dataset_config(
        load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    )
    checkpoint_path = resolve_project_path(
        f"results/checkpoints/{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}.pt"
    )
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        args.model,
        dataset_config,
        raw_checkpoint.get("config", {}).get("model_config", {}),
    )
    load_checkpoint(checkpoint_path, model)
    device = torch.device(args.device)
    model.to(device).eval()
    manifest = None
    if args.condition != "clean":
        manifest_path = args.manifest or resolve_project_path(
            f"manifests/outage_masks/{args.dataset}/fold{args.fold}/{args.split}_{args.condition}.jsonl"
        )
        manifest = Manifest.load(manifest_path)
    split = load_split(dataset_config["split_path"], args.fold)
    transform = manifest.transform(list(dataset_config["streams"])) if manifest else None
    dataset = WindowDataset(
        dataset_config["processed_path"],
        split[args.split],
        raw_checkpoint["normalizer"],
        dataset_config["sample_rate"],
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=TRAINING_DEFAULTS[args.dataset]["batch_size"],
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_windows,
    )
    rows = []
    targets = []
    predictions = []
    diagnostic_values: dict[str, list[np.ndarray]] = {}
    checksum = manifest.checksum() if manifest else "clean"
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            output = model(
                x=batch["x"],
                available=batch["available"],
                gap_age=batch["gap_age"],
                reentry=batch["reentry"],
                duration=batch["duration"],
                return_diagnostics=args.diagnostics,
            )
            probability = torch.softmax(output["logits"], dim=-1).cpu().numpy()
            predicted = probability.argmax(axis=-1)
            labels = batch["label"].cpu().numpy()
            targets.extend(labels.tolist())
            predictions.extend(predicted.tolist())
            for index, sample_id in enumerate(raw_batch["sample_id"]):
                entry = manifest.by_sample.get(sample_id) if manifest else None
                rows.append(
                    {
                        "dataset": args.dataset,
                        "fold": args.fold,
                        "seed": args.seed,
                        "subject_id": raw_batch["subject_id"][index],
                        "sample_id": sample_id,
                        "label": int(labels[index]),
                        "prediction": int(predicted[index]),
                        "probabilities": json.dumps(probability[index].tolist()),
                        "method": args.model,
                        "checkpoint_id": checkpoint_path.stem,
                        "condition": args.condition,
                        "pattern": entry.pattern if entry else "clean",
                        "missing_rate": entry.missing_rate if entry else 0.0,
                        "missing_streams": json.dumps(list(entry.segments) if entry else []),
                        "duration": entry.duration_seconds if entry else 0.0,
                        "recovery_horizon": entry.recovery_horizon_seconds if entry else 0.0,
                        "manifest_hash": checksum,
                    }
                )
            if args.diagnostics:
                for group in ["states", "repair_gates", "fusion_weights", "step_logits"]:
                    if group not in output:
                        continue
                    value = output[group]
                    if isinstance(value, dict):
                        for stream, tensor in value.items():
                            diagnostic_values.setdefault(f"{group}__{stream}", []).append(
                                tensor.cpu().numpy()
                            )
                    else:
                        diagnostic_values.setdefault(group, []).append(value.cpu().numpy())
    output_root = resolve_project_path("results/raw_predictions")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}_{args.condition}"
    with (output_root / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "dataset": args.dataset,
        "method": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "condition": args.condition,
        "macro_f1": macro_f1(np.asarray(targets), np.asarray(predictions), dataset_config["num_classes"]),
        "accuracy": accuracy(np.asarray(targets), np.asarray(predictions)),
        "samples": len(targets),
    }
    aggregate_root = resolve_project_path("results/aggregated")
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (aggregate_root / f"{stem}.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    if args.diagnostics:
        diagnostics_root = resolve_project_path("results/diagnostics")
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.concatenate(value, axis=0) for key, value in diagnostic_values.items()
        }
        arrays["sample_id"] = np.asarray([row["sample_id"] for row in rows])
        np.savez_compressed(diagnostics_root / f"{stem}.npz", **arrays)


if __name__ == "__main__":
    main()

