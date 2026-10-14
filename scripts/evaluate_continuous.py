from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dasr_har.config import load_config
from dasr_har.data.dataset import load_split
from dasr_har.data.mhealth import MHealthAdapter
from dasr_har.data.pamap2 import Pamap2Adapter
from dasr_har.data.wisdm import WisdmAdapter
from dasr_har.evaluation.metrics import macro_f1
from dasr_har.evaluation.recovery import event_aligned_curve
from dasr_har.models.factory import build_model
from dasr_har.outages.streaming import CausalRuntime
from dasr_har.training.checkpoint import load_checkpoint
from dasr_har.utils.configuration import resolved_dataset_config
from dasr_har.utils.io import resolve_project_path
from dasr_har.utils.seed import set_seed


def load_recordings(config):
    if config["dataset"] == "mhealth":
        return MHealthAdapter(
            config["raw_dir"], config["retained_labels"], config["sample_rate"]
        ).load()
    if config["dataset"] == "pamap2":
        return Pamap2Adapter(
            config["raw_dir"], config["retained_labels"], config["sample_rate"]
        ).load()
    return WisdmAdapter(config["raw_dir"], config["sample_rate"]).load()


def normalize_recording(recording, normalizer):
    return {
        name: np.nan_to_num(
            (values - np.asarray(normalizer[name]["mean"], dtype=np.float32))
            / np.asarray(normalizer[name]["std"], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        for name, values in recording.streams.items()
    }


def controlled_masks(recording, streams, sample_rate, duration_seconds, spacing_seconds, seed):
    length = len(recording.labels)
    masks = {name: np.ones(length, dtype=bool) for name in streams}
    duration = max(1, int(round(duration_seconds * sample_rate)))
    spacing = max(duration + 1, int(round(spacing_seconds * sample_rate)))
    digest = hashlib.sha256(f"{seed}:{recording.subject_id}:{recording.session_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    for start in range(spacing, length - duration - 1, spacing):
        stream = streams[int(rng.integers(0, len(streams)))]
        masks[stream][start : start + duration] = False
    return masks


def infer_recording(model, recording, values, synthetic, sample_rate, chunk_steps, device, native_events):
    streams = list(values)
    runtime = CausalRuntime(model, streams, sample_rate)
    predictions = []
    gates = {name: [] for name in streams}
    states = {name: [] for name in streams}
    event_values = []
    stream_event_values = {name: [] for name in streams}
    for start in range(0, len(recording.labels), chunk_steps):
        stop = min(len(recording.labels), start + chunk_steps)
        x = {
            name: torch.as_tensor(values[name][None, start:stop], device=device)
            for name in streams
        }
        native = {
            name: torch.as_tensor(recording.native_mask[name][None, start:stop], device=device)
            for name in streams
        }
        event_mask = {
            name: torch.as_tensor(synthetic[name][None, start:stop], device=device)
            for name in streams
        }
        available = {
            name: native[name] & event_mask[name] for name in streams
        }
        output = runtime.step(
            x,
            available,
            native if native_events else event_mask,
            return_diagnostics=True,
        )
        predictions.append(output["step_logits"].argmax(dim=-1).squeeze(0).cpu().numpy())
        chunk_events = np.zeros(stop - start, dtype=bool)
        for name in streams:
            if "repair_gates" in output:
                gate_value = output["repair_gates"][name].squeeze(0).cpu().numpy()
            else:
                gate_value = np.zeros((stop - start, output["states"][name].shape[-1]))
            gates[name].append(gate_value)
            states[name].append(output["states"][name].squeeze(0).cpu().numpy())
            source_mask = recording.native_mask[name] if native_events else synthetic[name]
            mask = source_mask[start:stop]
            previous = source_mask[start - 1] if start else True
            chunk_event = np.zeros(len(mask), dtype=bool)
            if len(mask):
                chunk_event[0] = mask[0] and not previous
                chunk_event[1:] = mask[1:] & ~mask[:-1]
            chunk_events |= chunk_event
            stream_event_values[name].append(chunk_event)
        event_values.append(chunk_events)
    return (
        np.concatenate(predictions),
        np.concatenate(event_values),
        {name: np.concatenate(value) for name, value in gates.items()},
        {name: np.concatenate(value) for name, value in states.items()},
        {name: np.concatenate(value) for name, value in stream_event_values.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--duration-seconds", type=float, default=1.5)
    parser.add_argument("--spacing-seconds", type=float, default=20.0)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-offset-seconds", type=float, default=2.5)
    parser.add_argument("--native-events", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    config = resolved_dataset_config(
        load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    )
    checkpoint_path = resolve_project_path(
        f"results/checkpoints/{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(args.model, config, checkpoint.get("config", {}).get("model_config", {}))
    load_checkpoint(checkpoint_path, model)
    device = torch.device(args.device)
    model.to(device).eval()
    split = load_split(config["split_path"], args.fold)
    recordings = [item for item in load_recordings(config) if item.subject_id in split["test"]]
    with np.load(config["processed_path"], allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    label_map = {value: index for index, value in enumerate(metadata["raw_labels"])}
    clean_targets = []
    clean_predictions = []
    outage_labels = []
    outage_predictions = []
    outage_events = []
    streams = list(config["streams"])
    diagnostics = {}
    with torch.no_grad():
        for recording in recordings:
            values = normalize_recording(recording, checkpoint["normalizer"])
            clean_mask = {name: np.ones(len(recording.labels), dtype=bool) for name in streams}
            synthetic = (
                clean_mask
                if args.native_events
                else controlled_masks(
                    recording,
                    streams,
                    config["sample_rate"],
                    args.duration_seconds,
                    args.spacing_seconds,
                    args.seed,
                )
            )
            chunk_steps = max(1, int(round(args.chunk_seconds * config["sample_rate"])))
            clean_prediction, _, _, clean_states, _ = infer_recording(
                model,
                recording,
                values,
                clean_mask,
                config["sample_rate"],
                chunk_steps,
                device,
                args.native_events,
            )
            outage_prediction, events, gates, outage_states, stream_events = infer_recording(
                model,
                recording,
                values,
                synthetic,
                config["sample_rate"],
                chunk_steps,
                device,
                args.native_events,
            )
            labels = np.asarray([label_map[str(value)] for value in recording.labels], dtype=np.int64)
            clean_targets.extend(labels.tolist())
            clean_predictions.extend(clean_prediction.tolist())
            outage_labels.append(labels)
            outage_predictions.append(outage_prediction)
            outage_events.append(events)
            key = f"{recording.subject_id}__{recording.session_id}"
            diagnostics[f"labels__{key}"] = labels
            diagnostics[f"predictions__{key}"] = outage_prediction
            diagnostics[f"events__{key}"] = events
            for name in streams:
                diagnostics[f"gates__{name}__{key}"] = gates[name]
                diagnostics[f"events__{name}__{key}"] = stream_events[name]
                diagnostics[f"clean_states__{name}__{key}"] = clean_states[name]
                diagnostics[f"outage_states__{name}__{key}"] = outage_states[name]
    clean_score = macro_f1(np.asarray(clean_targets), np.asarray(clean_predictions), config["num_classes"])
    maximum_offset = int(round(args.maximum_offset_seconds * config["sample_rate"]))
    curve = event_aligned_curve(
        outage_labels,
        outage_predictions,
        outage_events,
        maximum_offset,
        clean_score,
    )
    stem = f"{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}_continuous"
    output_root = resolve_project_path("results/continuous")
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_root / f"{stem}_diagnostics.npz", **diagnostics)
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "clean_macro_f1": clean_score,
        "offsets": (curve.offsets / config["sample_rate"]).tolist(),
        "scores": curve.scores.tolist(),
        "event_counts": curve.event_counts.tolist(),
        "rda_event": curve.rda,
    }
    (output_root / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

