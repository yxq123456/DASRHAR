from __future__ import annotations

import argparse
import json

import torch

from dasr_har.config import load_config
from dasr_har.evaluation.efficiency import latency, model_size_bytes, parameter_count, profile_flops
from dasr_har.models.factory import build_model
from dasr_har.training.trainer import move_to_device
from dasr_har.utils.configuration import resolved_dataset_config
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], default="mhealth")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/efficiency/profile.json")
    args = parser.parse_args()
    config = resolved_dataset_config(
        load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    )
    model = build_model(args.model, config)
    device = torch.device(args.device)
    model.to(device).eval()
    steps = int(round(config["window_seconds"] * config["sample_rate"]))
    x = {
        name: torch.zeros(args.batch_size, steps, dimension)
        for name, dimension in config["streams"].items()
    }
    available = {
        name: torch.ones(args.batch_size, steps, dtype=torch.bool) for name in config["streams"]
    }
    scalar = {
        name: torch.zeros(args.batch_size, steps) for name in config["streams"]
    }
    events = {
        name: torch.zeros(args.batch_size, steps, dtype=torch.bool) for name in config["streams"]
    }
    batch = move_to_device(
        {
            "x": x,
            "available": available,
            "gap_age": scalar,
            "duration": scalar,
            "reentry": events,
        },
        device,
    )
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "batch_size": args.batch_size,
        "parameters": parameter_count(model),
        "model_size_bytes": model_size_bytes(model),
        "flops": profile_flops(model, batch),
        **latency(model, batch, 50, 200),
    }
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
