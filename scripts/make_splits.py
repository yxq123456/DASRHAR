from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dasr_har.config import load_config
from dasr_har.data.dataset import load_split
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    split = load_split(resolve_project_path(config["split_path"]), args.fold)
    with np.load(resolve_project_path(config["processed_path"]), allow_pickle=False) as data:
        subjects = data["subject_id"]
        samples = data["sample_id"]
        result = {
            name: samples[np.isin(subjects, np.asarray(ids))].tolist()
            for name, ids in split.items()
        }
    output = args.output or resolve_project_path(
        f"manifests/subject_splits/{args.dataset}_fold{args.fold}_samples.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

