from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dasr_har.config import load_config
from dasr_har.utils.configuration import resolved_dataset_config
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = resolved_dataset_config(
        load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    )
    rows = []
    with np.load(config["processed_path"], allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for subject in sorted(np.unique(data["subject_id"]), key=lambda value: int(value)):
            selected = data["subject_id"] == subject
            for label in np.unique(data["label"][selected]):
                label_selected = selected & (data["label"] == label)
                row = {
                    "dataset": args.dataset,
                    "subject_id": subject,
                    "class_index": int(label),
                    "raw_label": metadata["raw_labels"][int(label)],
                    "windows": int(label_selected.sum()),
                }
                for stream in metadata["streams"]:
                    row[f"native_missing__{stream}"] = float(
                        1.0 - data[f"native__{stream}"][label_selected].mean()
                    )
                rows.append(row)
    output = args.output or resolve_project_path(f"results/audits/{args.dataset}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()

