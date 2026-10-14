from __future__ import annotations

import argparse
from pathlib import Path

from dasr_har.config import load_config
from dasr_har.data.mhealth import MHealthAdapter
from dasr_har.data.pamap2 import Pamap2Adapter
from dasr_har.data.preprocessing import save_windows
from dasr_har.data.windowing import window_recordings
from dasr_har.data.wisdm import WisdmAdapter
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path(f"configs/datasets/{args.dataset}.yaml")
    config = load_config(resolve_project_path(config_path))
    raw_dir = resolve_project_path(config["raw_dir"])
    if args.dataset == "mhealth":
        adapter = MHealthAdapter(raw_dir, config["retained_labels"], config["sample_rate"])
    elif args.dataset == "pamap2":
        adapter = Pamap2Adapter(raw_dir, config["retained_labels"], config["sample_rate"])
    else:
        adapter = WisdmAdapter(raw_dir, config["sample_rate"])
    recordings = adapter.load()
    windows = window_recordings(
        recordings,
        int(round(config["window_seconds"] * config["sample_rate"])),
        int(round(config["step_seconds"] * config["sample_rate"])),
        float(config["label_purity"]),
        config.get("max_windows_per_subject_activity"),
    )
    save_windows(windows, resolve_project_path(config["processed_path"]))


if __name__ == "__main__":
    main()

