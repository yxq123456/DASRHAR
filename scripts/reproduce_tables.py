from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dasr_har.evaluation.metrics import mean_robust
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/aggregated"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/main_results.csv"))
    args = parser.parse_args()
    input_root = resolve_project_path(args.input)
    records = []
    for path in sorted(input_root.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"No aggregate result files found under {input_root}")
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["dataset", "method", "condition"], as_index=False)["macro_f1"]
        .mean()
        .pivot(index=["dataset", "method"], columns="condition", values="macro_f1")
        .reset_index()
    )
    robust = []
    for _, row in summary.iterrows():
        values = row.to_dict()
        if all(name in values and pd.notna(values[name]) for name in [
            "point_30", "burst_30", "multi_30", "periodic_30", "burst_50"
        ]):
            average, worst = mean_robust(values)
        else:
            average, worst = float("nan"), float("nan")
        robust.append((average, worst))
    summary["mean_robust"] = [value[0] for value in robust]
    summary["worst"] = [value[1] for value in robust]
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)


if __name__ == "__main__":
    main()
