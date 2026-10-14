from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dasr_har.utils.io import resolve_project_path


def geometry(paths: list[Path], output: Path) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    frame = pd.DataFrame(records)
    summary = frame.groupby(["method", "condition"], as_index=False)["macro_f1"].mean()
    pivot = summary.pivot(index="condition", columns="method", values="macro_f1")
    axis = pivot.plot(kind="bar", figsize=(10, 4.8), width=0.82)
    axis.set_ylabel("Macro-F1")
    axis.set_xlabel("")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=300)


def recovery(paths: list[Path], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        axis.plot(value["offsets"], value["scores"], label=value["model"], linewidth=2)
    axis.set_xlabel("Time after re-entry (s)")
    axis.set_ylabel("Macro-F1")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=300)


def mechanism(paths: list[Path], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(value["state_gap"])
        for stream, group in frame.groupby("stream"):
            axis.plot(group["offset"], group["mean"], label=stream, linewidth=2)
    axis.set_xlabel("Offset after re-entry")
    axis.set_ylabel("StateGap")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["geometry", "recovery", "mechanism"], required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [resolve_project_path(path) for path in args.inputs]
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    {"geometry": geometry, "recovery": recovery, "mechanism": mechanism}[args.kind](
        paths, output
    )


if __name__ == "__main__":
    main()
