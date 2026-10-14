from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dasr_har.evaluation.metrics import state_gap
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--maximum-offset", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gaps = defaultdict(list)
    gates = defaultdict(list)
    with np.load(resolve_project_path(args.diagnostics), allow_pickle=False) as data:
        event_keys = [key for key in data.files if key.startswith("events__") and key.count("__") >= 3]
        for event_key in event_keys:
            _, stream, *recording_parts = event_key.split("__")
            recording = "__".join(recording_parts)
            clean_key = f"clean_states__{stream}__{recording}"
            outage_key = f"outage_states__{stream}__{recording}"
            gate_key = f"gates__{stream}__{recording}"
            if clean_key not in data or outage_key not in data:
                continue
            clean = data[clean_key]
            outage = data[outage_key]
            gap = state_gap(clean, outage)
            for event in np.flatnonzero(data[event_key]):
                gates[stream].append(float(data[gate_key][event].mean()))
                for offset in range(args.maximum_offset + 1):
                    if event + offset >= len(gap):
                        break
                    gaps[(stream, offset)].append(float(gap[event + offset]))
    result = {
        "state_gap": [
            {
                "stream": stream,
                "offset": offset,
                "mean": float(np.mean(values)),
                "events": len(values),
            }
            for (stream, offset), values in sorted(gaps.items())
        ],
        "repair_gate": [
            {
                "stream": stream,
                "mean": float(np.mean(values)),
                "events": len(values),
            }
            for stream, values in sorted(gates.items())
        ],
    }
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
