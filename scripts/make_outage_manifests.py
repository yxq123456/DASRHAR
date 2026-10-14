from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from dasr_har.config import load_config
from dasr_har.data.dataset import load_split
from dasr_har.outages.generator import generate_mask, overlapping_burst_masks
from dasr_har.outages.manifest import Manifest, ManifestEntry
from dasr_har.outages.patterns import mask_to_segments
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument(
        "--pattern", choices=["point", "single_burst", "multi_burst", "periodic"], default="single_burst"
    )
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--duration-ratio", type=float)
    parser.add_argument("--horizon-ratio", type=float)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--condition")
    parser.add_argument("--interrupted-streams")
    parser.add_argument("--overlap", type=float, choices=[0.0, 0.5, 1.0])
    parser.add_argument("--no-anchor", choices=["others_offline", "simultaneous_return"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    split = load_split(resolve_project_path(config["split_path"]), args.fold)
    streams = list(config["streams"])
    entries = []
    with np.load(resolve_project_path(config["processed_path"]), allow_pickle=False) as data:
        selected = np.flatnonzero(np.isin(data["subject_id"], np.asarray(split[args.split])))
        length = data[f"x__{streams[0]}"].shape[1]
        for index in selected:
            sample_id = str(data["sample_id"][index])
            digest = hashlib.sha256(f"{args.seed}:{sample_id}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            selected_streams = (
                args.interrupted_streams.split(",")
                if args.interrupted_streams
                else [streams[int(rng.integers(0, len(streams)))]]
            )
            horizon = None
            rate = args.missing_rate
            if args.duration_ratio is not None:
                rate = args.duration_ratio
            if args.horizon_ratio is not None:
                horizon = int(round(args.horizon_ratio * length))
            if len(selected_streams) > 1 and args.pattern == "single_burst":
                masks = overlapping_burst_masks(
                    length, rate, selected_streams, rng, args.overlap if args.overlap is not None else 1.0
                )
            else:
                masks = {
                    selected_streams[0]: generate_mask(
                        length,
                        rate,
                        args.pattern,
                        rng,
                        require_return=horizon is None and args.pattern == "single_burst",
                        horizon=horizon,
                    )
                }
            if args.no_anchor:
                target_mask = masks[selected_streams[0]]
                target_segments = mask_to_segments(target_mask)
                outage_start = target_segments[0][0] if target_segments else 0
                for other in streams:
                    if other in masks:
                        continue
                    other_mask = np.ones(length, dtype=bool)
                    if args.no_anchor == "others_offline":
                        other_mask[outage_start:] = False
                    else:
                        other_mask[:] = target_mask
                    masks[other] = other_mask
            segment_map = {name: mask_to_segments(mask) for name, mask in masks.items()}
            all_segments = [segment for segments in segment_map.values() for segment in segments]
            longest = max((stop - start + 1 for start, stop in all_segments), default=0)
            last_stop = max((stop for _, stop in all_segments), default=-1)
            actual_rate = float(np.mean([(~mask).mean() for mask in masks.values()]))
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    pattern=args.pattern,
                    missing_rate=actual_rate,
                    segments=segment_map,
                    duration_seconds=longest / config["sample_rate"],
                    recovery_horizon_seconds=(length - last_stop - 1) / config["sample_rate"],
                    mask_seed=args.seed,
                    condition=args.condition or f"{args.pattern}_{int(round(rate * 100))}",
                )
            )
    condition = args.condition or f"{args.pattern}_{int(round(rate * 100))}"
    output = args.output or resolve_project_path(
        f"manifests/outage_masks/{args.dataset}/fold{args.fold}/{args.split}_{condition}.jsonl"
    )
    Manifest(entries).save(output)


if __name__ == "__main__":
    main()
