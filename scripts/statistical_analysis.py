from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dasr_har.evaluation.metrics import macro_f1
from dasr_har.evaluation.statistics import exact_wilcoxon, holm_adjust, paired_subject_bootstrap
from dasr_har.utils.io import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = pd.read_csv(resolve_project_path(args.first))
    second = pd.read_csv(resolve_project_path(args.second))
    keys = ["dataset", "subject_id", "sample_id", "condition"]
    merged = first.merge(second, on=keys, suffixes=("_first", "_second"))
    merged["correct_first"] = (merged["label_first"] == merged["prediction_first"]).astype(float)
    merged["correct_second"] = (merged["label_second"] == merged["prediction_second"]).astype(float)
    rows = []
    for (dataset, condition), group in merged.groupby(["dataset", "condition"]):
        subject_rows = []
        for subject_id, values in group.groupby("subject_id"):
            if args.metric == "macro_f1":
                first_value = macro_f1(
                    values["label_first"].to_numpy(), values["prediction_first"].to_numpy()
                )
                second_value = macro_f1(
                    values["label_second"].to_numpy(), values["prediction_second"].to_numpy()
                )
            else:
                first_value = values["correct_first"].mean()
                second_value = values["correct_second"].mean()
            subject_rows.append(
                {"subject_id": subject_id, "first": first_value, "second": second_value}
            )
        subject = pd.DataFrame(subject_rows).set_index("subject_id")
        estimate, low, high = paired_subject_bootstrap(
            subject["first"].to_numpy(),
            subject["second"].to_numpy(),
            args.bootstrap_samples,
            args.seed,
        )
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_raw": exact_wilcoxon(
                    subject["first"].to_numpy(), subject["second"].to_numpy()
                ),
                "subjects": len(subject),
            }
        )
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()

