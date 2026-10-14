from __future__ import annotations

from typing import Any

from .baselines import (
    BRITSClassifier,
    CrossSensorReconstructionClassifier,
    MaskedEarlyFusion,
    MultiBranchGRU,
)
from .dasr import DASR
from .tcn import TCNBase, TCNDASR


DATASET_DEFAULTS = {
    "mhealth": {
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "duration_dim": 32,
        "repair_width": 128,
    },
    "pamap2": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "duration_dim": 16,
        "repair_width": 64,
    },
    "wisdm": {
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.1,
        "duration_dim": 16,
        "repair_width": 64,
    },
}


def build_model(
    model_name: str,
    dataset_config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
):
    values = dict(DATASET_DEFAULTS[dataset_config["dataset"]])
    values.update(model_config or {})
    values.pop("model", None)
    stream_dims = {key: int(value) for key, value in dataset_config["streams"].items()}
    common = {
        "stream_dims": stream_dims,
        "num_classes": int(dataset_config["num_classes"]),
        "hidden_dim": int(values["hidden_dim"]),
        "num_layers": int(values.get("num_layers", 1)),
        "dropout": float(values.get("dropout", 0.1)),
        "sample_rate": float(dataset_config["sample_rate"]),
        "encoder_kernels": tuple(values.get("encoder_kernels", (5, 3))),
    }
    if model_name == "matched_gru":
        return MultiBranchGRU(**common, strategy="update")
    if model_name == "state_hold":
        return MultiBranchGRU(**common, strategy="hold")
    if model_name == "modality_dropout":
        return MultiBranchGRU(**common, strategy="update", learned_fusion=False)
    if model_name == "hard_reset":
        return MultiBranchGRU(**common, strategy="reset")
    if model_name == "gru_d":
        return MultiBranchGRU(**common, strategy="decay")
    if model_name in {"masked_early_fusion", "clean_early_fusion"}:
        reduced = {key: value for key, value in common.items() if key != "sample_rate"}
        return MaskedEarlyFusion(**reduced)
    if model_name == "brits":
        return BRITSClassifier(**common)
    if model_name == "cross_sensor_reconstruction":
        return CrossSensorReconstructionClassifier(**common)
    dasr = {
        **common,
        "duration_dim": int(values.get("duration_dim", 16)),
        "repair_width": int(values.get("repair_width", 64)),
        "use_decay": bool(values.get("use_decay", True)),
        "use_duration": bool(values.get("use_duration", True)),
        "use_context": bool(values.get("use_context", True)),
        "adaptive_gate": bool(values.get("adaptive_gate", True)),
    }
    ablations = {
        "dasr_no_decay": {"use_decay": False},
        "dasr_no_duration": {"use_duration": False},
        "dasr_no_context": {"use_context": False},
        "dasr_fixed_gate": {"adaptive_gate": False},
    }
    if model_name == "dasr" or model_name in ablations:
        dasr.update(ablations.get(model_name, {}))
        return DASR(**dasr)
    if model_name == "tcn_dasr":
        return TCNDASR(
            tcn_layers=int(values.get("tcn_layers", values.get("num_layers", 3))),
            **dasr,
        )
    if model_name == "tcn_base":
        return TCNBase(
            tcn_layers=int(values.get("tcn_layers", values.get("num_layers", 3))),
            **common,
            strategy="hold",
        )
    raise ValueError(f"Unknown model: {model_name}")
