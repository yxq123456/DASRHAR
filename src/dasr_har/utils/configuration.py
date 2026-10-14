from __future__ import annotations

from typing import Any

from .io import resolve_project_path


def resolved_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for key in ["raw_dir", "processed_path", "split_path"]:
        result[key] = str(resolve_project_path(result[key]))
    return result


TRAINING_DEFAULTS = {
    "mhealth": {"learning_rate": 3e-4, "weight_decay": 1e-5, "batch_size": 64},
    "pamap2": {"learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 128},
    "wisdm": {"learning_rate": 1e-4, "weight_decay": 1e-5, "batch_size": 64},
}

