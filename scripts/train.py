from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dasr_har.config import load_config
from dasr_har.data.dataset import WindowDataset, build_dataloaders, collate_windows, load_split
from dasr_har.models.factory import build_model
from dasr_har.outages.manifest import Manifest
from dasr_har.training.losses import class_weights
from dasr_har.training.trainer import Trainer
from dasr_har.utils.configuration import TRAINING_DEFAULTS, resolved_dataset_config
from dasr_har.utils.io import resolve_project_path
from dasr_har.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mhealth", "pamap2", "wisdm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)
    dataset_config = resolved_dataset_config(
        load_config(resolve_project_path(f"configs/datasets/{args.dataset}.yaml"))
    )
    model_config = load_config(resolve_project_path(args.model_config)) if args.model_config else {}
    training = TRAINING_DEFAULTS[args.dataset]
    loaders, normalizer = build_dataloaders(
        dataset_config,
        args.fold,
        training["batch_size"],
        args.seed,
        args.workers,
        train_outages=args.model != "clean_early_fusion",
    )
    validation_loaders = {"clean": loaders["val"]}
    split = load_split(dataset_config["split_path"], args.fold)
    manifest_root = resolve_project_path(
        f"manifests/outage_masks/{args.dataset}/fold{args.fold}"
    )
    for path in sorted(manifest_root.glob("val_*.jsonl")) if manifest_root.exists() else []:
        manifest = Manifest.load(path)
        dataset = WindowDataset(
            dataset_config["processed_path"],
            split["val"],
            normalizer,
            dataset_config["sample_rate"],
            manifest.transform(list(dataset_config["streams"])),
        )
        validation_loaders[path.stem.removeprefix("val_")] = DataLoader(
            dataset,
            batch_size=training["batch_size"],
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_windows,
        )
    model = build_model(args.model, dataset_config, model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    train_dataset = loaders["train"].dataset
    labels = torch.as_tensor(
        train_dataset.data["label"][train_dataset.indices], dtype=torch.long
    )
    weights = class_weights(labels, dataset_config["num_classes"])
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        weights,
        torch.device(args.device),
        gradient_clip=1.0,
    )
    run_name = f"{args.dataset}_{args.model}_fold{args.fold}_seed{args.seed}"
    run_config = {
        "dataset": dataset_config,
        "model": args.model,
        "model_config": model_config,
        "fold": args.fold,
        "seed": args.seed,
        "training": training,
    }
    trainer.fit(
        loaders["train"],
        validation_loaders,
        args.epochs,
        args.patience,
        resolve_project_path(f"results/checkpoints/{run_name}.pt"),
        resolve_project_path(f"results/logs/{run_name}.jsonl"),
        run_config,
        normalizer,
        0.4,
    )


if __name__ == "__main__":
    main()

