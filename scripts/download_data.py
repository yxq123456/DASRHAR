from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


URLS = {
    "mhealth": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "pamap2": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
    "wisdm": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
}


def download(dataset: str, root: Path) -> None:
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{dataset}.zip"
    with urllib.request.urlopen(URLS[dataset]) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    with zipfile.ZipFile(archive) as compressed:
        compressed.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *URLS], default="all")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    datasets = list(URLS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        download(dataset, args.root)


if __name__ == "__main__":
    main()

