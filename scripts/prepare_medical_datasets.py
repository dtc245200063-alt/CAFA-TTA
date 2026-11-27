"""Download and organize high-resolution medical datasets into C:\\Code\\DataSet.

Creates ImageFolder-compatible directories ready for CAFA-TTA 4-stage training:
  <root>/histopathology/kather2016/<class>/*
  <root>/dermatology/ham10000/<class>/*
  <root>/chest_xray/pneumonia/{train,val}/<class>/*
  <root>/fundus/aptos2019/<class>/*
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

DEFAULT_ROOT = Path(os.environ.get("CAFA_DATASET_ROOT", r"C:\Code\DataSet"))

ZENODO_DATASETS = {
    "kather2016": {
        "modality": "histopathology",
        "url": "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1",
        "filename": "Kather_texture_2016_image_tiles_5000.zip",
    },
    "nct_crc_100k": {
        "modality": "histopathology",
        "url": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1",
        "filename": "NCT-CRC-HE-100K.zip",
    },
}

KAGGLE_DATASETS = {
    "ham10000": {
        "modality": "dermatology",
        "slug": "kmader/skin-cancer-mnist-ham10000",
        "kind": "dataset",
    },
    "chest_xray_pneumonia": {
        "modality": "chest_xray",
        "slug": "paultimothymooney/chest-xray-pneumonia",
        "kind": "dataset",
    },
    "aptos2019": {
        "modality": "fundus",
        "slug": "aptos2019-blindness-detection",
        "kind": "competition",
    },
}


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"[SKIP] Already downloaded: {dest}", flush=True)
        return
    print(f"[DOWNLOAD] {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[DONE] {dest} ({dest.stat().st_size / 1024**2:.1f} MB)", flush=True)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"[EXTRACT] {zip_path} -> {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def prepare_zenodo(name: str, root: Path) -> Path:
    info = ZENODO_DATASETS[name]
    target_dir = root / info["modality"] / name
    zip_path = root / "_downloads" / info["filename"]
    _download_file(info["url"], zip_path)
    _extract_zip(zip_path, target_dir)
    # Flatten single nested wrapper folder if present
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        nested = subdirs[0]
        for item in nested.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(nested, ignore_errors=True)
    print(f"[READY] {name} -> {target_dir}", flush=True)
    return target_dir


def _organize_csv_images(raw_dir: Path, out_dir: Path, csv_path: Path, id_col: str, label_col: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_map = {
        p.stem: p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    }
    moved = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            img_id = row[id_col].strip()
            label = row[label_col].strip()
            src = image_map.get(img_id)
            if src is None:
                continue
            cls_dir = out_dir / label
            cls_dir.mkdir(parents=True, exist_ok=True)
            dst = cls_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            moved += 1
    print(f"[ORGANIZED] {moved} images -> {out_dir}", flush=True)


def prepare_kaggle(name: str, root: Path) -> Path:
    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` and place your API token at "
            "`%USERPROFILE%\\.kaggle\\kaggle.json`."
        )
    info = KAGGLE_DATASETS[name]
    raw_dir = root / "_downloads" / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        [kaggle_exe, "datasets", "download", "-d", info["slug"], "-p", str(raw_dir), "--unzip"]
        if info["kind"] == "dataset"
        else [kaggle_exe, "competitions", "download", "-c", info["slug"], "-p", str(raw_dir)]
    )
    print("[KAGGLE]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    for zp in raw_dir.glob("*.zip"):
        _extract_zip(zp, raw_dir)

    target_dir = root / info["modality"] / name
    if name == "ham10000":
        csv_path = next(raw_dir.rglob("HAM10000_metadata.csv"))
        _organize_csv_images(raw_dir, target_dir, csv_path, "image_id", "dx")
    elif name == "aptos2019":
        csv_path = next(raw_dir.rglob("train.csv"))
        _organize_csv_images(raw_dir, target_dir, csv_path, "id_code", "diagnosis")
    elif name == "chest_xray_pneumonia":
        cxr_root = next(p for p in raw_dir.rglob("chest_xray") if (p / "train").is_dir())
        if not target_dir.exists():
            shutil.copytree(cxr_root, target_dir)
    print(f"[READY] {name} -> {target_dir}", flush=True)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & prepare medical datasets for CAFA-TTA")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*ZENODO_DATASETS.keys(), *KAGGLE_DATASETS.keys(), "all_free", "all"],
        default=["kather2016"],
        help="Datasets to download. 'all_free' downloads Zenodo datasets without Kaggle token.",
    )
    args = parser.parse_args()

    selected: list[str] = []
    for item in args.datasets:
        if item == "all_free":
            selected.extend(["kather2016"])
        elif item == "all":
            selected.extend(["kather2016", "ham10000", "chest_xray_pneumonia", "aptos2019"])
        else:
            selected.append(item)

    for name in dict.fromkeys(selected):
        if name in ZENODO_DATASETS:
            prepare_zenodo(name, args.root)
        else:
            prepare_kaggle(name, args.root)


if __name__ == "__main__":
    main()
