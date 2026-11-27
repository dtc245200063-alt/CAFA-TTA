"""Download and organize high-resolution medical datasets into C:\\Code\\DataSet.

Creates ImageFolder-compatible directories ready for CAFA-TTA 4-stage training:
  <root>/histopathology/kather2016/<class>/*
  <root>/histopathology/crc_val_he_7k/<class>/*
  <root>/dermatology/ham10000/<class>/*
  <root>/chest_xray/pneumonia/{train,val}/<class>/*
  <root>/fundus/aptos2019/<class>/*
  <root>/<modality>/<flag>_224/{train,val,test}/<class>/*.png   (MedMNIST+ export)

Gated datasets (MHIST, Chaoyang, ISIC, CheXpert, MIMIC-CXR) are never downloaded
here; see docs/DATASET_LICENSES.md for manual access instructions.

`--check-duplicates` hashes every prepared image under the root (SHA-256 of decoded
pixels + 64-bit average hash) and writes a CSV of exact and near-duplicate groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from PIL import Image

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
    # External validation: 7,180 patches, 50 patients disjoint from NCT-CRC-HE-100K,
    # 224x224 px at 0.5 um/px. Zenodo record 1214456 license: CC BY 4.0.
    "crc_val_he_7k": {
        "modality": "histopathology",
        "url": "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1",
        "filename": "CRC-VAL-HE-7K.zip",
    },
}

# MedMNIST+ (medmnist>=3.0) at size=224. License per medmnist.INFO 3.0.2 / medmnist.com:
# CC BY 4.0 except DermaMNIST (CC BY-NC 4.0). ChestMNIST is excluded on purpose: it is
# multi-label (14 findings), so it has no single-class ImageFolder layout.
MEDMNIST_DATASETS = {
    "dermamnist_224": {"modality": "dermatology", "flag": "dermamnist"},
    "pneumoniamnist_224": {"modality": "chest_xray", "flag": "pneumoniamnist"},
    "retinamnist_224": {"modality": "fundus", "flag": "retinamnist"},
    "bloodmnist_224": {"modality": "hematology", "flag": "bloodmnist"},
    "breastmnist_224": {"modality": "breast_ultrasound", "flag": "breastmnist"},
    "pathmnist_224": {"modality": "histopathology", "flag": "pathmnist"},
}
MEDMNIST_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

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


def _download_file(url: str, dest: Path, retries: int = 30, wait: float = 5.0) -> None:
    """Download to `<dest>.part`, resuming with HTTP Range after drops; rename only when complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"[SKIP] Already downloaded: {dest}", flush=True)
        return
    part = dest.with_name(dest.name + ".part")
    print(f"[DOWNLOAD] {url} -> {dest}", flush=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    have = 0
                length = resp.headers.get("Content-Length")
                total = have + int(length) if length is not None else None
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(resp, f, 8 * 1024 * 1024)
        except urllib.error.HTTPError as err:
            if err.code == 416 and have:  # requested range starts at the end: already complete
                total = have
            else:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            print(f"[RETRY {attempt}/{retries}] {type(err).__name__}: {err} "
                  f"({part.stat().st_size / 1024**2:.0f} MB so far)", flush=True)
            time.sleep(wait * min(attempt, 12))
            continue
        size = part.stat().st_size
        if total is None or size == total:
            part.replace(dest)
            print(f"[DONE] {dest} ({size / 1024**2:.1f} MB)", flush=True)
            return
        print(f"[RETRY {attempt}/{retries}] incomplete: {size} of {total} bytes", flush=True)
        time.sleep(wait * min(attempt, 12))
    raise RuntimeError(f"Download failed after {retries} attempts; partial file kept at {part}")


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
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
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


def _class_dir_name(index: str, label: str) -> str:
    # Index prefix keeps ImageFolder's alphabetical class order equal to MedMNIST label ids
    # (all exported subsets have <= 9 classes).
    slug = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    return f"{index}_{slug}"


def export_medmnist(name: str, root: Path, size: int = 224, download: bool = True) -> Path:
    """Export one MedMNIST+ subset to <root>/<modality>/<name>/{train,val,test}/<class>/*.png."""
    import medmnist
    from medmnist import INFO

    info = MEDMNIST_DATASETS[name]
    meta = INFO[info["flag"]]
    if meta["task"].startswith("multi-label"):
        raise ValueError(f"{name} is multi-label and has no ImageFolder layout")
    dataset_cls = getattr(medmnist, meta["python_class"])
    class_dirs = {int(k): _class_dir_name(k, v) for k, v in meta["label"].items()}
    cache_dir = root / "_downloads" / "medmnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir = root / info["modality"] / name

    for split in MEDMNIST_SPLITS:
        ds = dataset_cls(split=split, download=download, size=size, root=str(cache_dir))
        written = 0
        for i, (img, label) in enumerate(zip(ds.imgs, ds.labels)):
            cls_dir = target_dir / split / class_dirs[int(label[0])]
            dst = cls_dir / f"{split}_{i:06d}.png"
            if dst.exists():
                continue
            cls_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(dst)
            written += 1
        print(f"[EXPORT] {name}/{split}: {len(ds.labels)} images ({written} new)", flush=True)
    print(f"[READY] {name} ({meta['license']}) -> {target_dir}", flush=True)
    return target_dir


def _average_hash(img: Image.Image) -> int:
    small = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    pixels = small.tobytes()
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, px in enumerate(pixels) if px > mean)


def find_duplicates(root: Path, threshold: int = 5) -> list[dict]:
    """Return CSV rows for exact (pixel SHA-256) and near (aHash Hamming <= threshold) groups.

    Scans every image under `root` except `_downloads`. A near group is reported only when it
    contains more than one distinct pixel hash, so pure exact groups are not listed twice.
    """
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "_downloads" not in p.relative_to(root).parts
    )
    records = []
    for p in paths:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
        sha = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
        records.append({"path": p, "sha256": sha, "ahash": _average_hash(rgb)})

    rows: list[dict] = []

    def emit(kind: str, group_id: int, members: list[int]) -> None:
        for idx in members:
            rec = records[idx]
            rel = rec["path"].relative_to(root)
            rows.append({
                "group_type": kind,
                "group_id": group_id,
                "dataset": "/".join(rel.parts[:2]),
                "path": str(rel),
                "sha256": rec["sha256"],
                "ahash": f"{rec['ahash']:016x}",
            })

    by_sha: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_sha.setdefault(rec["sha256"], []).append(i)
    exact = [m for m in by_sha.values() if len(m) > 1]
    for gid, members in enumerate(exact):
        emit("exact", gid, members)

    # Pigeonhole: hashes within Hamming distance t agree exactly on at least one of t+1 bit bands.
    # shortcut: a band shared by many images (e.g. blank tiles) is compared pairwise; fine for
    # ~1e5 images, switch to a BK-tree or FAISS binary index if it gets slow.
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_bands = threshold + 1
    bounds = [round(64 * b / n_bands) for b in range(n_bands + 1)]
    for b in range(n_bands):
        mask = (1 << (bounds[b + 1] - bounds[b])) - 1
        buckets: dict[int, list[int]] = {}
        for i, rec in enumerate(records):
            buckets.setdefault((rec["ahash"] >> bounds[b]) & mask, []).append(i)
        for members in buckets.values():
            for ai, a in enumerate(members):
                for c in members[ai + 1:]:
                    if find(a) != find(c) and (records[a]["ahash"] ^ records[c]["ahash"]).bit_count() <= threshold:
                        parent[find(a)] = find(c)

    components: dict[int, list[int]] = {}
    for i in range(len(records)):
        components.setdefault(find(i), []).append(i)
    near = [m for m in components.values() if len({records[i]["sha256"] for i in m}) > 1]
    for gid, members in enumerate(near):
        emit("near", gid, members)
    print(
        f"[DUPLICATES] {len(records)} images, {len(exact)} exact groups, {len(near)} near groups "
        f"(Hamming <= {threshold})",
        flush=True,
    )
    return rows


def write_duplicates_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group_type", "group_id", "dataset", "path", "sha256", "ahash"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WROTE] {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & prepare medical datasets for CAFA-TTA")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*ZENODO_DATASETS.keys(), *KAGGLE_DATASETS.keys(), *MEDMNIST_DATASETS.keys(), "all_free", "all"],
        default=None,
        help="Datasets to download (default: kather2016, or none with --check-duplicates). "
        "'all_free' downloads Zenodo datasets without Kaggle token.",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="After preparing, hash all images under --root and write exact/near-duplicate groups to CSV.",
    )
    parser.add_argument("--hamming-threshold", type=int, default=5, help="Max aHash Hamming distance for near duplicates.")
    parser.add_argument("--duplicates-csv", type=Path, default=None, help="Default: <root>/duplicates.csv")
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = [] if args.check_duplicates else ["kather2016"]

    selected: list[str] = []
    for item in args.datasets:
        if item == "all_free":
            selected.extend(["kather2016", "crc_val_he_7k"])
        elif item == "all":
            selected.extend(["kather2016", "ham10000", "chest_xray_pneumonia", "aptos2019"])
        else:
            selected.append(item)

    for name in dict.fromkeys(selected):
        if name in ZENODO_DATASETS:
            prepare_zenodo(name, args.root)
        elif name in MEDMNIST_DATASETS:
            export_medmnist(name, args.root)
        else:
            prepare_kaggle(name, args.root)

    if args.check_duplicates:
        rows = find_duplicates(args.root, args.hamming_threshold)
        write_duplicates_csv(rows, args.duplicates_csv or args.root / "duplicates.csv")


if __name__ == "__main__":
    main()
