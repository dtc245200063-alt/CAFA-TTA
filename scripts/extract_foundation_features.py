"""Extract frozen Phikon CLS features for NCT-CRC-HE-100K, CRC-VAL-HE-7K and Kather 2016.

Writes runs/foundation/features_<name>.pt with float16 features, NCT-style labels
(Kather 2016 mapped to NCT names; 03_COMPLEX dropped), file paths and, for Kather, patient ids.
Research use only; Phikon weights are under the Owkin non-commercial licence.

    python scripts/extract_foundation_features.py --data-root C:/Code/DataSet/histopathology
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.foundation import PhikonEncoder, phikon_transform  # noqa: E402

LABELS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
KATHER_TO_NCT = {"01_TUMOR": "TUM", "02_STROMA": "STR", "04_LYMPHO": "LYM", "05_DEBRIS": "DEB",
                 "06_MUCOSA": "NORM", "07_ADIPOSE": "ADI", "08_EMPTY": "BACK"}
DATASETS = {"nct_crc_100k": "nct_crc_100k", "crc_val": "crc_val_he_7k", "kather2016": "kather2016"}


class Files(Dataset):
    def __init__(self, items):
        self.items, self.tf = items, phikon_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        with Image.open(self.items[i][0]) as im:
            return self.tf(im.convert("RGB")), self.items[i][1]


def list_items(folder: Path):
    items = []
    for p in sorted(folder.glob("*/*.tif")):
        name = KATHER_TO_NCT.get(p.parent.name, p.parent.name)
        if name in LABELS:
            items.append((p, LABELS.index(name)))
    return items


def patient_id(path: Path) -> str:
    match = re.search(r"CRC-Prim-HE-(\d+)", path.name)
    return match.group(1) if match else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/foundation")
    parser.add_argument("--only", default=",".join(DATASETS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = PhikonEncoder().to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only.split(","):
        items = list_items(args.data_root / DATASETS[name])
        loader = DataLoader(Files(items), batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device == "cuda")
        feats, t0 = [], time.time()
        for k, (x, _) in enumerate(loader):
            feats.append(encoder(x.to(device, non_blocking=True)).half().cpu())
            if k % 50 == 0:
                print(f"{name}: {k * args.batch_size}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"features": torch.cat(feats), "labels": torch.tensor([y for _, y in items]),
                    "paths": [str(p) for p, _ in items], "patients": [patient_id(p) for p, _ in items],
                    "label_names": LABELS}, args.out / f"features_{name}.pt")
        print(f"{name}: {len(items)} images in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
