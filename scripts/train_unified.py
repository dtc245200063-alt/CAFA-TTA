"""Train ONE ResNet18 with ONE 17-way head on PathMNIST + Kather 2016.

Labels are domain-prefixed to avoid claiming the datasets share a clinical taxonomy.
Requires torch, torchvision, numpy, pillow. Does not overwrite old checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
PATHMNIST_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_kather(root: Path):
    """Find the 8 class directories based on image-containing directories.

    Accepts Kather at arbitrary folder nesting, but refuses ambiguous layouts.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Kather directory not found: {root}")
    dirs = []
    for directory in [root, *root.rglob("*")]:
        if directory.is_dir():
            images = sorted(x for x in directory.iterdir() if x.is_file() and x.suffix.lower() in ALLOWED)
            if images:
                dirs.append((directory, images))
    if len(dirs) != 8:
        found = [str(d.relative_to(root)) for d, _ in dirs[:20]]
        raise ValueError(f"Need exactly 8 Kather class image folders; found {len(dirs)}: {found}")
    return sorted(dirs, key=lambda pair: pair[0].name.lower())


def split_items(items, seed: int):
    """Deterministic per-class 70/15/15 split; disjoint by image path."""
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for cls_id, paths in enumerate(items):
        ordered = list(paths)
        rng.shuffle(ordered)
        n = len(ordered)
        if n < 7:
            raise ValueError(f"Kather class {cls_id} has only {n} images")
        n_train, n_val = int(n * 0.7), int(n * 0.15)
        for name, group in (("train", ordered[:n_train]), ("val", ordered[n_train:n_train+n_val]),
                            ("test", ordered[n_train+n_val:])):
            result[name].extend((path, cls_id + 9) for path in group)
    return result


class PathMNISTDataset(Dataset):
    def __init__(self, npz_path: Path, split: str, tfm):
        if not npz_path.is_file():
            raise FileNotFoundError(f"PathMNIST not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            self.x = data[f"{split}_images"].copy()
            self.y = data[f"{split}_labels"].reshape(-1).astype(int).copy()
        if len(self.x) != len(self.y) or np.any((self.y < 0) | (self.y >= 9)):
            raise ValueError("Invalid PathMNIST images/labels")
        self.tfm = tfm

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tfm(Image.fromarray(self.x[i]).convert("RGB")), int(self.y[i])


class KatherDataset(Dataset):
    def __init__(self, records, tfm):
        self.records, self.tfm = records, tfm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, label = self.records[i]
        with Image.open(path) as img:
            return self.tfm(img.convert("RGB")), label


def build_model(num_classes=17):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_metrics(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    confusion = torch.zeros(17, 17, dtype=torch.int64)
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(1).cpu()
            y = y.long()
            count += len(y)
            correct += (pred == y).sum().item()
            for actual, predicted in zip(y.tolist(), pred.tolist()):
                confusion[actual, predicted] += 1
    f1s = []
    for c in range(17):
        tp = confusion[c, c].item()
        support = confusion[c, :].sum().item()
        if support == 0:
            continue
        fp = confusion[:, c].sum().item() - tp
        fn = support - tp
        denominator = 2 * tp + fp + fn
        f1s.append((2 * tp / denominator) if denominator else 0.0)
    return {"samples": count, "accuracy": (correct / count if count else 0.0),
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pathmnist", type=Path, default=ROOT / "data/raw/medmnist/pathmnist.npz")
    ap.add_argument("--kather-root", type=Path, default=ROOT / "data/raw/kather2016")
    ap.add_argument("--output", type=Path, default=ROOT / "checkpoints/cafa_tta_unified_best.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--quick-samples", type=int, default=0)
    ap.add_argument("--balanced-train", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="Validate dataset layout and split only")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32:
        ap.error("epochs/batch-size must be >=1 and size >=32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = collect_kather(args.kather_root)
    labels = [f"PathMNIST:{x}" for x in PATHMNIST_NAMES] + [f"Kather:{d.name}" for d, _ in dirs]
    splits = split_items([imgs for _, imgs in dirs], args.seed)
    test_paths = {str(p.resolve()) for p, _ in splits["test"]}
    train_paths = {str(p.resolve()) for p, _ in splits["train"]}
    val_paths = {str(p.resolve()) for p, _ in splits["val"]}
    assert not (test_paths & train_paths or test_paths & val_paths or train_paths & val_paths)
    with np.load(args.pathmnist, allow_pickle=False) as data:
        path_counts = {split: len(data[f"{split}_labels"]) for split in ("train", "val", "test")}
    print("ONE MODEL / ONE 17-CLASS HEAD")
    print("Labels:", json.dumps(labels, ensure_ascii=False))
    print("PathMNIST:", path_counts, "Kather:", {k: len(v) for k, v in splits.items()})
    if args.check_only:
        print("DATA CHECK: PASS (no training, no files changed)")
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {args.output}")

    train_tfm = transforms.Compose([
        transforms.Resize((args.size, args.size)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.size, args.size)), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "pin_memory": torch.cuda.is_available()}
    train = ConcatDataset([PathMNISTDataset(args.pathmnist, "train", train_tfm),
                           KatherDataset(splits["train"], train_tfm)])
    val_p = PathMNISTDataset(args.pathmnist, "val", eval_tfm)
    val_k = KatherDataset(splits["val"], eval_tfm)
    val = ConcatDataset([val_p, val_k])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.quick_samples or args.balanced_train:
        n_path = len(train.datasets[0])
        n_kather = len(train.datasets[1])
        weights = [0.5 / n_path] * n_path + [0.5 / n_kather] * n_kather
        sampler = WeightedRandomSampler(
            weights, num_samples=(args.quick_samples or len(train)), replacement=True
        )
        train_loader = DataLoader(train, sampler=sampler, **loader_args)
        print("QUICK TRAIN:", args.quick_samples,
              "balanced samples per epoch", flush=True)
    else:
        train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    print("Device:", device, "Train:", len(train), "Val:", len(val), flush=True)
    best = -1
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0., 0
        for batch_index, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_n += len(y)
            if batch_index % 20 == 0 or batch_index == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} - "
                      f"batch {batch_index}/{len(train_loader)}",
                      flush=True)
        metrics = compute_metrics(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / total_n, "val": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            # Save only one primary checkpoint, with stable labels and preprocessing metadata.
            torch.save({"model_state_dict": model.state_dict(), "labels": labels,
                        "architecture": "resnet18", "input_size": args.size,
                        "normalization": {"mean": [0.5]*3, "std": [0.5]*3},
                        "epoch": epoch, "seed": args.seed, "val": metrics}, args.output)
    state = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    results = {"hyperparameters": {k: str(v) for k, v in vars(args).items()},
               "labels": labels, "seed": args.seed, "best_epoch": state["epoch"],
               "selection": "best combined validation macro-F1", "history": history,
               "test_pathmnist": compute_metrics(model, DataLoader(
                   PathMNISTDataset(args.pathmnist, "test", eval_tfm), shuffle=False, **loader_args), device),
               "test_kather": compute_metrics(model, DataLoader(
                   KatherDataset(splits["test"], eval_tfm), shuffle=False, **loader_args), device)}
    result_path = args.output.with_suffix(".metrics.json")
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CHECKPOINT:", args.output)
    print("METRICS:", result_path)
    print("FINAL TEST:", json.dumps({k: results[k] for k in ("test_pathmnist", "test_kather")}))


if __name__ == "__main__":
    main()
