import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


DATA_ROOT = Path(
    "/kather/extracted/"
    "Kather_texture_2016_image_tiles_5000"
)

SPLIT_FILE = Path("/kather/split_seed42.json")
OUTPUT_ROOT = Path("/outputs")


class KatherDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = DATA_ROOT / record["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, int(record["label"])


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
        args.run_name
    ):
        parser.error("Invalid run name")

    if args.epochs < 1:
        parser.error("Epochs must be positive")

    output = OUTPUT_ROOT / args.run_name

    if output.exists():
        raise FileExistsError(output)

    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(SPLIT_FILE)

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    assert len(split["classes"]) == 8
    assert len(split["splits"]["train"]) == 3496
    assert len(split["splits"]["val"]) == 752
    assert len(split["splits"]["test"]) == 752

    all_paths = [
        item["path"]
        for name in ["train", "val", "test"]
        for item in split["splits"][name]
    ]

    assert len(all_paths) == len(set(all_paths))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_data = KatherDataset(
        split["splits"]["train"],
        train_transform,
    )

    val_data = KatherDataset(
        split["splits"]["val"],
        val_transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = models.resnet18(
        weights=None,
        num_classes=8,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    output.mkdir(parents=True)

    history = []
    best_acc = -1.0

    print("=" * 60, flush=True)
    print("CAFA-TTA | KATHER 2016 TRAINING", flush=True)
    print("Device: CPU", flush=True)
    print("Classes: 8", flush=True)
    print("Train:", len(train_data), flush=True)
    print("Validation:", len(val_data), flush=True)
    print("Epochs:", args.epochs, flush=True)
    print("Batch size: 16", flush=True)
    print("Image size: 96x96", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for batch, (images, labels) in enumerate(
            train_loader, start=1
        ):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(labels)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += len(labels)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"| Batch {batch}/{len(train_loader)} "
                    f"| Loss={loss.item():.4f}",
                    flush=True,
                )

        train_acc = correct / total
        train_loss = loss_sum / total
        val_acc = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_acc,
                "num_classes": 8,
                "classes": split["classes"],
            }, output / "best.pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_acc,
            "num_classes": 8,
            "classes": split["classes"],
        }, output / "latest.pt")

        (output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"EPOCH {epoch}/{args.epochs} "
            f"| Train Acc={train_acc:.4f} "
            f"| Val Acc={val_acc:.4f}",
            flush=True,
        )

    print("KATHER TRAINING COMPLETED", flush=True)
    print("BEST VALIDATION ACCURACY:", best_acc)
    print("CHECKPOINT:", output / "best.pt")
    print("FINAL STATUS: PASS", flush=True)


if __name__ == "__main__":
    main()
