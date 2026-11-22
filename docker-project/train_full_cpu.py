import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import models, transforms


DATA_ROOT = Path("/project/data/raw/medmnist")
OUTPUT_ROOT = Path("/outputs")


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
        parser.error("Epochs must be at least 1")

    output_dir = OUTPUT_ROOT / args.run_name

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    if not (DATA_ROOT / "pathmnist.npz").is_file():
        raise FileNotFoundError(
            DATA_ROOT / "pathmnist.npz"
        )

    output_dir.mkdir(parents=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])

    print("Loading PathMNIST...", flush=True)

    train_data = PathMNIST(
        split="train",
        root=str(DATA_ROOT),
        transform=transform,
        download=False,
        size=28,
    )

    val_data = PathMNIST(
        split="val",
        root=str(DATA_ROOT),
        transform=transform,
        download=False,
        size=28,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history = []

    print("=" * 60, flush=True)
    print("CAFA-TTA | FULL BASELINE TRAINING", flush=True)
    print("Device: CPU", flush=True)
    print("Train samples:", len(train_data), flush=True)
    print("Validation samples:", len(val_data), flush=True)
    print("Epochs:", args.epochs, flush=True)
    print("Batch size: 32", flush=True)
    print("Output:", output_dir, flush=True)
    print("=" * 60, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch, (images, labels) in enumerate(
            train_loader, start=1
        ):
            labels = labels.view(-1).long()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * images.size(0)
            )

            train_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            train_total += images.size(0)

            if batch % 200 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"| Batch {batch}/{len(train_loader)} "
                    f"| Loss={loss.item():.4f}",
                    flush=True
                )

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                labels = labels.view(-1).long()

                outputs = model(images)

                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

                val_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "num_classes": 9,
                },
                output_dir / "best.pt",
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "num_classes": 9,
            },
            output_dir / "latest.pt",
        )

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"EPOCH {epoch}/{args.epochs} "
            f"| Train Acc={train_acc:.4f} "
            f"| Val Acc={val_acc:.4f} "
            f"| Best Val={best_val_acc:.4f}",
            flush=True
        )

    print("FULL TRAINING COMPLETED", flush=True)
    print("BEST VALIDATION ACCURACY:", best_val_acc)
    print("CHECKPOINT:", output_dir / "best.pt")
    print("FINAL STATUS: PASS", flush=True)


if __name__ == "__main__":
    main()
