import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


DATA_PATH = Path(
    "/project/data/raw/medmnist/pathmnist.npz"
)
OUTPUT_ROOT = Path("/outputs")


def prepare_data(images, labels, count, rng):
    indices = rng.choice(
        len(images),
        size=count,
        replace=False
    )

    x = np.array(
        images[indices],
        dtype=np.uint8,
        copy=True
    )

    y = np.array(
        labels[indices].reshape(-1),
        dtype=np.int64,
        copy=True
    )

    x = torch.from_numpy(
        x.transpose(0, 3, 1, 2).copy()
    ).float()

    x = (x / 255.0 - 0.5) / 0.5
    y = torch.from_numpy(y)

    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        required=True
    )
    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
        args.run_name
    ):
        parser.error("Invalid run name")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    run_dir = OUTPUT_ROOT / args.run_name

    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}"
        )

    run_dir.mkdir(parents=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    print("=" * 65, flush=True)
    print("CAFA-TTA | DOCKER CPU TRAINING SMOKE TEST")
    print("=" * 65, flush=True)

    print("Device: CPU")
    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Training images: 256")
    print("Validation images: 128")
    print("Epochs: 1")
    print("Batch size: 8")
    print("Output:", run_dir, flush=True)

    print("\nLoading PathMNIST...", flush=True)

    with np.load(
        DATA_PATH,
        allow_pickle=False
    ) as dataset:
        rng = np.random.default_rng(seed)

        train_data = prepare_data(
            dataset["train_images"],
            dataset["train_labels"],
            256,
            rng
        )

        val_data = prepare_data(
            dataset["val_images"],
            dataset["val_labels"],
            128,
            rng
        )

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    model = models.resnet18(
        weights=None,
        num_classes=9
    )

    model = model.to("cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    print("\nStarting one training epoch...", flush=True)

    model.train()

    total_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_id, (images, labels) in enumerate(
        train_loader,
        start=1
    ):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

        if batch_id % 8 == 0:
            print(
                f"Batch {batch_id}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True
            )

    train_loss = total_loss / train_total
    train_acc = train_correct / train_total

    print("\nValidating...", flush=True)

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            outputs = model(images)

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            val_total += labels.size(0)

    val_acc = val_correct / val_total

    checkpoint_path = (
        run_dir / "resnet18_cpu_smoke.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "num_classes": 9,
            "smoke_test": True,
        },
        checkpoint_path
    )

    metrics = {
        "model": "ResNet18",
        "device": "cpu",
        "epochs": 1,
        "train_samples": 256,
        "val_samples": 128,
        "batch_size": 8,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "smoke_test": True,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("CPU TRAINING SMOKE TEST COMPLETED")
    print("=" * 65)

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    print("Checkpoint:", checkpoint_path)
    print("Metrics:", run_dir / "metrics.json")
    print("Original project modified: NO")
    print("Research checkpoint overwritten: NO")
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()
