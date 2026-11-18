
from pathlib import Path

import torch
import medmnist
from medmnist import PathMNIST

print("=" * 50)
print("CAFA-TTA | DATASET SETUP")
print("=" * 50)

root = Path("data/raw/medmnist")
root.mkdir(parents=True, exist_ok=True)

print("MedMNIST:", medmnist.__version__)
print("CUDA:", torch.cuda.is_available())

datasets = {}

for split in ["train", "val", "test"]:
    print(f"\nLoading {split} dataset...")

    dataset = PathMNIST(
        split=split,
        root=str(root),
        download=True,
        size=28
    )

    datasets[split] = dataset

    print(f"{split}: {len(dataset)} images")

image, label = datasets["train"][0]

print("\nFirst image:")
print("Image size:", image.size)
print("Label:", label)

print("\nDATASET READY")
print("Saved at:", root.resolve())
