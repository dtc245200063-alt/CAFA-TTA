
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 9
FEATURE_DIM = 512
BATCH_SIZE = 64

CHECKPOINTS = {
    "enhanced": "resnet18_pathmnist_enhanced_best.pt",
    "robust": "resnet18_pathmnist_robust_best.pt",
    "balanced": "resnet18_pathmnist_balanced_best.pt",
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5] * 3,
        std=[0.5] * 3,
    ),
])

dataset = PathMNIST(
    split="train",
    root=str(ROOT / "data/raw/medmnist"),
    transform=transform,
    download=False,
    size=28,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

print("=" * 60)
print("CAFA-TTA | MULTI-MODEL PROTOTYPE EXTRACTION")
print("Device:", DEVICE)
print("Training samples:", len(dataset))
print("Batch size:", BATCH_SIZE)
print("=" * 60)


def extract_prototypes(model_name, checkpoint_name):
    print("\n" + "=" * 60)
    print("MODEL:", model_name.upper())
    print("Checkpoint:", checkpoint_name)
    print("=" * 60)

    # 1. Load the corresponding checkpoint
    backbone = models.resnet18(weights=None)

    backbone.fc = nn.Linear(
        backbone.fc.in_features,
        NUM_CLASSES,
    )

    checkpoint_path = (
        ROOT / "checkpoints" / checkpoint_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 2. Extract penultimate-layer features
    model = ResNetFeatureExtractor(
        backbone
    ).to(DEVICE)

    model.eval()

    # 3. Class-wise accumulators
    feature_sums = torch.zeros(
        NUM_CLASSES,
        FEATURE_DIM,
        device=DEVICE,
        dtype=torch.float64,
    )

    class_counts = torch.zeros(
        NUM_CLASSES,
        device=DEVICE,
        dtype=torch.long,
    )

    # 4. Source features, no gradients
    with torch.inference_mode():
        for images, labels in tqdm(
            loader,
            desc=f"Extracting {model_name}",
        ):
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.view(-1).long().to(
                DEVICE,
                non_blocking=True,
            )

            features, _ = model(images)

            features = F.normalize(
                features.float(),
                p=2,
                dim=1,
            ).to(torch.float64)

            feature_sums.index_add_(
                0,
                labels,
                features,
            )

            class_counts += torch.bincount(
                labels,
                minlength=NUM_CLASSES,
            )

    # 5. Compute normalized class prototypes
    if torch.any(class_counts == 0):
        raise RuntimeError(
            f"Missing class in {model_name}"
        )

    prototypes = (
        feature_sums
        / class_counts.unsqueeze(1)
    ).float()

    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    )

    # 6. Validate before saving
    assert prototypes.shape == (
        NUM_CLASSES,
        FEATURE_DIM,
    )

    assert (
        int(class_counts.sum().item())
        == len(dataset)
    )

    assert torch.isfinite(prototypes).all()

    # 7. Save separately for each model
    output_dir = ROOT / "checkpoints"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"source_prototypes_{model_name}.pt"
    )

    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "counts": class_counts.cpu(),
            "num_classes": NUM_CLASSES,
            "feature_dim": FEATURE_DIM,
            "backbone": "resnet18",
            "dataset": "PathMNIST",
            "split": "train",
            "feature_mode": "source_eval",
            "source_checkpoint": checkpoint_name,
            "model_variant": model_name,
        },
        output_path,
    )

    print(
        "Prototype Shape:",
        tuple(prototypes.shape),
    )

    print(
        "Class Counts:",
        class_counts.cpu().tolist(),
    )

    print(
        "Total Samples:",
        int(class_counts.sum().item()),
    )

    print(
        "Contains NaN:",
        torch.isnan(prototypes).any().item(),
    )

    print("Saved:", output_path)

    del model
    del backbone

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


for model_name, checkpoint_name in CHECKPOINTS.items():
    extract_prototypes(
        model_name,
        checkpoint_name,
    )

print("\n" + "=" * 60)
print("MULTI-MODEL PROTOTYPE EXTRACTION COMPLETED")
print("=" * 60)
