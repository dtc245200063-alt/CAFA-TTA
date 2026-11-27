
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor

# ==========================================
# CAFA-TTA | SOURCE PROTOTYPE EXTRACTION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 9
FEATURE_DIM = 512
BATCH_SIZE = 64

print("=" * 55)
print("CAFA-TTA | SOURCE PROTOTYPE EXTRACTION")
print("Device:", DEVICE)
print("=" * 55)

# 1. Load source training dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
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
    pin_memory=DEVICE.type == "cuda"
)

print("Total images:", len(dataset))

# 2. Load trained ResNet-18
backbone = models.resnet18(weights=None)

backbone.fc = nn.Linear(
    backbone.fc.in_features,
    NUM_CLASSES
)

checkpoint_path = (
    ROOT / "checkpoints/resnet18_pathmnist_best.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

backbone.load_state_dict(
    checkpoint["model_state_dict"]
)

model = ResNetFeatureExtractor(
    backbone
).to(DEVICE)

# Source statistics must use the source model
# in evaluation mode.
model.eval()

# 3. Initialize class accumulators
feature_sums = torch.zeros(
    NUM_CLASSES,
    FEATURE_DIM,
    device=DEVICE,
    dtype=torch.float64
)

class_counts = torch.zeros(
    NUM_CLASSES,
    device=DEVICE,
    dtype=torch.long
)

# 4. Extract and accumulate embeddings
with torch.inference_mode():
    for images, labels in tqdm(
        loader, desc="Extracting features"
    ):
        images = images.to(
            DEVICE, non_blocking=True
        )

        labels = labels.view(-1).long().to(
            DEVICE, non_blocking=True
        )

        features, logits = model(images)

        # Normalize each source embedding
        features = F.normalize(
            features.float(),
            p=2,
            dim=1
        ).to(torch.float64)

        feature_sums.index_add_(
            0, labels, features
        )

        class_counts += torch.bincount(
            labels,
            minlength=NUM_CLASSES
        )

# 5. Compute mean prototype per class
if torch.any(class_counts == 0):
    raise RuntimeError(
        "Some source classes have no samples"
    )

prototypes = (
    feature_sums
    / class_counts.unsqueeze(1)
).float()

# Unit-normalize the resulting prototypes
prototypes = F.normalize(
    prototypes,
    p=2,
    dim=1
)

# 6. Save source statistics
output_dir = ROOT / "checkpoints"
output_dir.mkdir(exist_ok=True)

output_path = (
    output_dir / "source_prototypes.pt"
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
        "source_checkpoint": checkpoint_path.name,
    },
    output_path
)

print("\n" + "=" * 55)
print("PROTOTYPE EXTRACTION COMPLETED")
print("Prototype Shape:", tuple(prototypes.shape))
print("Class Counts:", class_counts.cpu().tolist())
print("Total Samples:", int(class_counts.sum().item()))
print("Contains NaN:", torch.isnan(prototypes).any().item())
print("Saved:", output_path)
print("=" * 55)

assert prototypes.shape == (9, 512)
assert int(class_counts.sum().item()) == len(dataset)
assert torch.isfinite(prototypes).all()
