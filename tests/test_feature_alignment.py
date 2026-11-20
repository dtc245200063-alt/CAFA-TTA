
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision import models

from models.feature_extractor import ResNetFeatureExtractor
from kernel.feature_alignment import (
    compute_prototypes,
    confidence_alignment_loss,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backbone = models.resnet18(weights=None)
    backbone.fc = torch.nn.Linear(
        backbone.fc.in_features, 9
    )

    model = ResNetFeatureExtractor(
        backbone
    ).to(device)

    # Synthetic test data
    images = torch.randn(
        12, 3, 28, 28,
        device=device
    )

    features, logits = model(images)

    labels = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2],
        device=device
    )

    prototypes, counts = compute_prototypes(
        features.detach(),
        labels,
        num_classes=9
    )

    pseudo_labels = logits.argmax(dim=1)

    confidences = torch.full(
        (12,),
        0.9,
        device=device
    )

    loss = confidence_alignment_loss(
        features,
        pseudo_labels,
        confidences,
        prototypes,
        threshold=0.7
    )

    loss.backward()

    print("=" * 50)
    print("CAFA-TTA | FEATURE ALIGNMENT TEST")
    print("Device:", device)
    print("Feature shape:", tuple(features.shape))
    print("Prototype shape:", tuple(prototypes.shape))
    print("Class counts:", counts.cpu().tolist())
    print("Alignment Loss:", loss.item())
    print("=" * 50)

    assert features.shape == (12, 512)
    assert prototypes.shape == (9, 512)
    assert torch.isfinite(loss)

    print("FEATURE ALIGNMENT TEST PASSED")


if __name__ == "__main__":
    main()
