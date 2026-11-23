
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torchvision import models

from models.feature_extractor import (
    ResNetFeatureExtractor,
)
from core.cafa_adapter import CAFAAdapter


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(
        backbone.fc.in_features, 9
    )

    model = ResNetFeatureExtractor(
        backbone
    ).to(device)

    prototypes = torch.randn(
        9, 512, device=device
    )

    adapter = CAFAAdapter(
        model=model,
        prototypes=prototypes,
        threshold=0.0,  # Exercise the update path
    )

    images = torch.randn(
        16, 3, 28, 28,
        device=device
    )

    # Save parameters to verify real updates.
    before = [
        p.detach().clone()
        for p in adapter.params
    ]

    predictions = adapter.predict(images)
    result = adapter.adapt(images)

    changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(
            before, adapter.params
        )
    )

    print("=" * 50)
    print("CAFA-TTA | ADAPTATION ENGINE TEST")
    print("Device:", device)
    print("Prediction shape:", predictions.shape)
    print("Selected:", result["selected"])
    print("Entropy Loss:", result["entropy_loss"])
    print("Alignment Loss:", result["alignment_loss"])
    print("Total Loss:", result["total_loss"])
    print("Parameters changed:", changed)
    print("=" * 50)

    assert predictions.shape == (16, 9)
    assert result["updated"]
    assert changed

    print("CAFA ADAPTER TEST PASSED")


def test_smoke():
    """pytest entry point: runs the original self-checking script."""
    main()


if __name__ == "__main__":
    main()
