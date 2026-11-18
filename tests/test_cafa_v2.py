
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torchvision import models

from models.feature_extractor import ResNetFeatureExtractor
from core.cafa_agreement_adapter import (
    CAFAAgreementAdapter
)


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

    images = torch.randn(
        16, 3, 28, 28,
        device=device
    )

    # Initialize prototypes from embeddings.
    # This is a synthetic smoke test, not a
    # validation of real prototype quality.
    model.eval()
    with torch.no_grad():
        features, _ = model(images)

    prototypes = features[:9].detach().clone()

    adapter = CAFAAgreementAdapter(
        model=model,
        prototypes=prototypes,
        threshold=0.0,
        lambda_align=0.05,
        min_samples=1,
    )

    before = [
        param.detach().clone()
        for param in adapter.params
    ]

    logits = adapter.predict(images)
    result = adapter.adapt(images)

    parameters_changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(
            before,
            adapter.params
        )
    )

    print("=" * 50)
    print("CAFA-TTA v2 | ADAPTER TEST")
    print("Device:", device)
    print("Output:", tuple(logits.shape))
    print("Selected:", result["selected"])
    print("Updated:", result["updated"])
    print("Parameters changed:", parameters_changed)
    print("=" * 50)

    assert logits.shape == (16, 9)

    if result["updated"]:
        assert parameters_changed
        assert torch.isfinite(
            torch.tensor(result["total_loss"])
        )

    print("CAFA V2 ADAPTER TEST PASSED")


if __name__ == "__main__":
    main()
