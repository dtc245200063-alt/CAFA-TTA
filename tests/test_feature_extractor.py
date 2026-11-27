import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
from torchvision import models

from models.feature_extractor import ResNetFeatureExtractor


def test_resnet18_returns_512_features_and_logits():
    backbone = models.resnet18(weights=None)
    backbone.fc = torch.nn.Linear(512, 9)
    extractor = ResNetFeatureExtractor(backbone).eval()

    with torch.no_grad():
        features, logits = extractor(torch.randn(2, 3, 64, 64))

    assert features.shape == (2, 512)
    assert logits.shape == (2, 9)


def test_densenet_rejected_at_construction():
    backbone = models.densenet121(weights=None)

    with pytest.raises(ValueError, match="requires a torchvision ResNet"):
        ResNetFeatureExtractor(backbone)
