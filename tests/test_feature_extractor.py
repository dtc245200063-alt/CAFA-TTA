import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
from torchvision import models

from core.training_utils import build_backbone
from models.feature_extractor import ResNetFeatureExtractor


@pytest.mark.parametrize(
    ("name", "dim"),
    [("resnet18", 512), ("resnet50", 2048), ("densenet121", 1024)],
)
def test_backbones_return_features_and_logits(name, dim):
    backbone = build_backbone(name, num_classes=5, pretrained=False)
    extractor = ResNetFeatureExtractor(backbone).eval()

    with torch.no_grad():
        features, logits = extractor(torch.randn(2, 3, 64, 64))

    assert features.shape == (2, dim)
    assert logits.shape == (2, 5)


def test_densenet_logits_match_torchvision_forward():
    backbone = models.densenet121(weights=None, num_classes=3).eval()

    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64)
        _, logits = ResNetFeatureExtractor(backbone)(x)
        expected = backbone(x)

    assert torch.allclose(logits, expected, atol=1e-5)


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="Unsupported backbone"):
        ResNetFeatureExtractor(torch.nn.Linear(4, 2))
