"""Unit tests for CAFA-TTA v3 (core/cafa_soft_adapter.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn
from torchvision import models

from core.cafa_soft_adapter import CAFASoftAdapter
from models.feature_extractor import ResNetFeatureExtractor

NUM_CLASSES = 9


def make_adapter(**kwargs):
    torch.manual_seed(0)
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    model = ResNetFeatureExtractor(backbone)
    prototypes = torch.randn(NUM_CLASSES, 512)
    adapter = CAFASoftAdapter(model=model, prototypes=prototypes, **kwargs)
    return adapter, model


def batch(n=16):
    torch.manual_seed(1)
    return torch.randn(n, 3, 28, 28)


@pytest.mark.parametrize("floor", [-0.1, 1.1])
def test_entropy_floor_must_be_in_unit_interval(floor):
    with pytest.raises(ValueError):
        make_adapter(entropy_floor=floor)


def test_predict_shape():
    adapter, _ = make_adapter()
    assert adapter.predict(batch()).shape == (16, NUM_CLASSES)


def test_update_is_finite_and_weights_in_range():
    floor = 0.2
    adapter, _ = make_adapter(entropy_floor=floor)
    result = adapter.adapt(batch())
    assert result["updated"] is True
    assert torch.isfinite(torch.tensor(result["total_loss"]))
    # w = floor + (1 - floor) * confidence, confidence in [0, 1]
    assert floor - 1e-6 <= result["entropy_weight_mean"] <= 1.0 + 1e-6
    assert 0.0 <= result["mean_confidence"] <= 1.0
    assert 0 <= result["selected"] <= 16


def test_only_batchnorm_affine_parameters_change():
    adapter, model = make_adapter()
    bn_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_param_ids.update(id(p) for p in (module.weight, module.bias))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    adapter.adapt(batch())
    for name, param in model.named_parameters():
        if id(param) not in bn_param_ids:
            assert torch.equal(before[name], param.detach()), f"{name} changed"
    assert any(
        not torch.equal(before[n], p.detach())
        for n, p in model.named_parameters() if id(p) in bn_param_ids
    ), "no BatchNorm affine parameter was updated"


def test_alignment_skipped_when_too_few_agreement_samples():
    adapter, _ = make_adapter(min_samples=10 ** 6)
    result = adapter.adapt(batch())
    assert result["updated"] is True
    assert result["alignment_loss"] == 0.0


def test_non_finite_input_does_not_update_parameters():
    adapter, model = make_adapter()
    before = [p.detach().clone() for p in adapter.params]
    images = batch()
    images[0, 0, 0, 0] = float("nan")
    result = adapter.adapt(images)
    if result["updated"] is False:
        assert result.get("reason") == "non_finite_loss"
        assert all(torch.equal(b, p.detach()) for b, p in zip(before, adapter.params))
