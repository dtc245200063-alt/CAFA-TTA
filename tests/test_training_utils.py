from pathlib import Path

from PIL import Image
import pytest
import torch
from torchvision import transforms

from core.training_utils import (
    IMAGENET_MEAN,
    MODALITY_PROFILES,
    SIMPLE_MEAN,
    build_backbone,
    load_image_datasets,
    resolve_input_config,
)


def test_modality_sweet_spots_and_bounds():
    assert MODALITY_PROFILES["histopathology"].min_size == 96
    assert MODALITY_PROFILES["histopathology"].optimal_size == 224
    assert MODALITY_PROFILES["histopathology"].max_size == 256

    assert MODALITY_PROFILES["dermatology"].min_size == 224
    assert MODALITY_PROFILES["dermatology"].optimal_size == 224
    assert MODALITY_PROFILES["dermatology"].max_size == 384

    assert MODALITY_PROFILES["chest_xray"].min_size == 224
    assert MODALITY_PROFILES["chest_xray"].optimal_size == 320
    assert MODALITY_PROFILES["chest_xray"].max_size == 512

    assert MODALITY_PROFILES["fundus"].min_size == 224
    assert MODALITY_PROFILES["fundus"].optimal_size == 384
    assert MODALITY_PROFILES["fundus"].max_size == 512


def test_resolve_input_config_validation():
    cfg = resolve_input_config("fundus", size=None, backbone=None, pretrained=True)
    assert cfg["input_size"] == 384
    assert cfg["backbone"] == "resnet50"
    assert cfg["normalization"]["mean"] == list(IMAGENET_MEAN)

    cfg_scratch = resolve_input_config("histopathology", size=96, backbone="resnet18", pretrained=False)
    assert cfg_scratch["input_size"] == 96
    assert cfg_scratch["normalization"]["mean"] == list(SIMPLE_MEAN)

    with pytest.raises(ValueError):
        resolve_input_config("fundus", size=128, backbone=None, pretrained=True)

    with pytest.raises(ValueError):
        resolve_input_config("histopathology", size=512, backbone=None, pretrained=True)


@pytest.mark.parametrize(
    ("backbone", "size"),
    [
        ("resnet18", 224),
        ("resnet50", 384),
        ("densenet121", 320),
    ],
)
def test_build_backbone_forward_shapes(backbone, size):
    model = build_backbone(backbone, num_classes=5, pretrained=False)
    model.eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 5)


def test_load_image_datasets_flat_and_presplit(tmp_path: Path):
    tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    # 1. Flat class directory -> deterministic 80/20 split
    flat = tmp_path / "flat"
    for cls in ("benign", "malignant"):
        (flat / cls).mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (128, 128), color=(i * 40, 20, 100)).save(flat / cls / f"{i}.png")

    tr, va, names = load_image_datasets(flat, tfm, tfm, seed=42)
    assert names == ["benign", "malignant"]
    assert len(tr) == 8
    assert len(va) == 2

    # 2. Pre-split train/val directory
    presplit = tmp_path / "presplit"
    for split, count in (("train", 4), ("val", 2)):
        for cls in ("normal", "dr"):
            (presplit / split / cls).mkdir(parents=True)
            for i in range(count):
                Image.new("RGB", (128, 128), color=(10, i * 50, 30)).save(presplit / split / cls / f"{i}.png")

    tr2, va2, names2 = load_image_datasets(presplit, tfm, tfm, seed=42)
    assert names2 == ["dr", "normal"]
    assert len(tr2) == 8
    assert len(va2) == 4
