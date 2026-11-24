"""Tests for scripts/train_unified.py (data split and metrics helpers)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest

torch = pytest.importorskip("torch")

from train_unified import build_model, compute_metrics, split_items  # noqa: E402


def fake_items(per_class=625, classes=8):
    return [[Path(f"c{c}/img_{i}.tif") for i in range(per_class)] for c in range(classes)]


def test_kather_split_sizes_and_disjoint():
    split = split_items(fake_items(), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"train": 3496, "val": 744, "test": 760}
    paths = {k: {p for p, _ in v} for k, v in split.items()}
    assert not (paths["train"] & paths["val"])
    assert not (paths["train"] & paths["test"])
    assert not (paths["val"] & paths["test"])


def test_kather_labels_are_offset_by_nine_and_balanced_per_class():
    split = split_items(fake_items(), seed=42)
    for name in ("train", "val", "test"):
        labels = {label for _, label in split[name]}
        assert labels == set(range(9, 17))
    counts = {c: 0 for c in range(9, 17)}
    for _, label in split["train"]:
        counts[label] += 1
    assert set(counts.values()) == {437}


def test_split_is_deterministic_and_seed_dependent():
    a = split_items(fake_items(), seed=42)
    b = split_items(fake_items(), seed=42)
    c = split_items(fake_items(), seed=7)
    assert a == b
    assert a != c


def test_split_rejects_tiny_classes():
    with pytest.raises(ValueError):
        split_items([[Path("a.tif")] * 3], seed=1)


def test_build_model_has_17_outputs():
    model = build_model()
    assert model.fc.out_features == 17


def test_compute_metrics_perfect_and_wrong_predictions():
    class Constant(torch.nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.cls = cls

        def forward(self, x):
            out = torch.zeros(len(x), 17)
            out[:, self.cls] = 1.0
            return out

    x = torch.zeros(4, 3, 8, 8)
    y = torch.tensor([3, 3, 3, 3])
    loader = [(x, y)]
    perfect = compute_metrics(Constant(3), loader, torch.device("cpu"))
    assert perfect["samples"] == 4
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == 1.0
    wrong = compute_metrics(Constant(5), loader, torch.device("cpu"))
    assert wrong["accuracy"] == 0.0
    assert wrong["macro_f1"] == 0.0
