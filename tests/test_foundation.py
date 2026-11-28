import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from core.feature_tta import FeatureSession
from core.foundation import train_head


def _blobs(shift=0.0, n=60):
    torch.manual_seed(0)
    centres = torch.eye(3, 8) * 4
    y = torch.arange(3).repeat_interleave(n)
    f = centres[y] + torch.randn(len(y), 8)
    f[:, 0] += shift  # pushes every sample towards class 0, like a stain shift
    return f, y


def test_train_head_separates_classes():
    f, y = _blobs()
    head = train_head(f, y, 3, epochs=50)
    assert (head(f).argmax(1) == y).float().mean() > 0.95


def test_feature_session_recovers_from_shift_and_never_sees_labels():
    f, y = _blobs()
    head = train_head(f, y, 3, epochs=50)
    shifted, y2 = _blobs(shift=5.0)
    perm = torch.randperm(len(y2))
    shifted, y2 = shifted[perm], y2[perm]
    source_acc = (head(shifted).argmax(1) == y2).float().mean()
    for kwargs in ({"stats": "balanced", "min_memory": 16}, {"stats": "balanced", "min_memory": 16, "t3a": True}):
        session = FeatureSession(head, **kwargs)
        preds = torch.cat([session.predict(shifted[i:i + 30]).argmax(1) for i in range(0, len(y2), 30)])
        accuracy = (preds == y2).float().mean()
        assert accuracy > (source_acc + 0.1 if not kwargs.get("t3a") else 0.5)
        assert all(len(bucket) <= 16 for bucket in session.memory)
