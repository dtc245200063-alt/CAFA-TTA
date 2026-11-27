import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import TensorDataset

from core.continual import CollapseMonitor, ContinualSession, ExperienceStore, OODGate
from core.registry import ModelRegistry, passes_gate
from core.trainer import fit_temperature, train_classifier
from core.training_utils import build_transforms, resolve_input_config


@pytest.mark.parametrize(
    "modality", ["histopathology", "dermatology", "chest_xray", "fundus", "hematology", "breast_ultrasound"]
)
def test_build_transforms_outputs_normalized_tensor(modality):
    cfg = resolve_input_config(modality=modality, size=224, backbone="resnet18")
    image = Image.fromarray(np.random.randint(0, 255, (300, 260, 3), dtype=np.uint8))
    for train in (True, False):
        assert build_transforms(cfg, train=train)(image).shape == (3, 224, 224)


def test_fit_temperature_softens_overconfident_logits():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (300,))
    logits = torch.randn(300, 3) * 10  # confident but random
    assert fit_temperature(logits, labels) > 2.0


def test_train_classifier_learns_separable_data():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(64, 4) + 3, torch.randn(64, 4) - 3])
    y = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    data = TensorDataset(x, y)
    model, history, metrics = train_classifier(
        torch.nn.Linear(4, 2), data, data, 2, epochs=3, lr=0.05, batch_size=16, device="cpu", log=lambda _: None,
    )
    assert metrics["accuracy"] > 0.95
    assert len(history) == 3


def test_registry_promote_rollback_and_weights_only_load(tmp_path):
    registry = ModelRegistry(tmp_path, "histopathology")
    v1 = registry.register({"w": torch.ones(2), "metrics": {"macro_f1": 0.5}})
    registry.promote(v1)
    v2 = registry.register({"w": torch.zeros(2), "metrics": {"macro_f1": 0.6}})
    registry.promote(v2)
    assert registry.current_version() == "v2"
    assert torch.equal(registry.load()["w"], torch.zeros(2))
    assert registry.rollback() == "v1"
    with pytest.raises(RuntimeError):
        registry.rollback()


def test_gate_rejects_regressions():
    current = {"macro_f1": 0.90, "per_class_recall": [0.9, 0.9]}
    assert passes_gate({"macro_f1": 0.91, "per_class_recall": [0.92, 0.88]}, current)[0]
    assert not passes_gate({"macro_f1": 0.80, "per_class_recall": [0.9, 0.9]}, current)[0]
    ok, reasons = passes_gate({"macro_f1": 0.90, "per_class_recall": [0.99, 0.70]}, current)
    assert not ok and "class 1" in reasons[0]


def test_ood_gate_flags_far_features():
    torch.manual_seed(0)
    prototypes = torch.eye(3, 8)
    features = prototypes.repeat(50, 1) + 0.05 * torch.randn(150, 8)
    logits = torch.eye(3).repeat(50, 1) * 8
    gate = OODGate.calibrate(features, logits, prototypes)
    far = -prototypes.sum(0, keepdim=True)  # opposite direction to every prototype
    assert gate.is_ood(far, torch.zeros(1, 3), prototypes).item()
    assert gate.is_ood(features, logits, prototypes).float().mean() < 0.2
    assert OODGate.from_dict(gate.to_dict()).energy_threshold == gate.energy_threshold


def test_collapse_monitor_triggers_on_disagreement():
    monitor = CollapseMonitor(window=20, min_count=10, max_disagreement=0.3)
    ent = torch.full((10,), 0.5)
    monitor.update(torch.zeros(10), torch.zeros(10), ent, ent)
    assert not monitor.collapsed()
    monitor.update(torch.ones(10), torch.zeros(10), ent, ent)
    assert monitor.collapsed()


def test_experience_store_dedups_and_strips_metadata(tmp_path):
    store = ExperienceStore(tmp_path)
    image = Image.new("RGB", (16, 16), (200, 10, 10))
    image.info["exif"] = b"Exif\x00\x00fake"
    first = store.add(image, {"prediction": "A"})
    assert store.add(image.copy(), {"prediction": "A"}) == first
    store.set_label(first, "B")
    (record,) = store.records()
    assert record["user_label"] == "B" and record["prediction"] == "A"
    assert "exif" not in Image.open(store.image_path(first)).info
    assert len(list(store.images.iterdir())) == 1


def _make_imagefolder(root: Path, per_class: int = 10, size: int = 96) -> None:
    rng = np.random.default_rng(0)
    for name, colour in (("ADI", (230, 200, 220)), ("TUM", (90, 30, 120))):
        (root / name).mkdir(parents=True)
        for i in range(per_class):
            arr = np.clip(np.array(colour) + rng.normal(0, 12, (size, size, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / name / f"{i}.png")


def test_end_to_end_train_session_consolidate_rollback(tmp_path):
    import consolidate
    import train_modality

    data = tmp_path / "data"
    _make_imagefolder(data)
    registry_root, experience_root = tmp_path / "registry", tmp_path / "experience"
    common = ["--modality", "histopathology", "--registry-root", str(registry_root)]

    assert train_modality.main(common + ["--data-dir", str(data), "--size", "96", "--no-pretrained",
                                         "--epochs", "2", "--batch-size", "4", "--grad-accum", "1"]) == "v1"
    registry = ModelRegistry(registry_root, "histopathology")
    checkpoint = registry.load()
    assert set(checkpoint) >= {"temperature", "prototypes", "ood", "labels", "metrics"}

    # Online session: predictions, adaptation on buffered batches, reset to source.
    session = ContinualSession(checkpoint, batch_size=4)
    session.gate = OODGate(float("inf"), float("-inf"))  # accept everything in this smoke test
    source_state = {k: v.clone() for k, v in session.adapter.model.state_dict().items()}
    images = [Image.open(p) for p in sorted((data / "TUM").glob("*.png"))[:8]]
    results = session.predict(images)
    assert len(results) == 8 and results[0]["label"] in checkpoint["labels"]
    assert session.updates >= 1
    session.reset()
    assert all(torch.equal(v, session.adapter.model.state_dict()[k]) for k, v in source_state.items())

    # Experience -> consolidation behind the gate.
    store = ExperienceStore(experience_root / "histopathology")
    for image, result in zip(images, results):
        image_id = store.add(image, {"prediction": result["label"], "confidence": result["confidence"],
                                     "ood": False, "agreement": True})
        store.set_label(image_id, "TUM")
    consolidate_args = common + ["--experience-root", str(experience_root), "--source-data-dir", str(data),
                                 "--epochs", "1", "--batch-size", "4", "--replay-per-class", "4"]

    assert consolidate.main(consolidate_args + ["--max-macro-f1-drop", "-1"]) is None  # gate forced to reject
    assert registry.current_version() == "v1"

    assert consolidate.main(consolidate_args + ["--max-macro-f1-drop", "1", "--max-class-recall-drop", "1"]) == "v2"
    assert registry.current_version() == "v2"
    assert consolidate.main(common + ["--rollback"]) == "v1"
