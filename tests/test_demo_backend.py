import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest

from core.continual import ExperienceStore, OODGate
from test_continual import _make_imagefolder


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    import train_modality
    from demo_gradio import DemoBackend

    root = tmp_path_factory.mktemp("demo")
    _make_imagefolder(root / "data", per_class=6)
    train_modality.main(["--modality", "histopathology", "--data-dir", str(root / "data"),
                         "--registry-root", str(root / "registry"), "--size", "96", "--no-pretrained",
                         "--epochs", "1", "--batch-size", "4", "--grad-accum", "1"])
    return DemoBackend(root / "registry", root / "experience", device="cpu"), root


def test_analyze_stores_in_distribution_and_skips_ood(backend):
    demo, root = backend
    paths = [str(p) for p in sorted((root / "data" / "ADI").glob("*.png"))[:2]]
    store = ExperienceStore(root / "experience" / "histopathology")

    _, session = demo.session("histopathology")
    session.gate = OODGate(float("inf"), float("-inf"))  # everything in-distribution
    rows = demo.analyze("histopathology", paths, adapt=False)
    assert all(r["id"] for r in rows) and len(store.records()) == 2

    session.gate = OODGate(float("-inf"), float("inf"))  # everything out-of-distribution
    rows = demo.analyze("histopathology", [str(p) for p in sorted((root / "data" / "TUM").glob("*.png"))[:2]], True)
    assert all(r["ood"] and r["id"] is None for r in rows) and len(store.records()) == 2
    assert session.updates == 0


def test_labels_versions_and_rollback(backend):
    demo, root = backend
    store = ExperienceStore(root / "experience" / "histopathology")
    image_id = store.records()[0]["id"]
    demo.set_label("histopathology", image_id, "TUM")
    assert any(r.get("user_label") == "TUM" for r in store.records())
    with pytest.raises(ValueError):
        demo.set_label("histopathology", image_id, "NOT_A_CLASS")
    assert demo.versions_table("histopathology")[0][0] == "v1"
    with pytest.raises(RuntimeError):
        demo.rollback("histopathology")  # only one version


def test_ui_builds(backend):
    pytest.importorskip("gradio")
    from demo_gradio import build_ui

    assert build_ui(backend[0]) is not None
