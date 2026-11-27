import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.training_utils import DEFAULT_SHIFT, load_domain_shift


def test_falls_back_to_default_when_metrics_missing(tmp_path):
    assert load_domain_shift(tmp_path) == DEFAULT_SHIFT


def test_reads_shift_from_metrics_file(tmp_path):
    metrics = tmp_path / "experiments" / "domain_shift"
    metrics.mkdir(parents=True)
    shift = {"brightness": 1.1, "contrast": 0.9, "saturation": 1.2}
    (metrics / "metrics.json").write_text(json.dumps({"shift": shift}), encoding="utf-8")

    assert load_domain_shift(tmp_path) == shift
