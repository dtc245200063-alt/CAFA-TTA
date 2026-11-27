"""Versioned per-modality model registry with promotion gate and rollback.

Layout: `<root>/<modality>/registry.json` plus `<root>/<modality>/vN/model.pt`.
Original research checkpoints in `checkpoints/` are never written here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch


def passes_gate(
    candidate: dict,
    current: dict,
    max_macro_f1_drop: float = 0.005,
    max_class_recall_drop: float = 0.05,
) -> tuple[bool, list[str]]:
    """Accept a candidate only if it does not regress on the same fixed evaluation set."""
    reasons = []
    drop = current["macro_f1"] - candidate["macro_f1"]
    if drop > max_macro_f1_drop:
        reasons.append(f"macro-F1 dropped by {drop:.4f} (> {max_macro_f1_drop})")
    for idx, (old, new) in enumerate(zip(current["per_class_recall"], candidate["per_class_recall"])):
        if old - new > max_class_recall_drop:
            reasons.append(f"class {idx} recall dropped {old:.3f} -> {new:.3f}")
    return not reasons, reasons


class ModelRegistry:
    def __init__(self, root: Path | str, modality: str):
        self.dir = Path(root) / modality
        self.index_path = self.dir / "registry.json"

    def _load(self) -> dict:
        if self.index_path.is_file():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"current": None, "history": [], "versions": {}}

    def _save(self, index: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def versions(self) -> dict:
        return self._load()["versions"]

    def current_version(self) -> str | None:
        return self._load()["current"]

    def register(self, checkpoint: dict, notes: str = "") -> str:
        index = self._load()
        version = f"v{len(index['versions']) + 1}"
        path = self.dir / version / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        index["versions"][version] = {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "parent": index["current"],
            "metrics": checkpoint.get("metrics", {}),
            "notes": notes,
        }
        self._save(index)
        return version

    def promote(self, version: str) -> None:
        index = self._load()
        if version not in index["versions"]:
            raise KeyError(f"Unknown version {version}")
        if index["current"] and index["current"] != version:
            index["history"].append(index["current"])
        index["current"] = version
        self._save(index)

    def rollback(self) -> str:
        index = self._load()
        if not index["history"]:
            raise RuntimeError("No earlier promoted version to roll back to")
        index["current"] = index["history"].pop()
        self._save(index)
        return index["current"]

    def load(self, version: str | None = None) -> dict:
        version = version or self.current_version()
        if version is None:
            raise FileNotFoundError(f"No promoted model in {self.dir}")
        # weights_only: refuse pickled code in checkpoints.
        return torch.load(self.dir / version / "model.pt", map_location="cpu", weights_only=True)
