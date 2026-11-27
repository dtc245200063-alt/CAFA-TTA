"""CAFA-TTA continual-learning research demo (Gradio, local only).

Per modality it loads the current model from `models_registry/<modality>/`.
Each analysis runs a ContinualSession: in-distribution images may adapt the model
online for this session only, and are saved to `data/experience/<modality>/`
with their predictions. Images flagged out-of-distribution are shown but not saved
and never used for learning. Labels you confirm or correct here are used by
`scripts/consolidate.py`, which creates a new version only if it passes the gate.

    python app/demo_gradio.py            # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
import torch

from core.continual import ContinualSession, ExperienceStore
from core.registry import ModelRegistry

REJECT = "Không thuộc modality này"
DISCLAIMER = ("**Research only. Not a clinical diagnostic tool.** Model outputs are unvalidated "
              "and must not be used for patient care. Uploaded images are stored locally only.")


class DemoBackend:
    def __init__(self, registry_root: Path, experience_root: Path, device=None):
        self.registry_root = Path(registry_root)
        self.experience_root = Path(experience_root)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions: dict[str, tuple[str, ContinualSession]] = {}

    def modalities(self) -> list[str]:
        if not self.registry_root.is_dir():
            return []
        return sorted(d.name for d in self.registry_root.iterdir()
                      if d.is_dir() and ModelRegistry(self.registry_root, d.name).current_version())

    def registry(self, modality: str) -> ModelRegistry:
        return ModelRegistry(self.registry_root, modality)

    def session(self, modality: str) -> tuple[str, ContinualSession]:
        version = self.registry(modality).current_version()
        cached = self.sessions.get(modality)
        if cached is None or cached[0] != version:
            cached = (version, ContinualSession(self.registry(modality).load(), self.device))
            self.sessions[modality] = cached
        return cached

    def labels(self, modality: str) -> list[str]:
        return self.session(modality)[1].labels

    def analyze(self, modality: str, paths: list[str], adapt: bool) -> list[dict]:
        version, session = self.session(modality)
        session.adapt_enabled = adapt
        images = [Image.open(p).convert("RGB") for p in paths]
        store = ExperienceStore(self.experience_root / modality)
        rows = []
        for image, result in zip(images, session.predict(images)):
            image_id = None
            if not result["ood"]:
                image_id = store.add(image, {
                    "modality": modality, "model_version": version, "prediction": result["label"],
                    "confidence": result["confidence"], "ood": False, "agreement": result["agreement"],
                    "source_label": result["source_label"],
                })
            rows.append({"id": image_id, **result})
        return rows

    def set_label(self, modality: str, image_id: str, label: str) -> None:
        if label != REJECT and label not in self.labels(modality):
            raise ValueError(f"Unknown label {label}")
        ExperienceStore(self.experience_root / modality).set_label(image_id, "rejected" if label == REJECT else label)

    def status(self, modality: str) -> str:
        version, session = self.session(modality)
        return (f"Model {modality}/{version} | online updates {session.updates} | "
                f"resets {session.resets} | buffered {len(session.pending)}")

    def versions_table(self, modality: str) -> list[list]:
        registry = self.registry(modality)
        current = registry.current_version()
        return [[v, info["created"], round(info["metrics"].get("macro_f1", float("nan")), 4),
                 info["notes"], "✓" if v == current else ""]
                for v, info in registry.versions().items()]

    def rollback(self, modality: str) -> str:
        version = self.registry(modality).rollback()
        self.sessions.pop(modality, None)
        return version


def build_ui(backend: DemoBackend):
    import gradio as gr

    modalities = backend.modalities()
    with gr.Blocks(title="CAFA-TTA continual demo") as demo:
        gr.Markdown("# CAFA-TTA | Continual learning demo")
        gr.Markdown(DISCLAIMER)
        if not modalities:
            gr.Markdown("No model in the registry yet. Train one first, for example:\n\n"
                        "`python scripts/train_modality.py --modality histopathology --data-dir <ImageFolder>`")
            return demo

        with gr.Row():
            modality = gr.Dropdown(modalities, value=modalities[0], label="Modality")
            adapt = gr.Checkbox(True, label="Adapt online in this session")
            reset_btn = gr.Button("Reset session")
        status = gr.Markdown(backend.status(modalities[0]))

        files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Images")
        analyze_btn = gr.Button("Analyze", variant="primary")
        table = gr.Dataframe(headers=["id", "prediction", "confidence", "OOD", "agreement", "source model"],
                             interactive=False, label="Results")
        top = gr.Label(num_top_classes=5, label="Probabilities (first image)")

        gr.Markdown("### Confirm or correct a prediction")
        with gr.Row():
            image_id = gr.Dropdown([], label="Image id")
            label = gr.Dropdown(backend.labels(modalities[0]) + [REJECT], label="Correct label")
            save_btn = gr.Button("Save label")
        feedback = gr.Markdown()

        gr.Markdown("### Model versions")
        versions = gr.Dataframe(backend.versions_table(modalities[0]),
                                headers=["version", "created", "val macro-F1", "notes", "current"],
                                interactive=False)
        rollback_btn = gr.Button("Roll back to previous version")
        gr.Markdown("New versions are created offline: "
                    "`python scripts/consolidate.py --modality <name> --source-data-dir <ImageFolder>`")

        def on_analyze(mod, paths, adapt_on):
            if not paths:
                return [], {}, gr.Dropdown(choices=[]), backend.status(mod)
            rows = backend.analyze(mod, paths, adapt_on)
            data = [[r["id"][:12] if r["id"] else "-", r["label"], round(r["confidence"], 3),
                     "yes" if r["ood"] else "", "yes" if r["agreement"] else "", r["source_label"]] for r in rows]
            ids = [r["id"] for r in rows if r["id"]]
            note = backend.status(mod) + (" | drift detected: session reset to source"
                                          if any(r["session_reset"] for r in rows) else "")
            return data, rows[0]["probabilities"], gr.Dropdown(choices=ids, value=ids[0] if ids else None), note

        def on_modality(mod):
            return (backend.status(mod), gr.Dropdown(choices=backend.labels(mod) + [REJECT], value=None),
                    backend.versions_table(mod))

        def on_save(mod, img_id, lab):
            if not img_id or not lab:
                return "Choose an image id and a label."
            backend.set_label(mod, img_id, lab)
            return f"Saved label '{lab}' for {img_id[:12]}."

        def on_reset(mod):
            backend.session(mod)[1].reset()
            return backend.status(mod)

        def on_rollback(mod):
            try:
                version = backend.rollback(mod)
            except RuntimeError as err:
                return backend.status(mod) + f" | {err}", backend.versions_table(mod)
            return backend.status(mod) + f" | rolled back to {version}", backend.versions_table(mod)

        analyze_btn.click(on_analyze, [modality, files, adapt], [table, top, image_id, status])
        modality.change(on_modality, modality, [status, label, versions])
        save_btn.click(on_save, [modality, image_id, label], feedback)
        reset_btn.click(on_reset, modality, status)
        rollback_btn.click(on_rollback, modality, [status, versions])
    return demo


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    parser.add_argument("--experience-root", type=Path, default=ROOT / "data/experience")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    # Bound to localhost: uploaded medical images must not be exposed on the network.
    build_ui(DemoBackend(args.registry_root, args.experience_root)).launch(
        server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
