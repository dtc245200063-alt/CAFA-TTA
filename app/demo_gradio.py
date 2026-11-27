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


def build_ui(backend: DemoBackend, data_root: Path | None = None, results_dir: Path | None = None):
    import gradio as gr

    modalities = backend.modalities()
    with gr.Blocks(title="CAFA-TTA continual demo") as demo:
        gr.Markdown("# CAFA-TTA | Continual learning demo")
        gr.Markdown(DISCLAIMER)
        if not modalities:
            gr.Markdown("No model in the registry yet. Train one first, for example:\n\n"
                        "`python scripts/train_modality.py --modality histopathology --data-dir <ImageFolder>`")
            return demo

        showcase = Path(results_dir or "").parent / "showcase"
        if (showcase / "showcase.json").is_file():
            with gr.Tab("Điểm nổi bật: trước và sau"):
                _showcase_tab(gr, showcase)
        with gr.Tab("Demo: phân tích ảnh"):
            _analysis_tab(gr, backend, modalities, data_root)
        if "histopathology" in modalities and data_root and Path(data_root).is_dir():
            with gr.Tab("So sánh 2 ảnh"):
                _pair_tab(gr, backend, Path(data_root))
            with gr.Tab("Test thực tế: so sánh phương pháp"):
                _live_test_tab(gr, backend, Path(data_root))
        if results_dir and Path(results_dir).is_dir():
            with gr.Tab("Kết quả thí nghiệm"):
                _results_tab(gr, Path(results_dir))
    return demo


def grouped_bars(frame, x: str, group: str, value: str):
    """Matplotlib grouped bar chart (one bar per `group` within each `x`), values in percent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = frame.pivot_table(index=x, columns=group, values=value, sort=False)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(table)), 4))
    width = 0.8 / max(1, len(table.columns))
    for k, name in enumerate(table.columns):
        bars = ax.bar(np.arange(len(table)) + k * width, table[name], width, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.set_xticks(np.arange(len(table)) + 0.4 - width / 2, table.index, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel(value)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _showcase_tab(gr, folder: Path):
    """Before/after view written by scripts/build_showcase.py."""
    import json
    import random

    data = json.loads((folder / "showcase.json").read_text(encoding="utf-8"))
    keys = list(data["accuracy"])
    first, last = keys[0], keys[-1]
    acc = {k: 100 * v for k, v in data["accuracy"].items()}
    rescued = [r for r in data["images"] if r[first] != r["true"] and r[last] == r["true"]]
    broken = [r for r in data["images"] if r[first] == r["true"] and r[last] != r["true"]]
    gr.Markdown(
        f"## Kather 2016: ảnh từ một phòng lab khác, mô hình chưa từng thấy\n"
        f"# {acc[first]:.1f}% → {acc[keys[1]]:.1f}% → **{acc[last]:.1f}%**\n"
        f"*{first}* → *{keys[1]}* → *{last}*. Trên {len(data['images']):,} ảnh có "
        f"**{len(rescued):,} ảnh từ sai thành đúng** và {len(broken):,} ảnh từ đúng thành sai. "
        "Phần tự thích nghi không dùng nhãn; nhãn chỉ dùng để chấm điểm. "
        "Một tập ngoài duy nhất, không phải đánh giá lâm sàng.")
    gr.Image(str(folder / "summary.png"), label="Tổng quan", show_label=False)
    gr.Image(str(folder / "examples.png"), label="Ví dụ", show_label=False)
    gr.Markdown("### Xem thêm ảnh ngẫu nhiên được sửa đúng")
    more = gr.Button("Lấy 8 ảnh khác")
    gallery = gr.Gallery(columns=4, height="auto", label="Trước ✗ → Sau ✓")

    def on_more():
        picks = random.sample(rescued, min(8, len(rescued)))
        # Browsers cannot show TIFF: hand Gradio PIL images, which it re-encodes as PNG.
        return [(Image.open(r["path"]).convert("RGB"), f"Thật {r['true']} | trước {r[first]} ✗ | sau {r[last]} ✓")
                for r in picks]

    more.click(on_more, None, gallery)


def _pair_tab(gr, backend: DemoBackend, data_root: Path):
    from app.live_test import DATASETS, PairComparer

    comparer = PairComparer(data_root, backend.device)
    registry = backend.registry("histopathology")
    labels = backend.labels("histopathology")
    gr.Markdown("Đặt hai ảnh cạnh nhau, ví dụ cùng loại mô nhưng từ hai phòng lab khác nhau. "
                "Mỗi ảnh được dự đoán bằng mô hình gốc và bằng session đã thích nghi trên 256 ảnh "
                "(không nhãn) của tập dữ liệu đã chọn cho ảnh đó. Có thể tải ảnh của bạn lên thay ảnh mẫu. "
                "Lần đầu với mỗi tập dữ liệu mất vài giây để thích nghi.")
    version = gr.Dropdown(list(registry.versions()), value=registry.current_version(), label="Phiên bản mô hình")
    sides = []
    with gr.Row():
        for name, ds, default in (("Ảnh A", 1, "TUM"), ("Ảnh B", 0, "TUM")):
            with gr.Column():
                gr.Markdown(f"### {name}")
                with gr.Row():
                    dataset = gr.Dropdown(list(DATASETS), value=list(DATASETS)[ds], label="Tập dữ liệu (lab)")
                    label = gr.Dropdown(labels, value=default, label="Loại mô")
                pick = gr.Button("Lấy ảnh mẫu ngẫu nhiên")
                image = gr.Image(comparer.random_image(DATASETS[list(DATASETS)[ds]], default, labels, seed=0),
                                 type="pil", label=name, height=300)
                truth = gr.State(default)
                sides.append((dataset, label, image, truth))

                def on_pick(ds_name, lab):
                    try:
                        return comparer.random_image(DATASETS[ds_name], lab, labels), lab
                    except ValueError as err:
                        raise gr.Error(str(err))

                pick.click(on_pick, [dataset, label], [image, truth])
                image.upload(lambda: None, None, truth)  # uploaded image: true label unknown
    compare_btn = gr.Button("So sánh", variant="primary")
    table = gr.Dataframe(label="Bảng thông số", interactive=False, wrap=True)
    plot = gr.Plot(label="Xác suất theo lớp: mô hình gốc và sau thích nghi")

    def on_compare(ver, ds_a, img_a, truth_a, ds_b, img_b, truth_b, progress=gr.Progress()):
        if img_a is None or img_b is None:
            raise gr.Error("Cần đủ hai ảnh: bấm 'Lấy ảnh mẫu ngẫu nhiên' hoặc tải ảnh lên.")
        progress(0.1, desc="Đang thích nghi và dự đoán")
        params, probs = comparer.compare(ver, registry.load(ver), [
            ("Ảnh A", img_a.convert("RGB"), DATASETS[ds_a], truth_a),
            ("Ảnh B", img_b.convert("RGB"), DATASETS[ds_b], truth_b)])
        return params, grouped_bars(probs, "Lớp", "Ảnh", "Xác suất (%)")

    (ds_a, _, img_a, truth_a), (ds_b, _, img_b, truth_b) = sides
    compare_btn.click(on_compare, [version, ds_a, img_a, truth_a, ds_b, img_b, truth_b], [table, plot])


def _live_test_tab(gr, backend: DemoBackend, data_root: Path):
    from app.live_test import DATASETS, METHODS, ORDERS, run_comparison

    gr.Markdown("Chạy một luồng ảnh **có nhãn** qua từng phương pháp, theo đúng giao thức online: "
                "dự đoán trước, rồi mới thích nghi trên lô đó. Nhãn chỉ dùng để chấm điểm, không dùng để học. "
                "Mỗi phương pháp bắt đầu lại từ mô hình gốc.")
    registry = backend.registry("histopathology")
    versions = list(registry.versions())
    with gr.Row():
        version = gr.Dropdown(versions, value=registry.current_version(), label="Phiên bản mô hình")
        dataset = gr.Dropdown(list(DATASETS), value=next(iter(DATASETS)), label="Tập dữ liệu")
        order = gr.Radio(list(ORDERS), value=next(iter(ORDERS)), label="Thứ tự ảnh đến")
    with gr.Row():
        n = gr.Slider(64, 2000, value=448, step=64, label="Số ảnh (chia đều theo lớp)")
        seed = gr.Number(0, precision=0, label="Seed")
        methods = gr.CheckboxGroup(list(METHODS), value=list(METHODS), label="Phương pháp")
    run_btn = gr.Button("Chạy test", variant="primary")
    summary = gr.Dataframe(label="Độ chính xác", interactive=False)
    recall = gr.Plot(label="Recall theo lớp mô")
    gallery = gr.Gallery(label="Ví dụ (✗ = dự đoán sai)", columns=4, height="auto")

    def on_run(ver, ds, order_name, count, s, chosen, progress=gr.Progress()):
        if not chosen:
            raise gr.Error("Chọn ít nhất một phương pháp.")
        table, per_class, pictures = run_comparison(
            registry.load(ver), data_root / DATASETS[ds], int(count), [METHODS[m] for m in chosen],
            ORDERS[order_name], int(s), backend.device, progress)
        names = {v: k for k, v in METHODS.items()}
        table["Phương pháp"] = table["Phương pháp"].map(names)
        per_class["Phương pháp"] = per_class["Phương pháp"].map(names)
        return table, grouped_bars(per_class, "Lớp", "Phương pháp", "Recall (%)"), pictures

    run_btn.click(on_run, [version, dataset, order, n, seed, methods], [summary, recall, gallery])


def _results_tab(gr, results_dir: Path):
    from app.live_test import results_frame

    frame = results_frame(results_dir)
    gr.Markdown(f"Kết quả đã lưu từ `scripts/evaluate_histopathology_external.py` (thư mục `{results_dir}`). "
                "Luồng Kather: trung bình 3 seed, so sánh theo cặp trên cùng dữ liệu, không phải thí nghiệm độc lập.")
    if frame.empty:
        gr.Markdown("Chưa có file kết quả `vN.json`.")
        return
    gr.Plot(grouped_bars(frame, "Chỉ số", "Mô hình", "Giá trị (%)"), label="So sánh mô hình")
    gr.Dataframe(frame.pivot(index="Chỉ số", columns="Mô hình", values="Giá trị (%)").reset_index(),
                 interactive=False, label="Bảng so sánh (%)")


def _example_images(data_root: Path | None) -> list[str]:
    """First image of each Kather 2016 class folder, if the dataset is present."""
    folder = Path(data_root or "") / "kather2016"
    if not folder.is_dir():
        return []
    return [str(next(d.glob("*.tif"))) for d in sorted(folder.iterdir()) if d.is_dir() and any(d.glob("*.tif"))]


def _analysis_tab(gr, backend: DemoBackend, modalities: list[str], data_root: Path | None):
    with gr.Row():
        modality = gr.Dropdown(modalities, value=modalities[0], label="Modality")
        adapt = gr.Checkbox(True, label="Adapt online in this session")
        reset_btn = gr.Button("Reset session")
    status = gr.Markdown(backend.status(modalities[0]))

    files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Images")
    examples = _example_images(data_root)
    if examples:
        gr.Examples([[[p]] for p in examples], inputs=files, label="Ảnh mẫu từ Kather 2016 (phòng lab khác)")
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    parser.add_argument("--experience-root", type=Path, default=ROOT / "data/experience")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-root", type=Path, default=Path("C:/Code/DataSet/histopathology"),
                        help="Folder with kather2016/ and crc_val_he_7k/ for the live test tab")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "runs/overnight",
                        help="Folder with vN.json from scripts/evaluate_histopathology_external.py")
    args = parser.parse_args(argv)
    # Bound to localhost: uploaded medical images must not be exposed on the network.
    allowed = [str(args.data_root)] if args.data_root.is_dir() else []
    allowed.append(str(args.results_dir.parent / "showcase"))
    build_ui(DemoBackend(args.registry_root, args.experience_root), args.data_root, args.results_dir).launch(
        server_name="127.0.0.1", server_port=args.port, allowed_paths=allowed)


if __name__ == "__main__":
    main()
