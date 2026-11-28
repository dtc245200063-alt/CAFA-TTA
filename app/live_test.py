"""Labelled live test and saved-results comparison for the Gradio demo (histopathology).

`run_comparison` streams a labelled subset of an external dataset through several
methods in the same online batch-prequential protocol as
`scripts/evaluate_histopathology_external.py`. Labels are used only for scoring.
`results_frame` reads the frozen JSON outputs of that evaluator (runs/overnight/vN.json).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_histopathology_external import KATHER_TO_NCT, Evaluator  # noqa: E402

DATASETS = {"Kather 2016 (phòng lab khác)": "kather2016",
            "Kather 2016, nhóm B (bệnh nhân 02, 04, 08, 10)": "kather2016#B",
            "CRC-VAL-HE-7K (cùng nguồn, độc lập)": "crc_val_he_7k"}
# Kather 2016 patients used, unlabelled, by the continual-learning rounds (scripts/continual_resnet.py).
# Group B never is, so it is the fair test set for a consolidated model.
KATHER_GROUP_B = {"02", "04", "08", "10"}
METHODS = {"Không thích nghi": "source", "BN-adapt": "bn_adapt", "CAFA v3 gốc": "cafa_v3",
           "Session mới (bộ nhớ cân bằng)": "session_b8"}
SHORT = {"source": "Gốc", "bn_adapt": "BN", "cafa_v3": "CAFA", "session_b8": "Session"}  # gallery captions
ORDERS = {"Xáo trộn": False, "Theo ca (từng loại mô liên tiếp)": True}


def labelled_paths(folder: Path, labels: list[str]) -> list[tuple[Path, int]]:
    """(path, label index) for an ImageFolder whose class names are NCT or Kather 2016 names.

    A folder named `kather2016#B` means Kather 2016 restricted to patient group B.
    """
    folder = Path(folder)
    group_b = folder.name.endswith("#B")
    if group_b:
        folder = folder.with_name(folder.name[:-2])
    out = []
    for p in sorted(folder.glob("*/*")):
        if p.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        if group_b and (re.search(r"HE-(\d+)", p.name) or [None, ""])[1] not in KATHER_GROUP_B:
            continue
        name = KATHER_TO_NCT.get(p.parent.name, p.parent.name)
        if name in labels:
            out.append((p, labels.index(name)))
    return out


def stratified_subset(items: list[tuple[Path, int]], n: int, seed: int) -> list[tuple[Path, int]]:
    """About n items with the same number per class (fewer if a class is small)."""
    rng = np.random.default_rng(seed)
    classes = sorted({y for _, y in items})
    per_class = max(1, n // max(1, len(classes)))
    chosen = []
    for c in classes:
        idx = [i for i, (_, y) in enumerate(items) if y == c]
        chosen += [items[i] for i in rng.choice(idx, min(per_class, len(idx)), replace=False)]
    return chosen


def run_comparison(checkpoint: dict, folder: Path, n: int, methods: list[str], class_ordered: bool,
                   seed: int = 0, device: str = "cpu", progress=None):
    """Return (summary DataFrame, per-class recall DataFrame, gallery [(image, caption)])."""
    ev = Evaluator(checkpoint, device)
    items = stratified_subset(labelled_paths(folder, ev.labels), n, seed)
    if not items:
        raise ValueError(f"No labelled images for this model's classes in {folder}")
    images = []
    for p, _ in items:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    y = np.array([label for _, label in items])

    summary, recall, preds = [], [], {}
    for k, method in enumerate(methods):
        if progress:
            progress(k / len(methods), desc=f"Đang chạy {method}")
        r = ev.stream(method, images, y, seed, class_ordered=class_ordered, keep_predictions=True)
        preds[method] = dict(zip(r["order"], r["predictions"]))
        summary.append({"Phương pháp": method, "Độ chính xác (%)": round(100 * r["accuracy"], 1),
                        "Chấm theo định nghĩa lớp Kather 2016 (%)": round(100 * r["accuracy_harmonised"], 1),
                        "Cập nhật online": r.get("updates", "-"), "Reset": r.get("resets", "-")})
        recall += [{"Phương pháp": method, "Lớp": c, "Recall (%)": round(100 * v, 1)} for c, v in r["recall"].items()]

    # Gallery: up to 2 images per class, caption shows true label and each method's prediction.
    gallery = []
    for c in sorted(set(y.tolist())):
        for i in np.flatnonzero(y == c)[:2]:
            caption = f"Đúng {ev.labels[c]} | " + " | ".join(
                f"{SHORT.get(m, m)} {ev.labels[preds[m][i]]}{'' if preds[m][i] == c else '✗'}" for m in methods)
            gallery.append((images[i], caption))
    return pd.DataFrame(summary), pd.DataFrame(recall), gallery


def image_stats(image: Image.Image) -> dict:
    """Colour and stain statistics that show the lab-to-lab difference by eye."""
    from skimage.color import rgb2hed

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255
    hed = rgb2hed(rgb)
    return {"rgb": tuple(int(v) for v in rgb.reshape(-1, 3).mean(0) * 255),
            "brightness": float(hsv[..., 2].mean()), "saturation": float(hsv[..., 1].mean()),
            "hematoxylin": float(hed[..., 0].mean()), "eosin": float(hed[..., 1].mean())}


class PairComparer:
    """Predict two images with the source model and with a session warmed on each image's dataset.

    The warm-up stream (unlabelled, class-balanced, shuffled) stands in for the images a lab would
    have seen before this one; the compared images themselves are predicted without adapting.
    """

    def __init__(self, data_root: Path, device: str = "cpu", warmup: int = 256):
        self.data_root, self.device, self.warmup = Path(data_root), device, warmup
        self.evaluators: dict[str, Evaluator] = {}
        self.sessions: dict[tuple[str, str], object] = {}

    def evaluator(self, version: str, checkpoint: dict) -> Evaluator:
        if version not in self.evaluators:
            self.evaluators[version] = Evaluator(checkpoint, self.device)
        return self.evaluators[version]

    def session(self, version: str, checkpoint: dict, dataset: str):
        from core.continual import ContinualSession

        key = (version, dataset)
        if key not in self.sessions:
            session = ContinualSession(checkpoint, self.device, batch_size=8)
            items = stratified_subset(labelled_paths(self.data_root / dataset, session.labels), self.warmup, seed=1)
            order = np.random.default_rng(1).permutation(len(items))
            for i in range(0, len(order), 64):
                batch = []
                for j in order[i:i + 64]:
                    with Image.open(items[j][0]) as im:
                        batch.append(im.convert("RGB"))
                session.predict(batch)
            session.adapt_enabled = False  # frozen after warm-up: comparing never changes it
            self.sessions[key] = session
        return self.sessions[key]

    def random_image(self, dataset: str, label: str, labels: list[str], seed=None) -> Image.Image:
        items = [p for p, y in labelled_paths(self.data_root / dataset, labels) if labels[y] == label]
        if not items:
            raise ValueError(f"No {label} images in {dataset}")
        with Image.open(items[np.random.default_rng(seed).integers(len(items))]) as im:
            return im.convert("RGB")

    def compare(self, version: str, checkpoint: dict, sides: list[tuple]):
        """sides: [(name, image, dataset folder, true label or None)].

        Returns (parameter table, probability table) as DataFrames.
        """
        from core.continual import OODGate

        ev = self.evaluator(version, checkpoint)
        table, probs = {}, []
        for name, image, dataset, truth in sides:
            f, l = ev.forward(ev.source, [image])
            p_src = torch.softmax(l.float() / ev.T, dim=1)[0].cpu()
            energy = float(OODGate.energy(l, ev.gate.temperature)[0])
            similarity = float(OODGate.max_similarity(f, ev.protos)[0])
            adapted = self.session(version, checkpoint, dataset).predict([image])[0]
            stats = image_stats(image)

            def verdict(label):
                return "" if truth is None else (" ✓" if label == truth else " ✗")

            src_label = ev.labels[int(p_src.argmax())]
            table[name] = [
                dataset, truth or "không rõ (ảnh tải lên)",
                f"{src_label} ({100 * float(p_src.max()):.1f}%){verdict(src_label)}",
                f"{adapted['label']} ({100 * adapted['confidence']:.1f}%){verdict(adapted['label'])}",
                "chặn, không học" if adapted["ood"] else "cho phép học",
                f"{energy:.2f} (chặn nếu > {ev.gate.energy_threshold:.2f})",
                f"{similarity:.3f} (chặn nếu < {ev.gate.similarity_threshold:.3f})",
                "R {} G {} B {}".format(*stats["rgb"]),
                f"{stats['brightness']:.3f}", f"{stats['saturation']:.3f}",
                f"{stats['hematoxylin']:.4f}", f"{stats['eosin']:.4f}",
            ]
            for k, label in enumerate(ev.labels):
                probs.append({"Lớp": label, "Ảnh": f"{name}, gốc", "Xác suất (%)": 100 * float(p_src[k])})
                probs.append({"Lớp": label, "Ảnh": f"{name}, sau thích nghi",
                              "Xác suất (%)": 100 * adapted["probabilities"][label]})
        rows = ["Tập dữ liệu", "Nhãn thật", "Mô hình gốc dự đoán", "Sau thích nghi dự đoán", "Cổng ảnh lạ",
                "Energy", "Tương đồng prototype", "Màu trung bình", "Độ sáng (0–1)", "Độ bão hòa (0–1)",
                "Hematoxylin (nhân, tím)", "Eosin (bào tương, hồng)"]
        return pd.DataFrame({"Thông số": rows, **table}), pd.DataFrame(probs)


RESULT_ROWS = [  # (label, key path in the evaluator JSON); every value is a fraction. "K16" = Kather 2016 class definitions
    ("CRC-VAL sạch", ("crc_val_clean", "accuracy")),
    ("CRC-VAL lệch màu", ("crc_val_colour_shift", "accuracy")),
    ("Kather, không thích nghi (nghiêm ngặt)", ("kather2016_source_static", "accuracy")),
    ("Kather, không thích nghi (K16)", ("kather2016_source_static", "accuracy_harmonised")),
    ("Kather xáo trộn, CAFA v3 gốc (K16)", ("kather2016_stream", "cafa_v3", "acc_harmonised_mean")),
    ("Kather xáo trộn, Session mới (K16)", ("kather2016_stream", "session_b8", "acc_harmonised_mean")),
    ("Kather theo ca, CAFA v3 gốc (K16)", ("kather2016_stream_class_ordered", "cafa_v3", "acc_harmonised_mean")),
    ("Kather theo ca, Session mới (K16)", ("kather2016_stream_class_ordered", "session_b8", "acc_harmonised_mean")),
    ("Ảnh không phải mô bị chặn", ("synthetic_non_tissue_rejected",)),
]


def results_frame(results_dir: Path) -> pd.DataFrame:
    """Long table (Chỉ số, Mô hình, Giá trị %) from every vN.json the evaluator wrote."""
    rows = []
    for path in sorted(Path(results_dir).glob("v[0-9]*.json")):
        if not path.stem[1:].isdigit():
            continue  # skip partial files such as v1_static.json
        data = json.loads(path.read_text(encoding="utf-8"))
        model = f"{data['version']} ({data['architecture']}, {data.get('augment') or 'light'})"
        for label, keys in RESULT_ROWS:
            value = data
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                rows.append({"Chỉ số": label, "Mô hình": model, "Giá trị (%)": round(100 * value, 1)})
    return pd.DataFrame(rows, columns=["Chỉ số", "Mô hình", "Giá trị (%)"])
