"""CAFA-TTA unified histopathology research demo.

Uses a single 17-output ResNet18 checkpoint trained on PathMNIST and Kather 2016.
The 17 labels are dataset-prefixed and are NOT a harmonized medical taxonomy.
This is source-model inference, NOT online CAFA-TTA adaptation.
"""
from __future__ import annotations

import json
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/medmnist/pathmnist.npz"
CHECKPOINT = ROOT / "checkpoints/cafa_unified_balanced_best.pt"
METRICS = ROOT / "experiments/unified_balanced_paper_01/cafa_unified_balanced_best.metrics.json"
OLD_RUN_ROOT = ROOT / "docker-project/docker_runs"


class CAFAApp:
    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("CAFA-TTA | Unified 17-Class Research AI")
        self.window.geometry("1060x780")
        self.window.minsize(850, 640)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.labels: list[str] = []
        self.preprocess = None
        self.photo = None
        self.current_image: Image.Image | None = None
        self.is_pathmnist = False
        self.true_class: int | None = None
        self.images = None
        self.targets = None
        if DATA.is_file():
            with np.load(DATA, allow_pickle=False) as data:
                self.images = data["test_images"].copy()
                self.targets = data["test_labels"].reshape(-1).copy()

        self.index_var = tk.StringVar(value="0")
        self.truth_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=f"Ready | {self.device.type.upper()}")
        self.pred_var = tk.StringVar(value="Choose an image, then click Run AI Prediction")
        self.conf_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="")
        self.build_ui()
        if self.images is not None:
            self.show_image()
        else:
            self.truth_var.set("PathMNIST dataset not installed; use Choose JPG/PNG/TIFF")
            self.source_var.set("Source: external image")

    def build_ui(self):
        ttk.Style().theme_use("clam")
        root = ttk.Frame(self.window, padding=16)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="CAFA-TTA | Unified Histopathology AI",
                  font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(root, text="ONE ResNet18 | ONE checkpoint | 17 dataset-prefixed output classes",
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 12))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Model: Unified ResNet18", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 12))
        ttk.Label(controls, text="PathMNIST index:").pack(side="left", padx=(0, 4))
        entry = ttk.Entry(controls, textvariable=self.index_var, width=7)
        entry.pack(side="left", padx=(0, 5))
        for text, command in (
            ("Load Image", self.show_image),
            ("Random Image", self.random_image),
            ("Choose JPG/PNG/TIFF", self.open_custom_image),
            ("Experiment Results", self.show_experiment_results),
        ):
            ttk.Button(controls, text=text, command=command).pack(side="left", padx=3)

        content = ttk.Frame(root)
        content.pack(fill="both", expand=True)
        left = ttk.LabelFrame(content, text="Input histopathology image", padding=12)
        left.pack(side="left", fill="y", padx=(0, 12))
        self.image_label = ttk.Label(left)
        self.image_label.pack(pady=12)
        ttk.Label(left, textvariable=self.truth_var, wraplength=275,
                  font=("Segoe UI", 11, "bold")).pack(pady=(8, 4))
        ttk.Label(left, textvariable=self.source_var, wraplength=275).pack(pady=(0, 14))
        self.run_button = ttk.Button(left, text="RUN AI PREDICTION", command=self.predict)
        self.run_button.pack(fill="x", pady=10, ipady=8)

        right = ttk.LabelFrame(content, text="AI prediction results", padding=12)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, textvariable=self.pred_var, font=("Segoe UI", 14, "bold"),
                  wraplength=550).pack(anchor="w", pady=(0, 8))
        ttk.Label(right, textvariable=self.conf_var,
                  font=("Segoe UI", 15)).pack(anchor="w", pady=4)
        ttk.Label(right, textvariable=self.result_var,
                  font=("Segoe UI", 11, "bold"), wraplength=520).pack(anchor="w", pady=(0, 12))
        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="17-CLASS PROBABILITIES",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=5)
        # Scrollable area: 17 probabilities cannot reliably fit in a fixed pane.
        canvas = tk.Canvas(right, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.bar_frame = ttk.Frame(canvas)
        self.bar_frame.bind("<Configure>",
                            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.bar_window = canvas.create_window((0, 0), window=self.bar_frame, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(self.bar_window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.bars = []
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", pady=(10, 3))
        ttk.Label(root, text="Research only. Not a clinical diagnostic tool. "
                  "CAFA-TTA v3 online adaptation is not yet enabled in this GUI.",
                  wraplength=1010).pack(anchor="w")

    def create_probability_bars(self, names: list[str]):
        for child in self.bar_frame.winfo_children():
            child.destroy()
        self.bars.clear()
        for name in names:
            line = ttk.Frame(self.bar_frame)
            line.pack(fill="x", pady=3)
            ttk.Label(line, text=name, width=24, anchor="w").pack(side="left")
            bar = ttk.Progressbar(line, maximum=100, value=0)
            bar.pack(side="left", fill="x", expand=True, padx=(4, 8))
            pct = ttk.Label(line, text="0.0%", width=7, anchor="e")
            pct.pack(side="right")
            self.bars.append((bar, pct))

    def reset_prediction(self):
        self.pred_var.set("Click Run AI Prediction")
        self.conf_var.set("")
        self.result_var.set("")
        for bar, pct in self.bars:
            bar["value"] = 0
            pct.configure(text="0.0%")

    def display_image(self, image: Image.Image):
        preview = image.copy()
        preview.thumbnail((285, 285))
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.photo)
        self.current_image = image
        self.reset_prediction()

    def show_image(self):
        if self.images is None:
            messagebox.showinfo("Dataset missing", f"PathMNIST not found at:\n{DATA}\nYou can still upload an image.")
            return
        try:
            i = int(self.index_var.get())
            if i < 0 or i >= len(self.images):
                raise ValueError(f"Image index must be 0–{len(self.images)-1}")
            image = Image.fromarray(self.images[i]).convert("RGB")
            self.is_pathmnist = True
            self.true_class = int(self.targets[i])
            self.display_image(image.resize((280, 280), Image.Resampling.NEAREST))
            # Keep *original* 28x28 image for the model; preview is for display only.
            self.current_image = image
            self.truth_var.set(f"True label: PathMNIST class {self.true_class}")
            self.source_var.set("Source: PathMNIST test split")
            self.status_var.set(f"PathMNIST image #{i} | {self.device.type.upper()}")
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))

    def random_image(self):
        if self.images is None:
            messagebox.showinfo("Dataset missing", "PathMNIST dataset is not installed.")
            return
        self.index_var.set(str(random.randrange(len(self.images))))
        self.show_image()

    def open_custom_image(self):
        path = filedialog.askopenfilename(
            title="Choose histopathology image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if not path:
            return
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            self.is_pathmnist = False
            self.true_class = None
            self.display_image(image)
            self.truth_var.set("External image — ground truth unknown")
            self.source_var.set(f"Source: {Path(path).name}")
            self.status_var.set(f"Loaded external image | {self.device.type.upper()}")
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))

    def load_model(self):
        if self.model is not None:
            return
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Unified checkpoint missing:\n{CHECKPOINT}")
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "resnet18":
            raise ValueError("Checkpoint is not a supported ResNet18")
        labels = checkpoint.get("labels")
        if not isinstance(labels, list) or len(labels) != 17 or len(set(labels)) != 17:
            raise ValueError("Expected exactly 17 distinct checkpoint labels")
        size = int(checkpoint["input_size"])
        norm = checkpoint["normalization"]
        mean, std = norm["mean"], norm["std"]
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError("Unexpected checkpoint normalization")
        # Match train_unified.py's build_model(): torchvision ResNet18 + 17-way fc.
        model = models.resnet18(weights=None, num_classes=17)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.labels = labels
        self.model = model
        self.create_probability_bars(self.labels)
        self.truth_var.set(self.truth_var.get())

    def predict(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Choose an image first.")
            return
        self.run_button.configure(state="disabled")
        try:
            self.status_var.set("Running unified inference...")
            self.window.update_idletasks()
            self.load_model()
            x = self.preprocess(self.current_image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                probabilities = self.model(x).softmax(1)[0].cpu().numpy()
            top = int(np.argmax(probabilities))
            self.pred_var.set(f"Predicted: {self.labels[top]}")
            self.conf_var.set(f"Confidence: {float(probabilities[top]) * 100:.2f}%")
            if self.is_pathmnist and self.true_class is not None:
                expected_index = self.true_class
                self.result_var.set("CORRECT PREDICTION" if top == expected_index
                                    else "INCORRECT PREDICTION")
            else:
                self.result_var.set("EXTERNAL IMAGE — NO VERIFIED GROUND TRUTH")
            for i, (bar, pct) in enumerate(self.bars):
                value = float(probabilities[i]) * 100
                bar["value"] = value
                pct.configure(text=f"{value:.1f}%")
            self.status_var.set(f"Completed | Unified ResNet18 | {self.device.type.upper()}")
        except Exception as exc:
            self.status_var.set("Inference failed")
            messagebox.showerror("AI Prediction Error", str(exc))
        finally:
            self.run_button.configure(state="normal")

    def show_experiment_results(self):
        dialog = tk.Toplevel(self.window)
        dialog.title("CAFA-TTA | Experiment Results")
        dialog.geometry("750x560")
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="CAFA-TTA Experiment Results",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 10))
        display = tk.Text(frame, wrap="word", font=("Consolas", 10))
        display.pack(fill="both", expand=True)

        def write(line=""):
            display.insert("end", str(line) + "\n")

        write("UNIFIED RESNET18 — CLEAN TEST (17 labels)")
        write("=" * 62)
        if METRICS.is_file():
            try:
                metrics = json.loads(METRICS.read_text(encoding="utf-8-sig"))
                for title, key in (("PathMNIST", "test_pathmnist"),
                                   ("Kather 2016", "test_kather")):
                    result = metrics[key]
                    write(f"{title}: Accuracy {result['accuracy'] * 100:.2f}% | "
                          f"Macro-F1 {result['macro_f1']:.4f} | "
                          f"n={result['samples']}")
                write(f"Training epochs: {len(metrics.get('history', []))} | "
                      f"selected epoch: {metrics.get('best_epoch', 'N/A')}")
            except Exception as exc:
                write(f"Unable to read unified metrics: {exc}")
        else:
            write(f"Unified metrics not found: {METRICS}")
        write()
        write("OLDER SEPARATE-MODEL BENCHMARKS (NOT THE UNIFIED MODEL)")
        write("=" * 62)
        for title, path in (
            ("PathMNIST baseline", OLD_RUN_ROOT / "baseline_full_cpu_01/test_metrics.json"),
            ("Kather baseline", OLD_RUN_ROOT / "kather2016_cpu_01/test_metrics.json"),
        ):
            if path.is_file():
                try:
                    result = json.loads(path.read_text(encoding="utf-8-sig"))
                    write(f"{title}: Accuracy {result['accuracy'] * 100:.2f}% | "
                          f"Macro-F1 {result['macro_f1']:.4f}")
                except Exception as exc:
                    write(f"{title}: cannot read: {exc}")
            else:
                write(f"{title}: not available in this copy")
        write()
        path = OLD_RUN_ROOT / "kather2016_cpu_01/cafa_comparison_shifted_seed42/comparison_metrics.json"
        write("Kather shifted comparison (OLDER SEPARATE MODEL)")
        if path.is_file():
            try:
                comparison = json.loads(path.read_text(encoding="utf-8-sig"))
                for result in comparison.get("results", []):
                    write(f"{result['method']}: accuracy {result['accuracy']*100:.2f}% "
                          f"macro-F1 {result['macro_f1']:.4f}")
            except Exception as exc:
                write(f"Cannot read old shifted comparison: {exc}")
        else:
            write("Not available in this copy")
        write()
        write("Research only. CAFA-TTA v3 adaptation is NOT yet integrated.")
        display.configure(state="disabled")


if __name__ == "__main__":
    app_window = tk.Tk()
    CAFAApp(app_window)
    app_window.mainloop()
