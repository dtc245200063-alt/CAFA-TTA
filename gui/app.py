from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import hashlib

import numpy as np
import torch
from torchvision import models
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/medmnist/pathmnist.npz"
CHECKPOINTS = ROOT / "checkpoints"

MODEL_FILES = {
    "Baseline": "resnet18_pathmnist_best.pt",
    "Enhanced": "resnet18_pathmnist_enhanced_best.pt",
    "Robust": "resnet18_pathmnist_robust_best.pt",
    "Balanced": "resnet18_pathmnist_balanced_best.pt",
}

CLASS_NAMES = [
    "ADI - Adipose",
    "BACK - Background",
    "DEB - Debris",
    "LYM - Lymphocytes",
    "MUC - Mucus",
    "MUS - Smooth muscle",
    "NORM - Normal mucosa",
    "STR - Stroma",
    "TUM - Tumor epithelium",
]

class CAFAApp:
    def __init__(self, window):
        self.window = window
        window.title("CAFA-TTA | Interactive Medical AI")
        window.geometry("980x740")
        window.minsize(850, 650)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not DATA.is_file():
            raise FileNotFoundError(DATA)

        with np.load(DATA, allow_pickle=False) as dataset:
            self.images = dataset["test_images"].copy()
            self.labels = dataset["test_labels"].reshape(-1).copy()

        assert self.images.shape == (7180, 28, 28, 3)
        assert len(self.labels) == 7180

        self.custom_image = None
        self.model = None
        self.loaded_model = None
        self.model_var = tk.StringVar(value="Enhanced")
        self.index_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(
            value="Ready | 7180 test images | " +
            str(self.device).upper()
        )
        self.pred_var = tk.StringVar(
            value="Click Run AI Prediction"
        )
        self.truth_var = tk.StringVar()
        self.conf_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="")

        self.build_ui()
        self.show_image()

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        outer = ttk.Frame(self.window, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="CAFA-TTA  |  Medical AI Dashboard",
            font=("Segoe UI", 20, "bold")
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text="Interactive inference on real PathMNIST test images",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 15))

        control = ttk.Frame(outer)
        control.pack(fill="x", pady=(0, 15))

        ttk.Label(control, text="Trained model:").pack(
            side="left", padx=(0, 6)
        )

        self.model_box = ttk.Combobox(
            control,
            textvariable=self.model_var,
            values=list(MODEL_FILES),
            state="readonly",
            width=15
        )
        self.model_box.pack(side="left", padx=(0, 18))
        self.model_box.bind(
            "<<ComboboxSelected>>",
            lambda event: self.reset_prediction()
        )

        ttk.Label(control, text="Image index:").pack(
            side="left", padx=(0, 6)
        )

        ttk.Entry(
            control,
            textvariable=self.index_var,
            width=8
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            control,
            text="Load Image",
            command=self.show_image
        ).pack(side="left", padx=4)

        ttk.Button(
            control,
            text="Random Image",
            command=self.random_image
        ).pack(side="left", padx=4)

        ttk.Button(
            control,
            text='Choose JPG/PNG',
            command=self.open_custom_image,
        ).pack(side='left', padx=4)

        ttk.Button(
            control,
            text="Experiment Results",
            command=self.show_experiment_results,
        ).pack(side="left", padx=4)

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)

        left = ttk.LabelFrame(
            content,
            text="Input Histopathology Image",
            padding=16
        )
        left.pack(side="left", fill="y", padx=(0, 12))

        self.image_label = ttk.Label(left)
        self.image_label.pack(pady=12)

        ttk.Label(
            left, textvariable=self.truth_var,
            font=("Segoe UI", 12, "bold"),
            wraplength=280
        ).pack(pady=12)

        ttk.Label(
            left,
            text="Source: PathMNIST test split",
        ).pack()

        self.run_button = ttk.Button(
            left,
            text="RUN AI PREDICTION",
            command=self.predict
        )
        self.run_button.pack(
            fill="x", pady=(32, 8), ipady=10
        )

        right = ttk.LabelFrame(
            content, text="AI Prediction Results",
            padding=16
        )
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(
            right,
            textvariable=self.pred_var,
            font=("Segoe UI", 15, "bold"),
            wraplength=480
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            right,
            textvariable=self.conf_var,
            font=("Segoe UI", 16)
        ).pack(anchor="w", pady=6)

        ttk.Label(
            right,
            textvariable=self.result_var,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(0, 20))

        ttk.Separator(right).pack(fill="x", pady=10)

        ttk.Label(
            right, text="PROBABILITIES BY CLASS",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 12))

        self.bars = []

        for name in CLASS_NAMES:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=4)

            ttk.Label(
                row, text=name, width=25
            ).pack(side="left")

            bar = ttk.Progressbar(
                row, maximum=100, value=0
            )
            bar.pack(side="left", fill="x", expand=True)

            value = ttk.Label(
                row, text="0.0%", width=8
            )
            value.pack(side="right", padx=(8, 0))

            self.bars.append((bar, value))

        ttk.Label(
            outer,
            textvariable=self.status_var,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(15, 0))

        ttk.Label(
            outer,
            text=(
                "Research demonstration only. "
                "Not a clinically validated diagnostic tool."
            )
        ).pack(anchor="w", pady=(5, 0))

    def open_custom_image(self):
        path = filedialog.askopenfilename(
            title="Choose a histopathology image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")],
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")

            small = image.resize(
                (28, 28), Image.Resampling.BILINEAR
            )
            self.custom_image = np.asarray(
                small, dtype=np.uint8
            ).copy()

            preview = image.copy()
            preview.thumbnail((280, 280))
            self.photo = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=self.photo)

            self.truth_var.set(
                "External image - ground truth unknown"
            )
            self.reset_prediction()
            self.status_var.set(
                "Image loaded: " + Path(path).name
            )
        except Exception as error:
            messagebox.showerror("Image error", str(error))

    def show_experiment_results(self):
        window = tk.Toplevel(self.window)
        window.title("CAFA-TTA | Experiment Results")
        window.geometry("720x540")

        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="CAFA-TTA Experiment Results",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        output = tk.Text(
            outer,
            wrap="word",
            font=("Consolas", 11),
        )
        output.pack(fill="both", expand=True)

        def add_line(value=""):
            output.insert("end", str(value) + "\n")

        run_root = ROOT / "docker-project" / "docker_runs"

        baseline_files = [
            (
                "PathMNIST Baseline - Clean Test",
                run_root / "baseline_full_cpu_01"
                / "test_metrics.json",
            ),
            (
                "Kather 2016 Baseline - Clean Test",
                run_root / "kather2016_cpu_01"
                / "test_metrics.json",
            ),
        ]

        for title, path in baseline_files:
            add_line("=" * 52)
            add_line(title)
            add_line("=" * 52)

            if not path.is_file():
                add_line("Results not available in this copy.")
                add_line()
                continue

            try:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )
                add_line(
                    f"Accuracy : {data['accuracy'] * 100:.2f}%"
                )
                add_line(
                    f"Macro-F1 : {data['macro_f1']:.4f}"
                )
                add_line(
                    f"Samples  : {data.get('samples', 'N/A')}"
                )
            except Exception as error:
                add_line(f"Cannot read results: {error}")

            add_line()

        add_line("=" * 52)
        add_line("Kather 2016 - Shifted Test Comparison")
        add_line("=" * 52)

        comparison = (
            run_root / "kather2016_cpu_01"
            / "cafa_comparison_shifted_seed42"
            / "comparison_metrics.json"
        )

        if comparison.is_file():
            try:
                data = json.loads(
                    comparison.read_text(encoding="utf-8")
                )
                add_line(
                    f"Seed: {data.get('seed', 'N/A')}"
                )
                for result in data["results"]:
                    add_line(
                        f"{result['method']:12s} "
                        f"Accuracy: {result['accuracy'] * 100:6.2f}% "
                        f"F1: {result['macro_f1']:.4f}"
                    )
            except Exception as error:
                add_line(f"Cannot read results: {error}")
        else:
            add_line("Results not available in this copy.")

        add_line()
        add_line(
            "Research demonstration only. "
            "Not a clinical diagnostic tool."
        )

        output.configure(state="disabled")

    def current_index(self):
        index = int(self.index_var.get())
        if not 0 <= index < len(self.images):
            raise ValueError(
                "Image index must be between 0 and 7179."
            )
        return index

    def reset_prediction(self):
        self.pred_var.set("Click Run AI Prediction")
        self.conf_var.set("")
        self.result_var.set("")
        for bar, label in self.bars:
            bar["value"] = 0
            label.config(text="0.0%")

    def show_image(self):
        try:
            index = self.current_index()
            self.custom_image = None
            image = Image.fromarray(self.images[index])
            image = image.resize(
                (280, 280),
                Image.Resampling.NEAREST
            )

            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)

            true_class = int(self.labels[index])
            self.truth_var.set(
                f"True label: {true_class} - "
                f"{CLASS_NAMES[true_class]}"
            )

            self.reset_prediction()

        except Exception as error:
            messagebox.showerror("Image error", str(error))

    def random_image(self):
        self.index_var.set(
            str(random.randrange(len(self.images)))
        )
        self.show_image()

    def load_model(self):
        selected = self.model_var.get()

        if (
            self.model is not None
            and self.loaded_model == selected
        ):
            return

        path = CHECKPOINTS / MODEL_FILES[selected]

        if not path.is_file():
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        if int(checkpoint["num_classes"]) != 9:
            raise ValueError("Unexpected number of classes")

        model = models.resnet18(
            weights=None,
            num_classes=9
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.model = model.to(self.device).eval()
        self.loaded_model = selected

    def predict(self):
        try:
            index = self.current_index()

            self.run_button.config(state="disabled")
            self.status_var.set("Running inference...")
            self.window.update_idletasks()

            self.load_model()

            image = (
                self.custom_image
                if self.custom_image is not None
                else self.images[index]
            )

            x = torch.from_numpy(
                image.transpose(2, 0, 1).copy()
            ).float().unsqueeze(0)

            x = ((x / 255.0 - 0.5) / 0.5).to(
                self.device
            )

            with torch.inference_mode():
                logits = self.model(x)
                probs = torch.softmax(
                    logits, dim=1
                )[0].cpu().numpy()

            prediction = int(probs.argmax())
            truth = (
                None if self.custom_image is not None
                else int(self.labels[index])
            )
            confidence = float(probs[prediction]) * 100

            self.pred_var.set(
                f"Predicted: {prediction} - "
                f"{CLASS_NAMES[prediction]}"
            )
            self.conf_var.set(
                f"Confidence: {confidence:.2f}%"
            )

            self.result_var.set(
                "EXTERNAL IMAGE - NO GROUND TRUTH"
                if truth is None else
                ("CORRECT PREDICTION" if prediction == truth
                 else "INCORRECT PREDICTION")
            )

            for i, (bar, label) in enumerate(self.bars):
                percentage = float(probs[i]) * 100
                bar["value"] = percentage
                label.config(
                    text=f"{percentage:.1f}%"
                )

            self.status_var.set(
                f"Completed | {self.model_var.get()} | "
                f"Image {index} | {self.device}"
            )

        except Exception as error:
            messagebox.showerror(
                "AI Prediction Error", str(error)
            )
            self.status_var.set("Inference failed")

        finally:
            self.run_button.config(state="normal")

if __name__ == "__main__":
    print("CAFA-TTA | Launching interactive desktop AI...")
    app_window = tk.Tk()
    app = CAFAApp(app_window)
    app_window.mainloop()
