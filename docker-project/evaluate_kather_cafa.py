import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/project")
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor
from kernel.tent import TentAdapter
from core.cafa_soft_adapter import CAFASoftAdapter


KATHER = Path("/kather")
DATA_ROOT = (
    KATHER / "extracted"
    / "Kather_texture_2016_image_tiles_5000"
)

RUN_DIR = Path("/outputs/kather2016_cpu_01")
CHECKPOINT = RUN_DIR / "best.pt"
PROTOTYPE_FILE = RUN_DIR / "source_prototypes.pt"
OUTPUT_DIR = RUN_DIR / "cafa_comparison_shifted_seed42"

SHIFT = {
    "brightness": 0.75,
    "contrast": 1.35,
    "saturation": 0.70,
}

SEED = 42
BATCH_SIZE = 16


class KatherTestDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]

        with Image.open(
            DATA_ROOT / item["path"]
        ) as img:
            image = img.convert("RGB")
            image = image.resize((96, 96))

            image = TF.adjust_brightness(
                image, SHIFT["brightness"]
            )
            image = TF.adjust_contrast(
                image, SHIFT["contrast"]
            )
            image = TF.adjust_saturation(
                image, SHIFT["saturation"]
            )

            image = TF.to_tensor(image)
            image = TF.normalize(
                image,
                [0.5] * 3,
                [0.5] * 3,
            )

        return image, int(item["label"]), index


def create_backbone():
    model = models.resnet18(
        weights=None,
        num_classes=8,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def make_loader(dataset):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def evaluate_method(method, dataset, prototypes):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    backbone = create_backbone()
    adapter = None

    if method == "source_only":
        backbone.eval()

    elif method == "tent":
        adapter = TentAdapter(
            backbone,
            learning_rate=1e-4,
        )

    elif method == "cafa_v3":
        feature_model = ResNetFeatureExtractor(
            backbone
        )

        adapter = CAFASoftAdapter(
            model=feature_model,
            prototypes=prototypes,
            lr=1e-4,
            lambda_align=0.05,
            entropy_floor=0.2,
            min_samples=2,
        )

    else:
        raise ValueError(method)

    predictions = []
    labels_all = []
    indices_all = []

    selected_total = 0
    updated_batches = 0

    loader = make_loader(dataset)

    for batch_id, (images, labels, indices) in enumerate(
        loader, start=1
    ):
        # Predict before adapting.
        with torch.no_grad():
            if adapter is None:
                logits = backbone(images)
            else:
                logits = adapter.predict(images)

            predicted = logits.argmax(1)

        predictions.extend(predicted.tolist())
        labels_all.extend(labels.tolist())
        indices_all.extend(indices.tolist())

        # Adapt without using test labels.
        if adapter is not None:
            result = adapter.adapt(images)

            if method == "cafa_v3":
                selected_total += int(
                    result["selected"]
                )

                if result["updated"]:
                    updated_batches += 1

        if batch_id % 10 == 0:
            print(
                f"{method} | "
                f"Batch {batch_id}/{len(loader)}",
                flush=True,
            )

    accuracy = accuracy_score(
        labels_all,
        predictions,
    )

    macro_f1 = f1_score(
        labels_all,
        predictions,
        labels=list(range(8)),
        average="macro",
        zero_division=0,
    )

    prediction_path = (
        OUTPUT_DIR / f"{method}_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        indices=np.array(indices_all),
        labels=np.array(labels_all),
        predictions=np.array(predictions),
    )

    metrics = {
        "method": method,
        "samples": len(dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "prediction_file": prediction_path.name,
    }

    if method == "cafa_v3":
        metrics["agreement_rate"] = (
            selected_total / len(dataset)
        )
        metrics["updated_batches"] = updated_batches
        metrics["total_batches"] = len(loader)

    return metrics


def main():
    torch.set_num_threads(1)

    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)

    split = json.loads(
        (KATHER / "split_seed42.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(split["classes"]) == 8
    assert len(split["splits"]["test"]) == 752

    prototype_data = torch.load(
        PROTOTYPE_FILE,
        map_location="cpu",
        weights_only=True,
    )

    assert prototype_data["num_classes"] == 8
    assert prototype_data["feature_dim"] == 512
    assert prototype_data["classes"] == split["classes"]

    prototypes = prototype_data["prototypes"]

    assert prototypes.shape == (8, 512)
    assert torch.isfinite(prototypes).all()

    dataset = KatherTestDataset(
        split["splits"]["test"]
    )

    OUTPUT_DIR.mkdir(parents=True)

    print("=" * 60, flush=True)
    print("KATHER 2016 | CAFA-TTA COMPARISON", flush=True)
    print("Device: CPU", flush=True)
    print("Test samples:", len(dataset), flush=True)
    print("Domain shift:", SHIFT, flush=True)
    print("Seed:", SEED, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)
    print("=" * 60, flush=True)

    results = []

    for method in [
        "source_only",
        "tent",
        "cafa_v3",
    ]:
        print("\nRunning:", method, flush=True)

        metrics = evaluate_method(
            method,
            dataset,
            prototypes,
        )

        results.append(metrics)

        (OUTPUT_DIR / "partial_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        print(
            f"{method} | "
            f"Accuracy={metrics['accuracy'] * 100:.2f}% | "
            f"Macro-F1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    (OUTPUT_DIR / "comparison_metrics.json").write_text(
        json.dumps(
            {
                "dataset": "Kather 2016",
                "shift": SHIFT,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nCOMPARISON SUMMARY", flush=True)

    for row in results:
        print(
            f"{row['method']:12s} | "
            f"Acc={row['accuracy'] * 100:.2f}% | "
            f"F1={row['macro_f1']:.4f}",
            flush=True,
        )

    print("FINAL STATUS: PASS", flush=True)


if __name__ == "__main__":
    main()
