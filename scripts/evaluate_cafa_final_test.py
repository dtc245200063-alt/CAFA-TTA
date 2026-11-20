
import sys
import csv
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor
from kernel.tent import TentAdapter
from core.cafa_adapter import CAFAAdapter
from core.cafa_agreement_adapter import CAFAAgreementAdapter
from core.cafa_soft_adapter import CAFASoftAdapter

# ==========================================
# CAFA-TTA | FINAL TEST EVALUATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

REPRO_DIR = (
    ROOT / "experiments/reproducibility"
)

with open(
    REPRO_DIR / "final_config.json",
    encoding="utf-8-sig",
) as f:
    config = json.load(f)

assert config["configuration_frozen"] is True
assert config["algorithm"] == "CAFA-TTA v3"

BATCH_SIZE = config["batch_size"]
LR = config["learning_rate"]
SEEDS = config["seeds"]
ENTROPY_FLOOR = config["entropy_floor"]
LAMBDA_ALIGN = config["lambda_align"]
MIN_SAMPLES = config["min_samples"]

assert BATCH_SIZE == 64
assert LR == 1e-4
assert SEEDS == [42, 123, 2026]
assert ENTROPY_FLOOR == 0.2
assert LAMBDA_ALIGN == 0.05
assert MIN_SAMPLES == 2

MODELS = {
    "enhanced": {
        "checkpoint": "resnet18_pathmnist_enhanced_best.pt",
        "prototypes": "source_prototypes_enhanced.pt",
    },
    "robust": {
        "checkpoint": "resnet18_pathmnist_robust_best.pt",
        "prototypes": "source_prototypes_robust.pt",
    },
    "balanced": {
        "checkpoint": "resnet18_pathmnist_balanced_best.pt",
        "prototypes": "source_prototypes_balanced.pt",
    },
}

METHODS = [
    "source_only",
    "bn_only",
    "tent",
    "cafa_v1",
    "alignment_only",
    "cafa_v2",
    "cafa_v3",
]

METHOD_CONFIGS = {
    "source_only": {},
    "bn_only": {},
    "tent": {
        "lr": LR,
    },
    "cafa_v1": {
        "threshold": 0.7,
        "lambda_align": 0.05,
    },
    "alignment_only": {
        "threshold": 0.0,
        "lambda_align": 0.10,
    },
    "cafa_v2": {
        "threshold": 0.7,
        "lambda_align": 0.05,
    },
    "cafa_v3": {
        "entropy_floor": ENTROPY_FLOOR,
        "lambda_align": LAMBDA_ALIGN,
    },
}

# ==========================================
# Verify frozen files
# ==========================================

def sha256_file(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest().upper()


def verify_hash_manifest(filename):

    verified = {}

    with open(
        REPRO_DIR / filename,
        encoding="utf-8-sig",
        newline="",
    ) as f:

        for row in csv.DictReader(f):

            original_path = Path(row["Path"])

            path = original_path

            if not path.exists():

                parts = original_path.parts

                for folder in [
                    "checkpoints",
                    "core",
                    "kernel",
                    "models",
                ]:

                    if folder in parts:

                        idx = parts.index(folder)

                        path = ROOT.joinpath(
                            *parts[idx:]
                        )

                        break

            if not path.exists():

                raise FileNotFoundError(path)

            actual_hash = sha256_file(path)

            expected_hash = (
                row["Hash"].strip().upper()
            )

            if actual_hash != expected_hash:

                raise RuntimeError(
                    f"Frozen file changed: {path}"
                )

            relative_path = (
                path.resolve()
                .relative_to(ROOT.resolve())
                .as_posix()
            )

            verified[relative_path] = actual_hash

    assert verified, (
        f"Empty hash manifest: {filename}"
    )

    return verified


checkpoint_hashes = verify_hash_manifest(
    "checkpoint_hashes.csv"
)

algorithm_hashes = verify_hash_manifest(
    "algorithm_hashes.csv"
)

for model_config in MODELS.values():

    for key in ["checkpoint", "prototypes"]:

        relative_path = (
            "checkpoints/" + model_config[key]
        )

        assert relative_path in checkpoint_hashes

assert (
    "core/cafa_soft_adapter.py"
    in algorithm_hashes
)

print("Checkpoint hashes: VERIFIED")
print("Algorithm hashes: VERIFIED")
print("Frozen configuration: VERIFIED")

# ==========================================
# Strong domain shift
# ==========================================

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8",
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"

shift = shift_config["shift"]

assert shift == {
    "brightness": 0.75,
    "contrast": 1.35,
    "saturation": 0.70,
}

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3,
)


def target_transform(image):

    image = TF.adjust_brightness(
        image, shift["brightness"]
    )

    image = TF.adjust_contrast(
        image, shift["contrast"]
    )

    image = TF.adjust_saturation(
        image, shift["saturation"]
    )

    return normalize(
        TF.to_tensor(image)
    )


# ==========================================
# Indexed test dataset
# ==========================================

class IndexedPathMNIST(Dataset):

    def __init__(self):

        self.dataset = PathMNIST(
            split="test",
            root=str(
                ROOT / "data/raw/medmnist"
            ),
            transform=target_transform,
            download=False,
            size=28,
        )

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        return image, label, index


dataset = IndexedPathMNIST()

assert len(dataset) == 7180

# ==========================================
# Reproducibility
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def make_loader(seed):

    generator = torch.Generator()

    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False,
    )


# ==========================================
# Model and prototype loading
# ==========================================

def create_backbone(checkpoint_name):

    backbone = models.resnet18(
        weights=None
    )

    backbone.fc = nn.Linear(
        backbone.fc.in_features,
        9,
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return backbone.to(DEVICE)


def load_prototypes(model_config):

    data = torch.load(
        ROOT
        / "checkpoints"
        / model_config["prototypes"],
        map_location="cpu",
        weights_only=True,
    )

    assert (
        data["source_checkpoint"]
        == model_config["checkpoint"]
    )

    prototypes = data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return prototypes


def configure_target_bn(model):

    model.eval()

    for param in model.parameters():

        param.requires_grad_(False)

    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):

            module.train()

            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


# ==========================================
# Output directories
# ==========================================

output_dir = (
    ROOT
    / "experiments"
    / "cafa_final_test_strong"
)

prediction_dir = (
    output_dir / "predictions"
)

prediction_dir.mkdir(
    parents=True,
    exist_ok=True,
)

partial_path = (
    output_dir / "partial_results.json"
)


def save_json(path, data):

    temp_path = path.with_suffix(".tmp")

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    temp_path.replace(path)


# ==========================================
# Evaluate one independent run
# ==========================================

def evaluate(
    model_name,
    model_config,
    method,
    seed,
):

    set_seed(seed)

    loader = make_loader(seed)

    backbone = create_backbone(
        model_config["checkpoint"]
    )

    adapter = None
    feature_model = None
    prototypes = None

    if method == "source_only":

        backbone.eval()

        for param in backbone.parameters():

            param.requires_grad_(False)

    elif method == "bn_only":

        backbone = configure_target_bn(
            backbone
        )

    elif method == "tent":

        adapter = TentAdapter(
            backbone,
            learning_rate=LR,
        )

    elif method in [
        "cafa_v1",
        "alignment_only",
        "cafa_v2",
        "cafa_v3",
    ]:

        feature_model = ResNetFeatureExtractor(
            backbone
        )

        prototypes = load_prototypes(
            model_config
        )

        if method == "cafa_v1":

            adapter = CAFAAdapter(
                model=feature_model,
                prototypes=prototypes,
                lr=LR,
                threshold=0.7,
                lambda_align=0.05,
                min_samples=MIN_SAMPLES,
            )

        elif method == "alignment_only":

            adapter = CAFAAdapter(
                model=feature_model,
                prototypes=prototypes,
                lr=LR,
                threshold=0.0,
                lambda_align=0.10,
                min_samples=MIN_SAMPLES,
            )

        elif method == "cafa_v2":

            adapter = CAFAAgreementAdapter(
                model=feature_model,
                prototypes=prototypes,
                lr=LR,
                threshold=0.7,
                lambda_align=0.05,
                min_samples=MIN_SAMPLES,
            )

        elif method == "cafa_v3":

            adapter = CAFASoftAdapter(
                model=feature_model,
                prototypes=prototypes,
                lr=LR,
                lambda_align=LAMBDA_ALIGN,
                entropy_floor=ENTROPY_FLOOR,
                min_samples=MIN_SAMPLES,
            )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    predictions = []
    labels_all = []
    indices_all = []

    selected_total = 0
    updated_batches = 0
    total_seen = 0

    for images, labels, indices in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Predict before adaptation.
        with torch.no_grad():

            if adapter is None:

                logits = backbone(images)

            else:

                logits = adapter.predict(images)

            pred = logits.argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )

        labels_all.extend(
            labels.view(-1).tolist()
        )

        indices_all.extend(
            indices.tolist()
        )

        total_seen += images.size(0)

        # Adapt without target labels.
        if adapter is not None:

            result = adapter.adapt(images)

            if method.startswith("cafa") or (
                method == "alignment_only"
            ):

                selected_total += int(
                    result["selected"]
                )

                if result["updated"]:

                    updated_batches += 1

    # Verify complete test coverage.

    indices_all = np.array(
        indices_all,
        dtype=np.int64,
    )

    labels_all = np.array(
        labels_all,
        dtype=np.int64,
    )

    predictions = np.array(
        predictions,
        dtype=np.int64,
    )

    assert total_seen == len(dataset)

    assert np.array_equal(
        np.sort(indices_all),
        np.arange(len(dataset)),
    )

    # Preserve stream order and original indices.
    # Required for paired statistical analysis.

    filename = (
        f"{model_name}_{method}_"
        f"seed{seed}.npz"
    )

    prediction_path = (
        prediction_dir / filename
    )

    np.savez_compressed(
        prediction_path,
        indices=indices_all,
        labels=labels_all,
        predictions=predictions,
        batch_size=BATCH_SIZE,
    )

    metrics = {
        "model": model_name,
        "checkpoint": (
            model_config["checkpoint"]
        ),
        "method": method,
        "seed": seed,
        "samples": total_seen,
        "accuracy": float(
            accuracy_score(
                labels_all,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions,
                labels=list(range(9)),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                labels_all,
                predictions,
                labels=list(range(9)),
            ).tolist()
        ),
        "predictions_file": (
            prediction_path
            .relative_to(ROOT)
            .as_posix()
        ),
        "predictions_sha256": (
            sha256_file(prediction_path)
        ),
    }

    if method.startswith("cafa") or (
        method == "alignment_only"
    ):

        rate_name = (
            "agreement_rate"
            if method == "cafa_v3"
            else "selection_rate"
        )

        metrics[rate_name] = float(
            selected_total / total_seen
        )

        metrics["updated_batches"] = (
            updated_batches
        )

        metrics["total_batches"] = len(loader)

    del adapter
    del feature_model
    del prototypes
    del backbone
    del loader

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    return metrics


# ==========================================
# Experiment metadata
# ==========================================

TOTAL_RUNS = (
    len(MODELS)
    * len(METHODS)
    * len(SEEDS)
)

metadata = {
    "protocol": "shuffled_online_prequential",
    "split": "test",
    "test_previously_accessed": True,
    "configuration_frozen": True,
    "shift_name": "strong",
    "shift": shift,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "entropy_floor": ENTROPY_FLOOR,
    "lambda_align": LAMBDA_ALIGN,
    "min_samples": MIN_SAMPLES,
    "seeds": SEEDS,
    "models": MODELS,
    "methods": METHODS,
    "method_configs": METHOD_CONFIGS,
    "checkpoint_hashes": checkpoint_hashes,
    "algorithm_hashes": algorithm_hashes,
    "evaluation_script_hash": sha256_file(
        Path(__file__)
    ),
    "torch_version": torch.__version__,
    "device": str(DEVICE),
}


def run_key(row):

    return (
        row["model"],
        row["method"],
        row["seed"],
    )


# ==========================================
# Resume completed experiments
# ==========================================

results = []
completed = set()

if partial_path.exists():

    with open(
        partial_path,
        encoding="utf-8",
    ) as f:

        previous = json.load(f)

    if previous["metadata"] != metadata:

        raise RuntimeError(
            "Saved metadata differs. "
            "Do not mix test configurations."
        )

    results = previous["results"]

    for row in results:

        prediction_path = (
            ROOT / row["predictions_file"]
        )

        assert prediction_path.exists()

        assert (
            sha256_file(prediction_path)
            == row["predictions_sha256"]
        )

    completed = {
        run_key(row)
        for row in results
    }

    assert len(completed) == len(results)

# ==========================================
# Execute final test
# ==========================================

print("=" * 75)
print("CAFA-TTA | FINAL TEST EVALUATION")
print("Device:", DEVICE)
print("Dataset: PathMNIST Test")
print("Samples:", len(dataset))
print("Shift:", shift)
print("Seeds:", SEEDS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Entropy floor:", ENTROPY_FLOOR)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("Total runs:", TOTAL_RUNS)
print("Already completed:", len(completed))
print("=" * 75)

for model_name, model_config in MODELS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("=" * 75)

    for seed in SEEDS:

        for method in METHODS:

            key = (
                model_name,
                method,
                seed,
            )

            if key in completed:

                print(
                    f"SKIP {method} "
                    f"Seed={seed}: completed",
                    flush=True,
                )

                continue

            metrics = evaluate(
                model_name,
                model_config,
                method,
                seed,
            )

            results.append(metrics)
            completed.add(key)

            line = (
                f"[{len(results):2d}/{TOTAL_RUNS}] "
                f"Seed={seed:4d} | "
                f"{method:15s} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['macro_f1']:.4f}"
            )

            if "agreement_rate" in metrics:

                line += (
                    f" | Agreement="
                    f"{metrics['agreement_rate']:.2%}"
                )

            if "selection_rate" in metrics:

                line += (
                    f" | Selection="
                    f"{metrics['selection_rate']:.2%}"
                )

            print(line, flush=True)

            save_json(
                partial_path,
                {
                    "metadata": metadata,
                    "completed_runs": len(results),
                    "total_runs": TOTAL_RUNS,
                    "results": results,
                },
            )


# ==========================================
# Final summary
# ==========================================

assert len(results) == TOTAL_RUNS

summary = {}

all_acc_deltas = []
all_f1_deltas = []

print("\n" + "=" * 75)
print("FINAL TEST SUMMARY")
print("=" * 75)

for model_name in MODELS:

    print("\nMODEL:", model_name.upper())

    summary[model_name] = {}

    model_rows = [
        row for row in results
        if row["model"] == model_name
    ]

    tent_by_seed = {
        row["seed"]: row
        for row in model_rows
        if row["method"] == "tent"
    }

    for method in METHODS:

        rows = [
            row for row in model_rows
            if row["method"] == method
        ]

        assert len(rows) == len(SEEDS)

        accuracies = np.array([
            row["accuracy"]
            for row in rows
        ])

        f1_values = np.array([
            row["macro_f1"]
            for row in rows
        ])

        acc_deltas = np.array([
            row["accuracy"]
            - tent_by_seed[row["seed"]]["accuracy"]
            for row in rows
        ]) * 100

        f1_deltas = np.array([
            row["macro_f1"]
            - tent_by_seed[row["seed"]]["macro_f1"]
            for row in rows
        ]) * 100

        wins = int(
            (acc_deltas > 0).sum()
        )

        ties = int(
            (acc_deltas == 0).sum()
        )

        losses = int(
            (acc_deltas < 0).sum()
        )

        summary[model_name][method] = {
            "accuracy_mean": float(
                accuracies.mean()
            ),
            "accuracy_std": float(
                accuracies.std(ddof=1)
            ),
            "macro_f1_mean": float(
                f1_values.mean()
            ),
            "macro_f1_std": float(
                f1_values.std(ddof=1)
            ),
            "accuracy_delta_vs_tent_pp": float(
                acc_deltas.mean()
            ),
            "macro_f1_delta_vs_tent_pp": float(
                f1_deltas.mean()
            ),
            "accuracy_per_seed_pp": (
                acc_deltas.tolist()
            ),
            "macro_f1_per_seed_pp": (
                f1_deltas.tolist()
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

        print(
            f"{method:15s} | "
            f"Acc={accuracies.mean():.4f} "
            f"+/- {accuracies.std(ddof=1):.4f} | "
            f"F1={f1_values.mean():.4f} "
            f"+/- {f1_values.std(ddof=1):.4f} | "
            f"Delta={acc_deltas.mean():+.3f} pp"
        )

        if method == "cafa_v3":

            all_acc_deltas.extend(
                acc_deltas.tolist()
            )

            all_f1_deltas.extend(
                f1_deltas.tolist()
            )

            print(
                "  CAFA v3 Wins/Ties/Losses:",
                f"{wins}/{ties}/{losses}"
            )


# ==========================================
# Overall CAFA v3 comparison
# ==========================================

all_acc_deltas = np.array(
    all_acc_deltas
)

all_f1_deltas = np.array(
    all_f1_deltas
)

overall = {
    "mean_accuracy_delta_pp": float(
        all_acc_deltas.mean()
    ),
    "mean_macro_f1_delta_pp": float(
        all_f1_deltas.mean()
    ),
    "wins": int(
        (all_acc_deltas > 0).sum()
    ),
    "ties": int(
        (all_acc_deltas == 0).sum()
    ),
    "losses": int(
        (all_acc_deltas < 0).sum()
    ),
}

print("\n" + "=" * 75)
print("FINAL CAFA v3 VS TENT COMPARISON")
print("=" * 75)

print(
    "Mean Accuracy Delta:",
    f"{overall['mean_accuracy_delta_pp']:+.3f} pp"
)

print(
    "Mean Macro-F1 Delta:",
    f"{overall['mean_macro_f1_delta_pp']:+.3f} pp"
)

print(
    "Wins/Ties/Losses:",
    f"{overall['wins']}/"
    f"{overall['ties']}/"
    f"{overall['losses']}"
)

# ==========================================
# Save final test results
# ==========================================

output_path = output_dir / "metrics.json"

save_json(
    output_path,
    {
        **metadata,
        "completed_runs": len(results),
        "total_runs": TOTAL_RUNS,
        "results": results,
        "summary": summary,
        "overall": overall,
    },
)

print("\n" + "=" * 75)
print("CAFA-TTA FINAL TEST COMPLETED")
print("Completed runs:", len(results), "/", TOTAL_RUNS)
print("Saved metrics:", output_path)
print("Saved predictions:", prediction_dir)
print("=" * 75)
