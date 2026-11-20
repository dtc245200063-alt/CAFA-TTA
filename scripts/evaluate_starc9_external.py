
import sys
import ast
import csv
import json
import hashlib
import gc
import time
import platform

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

BASE_SCRIPT = (
    ROOT / "scripts/evaluate_cafa_final_test.py"
)

OUTPUT_DIR = (
    ROOT / "experiments/starc9_external"
)

PROTOCOL_PATH = (
    OUTPUT_DIR / "external_protocol.json"
)

PARTIAL_PATH = (
    OUTPUT_DIR / "partial_results.json"
)

METRICS_PATH = (
    OUTPUT_DIR / "metrics.json"
)

PREDICTIONS_DIR = (
    OUTPUT_DIR / "predictions"
)

DATA_DIR = (
    ROOT / "data/processed/starc9"
)

DATASET_PATH = (
    DATA_DIR / "starc9_target_28.npz"
)

INDEX_PATH = (
    DATA_DIR / "sample_index.csv"
)

PREPARATION_PATH = (
    DATA_DIR / "preparation_manifest.json"
)

REPRO_DIR = (
    ROOT / "experiments/reproducibility"
)

EXPECTED_PROTOCOL_HASH = (
    "0880BA90A14046C10E006F99B53CAAB6E"
    "013D46F1C62B56D38CEBD3A992E9295"
)

EXPECTED_DATASET_HASH = (
    "308F5AB7C7FAF73F58FD088896D8FE83F"
    "A7D9CF660942BFA925B01DC49DDFA32"
)

EXPECTED_INDEX_HASH = (
    "77644B633C337C3DAE6DF664061206DD"
    "8C855478ECC1FCFE2993F9A193C3B5C3"
)

EXPECTED_PREPARATION_SCRIPT_HASH = (
    "26BBC75A6800631D796A3E730B887BF4E"
    "B66759FDA60A4E22F13F081F4E254AA"
)

TARGET_IDS = [0, 3, 4, 5, 6, 8]

ALL_IDS = list(range(9))

ABSENT_IDS = [1, 2, 7]

SEEDS = [42, 123, 2026]

METHODS = [
    "source_only",
    "bn_only",
    "tent",
    "cafa_v3",
]

MODEL_NAMES = [
    "enhanced",
    "robust",
    "balanced",
]

EXPECTED_IMAGES = 12000

EXPECTED_RUNS = 30

BATCH_SIZE = 64

LEARNING_RATE = 0.0001

ENTROPY_FLOOR = 0.2

LAMBDA_ALIGN = 0.05

MIN_SAMPLES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

LABEL_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}


# =========================================================
# GENERAL UTILITIES
# =========================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def sha256_file(path):

    require(
        path.is_file(),
        f"Missing file: {path}"
    )

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(
                8 * 1024 * 1024
            ),
            b"",
        ):

            hasher.update(chunk)

    return hasher.hexdigest().upper()


def read_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def save_json(path, data):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def relative_path(path):

    return path.relative_to(
        ROOT
    ).as_posix()


def run_key(row):

    return (
        row["model"],
        row["method"],
        row["seed"],
    )


# =========================================================
# REUSE ORIGINAL PATHMNIST IMPLEMENTATION
#
# Load definitions without executing the old experiment.
# =========================================================

def load_original_pipeline():

    source = BASE_SCRIPT.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    frozen_config = read_json(
        REPRO_DIR / "final_config.json"
    )

    require(
        frozen_config["configuration_frozen"] is True,
        "Original configuration not frozen"
    )

    require(
        frozen_config["algorithm"] == "CAFA-TTA v3",
        "Original algorithm mismatch"
    )

    # Explicitly provide only the globals needed
    # by the original evaluation functions.

    namespace = {
        "__file__": str(BASE_SCRIPT),
        "__name__": "_cafa_original_pipeline",

        "ROOT": ROOT,
        "DEVICE": DEVICE,
        "REPRO_DIR": REPRO_DIR,

        "config": frozen_config,

        "BATCH_SIZE": frozen_config["batch_size"],
        "LR": frozen_config["learning_rate"],
        "SEEDS": frozen_config["seeds"],
        "ENTROPY_FLOOR": frozen_config["entropy_floor"],
        "LAMBDA_ALIGN": frozen_config["lambda_align"],
        "MIN_SAMPLES": frozen_config["min_samples"],
    }

    required_functions = {
        "sha256_file",
        "set_seed",
        "make_loader",
        "create_backbone",
        "load_prototypes",
        "configure_target_bn",
        "evaluate",
    }

    definitions = []

    loaded_functions = set()

    models_count = 0

    for node in tree.body:

        if isinstance(
            node,
            (ast.Import, ast.ImportFrom)
        ):

            definitions.append(node)

        elif isinstance(node, ast.FunctionDef):

            if node.name in required_functions:

                definitions.append(node)

                loaded_functions.add(node.name)

        elif isinstance(node, ast.Assign):

            # MODELS is the only original
            # assignment that must be retained.

            if any(
                isinstance(target, ast.Name)
                and target.id == "MODELS"
                for target in node.targets
            ):

                models_count += 1

                # Ensure the model configuration
                # contains only literal data.

                ast.literal_eval(node.value)

                definitions.append(node)

    require(
        loaded_functions == required_functions,
        "Missing original evaluator functions: "
        + str(required_functions - loaded_functions)
    )

    require(
        models_count == 1,
        "Original MODELS assignment missing or duplicated"
    )

    module = ast.Module(
        body=definitions,
        type_ignores=[],
    )

    exec(
        compile(
            ast.fix_missing_locations(module),
            str(BASE_SCRIPT),
            "exec",
        ),
        namespace,
    )

    for name in required_functions:

        require(
            callable(namespace.get(name)),
            f"Original function unavailable: {name}"
        )

    print(
        "Original PathMNIST pipeline: LOADED"
    )

    print(
        "Original functions loaded:",
        len(loaded_functions)
    )

    print(
        "Frozen configuration injection: PASS"
    )

    print(
        "Original MODELS configuration: LOADED"
    )

    print(
        "PathMNIST dataset initialization: SKIPPED"
    )

    print(
        "PathMNIST synthetic shift initialization: SKIPPED"
    )

    print(
        "Original experiment execution: SKIPPED"
    )

    return namespace


# =========================================================
# VERIFY FROZEN ARTIFACTS
# =========================================================

def verify_csv_manifest(path):

    verified = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    for row in rows:

        require(
            row["Algorithm"].upper() == "SHA256",
            "Unexpected hash algorithm"
        )

        original = Path(
            row["Path"]
        )

        if original.is_absolute():

            try:

                relative = original.relative_to(
                    ROOT
                )

                current = ROOT / relative

            except ValueError:

                parent = original.parent.name

                current = (
                    ROOT / parent / original.name
                )

        else:

            current = ROOT / original

        actual = sha256_file(
            current
        )

        require(
            actual == row["Hash"].upper(),
            f"Frozen artifact hash mismatch: {current}"
        )

        verified[
            relative_path(current)
        ] = actual

        print(
            f"PASS | {relative_path(current)}"
        )

    return verified


def verify_all_inputs():

    print("\nFROZEN INPUT VERIFICATION")
    print("-" * 75)

    protocol_hash = sha256_file(
        PROTOCOL_PATH
    )

    require(
        protocol_hash
        == EXPECTED_PROTOCOL_HASH,
        "External protocol hash mismatch"
    )

    protocol = read_json(
        PROTOCOL_PATH
    )

    preparation = read_json(
        PREPARATION_PATH
    )

    require(
        protocol["configuration_frozen"] is True,
        "External protocol not frozen"
    )

    require(
        protocol["expected_runs"] == EXPECTED_RUNS,
        "Run count mismatch"
    )

    require(
        protocol["target_images"] == EXPECTED_IMAGES,
        "Target image count mismatch"
    )

    require(
        protocol["target_class_ids"] == TARGET_IDS,
        "Target class mismatch"
    )

    require(
        protocol["source_models"] == MODEL_NAMES,
        "Source model configuration mismatch"
    )

    require(
        protocol["methods"] == METHODS,
        "Method configuration mismatch"
    )

    expected_parameters = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "entropy_floor": ENTROPY_FLOOR,
        "lambda_align": LAMBDA_ALIGN,
        "min_samples": MIN_SAMPLES,
        "seeds": SEEDS,
    }

    for name, expected in (
        expected_parameters.items()
    ):

        require(
            protocol[name] == expected,
            f"Protocol parameter mismatch: {name}"
        )

    require(
        protocol["target_labels_used_for_adaptation"]
        is False,
        "Target-label adaptation must be disabled"
    )

    require(
        protocol["restrict_output_logits"]
        is False,
        "Output restriction must be disabled"
    )

    require(
        protocol["restrict_source_prototypes"]
        is False,
        "Prototype restriction must be disabled"
    )

    require(
        protocol["predict_before_parameter_update"]
        is True,
        "Unexpected prequential protocol"
    )

    require(
        protocol["hyperparameter_tuning_on_target"]
        is False,
        "Target tuning must be disabled"
    )

    require(
        preparation["status"]
        == "dataset_preparation_completed",
        "Dataset preparation incomplete"
    )

    require(
        preparation["protocol_sha256"]
        == EXPECTED_PROTOCOL_HASH,
        "Preparation protocol hash mismatch"
    )

    require(
        preparation["prepared_images"]
        == EXPECTED_IMAGES,
        "Prepared image count mismatch"
    )

    require(
        preparation["prepared_array_integrity_verified"]
        is True,
        "Prepared array integrity not verified"
    )

    require(
        preparation["sample_index_integrity_verified"]
        is True,
        "Sample index integrity not verified"
    )

    preprocessing = preparation[
        "preprocessing"
    ]

    require(
        preprocessing["normalization_applied"]
        is False,
        "Dataset unexpectedly normalized"
    )

    preparation_script = (
        ROOT / "scripts/prepare_starc9_dataset.py"
    )

    require(
        sha256_file(preparation_script)
        == EXPECTED_PREPARATION_SCRIPT_HASH,
        "Preparation script hash mismatch"
    )

    require(
        sha256_file(DATASET_PATH)
        == EXPECTED_DATASET_HASH,
        "Prepared dataset hash mismatch"
    )

    require(
        sha256_file(INDEX_PATH)
        == EXPECTED_INDEX_HASH,
        "Sample index hash mismatch"
    )

    print("Protocol SHA256: PASS")
    print("Prepared dataset SHA256: PASS")
    print("Sample index SHA256: PASS")
    print("Preparation script SHA256: PASS")

    for relative, expected in (
        protocol["input_hashes"].items()
    ):

        require(
            sha256_file(ROOT / relative)
            == expected.upper(),
            f"Frozen input mismatch: {relative}"
        )

    print("Frozen protocol input hashes: PASS")

    print("\nCHECKPOINT AND PROTOTYPE HASHES")

    checkpoint_hashes = verify_csv_manifest(
        REPRO_DIR / "checkpoint_hashes.csv"
    )

    print("\nALGORITHM HASHES")

    algorithm_hashes = verify_csv_manifest(
        REPRO_DIR / "algorithm_hashes.csv"
    )

    require(
        len(checkpoint_hashes) == 8,
        "Unexpected checkpoint manifest count"
    )

    require(
        len(algorithm_hashes) == 7,
        "Unexpected algorithm manifest count"
    )

    print("\nAll frozen artifact hashes: PASS")

    return (
        protocol,
        preparation,
        checkpoint_hashes,
        algorithm_hashes,
    )


# =========================================================
# EXTERNAL DATASET
# =========================================================

class IndexedSTARC9(Dataset):

    def __init__(self, images, labels):

        # Fixed source normalization:
        # uint8 -> float32 -> [-1, 1]
        self.images = (
            torch.from_numpy(images.copy())
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div(255.0)
            .sub(0.5)
            .div(0.5)
        )

        self.labels = torch.from_numpy(
            labels.copy()
        ).long()

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.images[index],
            self.labels[index],
            index,
        )


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_dataset():

    print("\nLOADING STARC-9")
    print("-" * 75)

    with np.load(
        DATASET_PATH,
        allow_pickle=False,
    ) as data:

        images = data["images"]

        labels = data["labels"]

        indices = data["indices"]

    require(
        images.shape
        == (EXPECTED_IMAGES, 28, 28, 3),
        "Invalid image shape"
    )

    require(
        images.dtype == np.uint8,
        "Invalid image dtype"
    )

    require(
        labels.shape == (EXPECTED_IMAGES,),
        "Invalid labels shape"
    )

    require(
        labels.dtype == np.int64,
        "Invalid labels dtype"
    )

    require(
        np.array_equal(
            indices,
            np.arange(EXPECTED_IMAGES),
        ),
        "Invalid sample IDs"
    )

    class_ids, counts = np.unique(
        labels,
        return_counts=True,
    )

    require(
        class_ids.tolist() == TARGET_IDS,
        "Unexpected target class IDs"
    )

    require(
        np.all(counts == 2000),
        "Target distribution mismatch"
    )

    dataset = IndexedSTARC9(
        images,
        labels,
    )

    require(
        torch.isfinite(
            dataset.images
        ).all().item(),
        "Non-finite input pixels"
    )

    require(
        dataset.images.min().item() >= -1.0,
        "Invalid normalized minimum"
    )

    require(
        dataset.images.max().item() <= 1.0,
        "Invalid normalized maximum"
    )

    print("Images:", len(dataset))
    print("Input tensor:", tuple(dataset.images.shape))
    print("Normalization: source mean/std 0.5")
    print("Target classes:", class_ids.tolist())
    print("Class counts:", counts.tolist())

    print("Dataset loading: PASS")

    return dataset, labels


# =========================================================
# FAIR TARGET STREAM
# =========================================================

def make_external_loader(dataset, seed):

    generator = torch.Generator()

    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


# =========================================================
# CHECK MODEL CONFIGURATION
# =========================================================

def get_model_config(namespace):

    configuration = namespace.get("MODELS")

    require(
        isinstance(configuration, dict),
        "Original MODELS configuration missing"
    )

    require(
        set(configuration) == set(MODEL_NAMES),
        "Original model names mismatch"
    )

    for model in MODEL_NAMES:

        item = configuration[model]

        require(
            item["checkpoint"]
            == f"resnet18_pathmnist_{model}_best.pt",
            f"Checkpoint configuration mismatch: {model}"
        )

        require(
            item["prototypes"]
            == f"source_prototypes_{model}.pt",
            f"Prototype configuration mismatch: {model}"
        )

    print("Original model configuration: MODELS")
    print("Model configuration verification: PASS")

    return configuration


# =========================================================
# READ METRICS FROM PREDICTIONS
# =========================================================

def calculate_metrics(labels, predictions):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    require(
        labels.shape
        == predictions.shape
        == (EXPECTED_IMAGES,),
        "Prediction array shape mismatch"
    )

    require(
        np.isin(
            predictions,
            ALL_IDS,
        ).all(),
        "Invalid predicted class IDs"
    )

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    macro_f1_target = float(
        f1_score(
            labels,
            predictions,
            labels=TARGET_IDS,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1_all = float(
        f1_score(
            labels,
            predictions,
            labels=ALL_IDS,
            average="macro",
            zero_division=0,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=ALL_IDS,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=ALL_IDS,
    )

    require(
        int(matrix.sum()) == EXPECTED_IMAGES,
        "Confusion matrix count mismatch"
    )

    require(
        np.isclose(
            np.trace(matrix) / matrix.sum(),
            accuracy,
        ),
        "Accuracy reconstruction mismatch"
    )

    absent_count = int(
        np.isin(
            predictions,
            ABSENT_IDS,
        ).sum()
    )

    per_class = []

    for i in ALL_IDS:

        per_class.append({
            "class_id": i,
            "class_name": LABEL_NAMES[i],
            "class_present_in_target": i in TARGET_IDS,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_count": int(
                matrix[:, i].sum()
            ),
        })

    return {
        "samples": EXPECTED_IMAGES,
        "correct_predictions": int(
            (labels == predictions).sum()
        ),
        "accuracy": accuracy,
        "macro_f1": macro_f1_target,
        "macro_f1_target_6_classes": macro_f1_target,
        "macro_f1_all_9_classes": macro_f1_all,
        "absent_class_prediction_count": (
            absent_count
        ),
        "absent_class_prediction_rate": (
            absent_count / EXPECTED_IMAGES
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


# =========================================================
# PREDICTION FILE VALIDATION
# =========================================================

def validate_prediction_file(
    path,
    canonical_labels,
    expected_order=None,
):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required_keys = {
            "indices",
            "labels",
            "predictions",
            "batch_size",
        }

        require(
            required_keys.issubset(
                set(data.files)
            ),
            f"Missing prediction keys: {path}"
        )

        indices = np.asarray(
            data["indices"]
        ).reshape(-1)

        labels = np.asarray(
            data["labels"]
        ).reshape(-1)

        predictions = np.asarray(
            data["predictions"]
        ).reshape(-1)

        batch_size = int(
            np.asarray(
                data["batch_size"]
            ).reshape(-1)[0]
        )

    require(
        batch_size == BATCH_SIZE,
        "Prediction batch size mismatch"
    )

    require(
        indices.shape
        == labels.shape
        == predictions.shape
        == (EXPECTED_IMAGES,),
        "Invalid prediction array sizes"
    )

    require(
        np.issubdtype(
            indices.dtype,
            np.integer,
        ),
        "Non-integer sample IDs"
    )

    require(
        np.issubdtype(
            predictions.dtype,
            np.integer,
        ),
        "Non-integer predictions"
    )

    require(
        np.array_equal(
            np.sort(indices),
            np.arange(EXPECTED_IMAGES),
        ),
        "Prediction samples missing or repeated"
    )

    require(
        np.array_equal(
            labels,
            canonical_labels[indices],
        ),
        "Prediction ground-truth mismatch"
    )

    if expected_order is not None:

        require(
            np.array_equal(
                indices,
                expected_order,
            ),
            "Seeded stream order mismatch"
        )

    metrics = calculate_metrics(
        labels,
        predictions,
    )

    return metrics, indices


# =========================================================
# ORIGINAL EVALUATOR INTEGRATION
# =========================================================

def configure_original_evaluator(
    namespace,
    dataset,
    work_dir,
):

    def external_loader(seed):

        return make_external_loader(
            dataset,
            seed,
        )

    namespace["make_loader"] = (
        external_loader
    )

    namespace["DEVICE"] = DEVICE

    namespace["BATCH_SIZE"] = BATCH_SIZE

    namespace["dataset"] = dataset

    namespace["prediction_dir"] = work_dir

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    require(
        len(namespace["dataset"]) == EXPECTED_IMAGES,
        "Original evaluator dataset size mismatch"
    )

    require(
        namespace["LR"] == LEARNING_RATE,
        "Original evaluator learning rate mismatch"
    )

    require(
        namespace["ENTROPY_FLOOR"] == ENTROPY_FLOOR,
        "Original evaluator entropy floor mismatch"
    )

    require(
        namespace["LAMBDA_ALIGN"] == LAMBDA_ALIGN,
        "Original evaluator alignment weight mismatch"
    )

    require(
        namespace["MIN_SAMPLES"] == MIN_SAMPLES,
        "Original evaluator min_samples mismatch"
    )

    print(
        "External dataset injection: PASS"
    )

    print(
        "External prediction directory:",
        namespace["prediction_dir"]
    )

    print(
        "Original evaluator configuration: PASS"
    )

    # Redirect known experiment-output paths
    # to a separate external working directory.
    for name, value in list(
        namespace.items()
    ):

        if not isinstance(value, Path):

            continue

        try:

            relative = value.resolve().relative_to(
                ROOT / "experiments"
            )

        except ValueError:

            continue

        if not relative.parts:

            continue

        if relative.parts[0] == (
            "cafa_final_test_strong"
        ):

            new_path = work_dir.joinpath(
                *relative.parts[1:]
            )

            namespace[name] = new_path

            print(
                f"Redirect {name}: {new_path}"
            )

    # Inspect the evaluator before execution.
    # Refuse unsafe hardcoded references to the
    # original PathMNIST output directory.
    tree = ast.parse(
        BASE_SCRIPT.read_text(
            encoding="utf-8-sig"
        )
    )

    evaluation_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "evaluate"
    )

    for node in ast.walk(
        evaluation_node
    ):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, str):

                require(
                    "cafa_final_test_strong"
                    not in node.value,
                    "Evaluator contains a hardcoded "
                    "PathMNIST output path"
                )

                require(
                    node.value
                    not in (
                        "PathMNIST",
                        "pathmnist",
                    ),
                    "Evaluator contains an "
                    "unexpected dataset dependency"
                )

    return namespace["evaluate"]


# =========================================================
# EXECUTE ONE ORIGINAL EVALUATION RUN
# =========================================================

def evaluate_one_run(
    namespace,
    evaluator,
    model_config,
    canonical_labels,
    model_name,
    method,
    seed,
    work_dir,
    reference_orders,
):

    print("\n" + "=" * 75)

    print(
        f"MODEL={model_name.upper()} | "
        f"METHOD={method.upper()} | "
        f"SEED={seed}"
    )

    print("=" * 75)

    gc.collect()

    torch.cuda.empty_cache()

    before_files = set(
        work_dir.rglob("*.npz")
    )

    start = time.perf_counter()

    row = evaluator(
        model_name,
        model_config,
        method,
        seed,
    )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    require(
        isinstance(row, dict),
        "Original evaluator must return a dictionary"
    )

    after_files = set(
        work_dir.rglob("*.npz")
    )

    generated_files = sorted(
        after_files - before_files
    )

    require(
        len(generated_files) == 1,
        "Expected exactly one original prediction "
        "file in external working directory; found "
        + str(generated_files)
    )

    generated_path = generated_files[0]

    destination = (
        PREDICTIONS_DIR
        / f"{model_name}_{method}_seed{seed}.npz"
    )

    require(
        not destination.exists(),
        f"Prediction already exists: {destination}"
    )

    metrics, indices = validate_prediction_file(
        generated_path,
        canonical_labels,
        reference_orders.get(seed),
    )

    if seed not in reference_orders:

        reference_orders[seed] = indices.copy()

    if "accuracy" in row:

        require(
            abs(
                float(row["accuracy"])
                - metrics["accuracy"]
            ) < 1e-10,
            "Original evaluator accuracy mismatch"
        )

    require(
        row["macro_f1"]
        == metrics["macro_f1_all_9_classes"],
        "Original evaluator Macro-F1 mismatch"
    )

    require(
        row["confusion_matrix"]
        == metrics["confusion_matrix"],
        "Original evaluator confusion matrix mismatch"
    )

    require(
        row["samples"] == EXPECTED_IMAGES,
        "Original evaluator sample count mismatch"
    )

    require(
        row["predictions_sha256"].upper()
        == sha256_file(generated_path),
        "Original evaluator prediction hash mismatch"
    )

    generated_path.replace(
        destination
    )

    row["predictions_file"] = relative_path(
        destination
    )

    row["predictions_sha256"] = sha256_file(
        destination
    )

    output = {
        "model": model_name,
        "method": method,
        "seed": seed,

        **metrics,

        "batch_size": BATCH_SIZE,
        "batches": 188,

        "elapsed_seconds": elapsed,

        "predictions_file": relative_path(
            destination
        ),

        "predictions_sha256": sha256_file(
            destination
        ),

        "original_evaluator_metrics": row,

        "dataset": "STARC-9",

        "protocol": "online_prequential",

        "target_labels_used_for_adaptation": False,
    }

    # Any filesystem Path returned by the
    # original evaluator is converted to text.
    output = json.loads(
        json.dumps(
            output,
            default=str,
            allow_nan=False,
        )
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        "Macro-F1 (target 6): "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        "Macro-F1 (all 9): "
        f"{metrics['macro_f1_all_9_classes']:.6f}"
    )

    print(
        "Absent-class prediction rate: "
        f"{metrics['absent_class_prediction_rate'] * 100:.4f}%"
    )

    print(
        f"Elapsed: {elapsed:.2f} seconds"
    )

    print("Prediction integrity: PASS")

    gc.collect()

    torch.cuda.empty_cache()

    return output


# =========================================================
# RESUME VALIDATED PARTIAL RESULTS
# =========================================================

def load_partial(
    context,
    canonical_labels,
    expected_keys,
):

    rows = []

    reference_orders = {}

    if not PARTIAL_PATH.exists():

        return rows, reference_orders

    partial = read_json(
        PARTIAL_PATH
    )

    require(
        partial["context"] == context,
        "Partial-result context mismatch"
    )

    seen = set()

    for row in partial["results"]:

        key = run_key(row)

        require(
            key in expected_keys,
            f"Unexpected partial run: {key}"
        )

        require(
            key not in seen,
            f"Duplicate partial run: {key}"
        )

        seen.add(key)

        path = ROOT / row[
            "predictions_file"
        ]

        require(
            sha256_file(path)
            == row["predictions_sha256"].upper(),
            "Partial prediction hash mismatch"
        )

        reconstructed, indices = (
            validate_prediction_file(
                path,
                canonical_labels,
                reference_orders.get(
                    row["seed"]
                ),
            )
        )

        for field in [
            "accuracy",
            "macro_f1",
            "macro_f1_all_9_classes",
            "absent_class_prediction_rate",
        ]:

            require(
                abs(
                    row[field]
                    - reconstructed[field]
                ) < 1e-10,
                f"Partial metric mismatch: {field}"
            )

        require(
            row["confusion_matrix"]
            == reconstructed["confusion_matrix"],
            "Partial confusion matrix mismatch"
        )

        reference_orders.setdefault(
            row["seed"],
            indices.copy(),
        )

        rows.append(row)

    print(
        "Validated partial runs:",
        len(rows)
    )

    return rows, reference_orders


# =========================================================
# SUMMARY UTILITIES
# =========================================================

def statistics(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(values.mean()),
        "std": (
            float(
                values.std(ddof=1)
            )
            if len(values) > 1
            else None
        ),
        "n": len(values),
    }


def summarize(rows):

    summary = {}

    paired = []

    for model in MODEL_NAMES:

        summary[model] = {}

        for method in METHODS:

            group = [
                row
                for row in rows
                if row["model"] == model
                and row["method"] == method
            ]

            expected = (
                1
                if method == "source_only"
                else 3
            )

            require(
                len(group) == expected,
                "Incomplete method group"
            )

            summary[model][method] = {
                field: statistics([
                    row[field]
                    for row in group
                ])
                for field in [
                    "accuracy",
                    "macro_f1",
                    "macro_f1_all_9_classes",
                    "absent_class_prediction_rate",
                ]
            }

        for seed in SEEDS:

            tent = next(
                row
                for row in rows
                if run_key(row)
                == (model, "tent", seed)
            )

            cafa = next(
                row
                for row in rows
                if run_key(row)
                == (model, "cafa_v3", seed)
            )

            accuracy_delta = (
                cafa["accuracy"]
                - tent["accuracy"]
            ) * 100

            f1_delta = (
                cafa["macro_f1"]
                - tent["macro_f1"]
            ) * 100

            paired.append({
                "model": model,
                "seed": seed,
                "accuracy_delta_pp": (
                    accuracy_delta
                ),
                "macro_f1_delta_pp": (
                    f1_delta
                ),
                "absent_prediction_delta_pp": (
                    (
                        cafa[
                            "absent_class_prediction_rate"
                        ]
                        - tent[
                            "absent_class_prediction_rate"
                        ]
                    ) * 100
                ),
            })

    return summary, paired


# =========================================================
# FINAL REPORT
# =========================================================

def print_summary(
    summary,
    pairs,
):

    print("\n" + "=" * 100)

    print(
        "STARC-9 EXTERNAL EVALUATION SUMMARY"
    )

    print("=" * 100)

    print(
        f"{'Model':12s} "
        f"{'Method':14s} "
        f"{'Accuracy':>12s} "
        f"{'F1-6':>12s} "
        f"{'F1-9':>12s} "
        f"{'Absent %':>12s}"
    )

    print("-" * 100)

    for model in MODEL_NAMES:

        for method in METHODS:

            item = summary[model][method]

            print(
                f"{model:12s} "
                f"{method:14s} "
                f"{item['accuracy']['mean']:.6f} "
                f"{item['macro_f1']['mean']:.6f} "
                f"{item['macro_f1_all_9_classes']['mean']:.6f} "
                f"{item['absent_class_prediction_rate']['mean'] * 100:12.4f}"
            )

    print("\nCAFA V3 VS TENT — PAIRED COMPARISONS")
    print("-" * 75)

    print(
        f"{'Model':12s} "
        f"{'Seed':>6s} "
        f"{'Acc Delta pp':>15s} "
        f"{'F1 Delta pp':>15s}"
    )

    for row in pairs:

        print(
            f"{row['model']:12s} "
            f"{row['seed']:6d} "
            f"{row['accuracy_delta_pp']:+15.6f} "
            f"{row['macro_f1_delta_pp']:+15.6f}"
        )

    for model in MODEL_NAMES:

        group = [
            row
            for row in pairs
            if row["model"] == model
        ]

        acc = np.mean([
            row["accuracy_delta_pp"]
            for row in group
        ])

        f1 = np.mean([
            row["macro_f1_delta_pp"]
            for row in group
        ])

        print(
            f"\n{model.upper()} MEAN:"
        )

        print(
            f"  Accuracy delta: {acc:+.6f} pp"
        )

        print(
            f"  Macro-F1 delta: {f1:+.6f} pp"
        )

    print("\nOVERALL CAFA V3 VS TENT")
    print("-" * 75)

    acc_deltas = np.asarray([
        row["accuracy_delta_pp"]
        for row in pairs
    ])

    f1_deltas = np.asarray([
        row["macro_f1_delta_pp"]
        for row in pairs
    ])

    wins = int(
        (acc_deltas > 1e-10).sum()
    )

    losses = int(
        (acc_deltas < -1e-10).sum()
    )

    ties = (
        len(pairs) - wins - losses
    )

    print(
        f"Accuracy Delta: "
        f"{acc_deltas.mean():+.6f} pp"
    )

    print(
        f"Macro-F1 Delta: "
        f"{f1_deltas.mean():+.6f} pp"
    )

    print(
        f"Wins/Ties/Losses: "
        f"{wins}/{ties}/{losses}"
    )

    return {
        "paired_comparisons": len(pairs),
        "mean_accuracy_delta_pp": float(
            acc_deltas.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            f1_deltas.mean()
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    require(
        DEVICE.type == "cuda",
        "CUDA is required for this evaluation"
    )

    print("=" * 75)

    print(
        "CAFA-TTA | STARC-9 EXTERNAL EVALUATION"
    )

    print("=" * 75)

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Device:", DEVICE)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    require(
        not METRICS_PATH.exists(),
        "Final metrics already exist. "
        "Refusing to overwrite completed evaluation."
    )

    (
        protocol,
        preparation,
        checkpoint_hashes,
        algorithm_hashes,
    ) = verify_all_inputs()

    dataset, canonical_labels = (
        load_dataset()
    )

    namespace = load_original_pipeline()

    model_configs = get_model_config(
        namespace
    )

    work_dir = (
        OUTPUT_DIR / "_evaluation_work"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluator = configure_original_evaluator(
        namespace,
        dataset,
        work_dir,
    )

    context = {
        "dataset": "STARC-9",
        "protocol_sha256": (
            EXPECTED_PROTOCOL_HASH
        ),
        "dataset_sha256": (
            EXPECTED_DATASET_HASH
        ),
        "sample_index_sha256": (
            EXPECTED_INDEX_HASH
        ),
        "original_evaluator_sha256": (
            sha256_file(BASE_SCRIPT)
        ),
        "external_evaluator_sha256": (
            sha256_file(
                Path(__file__).resolve()
            )
        ),
        "checkpoint_hashes": checkpoint_hashes,
        "algorithm_hashes": algorithm_hashes,
        "torch_version": str(torch.__version__),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0),
    }

    schedule = []

    for model in MODEL_NAMES:

        for method in METHODS:

            seeds = (
                [42]
                if method == "source_only"
                else SEEDS
            )

            for seed in seeds:

                schedule.append(
                    (model, method, seed)
                )

    require(
        len(schedule) == EXPECTED_RUNS,
        "Unexpected experiment schedule"
    )

    rows, reference_orders = load_partial(
        context,
        canonical_labels,
        set(schedule),
    )

    completed_keys = {
        run_key(row)
        for row in rows
    }

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 75)

    print("Target images:", len(dataset))
    print("Target classes:", TARGET_IDS)
    print("Output classes: 9")
    print("Batch size:", BATCH_SIZE)
    print("Seeds:", SEEDS)
    print("Expected runs:", EXPECTED_RUNS)
    print("Completed runs:", len(rows))

    print(
        "Target-label adaptation: DISABLED"
    )

    print(
        "Hyperparameter tuning: DISABLED"
    )

    for model, method, seed in schedule:

        key = (
            model,
            method,
            seed,
        )

        if key in completed_keys:

            print(
                "SKIP VERIFIED:",
                key
            )

            continue

        print(
            f"\nEXTERNAL RUN "
            f"{len(rows) + 1}/{EXPECTED_RUNS}",
            flush=True,
        )

        row = evaluate_one_run(
            namespace,
            evaluator,
            model_configs[model],
            canonical_labels,
            model,
            method,
            seed,
            work_dir,
            reference_orders,
        )

        rows.append(row)

        completed_keys.add(key)

        save_json(
            PARTIAL_PATH,
            {
                "context": context,
                "completed_runs": len(rows),
                "expected_runs": EXPECTED_RUNS,
                "results": rows,
            },
        )

        print(
            f"Saved partial results: "
            f"{len(rows)}/{EXPECTED_RUNS}",
            flush=True,
        )

    require(
        len(rows) == EXPECTED_RUNS,
        "Incomplete external evaluation"
    )

    require(
        completed_keys == set(schedule),
        "Final schedule mismatch"
    )

    for row in rows:

        path = ROOT / row[
            "predictions_file"
        ]

        require(
            sha256_file(path)
            == row["predictions_sha256"],
            "Final prediction hash mismatch"
        )

        validate_prediction_file(
            path,
            canonical_labels,
            reference_orders[row["seed"]],
        )

    summary, paired = summarize(
        rows
    )

    overall = print_summary(
        summary,
        paired,
    )

    final_report = {
        "project": "CAFA-TTA",
        "dataset": "STARC-9",
        "subset": protocol["target_subset"],

        "dataset_revision": (
            protocol["dataset_revision"]
        ),

        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "protocol": "online_prequential",
        "configuration_frozen": True,

        "partial_label_space": True,

        "target_images": EXPECTED_IMAGES,
        "target_class_ids": TARGET_IDS,
        "source_num_classes": 9,

        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
        "learning_rate": LEARNING_RATE,
        "entropy_floor": ENTROPY_FLOOR,
        "lambda_align": LAMBDA_ALIGN,
        "min_samples": MIN_SAMPLES,

        "models": MODEL_NAMES,
        "methods": METHODS,

        "expected_runs": EXPECTED_RUNS,
        "completed_runs": len(rows),

        "target_labels_used_for_adaptation": False,

        "hyperparameter_tuning_on_target": False,

        "source_only_repeated_across_seeds": False,

        "primary_metric": "accuracy",

        "macro_f1_primary_class_ids": TARGET_IDS,

        "context": context,

        "data_quality": {
            "original_duplicates": preparation[
                "original_byte_duplicate_analysis"
            ],
            "resized_duplicates": preparation[
                "resized_pixel_duplicate_analysis"
            ],
            "samples_removed": 0,
        },

        "results": rows,
        "summary": summary,
        "paired_comparisons": paired,
        "overall": overall,

        "prediction_integrity_verified": True,

        "limitations": protocol["limitations"],

        "status": "external_evaluation_completed",
    }

    save_json(
        METRICS_PATH,
        final_report,
    )

    print("\n" + "=" * 75)

    print("FINAL EXTERNAL EVALUATION STATUS")

    print("=" * 75)

    print(
        "Completed runs:",
        len(rows),
        "/",
        EXPECTED_RUNS,
    )

    print(
        "Prediction files:",
        len(list(
            PREDICTIONS_DIR.glob("*.npz")
        )),
    )

    print("Checkpoint integrity: PASS")
    print("Algorithm integrity: PASS")
    print("Dataset integrity: PASS")
    print("Protocol integrity: PASS")
    print("Prediction integrity: PASS")

    print(
        "Metrics SHA256:",
        sha256_file(METRICS_PATH),
    )

    print(
        "Saved metrics:",
        METRICS_PATH,
    )

    print(
        "Final status: "
        "EXTERNAL EVALUATION COMPLETED"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()
