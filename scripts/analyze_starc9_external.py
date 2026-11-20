
import csv
import json
import hashlib
import math
import platform

from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = ROOT / "experiments/starc9_external"

OUTPUT_DIR = EXPERIMENT_DIR / "statistical_analysis"

FIGURE_DIR = OUTPUT_DIR / "figures"

METRICS_PATH = EXPERIMENT_DIR / "metrics.json"

PROTOCOL_PATH = EXPERIMENT_DIR / "external_protocol.json"

DATA_DIR = ROOT / "data/processed/starc9"

DATASET_PATH = DATA_DIR / "starc9_target_28.npz"

INDEX_PATH = DATA_DIR / "sample_index.csv"

PREPARATION_PATH = DATA_DIR / "preparation_manifest.json"

EXPECTED_METRICS_HASH = (
    "44F3EC85515FD74197510E57CA2ED817B"
    "22BF1FF56373767AD4BF44B0872A364"
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

MODELS = [
    "enhanced",
    "robust",
    "balanced",
]

METHODS = [
    "source_only",
    "bn_only",
    "tent",
    "cafa_v3",
]

SEEDS = [42, 123, 2026]

TARGET_IDS = [0, 3, 4, 5, 6, 8]

ABSENT_IDS = [1, 2, 7]

ALL_IDS = list(range(9))

CLASS_NAMES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
]

NUM_IMAGES = 12000

EXPECTED_RUNS = 30

EXPECTED_PAIRS = 9

BATCH_SIZE = 64

BOOTSTRAP_REPEATS = 10000

BOOTSTRAP_SEED = 2026

BLOCK_LENGTHS = [1, 4, 8, 16]

ALPHA = 0.05

COLORS = {
    "source_only": "#64748B",
    "bn_only": "#2563EB",
    "tent": "#D97706",
    "cafa_v3": "#169B62",
}

DISPLAY = {
    "source_only": "Source-only",
    "bn_only": "BN-only",
    "tent": "TENT",
    "cafa_v3": "CAFA v3",
}

OUTPUT_FILES = []

INPUT_HASHES = {}


# =========================================================
# UTILITIES
# =========================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest().upper()


def relative(path):

    return path.relative_to(ROOT).as_posix()


def verify_input(path, expected=None):

    require(
        path.is_file(),
        f"Missing input: {path}"
    )

    actual = sha256_file(path)

    if expected is not None:

        require(
            actual == expected.upper(),
            f"Input hash mismatch: {path}"
        )

    INPUT_HASHES[relative(path)] = actual

    return actual


def read_json(path):

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )



def numpy_json_default(value):

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Unsupported JSON type: {type(value).__name__}"
    )


def save_json(name, data):

    path = OUTPUT_DIR / name

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=numpy_json_default,
        ),
        encoding="utf-8",
    )

    temp.replace(path)

    OUTPUT_FILES.append(path)

    print("JSON:", name)


def save_csv(name, rows):

    require(
        len(rows) > 0,
        f"No rows for {name}"
    )

    path = OUTPUT_DIR / name

    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)

    temp.replace(path)

    OUTPUT_FILES.append(path)

    print(
        f"CSV: {name} | rows={len(rows)}"
    )


def mean(values):

    return float(
        np.mean(values)
    )


def sample_std(values):

    if len(values) < 2:
        return None

    return float(
        np.std(values, ddof=1)
    )


def interval(values):

    lower, upper = np.quantile(
        values,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def key(row):

    return (
        row["model"],
        row["method"],
        row["seed"],
    )


def win_status(delta):

    if delta > 1e-10:
        return "win"

    if delta < -1e-10:
        return "loss"

    return "tie"


# =========================================================
# INPUT VALIDATION
# =========================================================

def load_inputs():

    print("\nINPUT VALIDATION")
    print("-" * 75)

    verify_input(
        METRICS_PATH,
        EXPECTED_METRICS_HASH,
    )

    verify_input(
        PROTOCOL_PATH,
        EXPECTED_PROTOCOL_HASH,
    )

    verify_input(
        DATASET_PATH,
        EXPECTED_DATASET_HASH,
    )

    verify_input(
        INDEX_PATH,
        EXPECTED_INDEX_HASH,
    )

    verify_input(
        PREPARATION_PATH
    )

    metrics = read_json(
        METRICS_PATH
    )

    protocol = read_json(
        PROTOCOL_PATH
    )

    preparation = read_json(
        PREPARATION_PATH
    )

    require(
        metrics["status"]
        == "external_evaluation_completed",
        "External evaluation incomplete"
    )

    require(
        metrics["completed_runs"]
        == EXPECTED_RUNS,
        "Unexpected run count"
    )

    require(
        len(metrics["results"])
        == EXPECTED_RUNS,
        "Missing evaluation results"
    )

    require(
        protocol["configuration_frozen"] is True,
        "Protocol not frozen"
    )

    require(
        protocol["target_class_ids"]
        == TARGET_IDS,
        "Target classes mismatch"
    )

    require(
        metrics["target_labels_used_for_adaptation"]
        is False,
        "Target labels were used for adaptation"
    )

    require(
        metrics["context"]["protocol_sha256"]
        == EXPECTED_PROTOCOL_HASH,
        "Metrics protocol mismatch"
    )

    require(
        metrics["context"]["dataset_sha256"]
        == EXPECTED_DATASET_HASH,
        "Metrics dataset mismatch"
    )

    with np.load(
        DATASET_PATH,
        allow_pickle=False,
    ) as dataset:

        canonical_labels = dataset[
            "labels"
        ].copy()

        sample_ids = dataset[
            "indices"
        ].copy()

    require(
        canonical_labels.shape
        == (NUM_IMAGES,),
        "Invalid canonical label shape"
    )

    require(
        np.array_equal(
            sample_ids,
            np.arange(NUM_IMAGES),
        ),
        "Invalid canonical sample IDs"
    )

    require(
        np.unique(
            canonical_labels
        ).tolist() == TARGET_IDS,
        "Invalid canonical labels"
    )

    require(
        all(
            value == 2000
            for value in Counter(
                canonical_labels.tolist()
            ).values()
        ),
        "Target distribution mismatch"
    )

    with INDEX_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        sample_rows = list(
            csv.DictReader(f)
        )

    require(
        len(sample_rows) == NUM_IMAGES,
        "Sample index size mismatch"
    )

    for i, row in enumerate(
        sample_rows
    ):

        require(
            int(row["sample_id"]) == i,
            "Sample index order mismatch"
        )

        require(
            int(row["mapped_label"])
            == canonical_labels[i],
            "Sample index label mismatch"
        )

    print("Input hashes: PASS")
    print("Canonical labels: PASS")
    print("Sample index: PASS")

    return (
        metrics,
        protocol,
        preparation,
        canonical_labels,
        sample_rows,
    )


# =========================================================
# METRIC RECONSTRUCTION
# =========================================================

def calculate_metrics(y, pred):

    return {
        "accuracy": float(
            accuracy_score(y, pred)
        ),

        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=TARGET_IDS,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_f1_all_9_classes": float(
            f1_score(
                y,
                pred,
                labels=ALL_IDS,
                average="macro",
                zero_division=0,
            )
        ),

        "absent_class_prediction_rate": float(
            np.isin(
                pred,
                ABSENT_IDS,
            ).mean()
        ),

        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=ALL_IDS,
        ).tolist(),
    }


# =========================================================
# VERIFY ALL 30 PREDICTION FILES
# =========================================================

def verify_predictions(
    metrics,
    canonical_labels,
):

    print("\nPREDICTION INTEGRITY AUDIT")
    print("-" * 75)

    runs = {}

    audit_rows = []

    reference_orders = {}

    expected_keys = {
        (model, method, seed)
        for model in MODELS
        for method in METHODS
        for seed in (
            [42]
            if method == "source_only"
            else SEEDS
        )
    }

    used_paths = set()

    for row in metrics["results"]:

        identifier = key(row)

        require(
            identifier in expected_keys,
            f"Unexpected run: {identifier}"
        )

        require(
            identifier not in runs,
            f"Duplicate run: {identifier}"
        )

        path = ROOT / row[
            "predictions_file"
        ]

        path = path.resolve()

        require(
            path.is_relative_to(
                (
                    EXPERIMENT_DIR
                    / "predictions"
                ).resolve()
            ),
            "Unexpected prediction location"
        )

        require(
            path not in used_paths,
            "Prediction file reused incorrectly"
        )

        used_paths.add(path)

        verify_input(
            path,
            row["predictions_sha256"],
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            indices = data[
                "indices"
            ].reshape(-1)

            labels = data[
                "labels"
            ].reshape(-1)

            predictions = data[
                "predictions"
            ].reshape(-1)

            batch_size = int(
                data["batch_size"].reshape(-1)[0]
            )

        require(
            indices.shape
            == labels.shape
            == predictions.shape
            == (NUM_IMAGES,),
            f"Invalid prediction arrays: {identifier}"
        )

        require(
            np.issubdtype(
                indices.dtype,
                np.integer,
            ),
            "Non-integer sample indices"
        )

        require(
            np.issubdtype(
                predictions.dtype,
                np.integer,
            ),
            "Non-integer predictions"
        )

        require(
            batch_size == BATCH_SIZE,
            "Batch size mismatch"
        )

        require(
            np.array_equal(
                np.sort(indices),
                np.arange(NUM_IMAGES),
            ),
            "Missing or repeated sample index"
        )

        require(
            np.array_equal(
                labels,
                canonical_labels[indices],
            ),
            "Ground truth mismatch"
        )

        require(
            np.isin(
                predictions,
                ALL_IDS,
            ).all(),
            "Predicted class out of range"
        )

        seed = row["seed"]

        if seed in reference_orders:

            require(
                np.array_equal(
                    indices,
                    reference_orders[seed],
                ),
                "Same-seed stream mismatch"
            )

        else:

            reference_orders[seed] = (
                indices.copy()
            )

        reconstructed = calculate_metrics(
            labels,
            predictions,
        )

        for field in [
            "accuracy",
            "macro_f1",
            "macro_f1_all_9_classes",
            "absent_class_prediction_rate",
        ]:

            require(
                abs(
                    reconstructed[field]
                    - row[field]
                ) < 1e-10,
                f"Metric mismatch: {identifier}/{field}"
            )

        require(
            reconstructed["confusion_matrix"]
            == row["confusion_matrix"],
            "Confusion matrix mismatch"
        )

        aligned_predictions = np.empty(
            NUM_IMAGES,
            dtype=np.int64,
        )

        aligned_predictions[
            indices
        ] = predictions

        correct = (
            aligned_predictions
            == canonical_labels
        )

        runs[identifier] = {
            "row": row,
            "indices": indices,
            "predictions": aligned_predictions,
            "correct": correct,
        }

        audit_rows.append({
            "model": row["model"],
            "method": row["method"],
            "seed": seed,
            "samples": NUM_IMAGES,
            "sha256_verified": True,
            "label_alignment_verified": True,
            "stream_order_verified": True,
            "metrics_verified": True,
            "accuracy": row["accuracy"],
            "macro_f1_target_6": row["macro_f1"],
            "absent_class_prediction_rate": row[
                "absent_class_prediction_rate"
            ],
        })

        print(
            f"PASS | {identifier}"
        )

    require(
        set(runs) == expected_keys,
        "Final run inventory mismatch"
    )

    actual_paths = {
        p.resolve()
        for p in (
            EXPERIMENT_DIR / "predictions"
        ).glob("*.npz")
    }

    require(
        actual_paths == used_paths,
        "Prediction directory inventory mismatch"
    )

    save_csv(
        "prediction_integrity.csv",
        audit_rows,
    )

    print("Verified predictions: 30/30")

    return runs


# =========================================================
# HOLM MULTIPLE-TEST CORRECTION
# =========================================================

def holm_adjust(p_values):

    p = np.asarray(
        p_values,
        dtype=np.float64,
    )

    n = len(p)

    order = np.argsort(p)

    adjusted = np.empty_like(p)

    maximum = 0.0

    for rank, index in enumerate(
        order
    ):

        value = (
            n - rank
        ) * p[index]

        maximum = max(
            maximum,
            value,
        )

        adjusted[index] = min(
            1.0,
            maximum,
        )

    return adjusted


# =========================================================
# PAIRED PREDICTIONS AND MCNEMAR
# =========================================================

def paired_analysis(
    runs,
    canonical_labels,
):

    rows = []

    vectors = []

    pair_keys = []

    print("\nPAIRED PREDICTION ANALYSIS")
    print("-" * 75)

    for model in MODELS:

        for seed in SEEDS:

            tent = runs[
                (model, "tent", seed)
            ]

            cafa = runs[
                (model, "cafa_v3", seed)
            ]

            require(
                np.array_equal(
                    tent["indices"],
                    cafa["indices"],
                ),
                "Paired stream mismatch"
            )

            t = tent["correct"]

            c = cafa["correct"]

            both_correct = int(
                (t & c).sum()
            )

            both_wrong = int(
                ((~t) & (~c)).sum()
            )

            corrected = int(
                ((~t) & c).sum()
            )

            harmed = int(
                (t & (~c)).sum()
            )

            net = corrected - harmed

            require(
                both_correct
                + both_wrong
                + corrected
                + harmed
                == NUM_IMAGES,
                "Paired contingency count mismatch"
            )

            delta = (
                c.astype(np.int8)
                - t.astype(np.int8)
            )

            acc_delta = (
                100.0 * delta.mean()
            )

            f1_delta = (
                cafa["row"]["macro_f1"]
                - tent["row"]["macro_f1"]
            ) * 100

            absent_delta = (
                cafa["row"][
                    "absent_class_prediction_rate"
                ]
                - tent["row"][
                    "absent_class_prediction_rate"
                ]
            ) * 100

            discordant = (
                corrected + harmed
            )

            p_value = (
                float(
                    binomtest(
                        min(corrected, harmed),
                        discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant > 0
                else 1.0
            )

            rows.append({
                "model": model,
                "seed": seed,
                "samples": NUM_IMAGES,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "corrected": corrected,
                "harmed": harmed,
                "net_correct": net,
                "accuracy_delta_pp": acc_delta,
                "macro_f1_delta_pp": f1_delta,
                "absent_delta_pp": absent_delta,
                "accuracy_outcome": win_status(
                    acc_delta
                ),
                "mcnemar_exact_p": p_value,
            })

            vectors.append(delta)

            pair_keys.append(
                (model, seed)
            )

    p_adjusted = holm_adjust([
        row["mcnemar_exact_p"]
        for row in rows
    ])

    for row, adjusted in zip(
        rows,
        p_adjusted,
    ):

        row["holm_adjusted_p"] = float(
            adjusted
        )

        row["raw_significant"] = (
            row["mcnemar_exact_p"]
            < ALPHA
        )

        row["holm_significant"] = (
            adjusted < ALPHA
        )

        print(
            f"{row['model']:10s} "
            f"seed={row['seed']:4d} | "
            f"+{row['corrected']} "
            f"-{row['harmed']} | "
            f"Net={row['net_correct']:+d} | "
            f"p={row['mcnemar_exact_p']:.6f} | "
            f"Holm={adjusted:.6f}"
        )

    save_csv(
        "paired_statistical_tests.csv",
        rows,
    )

    return (
        rows,
        np.stack(vectors),
        pair_keys,
    )


# =========================================================
# CONDITIONAL IMAGE-INDEX BOOTSTRAP
#
# Images are aligned across all runs.
# Repeated model/seed observations are NOT
# treated as independent images.
# =========================================================

def conditional_bootstrap(
    vectors,
    pair_keys,
):

    print("\nCONDITIONAL BOOTSTRAP")
    print("-" * 75)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    groups = {
        "overall": list(
            range(EXPECTED_PAIRS)
        )
    }

    for model in MODELS:

        groups[model] = [
            i
            for i, (name, seed)
            in enumerate(pair_keys)
            if name == model
        ]

    effects = np.stack([
        vectors[indices].mean(axis=0)
        for indices in groups.values()
    ])

    draws = np.empty(
        (
            len(groups),
            BOOTSTRAP_REPEATS,
        )
    )

    chunk = 50

    for start in range(
        0,
        BOOTSTRAP_REPEATS,
        chunk,
    ):

        end = min(
            start + chunk,
            BOOTSTRAP_REPEATS,
        )

        sampled = rng.integers(
            0,
            NUM_IMAGES,
            size=(
                end - start,
                NUM_IMAGES,
            ),
        )

        for i in range(
            len(groups)
        ):

            draws[i, start:end] = (
                effects[i][sampled]
                .mean(axis=1)
                * 100
            )

    output = []

    for i, name in enumerate(
        groups
    ):

        lower, upper = interval(
            draws[i]
        )

        output.append({
            "group": name,
            "pairs": len(groups[name]),
            "accuracy_delta_pp": float(
                effects[i].mean() * 100
            ),
            "ci95_lower_pp": lower,
            "ci95_upper_pp": upper,
            "bootstrap_repeats": (
                BOOTSTRAP_REPEATS
            ),
            "resampling_unit": (
                "aligned_original_image_index"
            ),
            "conditional_on_saved_trajectories": True,
        })

        print(
            f"{name:10s} | "
            f"{output[-1]['accuracy_delta_pp']:+.6f} pp | "
            f"95% CI [{lower:+.6f}, {upper:+.6f}]"
        )

    save_csv(
        "conditional_bootstrap.csv",
        output,
    )

    return output


# =========================================================
# CIRCULAR BLOCK BOOTSTRAP
#
# Block resampling preserves short stretches
# of the observed online batch sequence.
# =========================================================

def circular_block_bootstrap(
    runs,
    vectors,
    pair_keys,
):

    print("\nCIRCULAR BLOCK BOOTSTRAP")
    print("-" * 75)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + 1
    )

    results = []

    for i, (model, seed) in enumerate(
        pair_keys
    ):

        order = runs[
            (model, "tent", seed)
        ]["indices"]

        stream = vectors[i][order]

        sums = np.array([
            stream[start:start + BATCH_SIZE]
            .sum()
            for start in range(
                0,
                NUM_IMAGES,
                BATCH_SIZE,
            )
        ], dtype=np.float64)

        sizes = np.array([
            min(
                BATCH_SIZE,
                NUM_IMAGES - start,
            )
            for start in range(
                0,
                NUM_IMAGES,
                BATCH_SIZE,
            )
        ], dtype=np.float64)

        n_batches = len(sums)

        require(
            n_batches == 188,
            "Unexpected batch count"
        )

        for block_length in BLOCK_LENGTHS:

            n_blocks = math.ceil(
                n_batches / block_length
            )

            starts = rng.integers(
                0,
                n_batches,
                size=(
                    BOOTSTRAP_REPEATS,
                    n_blocks,
                ),
            )

            sampled = (
                starts[:, :, None]
                + np.arange(block_length)
            ) % n_batches

            sampled = sampled.reshape(
                BOOTSTRAP_REPEATS,
                -1,
            )[:, :n_batches]

            totals = sums[
                sampled
            ].sum(axis=1)

            denominators = sizes[
                sampled
            ].sum(axis=1)

            values = (
                totals
                / denominators
                * 100
            )

            lower, upper = interval(
                values
            )

            results.append({
                "model": model,
                "seed": seed,
                "block_length_batches": (
                    block_length
                ),
                "accuracy_delta_pp": float(
                    stream.mean() * 100
                ),
                "ci95_lower_pp": lower,
                "ci95_upper_pp": upper,
                "ci_excludes_zero": bool(
                    lower > 0 or upper < 0
                ),
                "bootstrap_repeats": (
                    BOOTSTRAP_REPEATS
                ),
                "conditional_on_saved_trajectory": True,
            })

    save_csv(
        "block_bootstrap.csv",
        results,
    )

    for length in BLOCK_LENGTHS:

        group = [
            row
            for row in results
            if row["block_length_batches"]
            == length
        ]

        positive = sum(
            row["ci95_lower_pp"] > 0
            for row in group
        )

        negative = sum(
            row["ci95_upper_pp"] < 0
            for row in group
        )

        print(
            f"Block {length:2d} | "
            f"Positive CI={positive} | "
            f"Negative CI={negative} | "
            f"Includes zero="
            f"{len(group)-positive-negative}"
        )

    return results


# =========================================================
# DUPLICATE SENSITIVITY ANALYSIS
# =========================================================

def duplicate_sensitivity(
    runs,
    labels,
    sample_rows,
    preparation,
    vectors,
    pair_keys,
):

    print("\nDUPLICATE SENSITIVITY")
    print("-" * 75)

    groups = defaultdict(list)

    for row in sample_rows:

        groups[
            row["resized_pixel_sha256"]
        ].append(
            int(row["sample_id"])
        )

    duplicate_groups = [
        members
        for members in groups.values()
        if len(members) > 1
    ]

    require(
        len(duplicate_groups)
        == preparation[
            "resized_pixel_duplicate_analysis"
        ]["duplicate_groups"],
        "Duplicate group count mismatch"
    )

    for members in duplicate_groups:

        require(
            len(
                np.unique(labels[members])
            ) == 1,
            "Unexpected cross-label duplicates"
        )

    representatives = np.array(
        sorted(
            min(members)
            for members in groups.values()
        ),
        dtype=np.int64,
    )

    weights = np.ones(
        NUM_IMAGES,
        dtype=np.float64,
    )

    for members in groups.values():

        weights[members] = (
            1.0 / len(members)
        )

    require(
        np.isclose(
            weights.sum(),
            len(groups),
        ),
        "Duplicate weights mismatch"
    )

    rows = []

    for i, (model, seed) in enumerate(
        pair_keys
    ):

        tent = runs[
            (model, "tent", seed)
        ]["predictions"]

        cafa = runs[
            (model, "cafa_v3", seed)
        ]["predictions"]

        scenarios = [
            (
                "all_samples",
                np.arange(NUM_IMAGES),
                None,
            ),
            (
                "one_representative_per_group",
                representatives,
                None,
            ),
            (
                "equal_weight_per_hash_group",
                np.arange(NUM_IMAGES),
                weights,
            ),
        ]

        for name, ids, w in scenarios:

            y = labels[ids]

            t = tent[ids]

            c = cafa[ids]

            t_acc = accuracy_score(
                y,
                t,
                sample_weight=w,
            )

            c_acc = accuracy_score(
                y,
                c,
                sample_weight=w,
            )

            t_f1 = f1_score(
                y,
                t,
                labels=TARGET_IDS,
                average="macro",
                sample_weight=w,
                zero_division=0,
            )

            c_f1 = f1_score(
                y,
                c,
                labels=TARGET_IDS,
                average="macro",
                sample_weight=w,
                zero_division=0,
            )

            rows.append({
                "model": model,
                "seed": seed,
                "scenario": name,
                "samples": len(ids),
                "effective_weight": (
                    float(w.sum())
                    if w is not None
                    else len(ids)
                ),
                "tent_accuracy": float(
                    t_acc
                ),
                "cafa_accuracy": float(
                    c_acc
                ),
                "accuracy_delta_pp": float(
                    (c_acc - t_acc) * 100
                ),
                "macro_f1_delta_pp": float(
                    (c_f1 - t_f1) * 100
                ),
                "adaptation_rerun": False,
            })

    save_csv(
        "duplicate_sensitivity.csv",
        rows,
    )

    duplicate_details = []

    for group_id, members in enumerate(
        duplicate_groups
    ):

        for index in members:

            duplicate_details.append({
                "duplicate_group_id": group_id,
                "sample_id": index,
                "label": int(labels[index]),
                "group_size": len(members),
                "representative": (
                    index == min(members)
                ),
            })

    if duplicate_details:

        save_csv(
            "duplicate_group_members.csv",
            duplicate_details,
        )

    summaries = []

    for name in [
        "all_samples",
        "one_representative_per_group",
        "equal_weight_per_hash_group",
    ]:

        group = [
            row
            for row in rows
            if row["scenario"] == name
        ]

        summaries.append({
            "scenario": name,
            "accuracy_delta_pp": mean([
                row["accuracy_delta_pp"]
                for row in group
            ]),
            "macro_f1_delta_pp": mean([
                row["macro_f1_delta_pp"]
                for row in group
            ]),
            "wins": sum(
                win_status(
                    row["accuracy_delta_pp"]
                ) == "win"
                for row in group
            ),
            "ties": sum(
                win_status(
                    row["accuracy_delta_pp"]
                ) == "tie"
                for row in group
            ),
            "losses": sum(
                win_status(
                    row["accuracy_delta_pp"]
                ) == "loss"
                for row in group
            ),
        })

    save_csv(
        "duplicate_sensitivity_summary.csv",
        summaries,
    )

    print(
        "Duplicate groups:",
        len(duplicate_groups)
    )

    print(
        "Unique resized pixel groups:",
        len(groups)
    )

    print(
        "Extra duplicate images:",
        NUM_IMAGES - len(groups)
    )

    for row in summaries:

        print(
            f"{row['scenario']} | "
            f"Acc={row['accuracy_delta_pp']:+.6f} pp | "
            f"F1={row['macro_f1_delta_pp']:+.6f} pp | "
            f"W/T/L="
            f"{row['wins']}/"
            f"{row['ties']}/"
            f"{row['losses']}"
        )

    return {
        "duplicate_groups": len(
            duplicate_groups
        ),
        "unique_pixel_groups": len(groups),
        "extra_duplicate_images": (
            NUM_IMAGES - len(groups)
        ),
        "sensitivity_results": summaries,
        "adaptation_rerun": False,
    }


# =========================================================
# PER-CLASS ANALYSIS
# =========================================================

def per_class_analysis(
    runs,
    labels,
    pair_keys,
):

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 75)

    per_run = []

    paired = []

    for identifier, run in runs.items():

        model, method, seed = identifier

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                labels,
                run["predictions"],
                labels=ALL_IDS,
                zero_division=0,
            )
        )

        matrix = np.asarray(
            run["row"]["confusion_matrix"]
        )

        for class_id in ALL_IDS:

            per_run.append({
                "model": model,
                "method": method,
                "seed": seed,
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "present_in_target": (
                    class_id in TARGET_IDS
                ),
                "precision": float(
                    precision[class_id]
                ),
                "recall": float(
                    recall[class_id]
                ),
                "f1": float(
                    f1[class_id]
                ),
                "support": int(
                    support[class_id]
                ),
                "predicted_count": int(
                    matrix[:, class_id].sum()
                ),
            })

    lookup = {
        (
            row["model"],
            row["method"],
            row["seed"],
            row["class_id"],
        ): row
        for row in per_run
    }

    for model, seed in pair_keys:

        t = runs[
            (model, "tent", seed)
        ]

        c = runs[
            (model, "cafa_v3", seed)
        ]

        for class_id in TARGET_IDS:

            mask = labels == class_id

            t_correct = t[
                "correct"
            ][mask]

            c_correct = c[
                "correct"
            ][mask]

            corrected = int(
                ((~t_correct) & c_correct).sum()
            )

            harmed = int(
                (t_correct & (~c_correct)).sum()
            )

            t_metrics = lookup[
                (
                    model,
                    "tent",
                    seed,
                    class_id,
                )
            ]

            c_metrics = lookup[
                (
                    model,
                    "cafa_v3",
                    seed,
                    class_id,
                )
            ]

            paired.append({
                "model": model,
                "seed": seed,
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "support": int(mask.sum()),
                "tent_precision": (
                    t_metrics["precision"]
                ),
                "cafa_precision": (
                    c_metrics["precision"]
                ),
                "tent_recall": (
                    t_metrics["recall"]
                ),
                "cafa_recall": (
                    c_metrics["recall"]
                ),
                "tent_f1": t_metrics["f1"],
                "cafa_f1": c_metrics["f1"],
                "precision_delta_pp": (
                    c_metrics["precision"]
                    - t_metrics["precision"]
                ) * 100,
                "recall_delta_pp": (
                    c_metrics["recall"]
                    - t_metrics["recall"]
                ) * 100,
                "f1_delta_pp": (
                    c_metrics["f1"]
                    - t_metrics["f1"]
                ) * 100,
                "corrected": corrected,
                "harmed": harmed,
                "net_correct": (
                    corrected - harmed
                ),
            })

    summaries = []

    for class_id in TARGET_IDS:

        group = [
            row
            for row in paired
            if row["class_id"] == class_id
        ]

        summary = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[
                class_id
            ],
            "support": 2000,
        }

        for field in [
            "tent_precision",
            "cafa_precision",
            "tent_recall",
            "cafa_recall",
            "tent_f1",
            "cafa_f1",
            "precision_delta_pp",
            "recall_delta_pp",
            "f1_delta_pp",
        ]:

            summary[field] = mean([
                row[field]
                for row in group
            ])

        for field in [
            "corrected",
            "harmed",
            "net_correct",
        ]:

            summary[field] = sum(
                row[field]
                for row in group
            )

        summaries.append(summary)

        print(
            f"{summary['class_name']:5s} | "
            f"Recall="
            f"{summary['recall_delta_pp']:+.6f} pp | "
            f"F1="
            f"{summary['f1_delta_pp']:+.6f} pp | "
            f"Net="
            f"{summary['net_correct']:+d}"
        )

    save_csv(
        "per_run_class_metrics.csv",
        per_run,
    )

    save_csv(
        "paired_class_metrics.csv",
        paired,
    )

    save_csv(
        "overall_class_summary.csv",
        summaries,
    )

    return summaries


# =========================================================
# NEGATIVE ADAPTATION
# =========================================================

def negative_adaptation(
    runs,
):

    print("\nNEGATIVE ADAPTATION")
    print("-" * 75)

    rows = []

    summaries = []

    for model in MODELS:

        source = runs[
            (model, "source_only", 42)
        ]["row"]

        for method in [
            "bn_only",
            "tent",
            "cafa_v3",
        ]:

            group = []

            for seed in SEEDS:

                item = runs[
                    (model, method, seed)
                ]["row"]

                accuracy_delta = (
                    item["accuracy"]
                    - source["accuracy"]
                ) * 100

                f1_delta = (
                    item["macro_f1"]
                    - source["macro_f1"]
                ) * 100

                row = {
                    "model": model,
                    "method": method,
                    "seed": seed,
                    "source_accuracy": (
                        source["accuracy"]
                    ),
                    "adapted_accuracy": (
                        item["accuracy"]
                    ),
                    "accuracy_delta_pp": (
                        accuracy_delta
                    ),
                    "macro_f1_delta_pp": (
                        f1_delta
                    ),
                    "negative_accuracy_adaptation": (
                        accuracy_delta < -1e-10
                    ),
                    "negative_f1_adaptation": (
                        f1_delta < -1e-10
                    ),
                }

                group.append(row)

                rows.append(row)

            summaries.append({
                "model": model,
                "method": method,
                "accuracy_delta_pp": mean([
                    row["accuracy_delta_pp"]
                    for row in group
                ]),
                "macro_f1_delta_pp": mean([
                    row["macro_f1_delta_pp"]
                    for row in group
                ]),
                "negative_accuracy_runs": sum(
                    row[
                        "negative_accuracy_adaptation"
                    ]
                    for row in group
                ),
                "negative_f1_runs": sum(
                    row[
                        "negative_f1_adaptation"
                    ]
                    for row in group
                ),
                "runs": len(group),
            })

    save_csv(
        "negative_adaptation.csv",
        rows,
    )

    save_csv(
        "negative_adaptation_summary.csv",
        summaries,
    )

    for method in [
        "bn_only",
        "tent",
        "cafa_v3",
    ]:

        group = [
            row
            for row in rows
            if row["method"] == method
        ]

        negative = sum(
            row[
                "negative_accuracy_adaptation"
            ]
            for row in group
        )

        print(
            f"{method:10s} | "
            f"Negative Accuracy: "
            f"{negative}/{len(group)}"
        )

    return summaries


# =========================================================
# ABSENT CLASS ANALYSIS
# =========================================================

def absent_class_analysis(
    runs,
    labels,
):

    rows = []

    for identifier, run in runs.items():

        model, method, seed = identifier

        pred = run["predictions"]

        for class_id in ABSENT_IDS:

            count = int(
                (pred == class_id).sum()
            )

            rows.append({
                "model": model,
                "method": method,
                "seed": seed,
                "absent_class_id": class_id,
                "absent_class_name": CLASS_NAMES[
                    class_id
                ],
                "predicted_count": count,
                "prediction_rate": (
                    count / NUM_IMAGES
                ),
            })

    save_csv(
        "absent_class_predictions.csv",
        rows,
    )

    breakdown = []

    for model in MODELS:

        for seed in SEEDS:

            t = runs[
                (model, "tent", seed)
            ]["predictions"]

            c = runs[
                (model, "cafa_v3", seed)
            ]["predictions"]

            t_absent = np.isin(
                t,
                ABSENT_IDS,
            )

            c_absent = np.isin(
                c,
                ABSENT_IDS,
            )

            for class_id in TARGET_IDS:

                mask = labels == class_id

                breakdown.append({
                    "model": model,
                    "seed": seed,
                    "true_class_id": class_id,
                    "true_class": CLASS_NAMES[
                        class_id
                    ],
                    "support": int(mask.sum()),
                    "tent_absent_predictions": int(
                        t_absent[mask].sum()
                    ),
                    "cafa_absent_predictions": int(
                        c_absent[mask].sum()
                    ),
                    "absent_to_correct": int(
                        (
                            t_absent[mask]
                            & (c[mask] == labels[mask])
                        ).sum()
                    ),
                    "correct_to_absent": int(
                        (
                            (t[mask] == labels[mask])
                            & c_absent[mask]
                        ).sum()
                    ),
                    "absent_rate_delta_pp": float(
                        (
                            c_absent[mask].mean()
                            - t_absent[mask].mean()
                        ) * 100
                    ),
                })

    save_csv(
        "absent_class_transitions.csv",
        breakdown,
    )


# =========================================================
# CONFUSION MATRICES
# =========================================================

def confusion_analysis(
    runs,
    pair_keys,
):

    t_matrix = np.zeros(
        (9, 9),
        dtype=np.int64,
    )

    c_matrix = np.zeros_like(
        t_matrix
    )

    for model, seed in pair_keys:

        t_matrix += np.asarray(
            runs[
                (model, "tent", seed)
            ]["row"]["confusion_matrix"],
            dtype=np.int64,
        )

        c_matrix += np.asarray(
            runs[
                (model, "cafa_v3", seed)
            ]["row"]["confusion_matrix"],
            dtype=np.int64,
        )

    require(
        t_matrix.sum()
        == NUM_IMAGES * EXPECTED_PAIRS,
        "TENT confusion count mismatch"
    )

    require(
        c_matrix.sum()
        == t_matrix.sum(),
        "CAFA confusion count mismatch"
    )

    difference = c_matrix - t_matrix

    counts = t_matrix.sum(
        axis=1,
        keepdims=True,
    )

    t_normalized = np.divide(
        t_matrix,
        counts,
        out=np.zeros(
            (9, 9),
            dtype=np.float64,
        ),
        where=counts != 0,
    ) * 100

    c_normalized = np.divide(
        c_matrix,
        counts,
        out=np.zeros(
            (9, 9),
            dtype=np.float64,
        ),
        where=counts != 0,
    ) * 100

    difference_pp = (
        c_normalized
        - t_normalized
    )

    rows = []

    for i in ALL_IDS:

        for j in ALL_IDS:

            rows.append({
                "true_id": i,
                "true_class": CLASS_NAMES[i],
                "predicted_id": j,
                "predicted_class": CLASS_NAMES[j],
                "tent_count": int(
                    t_matrix[i, j]
                ),
                "cafa_count": int(
                    c_matrix[i, j]
                ),
                "count_difference": int(
                    difference[i, j]
                ),
                "normalized_difference_pp": float(
                    difference_pp[i, j]
                ),
                "true_class_present": (
                    i in TARGET_IDS
                ),
            })

    save_csv(
        "confusion_changes.csv",
        rows,
    )

    save_json(
        "confusion_matrices.json",
        {
            "class_names": CLASS_NAMES,
            "target_class_ids": TARGET_IDS,
            "pairs": EXPECTED_PAIRS,
            "tent_counts": t_matrix.tolist(),
            "cafa_counts": c_matrix.tolist(),
            "difference_counts": (
                difference.tolist()
            ),
            "tent_normalized_percent": (
                t_normalized.tolist()
            ),
            "cafa_normalized_percent": (
                c_normalized.tolist()
            ),
            "difference_pp": (
                difference_pp.tolist()
            ),
            "absent_true_rows_meaning": (
                "No target samples; normalized values "
                "are placeholders, not measured recall."
            ),
        },
    )

    return (
        t_matrix,
        c_matrix,
        difference,
    )


# =========================================================
# LEAVE-ONE-GROUP-OUT SENSITIVITY
# =========================================================

def group_sensitivity(
    pair_rows,
):

    results = []

    for field, values in [
        ("model", MODELS),
        ("seed", SEEDS),
    ]:

        for value in values:

            remaining = [
                row
                for row in pair_rows
                if row[field] != value
            ]

            results.append({
                "excluded_group_type": field,
                "excluded_group": value,
                "remaining_pairs": len(
                    remaining
                ),
                "accuracy_delta_pp": mean([
                    row["accuracy_delta_pp"]
                    for row in remaining
                ]),
                "macro_f1_delta_pp": mean([
                    row["macro_f1_delta_pp"]
                    for row in remaining
                ]),
            })

    save_csv(
        "leave_one_group_out.csv",
        results,
    )

    return results


# =========================================================
# FIGURE CONFIGURATION
# =========================================================

def configure_figures():

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save_figure(fig, name):

    for extension in [
        "png",
        "pdf",
    ]:

        path = FIGURE_DIR / (
            f"{name}.{extension}"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        OUTPUT_FILES.append(path)

    plt.close(fig)

    print("FIGURE:", name)


# =========================================================
# FIGURE 1: EXTERNAL PERFORMANCE
# =========================================================

def figure_performance(
    runs,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        layout="constrained",
    )

    x = np.arange(len(MODELS))

    width = 0.19

    for ax, field, ylabel in [
        (
            axes[0],
            "accuracy",
            "Accuracy (%)",
        ),
        (
            axes[1],
            "macro_f1",
            "Macro-F1, six classes (%)",
        ),
    ]:

        for j, method in enumerate(
            METHODS
        ):

            values = []

            errors = []

            for model in MODELS:

                group = [
                    run["row"][field] * 100
                    for (m, a, seed), run
                    in runs.items()
                    if m == model
                    and a == method
                ]

                values.append(
                    mean(group)
                )

                errors.append(
                    sample_std(group) or 0.0
                )

            ax.bar(
                x + (j - 1.5) * width,
                values,
                width,
                color=COLORS[method],
                label=DISPLAY[method],
                yerr=(
                    None
                    if method == "source_only"
                    else errors
                ),
                capsize=2,
            )

        ax.set_xticks(
            x,
            [m.title() for m in MODELS],
        )

        ax.set_ylabel(ylabel)

        ax.set_ylim(0, 100)

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        ax.set_axisbelow(True)

    axes[0].legend(
        fontsize=8,
        loc="upper left",
    )

    fig.suptitle(
        "STARC-9 external evaluation"
    )

    save_figure(
        fig,
        "fig01_external_performance",
    )


# =========================================================
# FIGURE 2: PAIRED EFFECTS
# =========================================================

def figure_paired_effects(
    rows,
):

    fig, ax = plt.subplots(
        figsize=(11, 4.8),
        layout="constrained",
    )

    x = np.arange(len(rows))

    width = 0.36

    ax.bar(
        x - width / 2,
        [
            row["accuracy_delta_pp"]
            for row in rows
        ],
        width,
        label="Accuracy",
        color="#2563EB",
    )

    ax.bar(
        x + width / 2,
        [
            row["macro_f1_delta_pp"]
            for row in rows
        ],
        width,
        label="Macro-F1 (6 classes)",
        color="#169B62",
    )

    ax.set_xticks(
        x,
        [
            f"{row['model'].title()}\n"
            f"{row['seed']}"
            for row in rows
        ],
        fontsize=8,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "CAFA v3 - TENT (pp)"
    )

    ax.set_title(
        "Paired external evaluation effects"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    save_figure(
        fig,
        "fig02_paired_effects",
    )


# =========================================================
# FIGURE 3: CONDITIONAL FOREST PLOT
# =========================================================

def figure_forest(
    rows,
):

    fig, ax = plt.subplots(
        figsize=(8, 4.2),
        layout="constrained",
    )

    for i, row in enumerate(
        rows
    ):

        value = row[
            "accuracy_delta_pp"
        ]

        lower = row[
            "ci95_lower_pp"
        ]

        upper = row[
            "ci95_upper_pp"
        ]

        color = (
            "#169B62"
            if row["group"] == "overall"
            else "#2563EB"
        )

        ax.hlines(
            i,
            lower,
            upper,
            color=color,
            linewidth=2,
        )

        ax.scatter(
            value,
            i,
            color=color,
            marker=(
                "D"
                if row["group"] == "overall"
                else "o"
            ),
            s=45,
            zorder=3,
        )

    ax.axvline(
        0,
        linestyle="--",
        color="#64748B",
    )

    ax.set_yticks(
        np.arange(len(rows)),
        [
            row["group"].title()
            for row in rows
        ],
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Accuracy difference (pp)"
    )

    ax.set_title(
        "Conditional image-index bootstrap: 95% CIs"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    save_figure(
        fig,
        "fig03_conditional_forest",
    )


# =========================================================
# FIGURE 4: PREDICTION OUTCOMES
# =========================================================

def figure_prediction_outcomes(
    rows,
):

    fig, ax = plt.subplots(
        figsize=(10, 4.6),
        layout="constrained",
    )

    x = np.arange(len(rows))

    ax.bar(
        x,
        [
            row["corrected"]
            for row in rows
        ],
        color="#169B62",
        label="Corrected",
    )

    ax.bar(
        x,
        [
            -row["harmed"]
            for row in rows
        ],
        color="#DC5252",
        label="Harmed",
    )

    ax.scatter(
        x,
        [
            row["net_correct"]
            for row in rows
        ],
        color="black",
        marker="D",
        s=22,
        label="Net correct",
        zorder=3,
    )

    ax.set_xticks(
        x,
        [
            f"{row['model'].title()}\n"
            f"{row['seed']}"
            for row in rows
        ],
        fontsize=8,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Prediction events"
    )

    ax.set_title(
        "Paired corrected and harmed predictions"
    )

    ax.legend()

    save_figure(
        fig,
        "fig04_paired_outcomes",
    )


# =========================================================
# FIGURE 5: DUPLICATE SENSITIVITY
# =========================================================

def figure_duplicates(
    duplicate_report,
):

    rows = duplicate_report[
        "sensitivity_results"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4.5),
        layout="constrained",
    )

    x = np.arange(len(rows))

    width = 0.34

    ax.bar(
        x - width / 2,
        [
            row["accuracy_delta_pp"]
            for row in rows
        ],
        width,
        color="#2563EB",
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        [
            row["macro_f1_delta_pp"]
            for row in rows
        ],
        width,
        color="#169B62",
        label="Macro-F1",
    )

    ax.set_xticks(
        x,
        [
            "All images",
            "One representative\nper hash group",
            "Equal weight\nper hash group",
        ],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "CAFA v3 - TENT (pp)"
    )

    ax.set_title(
        "Duplicate sensitivity of saved predictions"
    )

    ax.legend()

    save_figure(
        fig,
        "fig05_duplicate_sensitivity",
    )


# =========================================================
# FIGURE 6: PER-CLASS EFFECTS
# =========================================================

def figure_classes(
    rows,
):

    fig, ax = plt.subplots(
        figsize=(9, 4.8),
        layout="constrained",
    )

    x = np.arange(len(rows))

    width = 0.24

    for i, field, name, color in [
        (
            0,
            "precision_delta_pp",
            "Precision",
            "#64748B",
        ),
        (
            1,
            "recall_delta_pp",
            "Recall",
            "#2563EB",
        ),
        (
            2,
            "f1_delta_pp",
            "F1",
            "#169B62",
        ),
    ]:

        ax.bar(
            x + (i - 1) * width,
            [
                row[field]
                for row in rows
            ],
            width,
            label=name,
            color=color,
        )

    ax.set_xticks(
        x,
        [
            row["class_name"]
            for row in rows
        ],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "CAFA v3 - TENT (pp)"
    )

    ax.set_title(
        "Per-class performance differences"
    )

    ax.legend()

    save_figure(
        fig,
        "fig06_per_class_effects",
    )


# =========================================================
# FIGURE 7: NEGATIVE ADAPTATION
# =========================================================

def figure_negative(
    rows,
):

    fig, ax = plt.subplots(
        figsize=(9, 4.6),
        layout="constrained",
    )

    x = np.arange(len(MODELS))

    width = 0.24

    for j, method in enumerate(
        [
            "bn_only",
            "tent",
            "cafa_v3",
        ]
    ):

        values = [
            next(
                row["accuracy_delta_pp"]
                for row in rows
                if row["model"] == model
                and row["method"] == method
            )
            for model in MODELS
        ]

        ax.bar(
            x + (j - 1) * width,
            values,
            width,
            color=COLORS[method],
            label=DISPLAY[method],
        )

    ax.set_xticks(
        x,
        [model.title() for model in MODELS],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Accuracy vs source-only (pp)"
    )

    ax.set_title(
        "Positive and negative adaptation"
    )

    ax.legend()

    save_figure(
        fig,
        "fig07_negative_adaptation",
    )


# =========================================================
# FIGURE 8: ABSENT CLASS PREDICTION
# =========================================================

def figure_absent(
    runs,
):

    fig, ax = plt.subplots(
        figsize=(9, 4.6),
        layout="constrained",
    )

    x = np.arange(len(MODELS))

    width = 0.19

    for i, method in enumerate(
        METHODS
    ):

        values = []

        for model in MODELS:

            group = [
                run["row"][
                    "absent_class_prediction_rate"
                ] * 100
                for (m, a, seed), run
                in runs.items()
                if m == model
                and a == method
            ]

            values.append(
                mean(group)
            )

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            color=COLORS[method],
            label=DISPLAY[method],
        )

    ax.set_xticks(
        x,
        [model.title() for model in MODELS],
    )

    ax.set_ylabel(
        "Predictions into absent classes (%)"
    )

    ax.set_title(
        "Partial-label-space prediction errors"
    )

    ax.legend()

    save_figure(
        fig,
        "fig08_absent_class_predictions",
    )


# =========================================================
# FIGURE 9: CONFUSION DIFFERENCE
# =========================================================

def figure_confusion(
    difference,
):

    matrix = difference[
        TARGET_IDS,
        :
    ]

    limit = max(
        1,
        int(
            np.abs(matrix).max()
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 6),
        layout="constrained",
    )

    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:+d}",
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white"
                    if abs(value) > limit * 0.6
                    else "black"
                ),
            )

    ax.set_xticks(
        np.arange(9),
        CLASS_NAMES,
        rotation=30,
    )

    ax.set_yticks(
        np.arange(len(TARGET_IDS)),
        [
            CLASS_NAMES[i]
            for i in TARGET_IDS
        ],
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True target class"
    )

    ax.set_title(
        "Confusion difference: CAFA v3 - TENT"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Prediction count difference",
    )

    save_figure(
        fig,
        "fig09_confusion_difference",
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 75)

    print(
        "CAFA-TTA | STARC-9 STATISTICAL ANALYSIS"
    )

    print("=" * 75)

    print("Python:", platform.python_version())

    print("NumPy:", np.__version__)

    print("Device: CPU")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    require(
        not (
            OUTPUT_DIR
            / "statistical_report.json"
        ).exists(),
        "Completed statistical report already exists"
    )

    (
        metrics,
        protocol,
        preparation,
        labels,
        sample_rows,
    ) = load_inputs()

    runs = verify_predictions(
        metrics,
        labels,
    )

    (
        paired_rows,
        vectors,
        pair_keys,
    ) = paired_analysis(
        runs,
        labels,
    )

    require(
        len(paired_rows) == EXPECTED_PAIRS,
        "Unexpected number of paired comparisons"
    )

    corrected = sum(
        row["corrected"]
        for row in paired_rows
    )

    harmed = sum(
        row["harmed"]
        for row in paired_rows
    )

    net = corrected - harmed

    overall_accuracy_delta = mean([
        row["accuracy_delta_pp"]
        for row in paired_rows
    ])

    overall_f1_delta = mean([
        row["macro_f1_delta_pp"]
        for row in paired_rows
    ])

    require(
        abs(
            overall_accuracy_delta
            - metrics["overall"][
                "mean_accuracy_delta_pp"
            ]
        ) < 1e-10,
        "Overall accuracy delta mismatch"
    )

    require(
        abs(
            overall_f1_delta
            - metrics["overall"][
                "mean_macro_f1_delta_pp"
            ]
        ) < 1e-10,
        "Overall F1 delta mismatch"
    )

    require(
        abs(
            overall_accuracy_delta
            - net / (
                NUM_IMAGES * EXPECTED_PAIRS
            ) * 100
        ) < 1e-10,
        "Net correction reconstruction mismatch"
    )

    bootstrap_rows = conditional_bootstrap(
        vectors,
        pair_keys,
    )

    block_rows = circular_block_bootstrap(
        runs,
        vectors,
        pair_keys,
    )

    duplicate_report = duplicate_sensitivity(
        runs,
        labels,
        sample_rows,
        preparation,
        vectors,
        pair_keys,
    )

    class_rows = per_class_analysis(
        runs,
        labels,
        pair_keys,
    )

    negative_rows = negative_adaptation(
        runs,
    )

    absent_class_analysis(
        runs,
        labels,
    )

    (
        t_matrix,
        c_matrix,
        difference,
    ) = confusion_analysis(
        runs,
        pair_keys,
    )

    sensitivity_rows = group_sensitivity(
        paired_rows,
    )

    require(
        sum(
            row["net_correct"]
            for row in class_rows
        ) == net,
        "Per-class net correction mismatch"
    )

    require(
        int(np.trace(difference)) == net,
        "Confusion diagonal difference mismatch"
    )

    mean_absent_delta = mean([
        row["absent_delta_pp"]
        for row in paired_rows
    ])

    raw_significant = sum(
        row["raw_significant"]
        for row in paired_rows
    )

    holm_significant = sum(
        row["holm_significant"]
        for row in paired_rows
    )

    wins = sum(
        row["accuracy_outcome"] == "win"
        for row in paired_rows
    )

    ties = sum(
        row["accuracy_outcome"] == "tie"
        for row in paired_rows
    )

    losses = sum(
        row["accuracy_outcome"] == "loss"
        for row in paired_rows
    )

    overall_ci = next(
        row
        for row in bootstrap_rows
        if row["group"] == "overall"
    )

    configure_figures()

    print("\nGENERATING RESEARCH FIGURES")
    print("-" * 75)

    figure_performance(runs)

    figure_paired_effects(
        paired_rows
    )

    figure_forest(
        bootstrap_rows
    )

    figure_prediction_outcomes(
        paired_rows
    )

    figure_duplicates(
        duplicate_report
    )

    figure_classes(
        class_rows
    )

    figure_negative(
        negative_rows
    )

    figure_absent(
        runs
    )

    figure_confusion(
        difference
    )

    report = {
        "project": "CAFA-TTA",

        "dataset": "STARC-9",

        "analysis_type": (
            "external_post_hoc_statistical_analysis"
        ),

        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "verified_runs": EXPECTED_RUNS,

        "paired_comparisons": (
            EXPECTED_PAIRS
        ),

        "target_images": NUM_IMAGES,

        "target_class_ids": TARGET_IDS,

        "overall": {
            "accuracy_delta_pp": (
                overall_accuracy_delta
            ),
            "macro_f1_delta_pp": (
                overall_f1_delta
            ),
            "absent_prediction_delta_pp": (
                mean_absent_delta
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "corrected_events": corrected,
            "harmed_events": harmed,
            "net_correct_events": net,
            "corrected_harmed_ratio": (
                corrected / harmed
                if harmed > 0
                else None
            ),
            "mean_net_correct_per_run": (
                net / EXPECTED_PAIRS
            ),
        },

        "statistical_tests": {
            "method": "exact McNemar",
            "multiple_testing_correction": "Holm",
            "alpha": ALPHA,
            "tests": EXPECTED_PAIRS,
            "raw_significant": (
                raw_significant
            ),
            "holm_significant": (
                holm_significant
            ),
            "interpretation": (
                "Exploratory marginal paired tests; "
                "sample and trajectory dependence "
                "limits inferential interpretation."
            ),
        },

        "conditional_bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "overall_ci95_lower_pp": overall_ci[
                "ci95_lower_pp"
            ],
            "overall_ci95_upper_pp": overall_ci[
                "ci95_upper_pp"
            ],
            "groups": bootstrap_rows,
            "conditional_on_saved_trajectories": True,
            "patient_level_ci": False,
        },

        "block_bootstrap": {
            "block_lengths_batches": (
                BLOCK_LENGTHS
            ),
            "records": len(
                block_rows
            ),
            "conditional_on_saved_trajectories": True,
        },

        "duplicate_analysis": (
            duplicate_report
        ),

        "per_class_summary": (
            class_rows
        ),

        "negative_adaptation_summary": (
            negative_rows
        ),

        "leave_one_group_out": (
            sensitivity_rows
        ),

        "figure_count": 9,

        "model_evaluation_rerun": False,

        "target_label_training": False,

        "limitations": [
            "Statistical analysis was specified after "
            "the external evaluation results were observed.",

            "The McNemar tests are exploratory and "
            "do not account fully for within-slide, "
            "within-patient or online dependence.",

            "The conditional image-index bootstrap "
            "resamples saved correctness differences, "
            "not complete adaptation trajectories.",

            "Block bootstrap is a temporal sensitivity "
            "analysis, not proof of independent samples.",

            "The same target images are reused across "
            "models and stream seeds.",

            "Nine paired comparisons are not nine "
            "independent external datasets.",

            "Duplicate sensitivity changes scoring "
            "weights only, not the adaptation stream.",

            "Patient and slide identifiers are unavailable "
            "in the prepared dataset.",

            "Target selection uses six approximately "
            "compatible tissue categories.",

            "The target images are stain-normalized "
            "and resized to 28x28.",

            "Macro-F1 confidence intervals were "
            "not estimated by this script.",

            "Prediction events are not unique patients "
            "or independent clinical diagnoses.",

            "A small positive accuracy effect does "
            "not establish clinical utility.",
        ],
    }

    save_json(
        "statistical_report.json",
        report,
    )

    manifest = {
        "status": (
            "external_statistical_analysis_completed"
        ),

        "dataset": "STARC-9",

        "input_hashes": INPUT_HASHES,

        "analysis_script": relative(
            Path(__file__).resolve()
        ),

        "analysis_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),

        "output_hashes": {
            path.relative_to(
                OUTPUT_DIR
            ).as_posix(): sha256_file(path)
            for path in OUTPUT_FILES
        },

        "figure_count": 9,

        "csv_count": sum(
            path.suffix == ".csv"
            for path in OUTPUT_FILES
        ),

        "json_count": sum(
            path.suffix == ".json"
            for path in OUTPUT_FILES
        ),

        "output_count_excluding_manifest": (
            len(OUTPUT_FILES)
        ),
    }

    manifest_path = (
        OUTPUT_DIR
        / "statistical_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)

    print(
        "STARC-9 STATISTICAL ANALYSIS SUMMARY"
    )

    print("=" * 75)

    print(
        "Verified predictions:",
        len(runs),
        "/ 30",
    )

    print(
        "Paired comparisons:",
        len(paired_rows),
    )

    print(
        "Accuracy Delta:",
        f"{overall_accuracy_delta:+.6f} pp",
    )

    print(
        "Macro-F1 Delta:",
        f"{overall_f1_delta:+.6f} pp",
    )

    print(
        "Absent-class prediction Delta:",
        f"{mean_absent_delta:+.6f} pp",
    )

    print(
        "Wins/Ties/Losses:",
        f"{wins}/{ties}/{losses}",
    )

    print(
        "Corrected events:",
        corrected,
    )

    print(
        "Harmed events:",
        harmed,
    )

    print(
        "Net correct events:",
        net,
    )

    print(
        "Mean net correct per run:",
        f"{net / EXPECTED_PAIRS:.4f}",
    )

    print(
        "McNemar raw significant:",
        f"{raw_significant}/9",
    )

    print(
        "McNemar Holm significant:",
        f"{holm_significant}/9",
    )

    print(
        "Conditional 95% CI:",
        f"[{overall_ci['ci95_lower_pp']:+.6f}, "
        f"{overall_ci['ci95_upper_pp']:+.6f}] pp",
    )

    print(
        "Block bootstrap records:",
        len(block_rows),
    )

    print(
        "Duplicate groups:",
        duplicate_report[
            "duplicate_groups"
        ],
    )

    print(
        "Unique resized pixel groups:",
        duplicate_report[
            "unique_pixel_groups"
        ],
    )

    print(
        "Extra duplicate images:",
        duplicate_report[
            "extra_duplicate_images"
        ],
    )

    print("\nDUPLICATE SENSITIVITY RESULTS")

    for row in duplicate_report[
        "sensitivity_results"
    ]:

        print(
            f"{row['scenario']} | "
            f"Acc={row['accuracy_delta_pp']:+.6f} pp | "
            f"F1={row['macro_f1_delta_pp']:+.6f} pp | "
            f"W/T/L="
            f"{row['wins']}/"
            f"{row['ties']}/"
            f"{row['losses']}"
        )

    print("\nPER-CLASS RESULTS")

    for row in class_rows:

        print(
            f"{row['class_name']:5s} | "
            f"Recall="
            f"{row['recall_delta_pp']:+.6f} pp | "
            f"F1="
            f"{row['f1_delta_pp']:+.6f} pp | "
            f"Net="
            f"{row['net_correct']:+d}"
        )

    print("\nNEGATIVE ADAPTATION SUMMARY")

    for method in [
        "bn_only",
        "tent",
        "cafa_v3",
    ]:

        group = [
            row
            for row in negative_rows
            if row["method"] == method
        ]

        print(
            f"{method:10s} | "
            f"Accuracy negative="
            f"{sum(row['negative_accuracy_runs'] for row in group)}/9 | "
            f"F1 negative="
            f"{sum(row['negative_f1_runs'] for row in group)}/9"
        )

    print("\nOUTPUT SUMMARY")

    print(
        "Research figures: 9"
    )

    print(
        "PNG files: 9"
    )

    print(
        "PDF files: 9"
    )

    print(
        "CSV files:",
        manifest["csv_count"],
    )

    print(
        "Verified output files:",
        len(OUTPUT_FILES),
    )

    print(
        "Manifest SHA256:",
        sha256_file(manifest_path),
    )

    print(
        "Prediction integrity: PASS"
    )

    print(
        "Metric reconstruction: PASS"
    )

    print(
        "Per-class reconstruction: PASS"
    )

    print(
        "Confusion reconstruction: PASS"
    )

    print(
        "Model evaluation rerun: NO"
    )

    print(
        "Saved outputs:",
        OUTPUT_DIR,
    )

    print(
        "Final status: "
        "EXTERNAL STATISTICAL ANALYSIS COMPLETED"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()
