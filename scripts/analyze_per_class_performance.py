
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

from medmnist import INFO
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ==========================================
# CAFA-TTA | PER-CLASS PERFORMANCE ANALYSIS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "statistical_analysis"
)

CLASS_OUTPUT_DIR = (
    OUTPUT_DIR / "per_class"
)

CLASS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 9
NUM_SAMPLES = 7180
NUM_RUNS = 423
NUM_PAIRS = 117

MODELS = [
    "enhanced",
    "robust",
    "balanced",
]

SEEDS = [42, 123, 2026]

CLASSES = list(range(NUM_CLASSES))

CLASS_NAMES = {
    int(key): value
    for key, value
    in INFO["pathmnist"]["label"].items()
}

assert set(CLASS_NAMES) == set(CLASSES)

# ==========================================
# Utility functions
# ==========================================

def require(condition, message):

    if not condition:

        raise RuntimeError(message)


def read_json(path):

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:

        return json.load(f)


def read_csv(path):

    with open(
        path,
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(csv.DictReader(f))


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
            allow_nan=False,
        )

    temp_path.replace(path)


def save_csv(path, rows):

    require(
        len(rows) > 0,
        f"Empty output: {path}",
    )

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            hasher.update(chunk)

    return hasher.hexdigest().upper()


def mean_value(rows, field):

    return float(
        np.mean([
            row[field]
            for row in rows
        ])
    )


def normalize_confusion(matrix):

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix),
        where=totals != 0,
    )


# ==========================================
# Read verified statistical data
# ==========================================

integrity_path = (
    OUTPUT_DIR / "integrity_report.json"
)

statistical_path = (
    OUTPUT_DIR / "statistical_report.json"
)

prediction_index_path = (
    OUTPUT_DIR / "prediction_integrity.csv"
)

paired_path = (
    OUTPUT_DIR / "paired_predictions.csv"
)

labels_path = (
    OUTPUT_DIR / "canonical_test_labels.npy"
)

integrity_report = read_json(
    integrity_path
)

statistical_report = read_json(
    statistical_path
)

prediction_rows = read_csv(
    prediction_index_path
)

pairs = read_csv(
    paired_path
)

canonical_labels = np.load(
    labels_path,
    allow_pickle=False,
)

require(
    integrity_report["status"] == "passed",
    "Prediction integrity audit not passed",
)

require(
    statistical_report["status"]
    == "completed",
    "Statistical analysis incomplete",
)

require(
    len(prediction_rows) == NUM_RUNS,
    "Unexpected prediction record count",
)

require(
    len(pairs) == NUM_PAIRS,
    "Unexpected paired comparison count",
)

require(
    canonical_labels.shape == (
        NUM_SAMPLES,
    ),
    "Invalid canonical labels shape",
)

class_support = np.bincount(
    canonical_labels,
    minlength=NUM_CLASSES,
)

require(
    len(class_support) == NUM_CLASSES,
    "Invalid class distribution",
)

require(
    np.all(class_support > 0),
    "Some classes are missing",
)

# Verify original metrics files.

for relative_path, expected_hash in (
    integrity_report[
        "input_metrics_hashes"
    ].items()
):

    require(
        sha256_file(
            ROOT / relative_path
        ) == expected_hash,
        f"Metrics changed: {relative_path}",
    )

# Verify inputs used by previous statistics.

for relative_path, expected_hash in (
    statistical_report[
        "input_hashes"
    ].items()
):

    require(
        sha256_file(
            ROOT / relative_path
        ) == expected_hash,
        f"Statistical input changed: {relative_path}",
    )

# ==========================================
# Print initial information
# ==========================================

print("=" * 75)
print("CAFA-TTA | PER-CLASS PERFORMANCE")
print("Device: CPU")
print("Prediction files:", NUM_RUNS)
print("Paired comparisons:", NUM_PAIRS)
print("Number of classes:", NUM_CLASSES)
print("Test samples:", NUM_SAMPLES)
print("Input integrity: PASS")
print("=" * 75)

print("\nPATHMNIST CLASS LABELS")

for class_id in CLASSES:

    print(
        f"Class {class_id}: "
        f"{CLASS_NAMES[class_id]} | "
        f"Support={class_support[class_id]}"
    )

# ==========================================
# Verify and load prediction arrays
# ==========================================

prediction_cache = {}

per_run_class_rows = []

run_confusion_matrices = {}

run_metadata = {}

run_class_lookup = {}

condition_groups = defaultdict(list)

print("\n" + "=" * 75)
print("CALCULATING PER-CLASS METRICS")
print("=" * 75)

for run_number, row in enumerate(
    prediction_rows,
    start=1,
):

    relative_path = row[
        "predictions_file"
    ]

    path = ROOT / relative_path

    require(
        sha256_file(path)
        == row["predictions_sha256"],
        f"Prediction file changed: {relative_path}",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        indices = data["indices"]

        labels = data["labels"]

        predictions = data[
            "predictions"
        ]

    require(
        indices.shape == (NUM_SAMPLES,),
        "Invalid prediction indices",
    )

    require(
        labels.shape == (NUM_SAMPLES,),
        "Invalid labels shape",
    )

    require(
        predictions.shape == (
            NUM_SAMPLES,
        ),
        "Invalid predictions shape",
    )

    order = np.argsort(indices)

    require(
        np.array_equal(
            indices[order],
            np.arange(NUM_SAMPLES),
        ),
        f"Invalid indices: {relative_path}",
    )

    labels = labels[order]

    predictions = predictions[order]

    require(
        np.array_equal(
            labels,
            canonical_labels,
        ),
        f"Label mismatch: {relative_path}",
    )

    require(
        np.all(
            (predictions >= 0)
            & (predictions < NUM_CLASSES)
        ),
        f"Invalid predictions: {relative_path}",
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=CLASSES,
            average=None,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=CLASSES,
    )

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    require(
        abs(
            accuracy
            - float(row["accuracy"])
        ) < 1e-10,
        "Per-class accuracy mismatch",
    )

    require(
        abs(
            float(f1.mean())
            - float(row["macro_f1"])
        ) < 1e-10,
        "Per-class Macro-F1 mismatch",
    )

    require(
        np.array_equal(
            support,
            class_support,
        ),
        "Class support mismatch",
    )

    prediction_cache[
        relative_path
    ] = predictions.astype(
        np.int8
    )

    run_confusion_matrices[
        relative_path
    ] = matrix

    run_metadata[
        relative_path
    ] = {
        "family": row["family"],
        "condition": row["condition"],
        "model": row["model"],
        "method": row["method"],
        "seed": int(row["seed"]),
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
    }

    run_class_lookup[
        relative_path
    ] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    group_key = (
        row["family"],
        row["condition"],
        row["model"],
        row["method"],
    )

    condition_groups[
        group_key
    ].append(relative_path)

    for class_id in CLASSES:

        per_run_class_rows.append({
            "family": row["family"],
            "condition": row["condition"],
            "model": row["model"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "class_id": class_id,
            "class_name": CLASS_NAMES[
                class_id
            ],
            "support": int(
                support[class_id]
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
            "true_positive": int(
                matrix[class_id, class_id]
            ),
            "false_negative": int(
                matrix[class_id, :].sum()
                - matrix[class_id, class_id]
            ),
            "false_positive": int(
                matrix[:, class_id].sum()
                - matrix[class_id, class_id]
            ),
            "predictions_file": relative_path,
        })

    if (
        run_number % 25 == 0
        or run_number == NUM_RUNS
    ):

        print(
            f"[{run_number:3d}/{NUM_RUNS}] "
            "Per-class metrics calculated",
            flush=True,
        )

require(
    len(per_run_class_rows)
    == NUM_RUNS * NUM_CLASSES,
    "Unexpected per-class record count",
)

# ==========================================
# Paired per-class comparison
# ==========================================

paired_class_rows = []

pair_confusion_data = []

print("\n" + "=" * 75)
print("PAIRED PER-CLASS ANALYSIS")
print("=" * 75)

for pair_number, pair in enumerate(
    pairs,
    start=1,
):

    tent_file = pair[
        "tent_predictions_file"
    ]

    cafa_file = pair[
        "cafa_predictions_file"
    ]

    tent_predictions = (
        prediction_cache[tent_file]
    )

    cafa_predictions = (
        prediction_cache[cafa_file]
    )

    tent_metrics = (
        run_class_lookup[tent_file]
    )

    cafa_metrics = (
        run_class_lookup[cafa_file]
    )

    tent_correct = (
        tent_predictions
        == canonical_labels
    )

    cafa_correct = (
        cafa_predictions
        == canonical_labels
    )

    tent_matrix = (
        run_confusion_matrices[tent_file]
    )

    cafa_matrix = (
        run_confusion_matrices[cafa_file]
    )

    pair_confusion_data.append({
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "condition": pair["condition"],
        "model": pair["model"],
        "seed": int(pair["seed"]),
        "tent": tent_matrix,
        "cafa": cafa_matrix,
    })

    pair_net = 0
    pair_corrected = 0
    pair_harmed = 0

    for class_id in CLASSES:

        mask = (
            canonical_labels == class_id
        )

        support = int(
            mask.sum()
        )

        corrected = int(
            np.sum(
                mask
                & ~tent_correct
                & cafa_correct
            )
        )

        harmed = int(
            np.sum(
                mask
                & tent_correct
                & ~cafa_correct
            )
        )

        net_correct = (
            corrected - harmed
        )

        recall_delta_pp = (
            cafa_metrics["recall"][class_id]
            - tent_metrics["recall"][class_id]
        ) * 100

        require(
            abs(
                recall_delta_pp
                - net_correct / support * 100
            ) < 1e-10,
            "Class recall delta mismatch",
        )

        require(
            net_correct
            == int(
                cafa_matrix[class_id, class_id]
                - tent_matrix[class_id, class_id]
            ),
            "Class confusion matrix mismatch",
        )

        pair_net += net_correct
        pair_corrected += corrected
        pair_harmed += harmed

        paired_class_rows.append({
            "pair_id": pair["pair_id"],
            "family": pair["family"],
            "condition": pair["condition"],
            "model": pair["model"],
            "seed": int(pair["seed"]),
            "class_id": class_id,
            "class_name": CLASS_NAMES[
                class_id
            ],
            "support": support,
            "tent_precision": float(
                tent_metrics["precision"][class_id]
            ),
            "cafa_precision": float(
                cafa_metrics["precision"][class_id]
            ),
            "precision_delta_pp": float(
                (
                    cafa_metrics["precision"][class_id]
                    - tent_metrics["precision"][class_id]
                ) * 100
            ),
            "tent_recall": float(
                tent_metrics["recall"][class_id]
            ),
            "cafa_recall": float(
                cafa_metrics["recall"][class_id]
            ),
            "recall_delta_pp": float(
                recall_delta_pp
            ),
            "tent_f1": float(
                tent_metrics["f1"][class_id]
            ),
            "cafa_f1": float(
                cafa_metrics["f1"][class_id]
            ),
            "f1_delta_pp": float(
                (
                    cafa_metrics["f1"][class_id]
                    - tent_metrics["f1"][class_id]
                ) * 100
            ),
            "corrected": corrected,
            "harmed": harmed,
            "net_correct": net_correct,
        })

    require(
        pair_net == int(
            pair["net_correct"]
        ),
        "Paired net correction mismatch",
    )

    require(
        pair_corrected == int(
            pair["tent_wrong_cafa_correct"]
        ),
        "Paired corrected count mismatch",
    )

    require(
        pair_harmed == int(
            pair["tent_correct_cafa_wrong"]
        ),
        "Paired harmed count mismatch",
    )

    if (
        pair_number % 15 == 0
        or pair_number == NUM_PAIRS
    ):

        print(
            f"[{pair_number:3d}/{NUM_PAIRS}] "
            "Paired class analysis completed",
            flush=True,
        )

require(
    len(paired_class_rows)
    == NUM_PAIRS * NUM_CLASSES,
    "Unexpected paired class record count",
)

# ==========================================
# Aggregate per-class performance
# ==========================================

def summarize_classes(rows):

    output = []

    for class_id in CLASSES:

        items = [
            row for row in rows
            if row["class_id"] == class_id
        ]

        require(
            len(items) > 0,
            f"Missing class: {class_id}",
        )

        corrected = sum(
            row["corrected"]
            for row in items
        )

        harmed = sum(
            row["harmed"]
            for row in items
        )

        output.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[
                class_id
            ],
            "support": int(
                class_support[class_id]
            ),
            "comparisons": len(items),
            "tent_precision": mean_value(
                items,
                "tent_precision",
            ),
            "cafa_precision": mean_value(
                items,
                "cafa_precision",
            ),
            "precision_delta_pp": mean_value(
                items,
                "precision_delta_pp",
            ),
            "tent_recall": mean_value(
                items,
                "tent_recall",
            ),
            "cafa_recall": mean_value(
                items,
                "cafa_recall",
            ),
            "recall_delta_pp": mean_value(
                items,
                "recall_delta_pp",
            ),
            "tent_f1": mean_value(
                items,
                "tent_f1",
            ),
            "cafa_f1": mean_value(
                items,
                "cafa_f1",
            ),
            "f1_delta_pp": mean_value(
                items,
                "f1_delta_pp",
            ),
            "corrected_events": corrected,
            "harmed_events": harmed,
            "net_correct_events": (
                corrected - harmed
            ),
            "mean_net_correct_per_run": (
                (corrected - harmed)
                / len(items)
            ),
        })

    return output


overall_classes = summarize_classes(
    paired_class_rows
)

by_model = {}

for model in MODELS:

    by_model[model] = summarize_classes([
        row for row in paired_class_rows
        if row["model"] == model
    ])

by_family = {}

for family in [
    "severity",
    "corruption",
]:

    by_family[family] = summarize_classes([
        row for row in paired_class_rows
        if row["family"] == family
    ])

by_condition = {}

condition_keys = sorted({
    (
        row["family"],
        row["condition"],
    )
    for row in paired_class_rows
})

for family, condition in condition_keys:

    name = f"{family}/{condition}"

    by_condition[name] = summarize_classes([
        row for row in paired_class_rows
        if row["family"] == family
        and row["condition"] == condition
    ])

by_corruption = {}

for corruption in [
    "gaussian_noise",
    "gaussian_blur",
    "hue_shift",
]:

    by_corruption[corruption] = (
        summarize_classes([
            row
            for row in paired_class_rows
            if row["family"] == "corruption"
            and row["condition"].startswith(
                corruption + "/"
            )
        ])
    )

# ==========================================
# Verify overall improvement reconstruction
# ==========================================

total_corrected = sum(
    row["corrected_events"]
    for row in overall_classes
)

total_harmed = sum(
    row["harmed_events"]
    for row in overall_classes
)

total_net = (
    total_corrected - total_harmed
)

require(
    total_corrected
    == integrity_report["overall"][
        "total_corrected_prediction_events"
    ],
    "Overall corrected count mismatch",
)

require(
    total_harmed
    == integrity_report["overall"][
        "total_harmed_prediction_events"
    ],
    "Overall harmed count mismatch",
)

reconstructed_accuracy_delta = (
    sum(
        row["recall_delta_pp"]
        * row["support"]
        for row in overall_classes
    ) / NUM_SAMPLES
)

reconstructed_macro_f1_delta = (
    mean_value(
        overall_classes,
        "f1_delta_pp",
    )
)

require(
    abs(
        reconstructed_accuracy_delta
        - integrity_report["overall"][
            "mean_accuracy_delta_pp"
        ]
    ) < 1e-10,
    "Reconstructed Accuracy mismatch",
)

require(
    abs(
        reconstructed_macro_f1_delta
        - integrity_report["overall"][
            "mean_macro_f1_delta_pp"
        ]
    ) < 1e-10,
    "Reconstructed Macro-F1 mismatch",
)

# ==========================================
# Minority class analysis
# ==========================================

# Descriptive definition:
# classes with support below average
# class support in the test set.

average_support = (
    NUM_SAMPLES / NUM_CLASSES
)

minority_ids = [
    class_id
    for class_id in CLASSES
    if class_support[class_id]
    < average_support
]

majority_ids = [
    class_id
    for class_id in CLASSES
    if class_id not in minority_ids
]


def summarize_class_group(class_ids):

    rows = [
        row for row in overall_classes
        if row["class_id"] in class_ids
    ]

    total_support = sum(
        row["support"]
        for row in rows
    )

    return {
        "classes": list(class_ids),
        "class_names": [
            CLASS_NAMES[class_id]
            for class_id in class_ids
        ],
        "unique_test_images": total_support,
        "mean_precision_delta_pp": (
            mean_value(
                rows,
                "precision_delta_pp",
            )
        ),
        "macro_recall_delta_pp": (
            mean_value(
                rows,
                "recall_delta_pp",
            )
        ),
        "support_weighted_recall_delta_pp": (
            sum(
                row["recall_delta_pp"]
                * row["support"]
                for row in rows
            ) / total_support
        ),
        "macro_f1_delta_pp": mean_value(
            rows,
            "f1_delta_pp",
        ),
        "corrected_events": sum(
            row["corrected_events"]
            for row in rows
        ),
        "harmed_events": sum(
            row["harmed_events"]
            for row in rows
        ),
        "net_correct_events": sum(
            row["net_correct_events"]
            for row in rows
        ),
    }


minority_summary = {
    "definition": (
        "class support below mean test class support"
    ),
    "average_class_support": average_support,
    "minority": summarize_class_group(
        minority_ids
    ),
    "majority": summarize_class_group(
        majority_ids
    ),
}

# ==========================================
# Confusion matrix analysis
# ==========================================

def summarize_confusions(items):

    tent_count_matrix = np.sum(
        [
            item["tent"]
            for item in items
        ],
        axis=0,
    )

    cafa_count_matrix = np.sum(
        [
            item["cafa"]
            for item in items
        ],
        axis=0,
    )

    difference_count_matrix = (
        cafa_count_matrix
        - tent_count_matrix
    )

    tent_normalized = (
        normalize_confusion(
            tent_count_matrix
        )
    )

    cafa_normalized = (
        normalize_confusion(
            cafa_count_matrix
        )
    )

    difference_pp = (
        (
            cafa_normalized
            - tent_normalized
        ) * 100
    )

    require(
        int(tent_count_matrix.sum())
        == len(items) * NUM_SAMPLES,
        "Invalid aggregate TENT matrix",
    )

    require(
        int(cafa_count_matrix.sum())
        == len(items) * NUM_SAMPLES,
        "Invalid aggregate CAFA matrix",
    )

    require(
        np.all(
            difference_count_matrix.sum(
                axis=1
            ) == 0
        ),
        "Confusion matrix difference mismatch",
    )

    return {
        "paired_comparisons": len(items),
        "tent_counts": (
            tent_count_matrix.tolist()
        ),
        "cafa_counts": (
            cafa_count_matrix.tolist()
        ),
        "difference_counts": (
            difference_count_matrix.tolist()
        ),
        "tent_row_normalized": (
            tent_normalized.tolist()
        ),
        "cafa_row_normalized": (
            cafa_normalized.tolist()
        ),
        "difference_percentage_points": (
            difference_pp.tolist()
        ),
    }


confusion_summary = {
    "class_names": CLASS_NAMES,
    "overall": summarize_confusions(
        pair_confusion_data
    ),
    "by_model": {},
    "by_family": {},
    "by_condition": {},
}

for model in MODELS:

    confusion_summary["by_model"][
        model
    ] = summarize_confusions([
        item
        for item in pair_confusion_data
        if item["model"] == model
    ])

for family in [
    "severity",
    "corruption",
]:

    confusion_summary["by_family"][
        family
    ] = summarize_confusions([
        item
        for item in pair_confusion_data
        if item["family"] == family
    ])

for family, condition in condition_keys:

    name = f"{family}/{condition}"

    confusion_summary["by_condition"][
        name
    ] = summarize_confusions([
        item
        for item in pair_confusion_data
        if item["family"] == family
        and item["condition"] == condition
    ])

# ==========================================
# Identify changed misclassification patterns
# ==========================================

overall_confusion = (
    confusion_summary["overall"]
)

difference_counts = np.asarray(
    overall_confusion[
        "difference_counts"
    ]
)

difference_pp = np.asarray(
    overall_confusion[
        "difference_percentage_points"
    ]
)

require(
    int(
        np.trace(difference_counts)
    ) == total_net,
    "Confusion matrix net improvement mismatch",
)

confusion_change_rows = []

for true_class in CLASSES:

    for predicted_class in CLASSES:

        if true_class == predicted_class:

            continue

        change = int(
            difference_counts[
                true_class,
                predicted_class,
            ]
        )

        confusion_change_rows.append({
            "true_class_id": true_class,
            "true_class_name": CLASS_NAMES[
                true_class
            ],
            "predicted_class_id": predicted_class,
            "predicted_class_name": CLASS_NAMES[
                predicted_class
            ],
            "difference_events": change,
            "difference_recall_row_pp": float(
                difference_pp[
                    true_class,
                    predicted_class,
                ]
            ),
        })

reduced_confusions = sorted(
    [
        row
        for row in confusion_change_rows
        if row["difference_events"] < 0
    ],
    key=lambda row: row[
        "difference_events"
    ],
)

increased_confusions = sorted(
    [
        row
        for row in confusion_change_rows
        if row["difference_events"] > 0
    ],
    key=lambda row: row[
        "difference_events"
    ],
    reverse=True,
)

# ==========================================
# Negative adaptation analysis
# ==========================================

negative_adaptation_rows = []

negative_class_rows = []

main_methods = [
    "bn_only",
    "tent",
    "cafa_v3",
]

for family, condition in condition_keys:

    for model in MODELS:

        source_files = condition_groups[
            (
                family,
                condition,
                model,
                "source_only",
            )
        ]

        source_files = sorted(
            source_files,
            key=lambda path: (
                run_metadata[path]["seed"]
            ),
        )

        require(
            len(source_files) in [1, 3],
            "Unexpected source-only run count",
        )

        source_file = source_files[0]

        source_predictions = (
            prediction_cache[source_file]
        )

        # Source-only should not depend
        # on the order of images.

        for path in source_files[1:]:

            require(
                np.array_equal(
                    source_predictions,
                    prediction_cache[path],
                ),
                "Source-only predictions differ across seeds",
            )

        source_metrics = (
            run_metadata[source_file]
        )

        source_class_metrics = (
            run_class_lookup[source_file]
        )

        for method in main_methods:

            method_files = condition_groups[
                (
                    family,
                    condition,
                    model,
                    method,
                )
            ]

            require(
                len(method_files) == 3,
                "Unexpected adaptation run count",
            )

            adapted_accuracy = float(
                np.mean([
                    run_metadata[path][
                        "accuracy"
                    ]
                    for path in method_files
                ])
            )

            adapted_macro_f1 = float(
                np.mean([
                    run_metadata[path][
                        "macro_f1"
                    ]
                    for path in method_files
                ])
            )

            accuracy_delta = (
                adapted_accuracy
                - source_metrics["accuracy"]
            ) * 100

            f1_delta = (
                adapted_macro_f1
                - source_metrics["macro_f1"]
            ) * 100

            negative_adaptation_rows.append({
                "family": family,
                "condition": condition,
                "model": model,
                "method": method,
                "source_accuracy": (
                    source_metrics["accuracy"]
                ),
                "adapted_accuracy": (
                    adapted_accuracy
                ),
                "accuracy_delta_vs_source_pp": (
                    accuracy_delta
                ),
                "source_macro_f1": (
                    source_metrics["macro_f1"]
                ),
                "adapted_macro_f1": (
                    adapted_macro_f1
                ),
                "macro_f1_delta_vs_source_pp": (
                    f1_delta
                ),
                "negative_accuracy_adaptation": (
                    accuracy_delta < -1e-10
                ),
                "negative_f1_adaptation": (
                    f1_delta < -1e-10
                ),
            })

            for class_id in CLASSES:

                source_recall = float(
                    source_class_metrics[
                        "recall"
                    ][class_id]
                )

                adapted_recall = float(
                    np.mean([
                        run_class_lookup[path][
                            "recall"
                        ][class_id]
                        for path in method_files
                    ])
                )

                source_f1 = float(
                    source_class_metrics[
                        "f1"
                    ][class_id]
                )

                adapted_f1 = float(
                    np.mean([
                        run_class_lookup[path][
                            "f1"
                        ][class_id]
                        for path in method_files
                    ])
                )

                negative_class_rows.append({
                    "family": family,
                    "condition": condition,
                    "model": model,
                    "method": method,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[
                        class_id
                    ],
                    "support": int(
                        class_support[class_id]
                    ),
                    "source_recall": source_recall,
                    "adapted_recall": adapted_recall,
                    "recall_delta_vs_source_pp": (
                        (
                            adapted_recall
                            - source_recall
                        ) * 100
                    ),
                    "source_f1": source_f1,
                    "adapted_f1": adapted_f1,
                    "f1_delta_vs_source_pp": (
                        (
                            adapted_f1
                            - source_f1
                        ) * 100
                    ),
                })

require(
    len(negative_adaptation_rows)
    == 117,
    "Unexpected negative adaptation records",
)

require(
    len(negative_class_rows)
    == 117 * NUM_CLASSES,
    "Unexpected class adaptation records",
)


def summarize_adaptation(rows):

    return {
        "conditions": len(rows),
        "negative_accuracy_conditions": sum(
            row["negative_accuracy_adaptation"]
            for row in rows
        ),
        "negative_f1_conditions": sum(
            row["negative_f1_adaptation"]
            for row in rows
        ),
        "mean_accuracy_delta_vs_source_pp": (
            mean_value(
                rows,
                "accuracy_delta_vs_source_pp",
            )
        ),
        "mean_macro_f1_delta_vs_source_pp": (
            mean_value(
                rows,
                "macro_f1_delta_vs_source_pp",
            )
        ),
    }


negative_adaptation_summary = {
    "overall": {},
    "by_model": {},
}

for method in main_methods:

    rows = [
        row
        for row in negative_adaptation_rows
        if row["method"] == method
    ]

    negative_adaptation_summary[
        "overall"
    ][method] = summarize_adaptation(
        rows
    )

for model in MODELS:

    negative_adaptation_summary[
        "by_model"
    ][model] = {}

    for method in main_methods:

        rows = [
            row
            for row in negative_adaptation_rows
            if row["model"] == model
            and row["method"] == method
        ]

        negative_adaptation_summary[
            "by_model"
        ][model][method] = (
            summarize_adaptation(rows)
        )

# ==========================================
# Print final per-class summary
# ==========================================

print("\n" + "=" * 75)
print("OVERALL PER-CLASS PERFORMANCE")
print("=" * 75)

for row in overall_classes:

    print(
        f"\nCLASS {row['class_id']}: "
        f"{row['class_name']}"
    )

    print(
        "  Support:",
        row["support"],
    )

    print(
        f"  Precision: "
        f"TENT={row['tent_precision']:.4f} | "
        f"CAFA={row['cafa_precision']:.4f} | "
        f"Delta={row['precision_delta_pp']:+.4f} pp"
    )

    print(
        f"  Recall:    "
        f"TENT={row['tent_recall']:.4f} | "
        f"CAFA={row['cafa_recall']:.4f} | "
        f"Delta={row['recall_delta_pp']:+.4f} pp"
    )

    print(
        f"  F1:        "
        f"TENT={row['tent_f1']:.4f} | "
        f"CAFA={row['cafa_f1']:.4f} | "
        f"Delta={row['f1_delta_pp']:+.4f} pp"
    )

    print(
        f"  Corrected="
        f"{row['corrected_events']} | "
        f"Harmed="
        f"{row['harmed_events']} | "
        f"Net="
        f"{row['net_correct_events']:+d}"
    )

# ==========================================
# Print per-class improvement rankings
# ==========================================

print("\n" + "=" * 75)
print("CLASS RANKING | RECALL IMPROVEMENT")
print("=" * 75)

ranked_classes = sorted(
    overall_classes,
    key=lambda row: row[
        "recall_delta_pp"
    ],
    reverse=True,
)

for rank, row in enumerate(
    ranked_classes,
    start=1,
):

    print(
        f"{rank}. "
        f"Class {row['class_id']} | "
        f"{row['class_name']:25s} | "
        f"Recall Delta="
        f"{row['recall_delta_pp']:+.4f} pp | "
        f"F1 Delta="
        f"{row['f1_delta_pp']:+.4f} pp"
    )

# ==========================================
# Print minority class analysis
# ==========================================

print("\n" + "=" * 75)
print("MINORITY VS MAJORITY CLASS ANALYSIS")
print("=" * 75)

print(
    "Mean class support:",
    f"{average_support:.2f}"
)

for group_name in [
    "minority",
    "majority",
]:

    row = minority_summary[
        group_name
    ]

    print(
        f"\n{group_name.upper()} CLASSES"
    )

    print(
        "  Class IDs:",
        row["classes"],
    )

    print(
        "  Unique test images:",
        row["unique_test_images"],
    )

    print(
        "  Macro Recall Delta:",
        f"{row['macro_recall_delta_pp']:+.4f} pp"
    )

    print(
        "  Macro F1 Delta:",
        f"{row['macro_f1_delta_pp']:+.4f} pp"
    )

    print(
        "  Weighted Recall Delta:",
        f"{row['support_weighted_recall_delta_pp']:+.4f} pp"
    )

    print(
        "  Corrected events:",
        row["corrected_events"],
    )

    print(
        "  Harmed events:",
        row["harmed_events"],
    )

    print(
        "  Net correct events:",
        row["net_correct_events"],
    )

# ==========================================
# Print confusion changes
# ==========================================

print("\n" + "=" * 75)
print("TOP REDUCED MISCLASSIFICATIONS")
print("=" * 75)

for row in reduced_confusions[:10]:

    print(
        f"{row['true_class_name']} "
        f"-> {row['predicted_class_name']} | "
        f"Events="
        f"{row['difference_events']:+d} | "
        f"Delta="
        f"{row['difference_recall_row_pp']:+.4f} pp"
    )

print("\n" + "=" * 75)
print("TOP INCREASED MISCLASSIFICATIONS")
print("=" * 75)

for row in increased_confusions[:10]:

    print(
        f"{row['true_class_name']} "
        f"-> {row['predicted_class_name']} | "
        f"Events="
        f"{row['difference_events']:+d} | "
        f"Delta="
        f"{row['difference_recall_row_pp']:+.4f} pp"
    )

# ==========================================
# Print negative adaptation summary
# ==========================================

print("\n" + "=" * 75)
print("NEGATIVE ADAPTATION | VS SOURCE-ONLY")
print("=" * 75)

for method, row in (
    negative_adaptation_summary[
        "overall"
    ].items()
):

    print(
        f"{method:12s} | "
        f"Negative Acc="
        f"{row['negative_accuracy_conditions']}"
        f"/{row['conditions']} | "
        f"Negative F1="
        f"{row['negative_f1_conditions']}"
        f"/{row['conditions']} | "
        f"Mean Acc Delta="
        f"{row['mean_accuracy_delta_vs_source_pp']:+.4f} pp"
    )

print("\nCAFA v3 BY SOURCE MODEL")

for model in MODELS:

    row = negative_adaptation_summary[
        "by_model"
    ][model]["cafa_v3"]

    print(
        f"{model:10s} | "
        f"Negative Acc="
        f"{row['negative_accuracy_conditions']}"
        f"/{row['conditions']} | "
        f"Negative F1="
        f"{row['negative_f1_conditions']}"
        f"/{row['conditions']} | "
        f"Mean Acc Delta="
        f"{row['mean_accuracy_delta_vs_source_pp']:+.4f} pp"
    )

# ==========================================
# Save detailed outputs
# ==========================================

save_csv(
    CLASS_OUTPUT_DIR
    / "per_run_class_metrics.csv",
    per_run_class_rows,
)

save_csv(
    CLASS_OUTPUT_DIR
    / "paired_class_metrics.csv",
    paired_class_rows,
)

save_csv(
    CLASS_OUTPUT_DIR
    / "overall_class_summary.csv",
    overall_classes,
)

save_csv(
    CLASS_OUTPUT_DIR
    / "confusion_changes.csv",
    confusion_change_rows,
)

save_csv(
    CLASS_OUTPUT_DIR
    / "negative_adaptation.csv",
    negative_adaptation_rows,
)

save_csv(
    CLASS_OUTPUT_DIR
    / "negative_adaptation_by_class.csv",
    negative_class_rows,
)

group_summary_rows = []

for group_type, groups in [
    ("model", by_model),
    ("family", by_family),
    ("condition", by_condition),
    ("corruption", by_corruption),
]:

    for group_name, rows in groups.items():

        for row in rows:

            group_summary_rows.append({
                "group_type": group_type,
                "group_name": group_name,
                **row,
            })

save_csv(
    CLASS_OUTPUT_DIR
    / "grouped_class_summary.csv",
    group_summary_rows,
)

# Confusion matrix CSV files.

for matrix_name, field in [
    ("tent_confusion_counts", "tent_counts"),
    ("cafa_confusion_counts", "cafa_counts"),
    (
        "confusion_difference_counts",
        "difference_counts",
    ),
    (
        "tent_confusion_normalized",
        "tent_row_normalized",
    ),
    (
        "cafa_confusion_normalized",
        "cafa_row_normalized",
    ),
    (
        "confusion_difference_pp",
        "difference_percentage_points",
    ),
]:

    matrix = overall_confusion[field]

    rows = []

    for class_id in CLASSES:

        row = {
            "true_class_id": class_id,
            "true_class_name": CLASS_NAMES[
                class_id
            ],
        }

        for predicted_class in CLASSES:

            row[
                f"predicted_class_{predicted_class}"
            ] = matrix[
                class_id
            ][predicted_class]

        rows.append(row)

    save_csv(
        CLASS_OUTPUT_DIR
        / f"{matrix_name}.csv",
        rows,
    )

save_json(
    CLASS_OUTPUT_DIR
    / "confusion_matrices.json",
    confusion_summary,
)

# ==========================================
# Save comprehensive per-class report
# ==========================================

final_report = {
    "status": "completed",
    "dataset": "PathMNIST",
    "split": "test",
    "class_names": CLASS_NAMES,
    "samples_per_run": NUM_SAMPLES,
    "verified_prediction_files": NUM_RUNS,
    "paired_comparisons": NUM_PAIRS,
    "per_run_class_records": len(
        per_run_class_rows
    ),
    "paired_class_records": len(
        paired_class_rows
    ),
    "accuracy_delta_pp": (
        reconstructed_accuracy_delta
    ),
    "macro_f1_delta_pp": (
        reconstructed_macro_f1_delta
    ),
    "total_corrected_events": total_corrected,
    "total_harmed_events": total_harmed,
    "total_net_correct_events": total_net,
    "overall_classes": overall_classes,
    "by_model": by_model,
    "by_family": by_family,
    "by_condition": by_condition,
    "by_corruption": by_corruption,
    "minority_class_analysis": (
        minority_summary
    ),
    "top_reduced_confusions": (
        reduced_confusions[:10]
    ),
    "top_increased_confusions": (
        increased_confusions[:10]
    ),
    "negative_adaptation": (
        negative_adaptation_summary
    ),
    "input_hashes": {
        path.relative_to(ROOT).as_posix():
        sha256_file(path)

        for path in [
            integrity_path,
            statistical_path,
            prediction_index_path,
            paired_path,
            labels_path,
        ]
    },
    "analysis_script_sha256": sha256_file(
        Path(__file__)
    ),
    "limitations": [
        "Per-class analysis is descriptive.",
        "Class groups are defined using test support.",
        "Minority classes are not defined by clinical rarity.",
        "The same test images are reused across conditions.",
        "Prediction events are not unique patients.",
        "Aggregate confusion counts are repeated prediction events.",
        "Per-class metrics are averaged across evaluated runs.",
        "Aggregated confusion matrices are normalized by true class.",
        "Confusion matrix differences do not identify individual prediction transitions.",
        "Negative adaptation is defined relative to source-only performance.",
        "No per-class significance claim is made.",
        "Synthetic shifts do not establish clinical or cross-hospital validity.",
    ],
}

save_json(
    CLASS_OUTPUT_DIR
    / "per_class_report.json",
    final_report,
)

# ==========================================
# Final status
# ==========================================

print("\n" + "=" * 75)
print("FINAL PER-CLASS ANALYSIS SUMMARY")
print("=" * 75)

print(
    "Verified prediction files:",
    len(prediction_cache),
    "/",
    NUM_RUNS,
)

print(
    "Per-run class records:",
    len(per_run_class_rows),
)

print(
    "Paired class records:",
    len(paired_class_rows),
)

print(
    "Negative adaptation records:",
    len(negative_adaptation_rows),
)

print(
    "Accuracy reconstruction:",
    f"{reconstructed_accuracy_delta:+.4f} pp"
)

print(
    "Macro-F1 reconstruction:",
    f"{reconstructed_macro_f1_delta:+.4f} pp"
)

print(
    "Total corrected events:",
    total_corrected,
)

print(
    "Total harmed events:",
    total_harmed,
)

print(
    "Total net correct events:",
    total_net,
)

print(
    "Class metrics consistency: PASS"
)

print(
    "Confusion matrix consistency: PASS"
)

print(
    "Aggregate metrics consistency: PASS"
)

print("Saved outputs:", CLASS_OUTPUT_DIR)

print("Final status: ANALYSIS COMPLETED")

print("=" * 75)
