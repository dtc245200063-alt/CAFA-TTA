
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# CAFA-TTA | PREDICTION INTEGRITY AUDIT
# ==========================================

NUM_CLASSES = 9
NUM_SAMPLES = 7180
BATCH_SIZE = 64

SEEDS = [42, 123, 2026]

MODELS = [
    "enhanced",
    "robust",
    "balanced",
]

EXPERIMENTS = {
    "strong": {
        "folder": "cafa_final_test_strong",
        "expected_records": 63,
        "expected_files": 63,
    },
    "severity": {
        "folder": "cafa_final_test_severity",
        "expected_records": 126,
        "expected_files": 90,
    },
    "corruption": {
        "folder": "cafa_final_test_corruptions",
        "expected_records": 270,
        "expected_files": 270,
    },
}

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "statistical_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOLERANCE = 1e-10

# ==========================================
# Utility functions
# ==========================================

def require(condition, message):

    if not condition:

        raise RuntimeError(message)


def sha256_file(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            hasher.update(chunk)

    return hasher.hexdigest().upper()


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
        f"No rows for {path}",
    )

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def read_json(path):

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:

        return json.load(f)


# ==========================================
# Load and verify experiment metadata
# ==========================================

print("=" * 75)
print("CAFA-TTA | FINAL TEST PREDICTION AUDIT")
print("Expected samples:", NUM_SAMPLES)
print("Expected unique prediction files: 423")
print("Expected paired comparisons: 117")
print("=" * 75)

experiment_data = {}
input_hashes = {}

for name, specification in EXPERIMENTS.items():

    folder = (
        ROOT
        / "experiments"
        / specification["folder"]
    )

    metrics_path = folder / "metrics.json"

    require(
        metrics_path.exists(),
        f"Missing metrics: {metrics_path}",
    )

    data = read_json(metrics_path)

    require(
        data["split"] == "test",
        f"Invalid split: {name}",
    )

    require(
        data["protocol"]
        == "shuffled_online_prequential",
        f"Protocol mismatch: {name}",
    )

    require(
        data["batch_size"] == BATCH_SIZE,
        f"Batch size mismatch: {name}",
    )

    require(
        data["seeds"] == SEEDS,
        f"Seed mismatch: {name}",
    )

    require(
        set(data["models"]) == set(MODELS),
        f"Model mismatch: {name}",
    )

    require(
        data["configuration_frozen"] is True,
        f"Configuration not frozen: {name}",
    )

    require(
        len(data["results"])
        == specification["expected_records"],
        f"Unexpected record count: {name}",
    )

    files = list(
        (folder / "predictions").rglob("*.npz")
    )

    require(
        len(files)
        == specification["expected_files"],
        f"Unexpected prediction count: {name}",
    )

    input_hashes[
        metrics_path.relative_to(ROOT).as_posix()
    ] = sha256_file(metrics_path)

    experiment_data[name] = data

    print(
        f"{name.upper():12s} | "
        f"Records={len(data['results'])} | "
        f"Files={len(files)} | "
        "Metadata=PASS"
    )


reference = experiment_data["strong"]

for name, data in experiment_data.items():

    for field in [
        "checkpoint_hashes",
        "algorithm_hashes",
        "learning_rate",
        "entropy_floor",
        "lambda_align",
        "min_samples",
        "models",
    ]:

        require(
            data[field] == reference[field],
            f"Metadata mismatch: {name}/{field}",
        )

print("\nCross-experiment metadata: PASS")

# ==========================================
# Remove duplicate strong-shift records
# ==========================================

records_by_file = {}

duplicate_records = 0

for experiment, data in experiment_data.items():

    for original_row in data["results"]:

        row = dict(original_row)

        if experiment == "strong":

            row["family"] = "severity"
            row["condition"] = "strong"

        elif experiment == "severity":

            row["family"] = "severity"
            row["condition"] = row["severity"]

        else:

            row["family"] = "corruption"

            row["condition"] = (
                row["corruption"]
                + "/"
                + row["severity"]
            )

        file_key = row["predictions_file"]

        if file_key in records_by_file:

            previous = records_by_file[file_key]

            for field in [
                "model",
                "method",
                "seed",
                "samples",
                "accuracy",
                "macro_f1",
                "confusion_matrix",
                "predictions_sha256",
                "family",
                "condition",
            ]:

                require(
                    row[field] == previous[field],
                    f"Duplicate mismatch: "
                    f"{file_key}/{field}",
                )

            duplicate_records += 1

        else:

            records_by_file[file_key] = row


records = list(
    records_by_file.values()
)

require(
    len(records) == 423,
    "Expected 423 unique prediction records",
)

require(
    duplicate_records == 36,
    "Expected 36 duplicate strong-shift records",
)

listed_paths = {
    (ROOT / row["predictions_file"]).resolve()
    for row in records
}

disk_paths = set()

for specification in EXPERIMENTS.values():

    folder = (
        ROOT
        / "experiments"
        / specification["folder"]
        / "predictions"
    )

    disk_paths.update(
        path.resolve()
        for path in folder.rglob("*.npz")
    )

require(
    listed_paths == disk_paths,
    "Metrics and prediction file lists differ",
)

print("\nUnique prediction records:", len(records))
print("Duplicate records removed:", duplicate_records)
print("Metrics/file inventory: PASS")

# ==========================================
# Verify every prediction file
# ==========================================

print("\n" + "=" * 75)
print("VERIFYING PREDICTION FILES")
print("=" * 75)

verified_data = {}
integrity_rows = []

canonical_labels = None

streams_by_seed = {}

maximum_accuracy_error = 0.0
maximum_f1_error = 0.0

for run_number, row in enumerate(
    records,
    start=1,
):

    relative_path = row["predictions_file"]

    path = ROOT / relative_path

    require(
        path.exists(),
        f"Missing prediction file: {path}",
    )

    actual_hash = sha256_file(path)

    require(
        actual_hash
        == row["predictions_sha256"].upper(),
        f"SHA-256 mismatch: {relative_path}",
    )

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
            required_keys.issubset(data.files),
            f"Missing array: {relative_path}",
        )

        indices = np.asarray(
            data["indices"]
        )

        labels = np.asarray(
            data["labels"]
        )

        predictions = np.asarray(
            data["predictions"]
        )

        saved_batch_size = int(
            data["batch_size"].item()
        )

    require(
        saved_batch_size == BATCH_SIZE,
        f"Saved batch size mismatch: {relative_path}",
    )

    for array_name, array in [
        ("indices", indices),
        ("labels", labels),
        ("predictions", predictions),
    ]:

        require(
            array.shape == (NUM_SAMPLES,),
            f"Wrong array shape: "
            f"{relative_path}/{array_name}",
        )

        require(
            np.issubdtype(
                array.dtype,
                np.integer,
            ),
            f"Non-integer array: "
            f"{relative_path}/{array_name}",
        )

    require(
        row["samples"] == NUM_SAMPLES,
        f"Sample count mismatch: {relative_path}",
    )

    require(
        np.array_equal(
            np.sort(indices),
            np.arange(NUM_SAMPLES),
        ),
        f"Invalid image indices: {relative_path}",
    )

    require(
        np.all(
            (labels >= 0)
            & (labels < NUM_CLASSES)
        ),
        f"Invalid label: {relative_path}",
    )

    require(
        np.all(
            (predictions >= 0)
            & (predictions < NUM_CLASSES)
        ),
        f"Invalid prediction: {relative_path}",
    )

    # Align labels by original image index.

    index_order = np.argsort(indices)

    indexed_labels = labels[index_order]

    if canonical_labels is None:

        canonical_labels = indexed_labels.copy()

    else:

        require(
            np.array_equal(
                indexed_labels,
                canonical_labels,
            ),
            f"Ground truth mismatch: {relative_path}",
        )

    # Verify same stream order for the same seed.

    seed = row["seed"]

    require(
        seed in SEEDS,
        f"Unexpected seed: {relative_path}",
    )

    if seed not in streams_by_seed:

        streams_by_seed[seed] = indices.copy()

    else:

        require(
            np.array_equal(
                indices,
                streams_by_seed[seed],
            ),
            f"Stream order mismatch: "
            f"{relative_path}",
        )

    # Recalculate classification metrics.

    calculated_accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    calculated_f1 = float(
        f1_score(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )

    calculated_cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    accuracy_error = abs(
        calculated_accuracy - row["accuracy"]
    )

    f1_error = abs(
        calculated_f1 - row["macro_f1"]
    )

    maximum_accuracy_error = max(
        maximum_accuracy_error,
        accuracy_error,
    )

    maximum_f1_error = max(
        maximum_f1_error,
        f1_error,
    )

    require(
        accuracy_error <= TOLERANCE,
        f"Accuracy mismatch: {relative_path}",
    )

    require(
        f1_error <= TOLERANCE,
        f"Macro-F1 mismatch: {relative_path}",
    )

    require(
        np.array_equal(
            calculated_cm,
            np.asarray(row["confusion_matrix"]),
        ),
        f"Confusion matrix mismatch: {relative_path}",
    )

    require(
        int(calculated_cm.sum()) == NUM_SAMPLES,
        f"Confusion matrix total mismatch: "
        f"{relative_path}",
    )

    verified_data[relative_path] = {
        "indices": indices,
        "labels": labels,
        "predictions": predictions,
        "accuracy": calculated_accuracy,
        "macro_f1": calculated_f1,
    }

    integrity_rows.append({
        "family": row["family"],
        "condition": row["condition"],
        "model": row["model"],
        "method": row["method"],
        "seed": row["seed"],
        "samples": NUM_SAMPLES,
        "accuracy": calculated_accuracy,
        "macro_f1": calculated_f1,
        "accuracy_error": accuracy_error,
        "macro_f1_error": f1_error,
        "sha256_verified": True,
        "labels_verified": True,
        "stream_verified": True,
        "confusion_matrix_verified": True,
        "predictions_file": relative_path,
        "predictions_sha256": actual_hash,
    })

    if (
        run_number % 25 == 0
        or run_number == len(records)
    ):

        print(
            f"[{run_number:3d}/423] "
            "All integrity checks PASS",
            flush=True,
        )

require(
    len(verified_data) == 423,
    "Prediction verification incomplete",
)

require(
    set(streams_by_seed) == set(SEEDS),
    "Missing verified seed streams",
)

# ==========================================
# Build paired TENT / CAFA v3 comparisons
# ==========================================

print("\n" + "=" * 75)
print("PAIRED PREDICTION ANALYSIS")
print("=" * 75)

pair_groups = defaultdict(dict)

for row in records:

    if row["method"] not in [
        "tent",
        "cafa_v3",
    ]:

        continue

    key = (
        row["family"],
        row["condition"],
        row["model"],
        row["seed"],
    )

    require(
        row["method"] not in pair_groups[key],
        f"Duplicate paired method: {key}",
    )

    pair_groups[key][row["method"]] = row

require(
    len(pair_groups) == 117,
    "Expected 117 paired comparisons",
)

expected_conditions = {
    "severity": [
        "original",
        "light",
        "medium",
        "strong",
    ],
    "corruption": [
        corruption + "/" + severity

        for corruption in [
            "gaussian_noise",
            "gaussian_blur",
            "hue_shift",
        ]

        for severity in [
            "light",
            "medium",
            "strong",
        ]
    ],
}

expected_pair_keys = {
    (family, condition, model, seed)

    for family, conditions
    in expected_conditions.items()

    for condition in conditions
    for model in MODELS
    for seed in SEEDS
}

require(
    set(pair_groups) == expected_pair_keys,
    "Unexpected paired conditions",
)

paired_rows = []

block_rows = []

for key in sorted(pair_groups):

    family, condition, model, seed = key

    methods = pair_groups[key]

    require(
        set(methods) == {"tent", "cafa_v3"},
        f"Incomplete pair: {key}",
    )

    tent_row = methods["tent"]

    cafa_row = methods["cafa_v3"]

    tent_file = tent_row["predictions_file"]

    cafa_file = cafa_row["predictions_file"]

    tent = verified_data[tent_file]

    cafa = verified_data[cafa_file]

    require(
        np.array_equal(
            tent["indices"],
            cafa["indices"],
        ),
        f"Paired index mismatch: {key}",
    )

    require(
        np.array_equal(
            tent["labels"],
            cafa["labels"],
        ),
        f"Paired label mismatch: {key}",
    )

    labels = tent["labels"]

    tent_correct = (
        tent["predictions"] == labels
    )

    cafa_correct = (
        cafa["predictions"] == labels
    )

    # a: both correct
    # b: TENT correct, CAFA wrong
    # c: TENT wrong, CAFA correct
    # d: both wrong

    a = int(
        np.sum(tent_correct & cafa_correct)
    )

    b = int(
        np.sum(tent_correct & ~cafa_correct)
    )

    c = int(
        np.sum(~tent_correct & cafa_correct)
    )

    d = int(
        np.sum(~tent_correct & ~cafa_correct)
    )

    require(
        a + b + c + d == NUM_SAMPLES,
        f"Invalid contingency counts: {key}",
    )

    net_correct = c - b

    accuracy_delta_pp = (
        net_correct / NUM_SAMPLES * 100
    )

    direct_delta_pp = (
        cafa["accuracy"] - tent["accuracy"]
    ) * 100

    require(
        abs(
            accuracy_delta_pp - direct_delta_pp
        ) <= TOLERANCE,
        f"Paired accuracy mismatch: {key}",
    )

    changed_predictions = int(
        np.sum(
            tent["predictions"]
            != cafa["predictions"]
        )
    )

    macro_f1_delta_pp = (
        cafa["macro_f1"] - tent["macro_f1"]
    ) * 100

    outcome = (
        "win"
        if net_correct > 0
        else "loss"
        if net_correct < 0
        else "tie"
    )

    pair_id = (
        f"{family}__"
        f"{condition.replace('/', '_')}__"
        f"{model}__"
        f"seed{seed}"
    )

    paired_rows.append({
        "pair_id": pair_id,
        "family": family,
        "condition": condition,
        "model": model,
        "seed": seed,
        "samples": NUM_SAMPLES,
        "tent_accuracy": tent["accuracy"],
        "cafa_accuracy": cafa["accuracy"],
        "tent_macro_f1": tent["macro_f1"],
        "cafa_macro_f1": cafa["macro_f1"],
        "accuracy_delta_pp": accuracy_delta_pp,
        "macro_f1_delta_pp": macro_f1_delta_pp,
        "both_correct": a,
        "tent_correct_cafa_wrong": b,
        "tent_wrong_cafa_correct": c,
        "both_wrong": d,
        "net_correct": net_correct,
        "discordant_correctness": b + c,
        "changed_predictions": changed_predictions,
        "outcome": outcome,
        "tent_predictions_file": tent_file,
        "cafa_predictions_file": cafa_file,
    })

    # ======================================
    # Preserve batch-level paired outcomes
    # for later dependence-aware analysis
    # ======================================

    pair_blocks = []

    for batch_index, start in enumerate(
        range(0, NUM_SAMPLES, BATCH_SIZE)
    ):

        end = min(
            start + BATCH_SIZE,
            NUM_SAMPLES,
        )

        t = tent_correct[start:end]

        v = cafa_correct[start:end]

        block_a = int(np.sum(t & v))
        block_b = int(np.sum(t & ~v))
        block_c = int(np.sum(~t & v))
        block_d = int(np.sum(~t & ~v))

        block = {
            "pair_id": pair_id,
            "family": family,
            "condition": condition,
            "model": model,
            "seed": seed,
            "batch_index": batch_index,
            "stream_start": start,
            "stream_end_exclusive": end,
            "samples": end - start,
            "both_correct": block_a,
            "tent_correct_cafa_wrong": block_b,
            "tent_wrong_cafa_correct": block_c,
            "both_wrong": block_d,
            "net_correct": block_c - block_b,
        }

        pair_blocks.append(block)

        block_rows.append(block)

    require(
        len(pair_blocks) == 113,
        f"Unexpected batch count: {key}",
    )

    require(
        sum(
            block["net_correct"]
            for block in pair_blocks
        ) == net_correct,
        f"Batch difference mismatch: {key}",
    )

    require(
        sum(
            block["samples"]
            for block in pair_blocks
        ) == NUM_SAMPLES,
        f"Batch sample mismatch: {key}",
    )

    if (
        len(paired_rows) % 15 == 0
        or len(paired_rows) == 117
    ):

        print(
            f"[{len(paired_rows):3d}/117] "
            "Paired predictions verified",
            flush=True,
        )


require(
    len(block_rows) == 117 * 113,
    "Unexpected batch-level record count",
)

# ==========================================
# Descriptive summary
# ==========================================

def summarize(rows):

    require(
        len(rows) > 0,
        "Empty summary group",
    )

    outcomes = Counter(
        row["outcome"]
        for row in rows
    )

    acc = np.asarray([
        row["accuracy_delta_pp"]
        for row in rows
    ])

    f1 = np.asarray([
        row["macro_f1_delta_pp"]
        for row in rows
    ])

    net = np.asarray([
        row["net_correct"]
        for row in rows
    ])

    corrected = np.asarray([
        row["tent_wrong_cafa_correct"]
        for row in rows
    ])

    harmed = np.asarray([
        row["tent_correct_cafa_wrong"]
        for row in rows
    ])

    return {
        "paired_comparisons": len(rows),
        "mean_accuracy_delta_pp": float(
            acc.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            f1.mean()
        ),
        "median_accuracy_delta_pp": float(
            np.median(acc)
        ),
        "min_accuracy_delta_pp": float(
            acc.min()
        ),
        "max_accuracy_delta_pp": float(
            acc.max()
        ),
        "mean_corrected_per_run": float(
            corrected.mean()
        ),
        "mean_harmed_per_run": float(
            harmed.mean()
        ),
        "mean_net_correct_per_run": float(
            net.mean()
        ),
        "total_corrected_prediction_events": int(
            corrected.sum()
        ),
        "total_harmed_prediction_events": int(
            harmed.sum()
        ),
        "total_net_correct_events": int(
            net.sum()
        ),
        "wins": int(outcomes["win"]),
        "ties": int(outcomes["tie"]),
        "losses": int(outcomes["loss"]),
    }


overall = summarize(paired_rows)

by_family = {}

for family in expected_conditions:

    rows = [
        row for row in paired_rows
        if row["family"] == family
    ]

    by_family[family] = summarize(rows)


by_model = {}

for model in MODELS:

    rows = [
        row for row in paired_rows
        if row["model"] == model
    ]

    by_model[model] = summarize(rows)


by_seed = {}

for seed in SEEDS:

    rows = [
        row for row in paired_rows
        if row["seed"] == seed
    ]

    by_seed[str(seed)] = summarize(rows)


by_condition = {}

for family, conditions in (
    expected_conditions.items()
):

    for condition in conditions:

        rows = [
            row for row in paired_rows
            if row["family"] == family
            and row["condition"] == condition
        ]

        name = f"{family}/{condition}"

        by_condition[name] = summarize(rows)


by_corruption = {}

for corruption in [
    "gaussian_noise",
    "gaussian_blur",
    "hue_shift",
]:

    rows = [
        row for row in paired_rows
        if row["family"] == "corruption"
        and row["condition"].startswith(
            corruption + "/"
        )
    ]

    by_corruption[corruption] = summarize(rows)

# ==========================================
# Verify aggregate results against metrics
# ==========================================

for experiment, family in [
    ("severity", "severity"),
    ("corruption", "corruption"),
]:

    saved = experiment_data[experiment]["overall"]

    recalculated = by_family[family]

    for field in [
        "mean_accuracy_delta_pp",
        "mean_macro_f1_delta_pp",
    ]:

        require(
            abs(
                saved[field] - recalculated[field]
            ) <= TOLERANCE,
            f"Aggregate mismatch: "
            f"{experiment}/{field}",
        )

    for field in [
        "wins",
        "ties",
        "losses",
    ]:

        require(
            saved[field] == recalculated[field],
            f"Aggregate outcome mismatch: "
            f"{experiment}/{field}",
        )

strong_summary = by_condition[
    "severity/strong"
]

for field in [
    "mean_accuracy_delta_pp",
    "mean_macro_f1_delta_pp",
]:

    require(
        abs(
            reference["overall"][field]
            - strong_summary[field]
        ) <= TOLERANCE,
        f"Strong-shift aggregate mismatch: {field}",
    )

print("\nAggregate metrics verification: PASS")

# ==========================================
# Report class distribution
# ==========================================

class_counts = np.bincount(
    canonical_labels,
    minlength=NUM_CLASSES,
)

class_distribution = {
    str(class_index): int(count)

    for class_index, count
    in enumerate(class_counts)
}

print("\n" + "=" * 75)
print("TEST CLASS DISTRIBUTION")
print("=" * 75)

for class_index, count in (
    class_distribution.items()
):

    print(
        f"Class {class_index}: "
        f"{count:4d} images "
        f"({count / NUM_SAMPLES:.2%})"
    )

# ==========================================
# Print summaries
# ==========================================

def print_summary(name, summary):

    print(
        f"{name:22s} | "
        f"Pairs={summary['paired_comparisons']:3d} | "
        f"Acc Delta="
        f"{summary['mean_accuracy_delta_pp']:+.4f} pp | "
        f"F1 Delta="
        f"{summary['mean_macro_f1_delta_pp']:+.4f} pp | "
        f"W/T/L="
        f"{summary['wins']}/"
        f"{summary['ties']}/"
        f"{summary['losses']}"
    )


print("\n" + "=" * 75)
print("PAIRED RESULTS | BY EXPERIMENT")
print("=" * 75)

for name, summary in by_family.items():

    print_summary(name, summary)


print("\n" + "=" * 75)
print("PAIRED RESULTS | BY SOURCE MODEL")
print("=" * 75)

for name, summary in by_model.items():

    print_summary(name, summary)


print("\n" + "=" * 75)
print("PAIRED RESULTS | BY CORRUPTION TYPE")
print("=" * 75)

for name, summary in by_corruption.items():

    print_summary(name, summary)


print("\n" + "=" * 75)
print("PAIRED RESULTS | BY SEED")
print("=" * 75)

for name, summary in by_seed.items():

    print_summary(name, summary)


print("\n" + "=" * 75)
print("OVERALL PAIRED PREDICTION SUMMARY")
print("=" * 75)

print_summary("All conditions", overall)

print(
    "Mean corrected predictions per run:",
    f"{overall['mean_corrected_per_run']:.4f}"
)

print(
    "Mean harmed predictions per run:",
    f"{overall['mean_harmed_per_run']:.4f}"
)

print(
    "Mean net correct predictions per run:",
    f"{overall['mean_net_correct_per_run']:+.4f}"
)

print(
    "Total corrected prediction events:",
    overall["total_corrected_prediction_events"],
)

print(
    "Total harmed prediction events:",
    overall["total_harmed_prediction_events"],
)

print(
    "Total net correct prediction events:",
    overall["total_net_correct_events"],
)

print(
    "NOTE: Prediction events are not "
    "independent or unique patients."
)

# ==========================================
# Save analysis outputs
# ==========================================

save_csv(
    OUTPUT_DIR / "prediction_integrity.csv",
    integrity_rows,
)

save_csv(
    OUTPUT_DIR / "paired_predictions.csv",
    paired_rows,
)

save_csv(
    OUTPUT_DIR / "paired_batch_outcomes.csv",
    block_rows,
)

np.save(
    OUTPUT_DIR / "canonical_test_labels.npy",
    canonical_labels,
    allow_pickle=False,
)

report = {
    "status": "passed",
    "dataset": "PathMNIST",
    "split": "test",
    "samples_per_run": NUM_SAMPLES,
    "batch_size": BATCH_SIZE,
    "seeds": SEEDS,
    "unique_prediction_files": len(records),
    "duplicate_records_removed": duplicate_records,
    "verified_prediction_files": len(verified_data),
    "paired_comparisons": len(paired_rows),
    "batch_level_records": len(block_rows),
    "maximum_accuracy_error": (
        maximum_accuracy_error
    ),
    "maximum_macro_f1_error": (
        maximum_f1_error
    ),
    "input_metrics_hashes": input_hashes,
    "audit_script_sha256": sha256_file(
        Path(__file__)
    ),
    "class_distribution": class_distribution,
    "overall": overall,
    "by_family": by_family,
    "by_model": by_model,
    "by_seed": by_seed,
    "by_condition": by_condition,
    "by_corruption": by_corruption,
    "limitations": [
        "The test set was accessed previously.",
        "The same test images are reused.",
        "Seeds represent stream order, not independent training.",
        "Online adaptation introduces dependence across batches.",
        "Prediction events are not unique images or patients.",
        "Synthetic shifts do not establish real-world cross-domain validity.",
        "No statistical significance claim is made by this audit.",
    ],
}

save_json(
    OUTPUT_DIR / "integrity_report.json",
    report,
)

# ==========================================
# Final audit status
# ==========================================

print("\n" + "=" * 75)
print("FINAL PREDICTION INTEGRITY AUDIT")
print("=" * 75)

print(
    "Verified prediction files:",
    len(verified_data),
    "/ 423",
)

print(
    "Duplicate records removed:",
    duplicate_records,
)

print("SHA-256 verification: PASS")
print("Accuracy verification: PASS")
print("Macro-F1 verification: PASS")
print("Confusion matrix verification: PASS")
print("Ground truth consistency: PASS")
print("Stream order consistency: PASS")
print("Aggregate metrics verification: PASS")

print(
    "Verified paired comparisons:",
    len(paired_rows),
    "/ 117",
)

print(
    "Batch-level records:",
    len(block_rows),
)

print(
    "Maximum Accuracy error:",
    f"{maximum_accuracy_error:.12e}"
)

print(
    "Maximum Macro-F1 error:",
    f"{maximum_f1_error:.12e}"
)

print("Final status: ALL CHECKS PASSED")
print("Saved outputs:", OUTPUT_DIR)
print("=" * 75)
