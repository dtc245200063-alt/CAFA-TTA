
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

# ==========================================
# CAFA-TTA | STATISTICAL ANALYSIS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "statistical_analysis"
)

ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 2026

BLOCK_LENGTHS = [1, 4, 8, 16]

NUM_SAMPLES = 7180
NUM_PAIRS = 117
NUM_BATCHES = 113

SEEDS = [42, 123, 2026]

MODELS = [
    "enhanced",
    "robust",
    "balanced",
]

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

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


def confidence_interval(values):

    lower, upper = np.quantile(
        values,
        [
            ALPHA / 2,
            1 - ALPHA / 2,
        ],
    )

    return float(lower), float(upper)


# ==========================================
# Load verified input data
# ==========================================

report_path = (
    OUTPUT_DIR / "integrity_report.json"
)

paired_path = (
    OUTPUT_DIR / "paired_predictions.csv"
)

blocks_path = (
    OUTPUT_DIR / "paired_batch_outcomes.csv"
)

integrity_path = (
    OUTPUT_DIR / "prediction_integrity.csv"
)

report = read_json(report_path)

pairs = read_csv(paired_path)

blocks = read_csv(blocks_path)

integrity_rows = read_csv(integrity_path)

require(
    report["status"] == "passed",
    "Prediction integrity audit not passed",
)

require(
    report["verified_prediction_files"] == 423,
    "Incomplete prediction verification",
)

require(
    len(pairs) == NUM_PAIRS,
    "Unexpected paired comparison count",
)

require(
    len(blocks) == NUM_PAIRS * NUM_BATCHES,
    "Unexpected batch record count",
)

require(
    len(integrity_rows) == 423,
    "Unexpected integrity record count",
)

# Check that original metrics have not changed.

for relative_path, expected_hash in (
    report["input_metrics_hashes"].items()
):

    require(
        sha256_file(ROOT / relative_path)
        == expected_hash,
        f"Metrics changed: {relative_path}",
    )

input_hashes = {
    path.relative_to(ROOT).as_posix():
        sha256_file(path)

    for path in [
        report_path,
        paired_path,
        blocks_path,
        integrity_path,
    ]
}

verified_hashes = {
    row["predictions_file"]:
        row["predictions_sha256"]

    for row in integrity_rows
}

require(
    len(verified_hashes) == 423,
    "Duplicate integrity records",
)

# ==========================================
# Convert numeric data
# ==========================================

PAIR_INT_FIELDS = [
    "seed",
    "samples",
    "both_correct",
    "tent_correct_cafa_wrong",
    "tent_wrong_cafa_correct",
    "both_wrong",
    "net_correct",
    "discordant_correctness",
    "changed_predictions",
]

PAIR_FLOAT_FIELDS = [
    "tent_accuracy",
    "cafa_accuracy",
    "tent_macro_f1",
    "cafa_macro_f1",
    "accuracy_delta_pp",
    "macro_f1_delta_pp",
]

for row in pairs:

    for field in PAIR_INT_FIELDS:

        row[field] = int(row[field])

    for field in PAIR_FLOAT_FIELDS:

        row[field] = float(row[field])

BLOCK_INT_FIELDS = [
    "seed",
    "batch_index",
    "stream_start",
    "stream_end_exclusive",
    "samples",
    "both_correct",
    "tent_correct_cafa_wrong",
    "tent_wrong_cafa_correct",
    "both_wrong",
    "net_correct",
]

for row in blocks:

    for field in BLOCK_INT_FIELDS:

        row[field] = int(row[field])

blocks_by_pair = defaultdict(list)

for row in blocks:

    blocks_by_pair[
        row["pair_id"]
    ].append(row)

pair_ids = {
    row["pair_id"]
    for row in pairs
}

require(
    len(pair_ids) == NUM_PAIRS,
    "Duplicate pair identifiers",
)

require(
    set(blocks_by_pair) == pair_ids,
    "Batch and pair identifiers differ",
)

print("=" * 75)
print("CAFA-TTA | STATISTICAL ANALYSIS")
print("Device: CPU")
print("Paired comparisons:", len(pairs))
print("Batch records:", len(blocks))
print("Bootstrap repeats:", BOOTSTRAP_REPEATS)
print("Block lengths:", BLOCK_LENGTHS)
print("Alpha:", ALPHA)
print("Input integrity: PASS")
print("=" * 75)

# ==========================================
# Holm multiple-testing correction
# ==========================================

def holm_adjust(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)

    running_maximum = 0.0

    for rank, p_value in enumerate(
        sorted_p
    ):

        adjusted = (
            (m - rank) * p_value
        )

        running_maximum = max(
            running_maximum,
            adjusted,
        )

        adjusted_sorted[rank] = min(
            running_maximum,
            1.0,
        )

    adjusted_p = np.empty(m)

    adjusted_p[order] = adjusted_sorted

    return adjusted_p


# ==========================================
# Exact McNemar test
# ==========================================

def exact_mcnemar(b, c):

    discordant = b + c

    if discordant == 0:

        return 1.0

    return float(
        binomtest(
            k=c,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


# ==========================================
# Image-level paired bootstrap
# ==========================================

def paired_bootstrap(
    harmed,
    corrected,
    samples,
):

    unchanged = (
        samples
        - harmed
        - corrected
    )

    require(
        unchanged >= 0,
        "Invalid contingency counts",
    )

    probabilities = np.array(
        [
            harmed,
            unchanged,
            corrected,
        ],
        dtype=float,
    ) / samples

    # Equivalent to resampling paired
    # correctness differences -1, 0, +1.

    draws = rng.multinomial(
        samples,
        probabilities,
        size=BOOTSTRAP_REPEATS,
    )

    deltas = (
        draws[:, 2]
        - draws[:, 0]
    ) / samples * 100

    return confidence_interval(deltas)


# ==========================================
# Circular moving-block bootstrap
# ==========================================

def circular_block_bootstrap(
    batch_net,
    batch_sizes,
    block_length,
):

    batch_net = np.asarray(
        batch_net,
        dtype=np.int64,
    )

    batch_sizes = np.asarray(
        batch_sizes,
        dtype=np.int64,
    )

    n_batches = len(batch_net)

    n_blocks = int(
        np.ceil(
            n_batches / block_length
        )
    )

    starts = rng.integers(
        0,
        n_batches,
        size=(
            BOOTSTRAP_REPEATS,
            n_blocks,
        ),
    )

    offsets = np.arange(
        block_length
    )

    indices = (
        starts[:, :, None]
        + offsets[None, None, :]
    ) % n_batches

    indices = indices.reshape(
        BOOTSTRAP_REPEATS,
        -1,
    )[:, :n_batches]

    net_counts = batch_net[
        indices
    ].sum(axis=1)

    sample_counts = batch_sizes[
        indices
    ].sum(axis=1)

    deltas = (
        net_counts
        / sample_counts
        * 100
    )

    return confidence_interval(deltas)


# ==========================================
# Verify batch-level data
# ==========================================

def verify_pair_blocks(pair):

    rows = sorted(
        blocks_by_pair[pair["pair_id"]],
        key=lambda row: row["batch_index"],
    )

    require(
        len(rows) == NUM_BATCHES,
        f"Invalid batch count: {pair['pair_id']}",
    )

    require(
        [r["batch_index"] for r in rows]
        == list(range(NUM_BATCHES)),
        f"Invalid batch indices: {pair['pair_id']}",
    )

    expected_start = 0

    for row in rows:

        for field in [
            "family",
            "condition",
            "model",
            "seed",
        ]:

            require(
                row[field] == pair[field],
                f"Batch metadata mismatch: {field}",
            )

        require(
            row["stream_start"] == expected_start,
            "Non-consecutive batch positions",
        )

        require(
            row["stream_end_exclusive"]
            - row["stream_start"]
            == row["samples"],
            "Invalid batch size",
        )

        require(
            row["samples"]
            == min(64, NUM_SAMPLES - expected_start),
            "Unexpected batch length",
        )

        for field in [
            "both_correct",
            "tent_correct_cafa_wrong",
            "tent_wrong_cafa_correct",
            "both_wrong",
        ]:

            require(
                row[field] >= 0,
                "Negative contingency count",
            )

        require(
            row["both_correct"]
            + row["tent_correct_cafa_wrong"]
            + row["tent_wrong_cafa_correct"]
            + row["both_wrong"]
            == row["samples"],
            "Invalid batch contingency table",
        )

        require(
            row["net_correct"]
            == row["tent_wrong_cafa_correct"]
            - row["tent_correct_cafa_wrong"],
            "Invalid batch net correction",
        )

        expected_start = row[
            "stream_end_exclusive"
        ]

    require(
        expected_start == NUM_SAMPLES,
        "Incomplete batch coverage",
    )

    for field in [
        "samples",
        "both_correct",
        "tent_correct_cafa_wrong",
        "tent_wrong_cafa_correct",
        "both_wrong",
        "net_correct",
    ]:

        require(
            sum(r[field] for r in rows)
            == pair[field],
            f"Batch aggregate mismatch: {field}",
        )

    return rows


# ==========================================
# Per-pair statistical analysis
# ==========================================

statistical_rows = []
block_ci_rows = []

# Store per-image correctness differences.
# Align all comparisons to original image IDs.

indexed_differences = []

print("\n" + "=" * 75)
print("PER-PAIR STATISTICAL ANALYSIS")
print("=" * 75)

for pair_number, pair in enumerate(
    pairs,
    start=1,
):

    a = pair["both_correct"]

    b = pair[
        "tent_correct_cafa_wrong"
    ]

    c = pair[
        "tent_wrong_cafa_correct"
    ]

    d = pair["both_wrong"]

    n = pair["samples"]

    require(
        n == NUM_SAMPLES,
        "Unexpected sample count",
    )

    require(
        a + b + c + d == n,
        "Invalid contingency table",
    )

    require(
        c - b == pair["net_correct"],
        "Net correction mismatch",
    )

    batch_rows = verify_pair_blocks(
        pair
    )

    # Reload verified prediction files.

    prediction_arrays = {}

    for method in ["tent", "cafa"]:

        relative_path = pair[
            f"{method}_predictions_file"
        ]

        path = ROOT / relative_path

        require(
            sha256_file(path)
            == verified_hashes[relative_path],
            f"Prediction changed: {relative_path}",
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as saved:

            prediction_arrays[method] = {
                "indices": saved["indices"],
                "labels": saved["labels"],
                "predictions": saved["predictions"],
            }

    tent = prediction_arrays["tent"]
    cafa = prediction_arrays["cafa"]

    require(
        np.array_equal(
            tent["indices"],
            cafa["indices"],
        ),
        "Paired image order mismatch",
    )

    require(
        np.array_equal(
            tent["labels"],
            cafa["labels"],
        ),
        "Paired label mismatch",
    )

    tent_correct = (
        tent["predictions"]
        == tent["labels"]
    )

    cafa_correct = (
        cafa["predictions"]
        == cafa["labels"]
    )

    differences = (
        cafa_correct.astype(np.int8)
        - tent_correct.astype(np.int8)
    )

    require(
        int((differences == -1).sum()) == b,
        "Harmed count mismatch",
    )

    require(
        int((differences == 1).sum()) == c,
        "Corrected count mismatch",
    )

    require(
        int(differences.sum()) == c - b,
        "Prediction net correction mismatch",
    )

    index_order = np.argsort(
        tent["indices"]
    )

    indexed_differences.append(
        differences[index_order]
    )

    # Exact McNemar p-value.

    p_value = exact_mcnemar(
        b,
        c,
    )

    image_lower, image_upper = (
        paired_bootstrap(
            b,
            c,
            n,
        )
    )

    # Effect size.
    # Raw ratio undefined when b = 0.
    # Also report a continuity-corrected ratio.

    raw_ratio = (
        c / b
        if b > 0
        else None
    )

    corrected_ratio = (
        (c + 0.5)
        / (b + 0.5)
    )

    entry = {
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "condition": pair["condition"],
        "model": pair["model"],
        "seed": pair["seed"],
        "samples": n,
        "tent_accuracy": pair["tent_accuracy"],
        "cafa_accuracy": pair["cafa_accuracy"],
        "accuracy_delta_pp": (
            (c - b) / n * 100
        ),
        "macro_f1_delta_pp": (
            pair["macro_f1_delta_pp"]
        ),
        "both_correct": a,
        "harmed": b,
        "corrected": c,
        "both_wrong": d,
        "net_correct": c - b,
        "discordant": b + c,
        "corrected_harmed_ratio": raw_ratio,
        "continuity_corrected_ratio": (
            corrected_ratio
        ),
        "mcnemar_p_raw": p_value,
        "image_bootstrap_ci_lower_pp": (
            image_lower
        ),
        "image_bootstrap_ci_upper_pp": (
            image_upper
        ),
        "image_ci_excludes_zero": (
            image_lower > 0
            or image_upper < 0
        ),
    }

    # Block-level sensitivity intervals.

    batch_net = [
        row["net_correct"]
        for row in batch_rows
    ]

    batch_sizes = [
        row["samples"]
        for row in batch_rows
    ]

    # Confirm saved batch differences.

    for row in batch_rows:

        start = row["stream_start"]

        end = row[
            "stream_end_exclusive"
        ]

        require(
            int(
                differences[start:end].sum()
            ) == row["net_correct"],
            "Prediction/batch mismatch",
        )

    for block_length in BLOCK_LENGTHS:

        lower, upper = (
            circular_block_bootstrap(
                batch_net,
                batch_sizes,
                block_length,
            )
        )

        block_ci_rows.append({
            "pair_id": pair["pair_id"],
            "family": pair["family"],
            "condition": pair["condition"],
            "model": pair["model"],
            "seed": pair["seed"],
            "block_length_batches": block_length,
            "accuracy_delta_pp": (
                entry["accuracy_delta_pp"]
            ),
            "ci_lower_pp": lower,
            "ci_upper_pp": upper,
            "excludes_zero": (
                lower > 0 or upper < 0
            ),
            "positive_interval": lower > 0,
            "negative_interval": upper < 0,
        })

        entry[
            f"block{block_length}_ci_lower_pp"
        ] = lower

        entry[
            f"block{block_length}_ci_upper_pp"
        ] = upper

    statistical_rows.append(entry)

    if (
        pair_number % 10 == 0
        or pair_number == NUM_PAIRS
    ):

        print(
            f"[{pair_number:3d}/{NUM_PAIRS}] "
            "Statistical analysis completed",
            flush=True,
        )

# ==========================================
# Multiple testing correction
# ==========================================

adjusted_p = holm_adjust([
    row["mcnemar_p_raw"]
    for row in statistical_rows
])

for row, adjusted in zip(
    statistical_rows,
    adjusted_p,
):

    row["mcnemar_p_holm"] = float(
        adjusted
    )

    row["significant_raw"] = (
        row["mcnemar_p_raw"] < ALPHA
    )

    row["significant_holm"] = (
        row["mcnemar_p_holm"] < ALPHA
    )

# ==========================================
# Common-image grouped bootstrap
# ==========================================

# All conditions reuse the same image IDs.
# Average correctness differences over runs
# for each original image, then resample
# common image IDs jointly across runs.
#
# This preserves cross-condition alignment
# but does not reproduce online adaptation
# or account for unknown patient clusters.

difference_matrix = np.asarray(
    indexed_differences,
    dtype=np.int8,
)

require(
    difference_matrix.shape
    == (NUM_PAIRS, NUM_SAMPLES),
    "Unexpected difference matrix shape",
)

def common_image_bootstrap(
    matrix,
):

    per_image_effect = matrix.mean(
        axis=0
    )

    n_images = len(per_image_effect)

    estimates = []

    # Small chunks to limit RAM usage.

    chunk_size = 128

    for start in range(
        0,
        BOOTSTRAP_REPEATS,
        chunk_size,
    ):

        count = min(
            chunk_size,
            BOOTSTRAP_REPEATS - start,
        )

        indices = rng.integers(
            0,
            n_images,
            size=(count, n_images),
        )

        values = (
            per_image_effect[indices]
            .mean(axis=1)
            * 100
        )

        estimates.extend(
            values.tolist()
        )

    lower, upper = confidence_interval(
        estimates
    )

    return {
        "mean_accuracy_delta_pp": float(
            per_image_effect.mean() * 100
        ),
        "conditional_ci_lower_pp": lower,
        "conditional_ci_upper_pp": upper,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "images": n_images,
        "paired_comparisons": len(matrix),
    }


print("\n" + "=" * 75)
print("COMMON-IMAGE BOOTSTRAP")
print("=" * 75)

overall_bootstrap = (
    common_image_bootstrap(
        difference_matrix
    )
)

group_bootstrap = {}

for family in [
    "severity",
    "corruption",
]:

    mask = np.array([
        row["family"] == family
        for row in statistical_rows
    ])

    group_bootstrap[family] = (
        common_image_bootstrap(
            difference_matrix[mask]
        )
    )


# ==========================================
# Joint block sensitivity by seed
# ==========================================

# Comparisons within each seed share an
# identical image stream.
#
# Resample identical batch positions jointly
# across all conditions within that seed.
#
# Do not treat the three seeds as independent
# training experiments.

joint_block_rows = []

print("\n" + "=" * 75)
print("JOINT BLOCK SENSITIVITY BY SEED")
print("=" * 75)

for seed in SEEDS:

    seed_pairs = [
        row for row in pairs
        if row["seed"] == seed
    ]

    require(
        len(seed_pairs) == 39,
        "Unexpected seed comparison count",
    )

    matrix = []

    batch_sizes = None

    for pair in seed_pairs:

        rows = verify_pair_blocks(
            pair
        )

        matrix.append([
            row["net_correct"]
            for row in rows
        ])

        sizes = np.array([
            row["samples"]
            for row in rows
        ])

        if batch_sizes is None:

            batch_sizes = sizes

        else:

            require(
                np.array_equal(
                    batch_sizes,
                    sizes,
                ),
                "Seed batch sizes differ",
            )

    matrix = np.asarray(matrix)

    combined_net = matrix.sum(axis=0)

    combined_sizes = (
        batch_sizes
        * len(seed_pairs)
    )

    point_estimate = float(
        combined_net.sum()
        / combined_sizes.sum()
        * 100
    )

    for block_length in BLOCK_LENGTHS:

        lower, upper = (
            circular_block_bootstrap(
                combined_net,
                combined_sizes,
                block_length,
            )
        )

        joint_block_rows.append({
            "seed": seed,
            "paired_comparisons": len(seed_pairs),
            "block_length_batches": block_length,
            "accuracy_delta_pp": point_estimate,
            "conditional_ci_lower_pp": lower,
            "conditional_ci_upper_pp": upper,
            "excludes_zero": (
                lower > 0 or upper < 0
            ),
        })

        print(
            f"Seed={seed:4d} | "
            f"Block={block_length:2d} | "
            f"Delta={point_estimate:+.4f} pp | "
            f"CI=[{lower:+.4f}, "
            f"{upper:+.4f}]",
            flush=True,
        )

# ==========================================
# Descriptive effect sizes
# ==========================================

def summarize(rows):

    require(
        len(rows) > 0,
        "Empty effect-size group",
    )

    delta = np.array([
        row["accuracy_delta_pp"]
        for row in rows
    ])

    f1_delta = np.array([
        row["macro_f1_delta_pp"]
        for row in rows
    ])

    corrected = sum(
        row["corrected"]
        for row in rows
    )

    harmed = sum(
        row["harmed"]
        for row in rows
    )

    return {
        "paired_comparisons": len(rows),
        "mean_accuracy_delta_pp": float(
            delta.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            f1_delta.mean()
        ),
        "median_accuracy_delta_pp": float(
            np.median(delta)
        ),
        "descriptive_std_pp": float(
            delta.std(ddof=1)
            if len(delta) > 1
            else 0.0
        ),
        "minimum_accuracy_delta_pp": float(
            delta.min()
        ),
        "maximum_accuracy_delta_pp": float(
            delta.max()
        ),
        "corrected_prediction_events": corrected,
        "harmed_prediction_events": harmed,
        "net_correct_prediction_events": (
            corrected - harmed
        ),
        "mean_net_correct_per_run": (
            (corrected - harmed)
            / len(rows)
        ),
        "corrected_harmed_ratio": (
            corrected / harmed
            if harmed > 0
            else None
        ),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "mcnemar_significant_raw": int(
            sum(
                row["significant_raw"]
                for row in rows
            )
        ),
        "mcnemar_significant_holm": int(
            sum(
                row["significant_holm"]
                for row in rows
            )
        ),
    }


overall = summarize(
    statistical_rows
)

by_seed = {}

for seed in SEEDS:

    by_seed[str(seed)] = summarize([
        row for row in statistical_rows
        if row["seed"] == seed
    ])

by_model = {}

for model in MODELS:

    by_model[model] = summarize([
        row for row in statistical_rows
        if row["model"] == model
    ])

by_condition = {}

condition_keys = sorted({
    (
        row["family"],
        row["condition"],
    )

    for row in statistical_rows
})

for family, condition in condition_keys:

    name = f"{family}/{condition}"

    by_condition[name] = summarize([
        row for row in statistical_rows
        if row["family"] == family
        and row["condition"] == condition
    ])

# ==========================================
# Leave-one-group-out sensitivity
# ==========================================

sensitivity_rows = []

for field, values in [
    ("model", MODELS),
    ("seed", SEEDS),
    ("family", ["severity", "corruption"]),
]:

    for value in values:

        remaining = [
            row for row in statistical_rows
            if row[field] != value
        ]

        sensitivity_rows.append({
            "excluded_group_type": field,
            "excluded_group": str(value),
            **summarize(remaining),
        })

for family, condition in condition_keys:

    remaining = [
        row for row in statistical_rows
        if not (
            row["family"] == family
            and row["condition"] == condition
        )
    ]

    sensitivity_rows.append({
        "excluded_group_type": "condition",
        "excluded_group": (
            f"{family}/{condition}"
        ),
        **summarize(remaining),
    })

# ==========================================
# Summarize block-level intervals
# ==========================================

block_summary = {}

for block_length in BLOCK_LENGTHS:

    rows = [
        row for row in block_ci_rows
        if row["block_length_batches"]
        == block_length
    ]

    require(
        len(rows) == NUM_PAIRS,
        "Incomplete block analysis",
    )

    block_summary[str(block_length)] = {
        "paired_comparisons": len(rows),
        "positive_intervals": int(
            sum(
                row["positive_interval"]
                for row in rows
            )
        ),
        "negative_intervals": int(
            sum(
                row["negative_interval"]
                for row in rows
            )
        ),
        "intervals_including_zero": int(
            sum(
                not row["excludes_zero"]
                for row in rows
            )
        ),
        "mean_interval_width_pp": float(
            np.mean([
                row["ci_upper_pp"]
                - row["ci_lower_pp"]
                for row in rows
            ])
        ),
    }

# ==========================================
# Cross-check aggregate results
# ==========================================

require(
    abs(
        overall["mean_accuracy_delta_pp"]
        - report["overall"][
            "mean_accuracy_delta_pp"
        ]
    ) < 1e-10,
    "Overall Accuracy delta mismatch",
)

require(
    abs(
        overall["mean_macro_f1_delta_pp"]
        - report["overall"][
            "mean_macro_f1_delta_pp"
        ]
    ) < 1e-10,
    "Overall Macro-F1 delta mismatch",
)

for field in [
    "wins",
    "ties",
    "losses",
]:

    require(
        overall[field]
        == report["overall"][field],
        f"Overall outcome mismatch: {field}",
    )

# ==========================================
# Print final statistical results
# ==========================================

print("\n" + "=" * 75)
print("EXACT MCNEMAR TEST SUMMARY")
print("=" * 75)

print(
    "Total tests:",
    len(statistical_rows),
)

print(
    "Significant before Holm correction:",
    overall["mcnemar_significant_raw"],
    "/",
    NUM_PAIRS,
)

print(
    "Significant after Holm correction:",
    overall["mcnemar_significant_holm"],
    "/",
    NUM_PAIRS,
)

print(
    "Minimum raw p-value:",
    f"{min(r['mcnemar_p_raw'] for r in statistical_rows):.8f}"
)

print(
    "Minimum Holm p-value:",
    f"{min(r['mcnemar_p_holm'] for r in statistical_rows):.8f}"
)

print(
    "NOTE: McNemar results are exploratory "
    "because predictions may be dependent."
)

print("\n" + "=" * 75)
print("TOP 10 LOWEST RAW P-VALUES")
print("=" * 75)

sorted_results = sorted(
    statistical_rows,
    key=lambda row: row["mcnemar_p_raw"],
)

for row in sorted_results[:10]:

    print(
        f"{row['model']:9s} | "
        f"{row['condition']:23s} | "
        f"Seed={row['seed']:4d} | "
        f"b={row['harmed']:3d} | "
        f"c={row['corrected']:3d} | "
        f"Raw={row['mcnemar_p_raw']:.6f} | "
        f"Holm={row['mcnemar_p_holm']:.6f}"
    )

print("\n" + "=" * 75)
print("BLOCK BOOTSTRAP SENSITIVITY")
print("=" * 75)

for block_length, row in (
    block_summary.items()
):

    print(
        f"Block={block_length:>2s} batches | "
        f"Positive={row['positive_intervals']:3d} | "
        f"Negative={row['negative_intervals']:3d} | "
        f"Includes zero="
        f"{row['intervals_including_zero']:3d} | "
        f"Mean width="
        f"{row['mean_interval_width_pp']:.4f} pp"
    )

print("\n" + "=" * 75)
print("OVERALL CONDITIONAL IMAGE BOOTSTRAP")
print("=" * 75)

print(
    "Mean Accuracy Delta:",
    f"{overall_bootstrap['mean_accuracy_delta_pp']:+.4f} pp"
)

print(
    "Conditional 95% CI:",
    f"[{overall_bootstrap['conditional_ci_lower_pp']:+.4f}, "
    f"{overall_bootstrap['conditional_ci_upper_pp']:+.4f}] pp"
)

for family, row in group_bootstrap.items():

    print(
        f"{family:12s} | "
        f"Delta={row['mean_accuracy_delta_pp']:+.4f} | "
        f"CI=[{row['conditional_ci_lower_pp']:+.4f}, "
        f"{row['conditional_ci_upper_pp']:+.4f}] pp"
    )

print(
    "NOTE: These intervals condition on "
    "the saved adaptation trajectories."
)

print("\n" + "=" * 75)
print("SEED SENSITIVITY")
print("=" * 75)

for seed, row in by_seed.items():

    print(
        f"Seed={seed:4s} | "
        f"Acc Delta={row['mean_accuracy_delta_pp']:+.4f} pp | "
        f"F1 Delta={row['mean_macro_f1_delta_pp']:+.4f} pp"
    )

sensitivity_values = [
    row["mean_accuracy_delta_pp"]
    for row in sensitivity_rows
]

print("\n" + "=" * 75)
print("LEAVE-ONE-GROUP-OUT SENSITIVITY")
print("=" * 75)

print(
    "Minimum remaining mean effect:",
    f"{min(sensitivity_values):+.4f} pp"
)

print(
    "Maximum remaining mean effect:",
    f"{max(sensitivity_values):+.4f} pp"
)

print(
    "All remaining means positive:",
    all(
        value > 0
        for value in sensitivity_values
    ),
)

print("\n" + "=" * 75)
print("OVERALL EFFECT SIZE")
print("=" * 75)

print(
    "Accuracy Delta:",
    f"{overall['mean_accuracy_delta_pp']:+.4f} pp"
)

print(
    "Macro-F1 Delta:",
    f"{overall['mean_macro_f1_delta_pp']:+.4f} pp"
)

print(
    "Mean net correct per run:",
    f"{overall['mean_net_correct_per_run']:+.4f}"
)

print(
    "Corrected prediction events:",
    overall["corrected_prediction_events"],
)

print(
    "Harmed prediction events:",
    overall["harmed_prediction_events"],
)

print(
    "Corrected/Harmed ratio:",
    f"{overall['corrected_harmed_ratio']:.4f}"
    if overall["corrected_harmed_ratio"]
    is not None
    else "Undefined"
)

print(
    "Wins/Ties/Losses:",
    f"{overall['wins']}/"
    f"{overall['ties']}/"
    f"{overall['losses']}"
)

# ==========================================
# Save outputs
# ==========================================

save_csv(
    OUTPUT_DIR / "statistical_tests.csv",
    statistical_rows,
)

save_csv(
    OUTPUT_DIR / "block_bootstrap.csv",
    block_ci_rows,
)

save_csv(
    OUTPUT_DIR / "seed_joint_block_bootstrap.csv",
    joint_block_rows,
)

save_csv(
    OUTPUT_DIR / "sensitivity_analysis.csv",
    sensitivity_rows,
)

statistical_report = {
    "status": "completed",
    "analysis_type": "exploratory_conditional",
    "alpha": ALPHA,
    "confidence_level": CONFIDENCE_LEVEL,
    "bootstrap_repeats": BOOTSTRAP_REPEATS,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "block_lengths_batches": BLOCK_LENGTHS,
    "paired_comparisons": NUM_PAIRS,
    "input_hashes": input_hashes,
    "analysis_script_sha256": sha256_file(
        Path(__file__)
    ),
    "overall": overall,
    "by_seed": by_seed,
    "by_model": by_model,
    "by_condition": by_condition,
    "conditional_image_bootstrap": (
        overall_bootstrap
    ),
    "conditional_bootstrap_by_family": (
        group_bootstrap
    ),
    "block_bootstrap_summary": block_summary,
    "joint_block_sensitivity_by_seed": (
        joint_block_rows
    ),
    "leave_one_group_out": {
        "minimum_mean_accuracy_delta_pp": float(
            min(sensitivity_values)
        ),
        "maximum_mean_accuracy_delta_pp": float(
            max(sensitivity_values)
        ),
        "all_positive": all(
            value > 0
            for value in sensitivity_values
        ),
    },
    "multiple_testing": {
        "method": "Holm",
        "tests": NUM_PAIRS,
        "significant_raw": (
            overall["mcnemar_significant_raw"]
        ),
        "significant_adjusted": (
            overall["mcnemar_significant_holm"]
        ),
        "minimum_raw_p": float(
            min(
                row["mcnemar_p_raw"]
                for row in statistical_rows
            )
        ),
        "minimum_adjusted_p": float(
            min(
                row["mcnemar_p_holm"]
                for row in statistical_rows
            )
        ),
    },
    "limitations": [
        "McNemar assumes independent paired units.",
        "Online TTA introduces dependence within and across batches.",
        "Holm correction does not repair invalid marginal p-values.",
        "Bootstrap intervals condition on saved predictions.",
        "Resampling does not rerun the adaptation trajectory.",
        "Common-image bootstrap preserves cross-condition image alignment, but not patient or slide clustering.",
        "Circular block bootstrap is a sensitivity analysis, not a guaranteed valid confidence interval.",
        "Three seeds reuse the same test images and are not independent training repetitions.",
        "No pooled McNemar test across all prediction events was performed.",
        "Macro-F1 differences are descriptive; confidence intervals here concern Accuracy.",
        "The analysis plan was finalized after observing descriptive results.",
        "The test set was previously accessed.",
        "Synthetic corruptions do not establish real-world cross-domain validity.",
        "Statistical significance does not imply clinical relevance.",
    ],
}

save_json(
    OUTPUT_DIR / "statistical_report.json",
    statistical_report,
)

print("\n" + "=" * 75)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 75)

print(
    "Analyzed paired comparisons:",
    len(statistical_rows),
    "/",
    NUM_PAIRS,
)

print(
    "Block bootstrap records:",
    len(block_ci_rows),
)

print(
    "Joint seed block records:",
    len(joint_block_rows),
)

print(
    "Sensitivity analysis records:",
    len(sensitivity_rows),
)

print("Saved outputs:", OUTPUT_DIR)

print(
    "Final status: ANALYSIS COMPLETED"
)

print("=" * 75)
