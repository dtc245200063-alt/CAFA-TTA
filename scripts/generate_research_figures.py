
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

except ImportError:

    raise SystemExit(
        "Matplotlib is required. "
        "Run: python -m pip install matplotlib"
    )

# ==========================================
# CAFA-TTA | RESEARCH FIGURES AND TABLES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

STAT_DIR = (
    ROOT / "experiments/statistical_analysis"
)

CLASS_DIR = STAT_DIR / "per_class"

OUTPUT_DIR = (
    ROOT / "experiments/research_outputs"
)

FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for directory in [
    OUTPUT_DIR,
    FIG_DIR,
    TABLE_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
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

ALL_METHODS = [
    "source_only",
    "bn_only",
    "tent",
    "cafa_v1",
    "alignment_only",
    "cafa_v2",
    "cafa_v3",
]

SEVERITIES = [
    "original",
    "light",
    "medium",
    "strong",
]

CORRUPTIONS = [
    "gaussian_noise",
    "gaussian_blur",
    "hue_shift",
]

METHOD_NAMES = {
    "source_only": "Source-only",
    "bn_only": "BN-only",
    "tent": "TENT",
    "cafa_v1": "CAFA v1",
    "alignment_only": "Alignment-only",
    "cafa_v2": "CAFA v2",
    "cafa_v3": "CAFA v3",
}

COLORS = {
    "source_only": "#94A3B8",
    "bn_only": "#D97706",
    "tent": "#2563EB",
    "cafa_v1": "#A78BFA",
    "alignment_only": "#64748B",
    "cafa_v2": "#9333EA",
    "cafa_v3": "#16A34A",
}

POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
NEUTRAL = "#94A3B8"

CLASS_SHORT_NAMES = [
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

# ==========================================
# Publication-style configuration
# ==========================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.20,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ==========================================
# Utility functions
# ==========================================

input_hashes = {}
generated_figures = []
generated_tables = []
warnings = []


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


def register_input(path):

    relative = path.relative_to(ROOT).as_posix()

    input_hashes[relative] = sha256_file(path)


def read_json(path):

    register_input(path)

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:

        return json.load(f)


def read_csv(path):

    register_input(path)

    with open(
        path,
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(csv.DictReader(f))


def save_csv(name, rows):

    require(
        len(rows) > 0,
        f"Empty table: {name}",
    )

    path = TABLE_DIR / name

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

    generated_tables.append(name)

    print("TABLE:", name, flush=True)


def save_figure(fig, name):

    fig.savefig(
        FIG_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig.savefig(
        FIG_DIR / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    generated_figures.append(name)

    print("FIGURE:", name, flush=True)


def percent_axis(ax):

    ax.yaxis.set_major_formatter(
        PercentFormatter(100)
    )


def mean_std(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(values.mean()),
        float(values.std(ddof=1))
        if len(values) > 1
        else 0.0,
    )


# ==========================================
# Load verified research data
# ==========================================

print("=" * 75)
print("CAFA-TTA | RESEARCH FIGURES AND TABLES")
print("Matplotlib:", matplotlib.__version__)
print("NumPy:", np.__version__)
print("Device: CPU")
print("=" * 75)

integrity = read_json(
    STAT_DIR / "integrity_report.json"
)

statistics = read_json(
    STAT_DIR / "statistical_report.json"
)

per_class = read_json(
    CLASS_DIR / "per_class_report.json"
)

confusions = read_json(
    CLASS_DIR / "confusion_matrices.json"
)

prediction_rows = read_csv(
    STAT_DIR / "prediction_integrity.csv"
)

paired_rows = read_csv(
    STAT_DIR / "paired_predictions.csv"
)

negative_rows = read_csv(
    CLASS_DIR / "negative_adaptation.csv"
)

statistical_tests = read_csv(
    STAT_DIR / "statistical_tests.csv"
)

require(
    integrity["status"] == "passed",
    "Integrity audit incomplete",
)

require(
    statistics["status"] == "completed",
    "Statistical analysis incomplete",
)

require(
    per_class["status"] == "completed",
    "Per-class analysis incomplete",
)

require(
    len(prediction_rows) == 423,
    "Expected 423 prediction rows",
)

require(
    len(paired_rows) == 117,
    "Expected 117 paired rows",
)

require(
    len(negative_rows) == 117,
    "Expected 117 adaptation rows",
)

require(
    len(statistical_tests) == 117,
    "Expected 117 statistical tests",
)

for report, field in [
    (integrity, "input_metrics_hashes"),
    (statistics, "input_hashes"),
    (per_class, "input_hashes"),
]:

    for relative_path, expected_hash in (
        report[field].items()
    ):

        path = ROOT / relative_path

        require(
            sha256_file(path)
            == expected_hash.upper(),
            f"Input changed: {relative_path}",
        )

        register_input(path)

require(
    abs(
        integrity["overall"][
            "mean_accuracy_delta_pp"
        ]
        - per_class["accuracy_delta_pp"]
    ) < 1e-10,
    "Accuracy reconstruction mismatch",
)

require(
    abs(
        integrity["overall"][
            "mean_macro_f1_delta_pp"
        ]
        - per_class["macro_f1_delta_pp"]
    ) < 1e-10,
    "Macro-F1 reconstruction mismatch",
)

print("Input integrity: PASS")

# ==========================================
# Group verified run results
# ==========================================

run_groups = defaultdict(list)

for row in prediction_rows:

    key = (
        row["family"],
        row["condition"],
        row["model"],
        row["method"],
    )

    run_groups[key].append(row)


def metric_summary(
    family,
    condition,
    model,
    method,
    metric,
):

    rows = run_groups[
        (
            family,
            condition,
            model,
            method,
        )
    ]

    require(
        len(rows) > 0,
        "Missing metric group: "
        f"{family}/{condition}/{model}/{method}",
    )

    return mean_std([
        float(row[metric]) * 100
        for row in rows
    ])


# ==========================================
# TABLE 1: Complete aggregated test results
# ==========================================

all_test_results = []

for key, rows in sorted(
    run_groups.items()
):

    family, condition, model, method = key

    accuracy, accuracy_std = mean_std([
        float(row["accuracy"]) * 100
        for row in rows
    ])

    macro_f1, macro_f1_std = mean_std([
        float(row["macro_f1"]) * 100
        for row in rows
    ])

    all_test_results.append({
        "family": family,
        "condition": condition,
        "model": model,
        "method": method,
        "runs": len(rows),
        "accuracy_mean_percent": accuracy,
        "accuracy_std_pp": accuracy_std,
        "macro_f1_mean_percent": macro_f1,
        "macro_f1_std_pp": macro_f1_std,
        "std_available": len(rows) > 1,
    })

save_csv(
    "table01_all_test_results.csv",
    all_test_results,
)

# ==========================================
# FIGURE 1: Strong-shift method comparison
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    layout="constrained",
)

x = np.arange(len(MODELS))

width = 0.11

for ax, metric, ylabel in [
    (
        axes[0],
        "accuracy",
        "Accuracy (%)",
    ),
    (
        axes[1],
        "macro_f1",
        "Macro-F1 (%)",
    ),
]:

    for i, method in enumerate(
        ALL_METHODS
    ):

        values = [
            metric_summary(
                "severity",
                "strong",
                model,
                method,
                metric,
            )
            for model in MODELS
        ]

        means = [
            value[0]
            for value in values
        ]

        stds = [
            value[1]
            for value in values
        ]

        positions = x + (
            i - 3
        ) * width

        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            color=COLORS[method],
            label=METHOD_NAMES[method],
        )

    ax.set_xticks(
        x,
        [m.title() for m in MODELS],
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)

    percent_axis(ax)

axes[0].legend(
    fontsize=8,
    ncol=2,
    loc="upper left",
)

fig.suptitle(
    "PathMNIST Test: Strong Color Shift"
)

save_figure(
    fig,
    "fig01_final_test_comparison",
)

# ==========================================
# TABLE 2: Strong-shift results
# ==========================================

save_csv(
    "table02_strong_shift.csv",
    [
        row for row in all_test_results
        if row["family"] == "severity"
        and row["condition"] == "strong"
    ],
)

# ==========================================
# FIGURE 2: Multi-severity curves
# ==========================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8),
    layout="constrained",
)

for column, model in enumerate(
    MODELS
):

    for row_index, metric in enumerate([
        "accuracy",
        "macro_f1",
    ]):

        ax = axes[
            row_index,
            column,
        ]

        for method in METHODS:

            values = [
                metric_summary(
                    "severity",
                    severity,
                    model,
                    method,
                    metric,
                )
                for severity in SEVERITIES
            ]

            means = [
                v[0] for v in values
            ]

            stds = [
                v[1] for v in values
            ]

            ax.errorbar(
                range(4),
                means,
                yerr=stds,
                color=COLORS[method],
                marker="o",
                markersize=4,
                linewidth=1.7,
                capsize=2,
                label=METHOD_NAMES[method],
            )

        ax.set_xticks(
            range(4),
            [
                s.title()
                for s in SEVERITIES
            ],
            fontsize=9,
        )

        ax.set_ylim(0, 100)

        ax.set_ylabel(
            "Accuracy (%)"
            if metric == "accuracy"
            else "Macro-F1 (%)"
        )

        percent_axis(ax)

        if row_index == 0:

            ax.set_title(
                model.title()
            )

axes[0, 0].legend(
    fontsize=8
)

fig.suptitle(
    "Performance Across Color-Shift Severity"
)

save_figure(
    fig,
    "fig02_shift_severity",
)

# ==========================================
# TABLE 3: Paired condition summary
# ==========================================

condition_rows = []

for name, summary in (
    integrity["by_condition"].items()
):

    condition_rows.append({
        "condition": name,
        "paired_comparisons": (
            summary["paired_comparisons"]
        ),
        "accuracy_delta_pp": (
            summary["mean_accuracy_delta_pp"]
        ),
        "macro_f1_delta_pp": (
            summary["mean_macro_f1_delta_pp"]
        ),
        "wins": summary["wins"],
        "ties": summary["ties"],
        "losses": summary["losses"],
    })

save_csv(
    "table03_paired_conditions.csv",
    condition_rows,
)

# ==========================================
# FIGURE 3: Multi-corruption improvement
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    layout="constrained",
)

x = np.arange(3)
width = 0.24

for j, severity in enumerate([
    "light",
    "medium",
    "strong",
]):

    summaries = [
        integrity["by_condition"][
            f"corruption/{corruption}/{severity}"
        ]
        for corruption in CORRUPTIONS
    ]

    for ax, field, ylabel in [
        (
            axes[0],
            "mean_accuracy_delta_pp",
            "Accuracy improvement (pp)",
        ),
        (
            axes[1],
            "mean_macro_f1_delta_pp",
            "Macro-F1 improvement (pp)",
        ),
    ]:

        ax.bar(
            x + (j - 1) * width,
            [
                s[field]
                for s in summaries
            ],
            width=width,
            label=severity.title(),
            color=[
                "#93C5FD",
                "#3B82F6",
                "#1E3A8A",
            ][j],
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8,
        )

        ax.set_xticks(
            x,
            [
                "Gaussian\nNoise",
                "Gaussian\nBlur",
                "Hue\nShift",
            ],
        )

        ax.set_ylabel(ylabel)

axes[0].legend()

fig.suptitle(
    "CAFA v3 versus TENT: Synthetic Corruptions"
)

save_figure(
    fig,
    "fig03_multi_corruption",
)

# ==========================================
# FIGURE 4: Validation ablation study
# ==========================================

# Read full-precision validation ablation results.
# Verify aggregate metrics against individual runs.


ablation_path = (
    ROOT
    / "experiments/cafa_v3_ablation/metrics.json"
)

ablation_data = read_json(ablation_path)

require(
    ablation_data["split"] == "val",
    "Invalid ablation split",
)

require(
    ablation_data["completed_runs"] == 36,
    "Incomplete ablation results",
)

require(
    len(ablation_data["results"]) == 36,
    "Unexpected ablation record count",
)

require(
    ablation_data["seeds"] == [42, 123, 2026],
    "Invalid ablation seeds",
)

require(
    ablation_data["batch_size"] == 64,
    "Invalid ablation batch size",
)

ablation_keys = [
    "entropy_baseline",
    "soft_entropy_only",
    "agreement_align_only",
    "full_cafa_v3",
]

ablation_summary = ablation_data["summary"]

ablation_values = {
    model: [
        float(
            ablation_summary[model][variant][
                "accuracy_difference_pp"
            ]
        )
        for variant in ablation_keys
    ]
    for model in MODELS
}

for model in MODELS:

    baseline = ablation_summary[model][
        "entropy_baseline"
    ]

    for variant in ablation_keys:

        item = ablation_summary[model][variant]

        raw_runs = [
            row for row in ablation_data["results"]
            if row["model"] == model
            and row["variant"] == variant
        ]

        require(
            len(raw_runs) == 3,
            f"Incomplete ablation group: {model}/{variant}",
        )

        require(
            {row["seed"] for row in raw_runs}
            == {42, 123, 2026},
            "Ablation seed mismatch",
        )

        for metric in ["accuracy", "macro_f1"]:

            values = [
                float(row[metric])
                for row in raw_runs
            ]

            require(
                abs(
                    np.mean(values)
                    - item[f"{metric}_mean"]
                ) < 1e-10,
                f"Ablation mean mismatch: {model}/{variant}/{metric}",
            )

            require(
                abs(
                    np.std(values, ddof=1)
                    - item[f"{metric}_std"]
                ) < 1e-10,
                f"Ablation std mismatch: {model}/{variant}/{metric}",
            )

            require(
                abs(
                    (
                        item[f"{metric}_mean"]
                        - baseline[f"{metric}_mean"]
                    ) * 100
                    - item[f"{metric}_difference_pp"]
                ) < 1e-10,
                f"Ablation delta mismatch: {model}/{variant}/{metric}",
            )

print(
    "Ablation source verification: PASS",
    flush=True,
)

ablation_names = [
    "Entropy baseline",
    "Soft weighting",
    "Alignment",
    "Full CAFA v3",
]

ablation_rows = []

for model, values in (
    ablation_values.items()
):

    for variant_index, (variant, value) in enumerate(
        zip(ablation_names, values)
    ):

        variant_key = ablation_keys[variant_index]

        item = ablation_summary[model][variant_key]

        ablation_rows.append({
            "model": model,
            "variant": variant,
            "accuracy_delta_vs_entropy_baseline_pp": (
                value
            ),
            "split": "val",
            "variant_key": variant_key,
            "accuracy_mean_percent": (
                item["accuracy_mean"] * 100
            ),
            "accuracy_std_pp": (
                item["accuracy_std"] * 100
            ),
            "macro_f1_mean_percent": (
                item["macro_f1_mean"] * 100
            ),
            "macro_f1_std_pp": (
                item["macro_f1_std"] * 100
            ),
            "macro_f1_delta_pp": (
                item["macro_f1_difference_pp"]
            ),
            "provenance": (
                ablation_path.relative_to(ROOT).as_posix()
            ),
        })

save_csv(
    "table04_ablation_verified.csv",
    ablation_rows,
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

x = np.arange(3)
width = 0.19

for i, variant in enumerate(
    ablation_names
):

    ax.bar(
        x + (i - 1.5) * width,
        [
            ablation_values[m][i]
            for m in MODELS
        ],
        width=width,
        label=variant,
        color=[
            "#94A3B8",
            "#D97706",
            "#2563EB",
            "#16A34A",
        ][i],
    )

ax.set_xticks(
    x,
    [m.title() for m in MODELS],
)

ax.set_ylabel(
    "Accuracy gain vs entropy baseline (pp)"
)

ax.set_title(
    "CAFA v3 Ablation Study\n"
    "Validation, strong color shift"
)

ax.legend(
    ncol=2,
    fontsize=9,
)

fig.text(
    0.5,
    0.01,
    "Full-precision results from the original "
    "validation ablation JSON.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.04, 1, 1]
)

save_figure(
    fig,
    "fig04_ablation_study",
)

print(
    "Figure 4 source: verified ablation JSON",
    flush=True,
)

# ==========================================
# FIGURE 5: Accuracy and computational cost
# ==========================================

# Read full-precision GPU timing results.
# Validation strong shift benchmark.
# Units: milliseconds per batch of 64.
# No training or benchmark is rerun here.


benchmark_path = (
    ROOT
    / "experiments/cafa_v3_benchmark/metrics.json"
)

benchmark_data = read_json(benchmark_path)

require(
    benchmark_data["split"] == "val",
    "Invalid benchmark split",
)

require(
    benchmark_data["completed_runs"] == 36,
    "Incomplete benchmark results",
)

require(
    len(benchmark_data["results"]) == 36,
    "Unexpected benchmark record count",
)

require(
    benchmark_data["batch_size"] == 64,
    "Invalid benchmark batch size",
)

require(
    benchmark_data["repeats"] == 3,
    "Invalid benchmark repeats",
)

require(
    benchmark_data["warmup_batches"] == 10,
    "Invalid warmup batches",
)

require(
    benchmark_data["measured_batches"] == 50,
    "Invalid measured batches",
)

require(
    benchmark_data["data_loading_included"] is False,
    "Unexpected benchmark data-loading protocol",
)

require(
    benchmark_data["host_to_device_transfer_included"]
    is False,
    "Unexpected benchmark transfer protocol",
)

benchmark_summary = benchmark_data["summary"]

benchmark_ms = {
    model: {
        method: float(
            benchmark_summary[model][method][
                "total_mean_ms"
            ]["mean"]
        )
        for method in METHODS
    }
    for model in MODELS
}

benchmark_fields = [
    "inference_mean_ms",
    "adaptation_mean_ms",
    "total_mean_ms",
    "throughput_images_per_second",
    "peak_gpu_memory_mib",
    "trainable_parameters",
    "prototype_memory_bytes",
]

for model in MODELS:

    for method in METHODS:

        raw_runs = [
            row for row in benchmark_data["results"]
            if row["model"] == model
            and row["method"] == method
        ]

        require(
            len(raw_runs) == 3,
            f"Incomplete benchmark group: {model}/{method}",
        )

        require(
            len({row["repeat"] for row in raw_runs}) == 3,
            "Duplicate benchmark repeats",
        )

        for field in benchmark_fields:

            values = [
                float(row[field])
                for row in raw_runs
            ]

            summary = benchmark_summary[
                model
            ][method][field]

            require(
                np.isclose(
                    np.mean(values),
                    summary["mean"],
                    rtol=1e-10,
                    atol=1e-10,
                ),
                f"Benchmark mean mismatch: {model}/{method}/{field}",
            )

            require(
                np.isclose(
                    np.std(values, ddof=1),
                    summary["std"],
                    rtol=1e-10,
                    atol=1e-10,
                ),
                f"Benchmark std mismatch: {model}/{method}/{field}",
            )

print(
    "Benchmark source verification: PASS",
    flush=True,
)

benchmark_rows = []

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.5),
    sharex=True,
    layout="constrained",
)

for ax, model in zip(
    axes,
    MODELS,
):

    for method in METHODS:

        latency = benchmark_ms[
            model
        ][method]

        acc, std = metric_summary(
            "severity",
            "strong",
            model,
            method,
            "accuracy",
        )

        overhead = (
            latency
            / benchmark_ms[model]["tent"]
            - 1
        ) * 100

        benchmark_rows.append({
            "model": model,
            "method": method,
            "test_strong_accuracy_percent": acc,
            "test_strong_accuracy_std_pp": std,
            "validation_benchmark_ms_per_batch": (
                latency
            ),
            "latency_overhead_vs_tent_percent": (
                overhead
            ),
            "batch_size": 64,
            "latency_std_ms": (
                benchmark_summary[model][method][
                    "total_mean_ms"
                ]["std"]
            ),
            "inference_mean_ms": (
                benchmark_summary[model][method][
                    "inference_mean_ms"
                ]["mean"]
            ),
            "adaptation_mean_ms": (
                benchmark_summary[model][method][
                    "adaptation_mean_ms"
                ]["mean"]
            ),
            "throughput_images_per_second": (
                benchmark_summary[model][method][
                    "throughput_images_per_second"
                ]["mean"]
            ),
            "throughput_std": (
                benchmark_summary[model][method][
                    "throughput_images_per_second"
                ]["std"]
            ),
            "peak_gpu_memory_mib": (
                benchmark_summary[model][method][
                    "peak_gpu_memory_mib"
                ]["mean"]
            ),
            "peak_gpu_memory_std_mib": (
                benchmark_summary[model][method][
                    "peak_gpu_memory_mib"
                ]["std"]
            ),
            "trainable_parameters": (
                benchmark_summary[model][method][
                    "trainable_parameters"
                ]["mean"]
            ),
            "prototype_memory_bytes": (
                benchmark_summary[model][method][
                    "prototype_memory_bytes"
                ]["mean"]
            ),
            "timing_provenance": (
                benchmark_path.relative_to(ROOT).as_posix()
            ),
        })

        ax.scatter(
            latency,
            acc,
            color=COLORS[method],
            s=85,
            label=METHOD_NAMES[method],
            zorder=3,
        )

    ax.set_title(
        model.title()
    )

    ax.set_xlabel(
        "Latency (ms/batch)"
    )

    ax.set_ylabel(
        "Test accuracy (%)"
    )

    ax.set_ylim(0, 100)

    percent_axis(ax)

axes[0].legend(
    fontsize=8
)

fig.suptitle(
    "Accuracy and GPU Computational Cost"
)

fig.text(
    0.5,
    -0.03,
    "Accuracy: strong-shift test. "
    "Latency: separate validation benchmark. "
    "RTX 3050 Laptop GPU, batch size 64.",
    ha="center",
    fontsize=9,
)

save_figure(
    fig,
    "fig05_accuracy_cost_tradeoff",
)

save_csv(
    "table05_benchmark_tradeoff.csv",
    benchmark_rows,
)

print(
    "Figure 5 source: verified benchmark JSON",
    flush=True,
)

# ==========================================
# FIGURE 6: Statistical forest plot
# ==========================================

forest_data = [
    (
        "Overall",
        statistics[
            "conditional_image_bootstrap"
        ],
    ),
    (
        "Color severity",
        statistics[
            "conditional_bootstrap_by_family"
        ]["severity"],
    ),
    (
        "Corruptions",
        statistics[
            "conditional_bootstrap_by_family"
        ]["corruption"],
    ),
]

forest_rows = []

fig, ax = plt.subplots(
    figsize=(8, 4)
)

for index, (name, row) in enumerate(
    forest_data
):

    estimate = row[
        "mean_accuracy_delta_pp"
    ]

    lower = row[
        "conditional_ci_lower_pp"
    ]

    upper = row[
        "conditional_ci_upper_pp"
    ]

    ax.errorbar(
        estimate,
        index,
        xerr=[
            [estimate - lower],
            [upper - estimate],
        ],
        fmt="s",
        color=POSITIVE,
        markersize=7,
        capsize=5,
        linewidth=2,
    )

    ax.text(
        upper + 0.004,
        index,
        f"{estimate:+.4f} "
        f"[{lower:+.4f}, {upper:+.4f}]",
        va="center",
        fontsize=9,
    )

    forest_rows.append({
        "group": name,
        "accuracy_delta_pp": estimate,
        "conditional_ci_lower_pp": lower,
        "conditional_ci_upper_pp": upper,
        "interval_type": (
            "Common-image conditional bootstrap"
        ),
    })

ax.axvline(
    0,
    linestyle="--",
    color="black",
    linewidth=1,
)

ax.set_yticks(
    range(len(forest_data)),
    [name for name, _ in forest_data],
)

ax.invert_yaxis()

max_upper = max(
    row["conditional_ci_upper_pp"]
    for row in forest_rows
)

ax.set_xlim(
    -0.01,
    max_upper + 0.09,
)

ax.set_xlabel(
    "Accuracy improvement (pp)"
)

ax.set_title(
    "Conditional 95% Bootstrap Intervals"
)

ax.grid(
    axis="y",
    visible=False,
)

fig.text(
    0.5,
    0.01,
    "Conditional on saved predictions; "
    "not independent population-level inference.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.06, 1, 1]
)

save_figure(
    fig,
    "fig06_conditional_forest_plot",
)

save_csv(
    "table06_conditional_intervals.csv",
    forest_rows,
)

# ==========================================
# FIGURE 7: Paired prediction outcomes
# ==========================================

overall = integrity["overall"]

corrected = overall[
    "total_corrected_prediction_events"
]

harmed = overall[
    "total_harmed_prediction_events"
]

net = corrected - harmed

require(
    net == overall[
        "total_net_correct_events"
    ],
    "Paired outcome mismatch",
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5),
    layout="constrained",
)

axes[0].pie(
    [corrected, harmed],
    labels=[
        "Corrected",
        "Harmed",
    ],
    colors=[
        POSITIVE,
        NEGATIVE,
    ],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={
        "width": 0.40,
        "edgecolor": "white",
    },
)

axes[0].text(
    0,
    0,
    f"{corrected / harmed:.2f}x",
    ha="center",
    va="center",
    fontsize=19,
    fontweight="bold",
)

axes[0].set_title(
    "Discordant Correctness Events"
)

outcomes = [
    overall["wins"],
    overall["ties"],
    overall["losses"],
]

bars = axes[1].bar(
    ["Wins", "Ties", "Losses"],
    outcomes,
    color=[
        POSITIVE,
        NEUTRAL,
        NEGATIVE,
    ],
)

axes[1].bar_label(
    bars,
    padding=3,
)

axes[1].set_ylim(
    0,
    max(outcomes) * 1.2,
)

axes[1].set_ylabel(
    "Paired comparisons"
)

axes[1].set_title(
    "Accuracy Outcomes"
)

fig.suptitle(
    "Paired Prediction Analysis"
)

save_figure(
    fig,
    "fig07_paired_prediction_outcomes",
)

# ==========================================
# FIGURE 8: Per-class performance
# ==========================================

class_rows = sorted(
    per_class["overall_classes"],
    key=lambda row: row["class_id"],
)

require(
    len(class_rows) == 9,
    "Invalid class count",
)

save_csv(
    "table07_per_class_performance.csv",
    class_rows,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6),
    layout="constrained",
)

names = [
    row["class_name"]
    for row in class_rows
]

y = np.arange(9)

for ax, field, ylabel in [
    (
        axes[0],
        "recall_delta_pp",
        "Recall improvement (pp)",
    ),
    (
        axes[1],
        "f1_delta_pp",
        "F1 improvement (pp)",
    ),
]:

    values = [
        row[field]
        for row in class_rows
    ]

    ax.barh(
        y,
        values,
        color=[
            POSITIVE
            if value >= 0
            else NEGATIVE

            for value in values
        ],
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_yticks(
        y,
        names,
        fontsize=9,
    )

    ax.invert_yaxis()
    ax.set_xlabel(ylabel)

fig.suptitle(
    "Class-wise CAFA v3 Improvements over TENT"
)

save_figure(
    fig,
    "fig08_per_class_performance",
)

# ==========================================
# FIGURE 9: Minority and majority classes
# ==========================================

minority_data = per_class[
    "minority_class_analysis"
]

minority = minority_data["minority"]
majority = minority_data["majority"]

group_rows = []

for name, row in [
    ("minority", minority),
    ("majority", majority),
]:

    group_rows.append({
        "group": name,
        "test_images": row["unique_test_images"],
        "macro_recall_delta_pp": (
            row["macro_recall_delta_pp"]
        ),
        "macro_f1_delta_pp": (
            row["macro_f1_delta_pp"]
        ),
        "weighted_recall_delta_pp": (
            row["support_weighted_recall_delta_pp"]
        ),
        "corrected_events": row["corrected_events"],
        "harmed_events": row["harmed_events"],
        "net_correct_events": row["net_correct_events"],
    })

save_csv(
    "table08_class_balance.csv",
    group_rows,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
    layout="constrained",
)

x = np.arange(3)

for i, (name, row) in enumerate([
    ("Minority", minority),
    ("Majority", majority),
]):

    values = [
        row["macro_recall_delta_pp"],
        row["macro_f1_delta_pp"],
        row["support_weighted_recall_delta_pp"],
    ]

    axes[0].bar(
        x + (i - 0.5) * 0.32,
        values,
        width=0.32,
        color=[
            "#64748B",
            "#16A34A",
        ][i],
        label=name,
    )

axes[0].set_xticks(
    x,
    [
        "Macro\nRecall",
        "Macro\nF1",
        "Weighted\nRecall",
    ],
)

axes[0].axhline(
    0,
    color="black",
    linewidth=0.8,
)

axes[0].set_ylabel(
    "Improvement (pp)"
)

axes[0].legend()

net_values = [
    minority["net_correct_events"],
    majority["net_correct_events"],
]

bars = axes[1].bar(
    ["Minority", "Majority"],
    net_values,
    color=[
        "#64748B",
        POSITIVE,
    ],
)

axes[1].bar_label(
    bars,
    padding=3,
)

axes[1].set_ylim(
    0,
    max(net_values) * 1.2,
)

axes[1].set_ylabel(
    "Net correct prediction events"
)

fig.suptitle(
    "Performance by Test Class Support"
)

save_figure(
    fig,
    "fig09_minority_majority",
)

# ==========================================
# FIGURES 10-12: Confusion matrices
# ==========================================

confusion_overall = confusions[
    "overall"
]

tent_cm = np.asarray(
    confusion_overall[
        "tent_row_normalized"
    ],
    dtype=float,
) * 100

cafa_cm = np.asarray(
    confusion_overall[
        "cafa_row_normalized"
    ],
    dtype=float,
) * 100

diff_cm = np.asarray(
    confusion_overall[
        "difference_percentage_points"
    ],
    dtype=float,
)

require(
    tent_cm.shape == (9, 9),
    "Invalid TENT confusion shape",
)

require(
    cafa_cm.shape == (9, 9),
    "Invalid CAFA confusion shape",
)

require(
    np.allclose(
        tent_cm.sum(axis=1),
        100,
    ),
    "TENT normalization mismatch",
)

require(
    np.allclose(
        cafa_cm.sum(axis=1),
        100,
    ),
    "CAFA normalization mismatch",
)

require(
    np.allclose(
        cafa_cm - tent_cm,
        diff_cm,
    ),
    "Difference matrix mismatch",
)


def plot_confusion(
    matrix,
    title,
    name,
    difference=False,
):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    if difference:

        maximum = max(
            np.abs(matrix).max(),
            1e-8,
        )

        image = ax.imshow(
            matrix,
            cmap="RdBu",
            vmin=-maximum,
            vmax=maximum,
        )

    else:

        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=100,
        )

    ax.set_xticks(
        range(9),
        CLASS_SHORT_NAMES,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(9),
        CLASS_SHORT_NAMES,
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(title)

    ax.grid(False)

    for i in range(9):

        for j in range(9):

            value = matrix[i, j]

            if difference:

                text = f"{value:+.3f}"

                white_text = (
                    abs(value)
                    > maximum * 0.65
                )

            else:

                text = f"{value:.1f}"

                white_text = value > 55

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if white_text
                    else "black"
                ),
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Difference (pp)"
        if difference
        else "Row-normalized percentage"
    )

    fig.text(
        0.5,
        0.01,
        "Aggregated across 117 paired runs; "
        "images are reused across runs.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    save_figure(
        fig,
        name,
    )


plot_confusion(
    tent_cm,
    "TENT: Row-normalized Confusion Matrix",
    "fig10_tent_confusion_matrix",
)

plot_confusion(
    cafa_cm,
    "CAFA v3: Row-normalized Confusion Matrix",
    "fig11_cafa_confusion_matrix",
)

plot_confusion(
    diff_cm,
    "Confusion Difference: CAFA v3 - TENT",
    "fig12_difference_confusion_matrix",
    difference=True,
)

# ==========================================
# FIGURE 13: Negative adaptation
# ==========================================

negative_summary = per_class[
    "negative_adaptation"
]

negative_table = []

for model in MODELS:

    for method in [
        "bn_only",
        "tent",
        "cafa_v3",
    ]:

        row = negative_summary[
            "by_model"
        ][model][method]

        negative_table.append({
            "model": model,
            "method": method,
            **row,
        })

save_csv(
    "table09_negative_adaptation.csv",
    negative_table,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    layout="constrained",
)

x = np.arange(3)
width = 0.24

for i, method in enumerate([
    "bn_only",
    "tent",
    "cafa_v3",
]):

    rows = [
        negative_summary[
            "by_model"
        ][model][method]

        for model in MODELS
    ]

    axes[0].bar(
        x + (i - 1) * width,
        [
            row["negative_accuracy_conditions"]
            for row in rows
        ],
        width=width,
        color=COLORS[method],
        label=METHOD_NAMES[method],
    )

    axes[1].bar(
        x + (i - 1) * width,
        [
            row[
                "mean_accuracy_delta_vs_source_pp"
            ]
            for row in rows
        ],
        width=width,
        color=COLORS[method],
        label=METHOD_NAMES[method],
    )

for ax in axes:

    ax.set_xticks(
        x,
        [
            model.title()
            for model in MODELS
        ],
    )

axes[0].set_ylim(
    0,
    13,
)

axes[0].set_ylabel(
    "Negative adaptation conditions / 13"
)

axes[0].set_title(
    "Frequency of Accuracy Degradation"
)

axes[1].set_ylabel(
    "Accuracy change vs source-only (pp)"
)

axes[1].set_title(
    "Mean Adaptation Effect"
)

axes[1].axhline(
    0,
    color="black",
    linewidth=0.8,
)

axes[0].legend()

fig.suptitle(
    "Negative Adaptation Analysis"
)

save_figure(
    fig,
    "fig13_negative_adaptation",
)

# ==========================================
# TABLE 10: Main statistical results
# ==========================================

stat_overall = statistics["overall"]

testing = statistics[
    "multiple_testing"
]

conditional = statistics[
    "conditional_image_bootstrap"
]

sensitivity = statistics[
    "leave_one_group_out"
]

main_results = [
    {
        "metric": "Verified prediction files",
        "value": 423,
        "unit": "files",
    },
    {
        "metric": "Paired comparisons",
        "value": 117,
        "unit": "comparisons",
    },
    {
        "metric": "Accuracy improvement",
        "value": stat_overall[
            "mean_accuracy_delta_pp"
        ],
        "unit": "pp",
    },
    {
        "metric": "Macro-F1 improvement",
        "value": stat_overall[
            "mean_macro_f1_delta_pp"
        ],
        "unit": "pp",
    },
    {
        "metric": "Conditional CI lower",
        "value": conditional[
            "conditional_ci_lower_pp"
        ],
        "unit": "pp",
    },
    {
        "metric": "Conditional CI upper",
        "value": conditional[
            "conditional_ci_upper_pp"
        ],
        "unit": "pp",
    },
    {
        "metric": "Wins",
        "value": overall["wins"],
        "unit": "comparisons",
    },
    {
        "metric": "Ties",
        "value": overall["ties"],
        "unit": "comparisons",
    },
    {
        "metric": "Losses",
        "value": overall["losses"],
        "unit": "comparisons",
    },
    {
        "metric": "Corrected events",
        "value": corrected,
        "unit": "prediction events",
    },
    {
        "metric": "Harmed events",
        "value": harmed,
        "unit": "prediction events",
    },
    {
        "metric": "Net correct events",
        "value": net,
        "unit": "prediction events",
    },
    {
        "metric": "McNemar raw significant",
        "value": testing[
            "significant_raw"
        ],
        "unit": "comparisons",
    },
    {
        "metric": "McNemar Holm significant",
        "value": testing[
            "significant_adjusted"
        ],
        "unit": "comparisons",
    },
    {
        "metric": "Minimum Holm p-value",
        "value": testing[
            "minimum_adjusted_p"
        ],
        "unit": "p-value",
    },
]

save_csv(
    "table10_main_statistical_results.csv",
    main_results,
)

# ==========================================
# TABLE 11: Block bootstrap sensitivity
# ==========================================

block_rows = []

for length, row in (
    statistics[
        "block_bootstrap_summary"
    ].items()
):

    block_rows.append({
        "block_length_batches": int(length),
        **row,
    })

save_csv(
    "table11_block_bootstrap.csv",
    block_rows,
)

# ==========================================
# TABLE 12: Source model and seed comparison
# ==========================================

grouped_results = []

for group_type, group_data in [
    ("model", integrity["by_model"]),
    ("seed", integrity["by_seed"]),
    (
        "corruption",
        integrity["by_corruption"],
    ),
]:

    for group_name, row in group_data.items():

        grouped_results.append({
            "group_type": group_type,
            "group_name": group_name,
            "paired_comparisons": (
                row["paired_comparisons"]
            ),
            "accuracy_delta_pp": (
                row["mean_accuracy_delta_pp"]
            ),
            "macro_f1_delta_pp": (
                row["mean_macro_f1_delta_pp"]
            ),
            "wins": row["wins"],
            "ties": row["ties"],
            "losses": row["losses"],
        })

save_csv(
    "table12_grouped_comparisons.csv",
    grouped_results,
)

# ==========================================
# Research findings
# ==========================================

acc_delta = stat_overall[
    "mean_accuracy_delta_pp"
]

f1_delta = stat_overall[
    "mean_macro_f1_delta_pp"
]

ci_lower = conditional[
    "conditional_ci_lower_pp"
]

ci_upper = conditional[
    "conditional_ci_upper_pp"
]

wins = overall["wins"]
ties = overall["ties"]
losses = overall["losses"]

majority_share = (
    majority["net_correct_events"]
    / net * 100
)

noise = integrity[
    "by_corruption"
]["gaussian_noise"]

hue = integrity[
    "by_corruption"
]["hue_shift"]

cafa_negative = negative_summary[
    "overall"
]["cafa_v3"]

best_recall = max(
    class_rows,
    key=lambda row: row[
        "recall_delta_pp"
    ],
)

worst_recall = min(
    class_rows,
    key=lambda row: row[
        "recall_delta_pp"
    ],
)

latency_overheads = [
    (
        benchmark_ms[model]["cafa_v3"]
        / benchmark_ms[model]["tent"]
        - 1
    ) * 100

    for model in MODELS
]

findings = f"""# CAFA-TTA: Research Findings

## Experimental setting

- Dataset: PathMNIST, 28 x 28 RGB images.
- Test images: 7,180.
- Source backbones: three ResNet-18 checkpoints.
- Models: Enhanced, Robust, and Balanced.
- Final method: CAFA-TTA v3.
- Batch size: 64.
- Learning rate: 0.0001.
- Entropy floor: 0.2.
- Alignment coefficient: 0.05.
- Stream-order seeds: 42, 123, and 2026.
- Online prediction before parameter adaptation.
- Current-batch statistics are used by target BN.
- Target labels are not used for adaptation.

## 1. Data integrity

All 423 saved prediction files passed SHA-256
verification and metric consistency checks.

The audit removed 36 duplicate strong-shift records,
leaving 117 unique TENT versus CAFA v3 comparisons.

## 2. Overall performance

CAFA v3 achieved a mean Accuracy improvement of
{acc_delta:+.4f} percentage points and a mean
Macro-F1 improvement of {f1_delta:+.4f} percentage
points over TENT.

Accuracy wins, ties, and losses were
{wins}, {ties}, and {losses}, respectively.

These differences are descriptive averages across
the evaluated models, conditions, and stream orders.

## 3. Paired prediction outcomes

CAFA corrected {corrected} prediction events that
TENT classified incorrectly, while introducing
{harmed} incorrect prediction events.

The net difference was {net:+d} correct prediction
events, corresponding to an average of
{net / 117:+.4f} additional correct predictions
per run of 7,180 images.

The corrected-to-harmed ratio was
{corrected / harmed:.4f}.

These events are not unique images or patients.

## 4. Statistical analysis

The common-image conditional bootstrap produced a
95% interval of [{ci_lower:+.4f}, {ci_upper:+.4f}]
percentage points for the mean Accuracy difference.

Of 117 two-sided exact McNemar tests,
{testing['significant_raw']} met the nominal 0.05
threshold before correction and
{testing['significant_adjusted']} after Holm correction.

The leave-one-group-out mean Accuracy differences
ranged from
{sensitivity['minimum_mean_accuracy_delta_pp']:+.4f}
to
{sensitivity['maximum_mean_accuracy_delta_pp']:+.4f}
percentage points.

The statistical findings are exploratory because
the same images are reused and online adaptation
creates dependence within and across batches.

## 5. Robustness to synthetic corruption

Gaussian noise produced a mean Accuracy improvement
of {noise['mean_accuracy_delta_pp']:+.4f} percentage
points, with {noise['wins']} wins across
{noise['paired_comparisons']} comparisons.

Hue shift produced a smaller average improvement of
{hue['mean_accuracy_delta_pp']:+.4f} percentage
points.

The results suggest that the benefit of CAFA is
dependent on the type of distribution shift.

## 6. Class-wise performance

The largest mean Recall improvement occurred in
{best_recall['class_name']}
({best_recall['recall_delta_pp']:+.4f} pp).

The largest Recall degradation occurred in
{worst_recall['class_name']}
({worst_recall['recall_delta_pp']:+.4f} pp).

The majority class group accounted for
{majority_share:.2f}% of the net improvement.

Minority and majority groups were defined by test
class support relative to mean class support,
not by clinical disease prevalence.

## 7. Negative adaptation

CAFA v3 reduced Accuracy relative to source-only
in {cafa_negative['negative_accuracy_conditions']}
of {cafa_negative['conditions']} model-condition
combinations.

CAFA therefore does not eliminate negative
adaptation, even when it improves upon TENT.

## 8. Computational trade-off

The verified validation GPU benchmark showed
approximately {min(latency_overheads):.1f}% to
{max(latency_overheads):.1f}% higher mean latency
for CAFA v3 relative to TENT.

The timing was measured separately on validation
batches and excludes preprocessing and data transfer.

The observed Accuracy improvement is small relative
to the additional computational cost.

Timing values are read directly from the original
benchmark JSON and checked against the individual
benchmark repetitions.

## 9. Natural cross-center evaluation

PathMNIST training and validation originate from
NCT-CRC-HE-100K, whereas its test split originates
from CRC-VAL-HE-7K, described by MedMNIST as data
from a different clinical center.

The original, uncorrupted test split therefore
provides a natural cross-center evaluation within
the PathMNIST benchmark.

The synthetic perturbation experiments evaluate
additional controlled shifts on the target data.

On the original test split, CAFA v3 achieved a
small average improvement over TENT but remained
below source-only Accuracy for all three source
models.

This does not establish generalization across
a broader collection of independent hospitals
or medical datasets.

Dataset reference: https://medmnist.com/

## 10. Overall conclusion

CAFA-TTA v3 demonstrated small but relatively
consistent empirical improvements over TENT across
the evaluated synthetic distribution shifts.

However, the improvements were uneven across
classes, negative adaptation remained unresolved,
and the statistical evidence does not establish
definitive population-level or clinical superiority.

The method should be interpreted as a modest
extension of entropy-based test-time adaptation,
not as a comprehensive solution to medical
cross-domain classification.

"""

# ==========================================
# Research limitations
# ==========================================

limitations = """# Research Limitations

## 1. Small absolute effect size

The average Accuracy gain over TENT is small.
Practical relevance must be evaluated alongside
computational overhead.

## 2. Negative adaptation

Adaptation can reduce performance compared with
source-only inference when the source model is
already robust.

## 3. Uneven class-wise benefits

The gains are concentrated primarily in classes
with higher test support. Several lower-support
classes experience Recall degradation.

## 4. Statistical dependence

The same test images are reused across source
models, synthetic shifts, and stream-order seeds.

Online parameter adaptation creates dependence
between predictions within and across batches.

McNemar p-values and bootstrap intervals are
therefore exploratory.

## 5. Conditional bootstrap intervals

The bootstrap analysis resamples saved prediction
outcomes. It does not rerun adaptation under a
newly sampled sequence of images.

## 6. Multiple comparisons

Holm correction reduces the multiple-testing
problem but does not resolve dependence or
violations of marginal test assumptions.

## 7. Previously accessed test set

The PathMNIST test set was accessed during earlier
experiments. The final CAFA v3 configuration was
frozen before the reported final evaluation, but
the test set cannot be described as pristine.

## 8. Limited source model diversity

The three source models share the ResNet-18
architecture and related training history.

Stream-order seeds are not independent model
training repetitions.

## 9. Scope of domain-shift evaluation

PathMNIST includes source and test data originating
from different clinical centers.

The original test split provides a natural
cross-center benchmark.

Brightness, contrast, saturation, Gaussian noise,
Gaussian blur, and hue changes provide additional
controlled synthetic perturbations.

Evaluation on one source-target benchmark does
not establish broad generalization across hospitals,
scanners, staining protocols, patient populations,
or medical imaging modalities.

## 10. Limited baseline diversity

The principal comparisons involve source-only,
BN-only, TENT, and internal CAFA variants.

Broader comparisons against other established
test-time adaptation algorithms remain desirable.

## 11. Patient and slide-level uncertainty

The analysis does not account for patient-level
or histology-slide-level clustering.

Patch classification results are not evidence of
patient-level diagnostic accuracy.

## 12. Computational benchmark scope

GPU timings exclude preprocessing and CPU-GPU
data transfer and use one hardware configuration.

They do not represent end-to-end clinical latency.

## 13. Hyperparameter and analysis selection

The final method was selected using validation
experiments. The statistical analysis plan was
completed after descriptive results were observed.

Independent confirmatory evaluation is required.

## 14. Figure provenance

The ablation and benchmark figures use the original
full-precision experimental JSON files.

Their source hashes are recorded in the research
manifest, and aggregate values are checked against
individual experimental runs.

The accuracy-cost figure combines test Accuracy
with latency measured in a separate validation
benchmark. These measurements should not be
interpreted as a single joint evaluation.

"""

future_work = """# Future Research Directions

1. Evaluate on independently obtained medical
   datasets with matched label definitions.

2. Investigate source-to-target hospital,
   scanner, and staining distribution shifts.

3. Develop adaptive gating to detect when
   test-time adaptation should be disabled.

4. Investigate class-aware weighting and
   minority-class preservation.

5. Reduce the extra forward-pass and
   feature-alignment computational overhead.

6. Compare with additional established
   test-time adaptation baselines.

7. Evaluate independently trained source
   checkpoints and multiple architectures.

8. Conduct patient-aware or slide-aware
   statistical evaluation where metadata permits.

9. Pre-register the evaluation protocol for
   an independent confirmatory test.

These directions are future work and are not
claimed as completed experiments.
"""

for filename, content in [
    ("research_findings.md", findings),
    ("research_limitations.md", limitations),
    ("future_work.md", future_work),
]:

    path = OUTPUT_DIR / filename

    path.write_text(
        content,
        encoding="utf-8",
    )

    print(
        "DOCUMENT:",
        filename,
    )

# ==========================================
# Figure captions and interpretation guide
# ==========================================

figure_captions = """# Research Figure Captions

## Figure 1

Accuracy and Macro-F1 under strong synthetic
color shift. Error bars show sample standard
deviations over three stream orders, not
confidence intervals.

## Figure 2

Performance under original, light, medium,
and strong color-shift conditions. Source-only
is evaluated once for non-strong conditions.
Zero error bars on single-run results do not
represent measured uncertainty.

## Figure 3

Mean paired CAFA v3 improvements relative to
TENT under Gaussian noise, Gaussian blur,
and hue shift.

## Figure 4

Validation ablation results relative to the
entropy baseline. Alignment denotes the
alignment-enabled entropy baseline, not an
alignment-only objective. Values are read from
the original full-precision validation ablation
JSON and verified against individual runs.

## Figure 5

Strong-shift test Accuracy compared with
separately measured validation GPU latency.
Timing values are read directly from the verified
benchmark JSON.

## Figure 6

Conditional 95% common-image bootstrap
intervals. Estimates are conditional on
saved adaptation trajectories.

## Figure 7

Discordant correctness outcomes and
Accuracy wins, ties, and losses across
117 paired comparisons.

## Figure 8

Mean per-class Recall and F1 differences.

## Figure 9

Performance differences between class groups
defined by sample support relative to the mean
test class support.

## Figures 10 and 11

Confusion matrices normalized by true class.
Counts are aggregated across repeated
prediction events.

## Figure 12

CAFA minus TENT row-normalized confusion
matrix in percentage points. Positive
off-diagonal entries indicate increased
misclassification frequency.

## Figure 13

Negative adaptation relative to source-only
inference, averaged over stream orders.

## Class abbreviations

- ADI: adipose
- BACK: background
- DEB: debris
- LYM: lymphocytes
- MUC: mucus
- MUS: smooth muscle
- NORM: normal colon mucosa
- STR: cancer-associated stroma
- TUM: colorectal adenocarcinoma epithelium
"""

(
    OUTPUT_DIR / "figure_captions.md"
).write_text(
    figure_captions,
    encoding="utf-8",
)

# ==========================================
# Verify generated outputs
# ==========================================

require(
    len(generated_figures) == 13,
    "Expected 13 research figures",
)

require(
    len(generated_tables) == 12,
    "Expected 12 research tables",
)

output_hashes = {}

for name in generated_figures:

    for extension in [
        ".png",
        ".pdf",
    ]:

        path = FIG_DIR / (
            name + extension
        )

        require(
            path.exists()
            and path.stat().st_size > 0,
            f"Missing figure: {path}",
        )

        output_hashes[
            path.relative_to(
                OUTPUT_DIR
            ).as_posix()
        ] = sha256_file(path)

for name in generated_tables:

    path = TABLE_DIR / name

    require(
        path.exists()
        and path.stat().st_size > 0,
        f"Missing table: {path}",
    )

    output_hashes[
        path.relative_to(
            OUTPUT_DIR
        ).as_posix()
    ] = sha256_file(path)

documents = [
    "research_findings.md",
    "research_limitations.md",
    "future_work.md",
    "figure_captions.md",
]

for name in documents:

    path = OUTPUT_DIR / name

    require(
        path.exists()
        and path.stat().st_size > 0,
        f"Missing document: {path}",
    )

    output_hashes[name] = sha256_file(path)

# ==========================================
# Save reproducibility manifest
# ==========================================

manifest = {
    "status": "generated_from_verified_sources",
    "project": "CAFA-TTA",
    "dataset": "PathMNIST",
    "split": "test_and_validation",
    "figures": generated_figures,
    "figure_count": len(generated_figures),
    "figure_formats": [
        "PNG 300 DPI",
        "PDF vector",
    ],
    "tables": generated_tables,
    "table_count": len(generated_tables),
    "documents": documents,
    "document_count": len(documents),
    "input_hashes": input_hashes,
    "output_hashes": output_hashes,
    "script_sha256": sha256_file(
        Path(__file__)
    ),
    "matplotlib_version": (
        matplotlib.__version__
    ),
    "numpy_version": np.__version__,
    "warnings": warnings,
}

manifest_path = (
    OUTPUT_DIR / "research_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

# ==========================================
# Final summary
# ==========================================

print("\n" + "=" * 75)
print("RESEARCH OUTPUT SUMMARY")
print("=" * 75)

print(
    "Research figures:",
    len(generated_figures),
    "/ 13",
)

print(
    "PNG files:",
    len(generated_figures),
)

print(
    "PDF files:",
    len(generated_figures),
)

print(
    "Research tables:",
    len(generated_tables),
    "/ 12",
)

print(
    "Markdown documents:",
    len(documents),
)

print(
    "Reproducibility manifest:",
    manifest_path.name,
)

print(
    "Verified generated outputs:",
    len(output_hashes),
)

print(
    "Accuracy Delta:",
    f"{acc_delta:+.4f} pp",
)

print(
    "Macro-F1 Delta:",
    f"{f1_delta:+.4f} pp",
)

print(
    "Conditional 95% CI:",
    f"[{ci_lower:+.4f}, {ci_upper:+.4f}] pp",
)

print(
    "Wins/Ties/Losses:",
    f"{wins}/{ties}/{losses}",
)

print(
    "Net correct events:",
    net,
)

print("\nPROVENANCE NOTES")

for index, warning in enumerate(
    warnings,
    start=1,
):

    print(
        f"{index}. {warning}"
    )

print("\nSaved outputs:", OUTPUT_DIR)

print(
    "Final status: "
    "RESEARCH OUTPUTS GENERATED"
)

print("=" * 75)
