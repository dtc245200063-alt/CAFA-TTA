
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = ROOT / "experiments"
OUTPUT = EXPERIMENTS / "summary"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment file: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. Read source-only and BN-only test results
source = read_json(
    EXPERIMENTS / "source_only_test/metrics.json"
)

bn_only = read_json(
    EXPERIMENTS / "bn_only_test/metrics.json"
)

# 2. Read TENT and CAFA-TTA v2 robustness results
final_test = read_json(
    EXPERIMENTS / "final_test/metrics.json"
)

assert source["split"] == "test"
assert bn_only["split"] == "test"
assert final_test["split"] == "test"

# Make sure all methods used the same domain shift.
assert source["shift"] == bn_only["shift"]
assert source["shift"] == final_test["shift"]

rows = []


def add_single_result(name, data):
    rows.append({
        "method": name,
        "accuracy_mean": data["accuracy"],
        "accuracy_std": None,
        "macro_f1_mean": data["macro_f1"],
        "macro_f1_std": None,
        "runs": 1,
    })


add_single_result("Source-only", source)
add_single_result("BN-only", bn_only)


def add_multi_seed_result(name, method_key):
    method_results = [
        item for item in final_test["results"]
        if item["method"] == method_key
    ]

    if len(method_results) != 3:
        raise ValueError(
            f"Expected 3 results for {method_key}, "
            f"got {len(method_results)}"
        )

    accuracy_values = [
        item["accuracy"]
        for item in method_results
    ]

    f1_values = [
        item["macro_f1"]
        for item in method_results
    ]

    rows.append({
        "method": name,
        "accuracy_mean": statistics.mean(
            accuracy_values
        ),
        "accuracy_std": statistics.stdev(
            accuracy_values
        ),
        "macro_f1_mean": statistics.mean(
            f1_values
        ),
        "macro_f1_std": statistics.stdev(
            f1_values
        ),
        "runs": len(method_results),
    })


add_multi_seed_result("TENT", "tent")
add_multi_seed_result("CAFA-TTA v2", "cafa_v2")


# 3. Save CSV
csv_path = OUTPUT / "final_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "runs",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# 4. Create Markdown report
def format_metric(mean, std):
    percentage = mean * 100

    if std is None:
        return f"{percentage:.2f}%"

    return (
        f"{percentage:.2f}% "
        f"± {std * 100:.2f}"
    )


lines = [
    "# CAFA-TTA — Final Test Results",
    "",
    "**Dataset:** PathMNIST",
    "",
    "**Test samples:** 7,180",
    "",
    "**Domain shift:** Synthetic color transformation",
    "",
    f"**Shift configuration:** `{source['shift']}`",
    "",
    "| Method | Accuracy | Macro-F1 | Runs |",
    "|---|---:|---:|---:|",
]

for row in rows:
    acc = format_metric(
        row["accuracy_mean"],
        row["accuracy_std"]
    )
    f1 = format_metric(
        row["macro_f1_mean"],
        row["macro_f1_std"]
    )

    lines.append(
        f"| {row['method']} | {acc} | "
        f"{f1} | {row['runs']} |"
    )

lines.extend([
    "",
    "## Experimental Notes",
    "",
    "- Source-only and BN-only use one fixed data order.",
    "- TENT and CAFA-TTA v2 report mean ± sample standard "
    "deviation across three shuffled batch orders.",
    "- Target labels are used only for evaluation, "
    "not for test-time parameter updates.",
    "- Hyperparameters were selected before final test evaluation.",
    "- The domain shift is simulated, not a real cross-hospital shift.",
    "- CAFA-TTA v2 has not demonstrated a robust "
    "advantage over TENT.",
])

md_path = OUTPUT / "final_results.md"
md_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


# 5. Display results
print("=" * 60)
print("CAFA-TTA | FINAL RESULTS SUMMARY")
print("=" * 60)

for row in rows:
    print(
        f"{row['method']:15s} | "
        f"Accuracy: "
        f"{format_metric(row['accuracy_mean'], row['accuracy_std'])}"
        f" | Macro-F1: "
        f"{format_metric(row['macro_f1_mean'], row['macro_f1_std'])}"
    )

print("\nSaved:")
print(csv_path)
print(md_path)
print("\nRESULT SUMMARY COMPLETED")
