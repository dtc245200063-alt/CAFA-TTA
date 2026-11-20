
import sys
import ast
import json
import copy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# ==========================================
# CAFA-TTA | FINAL TEST MULTI-SEVERITY
# ==========================================

BASE_SCRIPT = (
    ROOT / "scripts/evaluate_cafa_final_test.py"
)

REFERENCE_PATH = (
    ROOT
    / "experiments"
    / "cafa_final_test_strong"
    / "metrics.json"
)

# ==========================================
# Reuse existing evaluation functions
# without executing the 63 strong-shift runs
# ==========================================

source = BASE_SCRIPT.read_text(
    encoding="utf-8-sig"
)

tree = ast.parse(
    source,
    filename=str(BASE_SCRIPT),
)

required_functions = {
    "sha256_file",
    "verify_hash_manifest",
    "target_transform",
    "set_seed",
    "make_loader",
    "create_backbone",
    "load_prototypes",
    "configure_target_bn",
    "save_json",
    "evaluate",
}

selected_nodes = []

for node in tree.body:

    if isinstance(
        node,
        (ast.Import, ast.ImportFrom),
    ):

        selected_nodes.append(node)

    elif isinstance(node, ast.FunctionDef):

        if node.name in required_functions:

            selected_nodes.append(node)

    elif isinstance(node, ast.ClassDef):

        if node.name == "IndexedPathMNIST":

            selected_nodes.append(node)

found_functions = {
    node.name
    for node in selected_nodes
    if isinstance(node, ast.FunctionDef)
}

assert required_functions == found_functions

assert any(
    isinstance(node, ast.ClassDef)
    and node.name == "IndexedPathMNIST"
    for node in selected_nodes
)

shared_tree = ast.Module(
    body=selected_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(shared_tree)

namespace = {
    "__file__": str(BASE_SCRIPT),
    "__name__": "shared_evaluation",
    "ROOT": ROOT,
}

exec(
    compile(
        shared_tree,
        str(BASE_SCRIPT),
        "exec",
    ),
    namespace,
)

torch = namespace["torch"]
transforms = namespace["transforms"]

sha256_file = namespace["sha256_file"]
save_json = namespace["save_json"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# Frozen configuration
# ==========================================

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
ENTROPY_FLOOR = config["entropy_floor"]
LAMBDA_ALIGN = config["lambda_align"]
MIN_SAMPLES = config["min_samples"]
SEEDS = config["seeds"]

assert BATCH_SIZE == 64
assert LR == 1e-4
assert ENTROPY_FLOOR == 0.2
assert LAMBDA_ALIGN == 0.05
assert MIN_SAMPLES == 2
assert SEEDS == [42, 123, 2026]

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
    "cafa_v3",
]

SEVERITIES = {
    "original": {
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
    },
    "light": {
        "brightness": 0.95,
        "contrast": 1.05,
        "saturation": 0.95,
    },
    "medium": {
        "brightness": 0.85,
        "contrast": 1.15,
        "saturation": 0.85,
    },
    "strong": {
        "brightness": 0.75,
        "contrast": 1.35,
        "saturation": 0.70,
    },
}

# ==========================================
# Configure shared evaluation namespace
# ==========================================

namespace.update({
    "DEVICE": DEVICE,
    "REPRO_DIR": REPRO_DIR,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "ENTROPY_FLOOR": ENTROPY_FLOOR,
    "LAMBDA_ALIGN": LAMBDA_ALIGN,
    "MIN_SAMPLES": MIN_SAMPLES,
    "SEEDS": SEEDS,
    "MODELS": MODELS,
    "METHODS": METHODS,
    "normalize": transforms.Normalize(
        [0.5] * 3,
        [0.5] * 3,
    ),
})

# ==========================================
# Verify checkpoint and algorithm hashes
# ==========================================

verify_hash_manifest = namespace[
    "verify_hash_manifest"
]

checkpoint_hashes = verify_hash_manifest(
    "checkpoint_hashes.csv"
)

algorithm_hashes = verify_hash_manifest(
    "algorithm_hashes.csv"
)

for model_config in MODELS.values():

    for key in ["checkpoint", "prototypes"]:

        assert (
            "checkpoints/" + model_config[key]
            in checkpoint_hashes
        )

assert (
    "core/cafa_soft_adapter.py"
    in algorithm_hashes
)

print("Checkpoint hashes: VERIFIED")
print("Algorithm hashes: VERIFIED")
print("Frozen configuration: VERIFIED")

# ==========================================
# Verify and reuse strong-shift results
# ==========================================

with open(
    REFERENCE_PATH,
    encoding="utf-8",
) as f:

    reference = json.load(f)

assert reference["split"] == "test"

assert reference["protocol"] == (
    "shuffled_online_prequential"
)

assert reference["shift"] == SEVERITIES["strong"]
assert reference["batch_size"] == BATCH_SIZE
assert reference["learning_rate"] == LR
assert reference["entropy_floor"] == ENTROPY_FLOOR
assert reference["lambda_align"] == LAMBDA_ALIGN
assert reference["min_samples"] == MIN_SAMPLES
assert reference["seeds"] == SEEDS
assert reference["models"] == MODELS

assert (
    reference["checkpoint_hashes"]
    == checkpoint_hashes
)

assert (
    reference["algorithm_hashes"]
    == algorithm_hashes
)

assert (
    reference["evaluation_script_hash"]
    == sha256_file(BASE_SCRIPT)
)

assert (
    reference["torch_version"]
    == torch.__version__
)

assert reference["completed_runs"] == 63

strong_results = []

for row in reference["results"]:

    if row["method"] not in METHODS:
        continue

    entry = copy.deepcopy(row)

    entry["severity"] = "strong"
    entry["reused"] = True

    prediction_path = (
        ROOT / entry["predictions_file"]
    )

    assert prediction_path.exists()

    assert (
        sha256_file(prediction_path)
        == entry["predictions_sha256"]
    )

    strong_results.append(entry)

assert len(strong_results) == 36

strong_keys = {
    (r["model"], r["method"], r["seed"])
    for r in strong_results
}

assert strong_keys == {
    (model_name, method, seed)
    for model_name in MODELS
    for method in METHODS
    for seed in SEEDS
}

print("Strong-shift reference: VERIFIED")
print("Reused strong-shift runs:", len(strong_results))

# ==========================================
# Output configuration
# ==========================================

output_dir = (
    ROOT
    / "experiments"
    / "cafa_final_test_severity"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

partial_path = (
    output_dir / "partial_results.json"
)

TOTAL_NEW_RUNS = 90

metadata = {
    "protocol": "shuffled_online_prequential",
    "split": "test",
    "test_previously_accessed": True,
    "configuration_frozen": True,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "entropy_floor": ENTROPY_FLOOR,
    "lambda_align": LAMBDA_ALIGN,
    "min_samples": MIN_SAMPLES,
    "seeds": SEEDS,
    "models": MODELS,
    "methods": METHODS,
    "severities": SEVERITIES,
    "source_only_single_seed": True,
    "checkpoint_hashes": checkpoint_hashes,
    "algorithm_hashes": algorithm_hashes,
    "reference_metrics": (
        REFERENCE_PATH.relative_to(ROOT)
        .as_posix()
    ),
    "reference_metrics_hash": (
        sha256_file(REFERENCE_PATH)
    ),
    "base_script_hash": (
        sha256_file(BASE_SCRIPT)
    ),
    "evaluation_script_hash": (
        sha256_file(Path(__file__))
    ),
    "torch_version": torch.__version__,
    "device": str(DEVICE),
}


def run_key(row):

    return (
        row["severity"],
        row["model"],
        row["method"],
        row["seed"],
    )


# ==========================================
# Resume previous runs safely
# ==========================================

new_results = []
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

    new_results = previous["new_results"]

    for row in new_results:

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
        for row in new_results
    }

    assert len(completed) == len(new_results)

# ==========================================
# Execute new severity evaluations
# ==========================================

print("\n" + "=" * 75)
print("CAFA-TTA | FINAL TEST MULTI-SEVERITY")
print("Device:", DEVICE)
print("Dataset: PathMNIST Test")
print("Batch size:", BATCH_SIZE)
print("Seeds:", SEEDS)
print("New runs:", TOTAL_NEW_RUNS)
print("Reused strong runs:", len(strong_results))
print("Already completed:", len(completed))
print("=" * 75)

evaluate = namespace["evaluate"]

for severity, shift in SEVERITIES.items():

    if severity == "strong":
        continue

    print("\n" + "=" * 75)
    print("SEVERITY:", severity.upper())
    print("Shift:", shift)
    print("=" * 75)

    # Separate prediction folders prevent
    # overwriting files from other severities.

    prediction_dir = (
        output_dir
        / "predictions"
        / severity
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    namespace["shift"] = shift
    namespace["prediction_dir"] = prediction_dir

    dataset = namespace[
        "IndexedPathMNIST"
    ]()

    assert len(dataset) == 7180

    namespace["dataset"] = dataset

    for model_name, model_config in MODELS.items():

        print("\nMODEL:", model_name.upper())

        for seed in SEEDS:

            for method in METHODS:

                # Source-only does not adapt.
                # One seed is sufficient.

                if (
                    method == "source_only"
                    and seed != SEEDS[0]
                ):
                    continue

                key = (
                    severity,
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

                metrics["severity"] = (
                    severity
                )

                metrics["reused"] = False

                new_results.append(metrics)

                completed.add(key)

                line = (
                    f"[{len(new_results):2d}/"
                    f"{TOTAL_NEW_RUNS}] "
                    f"Seed={seed:4d} | "
                    f"{method:12s} | "
                    f"Acc={metrics['accuracy']:.4f} | "
                    f"F1={metrics['macro_f1']:.4f}"
                )

                if "agreement_rate" in metrics:

                    line += (
                        f" | Agreement="
                        f"{metrics['agreement_rate']:.2%}"
                    )

                print(
                    line,
                    flush=True,
                )

                save_json(
                    partial_path,
                    {
                        "metadata": metadata,
                        "completed_new_runs": (
                            len(new_results)
                        ),
                        "total_new_runs": (
                            TOTAL_NEW_RUNS
                        ),
                        "new_results": new_results,
                    },
                )

assert len(new_results) == TOTAL_NEW_RUNS

results = (
    new_results + strong_results
)

assert len(results) == 126

# ==========================================
# Aggregate results
# ==========================================

summary = {}

severity_comparison = {}

all_acc_deltas = []
all_f1_deltas = []

print("\n" + "=" * 75)
print("FINAL TEST MULTI-SEVERITY SUMMARY")
print("=" * 75)

for severity in SEVERITIES:

    summary[severity] = {}

    severity_acc_deltas = []
    severity_f1_deltas = []

    print("\n" + "=" * 75)
    print("SEVERITY:", severity.upper())
    print("=" * 75)

    for model_name in MODELS:

        model_summary = {}

        rows = [
            r for r in results
            if r["severity"] == severity
            and r["model"] == model_name
        ]

        tent_by_seed = {
            r["seed"]: r
            for r in rows
            if r["method"] == "tent"
        }

        assert set(tent_by_seed) == set(SEEDS)

        print("\nMODEL:", model_name.upper())

        for method in METHODS:

            method_rows = [
                r for r in rows
                if r["method"] == method
            ]

            expected_count = (
                1
                if method == "source_only"
                and severity != "strong"
                else 3
            )

            assert len(method_rows) == (
                expected_count
            )

            acc = np.array([
                r["accuracy"]
                for r in method_rows
            ])

            f1 = np.array([
                r["macro_f1"]
                for r in method_rows
            ])

            acc_std = (
                float(acc.std(ddof=1))
                if len(acc) > 1
                else None
            )

            f1_std = (
                float(f1.std(ddof=1))
                if len(f1) > 1
                else None
            )

            entry = {
                "accuracy_mean": float(
                    acc.mean()
                ),
                "accuracy_std": acc_std,
                "macro_f1_mean": float(
                    f1.mean()
                ),
                "macro_f1_std": f1_std,
                "runs": len(method_rows),
            }

            if method == "cafa_v3":

                acc_deltas = np.array([
                    r["accuracy"]
                    - tent_by_seed[
                        r["seed"]
                    ]["accuracy"]
                    for r in method_rows
                ]) * 100

                f1_deltas = np.array([
                    r["macro_f1"]
                    - tent_by_seed[
                        r["seed"]
                    ]["macro_f1"]
                    for r in method_rows
                ]) * 100

                entry.update({
                    "accuracy_delta_vs_tent_pp": (
                        float(acc_deltas.mean())
                    ),
                    "macro_f1_delta_vs_tent_pp": (
                        float(f1_deltas.mean())
                    ),
                    "accuracy_per_seed_pp": (
                        acc_deltas.tolist()
                    ),
                    "macro_f1_per_seed_pp": (
                        f1_deltas.tolist()
                    ),
                    "wins": int(
                        (acc_deltas > 0).sum()
                    ),
                    "ties": int(
                        (acc_deltas == 0).sum()
                    ),
                    "losses": int(
                        (acc_deltas < 0).sum()
                    ),
                })

                severity_acc_deltas.extend(
                    acc_deltas.tolist()
                )

                severity_f1_deltas.extend(
                    f1_deltas.tolist()
                )

            model_summary[method] = entry

            acc_std_text = (
                f"+/- {acc_std:.4f}"
                if acc_std is not None
                else "(single run)"
            )

            f1_std_text = (
                f"+/- {f1_std:.4f}"
                if f1_std is not None
                else "(single run)"
            )

            print(
                f"{method:12s} | "
                f"Acc={acc.mean():.4f} "
                f"{acc_std_text} | "
                f"F1={f1.mean():.4f} "
                f"{f1_std_text}"
            )

            if method == "cafa_v3":

                print(
                    "  Accuracy Delta:",
                    f"{acc_deltas.mean():+.3f} pp"
                )

                print(
                    "  Macro-F1 Delta:",
                    f"{f1_deltas.mean():+.3f} pp"
                )

                print(
                    "  Wins/Ties/Losses:",
                    f"{entry['wins']}/"
                    f"{entry['ties']}/"
                    f"{entry['losses']}"
                )

        summary[severity][model_name] = (
            model_summary
        )

    severity_acc_deltas = np.array(
        severity_acc_deltas
    )

    severity_f1_deltas = np.array(
        severity_f1_deltas
    )

    severity_comparison[severity] = {
        "mean_accuracy_delta_pp": float(
            severity_acc_deltas.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            severity_f1_deltas.mean()
        ),
        "wins": int(
            (severity_acc_deltas > 0).sum()
        ),
        "ties": int(
            (severity_acc_deltas == 0).sum()
        ),
        "losses": int(
            (severity_acc_deltas < 0).sum()
        ),
    }

    all_acc_deltas.extend(
        severity_acc_deltas.tolist()
    )

    all_f1_deltas.extend(
        severity_f1_deltas.tolist()
    )

# ==========================================
# CAFA v3 vs TENT by severity
# ==========================================

print("\n" + "=" * 75)
print("CAFA v3 VS TENT | BY SEVERITY")
print("=" * 75)

for severity, row in (
    severity_comparison.items()
):

    print(
        f"{severity:10s} | "
        f"Acc Delta="
        f"{row['mean_accuracy_delta_pp']:+.3f} pp | "
        f"F1 Delta="
        f"{row['mean_macro_f1_delta_pp']:+.3f} pp | "
        f"W/T/L={row['wins']}/"
        f"{row['ties']}/{row['losses']}"
    )

all_acc_deltas = np.array(
    all_acc_deltas
)

all_f1_deltas = np.array(
    all_f1_deltas
)

assert len(all_acc_deltas) == 36

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
    "paired_comparisons": 36,
}

print("\nOVERALL COMPARISON")

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
# Save final results
# ==========================================

output_path = (
    output_dir / "metrics.json"
)

save_json(
    output_path,
    {
        **metadata,
        "completed_new_runs": (
            len(new_results)
        ),
        "total_new_runs": TOTAL_NEW_RUNS,
        "reused_runs": len(strong_results),
        "total_result_records": len(results),
        "results": results,
        "summary": summary,
        "severity_comparison": (
            severity_comparison
        ),
        "overall": overall,
    },
)

print("\n" + "=" * 75)
print("FINAL TEST MULTI-SEVERITY COMPLETED")
print(
    "New runs:",
    len(new_results),
    "/",
    TOTAL_NEW_RUNS,
)
print(
    "Reused strong-shift runs:",
    len(strong_results),
)
print(
    "Total result records:",
    len(results),
)
print("Saved:", output_path)
print("=" * 75)
