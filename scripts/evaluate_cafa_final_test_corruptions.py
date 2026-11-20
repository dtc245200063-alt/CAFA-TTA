
import sys
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# ==========================================
# CAFA-TTA | FINAL TEST MULTI-CORRUPTION
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

VALIDATION_REFERENCE = (
    ROOT
    / "experiments"
    / "cafa_v3_corruptions"
    / "metrics.json"
)

# ==========================================
# Reuse verified evaluation functions
# without executing previous experiments
# ==========================================

tree = ast.parse(
    BASE_SCRIPT.read_text(
        encoding="utf-8-sig"
    ),
    filename=str(BASE_SCRIPT),
)

required_functions = {
    "sha256_file",
    "verify_hash_manifest",
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

found = {
    node.name
    for node in selected_nodes
    if isinstance(node, ast.FunctionDef)
}

assert found == required_functions

shared_tree = ast.Module(
    body=selected_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(
    shared_tree
)

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
Dataset = namespace["Dataset"]
PathMNIST = namespace["PathMNIST"]
TF = namespace["TF"]
transforms = namespace["transforms"]

sha256_file = namespace["sha256_file"]
save_json = namespace["save_json"]
evaluate = namespace["evaluate"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

CORRUPTIONS = {
    "gaussian_noise": {
        "light": 0.02,
        "medium": 0.05,
        "strong": 0.10,
    },
    "gaussian_blur": {
        "light": 0.5,
        "medium": 1.0,
        "strong": 2.0,
    },
    "hue_shift": {
        "light": 0.03,
        "medium": 0.07,
        "strong": 0.15,
    },
}

NOISE_SEED = 2026
BLUR_KERNEL = 13

# Limit CPU thread overhead.
# Model evaluation still runs on CUDA.

torch.set_num_threads(1)

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3,
)

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
})

# ==========================================
# Verify frozen checkpoints and algorithms
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

with open(
    REFERENCE_PATH,
    encoding="utf-8",
) as f:

    reference = json.load(f)

assert reference["split"] == "test"

assert reference["protocol"] == (
    "shuffled_online_prequential"
)

assert reference["completed_runs"] == 63

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

print("Checkpoint hashes: VERIFIED")
print("Algorithm hashes: VERIFIED")
print("Frozen configuration: VERIFIED")
print("Shared evaluation functions: VERIFIED")

# ==========================================
# Verify validation corruption configuration
# ==========================================

with open(
    VALIDATION_REFERENCE,
    encoding="utf-8",
) as f:

    validation_reference = json.load(f)

validation_corruptions = (
    validation_reference.get("corruptions")
)

if validation_corruptions is None:

    validation_corruptions = (
        validation_reference
        .get("metadata", {})
        .get("corruptions")
    )

if validation_corruptions is not None:

    assert validation_corruptions == CORRUPTIONS

    print(
        "Validation corruption strengths: VERIFIED"
    )

else:

    print(
        "WARNING: Corruption metadata key "
        "not found in validation results."
    )

# ==========================================
# Original PathMNIST test dataset
# ==========================================

raw_dataset = PathMNIST(
    split="test",
    root=str(
        ROOT / "data/raw/medmnist"
    ),
    transform=None,
    download=False,
    size=28,
)

assert len(raw_dataset) == 7180

# ==========================================
# Deterministic corruption functions
# ==========================================

def corrupt_image(
    image,
    index,
    corruption,
    strength,
):

    if corruption == "gaussian_noise":

        image = TF.to_tensor(image)

        generator = torch.Generator()

        generator.manual_seed(
            NOISE_SEED + index
        )

        noise = torch.randn(
            image.shape,
            generator=generator,
            dtype=image.dtype,
        )

        image = (
            image + strength * noise
        ).clamp(0.0, 1.0)

    elif corruption == "gaussian_blur":

        image = TF.to_tensor(image)

        image = TF.gaussian_blur(
            image,
            kernel_size=[
                BLUR_KERNEL,
                BLUR_KERNEL,
            ],
            sigma=[
                strength,
                strength,
            ],
        )

    elif corruption == "hue_shift":

        image = TF.adjust_hue(
            image,
            strength,
        )

        image = TF.to_tensor(image)

    else:

        raise ValueError(
            f"Unknown corruption: {corruption}"
        )

    return normalize(image)


# ==========================================
# Precompute one corruption condition
# ==========================================

class CachedCorruptionDataset(Dataset):

    def __init__(
        self,
        corruption,
        strength,
    ):

        self.images = torch.empty(
            (
                len(raw_dataset),
                3,
                28,
                28,
            ),
            dtype=torch.float32,
        )

        self.labels = torch.empty(
            len(raw_dataset),
            dtype=torch.long,
        )

        print(
            "Preparing deterministic "
            "corrupted images...",
            flush=True,
        )

        for index in range(
            len(raw_dataset)
        ):

            image, label = raw_dataset[index]

            self.images[index] = (
                corrupt_image(
                    image,
                    index,
                    corruption,
                    strength,
                )
            )

            self.labels[index] = int(
                np.asarray(label)
                .reshape(-1)[0]
            )

        assert torch.isfinite(
            self.images
        ).all()

        print(
            "Prepared:",
            len(self.images),
            "images",
            flush=True,
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.images[index],
            self.labels[index],
            index,
        )


# ==========================================
# Output configuration
# ==========================================

output_dir = (
    ROOT
    / "experiments"
    / "cafa_final_test_corruptions"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

partial_path = (
    output_dir / "partial_results.json"
)

TOTAL_RUNS = (
    len(MODELS)
    * sum(
        len(levels)
        for levels in CORRUPTIONS.values()
    )
    * (
        1
        + (len(METHODS) - 1)
        * len(SEEDS)
    )
)

assert TOTAL_RUNS == 270

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
    "corruptions": CORRUPTIONS,
    "noise_seed": NOISE_SEED,
    "noise_seed_per_image": (
        "noise_seed_plus_original_index"
    ),
    "blur_kernel": BLUR_KERNEL,
    "source_only_single_seed": True,
    "corrupted_images_cached_on_cpu": True,
    "checkpoint_hashes": checkpoint_hashes,
    "algorithm_hashes": algorithm_hashes,
    "reference_metrics_hash": (
        sha256_file(REFERENCE_PATH)
    ),
    "validation_reference_hash": (
        sha256_file(VALIDATION_REFERENCE)
    ),
    "base_script_hash": (
        sha256_file(BASE_SCRIPT)
    ),
    "evaluation_script_hash": (
        sha256_file(Path(__file__))
    ),
    "torch_version": torch.__version__,
    "device": str(DEVICE),
    "cpu_threads": torch.get_num_threads(),
}


def run_key(row):

    return (
        row["corruption"],
        row["severity"],
        row["model"],
        row["method"],
        row["seed"],
    )


expected_keys = {
    (
        corruption,
        severity,
        model_name,
        method,
        seed,
    )
    for corruption, levels
    in CORRUPTIONS.items()
    for severity in levels
    for model_name in MODELS
    for method in METHODS
    for seed in (
        [SEEDS[0]]
        if method == "source_only"
        else SEEDS
    )
}

assert len(expected_keys) == TOTAL_RUNS

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

    assert completed.issubset(
        expected_keys
    )

# ==========================================
# Execute multi-corruption evaluation
# ==========================================

print("\n" + "=" * 75)
print("CAFA-TTA | FINAL TEST MULTI-CORRUPTION")
print("Device:", DEVICE)
print("Dataset: PathMNIST Test")
print("Samples:", len(raw_dataset))
print("Batch size:", BATCH_SIZE)
print("Seeds:", SEEDS)
print("Corruptions:", CORRUPTIONS)
print("Total runs:", TOTAL_RUNS)
print("Already completed:", len(completed))
print("=" * 75)

for corruption, levels in CORRUPTIONS.items():

    for severity, strength in levels.items():

        print("\n" + "=" * 75)

        print(
            "CORRUPTION:",
            corruption.upper(),
        )

        print(
            "SEVERITY:",
            severity.upper(),
        )

        print("Strength:", strength)
        print("=" * 75)

        condition_keys = {
            key for key in expected_keys
            if key[0] == corruption
            and key[1] == severity
        }

        if condition_keys.issubset(
            completed
        ):

            print(
                "SKIP: Entire condition completed."
            )

            continue

        dataset = CachedCorruptionDataset(
            corruption,
            strength,
        )

        prediction_dir = (
            output_dir
            / "predictions"
            / corruption
            / severity
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        namespace["dataset"] = dataset

        namespace["prediction_dir"] = (
            prediction_dir
        )

        for model_name, model_config in MODELS.items():

            print(
                "\nMODEL:",
                model_name.upper(),
            )

            for seed in SEEDS:

                for method in METHODS:

                    if (
                        method == "source_only"
                        and seed != SEEDS[0]
                    ):

                        continue

                    key = (
                        corruption,
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

                    metrics["corruption"] = (
                        corruption
                    )

                    metrics["severity"] = (
                        severity
                    )

                    metrics["strength"] = (
                        strength
                    )

                    results.append(metrics)

                    completed.add(key)

                    line = (
                        f"[{len(results):3d}/"
                        f"{TOTAL_RUNS}] "
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
                            "completed_runs": (
                                len(results)
                            ),
                            "total_runs": TOTAL_RUNS,
                            "results": results,
                        },
                    )

        # Release cached images before
        # preparing the next condition.

        del namespace["dataset"]
        del dataset

assert completed == expected_keys
assert len(results) == TOTAL_RUNS

# ==========================================
# Aggregate statistics
# ==========================================

summary = {}

corruption_comparison = {}

severity_deltas = {
    "light": {
        "accuracy": [],
        "macro_f1": [],
    },
    "medium": {
        "accuracy": [],
        "macro_f1": [],
    },
    "strong": {
        "accuracy": [],
        "macro_f1": [],
    },
}

all_acc_deltas = []
all_f1_deltas = []


def comparison_stats(
    acc_deltas,
    f1_deltas,
):

    acc_deltas = np.asarray(
        acc_deltas,
        dtype=float,
    )

    f1_deltas = np.asarray(
        f1_deltas,
        dtype=float,
    )

    return {
        "mean_accuracy_delta_pp": float(
            acc_deltas.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            f1_deltas.mean()
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
        "paired_comparisons": len(
            acc_deltas
        ),
    }


print("\n" + "=" * 75)
print("FINAL TEST MULTI-CORRUPTION SUMMARY")
print("=" * 75)

for corruption, levels in CORRUPTIONS.items():

    summary[corruption] = {}

    corruption_acc_deltas = []
    corruption_f1_deltas = []

    print("\n" + "=" * 75)
    print("CORRUPTION:", corruption.upper())
    print("=" * 75)

    for severity in levels:

        summary[corruption][severity] = {}

        print("\nSEVERITY:", severity.upper())

        for model_name in MODELS:

            print(
                "\nMODEL:",
                model_name.upper(),
            )

            rows = [
                r for r in results
                if r["corruption"] == corruption
                and r["severity"] == severity
                and r["model"] == model_name
            ]

            tent_by_seed = {
                r["seed"]: r
                for r in rows
                if r["method"] == "tent"
            }

            assert set(
                tent_by_seed
            ) == set(SEEDS)

            model_summary = {}

            for method in METHODS:

                method_rows = [
                    r for r in rows
                    if r["method"] == method
                ]

                expected_count = (
                    1
                    if method == "source_only"
                    else 3
                )

                assert len(
                    method_rows
                ) == expected_count

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

                    comparison = comparison_stats(
                        acc_deltas,
                        f1_deltas,
                    )

                    entry.update(comparison)

                    entry["accuracy_per_seed_pp"] = (
                        acc_deltas.tolist()
                    )

                    entry["macro_f1_per_seed_pp"] = (
                        f1_deltas.tolist()
                    )

                    corruption_acc_deltas.extend(
                        acc_deltas.tolist()
                    )

                    corruption_f1_deltas.extend(
                        f1_deltas.tolist()
                    )

                    severity_deltas[severity][
                        "accuracy"
                    ].extend(
                        acc_deltas.tolist()
                    )

                    severity_deltas[severity][
                        "macro_f1"
                    ].extend(
                        f1_deltas.tolist()
                    )

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
                        f"{comparison['wins']}/"
                        f"{comparison['ties']}/"
                        f"{comparison['losses']}"
                    )

                model_summary[method] = (
                    entry
                )

            summary[corruption][severity][
                model_name
            ] = model_summary

    assert len(
        corruption_acc_deltas
    ) == 27

    corruption_comparison[corruption] = (
        comparison_stats(
            corruption_acc_deltas,
            corruption_f1_deltas,
        )
    )

    all_acc_deltas.extend(
        corruption_acc_deltas
    )

    all_f1_deltas.extend(
        corruption_f1_deltas
    )

# ==========================================
# Compare by corruption type
# ==========================================

print("\n" + "=" * 75)
print("CAFA v3 VS TENT | BY CORRUPTION")
print("=" * 75)

for corruption, row in (
    corruption_comparison.items()
):

    print(
        f"{corruption:17s} | "
        f"Acc Delta="
        f"{row['mean_accuracy_delta_pp']:+.3f} pp | "
        f"F1 Delta="
        f"{row['mean_macro_f1_delta_pp']:+.3f} pp | "
        f"W/T/L={row['wins']}/"
        f"{row['ties']}/"
        f"{row['losses']}"
    )

# ==========================================
# Compare by severity
# ==========================================

severity_comparison = {}

print("\n" + "=" * 75)
print("CAFA v3 VS TENT | BY SEVERITY")
print("=" * 75)

for severity, values in (
    severity_deltas.items()
):

    row = comparison_stats(
        values["accuracy"],
        values["macro_f1"],
    )

    severity_comparison[severity] = row

    print(
        f"{severity:10s} | "
        f"Acc Delta="
        f"{row['mean_accuracy_delta_pp']:+.3f} pp | "
        f"F1 Delta="
        f"{row['mean_macro_f1_delta_pp']:+.3f} pp | "
        f"W/T/L={row['wins']}/"
        f"{row['ties']}/"
        f"{row['losses']}"
    )

# ==========================================
# Overall comparison
# ==========================================

assert len(all_acc_deltas) == 81
assert len(all_f1_deltas) == 81

overall = comparison_stats(
    all_acc_deltas,
    all_f1_deltas,
)

print("\n" + "=" * 75)
print("OVERALL COMPARISON")
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

print(
    "Paired comparisons:",
    overall["paired_comparisons"],
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
        "completed_runs": len(results),
        "total_runs": TOTAL_RUNS,
        "results": results,
        "summary": summary,
        "corruption_comparison": (
            corruption_comparison
        ),
        "severity_comparison": (
            severity_comparison
        ),
        "overall": overall,
    },
)

print("\n" + "=" * 75)
print("FINAL TEST MULTI-CORRUPTION COMPLETED")
print(
    "Completed runs:",
    len(results),
    "/",
    TOTAL_RUNS,
)
print("Saved metrics:", output_path)
print(
    "Saved predictions:",
    output_dir / "predictions",
)
print("=" * 75)
