
import sys
import json
import random
import time
import gc
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor
from kernel.tent import TentAdapter
from core.cafa_soft_adapter import CAFASoftAdapter

# ==========================================
# CAFA-TTA v3 | COMPUTATIONAL BENCHMARK
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
BATCH_SIZE = 64

LR = 1e-4
ENTROPY_FLOOR = 0.2
LAMBDA_ALIGN = 0.05
MIN_SAMPLES = 2

WARMUP_BATCHES = 10
MEASURE_BATCHES = 50
REPEATS = 3

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

# ==========================================
# Environment
# ==========================================

environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_version": torch.version.cuda,
    "device": str(DEVICE),
    "gpu": (
        torch.cuda.get_device_name(DEVICE)
        if DEVICE.type == "cuda"
        else None
    ),
    "gpu_total_memory_mib": (
        torch.cuda.get_device_properties(
            DEVICE
        ).total_memory / (1024 ** 2)
        if DEVICE.type == "cuda"
        else None
    ),
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def cleanup():
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        synchronize()


# ==========================================
# Dataset and strong domain shift
# ==========================================

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8",
) as f:

    shift_data = json.load(f)

assert shift_data["split"] == "val"

shift = shift_data["shift"]

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


dataset = PathMNIST(
    split="val",
    root=str(ROOT / "data/raw/medmnist"),
    transform=target_transform,
    download=False,
    size=28,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

# Prepare identical input batches on CPU.
# Dataset loading is excluded from timing.

cached_batches = []

for images, labels in loader:

    cached_batches.append(images)

    if len(cached_batches) >= (
        WARMUP_BATCHES + MEASURE_BATCHES
    ):
        break

assert len(cached_batches) == (
    WARMUP_BATCHES + MEASURE_BATCHES
)

# ==========================================
# Model loading
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


def load_prototypes(config):

    data = torch.load(
        ROOT / "checkpoints" / config["prototypes"],
        map_location="cpu",
        weights_only=True,
    )

    assert (
        data["source_checkpoint"]
        == config["checkpoint"]
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
# Benchmark one independent run
# ==========================================

def benchmark(model_name, config, method, repeat):

    cleanup()
    set_seed()

    backbone = create_backbone(
        config["checkpoint"]
    )

    adapter = None
    feature_model = None
    prototypes = None

    prototype_memory_bytes = 0
    prototype_file_bytes = 0

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

    elif method == "cafa_v3":

        feature_model = ResNetFeatureExtractor(
            backbone
        )

        prototypes = load_prototypes(
            config
        )

        adapter = CAFASoftAdapter(
            model=feature_model,
            prototypes=prototypes,
            lr=LR,
            lambda_align=LAMBDA_ALIGN,
            entropy_floor=ENTROPY_FLOOR,
            min_samples=MIN_SAMPLES,
        )

        prototype_memory_bytes = (
            adapter.prototypes.numel()
            * adapter.prototypes.element_size()
        )

        prototype_file_bytes = (
            ROOT
            / "checkpoints"
            / config["prototypes"]
        ).stat().st_size

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    total_params = sum(
        p.numel()
        for p in backbone.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in backbone.parameters()
        if p.requires_grad
    )

    def predict(images):

        with torch.no_grad():

            if adapter is None:
                logits = backbone(images)
            else:
                logits = adapter.predict(images)

            return logits.argmax(dim=1)

    # ======================================
    # Warm-up
    # ======================================

    for cpu_images in cached_batches[
        :WARMUP_BATCHES
    ]:

        images = cpu_images.to(DEVICE)

        predictions = predict(images)

        if adapter is not None:
            adapter.adapt(images)

        del images
        del predictions

    synchronize()

    # Measure memory after warm-up.
    # Optimizer state has already been created.

    if DEVICE.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

        starting_memory = (
            torch.cuda.memory_allocated(
                DEVICE
            ) / (1024 ** 2)
        )

    else:

        starting_memory = None

    inference_times = []
    adaptation_times = []
    total_times = []

    updated_batches = 0

    # ======================================
    # Timed measurements
    # ======================================

    for cpu_images in cached_batches[
        WARMUP_BATCHES:
    ]:

        # CPU-to-GPU transfer is excluded.
        images = cpu_images.to(DEVICE)

        synchronize()

        start = time.perf_counter()

        predictions = predict(images)

        synchronize()

        inference_end = time.perf_counter()

        if adapter is not None:

            result = adapter.adapt(images)

            synchronize()

            adaptation_end = time.perf_counter()

            adaptation_ms = (
                adaptation_end - inference_end
            ) * 1000

            if method == "cafa_v3":
                if result["updated"]:
                    updated_batches += 1

        else:

            adaptation_end = inference_end
            adaptation_ms = 0.0

        inference_ms = (
            inference_end - start
        ) * 1000

        total_ms = (
            adaptation_end - start
        ) * 1000

        inference_times.append(
            inference_ms
        )

        adaptation_times.append(
            adaptation_ms
        )

        total_times.append(
            total_ms
        )

        del images
        del predictions

    synchronize()

    if DEVICE.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated(
                DEVICE
            ) / (1024 ** 2)
        )

        peak_reserved_memory = (
            torch.cuda.max_memory_reserved(
                DEVICE
            ) / (1024 ** 2)
        )

    else:

        peak_memory = None
        peak_reserved_memory = None

    total_samples = (
        MEASURE_BATCHES * BATCH_SIZE
    )

    total_seconds = (
        sum(total_times) / 1000
    )

    metrics = {
        "model": model_name,
        "method": method,
        "repeat": repeat,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (
            trainable_params
            / total_params
            * 100
        ),
        "prototype_memory_bytes": (
            prototype_memory_bytes
        ),
        "prototype_file_bytes": (
            prototype_file_bytes
        ),
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURE_BATCHES,
        "measured_samples": total_samples,
        "inference_mean_ms": float(
            np.mean(inference_times)
        ),
        "adaptation_mean_ms": float(
            np.mean(adaptation_times)
        ),
        "total_mean_ms": float(
            np.mean(total_times)
        ),
        "total_median_ms": float(
            np.median(total_times)
        ),
        "total_p95_ms": float(
            np.percentile(
                total_times,
                95
            )
        ),
        "throughput_images_per_second": (
            total_samples / total_seconds
        ),
        "starting_gpu_memory_mib": (
            starting_memory
        ),
        "peak_gpu_memory_mib": peak_memory,
        "peak_reserved_gpu_memory_mib": (
            peak_reserved_memory
        ),
        "additional_peak_gpu_memory_mib": (
            peak_memory - starting_memory
            if peak_memory is not None
            else None
        ),
    }

    if method == "cafa_v3":

        metrics["updated_batches"] = (
            updated_batches
        )

    del predict
    del adapter
    del feature_model
    del prototypes
    del backbone

    cleanup()

    return metrics


# ==========================================
# Output configuration
# ==========================================

output_dir = (
    ROOT / "experiments" / "cafa_v3_benchmark"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

metadata = {
    "environment": environment,
    "split": "val",
    "shift": shift,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "entropy_floor": ENTROPY_FLOOR,
    "lambda_align": LAMBDA_ALIGN,
    "warmup_batches": WARMUP_BATCHES,
    "measured_batches": MEASURE_BATCHES,
    "repeats": REPEATS,
    "timing_protocol": (
        "synchronized_predict_then_adapt"
    ),
    "data_loading_included": False,
    "host_to_device_transfer_included": False,
    "models": MODELS,
    "methods": METHODS,
}


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
# Execute benchmark
# ==========================================

TOTAL_RUNS = (
    len(MODELS)
    * len(METHODS)
    * REPEATS
)

print("=" * 75)
print("CAFA-TTA v3 | COMPUTATIONAL BENCHMARK")
print("Device:", DEVICE)
print("GPU:", environment["gpu"])
print("PyTorch:", environment["torch"])
print("CUDA:", environment["cuda_version"])
print("Batch size:", BATCH_SIZE)
print("Warm-up batches:", WARMUP_BATCHES)
print("Measurement batches:", MEASURE_BATCHES)
print("Repeats:", REPEATS)
print("Total runs:", TOTAL_RUNS)
print("=" * 75)

results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("=" * 75)

    for repeat in range(1, REPEATS + 1):

        # Rotate method order to reduce order bias.
        offset = (repeat - 1) % len(METHODS)

        method_order = (
            METHODS[offset:]
            + METHODS[:offset]
        )

        for method in method_order:

            metrics = benchmark(
                model_name,
                config,
                method,
                repeat,
            )

            results.append(metrics)

            memory = metrics[
                "peak_gpu_memory_mib"
            ]

            memory_text = (
                f"{memory:.1f} MiB"
                if memory is not None
                else "N/A"
            )

            print(
                f"[{len(results):2d}/{TOTAL_RUNS}] "
                f"Repeat={repeat} | "
                f"{method:12s} | "
                f"Infer="
                f"{metrics['inference_mean_ms']:.2f} ms | "
                f"Adapt="
                f"{metrics['adaptation_mean_ms']:.2f} ms | "
                f"Total="
                f"{metrics['total_mean_ms']:.2f} ms | "
                f"GPU={memory_text}",
                flush=True,
            )

            save_json(
                output_dir / "partial_results.json",
                {
                    **metadata,
                    "completed_runs": len(results),
                    "total_runs": TOTAL_RUNS,
                    "results": results,
                },
            )


# ==========================================
# Aggregate benchmark results
# ==========================================

summary = {}

METRIC_KEYS = [
    "inference_mean_ms",
    "adaptation_mean_ms",
    "total_mean_ms",
    "total_median_ms",
    "total_p95_ms",
    "throughput_images_per_second",
    "peak_gpu_memory_mib",
    "peak_reserved_gpu_memory_mib",
    "additional_peak_gpu_memory_mib",
    "trainable_parameters",
    "trainable_percentage",
    "prototype_memory_bytes",
    "prototype_file_bytes",
]

print("\n" + "=" * 75)
print("FINAL BENCHMARK SUMMARY")
print("=" * 75)

for model_name in MODELS:

    summary[model_name] = {}

    print("\nMODEL:", model_name.upper())

    for method in METHODS:

        rows = [
            row for row in results
            if row["model"] == model_name
            and row["method"] == method
        ]

        assert len(rows) == REPEATS

        method_summary = {}

        for key in METRIC_KEYS:

            values = [
                row[key]
                for row in rows
                if row[key] is not None
            ]

            method_summary[key] = {
                "mean": (
                    float(np.mean(values))
                    if values else None
                ),
                "std": (
                    float(
                        np.std(values, ddof=1)
                    )
                    if len(values) > 1
                    else None
                ),
            }

        summary[model_name][method] = (
            method_summary
        )

        total = method_summary[
            "total_mean_ms"
        ]

        throughput = method_summary[
            "throughput_images_per_second"
        ]["mean"]

        memory = method_summary[
            "peak_gpu_memory_mib"
        ]["mean"]

        params = method_summary[
            "trainable_parameters"
        ]["mean"]

        memory_text = (
            f"{memory:.1f} MiB"
            if memory is not None
            else "N/A"
        )

        print(
            f"\n{method.upper()}"
        )

        print(
            "  Inference:",
            f"{method_summary['inference_mean_ms']['mean']:.3f} ms"
        )

        print(
            "  Adaptation:",
            f"{method_summary['adaptation_mean_ms']['mean']:.3f} ms"
        )

        print(
            "  Total latency:",
            f"{total['mean']:.3f} "
            f"+/- {total['std']:.3f} ms/batch"
        )

        print(
            "  Throughput:",
            f"{throughput:.2f} images/s"
        )

        print(
            "  Peak GPU memory:",
            memory_text
        )

        print(
            "  Trainable parameters:",
            f"{params:,.0f}"
        )

        print(
            "  Prototype storage:",
            f"{method_summary['prototype_memory_bytes']['mean']/1024:.2f} KiB"
        )

    # ======================================
    # CAFA v3 overhead relative to TENT
    # ======================================

    tent = summary[model_name]["tent"]
    cafa = summary[model_name]["cafa_v3"]

    tent_latency = tent[
        "total_mean_ms"
    ]["mean"]

    cafa_latency = cafa[
        "total_mean_ms"
    ]["mean"]

    tent_memory = tent[
        "peak_gpu_memory_mib"
    ]["mean"]

    cafa_memory = cafa[
        "peak_gpu_memory_mib"
    ]["mean"]

    tent_throughput = tent[
        "throughput_images_per_second"
    ]["mean"]

    cafa_throughput = cafa[
        "throughput_images_per_second"
    ]["mean"]

    comparison = {
        "additional_latency_ms": (
            cafa_latency - tent_latency
        ),
        "latency_overhead_percentage": (
            (
                cafa_latency / tent_latency
            ) - 1
        ) * 100,
        "additional_peak_memory_mib": (
            cafa_memory - tent_memory
            if cafa_memory is not None
            and tent_memory is not None
            else None
        ),
        "throughput_change_percentage": (
            (
                cafa_throughput / tent_throughput
            ) - 1
        ) * 100,
    }

    summary[model_name][
        "cafa_vs_tent"
    ] = comparison

    print("\nCAFA v3 VS TENT OVERHEAD")

    print(
        "Additional latency:",
        f"{comparison['additional_latency_ms']:+.3f} ms"
    )

    print(
        "Latency overhead:",
        f"{comparison['latency_overhead_percentage']:+.2f}%"
    )

    print(
        "Throughput change:",
        f"{comparison['throughput_change_percentage']:+.2f}%"
    )

    if comparison[
        "additional_peak_memory_mib"
    ] is not None:

        print(
            "Additional peak GPU memory:",
            f"{comparison['additional_peak_memory_mib']:+.2f} MiB"
        )


# ==========================================
# Save final results
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
    },
)

print("\n" + "=" * 75)
print("CAFA-TTA v3 BENCHMARK COMPLETED")
print("Completed runs:", len(results), "/", TOTAL_RUNS)
print("Saved:", output_path)
print("=" * 75)
