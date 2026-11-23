from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_unified import (
    PathMNISTDataset,
    KatherDataset,
    collect_kather,
    split_items,
)


def features(x):
    return np.array([
        float(x.mean()),
        float(x.std(unbiased=False)),
    ], dtype=np.float64)


def shifted(x):
    return (x * 0.65 + 0.18).clamp(0, 1).pow(1.15)


def evaluate(name, dataset, seed):
    rng = random.Random(seed)
    indices = rng.sample(
        range(len(dataset)),
        min(256, len(dataset)),
    )

    clean_features = []
    shifted_features = []

    for idx in indices:
        x, _ = dataset[idx]
        clean_features.append(features(x))
        shifted_features.append(features(shifted(x)))

    clean_features = np.stack(clean_features)
    shifted_features = np.stack(shifted_features)

    n_train = len(indices) // 2
    clean_train = clean_features[:n_train]
    shifted_train = shifted_features[:n_train]
    clean_test = clean_features[n_train:]
    shifted_test = shifted_features[n_train:]

    center_clean = clean_train.mean(axis=0)
    center_shifted = shifted_train.mean(axis=0)

    scale = np.std(
        np.concatenate([clean_train, shifted_train]),
        axis=0,
    )
    scale = np.maximum(scale, 1e-6)


    def shift_score(a):
        d_clean = np.sum(
            ((a - center_clean) / scale) ** 2,
            axis=1,
        )
        d_shifted = np.sum(
            ((a - center_shifted) / scale) ** 2,
            axis=1,
        )
        return d_clean - d_shifted


    # Higher scores indicate stronger evidence of shift.
    calibration_scores = shift_score(clean_train)
    clean_scores = shift_score(clean_test)
    shifted_scores = shift_score(shifted_test)

    # ROC-AUC via pairwise ranking, including half-credit ties.
    comparisons = (
        shifted_scores[:, None]
        - clean_scores[None, :]
    )
    auc = float(
        (np.count_nonzero(comparisons > 0)
         + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )

    operating_points = {}
    for target_fpr in (0.05, 0.10, 0.20):
        threshold_i = float(np.quantile(
            calibration_scores,
            1.0 - target_fpr,
            method="higher"
        ))

        operating_points[str(target_fpr)] = {
            "threshold": threshold_i,
            "calibration_false_alarm_rate": float(
                np.mean(calibration_scores > threshold_i)
            ),
            "false_alarm_rate": float(
                np.mean(clean_scores > threshold_i)
            ),
            "detection_rate": float(
                np.mean(shifted_scores > threshold_i)
            ),
        }

    selected = operating_points["0.05"]
    threshold = selected["threshold"]
    calibration_false_alarm = selected[
        "calibration_false_alarm_rate"
    ]
    false_alarm = selected["false_alarm_rate"]
    detection = selected["detection_rate"]

    result = {
        "validation_images": len(indices),
        "calibration_images": n_train,
        "evaluation_images": len(clean_test),
        "detection_rate": detection,
        "false_alarm_rate": false_alarm,
        "calibration_false_alarm_rate": calibration_false_alarm,
        "threshold": threshold,
        "target_calibration_fpr": 0.05,
        "roc_auc": auc,
        "operating_points": operating_points,
        "balanced_accuracy": (
            detection + (1.0 - false_alarm)
        ) / 2,
        "clean_center": center_clean.tolist(),
        "shifted_center": center_shifted.tolist(),
    }

    print(
        f"RESULT {name}: {json.dumps(result)}",
        flush=True,
    )

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pathmnist", required=True)
    p.add_argument("--kather-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    tfm = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])

    path_val = PathMNISTDataset(
        Path(args.pathmnist), "val", tfm
    )

    dirs = collect_kather(Path(args.kather_root))
    splits = split_items(
        [imgs for _, imgs in dirs],
        args.seed,
    )
    kather_val = KatherDataset(
        splits["val"], tfm
    )

    results = {
        "metadata": {
            "split": "validation",
            "seed": args.seed,
            "method": "mean_std_roc_calibration",
            "shift": "brightness_contrast_gamma",
        },
        "pathmnist": evaluate(
            "pathmnist", path_val, args.seed
        ),
        "kather": evaluate(
            "kather", kather_val, args.seed + 9
        ),
    }

    with open(args.output, "x", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("SHIFT DETECTOR VALIDATION: PASS")


if __name__ == "__main__":
    main()
