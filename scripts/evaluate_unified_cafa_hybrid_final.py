"""Evaluate ONE unified 17-class checkpoint: Source, BN-only, CAFA-TTA v3.

All methods see exactly the same deterministic, synthetically shifted test images.
CAFA adaptation uses no test labels. Separate online adaptation streams are reset
at the start of each dataset; the model checkpoint itself is not modified.

Example Docker invocation is in --help epilog / accompanying chat instructions.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from train_unified import (
    KatherDataset, PATHMNIST_NAMES, PathMNISTDataset, build_model,
    collect_kather, split_items,
)
from core.cafa_soft_adapter import CAFASoftAdapter
from models.feature_extractor import ResNetFeatureExtractor


class DeterministicShift(Dataset):
    """An identical deterministic RGB shift for all evaluation methods."""

    def __init__(self, dataset, mode: str):
        self.dataset = dataset
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.mode == "shifted":
            # Operate in RGB [0,1] *before* the training-time [-1,1] normalization.
            x = (x * 0.65 + 0.18).clamp(0.0, 1.0)
            x = x.pow(1.15)
        return x.mul(2.0).sub(1.0), y


def metrics(true, predicted, domain_start, domain_end):
    labels = list(range(domain_start, domain_end))
    n = len(true)
    correct = sum(int(a == b) for a, b in zip(true, predicted))
    f1s = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, predicted))
        fp = sum(a != label and b == label for a, b in zip(true, predicted))
        fn = sum(a == label and b != label for a, b in zip(true, predicted))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {"samples": n, "accuracy": correct / n if n else 0.0,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0}



class StableBNFeatures(ResNetFeatureExtractor):
    """Keep source BatchNorm statistics during CAFA adaptation."""
    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        return self


def eval_domain(name, dataset, base_model, prototypes, args, device):
    start, end = (0, 9) if name == "pathmnist" else (9, 17)
    if args.max_samples:
        # Same sampled indices for all methods; deterministic, without replacement.
        rng = random.Random(args.seed + start)
        chosen = rng.sample(range(len(dataset)), min(args.max_samples, len(dataset)))
        dataset = Subset(dataset, sorted(chosen))
    dataset = DeterministicShift(dataset, args.condition)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=False)

    source = copy.deepcopy(base_model).to(device).eval()
    # BN-only uses target batch statistics; Source uses source running statistics.
    # Hybrid-BN interpolates them. CAFA adds online parameter adaptation.
    bn_only = copy.deepcopy(base_model).to(device)
    wrapped_bn = ResNetFeatureExtractor(bn_only)
    bn_only.requires_grad_(False)
    for module in bn_only.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    wrapped_bn.train()

    # Hybrid-BN baseline without online adaptation.
    hybrid_only = copy.deepcopy(base_model).to(device).eval()
    hybrid_only.requires_grad_(False)

    adapted_backbone = copy.deepcopy(base_model).to(device)
    adapter = CAFASoftAdapter(
        model=StableBNFeatures(adapted_backbone),
        prototypes=prototypes,
        lr=args.lr,
        lambda_align=args.lambda_align,
        entropy_floor=args.entropy_floor,
        min_samples=2,
    )
    # Restore pretrained BN statistics after CAFAAdapter initialization.
    # CAFAAdapter normally disables these statistics.
    for src, dst in zip(base_model.modules(), adapted_backbone.modules()):
        if isinstance(src, torch.nn.BatchNorm2d):
            dst.track_running_stats = True
            dst.running_mean = src.running_mean.detach().clone()
            dst.running_var = src.running_var.detach().clone()
            dst.num_batches_tracked = src.num_batches_tracked.detach().clone()
            dst.eval()


    # Hybrid-BN mixes source and target statistics using alpha: PathMNIST=0.75, Kather=0.25 by default.
    # Preserve BN affine parameters for CAFA adaptation.
    selected_alpha = (args.alpha if args.alpha is not None
                      else (0.75 if name == "pathmnist" else 0.25))
    if not 0.0 <= selected_alpha <= 1.0:
        raise ValueError("Hybrid-BN alpha must be between 0 and 1")
    from types import MethodType

    def hybrid_bn_forward(bn, x):
        source_mean = bn.running_mean
        source_var = bn.running_var

        target_mean = x.detach().mean(dim=(0, 2, 3))
        target_var = x.detach().var(
            dim=(0, 2, 3), unbiased=False
        )

        alpha = selected_alpha
        mixed_mean = (
            (1.0 - alpha) * source_mean
            + alpha * target_mean
        )
        mixed_var = (
            (1.0 - alpha) * source_var
            + alpha * target_var
        )

        return torch.nn.functional.batch_norm(
            x, mixed_mean, mixed_var,
            bn.weight, bn.bias,
            training=False, eps=bn.eps
        )

    for backbone in (hybrid_only, adapted_backbone):
        for module in backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.forward = MethodType(hybrid_bn_forward, module)

    print(f"HYBRID BN: source={1-selected_alpha:.0%}, target={selected_alpha:.0%}", flush=True)

    outcomes = {key: {"true": [], "pred": []} for key in
                ("source", "bn_only", "hybrid_bn_no_adapt", "cafa_v3_hybrid_bn")}
    initial_cafa_params = [
        param.detach().clone() for param in adapter.params
    ]
    updates, agreements, skipped_singletons = 0, 0, 0
    for step, (x, y) in enumerate(loader, 1):
        # ResNet BN with 1x1 end features cannot train on singleton batches.
        if len(y) < 2:
            skipped_singletons += len(y)
            continue
        x = x.to(device)
        truth = y.tolist()
        with torch.inference_mode():
            source_pred = source(x).argmax(1).cpu().tolist()
            _, bn_logits = wrapped_bn(x)
            bn_pred = bn_logits.argmax(1).cpu().tolist()
            hybrid_pred = hybrid_only(x).argmax(1).cpu().tolist()
        report = adapter.adapt(x)
        with torch.no_grad():
            cafa_pred = adapter.predict(x).argmax(1).cpu().tolist()
        updates += bool(report.get("updated", False))
        agreements += int(report.get("selected", 0))
        for method, preds in (
            ("source", source_pred),
            ("bn_only", bn_pred),
            ("hybrid_bn_no_adapt", hybrid_pred),
            ("cafa_v3_hybrid_bn", cafa_pred),
        ):
            outcomes[method]["true"].extend(truth)
            outcomes[method]["pred"].extend(preds)
        if step % args.log_every == 0 or step == len(loader):
            print(f"{name.upper()} {step}/{len(loader)} batches", flush=True)

    with torch.no_grad():
        delta_max = max(
            (param.detach() - initial).abs().max().item()
            for param, initial in zip(adapter.params, initial_cafa_params)
        )
        delta_l2 = sum(
            (param.detach() - initial).float().square().sum().item()
            for param, initial in zip(adapter.params, initial_cafa_params)
        ) ** 0.5
    print(
        f"CAFA PARAMETER CHANGE {name}: "
        f"max_abs={delta_max:.10e}, l2={delta_l2:.10e}",
        flush=True
    )
    result = {method: metrics(data["true"], data["pred"], start, end)
              for method, data in outcomes.items()}
    result["adaptation"] = {"updated_batches": updates, "agreement_samples": agreements,
                            "skipped_singleton_samples": skipped_singletons}
    # Paired correctness comparison: same samples, same batch order.
    from math import comb
    truths = outcomes["cafa_v3_hybrid_bn"]["true"]
    hybrid_preds = outcomes["hybrid_bn_no_adapt"]["pred"]
    cafa_preds = outcomes["cafa_v3_hybrid_bn"]["pred"]

    both_correct = both_wrong = hybrid_only_correct = cafa_only_correct = 0

    for truth, hp, cp in zip(truths, hybrid_preds, cafa_preds):
        h = hp == truth
        c = cp == truth
        if h and c:
            both_correct += 1
        elif h:
            hybrid_only_correct += 1
        elif c:
            cafa_only_correct += 1
        else:
            both_wrong += 1

    discordant = hybrid_only_correct + cafa_only_correct
    if discordant:
        k = min(hybrid_only_correct, cafa_only_correct)
        p_exact = min(
            1.0,
            2 * sum(comb(discordant, i) for i in range(k + 1))
            / (2 ** discordant)
        )
    else:
        p_exact = 1.0

    assert (
        both_correct + both_wrong
        + hybrid_only_correct + cafa_only_correct
        == len(truths)
    )

    result["paired_hybrid_vs_cafa"] = {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "hybrid_only_correct": hybrid_only_correct,
        "cafa_only_correct": cafa_only_correct,
        "discordant": discordant,
        "mcnemar_exact_p": p_exact,
    }
    print(
        f"PAIRED AUDIT {name}: "
        f"Hybrid-only={hybrid_only_correct}, "
        f"CAFA-only={cafa_only_correct}, "
        f"McNemar p={p_exact:.8g}",
        flush=True
    )

    result["cafa_vs_hybrid_no_adapt_different"] = sum(
        a != b for a, b in zip(
            outcomes["cafa_v3_hybrid_bn"]["pred"],
            outcomes["hybrid_bn_no_adapt"]["pred"]
        )
    )
    disagreement = sum(
        a != b for a, b in zip(
            outcomes["bn_only"]["pred"],
            outcomes["cafa_v3_hybrid_bn"]["pred"]
        )
    )
    for method, data in outcomes.items():
        predictions = data["pred"]
        in_domain = sum(start <= x < end for x in predictions)
        print(
            f"DIAGNOSTIC {name} {method}: "
            f"in_domain={in_domain}/{len(predictions)}",
            flush=True
        )
    print(
        f"DIAGNOSTIC {name}: CAFA_vs_BN_different={disagreement}",
        flush=True
    )
    print(f"RESULT {name}: {json.dumps(result)}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prototypes", type=Path, required=True)
    p.add_argument("--pathmnist", type=Path, required=True)
    p.add_argument("--kather-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=256,
                   help="Per-domain subset; 0 means entire test set (slower)")
    p.add_argument("--condition", choices=("shifted", "clean"), default="shifted")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-align", type=float, default=0.05)
    p.add_argument("--entropy-floor", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=None,
                   help="Target-statistics weight for Hybrid-BN. Default: per-domain "
                        "values chosen on validation (PathMNIST 0.75, Kather 0.25).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=4)
    args = p.parse_args()
    if args.alpha is not None and not 0 <= args.alpha <= 1:
        p.error("--alpha must be between 0 and 1")
    if args.batch_size < 2 or args.max_samples < 0 or args.log_every < 1:
        p.error("batch-size >= 2, max-samples >= 0, log-every >= 1 are required")
    if args.output.exists():
        p.error(f"Refusing to overwrite existing results: {args.output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    proto_state = torch.load(args.prototypes, map_location="cpu", weights_only=True)
    labels = list(ckpt["labels"])
    prototypes = proto_state["prototypes"]
    if len(labels) != 17 or proto_state["labels"] != labels or tuple(prototypes.shape) != (17, 512):
        raise ValueError("Checkpoint and prototypes must contain the SAME 17 labels and 17x512 prototypes")
    if proto_state.get("split") != "train-only":
        raise ValueError("Refusing prototypes that are not labeled train-only")
    if int(proto_state["source_seed"]) != int(ckpt["seed"]):
        raise ValueError("Prototype split seed and checkpoint differ")
    dirs = collect_kather(args.kather_root)
    expected = [f"PathMNIST:{name}" for name in PATHMNIST_NAMES] + [f"Kather:{d.name}" for d, _ in dirs]
    if labels != expected:
        raise ValueError("Class order in test data differs from checkpoint")
    seed = int(ckpt["seed"])
    splits = split_items([imgs for _, imgs in dirs], seed)
    size = int(ckpt["input_size"])
    norm = ckpt["normalization"]
    if norm["mean"] != [0.5]*3 or norm["std"] != [0.5]*3:
        raise ValueError("This shift evaluator expects checkpoint normalization mean/std == 0.5")
    eval_tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    path_test = PathMNISTDataset(args.pathmnist, "test", eval_tfm)
    kather_test = KatherDataset(splits["test"], eval_tfm)
    model = build_model(len(labels))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EVALUATION: {device}; {args.condition}; max_samples={args.max_samples}; batch={args.batch_size}", flush=True)
    result = {"metadata": {"checkpoint": args.checkpoint.name, "prototypes": args.prototypes.name,
                           "seed": seed, "labels": labels, "condition": args.condition,
                           "max_samples_per_domain": args.max_samples,
                           "hybrid_bn_alpha": (args.alpha if args.alpha is not None else
                                               {"pathmnist": 0.75, "kather": 0.25}),
                           "online_adaptation": True,
                           "comparison_methods": {
                               "source": "source running BN statistics",
                               "bn_only": "target batch BN statistics",
                               "hybrid_bn_no_adapt": "mixed BN without updates",
                               "cafa_v3_hybrid_bn": "mixed BN with CAFA updates"},
                           "protocol": "adapt on unlabeled target batch, then predict same batch; reset between datasets",
                           "macro_f1_class_scope": "9 PathMNIST labels or 8 Kather labels"}}
    result["pathmnist"] = eval_domain("pathmnist", path_test, model, prototypes, args, device)
    result["kather"] = eval_domain("kather", kather_test, model, prototypes, args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("COMPARISON SAVED:", args.output, flush=True)


if __name__ == "__main__":
    main()
