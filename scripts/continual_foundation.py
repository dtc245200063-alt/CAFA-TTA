"""Continual learning by consolidation, measured on held-out patients (Phikon features).

Kather 2016 is split by patient. Each round streams the unlabelled group-A images (new order
each round) through the current head with online feature TTA, keeps confident pseudo-labels
(confidence >= --min-confidence after temperature scaling, agreement with the nearest
prototype, not flagged by the OOD gate), then fine-tunes the head on those plus NCT replay with
distillation from the current head (LwF). The new head is kept only if it passes the same gate
as `scripts/consolidate.py` on the fixed NCT validation split. Group B (other patients) is never
used for learning; CRC-VAL-HE-7K tracks forgetting. Labels are used only for scoring.
Research use only; one external dataset.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.continual import OODGate  # noqa: E402
from core.feature_tta import FeatureSession  # noqa: E402
from core.foundation import train_head  # noqa: E402
from core.registry import passes_gate  # noqa: E402
from core.trainer import classification_metrics  # noqa: E402
from evaluate_foundation import GROUP_A, METHODS, build_head, load, static, stream  # noqa: E402


@torch.no_grad()
def pseudo_labels(head, extras, features, seed, method, min_confidence, fraction=None):
    """Confident, prototype-agreeing, in-distribution pseudo-labels.

    With `fraction` (CBST, Zou et al. ECCV 2018), the confidence threshold is replaced by keeping the
    top `fraction` most confident images of each predicted class, so rare or shifted classes still get labels.
    """
    order = np.random.default_rng(seed).permutation(len(features))
    session = FeatureSession(head, **METHODS[method])
    gate = OODGate.from_dict(extras["ood"])
    keep_idx, keep_y, keep_conf = [], [], []
    for i in range(0, len(order), 64):
        idx = order[i:i + 64]
        f = features[idx].float()
        logits = session.predict(f)
        prob = torch.softmax(logits / extras["temperature"], 1)
        conf, pred = prob.max(1)
        z = head.normalise(f)
        agree = (F.normalize(z, dim=1) @ extras["prototypes"].T).argmax(1) == pred
        ood = gate.is_ood(z, head(f), extras["prototypes"])
        ok = agree & ~ood if fraction is not None else (conf >= min_confidence) & agree & ~ood
        keep_idx += idx[ok.numpy()].tolist()
        keep_y += pred[ok].tolist()
        keep_conf += conf[ok].tolist()
    keep_idx, keep_y, keep_conf = np.array(keep_idx, dtype=int), np.array(keep_y), np.array(keep_conf)
    if fraction is not None:
        chosen = []
        for c in np.unique(keep_y):
            members = np.flatnonzero(keep_y == c)
            chosen += members[np.argsort(-keep_conf[members])[:max(1, int(fraction * len(members)))]].tolist()
        keep_idx, keep_y = keep_idx[chosen], keep_y[chosen]
    return keep_idx, torch.tensor(keep_y, dtype=torch.long)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--method", default="feat_balanced", help="Online TTA used while streaming group A")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--cbst", default="", help="Comma list of per-round CBST fractions, e.g. 0.3,0.5,0.7")
    parser.add_argument("--replay-per-class", type=int, default=500)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/foundation/continual")
    args = parser.parse_args(argv)
    torch.manual_seed(0)

    nct, crc, kat = load("nct_crc_100k"), load("crc_val"), load("kather2016")
    names = nct["label_names"]
    head, extras, val = build_head(nct, "cpu")  # same seed and split as evaluate_foundation.py
    f_nct, y_nct = nct["features"].float(), nct["labels"]
    train_idx = np.setdiff1d(np.arange(len(y_nct)), val)
    groups = np.array(["A" if p in GROUP_A else "B" for p in kat["patients"]])
    f_a = kat["features"][torch.from_numpy(groups == "A")].float()
    y_a = kat["labels"].numpy()[groups == "A"]
    mask_b = torch.from_numpy(groups == "B")
    f_b, y_b = kat["features"][mask_b].float(), kat["labels"].numpy()[groups == "B"]

    def val_metrics(h):
        with torch.no_grad():
            return classification_metrics(h(f_nct[val]), y_nct[val], len(names))

    def evaluate(h, round_no, accepted, n_pseudo=0, pseudo_acc=None):
        b = static(h, extras, kat, "cpu", mask_b)
        tta = [stream(h, f_b, y_b, args.method, s, False, names=names) for s in (0, 1, 2)]
        tta_class = [stream(h, f_b, y_b, args.method, s, True, names=names) for s in (0, 1, 2)]
        row = {"round": round_no, "accepted": accepted, "pseudo_labels": n_pseudo, "pseudo_label_accuracy": pseudo_acc,
               "B_static_acc": b["accuracy"], "B_static_acc_harmonised": b["accuracy_harmonised"],
               "B_static_mean_recall": b["mean_recall_present"], "B_recall": b["per_class_recall"],
               "B_tta_iid_acc": round(float(np.mean([r["accuracy"] for r in tta])), 4),
               "B_tta_iid_acc_harmonised": round(float(np.mean([r["accuracy_harmonised"] for r in tta])), 4),
               "B_tta_class_acc": round(float(np.mean([r["accuracy"] for r in tta_class])), 4),
               "B_tta_class_acc_harmonised": round(float(np.mean([r["accuracy_harmonised"] for r in tta_class])), 4),
               "crc_val_acc": static(h, extras, crc, "cpu")["accuracy"],
               "nct_val_macro_f1": round(val_metrics(h)["macro_f1"], 4)}
        print(json.dumps({k: v for k, v in row.items() if k != "B_recall"}), flush=True)
        return row

    history = [evaluate(head, 0, True)]
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for r in range(1, args.rounds + 1):
        fractions = [float(x) for x in args.cbst.split(",")] if args.cbst else None
        idx, y_pseudo = pseudo_labels(head, extras, f_a, seed=100 + r, method=args.method,
                                      min_confidence=args.min_confidence,
                                      fraction=fractions[min(r, len(fractions)) - 1] if fractions else None)
        pseudo_acc = round(float((y_pseudo.numpy() == y_a[idx]).mean()), 4) if len(idx) else None
        replay = np.concatenate([rng.choice(train_idx[y_nct[train_idx].numpy() == c],
                                            args.replay_per_class, replace=False) for c in range(len(names))])
        feats = torch.cat([f_nct[replay], f_a[idx]])
        labels = torch.cat([y_nct[replay], y_pseudo])
        candidate = train_head(feats, labels, len(names), epochs=args.epochs, lr=args.lr,
                               init=copy.deepcopy(head), teacher=head, kd_weight=args.kd_weight, seed=r)
        ok, reasons = passes_gate(val_metrics(candidate), val_metrics(head))
        if ok:
            head = candidate
            torch.save(head.state_dict(), args.out / f"head_round{r}.pt")
        history.append(evaluate(head, r, ok, len(idx), pseudo_acc))
        if not ok:
            print(f"round {r} rejected: {reasons}", flush=True)
    (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
