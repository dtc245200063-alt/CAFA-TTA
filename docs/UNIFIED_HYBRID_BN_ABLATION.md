# Unified 17-class Hybrid-BN ablation

## Evaluation protocol

- Checkpoint: `cafa_unified_balanced_best.pt`
- Prototypes: `unified_17_prototypes_balanced.pt`
- Checkpoint seed: 42; evaluation batch size: 32
- Full test: 7,180 PathMNIST and 760 Kather images
- Hybrid-BN target-statistics weight: PathMNIST 0.75, Kather 0.25
- CAFA adapts on unlabeled target batches before predicting them

## Accuracy results

| Condition | Domain | Source | BN-only | Hybrid-BN no adaptation | CAFA Hybrid-BN | CAFA delta (pp) |
|---|---|---:|---:|---:|---:|---:|
| Clean | pathmnist | 84.18% | 56.62% | 61.24% | 62.26% | +1.0167 |
| Clean | kather | 88.29% | 12.76% | 47.76% | 48.68% | +0.9211 |

Source JSON SHA256 (Clean): `a1f85910a7a47d65a1dd29919b5ab73864ee3334b7bd3e464462950e10aec90d`

| Shifted | pathmnist | 2.9526% | 51.4345% | 54.5265% | 55.6407% | +1.1142 |
| Shifted | kather | 28.2895% | 13.5526% | 38.6842% | 38.1579% | -0.5263 |

Source JSON SHA256 (Shifted): `f4f800e5114dda9291a9cf255d504eb1081c79aacecc5f9362fad9f63365a1a7`

## Interpretation

CAFA is compared against the same Hybrid-BN formulation without
online optimization. This isolates the observed contribution of
CAFA adaptation at the classification-metric level.

The result is not uniformly positive across domains.
Accuracy improvements alone do not establish statistical
significance or generalization across independent training seeds.

Exact McNemar testing has now been completed using paired
correctness counts from the same test samples. The published
JSON files contain the four correctness counts and exact p-value.
Per-sample prediction arrays are not required to recompute the
exact McNemar p-value from these counts.

The archived statistical bootstrap concerns a separate paired
evaluation and must not be presented as significance testing
for this new Hybrid-BN ablation.

## Paired McNemar significance audit

Exact two-sided McNemar tests compare CAFA Hybrid-BN
with non-adaptive Hybrid-BN on the same test samples.
Holm correction is applied across all four comparisons.

| Condition | Domain | CAFA delta (pp) | Hybrid-only correct | CAFA-only correct | Exact p | Holm p | Significant |
|---|---|---:|---:|---:|---:|---:|---|
| clean | pathmnist | +1.0167 | 40 | 113 | 2.94681723e-09 | 8.84045168e-09 | Yes |
| clean | kather | +0.9211 | 1 | 8 | 0.0390625 | 0.078125 | No |
| shifted | pathmnist | +1.1142 | 29 | 109 | 4.20155793e-12 | 1.68062317e-11 | Yes |
| shifted | kather | -0.5263 | 4 | 0 | 0.125 | 0.125 | No |

## Statistical limitations

- Results use one trained checkpoint (seed 42).
- Significance is conditional on this checkpoint, dataset,
  sample ordering, and evaluation protocol.
- PathMNIST is significant in both conditions after Holm.
- Kather is not significant in either condition.
- This analysis does not establish robustness across
  independently trained checkpoints or external institutions.
