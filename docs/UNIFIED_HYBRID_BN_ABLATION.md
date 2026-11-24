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
| Clean | pathmnist | 84.1783% | 56.6156% | 61.2396% | 62.2563% | +1.0167 |
| Clean | kather | 88.2895% | 12.7632% | 47.7632% | 48.6842% | +0.9211 |

Source JSON SHA256 (Clean): `3e3cf3a86ea3f65b2a35a094bbd30283405f4e926c8f0fae736be94efb0f4be7`

| Shifted | pathmnist | 2.9526% | 51.4345% | 54.5265% | 55.6407% | +1.1142 |
| Shifted | kather | 28.2895% | 13.5526% | 38.6842% | 38.1579% | -0.5263 |

Source JSON SHA256 (Shifted): `1e39e1659a961d22b794e6fc3804ae3011b6dca10acfd38fed68c1f9898de95f`

## Interpretation

CAFA is compared against the same Hybrid-BN formulation without
online optimization. This isolates the observed contribution of
CAFA adaptation at the classification-metric level.

The result is not uniformly positive across domains.
Accuracy improvements alone do not establish statistical
significance or generalization across independent training seeds.

Per-sample paired predictions are needed for exact McNemar
testing of CAFA versus non-adaptive Hybrid-BN.
The present JSON summaries do not contain those prediction arrays.

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
