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
