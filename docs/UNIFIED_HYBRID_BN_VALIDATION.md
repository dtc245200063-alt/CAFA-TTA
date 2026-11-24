# Unified Hybrid-BN validation sensitivity

## Protocol

- Unified 17-class checkpoint trained with seed 42.
- Shifted validation data, 256 samples per domain.
- Evaluation seeds: 42, 123, 2026.
- Batch size: 32.
- Target BN weight: PathMNIST 0.75, Kather 0.25.
- These are NOT independently trained checkpoints.

## Results

| Evaluation seed | Domain | Hybrid-BN accuracy | CAFA accuracy | CAFA delta (pp) | Exact McNemar p |
|---|---|---:|---:|---:|---:|
| 42 | pathmnist | 61.7188% | 61.3281% | -0.3906 | 1 |
| 42 | kather | 45.3125% | 45.3125% | +0.0000 | 1 |
| 123 | pathmnist | 68.3594% | 68.3594% | +0.0000 | 1 |
| 123 | kather | 38.2812% | 38.2812% | +0.0000 | 1 |
| 2026 | pathmnist | 62.1094% | 62.1094% | +0.0000 | 1 |
| 2026 | kather | 38.2812% | 38.2812% | +0.0000 | 1 |

## Interpretation

CAFA did not improve accuracy over non-adaptive Hybrid-BN
in these six 256-image shifted validation evaluations.
One PathMNIST evaluation lost one correct prediction.

The same trained checkpoint is used throughout.
Evaluation seeds may affect sampling and random state.
The results do not measure between-training-seed variance.

Full-test findings are documented separately in
`docs/UNIFIED_HYBRID_BN_ABLATION.md`.

Independent training runs and external-domain validation
remain future research requirements.
