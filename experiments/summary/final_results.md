# CAFA-TTA — Summary of Final Results (PathMNIST, synthetic colour shift)

> This file replaces an older summary that described CAFA-TTA **v2**.
> The numbers below are copied from the main `README.md` (CAFA-TTA **v3**); they are
> not regenerated here. For the full tables, statistics and caveats see `README.md`,
> `docs/EXPERIMENTS.md` and `experiments/research_outputs/`.

**Dataset:** PathMNIST test set (7,180 images)
**Shift:** synthetic colour transformation (strong shift: brightness 0.75, contrast 1.35, saturation 0.7)

## Mean accuracy under strong colour shift (per source model)

| Method | Enhanced | Robust | Balanced |
|---|---:|---:|---:|
| Source-only | 17.84% | 81.63% | 78.22% |
| BN-only | 70.25% | 78.70% | 77.27% |
| TENT | 71.03% | 79.97% | 78.52% |
| CAFA v3 | 71.12% | 80.01% | 78.60% |

## Paired CAFA v3 vs TENT (117 PathMNIST comparisons)

- Mean accuracy gain: +0.0621 pp; mean macro-F1 gain: +0.0448 pp.
- Accuracy wins / ties / losses: 95 / 11 / 11.
- Holm-adjusted p < 0.05 in only 1 of 117 comparisons.

## Notes

- Target labels are used only for evaluation, not for test-time parameter updates.
- The PathMNIST test set was also accessed during earlier method development, so
  the final numbers are not a fully untouched hold-out (see Limitations in `README.md`).
- The shift is simulated, not a real cross-hospital shift.
- Gains over TENT are small; the Robust source-only model beats every adaptation
  method under strong shift. CAFA does not remove negative adaptation.
- Unified 17-class model results: see "Unified 17-Class ResNet18" in `README.md`.
