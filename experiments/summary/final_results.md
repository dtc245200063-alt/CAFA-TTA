# CAFA-TTA — Final Test Results

**Dataset:** PathMNIST

**Test samples:** 7,180

**Domain shift:** Synthetic color transformation

**Shift configuration:** `{'brightness': 0.75, 'contrast': 1.35, 'saturation': 0.7}`

| Method | Accuracy | Macro-F1 | Runs |
|---|---:|---:|---:|
| Source-only | 15.01% | 6.51% | 1 |
| BN-only | 73.93% | 68.11% | 1 |
| TENT | 74.54% ± 0.10 | 68.67% ± 0.14 | 3 |
| CAFA-TTA v2 | 74.58% ± 0.13 | 68.71% ± 0.17 | 3 |

## Experimental Notes

- Source-only and BN-only use one fixed data order.
- TENT and CAFA-TTA v2 report mean ± sample standard deviation across three shuffled batch orders.
- Target labels are used only for evaluation, not for test-time parameter updates.
- Hyperparameters were selected before final test evaluation.
- The domain shift is simulated, not a real cross-hospital shift.
- CAFA-TTA v2 has not demonstrated a robust advantage over TENT.
