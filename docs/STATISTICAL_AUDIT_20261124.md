# Statistical audit — 2026-11-24

## PathMNIST paired evaluation

- 117 paired comparisons: 95 wins, 11 ties, 11 losses.
- Confusion-count reconciliation: 0 errors.
- Exact McNemar: 31 unadjusted significant comparisons.
- Holm correction: 1 significant comparison.

## STARC-9

- 9 paired comparisons.
- Exact McNemar and Holm values independently checked
  against the published statistical CSV.
- 9 unadjusted and 9 Holm-significant comparisons.
- 0 detected calculation inconsistencies.

## Conditional joint moving-block bootstrap

- 117 comparisons, 13,221 batch records.
- Original batch data found in the archived project.
- 12 seed/block-length configurations checked.
- 10,000 independent Monte Carlo replicates per configuration.
- Maximum observed difference in CI endpoints: 0.00115
  percentage points.
- 0 batch reconciliation errors; 0 comparison warnings.
- All 12 conditional CIs have positive lower bounds.

## Interpretation and limitations

The bootstrap results are conditional on the archived
prediction streams, selected seeds and experimental shifts.
Overlapping comparisons of the same samples must not be
interpreted as independent observations.

On the unified 17-class evaluation, the non-adaptive
Hybrid-BN baseline matched CAFA's classification accuracy
on both shifted 256-image subsets. CAFA parameters changed,
but no classification labels differed between these methods.
This limits attribution of the observed improvement to the
CAFA adaptation mechanism.

These audits do not establish generalization to unseen
medical institutions or prove superiority of CAFA over all
non-adaptive Hybrid-BN configurations.


## Domain-shift protocol audit

The project contains distinct evaluation protocols. Their results
must not be treated as directly comparable without matching the
checkpoint, dataset split, sample selection, preprocessing and
adaptation settings.

### Unified 17-class Hybrid-BN

The unified evaluators use a deterministic RGB transformation
before normalization:

    x = clamp(0.65 * x + 0.18, 0, 1)
    x = x ** 1.15

The validation evaluator uses validation data, while the final
evaluator uses test data.

### PathMNIST 9-class BN-only

The separate PathMNIST baseline uses torchvision adjustments
for brightness, contrast and saturation, with a different
9-class checkpoint.

### Docker Kather CAFA

The Docker Kather evaluator also uses brightness, contrast
and saturation adjustments. Exact transformation parameters
and sampling protocols must be matched before making
cross-experiment accuracy comparisons.

### Alpha selection and metadata

The unified final evaluator defaults to target-statistics
weights of 0.75 for PathMNIST and 0.25 for Kather.
Some archived Hybrid-BN result files do not record alpha
in their metadata. They are retained unchanged.

### Attribution of CAFA gains

In the new 256-image shifted ablation, adaptive CAFA and
non-adaptive Hybrid-BN achieved the same accuracy for
both domains. CAFA's BatchNorm affine parameters did change,
but the final classification labels did not.

Consequently, improvements over source-only and BN-only
cannot be attributed solely to CAFA's optimization step.

### Remaining experimental limitations

- Full-test clean/shifted ablation with the non-adaptive
  Hybrid-BN baseline is complete. Three 256-image shifted
  validation evaluations also exist, but they share one
  trained checkpoint and are not independent training seeds.
- Independent external-domain generalization remains unproven.
- The archived batch-level bootstrap source is available in
  the original project but is not included in this repository.

### Updated validation interpretation

The earlier 256-image shifted ablation reported identical final
classification labels for CAFA and non-adaptive Hybrid-BN.
This statement is specific to that experiment.

In the later shifted validation sensitivity evaluation, seed 42
produced one fewer correct PathMNIST prediction with CAFA.
The other five dataset/seed accuracy comparisons were tied.
See `docs/UNIFIED_HYBRID_BN_VALIDATION.md`.

The full-test paired McNemar results are documented in
`docs/UNIFIED_HYBRID_BN_ABLATION.md`.
