# CAFA-TTA: Research Findings

## Experimental setting

- Dataset: PathMNIST, 28 x 28 RGB images.
- Test images: 7,180.
- Source backbones: three ResNet-18 checkpoints.
- Models: Enhanced, Robust, and Balanced.
- Final method: CAFA-TTA v3.
- Batch size: 64.
- Learning rate: 0.0001.
- Entropy floor: 0.2.
- Alignment coefficient: 0.05.
- Stream-order seeds: 42, 123, and 2026.
- Online prediction before parameter adaptation.
- Current-batch statistics are used by target BN.
- Target labels are not used for adaptation.

## 1. Data integrity

All 423 saved prediction files passed SHA-256
verification and metric consistency checks.

The audit removed 36 duplicate strong-shift records,
leaving 117 unique TENT versus CAFA v3 comparisons.

## 2. Overall performance

CAFA v3 achieved a mean Accuracy improvement of
+0.0621 percentage points and a mean
Macro-F1 improvement of +0.0448 percentage
points over TENT.

Accuracy wins, ties, and losses were
95, 11, and 11, respectively.

These differences are descriptive averages across
the evaluated models, conditions, and stream orders.

## 3. Paired prediction outcomes

CAFA corrected 812 prediction events that
TENT classified incorrectly, while introducing
290 incorrect prediction events.

The net difference was +522 correct prediction
events, corresponding to an average of
+4.4615 additional correct predictions
per run of 7,180 images.

The corrected-to-harmed ratio was
2.8000.

These events are not unique images or patients.

## 4. Statistical analysis

The common-image conditional bootstrap produced a
95% interval of [+0.0539, +0.0705]
percentage points for the mean Accuracy difference.

Of 117 two-sided exact McNemar tests,
31 met the nominal 0.05
threshold before correction and
1 after Holm correction.

The leave-one-group-out mean Accuracy differences
ranged from
+0.0553
to
+0.0700
percentage points.

The statistical findings are exploratory because
the same images are reused and online adaptation
creates dependence within and across batches.

## 5. Robustness to synthetic corruption

Gaussian noise produced a mean Accuracy improvement
of +0.0830 percentage
points, with 27 wins across
27 comparisons.

Hue shift produced a smaller average improvement of
+0.0361 percentage
points.

The results suggest that the benefit of CAFA is
dependent on the type of distribution shift.

## 6. Class-wise performance

The largest mean Recall improvement occurred in
background
(+0.1221 pp).

The largest Recall degradation occurred in
cancer-associated stroma
(-0.1096 pp).

The majority class group accounted for
97.13% of the net improvement.

Minority and majority groups were defined by test
class support relative to mean class support,
not by clinical disease prevalence.

## 7. Negative adaptation

CAFA v3 reduced Accuracy relative to source-only
in 15
of 39 model-condition
combinations.

CAFA therefore does not eliminate negative
adaptation, even when it improves upon TENT.

## 8. Computational trade-off

The verified validation GPU benchmark showed
approximately 34.1% to
45.4% higher mean latency
for CAFA v3 relative to TENT.

The timing was measured separately on validation
batches and excludes preprocessing and data transfer.

The observed Accuracy improvement is small relative
to the additional computational cost.

Timing values are read directly from the original
benchmark JSON and checked against the individual
benchmark repetitions.

## 9. Natural cross-center evaluation

PathMNIST training and validation originate from
NCT-CRC-HE-100K, whereas its test split originates
from CRC-VAL-HE-7K, described by MedMNIST as data
from a different clinical center.

The original, uncorrupted test split therefore
provides a natural cross-center evaluation within
the PathMNIST benchmark.

The synthetic perturbation experiments evaluate
additional controlled shifts on the target data.

On the original test split, CAFA v3 achieved a
small average improvement over TENT but remained
below source-only Accuracy for all three source
models.

This does not establish generalization across
a broader collection of independent hospitals
or medical datasets.

Dataset reference: https://medmnist.com/

## 10. Overall conclusion

CAFA-TTA v3 demonstrated small but relatively
consistent empirical improvements over TENT across
the evaluated synthetic distribution shifts.

However, the improvements were uneven across
classes, negative adaptation remained unresolved,
and the statistical evidence does not establish
definitive population-level or clinical superiority.

The method should be interpreted as a modest
extension of entropy-based test-time adaptation,
not as a comprehensive solution to medical
cross-domain classification.

