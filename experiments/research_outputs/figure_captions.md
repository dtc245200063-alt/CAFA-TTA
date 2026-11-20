# Research Figure Captions

## Figure 1

Accuracy and Macro-F1 under strong synthetic
color shift. Error bars show sample standard
deviations over three stream orders, not
confidence intervals.

## Figure 2

Performance under original, light, medium,
and strong color-shift conditions. Source-only
is evaluated once for non-strong conditions.
Zero error bars on single-run results do not
represent measured uncertainty.

## Figure 3

Mean paired CAFA v3 improvements relative to
TENT under Gaussian noise, Gaussian blur,
and hue shift.

## Figure 4

Validation ablation results relative to the
entropy baseline. Alignment denotes the
alignment-enabled entropy baseline, not an
alignment-only objective. Values are read from
the original full-precision validation ablation
JSON and verified against individual runs.

## Figure 5

Strong-shift test Accuracy compared with
separately measured validation GPU latency.
Timing values are read directly from the verified
benchmark JSON.

## Figure 6

Conditional 95% common-image bootstrap
intervals. Estimates are conditional on
saved adaptation trajectories.

## Figure 7

Discordant correctness outcomes and
Accuracy wins, ties, and losses across
117 paired comparisons.

## Figure 8

Mean per-class Recall and F1 differences.

## Figure 9

Performance differences between class groups
defined by sample support relative to the mean
test class support.

## Figures 10 and 11

Confusion matrices normalized by true class.
Counts are aggregated across repeated
prediction events.

## Figure 12

CAFA minus TENT row-normalized confusion
matrix in percentage points. Positive
off-diagonal entries indicate increased
misclassification frequency.

## Figure 13

Negative adaptation relative to source-only
inference, averaged over stream orders.

## Class abbreviations

- ADI: adipose
- BACK: background
- DEB: debris
- LYM: lymphocytes
- MUC: mucus
- MUS: smooth muscle
- NORM: normal colon mucosa
- STR: cancer-associated stroma
- TUM: colorectal adenocarcinoma epithelium
