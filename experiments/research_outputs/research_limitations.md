# Research Limitations

## 1. Small absolute effect size

The average Accuracy gain over TENT is small.
Practical relevance must be evaluated alongside
computational overhead.

## 2. Negative adaptation

Adaptation can reduce performance compared with
source-only inference when the source model is
already robust.

## 3. Uneven class-wise benefits

The gains are concentrated primarily in classes
with higher test support. Several lower-support
classes experience Recall degradation.

## 4. Statistical dependence

The same test images are reused across source
models, synthetic shifts, and stream-order seeds.

Online parameter adaptation creates dependence
between predictions within and across batches.

McNemar p-values and bootstrap intervals are
therefore exploratory.

## 5. Conditional bootstrap intervals

The bootstrap analysis resamples saved prediction
outcomes. It does not rerun adaptation under a
newly sampled sequence of images.

## 6. Multiple comparisons

Holm correction reduces the multiple-testing
problem but does not resolve dependence or
violations of marginal test assumptions.

## 7. Previously accessed test set

The PathMNIST test set was accessed during earlier
experiments. The final CAFA v3 configuration was
frozen before the reported final evaluation, but
the test set cannot be described as pristine.

## 8. Limited source model diversity

The three source models share the ResNet-18
architecture and related training history.

Stream-order seeds are not independent model
training repetitions.

## 9. Scope of domain-shift evaluation

PathMNIST includes source and test data originating
from different clinical centers.

The original test split provides a natural
cross-center benchmark.

Brightness, contrast, saturation, Gaussian noise,
Gaussian blur, and hue changes provide additional
controlled synthetic perturbations.

Evaluation on one source-target benchmark does
not establish broad generalization across hospitals,
scanners, staining protocols, patient populations,
or medical imaging modalities.

## 10. Limited baseline diversity

The principal comparisons involve source-only,
BN-only, TENT, and internal CAFA variants.

Broader comparisons against other established
test-time adaptation algorithms remain desirable.

## 11. Patient and slide-level uncertainty

The analysis does not account for patient-level
or histology-slide-level clustering.

Patch classification results are not evidence of
patient-level diagnostic accuracy.

## 12. Computational benchmark scope

GPU timings exclude preprocessing and CPU-GPU
data transfer and use one hardware configuration.

They do not represent end-to-end clinical latency.

## 13. Hyperparameter and analysis selection

The final method was selected using validation
experiments. The statistical analysis plan was
completed after descriptive results were observed.

Independent confirmatory evaluation is required.

## 14. Figure provenance

The ablation and benchmark figures use the original
full-precision experimental JSON files.

Their source hashes are recorded in the research
manifest, and aggregate values are checked against
individual experimental runs.

The accuracy-cost figure combines test Accuracy
with latency measured in a separate validation
benchmark. These measurements should not be
interpreted as a single joint evaluation.

