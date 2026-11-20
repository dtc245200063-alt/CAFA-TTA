

# Experimental Design and Results



## CAFA-TTA



**Confidence-Aware Feature Alignment for Test-Time

Adaptation in Cross-Domain Medical Image Classification**



Last updated: 2026-11-19.



---



## 1. Research Objective



CAFA-TTA investigates whether confidence-aware

feature alignment can improve unsupervised

test-time adaptation for medical image classification.



The proposed method combines:



- Confidence-weighted entropy minimization.

- Source class prototypes.

- Classifier-prototype agreement.

- Confidence-aware feature alignment.

- Online batch-normalization adaptation.



The primary experimental comparison is

CAFA-TTA v3 versus TENT.



Source-only and BN-only methods are included

to evaluate absolute adaptation effectiveness.



The study examines both positive adaptation

and negative adaptation under domain shift.



---



## 2. Research Questions



The experimental framework addresses

the following research questions:



### RQ1: Adaptation Effectiveness



Does CAFA-TTA improve classification performance

relative to conventional entropy minimization?



### RQ2: Domain-Shift Robustness



Are the improvements consistent across

different corruption types and severity levels?



### RQ3: Source Model Dependency



How does source model training affect

test-time adaptation performance?



### RQ4: Component Contribution



What are the contributions of confidence-weighted

entropy and source-prototype alignment?



### RQ5: External Generalization



Does CAFA-TTA retain its effectiveness

on a separate histopathology dataset?



### RQ6: Computational Efficiency



What computational overhead is introduced

by confidence-aware feature alignment?



### RQ7: Statistical Consistency



Are improvements consistent across

paired experimental conditions and random seeds?



### RQ8: Negative Adaptation



Can CAFA-TTA reduce prediction errors without

eliminating or concealing harmful adaptation?



---



## 3. Experimental Datasets



| Property | PathMNIST | STARC-9 |
|---|---|---|
| Image modality | Histopathology | Histopathology |
| Research role | Training and evaluation | External evaluation |
| Input resolution | 28 x 28 | Resized to 28 x 28 |
| Source output classes | 9 | 9 |
| Evaluated target classes | 9 | 6 |
| Final evaluation samples | 7,180 | 12,000 |
| Synthetic corruptions | Yes | No |



PathMNIST provides 89,996 training images,

10,004 validation images, and 7,180 test images.



STARC-9 provides an additional external

cross-domain evaluation subset.



The selected STARC-9 subset contains

six approximately corresponding tissue categories.



The classifier retains all nine source outputs.



Refer to DATASET_LICENSES.md for source attribution,

dataset preparation, and licensing information.



---



## 4. Experimental Environment



| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| GPU | NVIDIA RTX 3050 Laptop GPU |
| GPU memory | 4 GB |
| PyTorch | 2.10.0+cu128 |
| Torchvision | 0.25.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 9.10.2 |
| Batch size | 64 |



The original software dependency audit

confirmed 39 matching package versions

and no broken dependencies.



The complete software environment is described

in REPRODUCIBILITY.md.



---



## 5. Source Model Experiments



Four related ResNet-18 models were developed.



| Model | Training Epochs | Validation Accuracy |
|---|---:|---:|
| Baseline | 3 | 84.73% |
| Enhanced | 15 | 95.40% |
| Robust | 8 | 89.57% |
| Balanced | 6 | 93.11% |



The final adaptation experiments use

the Enhanced, Robust, and Balanced models.



The Enhanced model is initialized

from the Baseline checkpoint.



The Robust and Balanced models are separately

fine-tuned from the Enhanced checkpoint.



These models represent related source training

strategies rather than independent training seeds.



Training implementations:



- scripts/train_enhanced.py

- scripts/train_robust.py

- scripts/train_balanced.py



Associated experiment directories:



- experiments/enhanced_training/

- experiments/robust_training/

- experiments/balanced_training/



All eight original model and prototype artifacts

passed SHA-256 integrity verification.



See CHECKPOINTS.md for further details.



---



## 6. Baseline Methods



### 6.1 Source-only



The source classifier is evaluated

without target-domain parameter adaptation.



This method provides the reference

for determining positive or negative adaptation.



### 6.2 BN-only



Batch-normalization behavior is adapted

using target batch statistics.



No entropy or feature-alignment

optimization is performed.



### 6.3 TENT



Test-time entropy minimization adapts

batch-normalization affine parameters

using unlabeled target samples.



Implementation:



kernel/tent.py



TENT is the primary comparison method

for measuring the incremental contribution

of CAFA-TTA.



---



## 7. CAFA-TTA Development



### 7.1 CAFA-TTA v1



The original method combines entropy minimization

with confidence-aware source-prototype alignment.



A confidence threshold selects target samples

for feature alignment.



Implementation:



core/cafa_adapter.py



### 7.2 CAFA-TTA v2



The second version introduces agreement

between classifier predictions and

nearest source prototypes.



Alignment is restricted by confidence

and classifier-prototype agreement.



Implementation:



core/cafa_agreement_adapter.py



### 7.3 CAFA-TTA v3



The final method uses soft confidence weighting

rather than hard confidence rejection

for the entropy objective.



Classifier-prototype agreement controls

the feature-alignment objective.



Implementation:



core/cafa_soft_adapter.py



The final adaptation objective is:



    L_total = L_weighted_entropy

              + lambda_align \* L_alignment



Confidence is derived from normalized

prediction entropy.



The entropy weight is:



    w_entropy = entropy_floor

                + (1 - entropy_floor) \* confidence



Feature alignment uses confidence weights

and classifier-prototype agreement.



Alignment is skipped when fewer than

the configured minimum number of samples

satisfy the agreement condition.



Only batch-normalization affine parameters

are optimized during adaptation.



---



## 8. Frozen Final Configuration



| Parameter | Value |
|---|---|
| Algorithm | CAFA-TTA v3 |
| Backbone | ResNet-18 |
| Feature dimension | 512 |
| Source classes | 9 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Entropy floor | 0.2 |
| Alignment weight | 0.05 |
| Minimum agreement samples | 2 |
| Adaptation seeds | 42, 123, 2026 |
| Protocol | Online batch-prequential |



Authoritative configuration:



experiments/reproducibility/final_config.json



The final configuration was frozen

before the external STARC-9 evaluation.



The method was developed using earlier

PathMNIST experiments.



The PathMNIST test set had been accessed

during previous development stages.



The final PathMNIST results must therefore

not be interpreted as an entirely untouched

confirmatory test evaluation.



---



## 9. Source Prototype Validation



Source prototypes are constructed from

labeled PathMNIST training features.



Each source model uses its corresponding

nine-class prototype tensor.



The prototype experiments investigate:



- Source feature representation quality.

- Target feature alignment.

- Nearest-prototype classification.

- Classifier-prototype agreement.

- Batch-normalization effects.



Prototype extraction:



scripts/extract_prototypes_models.py



Prototype validation:



scripts/validate_prototypes_models.py



Associated results:



experiments/prototype_validation_models/



These experiments provide empirical motivation

for classifier-prototype agreement gating.



They do not independently establish

a causal relationship between prototype

agreement and adaptation improvement.



---



## 10. Hyperparameter Exploration



Development experiments examined:



- Confidence thresholds.

- Feature-alignment weights.

- Prototype agreement.

- Entropy weighting.

- Source model dependency.



Relevant implementations:



- scripts/run_threshold_sweep.py

- scripts/run_alignment_sweep.py

- scripts/run_ablation_models.py



Associated directories:



- experiments/threshold_sweep/

- experiments/alignment_sweep/

- experiments/ablation_models/



These development experiments are distinct

from the frozen final evaluation.



Their results should not be treated

as independent confirmatory evidence.



---



## 11. Pseudo-Label Analysis



Pseudo-label analysis examines the relationship

between classifier confidence, prediction correctness,

and prototype agreement.



The analysis investigates:



- Pseudo-label precision.

- Confidence threshold coverage.

- Classifier-prototype agreement.

- Rejected correct predictions.

- Potential alignment errors.



Implementation:



scripts/analyze_pseudo_labels.py



Results:



experiments/pseudo_label_analysis/



The findings motivate retaining a nonzero

entropy weight for lower-confidence samples.



However, the observed relationships

do not establish that confidence weighting

will improve every target-domain condition.



---



## 12. Validation Robustness Experiments



The final CAFA-TTA v3 configuration

was evaluated under shuffled strong color shift.



Three source models and three adaptation seeds

were considered.



| Model | TENT Accuracy | CAFA v3 Accuracy |
|---|---:|---:|
| Enhanced | 83.81% | 83.86% |
| Robust | 81.49% | 81.58% |
| Balanced | 83.10% | 83.19% |



Across nine paired validation comparisons:



- Accuracy wins: 9.

- Accuracy ties: 0.

- Accuracy losses: 0.

- Mean accuracy improvement: approximately 0.075 pp.

- Mean macro-F1 improvement: approximately 0.068 pp.



Implementation:



scripts/evaluate_cafa_v3_robustness.py



Results:



experiments/cafa_v3_robustness/



These validation results supported

the final method configuration.



Their role is developmental rather than

independent confirmatory evaluation.



---



## 13. Ablation Study



A factorial ablation study examines

two objective components:



1. Confidence-weighted entropy minimization.

2. Source-prototype feature alignment.



| Variant | Entropy Floor | Alignment Weight |
|---|---:|---:|
| Entropy baseline | 1.0 | 0.00 |
| Soft entropy | 0.2 | 0.00 |
| Alignment-enabled entropy | 1.0 | 0.05 |
| Full CAFA v3 | 0.2 | 0.05 |



The experiment contains:



- Four objective configurations.

- Three source models.

- Three adaptation seeds.

- 36 experimental runs.



Approximate mean validation accuracy:



| Model | Entropy Baseline | Soft Entropy | Alignment-Enabled | Full CAFA v3 |
|---|---:|---:|---:|---:|
| Enhanced | 83.82% | 83.83% | 83.84% | 83.86% |
| Robust | 81.50% | 81.56% | 81.53% | 81.58% |
| Balanced | 83.10% | 83.14% | 83.13% | 83.19% |



Both components contribute small

performance improvements in the evaluated settings.



Their contributions are not uniformly additive.



The alignment-enabled variant retains

an entropy objective and must not

be described as pure alignment-only adaptation.



Implementation:



scripts/evaluate_cafa_v3_ablation.py



Primary results:



experiments/cafa_v3_ablation/metrics.json



Verified research table:



experiments/research_outputs/tables/table04_ablation_verified.csv



---



## 14. Controlled Domain-Shift Evaluation



### 14.1 Color Shift



| Severity | Brightness | Contrast | Saturation |
|---|---:|---:|---:|
| Original | 1.00 | 1.00 | 1.00 |
| Light | 0.95 | 1.05 | 0.95 |
| Medium | 0.85 | 1.15 | 0.85 |
| Strong | 0.75 | 1.35 | 0.70 |



### 14.2 Additional Corruptions



| Corruption | Light | Medium | Strong |
|---|---:|---:|---:|
| Gaussian noise std | 0.02 | 0.05 | 0.10 |
| Gaussian blur sigma | 0.5 | 1.0 | 2.0 |
| Hue shift | 0.03 | 0.07 | 0.15 |



The final corruption evaluator uses

a Gaussian blur kernel size of 13.



Gaussian noise is generated using

deterministic sample-specific randomization.



The exact transformation implementation

is defined by the corresponding evaluator.



Earlier validation experiments and final

test experiments should not be combined

without checking their execution protocols.



---



## 15. Final PathMNIST Evaluation



The final evaluation contains three

complementary experiment groups.



| Experiment | Unique Prediction Files |
|---|---:|
| Strong color shift | 63 |
| Severity evaluation | 90 |
| Multiple corruptions | 270 |
| Total | 423 |



All 423 unique final prediction files

passed the original integrity verification.



The evaluation uses a batch size of 64

and shuffled target sample streams.



Ground-truth labels are excluded

from adaptation optimization.



Predictions are obtained before the parameter

update associated with the current batch.



Target batch statistics may depend on

the other samples in that batch.



Consequently, the protocol is batch-prequential,

not strictly individual-sample causal.



---



## 16. Final Strong-Shift Results



Mean classification accuracy under

strong PathMNIST color shift:



| Method | Enhanced | Robust | Balanced |
|---|---:|---:|---:|
| Source-only | 17.84% | 81.63% | 78.22% |
| BN-only | 70.25% | 78.70% | 77.27% |
| TENT | 71.03% | 79.97% | 78.52% |
| CAFA v1 | 71.03% | 79.93% | 78.69% |
| Alignment variant | 71.07% | 79.97% | 78.56% |
| CAFA v2 | 71.15% | 79.91% | 78.71% |
| CAFA v3 | 71.12% | 80.01% | 78.60% |



The alignment variant is named according

to its historical experiment configuration.



It retains an entropy objective.



Mean macro-F1:



| Method | Enhanced | Robust | Balanced |
|---|---:|---:|---:|
| Source-only | 9.50% | 76.17% | 72.37% |
| BN-only | 65.22% | 74.21% | 72.53% |
| TENT | 65.86% | 75.41% | 73.58% |
| CAFA v3 | 65.94% | 75.42% | 73.64% |



CAFA v3 provides incremental improvements

over TENT in these aggregated comparisons.



However, CAFA v3 is not the highest-accuracy

adaptation variant for every source model.



The Robust source-only classifier also

outperforms the adaptation methods

in this strong-shift condition.



Implementation:



scripts/evaluate_cafa_final_test.py



Results:



experiments/cafa_final_test_strong/metrics.json



---



## 17. Final Severity Evaluation



Four color-shift levels are evaluated

across three source models.



| Severity | Mean CAFA v3 Accuracy Gain vs TENT |
|---|---:|
| Original | +0.032 pp |
| Light | +0.036 pp |
| Medium | +0.082 pp |
| Strong | +0.071 pp |



Across 36 paired comparisons:



- Accuracy wins: 27.

- Accuracy ties: 7.

- Accuracy losses: 2.

- Mean accuracy gain: +0.0553 pp.

- Mean macro-F1 gain: +0.0429 pp.



The severity evaluator reuses compatible

strong-shift results from the previous experiment.



Repeated strong-shift records are not

counted twice in the final paired analysis.



Implementation:



scripts/evaluate_cafa_final_test_severity.py



Results:



experiments/cafa_final_test_severity/metrics.json



---



## 18. Final Multi-Corruption Evaluation



Three corruption families and three

severity levels are evaluated.



| Corruption | Paired Comparisons | Mean Accuracy Gain vs TENT |
|---|---:|---:|
| Gaussian noise | 27 | +0.0830 pp |
| Gaussian blur | 27 | +0.0763 pp |
| Hue shift | 27 | +0.0361 pp |



Across 81 paired comparisons:



- Accuracy wins: 68.

- Accuracy ties: 4.

- Accuracy losses: 9.

- Mean accuracy gain: +0.0652 pp.

- Mean macro-F1 gain: +0.0457 pp.



Hue shift produces less consistent improvements

than the other evaluated corruption families.



Implementation:



scripts/evaluate_cafa_final_test_corruptions.py



Results:



experiments/cafa_final_test_corruptions/metrics.json



---



## 19. Overall PathMNIST Paired Results



The final paired analysis combines

compatible severity and corruption results.



Repeated strong-shift comparisons

are excluded from aggregation.



| Metric | Result |
|---|---:|
| Paired comparisons | 117 |
| Accuracy wins | 95 |
| Accuracy ties | 11 |
| Accuracy losses | 11 |
| Mean accuracy gain | +0.0621 pp |
| Mean macro-F1 gain | +0.0448 pp |
| Corrected predictions | 812 |
| Harmed predictions | 290 |
| Net corrected predictions | 522 |



Corrected predictions refer to samples

that are classified incorrectly by TENT

and correctly by CAFA v3.



Harmed predictions refer to samples

that are classified correctly by TENT

and incorrectly by CAFA v3.



The corrected-to-harmed prediction ratio is 2.8.



These counts are aggregated across experimental runs.



They do not represent unique independent patients

or unique underlying image samples.



The observed average accuracy gain is small

in absolute percentage-point terms.



---



## 20. PathMNIST Statistical Analysis



The statistical analysis uses:



- Exact paired McNemar tests.

- Holm multiple-comparison correction.

- Conditional image-index bootstrap.

- Circular block bootstrap.

- Leave-one-group-out sensitivity analysis.



| Statistical Result | Value |
|---|---:|
| Paired tests | 117 |
| Unadjusted p-values below 0.05 | 31 |
| Holm-adjusted p-values below 0.05 | 1 |
| Bootstrap repetitions | 10,000 |



The overall conditional 95% bootstrap interval

for mean accuracy improvement is approximately:



    \[+0.0539, +0.0705] percentage points



This interval is conditional on the saved

adaptation trajectories and evaluation samples.



It is not a patient-level confidence interval.



The analysis does not assume that all

117 experimental comparisons are independent.



Multiple evaluation conditions reuse

the same underlying image samples.



The statistical analysis is exploratory.



The results should not be presented

as definitive confirmatory evidence.



Implementations:



- scripts/verify_final_test_predictions.py

- scripts/analyze_statistical_significance.py



Results:



experiments/statistical_analysis/



---



## 21. PathMNIST Per-Class Performance



The class-level analysis investigates

whether aggregate improvements are

distributed consistently across tissue classes.



Compared with TENT:



- Recall improves in six of nine classes.

- Macro class F1 improves in seven of nine classes.

- Some classes experience recall degradation.



Most net corrected predictions are associated

with the higher-support evaluation classes.



The majority-class group accounts for

507 of the 522 net corrected predictions.



The minority-class group accounts for

15 net corrected predictions.



The grouping is based on evaluation-set

class support rather than clinical prevalence.



These findings do not establish that

CAFA-TTA improves rare-disease recognition

or clinical fairness.



Implementation:



scripts/analyze_per_class_performance.py



Results:



experiments/statistical_analysis/per_class/



---



## 22. Negative Adaptation



Negative adaptation occurs when an adaptation

method performs worse than the source-only classifier.



Across 39 PathMNIST model-condition groups,

using mean results across the available seeds:



| Method | Negative Accuracy Groups | Negative Macro-F1 Groups |
|---|---:|---:|
| BN-only | 15 | 14 |
| TENT | 15 | 14 |
| CAFA v3 | 15 | 14 |



CAFA v3 does not eliminate negative adaptation.



The source-only classifier outperforms

adaptation methods under several conditions,

including the original uncorrupted PathMNIST

test condition.



Improvements over TENT should therefore

not be confused with guaranteed improvements

over the original source model.



The choice to adapt a model remains

an important research problem.



---



## 23. Computational Efficiency



The computational benchmark evaluates:



- Source-only.

- BN-only.

- TENT.

- CAFA-TTA v3.



Benchmark configuration:



| Parameter | Value |
|---|---:|
| GPU | NVIDIA RTX 3050 Laptop |
| Batch size | 64 |
| Warm-up batches | 10 |
| Measured batches | 50 |
| Timing repetitions | 3 |
| Source models | 3 |



Mean batch processing times:



| Model | TENT (ms) | CAFA v3 (ms) | Relative Overhead |
|---|---:|---:|---:|
| Enhanced | 10.637 | 14.705 | 38.25% |
| Robust | 10.515 | 14.097 | 34.07% |
| Balanced | 9.765 | 14.197 | 45.39% |



CAFA v3 introduces additional computation

associated with confidence estimation

and prototype-based feature alignment.



The additional peak allocated GPU memory

relative to TENT is approximately

0.4 to 0.52 MiB in the reference benchmark.



The trainable batch-normalization affine

parameter count is 9,600.



A float32 prototype tensor with dimensions

9 by 512 occupies 18 KiB before serialization.



The timing benchmark is conducted

on a validation workload.



Data acquisition, transfer, and preprocessing

are excluded from the reported timing.



These measurements do not represent

end-to-end clinical inference latency.



Implementation:



scripts/benchmark_cafa_v3.py



Results:



experiments/cafa_v3_benchmark/metrics.json



---



## 24. External STARC-9 Evaluation



### 24.1 External Dataset



The external evaluation uses a pinned

STARC-9 Stanford validation archive.



A six-category target subset is selected.



| Property | Value |
|---|---:|
| Selected samples | 12,000 |
| Target categories | 6 |
| Samples per category | 2,000 |
| Source output classes | 9 |
| Batch size | 64 |
| Batches per run | 188 |
| Total evaluation runs | 30 |



The external protocol is frozen

before model evaluation.



No external target-label-based

adaptation or hyperparameter tuning is performed.



External labels are used for category selection

and evaluation metrics.



The selected category mapping is approximate.



The original classifier and all nine

source prototypes remain available.



### 24.2 Evaluation Methods



The external experiment compares:



- Source-only.

- BN-only.

- TENT.

- CAFA v3.



Source-only is evaluated once per model.



The adaptation methods are evaluated

using three seeds per model.



Three related source models are considered.



### 24.3 External Classification Accuracy



| Model | Source-only | BN-only | TENT | CAFA v3 |
|---|---:|---:|---:|---:|
| Enhanced | 22.3250% | 55.0139% | 56.4472% | 56.5917% |
| Robust | 73.4000% | 71.8806% | 74.1222% | 74.2278% |
| Balanced | 72.2583% | 69.4028% | 71.6278% | 71.7111% |



### 24.4 External Six-Class Macro-F1



| Model | Source-only | BN-only | TENT | CAFA v3 |
|---|---:|---:|---:|---:|
| Enhanced | 19.4651% | 60.3133% | 61.3416% | 61.4669% |
| Robust | 76.3687% | 76.7877% | 78.2515% | 78.3286% |
| Balanced | 74.2878% | 75.1022% | 76.6929% | 76.7495% |



The Robust model achieves the highest

absolute external accuracy and macro-F1.



The substantial Enhanced model improvement

relative to source-only is largely associated

with batch normalization and entropy adaptation.



The incremental contribution of CAFA v3

must be measured relative to TENT.



The Balanced model experiences negative

accuracy adaptation relative to source-only.



### 24.5 Incremental CAFA v3 Improvements



| Model | Accuracy Gain vs TENT | Macro-F1 Gain vs TENT |
|---|---:|---:|
| Enhanced | +0.144444 pp | +0.125346 pp |
| Robust | +0.105556 pp | +0.077075 pp |
| Balanced | +0.083333 pp | +0.056611 pp |



Across nine paired comparisons:



- Accuracy wins: 9.

- Accuracy ties: 0.

- Accuracy losses: 0.

- Macro-F1 wins: 9.

- Macro-F1 ties: 0.

- Macro-F1 losses: 0.

- Mean accuracy gain: +0.111111 pp.

- Mean macro-F1 gain: +0.086344 pp.



Implementation:



scripts/evaluate_starc9_external.py



Results:



experiments/starc9_external/metrics.json



---



## 25. External Statistical Analysis



All 30 external prediction files

passed the original integrity verification.



Across nine paired TENT versus CAFA v3 comparisons:



| Metric | Value |
|---|---:|
| Corrected predictions | 141 |
| Harmed predictions | 21 |
| Net corrected predictions | 120 |
| Corrected-to-harmed ratio | 6.714 |
| Unadjusted p-values below 0.05 | 9 |
| Holm-adjusted p-values below 0.05 | 9 |



The overall conditional 95% bootstrap interval

for mean accuracy improvement is:



    \[+0.087963, +0.134259] percentage points



Circular block bootstrap sensitivity analyses

were conducted using block lengths of:



- One batch.

- Four batches.

- Eight batches.

- Sixteen batches.



The evaluated external paired comparisons

retained positive bootstrap intervals

under these block-length settings.



These analyses remain conditional

on the saved adaptation trajectories.



Patient-level and slide-level clustering

information is unavailable.



The statistical findings are exploratory,

rather than independent clinical confirmation.



---



## 26. External Duplicate Sensitivity



The selected STARC-9 subset contains:



- 49 duplicate image groups.

- 98 images involved in duplicate groups.

- 49 additional duplicate images.

- 11,951 unique resized pixel groups.

- No cross-label duplicate groups.



The primary evaluation retains

all 12,000 selected images.



Duplicate sensitivity is evaluated

using three scoring schemes.



| Scoring Scheme | Accuracy Gain vs TENT |
|---|---:|
| Original samples | +0.111111 pp |
| One representative per group | +0.111567 pp |
| Equal weight per pixel group | +0.111567 pp |



The observed incremental improvement

is retained under these scoring alternatives.



Adaptation is not rerun after removing duplicates.



Consequently, this analysis evaluates

scoring sensitivity rather than the effects

of duplicate removal on adaptation dynamics.



---



## 27. External Per-Class Analysis



| Target Class | Mean Recall Gain | Mean F1 Gain |
|---|---:|---:|
| ADI | +0.038889 pp | +0.041105 pp |
| LYM | +0.194444 pp | +0.157264 pp |
| MUC | 0.000000 pp | -0.005060 pp |
| MUS | +0.266667 pp | +0.251141 pp |
| NOR | +0.061111 pp | +0.034669 pp |
| TUM | +0.105556 pp | +0.038944 pp |



Five of six target classes improve

in both recall and F1.



Smooth muscle and lymphocyte categories

account for 83 of the 120 net corrected

predictions across the paired runs.



The results indicate heterogeneous

class-level adaptation effects.



---



## 28. External Absent-Class Predictions



The external evaluation contains

six selected target categories.



The classifier retains nine output classes.



Predictions into the three absent

source classes are counted as errors.



Compared with TENT, CAFA v3 reduces

the mean absent-class prediction rate by:



    0.092593 percentage points



The absent-class analysis is relevant

to the partial overlap between

source and target label spaces.



The evaluation does not use target labels

to restrict classifier outputs or prototypes.



---



## 29. External Negative Adaptation



Across nine model-seed comparisons:



| Method | Negative Accuracy Comparisons |
|---|---:|
| BN-only | 6 |
| TENT | 3 |
| CAFA v3 | 3 |



None of these methods produces negative

six-class macro-F1 adaptation in the

nine evaluated external comparisons.



CAFA v3 nevertheless retains negative

accuracy adaptation for some source-model

and target-stream configurations.



The method does not guarantee improvement

relative to source-only.



External statistical implementation:



scripts/analyze_starc9_external.py



Results:



experiments/starc9_external/statistical_analysis/



---



## 30. Combined Research Summary



The two final evaluation packages contain:



| Research Artifact | PathMNIST | STARC-9 | Total |
|---|---:|---:|---:|
| Unique prediction files | 423 | 30 | 453 |
| Paired comparisons | 117 | 9 | 126 |
| Accuracy wins vs TENT | 95 | 9 | 104 |
| Accuracy ties vs TENT | 11 | 0 | 11 |
| Accuracy losses vs TENT | 11 | 0 | 11 |
| Research figures | 13 | 9 | 22 |
| Selected CSV tables | 12 | 16 | 28 |



The combined counts are descriptive.



The PathMNIST and STARC-9 experiments

use different target distributions,

sample counts, and label spaces.



No pooled statistical significance test

is asserted across the two datasets.



The six-class external macro-F1

and nine-class PathMNIST macro-F1

are not directly interchangeable.



---



## 31. Research Figures



### 31.1 PathMNIST Figures



Directory:



experiments/research_outputs/figures/



| Figure | Description |
|---|---|
| fig01_final_test_comparison | Final test comparison |
| fig02_shift_severity | Domain-shift severity |
| fig03_multi_corruption | Multiple corruptions |
| fig04_ablation_study | Component ablation |
| fig05_accuracy_cost_tradeoff | Accuracy and computation |
| fig06_conditional_forest_plot | Conditional bootstrap |
| fig07_paired_prediction_outcomes | Corrected and harmed predictions |
| fig08_per_class_performance | Class-level performance |
| fig09_minority_majority | Class support groups |
| fig10_tent_confusion_matrix | TENT confusion matrix |
| fig11_cafa_confusion_matrix | CAFA confusion matrix |
| fig12_difference_confusion_matrix | Confusion matrix differences |
| fig13_negative_adaptation | Negative adaptation |



### 31.2 STARC-9 Figures



Directory:



experiments/starc9_external/statistical_analysis/figures/



| Figure | Description |
|---|---|
| fig01_external_performance | External classification results |
| fig02_paired_effects | Paired adaptation effects |
| fig03_conditional_forest | Conditional bootstrap |
| fig04_paired_outcomes | Corrected and harmed predictions |
| fig05_duplicate_sensitivity | Duplicate sensitivity |
| fig06_per_class_effects | Class-level effects |
| fig07_negative_adaptation | Negative adaptation |
| fig08_absent_class_predictions | Absent-class outputs |
| fig09_confusion_difference | Confusion differences |



All 22 figures are available in the original

research package in PNG and PDF formats.



---



## 32. Experimental Script Index



| Research Stage | Implementation |
|---|---|
| Enhanced training | scripts/train_enhanced.py |
| Robust training | scripts/train_robust.py |
| Balanced training | scripts/train_balanced.py |
| Prototype extraction | scripts/extract_prototypes_models.py |
| Prototype validation | scripts/validate_prototypes_models.py |
| Threshold exploration | scripts/run_threshold_sweep.py |
| Alignment exploration | scripts/run_alignment_sweep.py |
| Development ablation | scripts/run_ablation_models.py |
| Pseudo-label analysis | scripts/analyze_pseudo_labels.py |
| CAFA v3 evaluation | scripts/evaluate_cafa_v3_models.py |
| CAFA v3 robustness | scripts/evaluate_cafa_v3_robustness.py |
| CAFA v3 ablation | scripts/evaluate_cafa_v3_ablation.py |
| Validation severity | scripts/evaluate_cafa_v3_severity.py |
| Validation corruptions | scripts/evaluate_cafa_v3_corruptions.py |
| Efficiency benchmark | scripts/benchmark_cafa_v3.py |
| Final strong shift | scripts/evaluate_cafa_final_test.py |
| Final severity | scripts/evaluate_cafa_final_test_severity.py |
| Final corruptions | scripts/evaluate_cafa_final_test_corruptions.py |
| Prediction verification | scripts/verify_final_test_predictions.py |
| Statistical analysis | scripts/analyze_statistical_significance.py |
| Per-class analysis | scripts/analyze_per_class_performance.py |
| Research figures | scripts/generate_research_figures.py |
| External data preparation | scripts/prepare_starc9_dataset.py |
| External evaluation | scripts/evaluate_starc9_external.py |
| External statistical analysis | scripts/analyze_starc9_external.py |



Refer to the corresponding source files

for implementation-level configurations.



The repository also contains historical

development and comparison scripts.



Historical experiment names should be interpreted

using their actual objective implementations.



---



## 33. Research Integrity



The original experimental artifacts

passed the following integrity audits.



| Audit | Verified |
|---|---:|
| Model and prototype files | 8/8 |
| PathMNIST final predictions | 423/423 |
| STARC-9 final predictions | 30/30 |
| PathMNIST research manifest inputs | 15/15 |
| PathMNIST research manifest outputs | 42/42 |
| STARC-9 statistical manifest inputs | 35/35 |
| STARC-9 statistical manifest outputs | 36/36 |



The final prediction content audit confirmed

that all 453 NPZ archives contain only

the expected integer-valued arrays.



The original experiment manifests are:



- experiments/research_outputs/research_manifest.json

- experiments/starc9_external/statistical_analysis/statistical_manifest.json



These integrity results apply to

the original local research artifacts.



They do not constitute certification

of a new repository checkout.



Artifact availability, redistribution rights,

and clean-environment reproduction

remain distinct requirements.



---



## 34. Principal Findings



The experiments support the following

bounded research conclusions.



### Finding 1



CAFA-TTA v3 produces small positive

average improvements over TENT

in the evaluated PathMNIST conditions.



### Finding 2



The incremental improvement is retained

on the selected external STARC-9 subset.



### Finding 3



The source model training strategy

substantially affects absolute

target-domain performance.



### Finding 4



Confidence-weighted entropy and prototype

alignment contribute to the observed

validation results, although their

interaction is not uniformly additive.



### Finding 5



The proposed method introduces additional

runtime overhead compared with TENT.



### Finding 6



Class-level improvements are heterogeneous

and may favor higher-support classes.



### Finding 7



CAFA-TTA does not eliminate negative adaptation

relative to the source-only classifier.



### Finding 8



The current evidence supports further

methodological investigation rather than

clinical deployment claims.



---



## 35. Limitations



The following limitations are essential

when interpreting the experimental results.



1. The absolute mean improvements over

   TENT are small.



2. The PathMNIST test set was accessed

   during earlier development experiments.



3. Multiple experimental conditions reuse

   the same underlying image samples.



4. The source models share a related

   training lineage.



5. Three adaptation seeds do not capture

   all possible training and stream variability.



6. The statistical analyses are exploratory.



7. Bootstrap intervals are conditional

   on the saved adaptation trajectories.



8. Patient-level and slide-level clustering

   information is unavailable.



9. STARC-9 uses a selected six-class subset

   with an approximate semantic label mapping.



10. Exact duplicate images occur in the

    selected external dataset.



11. Resizing to 28 x 28 removes substantial

    histopathological detail.



12. The external dataset is stain-normalized,

    limiting the scope of external domain shift.



13. Test-time adaptation can reduce performance

    relative to source-only inference.



14. Computational benchmarks are hardware

    and workload dependent.



15. Additional adaptation baselines, model

    architectures, and medical image modalities

    remain to be evaluated.



16. Public availability of original checkpoints,

    predictions, and required dataset metadata

    remains subject to separate review.



17. Independent clean-environment reproduction

    has not yet been completed.



The project does not establish clinical safety,

patient-level diagnosis, or diagnostic effectiveness.



---



## 36. Future Research Directions



Potential extensions include:



- Negative-adaptation detection.

- Adaptive alignment strength.

- Uncertainty-aware prototype selection.

- Class-balanced adaptation.

- Continual and nonstationary target streams.

- Additional test-time adaptation baselines.

- Evaluation with larger image resolutions.

- Additional source architectures.

- Patient-level and slide-level evaluation.

- Independent external validation.

- Prospective and confirmatory experimental design.

- Improved computational efficiency.

- Independent reproduction of published artifacts.



These directions are proposed future work.



They are not presented as completed experiments.



---



## 37. Related Documentation



- \[Dataset Licenses](DATASET_LICENSES.md)

- \[Reproducibility Guide](REPRODUCIBILITY.md)

- \[Model Checkpoints](CHECKPOINTS.md)

- \[Project README](../README.md)



Primary research findings and limitations:



- experiments/research_outputs/research_findings.md

- experiments/research_outputs/research_limitations.md

- experiments/research_outputs/future_work.md



---



## 38. Scientific Transparency Statement



CAFA-TTA is an experimental approach

to unsupervised test-time adaptation

for medical image classification.



The project preserves positive outcomes,

negative outcomes, component ablations,

computational costs, and statistical limitations.



The reported improvements are interpreted

within the evaluated datasets, model configurations,

and adaptation protocols.



No claim of universal superiority,

clinical effectiveness, or guaranteed

positive adaptation is made.



The original computational records

provide the reference for subsequent

verification and independent reproduction.
