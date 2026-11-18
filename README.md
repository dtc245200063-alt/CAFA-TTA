\# CAFA-TTA: Confidence-Aware Feature Alignment for Test-Time Adaptation



A research prototype for test-time adaptation in medical image classification under simulated domain shift.



\## Overview



CAFA-TTA explores whether confidence-aware sample selection and source-prototype feature alignment can improve test-time adaptation beyond entropy minimization.



The prototype is evaluated on \*\*PathMNIST\*\*, a nine-class histopathology image classification dataset. Domain shift is simulated using brightness, contrast, and saturation transformations.



\*\*Research status:\*\* Proof of concept. The experiments do not establish performance on real cross-hospital domain shifts.



\## Methods



Four principal methods are evaluated:



\- \*\*Source-only:\*\* Frozen ResNet-18 using source-domain BatchNorm statistics.

\- \*\*BN-only:\*\* Frozen ResNet-18 using target-batch BatchNorm statistics.

\- \*\*TENT:\*\* Test-time entropy minimization with updates to BatchNorm affine parameters.

\- \*\*CAFA-TTA v2:\*\* Entropy minimization combined with confidence filtering, classifier–prototype agreement, and source-prototype feature alignment.



CAFA-TTA v2 selects target samples when their prediction confidence meets a threshold and the classifier prediction agrees with the nearest source prototype.



Target labels are not used for test-time parameter updates.



\## Experimental Setup



| Setting | Value |

|---|---|

| Dataset | PathMNIST |

| Backbone | ResNet-18 |

| Number of classes | 9 |

| Image size | 28 × 28 |

| Training images | 89,996 |

| Validation images | 10,004 |

| Test images | 7,180 |

| Batch size | 64 for TTA |

| TTA learning rate | 0.0001 |

| Confidence threshold | 0.7 |

| Alignment loss weight | 0.05 |

| Shift brightness | 0.75 |

| Shift contrast | 1.35 |

| Shift saturation | 0.70 |



The evaluation follows an online \*predict-before-adapt\* protocol: predictions are recorded before updating on the current batch.



\## Final Test Results



| Method | Accuracy | Macro-F1 |

|---|---:|---:|

| Source-only | 15.01% | 6.51% |

| BN-only | 73.93% | 68.11% |

| TENT | 74.54% ± 0.10 | 68.67% ± 0.14 |

| CAFA-TTA v2 | \*\*74.58% ± 0.13\*\* | \*\*68.71% ± 0.17\*\* |



TENT and CAFA-TTA v2 results are reported as mean ± sample standard deviation across three shuffled batch orders (seeds 42, 123, and 2026). Source-only and BN-only were evaluated with one fixed data order.



CAFA-TTA v2 achieved a paired mean accuracy difference of approximately \*\*+0.042 percentage points\*\* against TENT on the final test set.



\*\*This difference is too small to establish a robust advantage over TENT.\*\* On validation, CAFA-TTA v2 performed slightly worse on average than TENT.



\## Ablation Study



Validation results under the simulated domain shift:



| Configuration | Accuracy | Macro-F1 |

|---|---:|---:|

| Entropy only | 79.86% | 80.05% |

| Confidence + Entropy | 79.78% | 79.97% |

| Entropy + Alignment | 79.82% | 80.01% |

| Full CAFA-TTA v1 | 79.79% | 79.99% |



These experiments did not show a consistent improvement from adding confidence filtering or feature alignment to entropy minimization.



\## Installation



Create and activate a Python virtual environment, then install the project's dependencies.



The main dependencies are:



\- PyTorch and torchvision

\- medmnist

\- scikit-learn

\- NumPy

\- pandas

\- matplotlib

\- tqdm

\- PyYAML



Use a PyTorch installation appropriate for the available CPU or GPU environment.



\## Reproducing Experiments



Run these commands from the repository root.



Download the dataset:



```bash

python scripts/download\_data.py

```



Train the source model:



```bash

python scripts/train\_baseline.py

```



Extract source prototypes:



```bash

python scripts/extract\_prototypes.py

```



Run domain-shift and adaptation experiments:



```bash

python scripts/evaluate\_domain\_shift.py

python scripts/evaluate\_tent.py

python scripts/evaluate\_cafa\_v2.py

```



Run ablation and robustness evaluations:



```bash

python scripts/run\_ablation.py --mode full\_cafa

python scripts/evaluate\_robustness.py

```



Summarize saved final-test results:



```bash

python scripts/summarize\_results.py

```



\*\*Reproducibility note:\*\* The training checkpoint and source prototypes are excluded from version control. Reproducing the reported numbers requires recreating the trained source model and prototypes. Retraining can produce different results.



\## Limitations



1\. Domain shifts are simulated through deterministic color transformations rather than real hospital or scanner differences.

2\. The model was trained using one source-model training run.

3\. Robustness evaluation covers three shuffled batch orders, not multiple independently trained source models.

4\. Confidence derived from entropy is not a calibrated estimate of prediction correctness.

5\. Source prototypes are extracted from labeled source training data.

6\. CAFA-TTA v2 has not demonstrated a meaningful, stable advantage over TENT.

7\. Continual learning and real cross-hospital adaptation remain future work.



\## Future Work



\- Evaluate real cross-hospital domain shifts.

\- Test multiple source-training seeds.

\- Investigate more reliable pseudo-label filtering.

\- Study sensitivity to batch size and adaptation order.

\- Explore continual test-time adaptation and catastrophic forgetting.



\## Research Conclusion



The experiments demonstrate that confidence-aware prototype alignment can be integrated into a practical test-time adaptation pipeline for medical image classification.



Most of the observed recovery under the simulated shift comes from using target-batch BatchNorm statistics. The additional improvement from CAFA-TTA v2 compared with TENT is marginal and requires further validation.

