

# Reproducibility Guide



## CAFA-TTA



**Confidence-Aware Feature Alignment for Test-Time

Adaptation in Cross-Domain Medical Image Classification**



Last updated: 2026-11-19.



---



## 1. Overview



This document describes the computational environment,

datasets, model configurations, evaluation protocols,

statistical analyses, and artifact verification procedures

used in the CAFA-TTA research project.



The study investigates confidence-aware entropy

minimization and source-prototype feature alignment

for unsupervised test-time adaptation.



Two histopathology datasets are considered:



- PathMNIST for source training, validation,

  and controlled domain-shift evaluation.

- STARC-9 for external cross-domain evaluation.



This guide distinguishes between:



1. Verification of existing research artifacts.

2. Reproduction of experimental results.

3. Independent replication in a new environment.



The original research artifacts have been audited

in the development environment.



End-to-end reproduction from a fresh repository clone

has not yet been independently verified.



---



## 2. Hardware Environment



The original experiments were conducted using

the following hardware configuration:



| Component | Specification |
|---|---|
| Operating system | Windows |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4 GB |
| CUDA acceleration | Enabled |
| Evaluation batch size | 64 |



The implementation is designed to operate

under limited GPU memory.



Training time, inference latency, throughput,

and memory consumption depend on the hardware,

software environment, and GPU operating conditions.



The reported computational benchmarks should not

be interpreted as hardware-independent measurements.



---



## 3. Software Environment



| Component | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.10.0+cu128 |
| Torchvision | 0.25.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 9.10.2 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| MedMNIST | 3.0.2 |
| Scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Pillow | 13.0.0 |



The complete dependency specification is provided in:



- `requirements.txt`

- `experiments/reproducibility/requirements_frozen.txt`



The dependency audit confirmed:



- 39 packages in the installation requirements.

- 39 packages in the frozen requirements.

- 39 packages in the original active environment.

- Zero package version differences.

- No broken requirements reported by pip check.



The installation requirements were converted

from UTF-16 to UTF-8 without changing package versions.



The original frozen requirements file remains unchanged.



---



## 4. Environment Setup



### 4.1 Clone the Repository



After the research repository has been published:



```powershell

git clone https://github.com/dtc245200063-alt/CAFA-TTA.git

cd CAFA-TTA

```



Use the final research commit or release tag

when it becomes available.



Record the exact repository commit:



```powershell

git rev-parse HEAD

```



### 4.2 Create a Virtual Environment



Install Python 3.12.10.



Create a new environment:



```powershell

python -m venv .venv

```



For Windows PowerShell:



```powershell

.\.venv\Scripts\Activate.ps1

```



If the activation script is blocked,

a process-scoped execution policy may be used:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1

```



Alternatively, invoke the environment's Python executable

directly without changing the execution policy.



### 4.3 Install PyTorch with CUDA 12.8



Install the CUDA-enabled PyTorch and Torchvision versions:



```powershell

python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

```



Install the remaining pinned dependencies:



```powershell

python -m pip install -r requirements.txt

```



Official PyTorch installation reference:



https://pytorch.org/get-started/previous-versions/



An NVIDIA driver compatible with the CUDA 12.8

runtime is required for GPU execution.



Do not silently substitute newer dependency versions

when attempting exact experimental reproduction.



If a pinned package is unavailable, record the issue

and document any alternative environment separately.



### 4.4 Verify the Installation



```powershell

python --version

python -m pip check

python -m pip freeze

```



Verify CUDA availability:



```powershell

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

```



Expected reference outputs:



```text

2.10.0+cu128

12.8

True

```



Additional GPU information:



```powershell

python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.backends.cudnn.version())"

```



A project-level CUDA test is also available:



```powershell

python test_cuda.py

```



These installation instructions describe the intended

reconstruction of the original environment.



Successful installation in a fresh environment

remains to be verified.



---



## 5. Project Structure



The main research directories are:



```text

CAFA-TTA/

|-- core/

|-- kernel/

|-- models/

|-- scripts/

|-- tests/

|-- docs/

|-- data/

|   |-- raw/

|   `-- processed/

|-- checkpoints/

|-- experiments/

|   |-- reproducibility/

|   |-- cafa_final_test_strong/

|   |-- cafa_final_test_severity/

|   |-- cafa_final_test_corruptions/

|   |-- statistical_analysis/

|   |-- research_outputs/

|   `-- starc9_external/

|-- requirements.txt

|-- README.md

`-- .gitignore

```



Some directories and binary artifacts are local-only

and are not included in the source repository.



Create missing local directories when required.



The original medical image archives, processed image

arrays, trained checkpoints, temporary files,

and virtual environment are excluded from Git tracking.



---



## 6. Dataset Preparation



### 6.1 PathMNIST



PathMNIST is obtained from the MedMNIST collection.



Official source:



https://medmnist.com/



The original data partitions are:



| Split | Samples |
|---|---:|
| Training | 89,996 |
| Validation | 10,004 |
| Test | 7,180 |



Images have the following representation:



- Resolution: 28 x 28.

- Channels: 3.

- Color format: RGB.

- Number of classes: 9.



Fixed source normalization:



```text

Mean = [0.5, 0.5, 0.5]

Std  = [0.5, 0.5, 0.5]

```



The original split definitions must be preserved.



Source training and source prototype extraction

use the training partition.



Validation experiments support method development

and hyperparameter selection.



The test partition is used for final evaluation.



However, the test partition had been accessed

in earlier development experiments.



It must not be described as a completely untouched

or independently held-out confirmatory test set.



Use the dataset acquisition logic in the corresponding

training and evaluation scripts.



Refer to DATASET_LICENSES.md for attribution

and applicable dataset terms.



### 6.2 STARC-9



The external evaluation uses the stain-normalized

Stanford colorectal histopathology validation subset.



Dataset provider:



https://huggingface.co/datasets/Path2AI/STARC-9



Pinned revision:



```text

06e80900204d099b2f0409772f94baf1687f754f

```



Archive filename:



```text

STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip

```



Pinned download URL:



https://huggingface.co/datasets/Path2AI/STARC-9/resolve/06e80900204d099b2f0409772f94baf1687f754f/Validation_data/STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip



Expected archive size:



```text

2149005637 bytes

```



Expected SHA-256:



```text

088281de09bfe4d57b4a8e356a99cdb9b002a08a0ff7c44910aee0b4de81d3f3

```



Place the archive at:



```text

data/raw/starc9/STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip

```



Verify its integrity:



```powershell

(Get-FileHash data\raw\starc9\STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip -Algorithm SHA256).Hash

```



The external protocol selects 12,000 images

from six categories of the original 18,000-image archive.



The model retains all nine PathMNIST source classes.



The semantic target mapping is:



| Target Category | Source Label ID |
|---|---:|
| ADI | 0 |
| LYM | 3 |
| MUC | 4 |
| MUS | 5 |
| NOR | 6 |
| TUM | 8 |



BLD, FCT, and NCS are excluded.



The selected images are decoded as RGB and resized

to 28 x 28 using Pillow bilinear interpolation.



No synthetic corruption is applied.



The preprocessing implementation is:



```text

scripts/prepare_starc9_dataset.py

```



Preparation entry point:



```powershell

python scripts\prepare_starc9_dataset.py

```



Before running this script, restore the required

dataset acquisition and inspection metadata

and verify the frozen external protocol.



Expected prepared dataset location:



```text

data/processed/starc9/starc9_target_28.npz

```



Expected original prepared dataset SHA-256:



```text

308F5AB7C7FAF73F58FD088896D8FE83FA7D9CF660942BFA925B01DC49DDFA32

```



Expected original sample index SHA-256:



```text

77644B633C337C3DAE6DF664061206DD8C855478ECC1FCFE2993F9A193C3B5C3

```



The original metadata includes:



```text

data/raw/starc9/dataset_manifest.json

data/raw/starc9/dataset_inspection.json

data/processed/starc9/preparation_manifest.json

data/processed/starc9/sample_index.csv

```



These local metadata files are required for

complete verification of the original external protocol.



Their original bytes must be preserved whenever

they are referenced by frozen SHA-256 manifests.



Regenerating metadata may introduce timestamp,

serialization, or line-ending differences.



Equivalent prepared image arrays do not necessarily

produce byte-identical archive files.



If regenerated artifacts fail the original hash checks,

document the differences and do not modify the frozen

reference hashes merely to bypass verification.



---



## 7. Source Model Training



### 7.1 Model Architecture



The source classifier uses:



- ResNet-18 backbone.

- 512-dimensional feature representation.

- Nine output classes.



Feature extraction is implemented in:



```text

models/feature_extractor.py

```



### 7.2 Source Model Variants



Three final source model variants are evaluated.



| Model | Training Strategy |
|---|---|
| Enhanced | Augmentation-enhanced fine-tuning |
| Robust | Strong color augmentation |
| Balanced | Moderate color augmentation |



The training scripts are:



```text

scripts/train_enhanced.py

scripts/train_robust.py

scripts/train_balanced.py

```



The Enhanced model is initialized from the original

PathMNIST baseline checkpoint.



The Robust and Balanced variants are fine-tuned

from the Enhanced checkpoint.



The original baseline checkpoint is therefore

required when reconstructing the full training sequence.



### 7.3 Training Configuration



| Model | Epochs | Learning Rate |
|---|---:|---:|
| Baseline | 3 | See original training script |
| Enhanced | 15 | 0.0001 |
| Robust | 8 | 0.00005 |
| Balanced | 6 | 0.00003 |



The training scripts contain the original implementation

of augmentation, checkpoint selection, and optimization.



Consult the script-level configuration before execution.



Retraining must be performed in a separate workspace

to avoid overwriting the original research checkpoints.



Newly trained checkpoints may not be byte-identical

to the original checkpoints, even under similar settings.



---



## 8. Source Prototype Extraction



Source prototypes are constructed using labeled

PathMNIST training samples.



The prototype extraction implementation is:



```text

scripts/extract_prototypes_models.py

```



Entry point:



```powershell

python scripts\extract_prototypes_models.py

```



Each final source model has its own prototype tensor.



The prototype representation contains:



```text

Number of classes: 9

Feature dimension: 512

Tensor shape: [9, 512]

```



Class-level source feature means are computed

and L2-normalized.



The expected final prototype files are:



```text

checkpoints/source_prototypes_enhanced.pt

checkpoints/source_prototypes_robust.pt

checkpoints/source_prototypes_balanced.pt

```



The corresponding source checkpoint must be used

when extracting each prototype tensor.



Target ground-truth labels are not used

for source prototype construction.



---



## 9. CAFA-TTA Configuration



### 9.1 Method Implementations



The main adaptation components are:



```text

kernel/tent.py

core/cafa_adapter.py

core/cafa_agreement_adapter.py

core/cafa_soft_adapter.py

```



The final proposed method is CAFA-TTA v3.



Its implementation uses confidence-weighted

entropy minimization and confidence-aware

source-prototype feature alignment.



### 9.2 Frozen Hyperparameters



| Parameter | Value |
|---|---|
| Algorithm | CAFA-TTA v3 |
| Backbone | ResNet-18 |
| Feature dimension | 512 |
| Number of source classes | 9 |
| Batch size | 64 |
| Adaptation learning rate | 0.0001 |
| Entropy floor | 0.2 |
| Alignment weight | 0.05 |
| Minimum agreement samples | 2 |
| Random seeds | 42, 123, 2026 |
| Adaptation protocol | Online prequential |



The authoritative frozen configuration is:



```text

experiments/reproducibility/final_config.json

```



No target-label-based parameter adaptation is performed.



### 9.3 Objective Function



CAFA-TTA v3 combines two objective components:



```text

L_total = L_weighted_entropy + lambda_align * L_alignment

```



Confidence is derived from normalized

classifier prediction entropy.



The entropy weight uses a nonzero confidence floor:



```text

w_entropy = entropy_floor

            + (1 - entropy_floor) * confidence

```



Feature alignment is weighted by confidence

and classifier-prototype agreement.



Alignment is skipped when the number of valid

agreement samples is below the configured minimum.



Adaptation updates batch-normalization affine parameters.



The exact objective implementation, optimizer behavior,

gradient clipping, and numerical safeguards are defined

in the original source code.



---



## 10. Evaluation Protocol



The final evaluation follows a shuffled

online batch-prequential adaptation protocol.



For each independent evaluation run:



1. Restore the designated source checkpoint.

2. Initialize the adaptation method.

3. Reset adaptation state and optimizer state.

4. Construct the target sample stream.

5. Obtain predictions for the current batch.

6. Perform unsupervised adaptation.

7. Continue to the next target batch.

8. Compute evaluation metrics from saved predictions.



Predictions are obtained before the parameter update

associated with the current evaluation batch.



Target batch statistics may nevertheless depend

on other samples within the current batch.



Therefore, this protocol is batch-prequential

rather than strictly individual-sample causal.



Ground-truth labels are used for metric calculation,

not adaptation optimization.



The final evaluation seeds are:



```text

42

123

2026

```



Methods are paired using the corresponding

target sample orders wherever applicable.



---



## 11. Evaluation Methods



The research compares the following methods:



| Method | Description |
|---|---|
| Source-only | Fixed source classifier |
| BN-only | Target batch-normalization adaptation |
| TENT | Test-time entropy minimization |
| CAFA v1 | Original confidence-aware alignment |
| CAFA v2 | Classifier-prototype agreement gating |
| CAFA v3 | Soft confidence weighting and alignment |



Additional component variants are considered

in the alignment and ablation experiments.



Alignment-enabled variants should not be described

as pure alignment-only methods when entropy

minimization remains active in their objectives.



---



## 12. PathMNIST Domain-Shift Evaluation



### 12.1 Color Shift Severity



Four color-shift conditions are evaluated:



| Condition | Brightness | Contrast | Saturation |
|---|---:|---:|---:|
| Original | 1.00 | 1.00 | 1.00 |
| Light | 0.95 | 1.05 | 0.95 |
| Medium | 0.85 | 1.15 | 0.85 |
| Strong | 0.75 | 1.35 | 0.70 |



These transformations are applied using

the original evaluator implementation.



### 12.2 Additional Corruptions



Three corruption families are evaluated:



| Corruption | Light | Medium | Strong |
|---|---:|---:|---:|
| Gaussian noise std | 0.02 | 0.05 | 0.10 |
| Gaussian blur sigma | 0.5 | 1.0 | 2.0 |
| Hue shift | 0.03 | 0.07 | 0.15 |



The final Gaussian blur implementation uses

a kernel size of 13.



The final Gaussian noise implementation uses

deterministic sample-specific randomization.



Exact transformation order, clipping, normalization,

and random-number generation are defined in the scripts.



### 12.3 Final Evaluation Entry Points



Strong shift:



```powershell

python scripts\evaluate_cafa_final_test.py

```



Severity evaluation:



```powershell

python scripts\evaluate_cafa_final_test_severity.py

```



Multiple corruptions:



```powershell

python scripts\evaluate_cafa_final_test_corruptions.py

```



The severity evaluator reuses compatible

previously saved strong-shift results.



Execute the strong-shift evaluator first

when reconstructing the complete evaluation.



Run reproduction experiments in an isolated workspace.



Do not overwrite the frozen research artifacts.



### 12.4 Prediction File Inventory



| Evaluation | Unique Prediction Files |
|---|---:|
| Strong shift | 63 |
| Severity evaluation | 90 |
| Multiple corruptions | 270 |
| Total | 423 |



All 423 original prediction files passed

the final prediction integrity audit.



Compatible repeated records were excluded

from paired statistical aggregation.



---



## 13. External STARC-9 Evaluation



The frozen external evaluation protocol is stored in:



```text

experiments/starc9_external/external_protocol.json

```



The protocol was fixed before external model evaluation.



The external evaluation uses:



- Three source models.

- Four evaluation methods.

- Source-only evaluation once per source model.

- Three seeds for each adaptation method.

- A total of 30 evaluation runs.

- 12,000 target images per run.

- 188 batches per run.

- Batch size of 64.



The evaluation implementation is:



```text

scripts/evaluate_starc9_external.py

```



After all dependencies, metadata, datasets,

and checkpoints have been restored:



```powershell

python scripts\evaluate_starc9_external.py

```



The expected output directory is:



```text

experiments/starc9_external/

```



The primary external macro-F1 metric is computed

over the six target categories.



A secondary macro-F1 metric is computed

over all nine source output classes.



The other three source classes remain valid

classifier outputs during adaptation and prediction.



The six-class external macro-F1 and nine-class

PathMNIST macro-F1 are not directly interchangeable.



---



## 14. Prediction Artifact Format



The original final evaluation generated:



```text

PathMNIST prediction files: 423

STARC-9 prediction files: 30

Total prediction files: 453

```



Each inspected NPZ archive contains:



```text

indices

labels

predictions

batch_size

```



The stored arrays contain integer-valued sample indices,

ground-truth labels, predicted labels, and batch size.



No original image arrays or explicit patient identifiers

are present in the inspected NPZ schemas.



The combined prediction artifact size is approximately

13.13 MiB.



The prediction files are currently retained locally,

pending applicable redistribution and privacy review.



Complete independent recomputation of prediction-based

statistics requires access to these original files

or a suitably reproduced prediction collection.



---



## 15. Statistical Analysis



### 15.1 PathMNIST Prediction Verification



Implementation:



```powershell

python scripts\verify_final_test_predictions.py

```



The original verification included:



- Prediction file completeness.

- SHA-256 verification.

- Sample index consistency.

- Ground-truth label consistency.

- Prediction order consistency.

- Accuracy reconstruction.

- Macro-F1 reconstruction.

- Confusion matrix reconstruction.



All 423 final PathMNIST prediction files passed

the original integrity verification.



### 15.2 Paired Statistical Analysis



Implementation:



```powershell

python scripts\analyze_statistical_significance.py

```



The analysis includes:



- 117 paired TENT versus CAFA-TTA v3 comparisons.

- Exact McNemar tests.

- Holm multiple-comparison correction.

- Conditional image-index bootstrap.

- Circular block bootstrap.

- Leave-one-group-out sensitivity analysis.



Bootstrap resampling uses 10,000 iterations

with the recorded random seed.



The primary paired comparison is:



```text

CAFA-TTA v3 minus TENT

```



Accuracy and macro-F1 differences are expressed

in percentage points where indicated.



### 15.3 Per-Class Performance



Implementation:



```powershell

python scripts\analyze_per_class_performance.py

```



The analysis includes:



- Class-level precision, recall, and F1.

- Class support.

- Corrected and harmed predictions.

- Confusion matrix changes.

- Minority and majority class comparisons.

- Negative adaptation analysis.



Class imbalance groups are defined using

evaluation-set support, not clinical disease prevalence.



### 15.4 External Statistical Analysis



Implementation:



```powershell

python scripts\analyze_starc9_external.py

```



The external analysis includes:



- Verification of all 30 saved prediction files.

- Nine paired TENT versus CAFA-TTA v3 comparisons.

- Exact McNemar tests.

- Holm correction.

- Conditional bootstrap.

- Circular block bootstrap.

- Duplicate-image sensitivity.

- Per-class performance.

- Absent-class predictions.

- Negative adaptation.

- Leave-one-group-out sensitivity.



---



## 16. Computational Efficiency Benchmark



Benchmark implementation:



```powershell

python scripts\benchmark_cafa_v3.py

```



The original benchmark compares:



- Source-only.

- BN-only.

- TENT.

- CAFA-TTA v3.



Reference settings:



```text

Batch size: 64

Warm-up batches: 10

Measured batches: 50

Timing repetitions: 3

Source model variants: 3

```



The benchmark uses CUDA synchronization

for GPU timing measurements.



The measurements characterize the original validation

benchmark workload on the reference GPU.



Data loading, transfer, and preprocessing costs

are excluded from the reported adaptation timing.



Consequently, the measurements are not equivalent

to complete clinical deployment latency.



---



## 17. Research Tables and Figures



PathMNIST research output generation:



```powershell

python scripts\generate_research_figures.py

```



Output directory:



```text

experiments/research_outputs/

```



The original PathMNIST research package contains:



- 13 figures in PNG format.

- 13 figures in PDF format.

- 12 research tables in CSV format.

- Four Markdown research documents.

- One research integrity manifest.



External STARC-9 statistical output directory:



```text

experiments/starc9_external/statistical_analysis/

```



The external research package contains:



- Nine figures in PNG format.

- Nine figures in PDF format.

- 16 statistical CSV tables.

- Confusion matrix data.

- Statistical reports.

- An integrity manifest.



Together, the two research packages contain

22 distinct research figures.



These figures are provided in both raster

and vector formats in the original research package.



Regenerating figures may introduce byte-level changes

because of library versions, fonts, rendering metadata,

or the operating environment.



Visual equivalence does not guarantee identical hashes.



---



## 18. Experimental Execution Order



The recommended logical reproduction sequence is:



```text

Stage 1: Reconstruct the software environment.



Stage 2: Obtain and verify the original datasets.



Stage 3: Restore the original model checkpoints

         or reproduce source model training.



Stage 4: Restore or extract source prototypes.



Stage 5: Confirm the frozen configuration.



Stage 6: Reproduce validation experiments,

         ablations, and computational benchmarks

         when required.



Stage 7: Execute the PathMNIST strong-shift evaluator.



Stage 8: Execute the PathMNIST severity evaluator.



Stage 9: Execute the PathMNIST corruption evaluator.



Stage 10: Verify PathMNIST prediction files.



Stage 11: Perform paired and per-class analysis.



Stage 12: Generate PathMNIST research outputs.



Stage 13: Restore the frozen external protocol

          and required dataset metadata.



Stage 14: Prepare the STARC-9 target dataset.



Stage 15: Execute STARC-9 external evaluation.



Stage 16: Perform external statistical analysis.



Stage 17: Verify generated research artifacts.

```



This sequence describes the dependency relationships

between major experiment stages.



It is not yet a certified automated,

end-to-end execution pipeline.



Consult the experiment-specific scripts

and configuration before running each stage.



---



## 19. Artifact Integrity Verification



The original research records include:



```text

experiments/reproducibility/final_config.json

experiments/reproducibility/requirements_frozen.txt

experiments/reproducibility/checkpoint_hashes.csv

experiments/reproducibility/algorithm_hashes.csv

```



PathMNIST research manifest:



```text

experiments/research_outputs/research_manifest.json

```



Expected original SHA-256:



```text

CB472628C8F4F042ABF49A8020CF96EB25342EB60D35BD14C305D51B73514E41

```



STARC-9 external statistical manifest:



```text

experiments/starc9_external/statistical_analysis/statistical_manifest.json

```



Expected original SHA-256:



```text

05AD20849369A1A05B9C60BC2403ED015901EA6EB0E977D58E023EE734EB4FE5

```



The original audits confirmed:



| Integrity Audit | Result |
|---|---|
| PathMNIST prediction files | 423/423 PASS |
| STARC-9 prediction files | 30/30 PASS |
| PathMNIST research manifest inputs | 15/15 PASS |
| PathMNIST research manifest outputs | 42/42 PASS |
| STARC-9 statistical manifest inputs | 35/35 PASS |
| STARC-9 statistical manifest outputs | 36/36 PASS |



These results refer to the original audited artifacts.



They do not certify a subsequent Git checkout.



To verify a file on Windows:



```powershell

(Get-FileHash path_to_file -Algorithm SHA256).Hash

```



The Python equivalent is:



```python

from pathlib import Path

import hashlib



path = Path("path_to_file")



digest = hashlib.sha256(

    path.read_bytes()

).hexdigest()



print(digest)

```



Some historical manifests contain absolute paths

from the original Windows development environment.



Resolve these paths against the corresponding local

artifact locations without changing the reference hashes.



SHA-256 verification operates on exact file bytes.



Changes in line endings, encoding, serialization,

or embedded metadata may change a file's digest.



Preserve frozen artifact bytes during publication

and verify the hashes of the checked-out files.



---



## 20. Checkpoint Availability



The project maintains four trained source checkpoints

and four source prototype files, including

the original baseline model and prototypes.



Checkpoint and prototype hashes are recorded

in the reproducibility directory.



Original model artifacts are excluded from Git tracking.



Public release of these artifacts remains subject to

applicable rights and redistribution review.



No public checkpoint download is assumed at this stage.



Without access to the original checkpoints,

independent users must reconstruct source training

or obtain the artifacts through an authorized release.



Retraining is not equivalent to verifying the exact

original model parameters.



Only load checkpoint files obtained from trusted sources.



---



## 21. Research Limitations



The following limitations must be considered

when interpreting reproducibility and results:



1. The final PathMNIST test set had been accessed

   during earlier development stages.



2. Multiple experiments reuse the same underlying

   histopathology samples.



3. The statistical comparisons are exploratory

   and should not be interpreted as independent

   confirmatory clinical trials.



4. Bootstrap intervals are conditional on the saved

   adaptation trajectories and evaluation samples.



5. Patient-level and slide-level clustering information

   is unavailable for the prepared evaluation datasets.



6. The external STARC-9 evaluation uses six selected

   tissue categories and an approximate label mapping.



7. STARC-9 contains exact duplicate images.



8. Resizing images to 28 x 28 removes substantial

   high-resolution morphological information.



9. Adaptation can degrade performance relative

   to the source-only classifier.



10. Improvements over TENT do not imply improvements

    over every baseline or evaluation condition.



11. Three adaptation seeds do not represent all

    possible deployment streams or training variability.



12. Runtime measurements are hardware-dependent.



13. GPU operations, random-number generation,

    library changes, and serialization behavior

    may affect exact reproduction.



14. A clean-environment end-to-end reproduction

    has not yet been completed.



15. Original prediction files, dataset metadata,

    and trained model artifacts are subject

    to availability and publication constraints.



The study evaluates computational research methods.



It does not establish clinical safety,

diagnostic validity, or patient-level effectiveness.



---



## 22. Reproducibility Checklist



### Environment



- \[ ] Python version matches the reference environment.

- \[ ] Pinned dependencies are installed.

- \[ ] Dependency health check passes.

- \[ ] CUDA-enabled PyTorch is available.

- \[ ] GPU configuration is recorded.



### Data and Models



- \[ ] Original dataset sources are identified.

- \[ ] Dataset attribution requirements are satisfied.

- \[ ] Original data partitions are preserved.

- \[ ] External archive SHA-256 is verified.

- \[ ] Required dataset metadata is restored.

- \[ ] Checkpoint SHA-256 values are verified.

- \[ ] Source prototype SHA-256 values are verified.



### Experiments



- \[ ] Frozen configuration is preserved.

- \[ ] Original evaluation scripts are available.

- \[ ] Adaptation seeds are preserved.

- \[ ] Target batch ordering is preserved.

- \[ ] Adaptation state is reset between runs.

- \[ ] Target labels are excluded from adaptation.

- \[ ] Final predictions are saved separately.

- \[ ] Frozen research outputs are not overwritten.



### Analysis



- \[ ] Accuracy and macro-F1 are reconstructed.

- \[ ] Confusion matrices are verified.

- \[ ] Paired statistics are reproduced.

- \[ ] Multiple-comparison correction is applied.

- \[ ] Conditional bootstrap assumptions are retained.

- \[ ] Duplicate sensitivity analysis is performed.

- \[ ] Negative adaptation is evaluated.

- \[ ] Research tables and figures are generated.



### Publication



- \[ ] Dataset and artifact rights are reviewed.

- \[ ] Required metadata is available.

- \[ ] Artifact SHA-256 values survive Git checkout.

- \[ ] Repository commit or release tag is recorded.

- \[ ] Clean-environment installation is verified.

- \[ ] Independent end-to-end reproduction is tested.



---



## 23. Reproducibility Status



| Component | Original Environment Status |
|---|---|
| Software dependency consistency | Verified |
| Frozen method configuration | Recorded |
| Source checkpoints | Locally retained and hash-recorded |
| Source prototypes | Locally retained and hash-recorded |
| PathMNIST final evaluation | Completed |
| STARC-9 external evaluation | Completed |
| Prediction integrity | Verified |
| Statistical analysis | Completed |
| Research output integrity | Verified |
| Public checkpoint availability | Pending |
| Public prediction availability | Pending |
| Fresh clone integrity | Pending |
| Independent reproduction | Not yet verified |



The original experiment results are computationally

documented and supported by integrity-checked artifacts.



Full independent reproducibility requires access to

the necessary original datasets, metadata, checkpoints,

predictions, and compatible software environment.



---



## 24. Related Documentation



- DATASET_LICENSES.md

- CHECKPOINTS.md

- EXPERIMENTS.md

- Repository README.md

- Frozen configuration and reproducibility manifests



The accompanying checkpoint and experiment guides

provide further artifact-specific details

as they are completed.



---



## 25. Research Transparency Statement



CAFA-TTA is presented as an experimental research method

for unsupervised test-time adaptation in medical image

classification.



The project preserves positive results, negative results,

ablation findings, computational costs, and limitations.



The objective is transparent scientific evaluation,

not the selective reporting of favorable outcomes.



All reported conclusions should be interpreted

within the scope of the evaluated datasets,

models, adaptation protocols, and statistical assumptions.


## Unified 17-class model (added later)

- Data: `python scripts/prepare_kather2016.py` downloads and MD5-checks Kather 2016 into `data/raw/kather2016`;
  PathMNIST goes to `data/raw/medmnist/pathmnist.npz`.
- Split: `train_unified.split_items` makes a per-class 70/15/15 split with seed 42
  (3,496 / 744 / 760). The Docker Kather split in
  `docker-project/dataset_splits/kather2016_seed42.json` (3,496 / 752 / 752) is **different**; do not
  mix checkpoints, prototypes or results between the two.
- Train: `python scripts/train_unified.py --check-only` validates the layout; the committed checkpoint's
  full command was not recorded (see `CHECKPOINTS.md` section 21).
- Verify artifacts: `sha256sum -c experiments/unified_balanced_paper_01/SHA256SUMS.txt`.
- Fresh-clone, end-to-end reproduction of this pipeline has not been verified.
