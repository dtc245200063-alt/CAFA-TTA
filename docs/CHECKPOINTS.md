

# Model Checkpoints and Source Prototypes



## CAFA-TTA



**Confidence-Aware Feature Alignment for Test-Time

Adaptation in Cross-Domain Medical Image Classification**



Last updated: 2026-11-19.



---



## 1. Overview



This document describes the trained source models,

source feature prototypes, checkpoint provenance,

integrity verification, and intended distribution policy

of the CAFA-TTA research project.



The project maintains eight model-related artifacts:



- Four trained ResNet-18 source checkpoints.

- Four source prototype files.



All eight original artifacts passed SHA-256

integrity verification on November 19, 2026.



The original checkpoint files remain unchanged.



---



## 2. Model Architecture



All source classifiers use the ResNet-18 backbone

for colorectal histopathology image classification.



| Property | Configuration |
|---|---|
| Backbone | ResNet-18 |
| Input resolution | 28 x 28 |
| Input channels | 3 |
| Feature dimension | 512 |
| Output classes | 9 |
| Source dataset | PathMNIST |
| Source training samples | 89,996 |
| Validation samples | 10,004 |



Fixed image normalization:



- Mean: \[0.5, 0.5, 0.5]

- Standard deviation: \[0.5, 0.5, 0.5]



Feature extraction implementation:



`models/feature_extractor.py`



---



## 3. Source Model Lineage



The source models are related through

sequential training and fine-tuning.



```text

Baseline ResNet-18

        |

        v

Enhanced ResNet-18

        |

        +------------------+

        |                  |

        v                  v

Robust ResNet-18    Balanced ResNet-18

```



The Robust and Balanced models are independently

fine-tuned from the Enhanced checkpoint.



These models represent different source training

strategies rather than independent random

initializations of the same training procedure.



---



## 4. Source Model Training



### 4.1 Baseline Model



Checkpoint:



`resnet18_pathmnist_best.pt`



The baseline model provides the initial

PathMNIST classification representation.



Training duration:



- 3 epochs.



Recorded validation accuracy:



- Approximately 84.73%.



The baseline checkpoint initializes

the Enhanced source model.



### 4.2 Enhanced Model



Checkpoint:



`resnet18_pathmnist_enhanced_best.pt`



Training configuration:



- Initialization: Baseline checkpoint.

- Training duration: 15 epochs.

- Learning rate: 0.0001.

- Augmentation-enhanced fine-tuning.

- Mixed-precision training.



Recorded validation results:



| Metric | Value |
|---|---:|
| Accuracy | 95.40% |
| Macro-F1 | 95.39% |
| Best checkpoint epoch | 13 |



Training implementation:



`scripts/train_enhanced.py`



### 4.3 Robust Model



Checkpoint:



`resnet18_pathmnist_robust_best.pt`



Training configuration:



- Initialization: Enhanced checkpoint.

- Training duration: 8 epochs.

- Learning rate: 0.00005.

- Strong color augmentation.



Color jitter configuration:



- Brightness: 0.35.

- Contrast: 0.35.

- Saturation: 0.35.

- Hue: 0.04.



Recorded validation results:



| Metric | Value |
|---|---:|
| Accuracy | 89.57% |
| Macro-F1 | 89.69% |
| Best checkpoint epoch | 7 |



Training implementation:



`scripts/train_robust.py`



### 4.4 Balanced Model



Checkpoint:



`resnet18_pathmnist_balanced_best.pt`



Training configuration:



- Initialization: Enhanced checkpoint.

- Training duration: 6 epochs.

- Learning rate: 0.00003.

- Moderate color augmentation.



Color jitter configuration:



- Brightness: 0.25.

- Contrast: 0.25.

- Saturation: 0.25.

- Hue: 0.02.

- Application probability: 0.6.



Recorded validation results:



| Metric | Value |
|---|---:|
| Accuracy | 93.11% |
| Macro-F1 | 93.10% |
| Best checkpoint epoch | 5 |



Training implementation:



`scripts/train_balanced.py`



The original training scripts remain the

authoritative reference for optimizer settings,

augmentation order, checkpoint selection,

and numerical implementation details.



---



## 5. Source Model Summary



| Model | Epochs | Best Epoch | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 3 | - | 84.73% | - |
| Enhanced | 15 | 13 | 95.40% | 95.39% |
| Robust | 8 | 7 | 89.57% | 89.69% |
| Balanced | 6 | 5 | 93.11% | 93.10% |



Validation results describe the recorded source

training experiments.



They do not establish target-domain robustness.



The source model with the highest validation

accuracy is not necessarily the strongest model

under domain shift.



---



## 6. Source Prototype Construction



Source prototypes represent class-level source features.



Each final source model has a corresponding

prototype representation.



Prototype construction uses labeled PathMNIST

training samples.



The feature extraction procedure computes

class-level feature means and applies

L2 normalization.



Prototype dimensions:



| Property | Value |
|---|---:|
| Number of classes | 9 |
| Feature dimension | 512 |
| Tensor shape | \[9, 512] |



The final prototype extraction implementation is:



`scripts/extract_prototypes_models.py`



Expected model-prototype associations:



| Source Model | Prototype File |
|---|---|
| Baseline | source_prototypes.pt |
| Enhanced | source_prototypes_enhanced.pt |
| Robust | source_prototypes_robust.pt |
| Balanced | source_prototypes_balanced.pt |



Each final prototype file must be paired

with the corresponding source checkpoint.



Mixing checkpoints and prototypes from

different source models is not supported

by the frozen experimental protocol.



Target ground-truth labels are not used

to construct the source prototypes.



---



## 7. Prototype Class Definitions



The prototype indices follow

the original PathMNIST class ordering.



| ID | Tissue Class |
|---:|---|
| 0 | Adipose |
| 1 | Background |
| 2 | Debris |
| 3 | Lymphocytes |
| 4 | Mucus |
| 5 | Smooth muscle |
| 6 | Normal colon mucosa |
| 7 | Cancer-associated stroma |
| 8 | Colorectal adenocarcinoma epithelium |



All nine source output classes and prototypes

are retained during external STARC-9 adaptation.



The external target subset contains six

approximately corresponding tissue categories.



No target-label-based prototype restriction

is performed during external evaluation.



---



## 8. Checkpoint Inventory



All model artifacts are stored locally under:



`checkpoints/`



| Filename | Size (MiB) |
|---|---:|
| resnet18_pathmnist_best.pt | 42.7303 |
| resnet18_pathmnist_enhanced_best.pt | 42.7315 |
| resnet18_pathmnist_robust_best.pt | 42.7313 |
| resnet18_pathmnist_balanced_best.pt | 42.7315 |
| source_prototypes.pt | 0.0197 |
| source_prototypes_enhanced.pt | 0.0199 |
| source_prototypes_robust.pt | 0.0199 |
| source_prototypes_balanced.pt | 0.0199 |



Total model artifact size:



Approximately 171 MiB.



The original checkpoint and prototype files

are excluded from ordinary Git tracking.



---



## 9. Frozen SHA-256 Checksums



The following checksums identify

the original experimental model artifacts.



### 9.1 Baseline Checkpoint



File:



`checkpoints/resnet18_pathmnist_best.pt`



SHA-256:



```text

D615D4D1C667CC0996CA15054039E66970B1E6AB62E4278C3F29F77AF15CDB57

```



### 9.2 Enhanced Checkpoint



File:



`checkpoints/resnet18_pathmnist_enhanced_best.pt`



SHA-256:



```text

89B5940B7A881DC3D3BF166D82399D21C14B6F3A0B0CAC7CF22FB365B042CCA6

```



### 9.3 Robust Checkpoint



File:



`checkpoints/resnet18_pathmnist_robust_best.pt`



SHA-256:



```text

41407D1FA0587D76FDCA917EC70328529BCDA7CD5F6B1C91142F46B018C3AA4F

```



### 9.4 Balanced Checkpoint



File:



`checkpoints/resnet18_pathmnist_balanced_best.pt`



SHA-256:



```text

3B7B513A7812379C32CC9A31F81823DBDE19B8C5361395A423F1FD275489F071

```



### 9.5 Baseline Prototypes



File:



`checkpoints/source_prototypes.pt`



SHA-256:



```text

815477FC85619893707E831CD41F550149BCAD947752B986215812A0B47B69D2

```



### 9.6 Enhanced Prototypes



File:



`checkpoints/source_prototypes_enhanced.pt`



SHA-256:



```text

928A04612A5B38A277E1B3F9F7202C463C4BEB6ACB5AC5694ADD62911F85136C

```



### 9.7 Robust Prototypes



File:



`checkpoints/source_prototypes_robust.pt`



SHA-256:



```text

253E4F69E338E59BCD50033501E660F375196CE05F94CD1FF8E0FB1A0D9EED23

```



### 9.8 Balanced Prototypes



File:



`checkpoints/source_prototypes_balanced.pt`



SHA-256:



```text

E3CFF7DDE28868B5989D6B033DBEFEB96EC5633D5DEBAA5B08C184AF515C9666

```



The authoritative machine-readable checksum record is:



`experiments/reproducibility/checkpoint_hashes.csv`



---



## 10. Checkpoint Integrity Verification



### 10.1 Individual File Verification



Windows PowerShell example:



```powershell

(Get-FileHash checkpoints\resnet18_pathmnist_enhanced_best.pt -Algorithm SHA256).Hash

```



Compare the resulting digest with

the corresponding frozen checksum.



### 10.2 Complete Checkpoint Verification



Run the following PowerShell commands

from the repository root:



```powershell

$records = Import-Csv experiments\reproducibility\checkpoint_hashes.csv



$passed = 0

$failed = 0



foreach ($record in $records) {



    $name = [System.IO.Path]::GetFileName($record.Path)



    $path = Join-Path checkpoints $name



    if (-not (Test-Path $path -PathType Leaf)) {



        Write-Host "MISSING: $name"

        $failed++

        continue

    }



    $actual = (Get-FileHash $path -Algorithm SHA256).Hash



    if ($actual -eq $record.Hash) {



        Write-Host "PASS: $name"

        $passed++



    } else {



        Write-Host "FAIL: $name"

        $failed++

    }

}



Write-Host "Verified: $passed"

Write-Host "Failed: $failed"



if ($passed -eq 8 -and $failed -eq 0) {



    Write-Host "CHECKPOINT INTEGRITY PASSED"



} else {



    Write-Host "CHECKPOINT REVIEW REQUIRED"

}

```



Expected results when all original files are available:



```text

Verified: 8

Failed: 0

CHECKPOINT INTEGRITY PASSED

```



Historical absolute Windows paths in the frozen

CSV are resolved using the corresponding filenames.



The frozen checksums must not be changed

to accommodate missing or modified checkpoints.



---



## 11. Original Integrity Audit



Audit date:



2026-11-19.



| Artifact Category | Expected | Verified |
|---|---:|---:|
| Trained checkpoints | 4 | 4 |
| Source prototypes | 4 | 4 |
| Total | 8 | 8 |
| SHA-256 mismatches | 0 | 0 |



All eight actual SHA-256 values matched

the original frozen checksum records.



No checkpoint was retrained, modified,

or replaced during this verification.



This audit establishes byte-level consistency

with the recorded experimental artifacts.



It does not independently establish model safety,

clinical validity, or redistribution rights.



---



## 12. Algorithm Integrity References



The frozen algorithm checksum inventory

contains the following seven implementation files:



```text

core/cafa_adapter.py

core/cafa_agreement_adapter.py

core/cafa_soft_adapter.py

kernel/confidence.py

kernel/feature_alignment.py

kernel/tent.py

models/feature_extractor.py

```



Checksum manifest:



`experiments/reproducibility/algorithm_hashes.csv`



The recorded algorithm files must be preserved

when reproducing the original frozen experiments.



The algorithm checksum inventory is separate

from the model checkpoint checksum inventory.



A fresh repository checkout must be verified

against the recorded hashes before claiming

byte-identical experimental reproduction.



---



## 13. Checkpoint Restoration



Create the local checkpoint directory if necessary:



```powershell

New-Item -ItemType Directory -Path checkpoints -Force

```



Restore authorized copies of the original

checkpoint and prototype files into this directory.



The expected directory structure is:



```text

checkpoints/

|-- resnet18_pathmnist_best.pt

|-- resnet18_pathmnist_enhanced_best.pt

|-- resnet18_pathmnist_robust_best.pt

|-- resnet18_pathmnist_balanced_best.pt

|-- source_prototypes.pt

|-- source_prototypes_enhanced.pt

|-- source_prototypes_robust.pt

`-- source_prototypes_balanced.pt

```



Verify file integrity before evaluation.



Do not substitute a different checkpoint

with the same filename without documenting

the change in experimental provenance.



---



## 14. Checkpoint Reconstruction



If the original pretrained artifacts are unavailable,

source model training may be reconstructed

using the original implementation.



The intended training sequence is:



1. Train the original baseline source model.

2. Fine-tune the Enhanced source model.

3. Fine-tune the Robust source model.

4. Fine-tune the Balanced source model.

5. Extract model-specific source prototypes.

6. Record new checkpoint and prototype checksums.



The final model training entry points are:



```powershell

python scripts\train_enhanced.py

python scripts\train_robust.py

python scripts\train_balanced.py

```



These commands require the appropriate

initialization checkpoints and datasets.



Prototype extraction:



```powershell

python scripts\extract_prototypes_models.py

```



Reconstruction should be performed

in an isolated experimental workspace.



Do not overwrite the original frozen checkpoints.



Retraining may produce different weights

because of optimization variability,

initialization, numerical behavior,

and hardware-dependent operations.



A reconstructed model is not necessarily

byte-identical to the original model.



New model artifacts must be treated as

a separate reproduction experiment.



---



## 15. Test-Time Adaptation State



The original source checkpoints represent

the initial state before target-domain adaptation.



For every independent evaluation run:



- Restore the designated source checkpoint.

- Use the corresponding source prototypes.

- Initialize the selected adaptation method.

- Reset the optimizer and adaptation state.

- Apply the designated target sample order.

- Perform unsupervised test-time adaptation.



The original source checkpoints must not

be overwritten with intermediate adapted weights.



The final proposed method adjusts

batch-normalization affine parameters

during online adaptation.



The frozen configuration is recorded in:



`experiments/reproducibility/final_config.json`



---



## 16. Distribution Policy



The original checkpoint and prototype files

are currently retained in the local

research environment.



They are excluded by the repository's

Git ignore rules.



The proposed future distribution mechanism

is a separately versioned artifact release,

such as GitHub Releases.



No public checkpoint download URL is

asserted in this document.



Publication remains subject to applicable

dataset terms, software licenses,

third-party rights, and privacy considerations.



| Artifact | Current Status |
|---|---|
| Checkpoint filenames | Documented |
| Checkpoint checksums | Verified |
| Source model provenance | Documented |
| Original model files | Locally retained |
| Public artifact release | Pending |
| Redistribution review | Pending |
| Independent download verification | Pending |



If an authorized public release is created,

it should include:



- Exact original artifact files.

- SHA-256 checksums.

- Model architecture information.

- Repository commit or release tag.

- License and attribution notices.

- Applicable usage restrictions.

- Research limitations.



---



## 17. Security Considerations



Only load model artifacts obtained

from trusted and authorized sources.



PyTorch checkpoint deserialization

may involve executable serialization objects.



Use safer weight-only loading when compatible

with the original checkpoint format.



A matching SHA-256 checksum demonstrates

consistency with a trusted reference artifact,

but does not independently establish

that the artifact is safe to execute.



---



## 18. Research Limitations



The source classifiers operate on

low-resolution histopathology image patches.



They are experimental research models.



The checkpoints have not been validated

for clinical diagnostic deployment.



The source models share a related

training lineage.



Their results must not be interpreted

as independent training replications.



Higher source-domain validation accuracy

does not guarantee better target-domain

classification performance.



Test-time adaptation may reduce accuracy

relative to the fixed source classifier

under some target conditions.



The proposed CAFA-TTA method does not

eliminate negative adaptation.



---



## 19. Related Documentation



- \[Reproducibility Guide](REPRODUCIBILITY.md)

- \[Dataset Licenses and Attribution](DATASET_LICENSES.md)

- Experiment documentation

- Frozen checkpoint and algorithm manifests

- Final research configuration



---



## 20. Model Artifact Transparency Statement



The CAFA-TTA project preserves the original

trained source models and corresponding

prototype representations.



All eight original model-related artifacts

have passed SHA-256 integrity verification.



The project distinguishes between:



- Verification of original model artifacts.

- Reconstruction through source training.

- Independent replication of research results.

- Authorization for public artifact distribution.



These are separate requirements and should

not be treated as interchangeable.



The original research checkpoints provide

a fixed computational reference for

the reported CAFA-TTA experiments.


## 21. Unified 17-Class Checkpoint (added after the original audit)

This section is **not** covered by the frozen checksums and audit in sections 9-11.

| Artifact | SHA-256 |
|---|---|
| `checkpoints/cafa_unified_balanced_best.pt` | `2f6c28cc9187c87ef6a8a2b5b2266e16f2286116c2a2446b45bb60f2e22d80a4` |
| `checkpoints/unified_17_prototypes_balanced.pt` | `82ef6ff4c7044b365f04ee3f0c4fdb2f1c417c9529d86ca88f01ec629715320c` |

- Architecture: ResNet-18, one linear head with 17 dataset-prefixed outputs
  (9 `PathMNIST:*` + 8 `Kather:*`). The labels are not a harmonised medical taxonomy.
- Training script: `scripts/train_unified.py` (AdamW, cross-entropy, flips only, normalisation mean=std=0.5).
- Metrics file: `experiments/unified_balanced_paper_01/cafa_unified_balanced_best.metrics.json`
  (10 epochs, epoch 8 selected by combined validation macro-F1, seed 42).
- Prototypes: train-split only, from `scripts/extract_unified_prototypes.py`.
- **Not recorded for this committed checkpoint:** the exact command-line (image size, learning
  rate, batch size, `--balanced-train`). Runs made with the current `train_unified.py` store them under
  `hyperparameters` in the metrics file.
- Verification: `sha256sum -c experiments/unified_balanced_paper_01/SHA256SUMS.txt` from the repository root.
- Whether these weights are redistributed publicly must be decided consistently with `PUBLICATION_SCOPE.md`.
