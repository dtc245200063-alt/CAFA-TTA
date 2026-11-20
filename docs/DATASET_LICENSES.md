

# Dataset Licenses and Attribution



## CAFA-TTA Research Project



**Confidence-Aware Feature Alignment for Test-Time Adaptation in Cross-Domain Medical Image Classification**



This document describes the datasets used in CAFA-TTA,

their declared licenses, attribution requirements,

preprocessing procedures, and publication policy.



The project acknowledges the original dataset creators

and does not claim ownership of the original medical images.



Last reviewed: 2026-11-19.



---



## 1. Dataset Overview



| Dataset | Research Role | Classes Used | Declared License |
|---|---|---:|---|
| PathMNIST | Source training, validation and test evaluation | 9 | CC BY 4.0 |
| STARC-9 | External cross-domain evaluation | 6 of 9 | CC BY 4.0 (declared in pinned README) |



Both datasets contain colorectal histopathology images.



CAFA-TTA is intended for academic research and experimental

evaluation, not clinical diagnosis.



---



## 2. PathMNIST



### 2.1 Source and Ownership



PathMNIST is part of the MedMNIST benchmark collection.



It is derived from the following colorectal histopathology

datasets:



- NCT-CRC-HE-100K

- CRC-VAL-HE-7K



Official sources:



- MedMNIST: https://medmnist.com/

- Repository: https://github.com/MedMNIST/MedMNIST

- Dataset: https://doi.org/10.5281/zenodo.10519652

- Original dataset: https://doi.org/10.5281/zenodo.1214456



### 2.2 License



PathMNIST is distributed under the Creative Commons

Attribution 4.0 International license (CC BY 4.0).



License text:



https://creativecommons.org/licenses/by/4.0/



Users must provide appropriate attribution, retain applicable

notices, link to the license, and indicate modifications.



The dataset license is distinct from the license

of the MedMNIST software and the CAFA-TTA source code.



### 2.3 Dataset Configuration



| Split | Number of Images |
|---|---:|
| Training | 89,996 |
| Validation | 10,004 |
| Test | 7,180 |
| Total | 107,180 |



Image dimensions: 28 x 28 pixels.



Color format: RGB.



Number of source classes: 9.



### 2.4 Research Usage



PathMNIST is used for:



- Source model training.

- Source prototype extraction.

- Test-time adaptation development.

- Hyperparameter selection using validation experiments.

- Ablation studies.

- Synthetic domain-shift evaluation.

- Statistical and computational efficiency analysis.



Three ResNet-18 source models are evaluated:



- Enhanced.

- Robust.

- Balanced.



Target labels are used for evaluation metrics,

not for test-time parameter adaptation.



### 2.5 Modifications and Preprocessing



The CAFA-TTA pipeline applies source normalization

using the following fixed parameters:



- Mean: \[0.5, 0.5, 0.5]

- Standard deviation: \[0.5, 0.5, 0.5]



Controlled domain-shift experiments include:



- Brightness changes.

- Contrast changes.

- Saturation changes.

- Gaussian noise.

- Gaussian blur.

- Hue shifts.



These transformations are applied for experimental purposes.



They do not represent additional original medical datasets.



### 2.6 Attribution



Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L.,

Ke, B., Pfister, H., & Ni, B. (2023).



MedMNIST v2: A large-scale lightweight benchmark

for 2D and 3D biomedical image classification.



Scientific Data, 10, 41.



https://doi.org/10.1038/s41597-022-01721-8



Original histopathology dataset:



Kather, J. N., Halama, N., & Marx, A. (2018).



100,000 histological images of human colorectal

cancer and healthy tissue.



Zenodo.



https://doi.org/10.5281/zenodo.1214456



---



## 3. STARC-9



### 3.1 Source and Ownership



STARC-9 is a colorectal histopathology dataset

developed by the original STARC-9 research team.



Official sources:



- Dataset:

  https://huggingface.co/datasets/Path2AI/STARC-9



- Repository:

  https://github.com/Path2AI/STARC-9



- Publication:

  https://arxiv.org/abs/2511.00383



- OpenReview:

  https://openreview.net/forum?id=rGWjTlK6Ev



### 3.2 License Declaration



The pinned dataset README declares the CC BY 4.0 license.



License text:



https://creativecommons.org/licenses/by/4.0/



The exact dataset revision used by CAFA-TTA is:



`06e80900204d099b2f0409772f94baf1687f754f`



Pinned README:



https://huggingface.co/datasets/Path2AI/STARC-9/blob/06e80900204d099b2f0409772f94baf1687f754f/README.md



The pinned README contains a CC BY 4.0 declaration,

but the structured dataset card contains incomplete fields.



The declaration is therefore recorded as published by

the dataset authors, rather than as an independent

legal or privacy certification.



### 3.3 External Evaluation Subset



CAFA-TTA uses the following archive:



`STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip`



Archive SHA-256:



`088281de09bfe4d57b4a8e356a99cdb9b002a08a0ff7c44910aee0b4de81d3f3`



| Property | Value |
|---|---|
| Original archive images | 18,000 |
| Original tissue categories | 9 |
| Selected target images | 12,000 |
| Selected target categories | 6 |
| Samples per selected category | 2,000 |
| Excluded images | 6,000 |
| Model output classes | 9 |
| Stain-normalized images | Yes |
| Prepared image size | 28 x 28 RGB |



The original archive is downloaded directly

from the dataset provider.



It is not redistributed in this repository.



### 3.4 Target Label Mapping



The following approximate semantic mapping is used

between STARC-9 and PathMNIST:



| STARC-9 Class | PathMNIST Class | Source Class ID |
|---|---|---:|
| ADI | Adipose | 0 |
| LYM | Lymphocytes | 3 |
| MUC | Mucus | 4 |
| MUS | Smooth muscle | 5 |
| NOR | Normal colon mucosa | 6 |
| TUM | Colorectal adenocarcinoma epithelium | 8 |



Excluded STARC-9 classes:



- BLD

- FCT

- NCS



Source classes absent from the selected target subset:



- Background (ID 1).

- Debris (ID 2).

- Cancer-associated stroma (ID 7).



The mapping is an experimental semantic approximation

and does not establish exact equivalence between

the original annotation protocols.



### 3.5 Modifications and Preprocessing



CAFA-TTA performs the following preprocessing:



1. Select six compatible tissue categories.

2. Sort original archive paths.

3. Decode RGB images.

4. Resize images to 28 x 28 using Pillow bilinear interpolation.

5. Preserve the original nine-class source label IDs.

6. Store processed images locally as uint8 arrays.

7. Apply fixed source normalization during evaluation.



No synthetic corruption is applied to STARC-9.



No target-based normalization tuning is performed.



The model retains all nine source output classes.



Source prototypes are not restricted to the six target classes.



Predictions into absent source classes are counted as errors.



### 3.6 External Evaluation Protocol



STARC-9 is used exclusively for external evaluation

and unsupervised test-time adaptation in this study.



The source models are not retrained using STARC-9.



The CAFA-TTA hyperparameters are fixed before

external model evaluation.



Ground-truth target labels are used for subset selection

and metric calculation, but not for adaptation.



The evaluation protocol is recorded in:



`experiments/starc9_external/external_protocol.json`



Protocol SHA-256:



`0880ba90a14046c10e006f99b53caab6e013d46f1c62b56d38cebd3a992e9295`



### 3.7 Data Quality and Limitations



Dataset inspection identified:



- 49 exact duplicate image groups.

- 98 images involved in duplicate groups.

- 49 additional duplicate images.

- 11,951 unique resized pixel groups.

- No cross-label duplicate groups.



All 12,000 selected images are retained

in the primary evaluation.



Duplicate sensitivity analysis is performed using

saved predictions without rerunning adaptation.



Patient and slide identifiers are unavailable

in the prepared evaluation dataset.



Consequently, patch-level results must not be interpreted

as independent patient-level observations.



Resizing histopathology images to 28 x 28 also reduces

available morphological detail.



### 3.8 Attribution



Subramanian, B., Jeyaraj, R., Peterson, M. N.,

Guo, T., Shah, N., Langlotz, C., Ng, A. Y.,

& Shen, J. (2025).



STARC-9: A Large-scale Dataset for Multi-Class Tissue

Classification for CRC Histopathology.



The Thirty-Ninth Annual Conference on Neural Information

Processing Systems, Datasets and Benchmarks Track.



https://openreview.net/forum?id=rGWjTlK6Ev



https://arxiv.org/abs/2511.00383



---



## 4. Derived Research Artifacts



The CAFA-TTA project generates the following

research artifacts:



- Source model checkpoints.

- Source feature prototypes.

- Evaluation metrics.

- Per-sample classification predictions.

- Confusion matrices.

- Statistical analysis reports.

- Research tables and figures.

- Reproducibility manifests.



These artifacts are generated through computational

processing of the original datasets.



Their publication must respect applicable dataset licenses,

third-party rights, and privacy requirements.



A dataset license does not automatically establish

the licensing status of every derived model or artifact.



---



## 5. Prediction File Contents



The project maintains 453 final evaluation prediction files.



| Evaluation | Prediction Files |
|---|---:|
| PathMNIST strong shift | 63 |
| PathMNIST severity | 90 |
| PathMNIST corruptions | 270 |
| STARC-9 external evaluation | 30 |
| Total | 453 |



All 453 files passed structural content inspection.



Each inspected NPZ file contains only the following

integer-valued fields:



- `indices`: dataset-relative sample indices.

- `labels`: ground-truth class IDs.

- `predictions`: predicted class IDs.

- `batch_size`: evaluation batch size.



The inspected prediction files do not contain

original medical image arrays or explicit patient identifiers.



However, sample indices and ground-truth labels are

dataset-derived information and may be linkable

to the corresponding public datasets.



Structural inspection is not a formal anonymization

or privacy certification.



Publication of per-sample predictions remains subject

to review of applicable rights and privacy obligations.



---



## 6. Repository Publication Policy



| Artifact | Publication Policy |
|---|---|
| Project source code | Intended for publication |
| Research scripts | Intended for publication |
| Aggregate evaluation metrics | Intended for publication |
| Statistical tables | Intended for publication |
| Research figures | Intended for publication |
| Reproducibility manifests | Intended for publication |
| Per-sample predictions | Pending redistribution and privacy review |
| Trained checkpoints | Pending release and rights review |
| Source prototypes | Pending release and rights review |
| Original medical images | Not redistributed |
| Processed image datasets | Recreated locally |
| Virtual environment | Not redistributed |
| Temporary and backup files | Not redistributed |



Dataset preparation scripts are provided to support

local reconstruction of the evaluation datasets.



Users should obtain the original images directly

from the official dataset providers.



---



## 7. Ethical and Research Considerations



CAFA-TTA is a research prototype.



It has not been clinically validated or approved

for medical diagnosis.



The experimental results do not establish:



- Patient-level diagnostic performance.

- Clinical safety or effectiveness.

- Generalization to all hospitals or populations.

- Elimination of negative adaptation.

- Independence of histopathology image patches.



The absence of patient identifiers in the project files

does not independently establish full anonymization.



Users remain responsible for complying with applicable

ethical requirements, privacy rules, and dataset terms.



---



## 8. License Separation



Third-party datasets retain their original licensing

and attribution requirements.



The licensing of the original CAFA-TTA source code

is documented separately by the project license.



No project-level license supersedes the rights

of the original dataset creators.



The release of model checkpoints and source prototypes

requires a separate review of applicable rights.



---



## 9. Acknowledgments



The CAFA-TTA project gratefully acknowledges:



- The MedMNIST research team.

- The creators of NCT-CRC-HE-100K.

- The creators of CRC-VAL-HE-7K.

- The STARC-9 research team.

- The institutions and contributors responsible

  for collecting and preparing the original datasets.



This project is independent research and does not imply

endorsement by the original dataset creators.



---



## 10. Disclaimer



This document records dataset sources, declared licenses,

research transformations, and intended publication practices.



It is provided for research transparency and attribution,

not as legal advice or a certification of redistribution rights.



Users should consult the original dataset licenses

and any applicable additional terms before reuse.
