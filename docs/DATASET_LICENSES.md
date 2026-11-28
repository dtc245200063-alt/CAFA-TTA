

# Dataset Licenses and Attribution



## CAFA-TTA Research Project



**Confidence-Aware Feature Alignment for Test-Time Adaptation in Cross-Domain Medical Image Classification**



This document describes the datasets used in CAFA-TTA,

their declared licenses, attribution requirements,

preprocessing procedures, and publication policy.



The project acknowledges the original dataset creators

and does not claim ownership of the original medical images.



Last reviewed: 2026-11-27 (Section 11 added; Sections 1-10 unchanged since 2026-11-19).



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



---



## 11. Multi-Modality Preparation Datasets



`scripts/prepare_medical_datasets.py` prepares additional datasets for the
multi-modality continual-learning extension. These datasets are not used
by the frozen results in `experiments/` and are not redistributed here.

Output layout: `<root>/<modality>/<name>/<class>/*`, with
`train/`, `val/`, `test/` split folders where the source provides splits.
The default root is `CAFA_DATASET_ROOT` or `C:\Code\DataSet`.



### 11.1 Dataset Table



"Auto-download" means the script fetches the files itself.
"Verified" licenses were checked on 2026-11-27 against the source named
in the license column. Entries marked "not re-verified" were not checked in
that review and should be confirmed at the source before reuse.

| Dataset | Modality | Resolution | License | Access method | Auto-download |
|---|---|---|---|---|---|
| Kather 2016 (`kather2016`) | histopathology | 150 x 150 px | CC BY 4.0 declared on Zenodo record 53169 (not re-verified) | Zenodo | yes |
| NCT-CRC-HE-100K (`nct_crc_100k`) | histopathology | 224 x 224 px, 0.5 µm/px | CC BY 4.0 (verified, Zenodo record 1214456) | Zenodo | yes |
| CRC-VAL-HE-7K (`crc_val_he_7k`) | histopathology | 224 x 224 px, 0.5 µm/px | CC BY 4.0 (verified, Zenodo record 1214456) | Zenodo | yes |
| HAM10000 (`ham10000`) | dermatology | 600 x 450 px | CC BY-NC 4.0 declared by the original release (not re-verified) | Kaggle CLI with the user's own token | yes, via Kaggle token |
| Chest X-Ray Pneumonia (`chest_xray_pneumonia`) | chest_xray | variable | Declared on the Kaggle and Mendeley pages (not re-verified) | Kaggle CLI with the user's own token | yes, via Kaggle token |
| APTOS 2019 (`aptos2019`) | fundus | variable | Kaggle competition rules (not re-verified) | Kaggle CLI; the user must accept the competition rules on Kaggle first | yes, via Kaggle token |
| DermaMNIST+ (`dermamnist_224`) | dermatology | 224 x 224 px | CC BY-NC 4.0 (verified, `medmnist.INFO` 3.0.2 and medmnist.com) | `medmnist` package | yes |
| PneumoniaMNIST+ (`pneumoniamnist_224`) | chest_xray | 224 x 224 px | CC BY 4.0 (verified, same sources) | `medmnist` package | yes |
| RetinaMNIST+ (`retinamnist_224`) | fundus | 224 x 224 px | CC BY 4.0 (verified, same sources) | `medmnist` package | yes |
| BloodMNIST+ (`bloodmnist_224`) | hematology | 224 x 224 px | CC BY 4.0 (verified, same sources) | `medmnist` package | yes |
| BreastMNIST+ (`breastmnist_224`) | breast_ultrasound | 224 x 224 px | CC BY 4.0 (verified, same sources) | `medmnist` package | yes |
| PathMNIST+ (`pathmnist_224`) | histopathology | 224 x 224 px | CC BY 4.0 (verified, same sources) | `medmnist` package | yes |
| ChestMNIST+ | chest_xray | 224 x 224 px | CC BY 4.0 (verified, same sources) | Not exported (multi-label) | no |
| MHIST | histopathology | 224 x 224 px | Dataset Research Use Agreement (not re-verified) | Manual request form | no |
| Chaoyang | histopathology | 512 x 512 px | Non-commercial, no redistribution (not re-verified) | Manual request form | no |
| ISIC Archive / challenges | dermatology | variable | Per-image or per-challenge licenses (not re-verified) | Manual download from the ISIC Archive | no |
| CheXpert | chest_xray | variable | Stanford Research Use Agreement (not re-verified) | Manual registration with Stanford AIMI | no |
| MIMIC-CXR | chest_xray | variable | PhysioNet Credentialed Health Data License (not re-verified) | PhysioNet credentialed access | no |

The medmnist.com license statement says MedMNIST is CC BY 4.0 except
DermaMNIST, which is CC BY-NC 4.0. DermaMNIST is therefore limited to
non-commercial use. The MedMNIST software is licensed separately (Apache-2.0).
For entries marked "not re-verified", the listed resolution was not re-checked either.



### 11.2 Notes on Specific Datasets



**CRC-VAL-HE-7K.** The Zenodo record describes 7,180 patches from 50 patients,
with no patient overlap with NCT-CRC-HE-100K, at 224 x 224 px and 0.5 µm/px.
The Zenodo page states Macenko normalization only for NCT-CRC-HE-100K and
does not state it for CRC-VAL-HE-7K. A Hugging Face mirror describes
CRC-VAL-HE-7K as not Macenko-normalized. This was not confirmed against the
original publication, so the normalization status should be treated as uncertain.
CRC-VAL-HE-7K is also the source of the PathMNIST test split, so it is not
an independent external set with respect to PathMNIST-trained models.

**MedMNIST+ export.** Images are exported as PNG with the official
train/val/test splits. Class folders are named `<label id>_<label name>`
from `medmnist.INFO`, so ImageFolder class order matches the MedMNIST label ids.
RetinaMNIST labels are ordinal grades (0-4) exported as classes.
MedMNIST+ images are resized by the MedMNIST authors; they are not the
original-resolution source images.

**ChestMNIST.** ChestMNIST is multi-label (14 findings per image, any number
may be positive). A single-class ImageFolder layout cannot represent it
without discarding or duplicating labels, so the script does not export it.

**Overlap between sources.** Some prepared datasets share origins.
PathMNIST is derived from NCT-CRC-HE-100K and CRC-VAL-HE-7K, and Kather 2016
comes from the same research group as NCT-CRC-HE-100K. Overlap between them
has not been measured. Run `--check-duplicates` before combining datasets.



### 11.3 Duplicate and Leakage Check



```powershell
python scripts/prepare_medical_datasets.py --check-duplicates
```

The check hashes every image under the root, except `_downloads/`:

- Exact duplicates: SHA-256 of the decoded RGB pixels.
- Near duplicates: 64-bit average hash (8 x 8 grayscale), grouped when the
  Hamming distance is at most `--hamming-threshold` (default 5).

The CSV (`<root>/duplicates.csv` by default) lists each group with the
`modality/dataset` of every member, so cross-dataset groups are visible.
Average hash is a coarse heuristic. Near-duplicate groups can contain false
positives (for example, uniform background tiles) and can miss cropped or
recolored copies. Groups should be inspected before any image is removed.



### 11.4 Gated Datasets: Manual Access Only



The script does not download these datasets, does not accept agreements,
and does not store credentials. Each user must request access personally
and accept the provider's terms. After download, place the images in the
ImageFolder layout under `<root>/<modality>/<name>/<class>/`.

- **MHIST** (https://bmirds.github.io/MHIST/): submit the request form and
  accept the Dataset Research Use Agreement. The emailed download links are
  reported to expire after 4 hours.
- **Chaoyang** (https://bupt-ai-cz.github.io/HSA-NRL/): submit the request form.
  The declared terms are non-commercial use and no redistribution.
- **ISIC** (https://www.isic-archive.com/): download from the ISIC Archive or
  challenge pages. Licenses differ per image and per challenge; check the
  license field of each image or challenge release.
- **CheXpert** (https://stanfordmlgroup.github.io/competitions/chexpert/):
  register with Stanford AIMI and accept the Research Use Agreement.
- **MIMIC-CXR** (https://physionet.org/content/mimic-cxr/): requires a
  credentialed PhysioNet account, completed human-subjects training, and a
  signed data use agreement.

Access terms for these datasets were not re-verified in the 2026-11-27 review.
Users should read the current terms at each provider before requesting access.
