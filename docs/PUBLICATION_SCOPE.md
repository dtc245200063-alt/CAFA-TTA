# Publication Scope and Artifact Availability

## CAFA-TTA Research Project

This document defines the intended publication scope of
CAFA-TTA and distinguishes publicly distributable research
materials from locally retained artifacts.

## 1. Intended Public Materials

Subject to final publication review, the repository includes:

- Original CAFA-TTA source code.
- Training and evaluation scripts.
- Statistical analysis scripts.
- Aggregate evaluation metrics.
- Aggregate statistical tables.
- Research figures.
- Experimental configurations.
- Reproducibility manifests.
- Environment and dependency specifications.
- Research documentation.

## 2. Locally Retained Research Artifacts

The following artifacts are retained locally and excluded
from the intended public repository.

| Artifact | Count |
|---|---:|
| PathMNIST strong-shift prediction files | 63 |
| PathMNIST severity prediction files | 90 |
| PathMNIST corruption prediction files | 270 |
| STARC-9 external prediction files | 30 |
| Total prediction files | 453 |
| Model checkpoints and source prototypes | 8 |
| Archived STARC-9 metadata files | 4 |
| Detailed statistical CSV files | 2 |

The two detailed statistical CSV files are:

- `experiments/starc9_external/statistical_analysis/duplicate_group_members.csv`
- `experiments/statistical_analysis/paired_batch_outcomes.csv`

These files remain part of the local research record.

Their exclusion does not alter the reported aggregate
research results.

## 3. Dataset Availability

Original and processed medical image datasets are not
redistributed through this repository.

Users should obtain datasets directly from their
respective official providers.

Dataset preparation and evaluation scripts are intended
to support local reconstruction.

See [Dataset Licenses](DATASET_LICENSES.md).

## 4. Frozen Research Manifests

Original research manifests retain their recorded
SHA-256 values and historical artifact references.

Some referenced files are intentionally excluded
from the public repository.

Consequently, a fresh repository clone does not contain
every input and output of the original research workflow.

Missing locally retained artifacts must not be interpreted
as evidence of file corruption.

Available files should still match their recorded hashes.

Original manifests must not be rewritten merely to remove
references to locally retained artifacts.

## 5. Reproducibility Boundaries

The repository provides the algorithm, configurations,
documented methodology, and aggregate research results.

Complete independent reproduction requires:

- Access to the original datasets.
- Reconstruction of the preprocessing pipeline.
- Reconstruction or authorized access to source models.
- Reconstruction of source feature prototypes.
- Execution of the documented evaluation workflow.
- Regeneration of detailed prediction artifacts.
- A compatible software and hardware environment.

Exact byte-level reconstruction of every locally retained
artifact has not been independently demonstrated.

The public repository should therefore not be described
as a complete self-contained reproduction package.

See [Reproducibility Guide](REPRODUCIBILITY.md).

## 6. Licensing and Redistribution

The project-level MIT license applies to original
CAFA-TTA materials that the author has rights to license.

Third-party datasets and software retain their
respective licenses and attribution requirements.

See:

- [Project License](../LICENSE)
- [Third-Party Notices](../THIRD_PARTY_NOTICES.md)
- [Dataset Licenses](DATASET_LICENSES.md)

Publication of model weights, source prototypes,
sample-linked metadata, or detailed prediction artifacts
requires separate review.

## 7. Research and Clinical Limitations

CAFA-TTA is an experimental research framework.

It is not a clinically validated medical diagnosis system.

Artifact inspection does not constitute formal privacy,
anonymization, security, or legal certification.

## 8. Future Artifact Releases

Additional research artifacts may be considered for
release following appropriate rights, privacy, security,
and reproducibility review.

No future release is guaranteed by this document.
