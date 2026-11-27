# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CAFA-TTA: research prototype for confidence-aware test-time adaptation (TTA) of ResNet-18 histopathology classifiers. Source domain is PathMNIST (9 classes); external evaluation on a 6-class STARC-9 subset; a later "unified" line trains one 17-way head on PathMNIST + Kather 2016. Windows-first (reference env: Python 3.12.10, PyTorch 2.10.0+cu128, RTX 3050 4 GB). Scripts are sized for 4 GB VRAM.

## Commands

Setup (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_cuda.py
```

Tests: install dev deps with `python -m pip install -r requirements-dev.txt` (adds `pytest`). Run from repo root with `python -m pytest` so the repo root is on `sys.path` (`tests/test_training_utils.py` imports `core.*` without its own path insert).

```powershell
python -m pytest
python -m pytest tests/test_cafa_soft_adapter.py
python -m pytest tests/test_cafa_soft_adapter.py::test_name
```

No linter/formatter is configured.

Main pipeline (see README "Quick Start" and `docs/REPRODUCIBILITY.md` for full order and prerequisites):

- Data: `scripts/download_data.py` (PathMNIST to `data/raw/medmnist/pathmnist.npz`), `scripts/prepare_starc9_dataset.py`, `scripts/prepare_kather2016.py`.
- Source training is a chained sequence; each stage needs the previous checkpoint: `train_baseline.py` -> `train_enhanced.py` -> `train_robust.py` / `train_balanced.py` (both from enhanced). Unified model: `scripts/train_unified.py`.
- Prototypes: `scripts/extract_prototypes_models.py` (or `extract_unified_prototypes.py`).
- Final evaluation: `evaluate_cafa_final_test.py` must run before `evaluate_cafa_final_test_severity.py` (severity reuses strong-shift results), then `evaluate_cafa_final_test_corruptions.py`.
- Analysis: `verify_final_test_predictions.py`, `analyze_statistical_significance.py`, `generate_research_figures.py`.
- GUI demo (source-model inference only, no adaptation): `OPEN-CAFA-TTA-GUI.bat` or `python gui/app.py`; loads `checkpoints/cafa_unified_balanced_best.pt`.

Docker training (`docker-project/`, README in Vietnamese): `docker compose build`, `docker compose run --rm trainer --check-gpu`, `docker compose run --rm trainer --model baseline --run-name <name>`. Project is mounted read-only at `/project`; outputs go to `docker-project/docker_runs/<run-name>/`. CPU variant: `compose.cpu.yaml`.

## Architecture

Layering: `kernel/` (pure loss/math functions) -> `core/` (adapter classes) -> `scripts/` (one entry-point script per experiment). Scripts are standalone and add the repo root to `sys.path` themselves; there is no package install. Several scripts import helpers from `scripts/train_unified.py` by also adding `scripts/` to `sys.path`. `scripts/legacy/` holds superseded v1/v2, BN-only and TENT scripts (their `ROOT` is `parents[2]`); add new entry points to `scripts/`, not there. All `docker-project/*.py` files belong to documented workflows (`Dockerfile`, `Dockerfile.cpu`, `README_CPU.md`).

- `models/feature_extractor.py`: `ResNetFeatureExtractor` wraps a torchvision ResNet and returns `(features[B,512], logits)`. The CAFA adapters require this tuple interface; `kernel/tent.py::TentAdapter` expects a plain model returning only logits.
- `kernel/confidence.py`: confidence = `1 - H(p)/log(K)` (entropy heuristic, not calibrated). `kernel/feature_alignment.py`: L2-normalized class-mean prototypes and alignment loss.
- Adapter versions (all keep v1/v2 for comparison; v3 is the final method):
  - v1 `core/cafa_adapter.py::CAFAAdapter` — base class. Freezes everything, enables only BatchNorm2d affine params, disables running stats (uses target-batch stats), Adam optimizer. Hard confidence threshold.
  - v2 `core/cafa_agreement_adapter.py` — adds classifier/nearest-prototype agreement filter.
  - v3 `core/cafa_soft_adapter.py::CAFASoftAdapter` — soft entropy weight `entropy_floor + (1-floor)*confidence` on all samples; alignment only on agreement samples, skipped below `min_samples`; grad clip 1.0; skips update on non-finite loss.
- Evaluation is online batch-prequential: predict each batch, then adapt on it. Target labels are never used for adaptation. Adapters mutate the model in place, so evaluators deep-copy/reload the source model per run/seed.
- `core/training_utils.py` (newer, shared by `train_*.py` and `docker-project/docker_train.py`): modality profiles (histopathology/dermatology/chest_xray/fundus) that set input size, backbone, and 4 GB batch/grad-accum defaults; `add_common_train_args` defines the shared CLI (`--modality`, `--size`, `--backbone`, `--data-dir`, `--init-checkpoint`, `--output-checkpoint`, ...). Omitting `--modality`/`--data-dir` gives legacy 28px PathMNIST behavior. Note `ResNetFeatureExtractor` only supports ResNet backbones, not `densenet121`.

## Continual-learning line (multi-modality, 224 px)

Separate from the frozen research pipeline above; it never writes `checkpoints/` or `experiments/`.

```powershell
python scripts/train_modality.py --modality histopathology --data-dir <ImageFolder>   # registers v1
python -m pip install -r requirements-demo.txt
python app/demo_gradio.py                                                             # http://127.0.0.1:7860
python scripts/consolidate.py --modality histopathology --source-data-dir <ImageFolder>
python scripts/consolidate.py --modality histopathology --rollback
```

- `core/trainer.py`: generic loop (label smoothing, warmup+cosine, EMA with warmup, class-balanced sampler, optional distillation) and temperature scaling.
- `core/continual.py`: `ContinualSession` runs CAFA-TTA v3 with BN always in eval mode (`StableBNFeatures`); BN running statistics start at source and follow the stream by EMA, updated every 64 images from a class-balanced memory (8 latest images per predicted class, RoTTA-style; all images, since the OOD gate cannot separate shifted tissue from non-tissue), while CAFA gradient updates use only in-distribution images. A stats-only reference model drives the OOD gate and `CollapseMonitor`; drift resets to source. `OODGate` (energy + prototype similarity, calibrated on val) decides what may be learned from; confidence alone is not enough (an untrained-domain image can get confidence 1.0). `ExperienceStore` keeps pixel-hashed PNGs + `records.jsonl` under `data/experience/<modality>/`. `build_checkpoint` packages weights with temperature, prototypes and OOD thresholds.
- `core/registry.py`: `models_registry/<modality>/vN/model.pt` + `registry.json`; loads with `weights_only=True`. `passes_gate` compares candidate vs current on the same fixed split; consolidation promotes only on pass and logs every attempt to `consolidation_log.jsonl`.
- `data/experience/` and `models_registry/` are gitignored (user images stay local). The demo binds to 127.0.0.1 only.

## Research-integrity constraints

- `experiments/` holds frozen research outputs (predictions, tables, figures, manifests) referenced by docs and hash manifests (`experiments/reproducibility/`, `experiments/research_outputs/research_manifest.json`, `experiments/starc9_external/external_protocol.json`). Run new experiments into a separate output directory; do not overwrite these files.
- The STARC-9 evaluator checks frozen config and artifact hashes. Do not bypass integrity checks by editing reference hashes.
- `data/`, `checkpoints/`, `*.pt`, `*.npz`, `runs/`, and `.env` are gitignored; datasets and most checkpoints are not in a fresh clone.
- Unified 17-class labels are dataset-prefixed (`PathMNIST:*`, `Kather:*`) and are not a harmonized clinical taxonomy; keep that distinction in code and docs.
- Docs (`docs/EXPERIMENTS.md`, `docs/PUBLICATION_SCOPE.md`, statistical audit/ablation docs) use deliberately hedged claims (no clinical validation, paired comparisons not independent). Keep that tone when editing results text.
