# CAFA-TTA - Docker CPU

This workflow was tested on Ubuntu VMware with Docker Compose and a CPU-only PyTorch image.

## Project structure
- ../core, ../kernel, ../models: main CAFA-TTA source code.
- train_full_cpu.py: PathMNIST baseline training (9 classes).
- train_kather_cpu.py: Kather 2016 training (8 classes).
- extract_kather_prototypes.py: extract Kather source prototypes.
- evaluate_docker_cpu.py: evaluate PathMNIST baseline.
- evaluate_kather_cpu.py: evaluate Kather baseline.
- evaluate_kather_cafa.py: compare Source Only, TENT and CAFA v3.
- dataset_splits/kather2016_seed42.json: reproducible image-level Kather split.

## Requirements
Docker Engine with Docker Compose; download datasets separately. Dataset images and trained checkpoints are not committed to Git.

## Build CPU image
From the docker-project directory:
docker compose -f compose.cpu.yaml build

## Train PathMNIST
Put the PathMNIST MedMNIST dataset under ../data/raw/medmnist/.
docker compose -f compose.cpu.yaml run --rm -v "$PWD/train_full_cpu.py:/runner/train_full_cpu.py:ro" --entrypoint python cafa-cpu-smoke -u /runner/train_full_cpu.py --run-name example_pathmnist_01 --epochs 3

## Kather 2016
Source: https://zenodo.org/records/53169
Extract the 5000 image tiles into external_datasets/kather2016/extracted/Kather_texture_2016_image_tiles_5000/.
Copy dataset_splits/kather2016_seed42.json to external_datasets/kather2016/split_seed42.json.
The fixed image-level split has 3496 training, 752 validation and 752 test images across 8 classes.
docker compose -f compose.cpu.yaml run --rm -v "$PWD/train_kather_cpu.py:/runner/train_kather_cpu.py:ro" -v "$PWD/external_datasets/kather2016:/kather:ro" --entrypoint python cafa-cpu-smoke -u /runner/train_kather_cpu.py --run-name example_kather_01 --epochs 3

## Evaluation notes
For CAFA Kather evaluation, use --run-dir (containing best.pt and source_prototypes.pt) and --kather-root (containing split_seed42.json and extracted/). Other evaluators may still need configured paths.
CAFA evaluation requires Kather source prototypes and uses main project modules from /project.
Test-time adaptation uses test images without target labels for parameter updates.
The image-level Kather split does not guarantee patient-level independence.

## Important
Use a new run name for each training run. Do not overwrite existing checkpoints.
CPU experiments and the separate GPU Docker configuration are not interchangeable.
Docker builds do not automatically start training.


## CAFA Kather evaluation with explicit paths

From `docker-project/`, with `external_datasets/kather2016` and checkpoint files prepared:

```bash
docker compose -f compose.cpu.yaml run --rm -v "$PWD/evaluate_kather_cafa.py:/runner/evaluate_kather_cafa.py:ro" -v "$PWD/external_datasets/kather2016:/kather:ro" --entrypoint python cafa-cpu-smoke -u /runner/evaluate_kather_cafa.py --run-dir /outputs/example_kather_01 --kather-root /kather
```

Use a new evaluation run directory to avoid overwriting earlier results.
