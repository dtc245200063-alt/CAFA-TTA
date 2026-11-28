"""User-operated, isolated Docker training launcher for CAFA-TTA scripts.

The original project's code and artifacts are mounted read-only at /project.
High-resolution external datasets can be mounted read-only at /datasets.
Each requested run has its own writable /outputs/<run-name>/workspace.
No experiment starts until the user invokes docker compose run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import runpy
import shutil
import sys

SOURCE = Path("/project")
DATASETS = Path("/datasets")
OUTPUT = Path("/outputs")
SCRIPTS = {
    "baseline": "train_baseline.py",
    "enhanced": "train_enhanced.py",
    "robust": "train_robust.py",
    "balanced": "train_balanced.py",
}
NEEDED_CHECKPOINTS = {
    "baseline": [],
    "enhanced": ["resnet18_pathmnist_best.pt"],
    "robust": ["resnet18_pathmnist_enhanced_best.pt"],
    "balanced": ["resnet18_pathmnist_enhanced_best.pt"],
}


def _find_latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "workspace" / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found in previous run: {ckpt_dir}")
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")
    return pts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="CAFA-TTA Docker training (manual start only)")
    parser.add_argument("--model", choices=SCRIPTS, help="Which training stage script to run")
    parser.add_argument("--run-name", help="Unique output directory name (letters, digits, underscore, hyphen)")
    parser.add_argument("--from-run", help="Use prerequisite checkpoint from an earlier Docker run")
    parser.add_argument("--modality", choices=["histopathology", "dermatology", "chest_xray", "fundus"], default=None)
    parser.add_argument("--size", type=int, default=None, help="Override input image size (validated by modality)")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "densenet121"], default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset path inside container (e.g. /datasets/isic)")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch size override")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--check-gpu", action="store_true", help="Only inspect GPU, then exit")
    args = parser.parse_args()

    import torch
    print("Python:", sys.version.split()[0], flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    if args.check_gpu:
        return
    if not args.model or not args.run_name:
        parser.error("Training requires both --model and --run-name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("--run-name may contain only A-Z, 0-9, _ and - (up to 64 chars)")
    if args.epochs is not None and not (1 <= args.epochs <= 200):
        parser.error("--epochs must be between 1 and 200")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA unavailable in Docker: training NOT started")

    run_dir = OUTPUT / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run name already exists: {run_dir}. Pick another name.")

    script_file = SOURCE / "scripts" / SCRIPTS[args.model]
    if not script_file.is_file():
        raise FileNotFoundError(script_file)

    is_legacy = args.modality is None and args.data_dir is None
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "scripts").mkdir()
    (workspace / "checkpoints").mkdir()

    staged_script = workspace / "scripts" / SCRIPTS[args.model]
    shutil.copy2(script_file, staged_script)

    if is_legacy:
        data_file = SOURCE / "data/raw/medmnist/pathmnist.npz"
        if not data_file.is_file():
            raise FileNotFoundError(data_file)
        (workspace / "data/raw/medmnist").mkdir(parents=True)
        (workspace / "data/raw/medmnist/pathmnist.npz").symlink_to(data_file)

    if args.model in {"robust", "balanced"}:
        shift_src = SOURCE / "experiments/domain_shift/metrics.json"
        if shift_src.is_file():
            shift_dst = workspace / "experiments/domain_shift"
            shift_dst.mkdir(parents=True)
            shutil.copy2(shift_src, shift_dst / "metrics.json")

    init_ckpt_arg: Path | None = None
    if args.from_run:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.from_run):
            parser.error("--from-run contains invalid characters")
        if args.from_run == args.run_name:
            parser.error("--from-run must differ from --run-name")
        if not NEEDED_CHECKPOINTS[args.model]:
            parser.error("--from-run is not needed for baseline")
        prev_ckpt = _find_latest_checkpoint(OUTPUT / args.from_run)
        staged_init = workspace / "checkpoints" / prev_ckpt.name
        shutil.copy2(prev_ckpt, staged_init)
        init_ckpt_arg = staged_init
    elif is_legacy:
        for ckpt in NEEDED_CHECKPOINTS[args.model]:
            src_ckpt = SOURCE / "checkpoints" / ckpt
            if not src_ckpt.is_file():
                raise FileNotFoundError(f"Required prerequisite checkpoint absent: {src_ckpt}")
            shutil.copy2(src_ckpt, workspace / "checkpoints" / ckpt)

    cli_args = [str(staged_script)]
    if args.modality:
        cli_args.extend(["--modality", args.modality])
    if args.size is not None:
        cli_args.extend(["--size", str(args.size)])
    if args.backbone:
        cli_args.extend(["--backbone", args.backbone])
    if args.no_pretrained:
        cli_args.append("--no-pretrained")
    if args.data_dir:
        cli_args.extend(["--data-dir", args.data_dir])
    if args.epochs is not None:
        cli_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cli_args.extend(["--batch-size", str(args.batch_size)])
    if args.grad_accum is not None:
        cli_args.extend(["--grad-accum", str(args.grad_accum)])
    if args.lr is not None:
        cli_args.extend(["--lr", str(args.lr)])
    if init_ckpt_arg is not None:
        cli_args.extend(["--init-checkpoint", str(init_ckpt_arg)])

    print("\nMODEL:", args.model)
    print("RUN:", args.run_name)
    print("SOURCE:", SOURCE, "(READ ONLY)")
    print("OUTPUT:", run_dir)
    print("CLI ARGS:", " ".join(cli_args[1:]) or "(legacy defaults)", flush=True)

    os.chdir(workspace)
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(workspace))
    old_argv = sys.argv
    sys.argv = cli_args
    try:
        runpy.run_path(str(staged_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
