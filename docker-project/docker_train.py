"""User-operated, isolated Docker training launcher for the original CAFA-TTA scripts.

The original project's code and artifacts are mounted read-only at /project.
Each requested run has its own writable /outputs/<run-name>/workspace.
No experiment starts until the user invokes docker compose run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import runpy
import shutil
import sys
import os

SOURCE = Path('/project')
OUTPUT = Path('/outputs')
SCRIPTS = {
    'baseline': 'train_baseline.py',
    'enhanced': 'train_enhanced.py',
    'robust': 'train_robust.py',
    'balanced': 'train_balanced.py',
}
NEEDED_CHECKPOINTS = {
    'baseline': [],
    'enhanced': ['resnet18_pathmnist_best.pt'],
    'robust': ['resnet18_pathmnist_enhanced_best.pt'],
    'balanced': ['resnet18_pathmnist_enhanced_best.pt'],
}


def main() -> None:
    parser = argparse.ArgumentParser(description='CAFA-TTA Docker training (manual start only)')
    parser.add_argument('--model', choices=SCRIPTS, help='Which ORIGINAL model script to run')
    parser.add_argument('--epochs', type=int, help='Optional smoke-test epoch count (staged copy only)')
    parser.add_argument('--run-name', help='Unique output directory name (letters, digits, underscore, hyphen)')
    parser.add_argument('--check-gpu', action='store_true', help='Only inspect GPU, then exit')
    args = parser.parse_args()

    import torch
    print('Python:', sys.version.split()[0], flush=True)
    print('PyTorch:', torch.__version__, flush=True)
    print('CUDA available:', torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    if args.check_gpu:
        return
    if not args.model or not args.run_name:
        parser.error('Training requires both --model and --run-name')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_name):
        parser.error('--run-name may contain only A-Z, 0-9, _ and - (up to 64 chars)')
    if args.epochs is not None and not (1 <= args.epochs <= 100):
        parser.error('--epochs must be between 1 and 100')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU CUDA unavailable in Docker: training NOT started')
    run_dir = OUTPUT / args.run_name
    if run_dir.exists():
        raise FileExistsError(f'Run name already exists: {run_dir}. Pick another name; existing results are never overwritten.')
    script_file = SOURCE / 'scripts' / SCRIPTS[args.model]
    data_file = SOURCE / 'data/raw/medmnist/pathmnist.npz'
    for file in [script_file, data_file]:
        if not file.is_file():
            raise FileNotFoundError(file)
    for ckpt in NEEDED_CHECKPOINTS[args.model]:
        file = SOURCE / 'checkpoints' / ckpt
        if not file.is_file():
            raise FileNotFoundError(f'Required prerequisite checkpoint absent: {file}')
    if args.model in {'robust', 'balanced'}:
        required = SOURCE / 'experiments/domain_shift/metrics.json'
        if not required.is_file():
            raise FileNotFoundError(required)

    # Preserve original files. All writes by the training script occur in this staged workspace.
    workspace = run_dir / 'workspace'
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / 'scripts').mkdir()
    (workspace / 'checkpoints').mkdir()
    (workspace / 'data/raw/medmnist').mkdir(parents=True)
    staged_script = workspace / 'scripts' / SCRIPTS[args.model]
    shutil.copy2(script_file, staged_script)
    (workspace / 'data/raw/medmnist/pathmnist.npz').symlink_to(data_file)
    for ckpt in NEEDED_CHECKPOINTS[args.model]:
        shutil.copy2(SOURCE / 'checkpoints' / ckpt, workspace / 'checkpoints' / ckpt)
    if args.model in {'robust', 'balanced'}:
        path = workspace / 'experiments/domain_shift'
        path.mkdir(parents=True)
        shutil.copy2(SOURCE / 'experiments/domain_shift/metrics.json', path / 'metrics.json')

    if args.epochs is not None:
        src = staged_script.read_text(encoding='utf-8')
        updated, n = re.subn(r'(?m)^EPOCHS[ \t]*=[ \t]*\d+[ \t]*$', f'EPOCHS = {args.epochs}', src)
        if n != 1:
            raise RuntimeError('Could not safely update EPOCHS in staged script')
        staged_script.write_text(updated, encoding='utf-8')

    print('\nMODEL:', args.model)
    print('RUN:', args.run_name)
    print('SOURCE:', SOURCE, '(READ ONLY)')
    print('OUTPUT:', run_dir)
    print('TRAINING SCRIPT:', staged_script)
    print('EPOCH OVERRIDE:', args.epochs if args.epochs else 'original configuration')
    print('No original code/checkpoints will be modified.\n', flush=True)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    try:
        runpy.run_path(str(staged_script), run_name='__main__')
    finally:
        print('\nRun folder:', run_dir, flush=True)


if __name__ == '__main__':
    main()
