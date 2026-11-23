"""Extract 17 source-class feature prototypes from TRAIN splits only.

Compatible with CAFA-TTA v3 CAFASoftAdapter and ResNetFeatureExtractor.
Does not overwrite any checkpoint. Outputs a separate torch file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from train_unified import (
    PathMNISTDataset, KatherDataset, build_model, collect_kather, split_items,
    PATHMNIST_NAMES,
)


def forward_features(backbone, x):
    """Matches models.feature_extractor.ResNetFeatureExtractor."""
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return torch.flatten(backbone.avgpool(x), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pathmnist', type=Path, required=True)
    p.add_argument('--kather-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100)
    args = p.parse_args()
    if args.output.exists():
        p.error(f'Output already exists (will not overwrite): {args.output}')
    if args.batch_size <= 0:
        p.error('--batch-size must be positive')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    labels = list(state['labels'])
    if len(labels) != 17:
        raise ValueError('Checkpoint must contain exactly 17 class labels')
    seed = int(state['seed'])
    size = int(state['input_size'])
    norm = state['normalization']
    dirs = collect_kather(args.kather_root)
    expected = [f'PathMNIST:{x}' for x in PATHMNIST_NAMES] + [f'Kather:{d.name}' for d, _ in dirs]
    if labels != expected:
        raise ValueError('Dataset class-folder order differs from checkpoint labels')
    splits = split_items([imgs for _, imgs in dirs], seed)
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(norm['mean'], norm['std']),
    ])
    train = ConcatDataset([
        PathMNISTDataset(args.pathmnist, 'train', tfm),
        KatherDataset(splits['train'], tfm),
    ])
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(labels))
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device).eval()
    sums = torch.zeros(17, 512, dtype=torch.float64)
    counts = torch.zeros(17, dtype=torch.long)
    print(f'Extracting source prototypes: {len(train)} images, {len(loader)} batches, {device}', flush=True)
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader, 1):
            features = F.normalize(forward_features(model, images.to(device)).float(), dim=1).cpu()
            targets = targets.long()
            for label in targets.unique():
                idx = int(label)
                mask = targets == idx
                sums[idx] += features[mask].double().sum(dim=0)
                counts[idx] += int(mask.sum())
            if step % args.log_every == 0 or step == len(loader):
                print(f'PROTOTYPES {step}/{len(loader)}', flush=True)
    if (counts == 0).any():
        raise RuntimeError(f'Missing class(es): {(counts == 0).nonzero().flatten().tolist()}')
    prototypes = F.normalize((sums / counts[:, None]).float(), dim=1)
    # Keep source checkpoint separate; never mutate the trained model.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'prototypes': prototypes,
        'labels': labels,
        'counts': counts,
        'source_checkpoint': args.checkpoint.name,
        'source_seed': seed,
        'feature_dim': 512,
        'normalization': norm,
        'input_size': size,
        'split': 'train-only',
    }, args.output)
    print('PROTOTYPES SAVED:', args.output, flush=True)
    print('COUNTS:', json.dumps(dict(zip(labels, counts.tolist())), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
