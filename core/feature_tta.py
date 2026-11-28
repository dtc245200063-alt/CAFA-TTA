"""Online test-time adaptation on frozen-backbone features (no backpropagation).

`FeatureSession` keeps the same ideas as `ContinualSession` for a frozen foundation backbone:
  - feature statistics follow the stream from a class-balanced memory (latest `per_class`
    features per predicted class), mixed with the source statistics; this is the feature-space
    counterpart of the BN statistics memory;
  - optional T3A (Iwasawa & Matsuo, NeurIPS 2021): the linear head is replaced by class
    prototypes built from the head weights plus low-entropy pseudo-labelled target features.
Labels are never used. Predict-then-update, so every prediction is made before its own update.
"""
from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F


class FeatureSession:
    def __init__(self, head, *, stats: str = "balanced", per_class: int = 16, min_memory: int = 32,
                 source_weight: float = 0.0, recent: int = 256, t3a: bool = False, t3a_keep: int = 20):
        self.head = head.eval()
        self.stats, self.source_weight = stats, source_weight
        num_classes = head.linear.out_features
        self.memory = [deque(maxlen=per_class) for _ in range(num_classes)]
        self.recent = deque(maxlen=recent)
        self.min_memory = min_memory
        self.t3a, self.t3a_keep = t3a, t3a_keep
        w = F.normalize(head.linear.weight.detach().float(), dim=1)
        self.supports = [[(w[c], 0.0)] for c in range(num_classes)]  # (normalised feature, entropy)

    def _target_stats(self):
        pool = [f for bucket in self.memory for f in bucket] if self.stats == "balanced" else list(self.recent)
        if self.stats == "source" or len(pool) < self.min_memory:
            return None, None
        x = torch.stack(pool)
        a = self.source_weight
        return a * self.head.mean + (1 - a) * x.mean(0), a * self.head.std + (1 - a) * (x.std(0) + 1e-6)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        f = features.float().to(self.head.mean.device)
        mean, std = self._target_stats()
        logits = self.head(f, mean, std)
        if self.t3a:
            z = F.normalize(self.head.normalise(f, mean, std), dim=1)
            protos = torch.stack([F.normalize(torch.stack([s for s, _ in sup]).mean(0), dim=0)
                                  for sup in self.supports])
            t3a_logits = 10.0 * z @ protos.T
        pred = logits.argmax(1)
        # Update after predicting.
        for i, c in enumerate(pred.tolist()):
            self.memory[c].append(f[i])
            self.recent.append(f[i])
        if self.t3a:
            ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
            for i, c in enumerate(pred.tolist()):
                self.supports[c].append((z[i], float(ent[i])))
                if len(self.supports[c]) > self.t3a_keep:
                    self.supports[c] = sorted(self.supports[c], key=lambda s: s[1])[:self.t3a_keep]
            return t3a_logits
        return logits
