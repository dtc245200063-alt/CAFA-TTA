
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from kernel.confidence import (
    normalized_entropy,
    confidence_score,
    confidence_mask,
    weighted_entropy_loss,
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logits = torch.tensor(
        [
            [8.0, 0.2, 0.1],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.5],
        ],
        device=device,
        requires_grad=True,
    )

    entropy = normalized_entropy(logits)
    confidence = confidence_score(logits)

    mask, scores = confidence_mask(
        logits, threshold=0.7
    )

    loss = weighted_entropy_loss(
        logits, threshold=0.7
    )

    loss.backward()

    print("=" * 50)
    print("CAFA-TTA | CONFIDENCE KERNEL TEST")
    print("Device:", device)
    print("Entropy:", entropy.detach().cpu().tolist())
    print("Confidence:", confidence.detach().cpu().tolist())
    print("Selected mask:", mask.cpu().tolist())
    print("Weighted Loss:", loss.item())
    print("=" * 50)

    assert torch.all(
        (confidence >= 0) & (confidence <= 1)
    )

    assert confidence[0] > confidence[1]
    assert mask[0].item()
    assert not mask[1].item()

    assert torch.isfinite(loss)
    assert logits.grad is not None

    print("CONFIDENCE KERNEL TEST PASSED")


if __name__ == "__main__":
    main()
