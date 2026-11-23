
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision import models

from kernel.tent import TentAdapter


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(weights=None).to(device)

    adapter = TentAdapter(model, learning_rate=1e-4)

    images = torch.randn(
        8, 3, 28, 28,
        device=device,
    )

    loss = adapter.adapt(images)
    logits = adapter.predict(images)

    print("Device:", device)
    print("TENT Loss:", loss)
    print("Output shape:", tuple(logits.shape))

    assert torch.isfinite(torch.tensor(loss))
    assert logits.shape == (8, 1000)

    print("TENT KERNEL TEST PASSED")


def test_smoke():
    """pytest entry point: runs the original self-checking script."""
    main()


if __name__ == "__main__":
    main()
