import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


WORKSPACE = Path("/project")
sys.path.insert(0, str(WORKSPACE))

from models.feature_extractor import ResNetFeatureExtractor


DATA_ROOT = Path(
    "/kather/extracted/"
    "Kather_texture_2016_image_tiles_5000"
)

SPLIT_FILE = Path("/kather/split_seed42.json")

CHECKPOINT = Path(
    "/outputs/kather2016_cpu_01/best.pt"
)

OUTPUT_FILE = Path(
    "/outputs/kather2016_cpu_01/"
    "source_prototypes.pt"
)


class KatherDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * 3,
                [0.5] * 3
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        path = DATA_ROOT / item["path"]

        with Image.open(path) as img:
            image = self.transform(
                img.convert("RGB")
            )

        return image, int(item["label"])


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    if OUTPUT_FILE.exists():
        raise FileExistsError(OUTPUT_FILE)

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    assert len(split["classes"]) == 8

    records = split["splits"]["train"]
    assert len(records) == 3496

    dataset = KatherDataset(records)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    backbone = models.resnet18(
        weights=None,
        num_classes=8,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["num_classes"] == 8
    assert checkpoint["classes"] == split["classes"]

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ResNetFeatureExtractor(backbone)
    model.eval()

    feature_sums = torch.zeros(
        8, 512, dtype=torch.float64
    )

    class_counts = torch.zeros(
        8, dtype=torch.long
    )

    print("=" * 60, flush=True)
    print("KATHER 2016 | SOURCE PROTOTYPE EXTRACTION",
          flush=True)
    print("Training samples:", len(dataset), flush=True)
    print("Classes: 8", flush=True)
    print("Feature dimension: 512", flush=True)
    print("Device: CPU", flush=True)
    print("=" * 60, flush=True)

    with torch.inference_mode():
        for batch, (images, labels) in enumerate(
            loader, start=1
        ):
            features, _ = model(images)

            features = F.normalize(
                features.float(),
                p=2,
                dim=1
            ).to(torch.float64)

            feature_sums.index_add_(
                0, labels, features
            )

            class_counts += torch.bincount(
                labels, minlength=8
            )

            if batch % 50 == 0:
                print(
                    f"Batch {batch}/{len(loader)}",
                    flush=True
                )

    if torch.any(class_counts == 0):
        raise RuntimeError("Missing class")

    prototypes = (
        feature_sums
        / class_counts.unsqueeze(1)
    ).float()

    prototypes = F.normalize(
        prototypes, p=2, dim=1
    )

    assert prototypes.shape == (8, 512)
    assert torch.isfinite(prototypes).all()
    assert class_counts.sum().item() == 3496

    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "counts": class_counts.cpu(),
            "num_classes": 8,
            "feature_dim": 512,
            "backbone": "resnet18",
            "dataset": "Kather 2016",
            "split": "train",
            "feature_mode": "source_eval",
            "source_checkpoint": "best.pt",
            "classes": split["classes"],
        },
        OUTPUT_FILE
    )

    print("Prototype shape:",
          tuple(prototypes.shape))
    print("Class counts:",
          class_counts.tolist())
    print("Saved:", OUTPUT_FILE)
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()
