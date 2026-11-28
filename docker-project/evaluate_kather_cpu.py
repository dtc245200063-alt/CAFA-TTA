import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_ROOT = Path(
    "/kather/extracted/"
    "Kather_texture_2016_image_tiles_5000"
)

SPLIT_FILE = Path("/kather/split_seed42.json")
CHECKPOINT = Path("/outputs/kather2016_cpu_01/best.pt")
RESULT_FILE = Path("/outputs/kather2016_cpu_01/test_metrics.json")


class KatherDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        path = DATA_ROOT / item["path"]

        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))

        return image, int(item["label"])


def main():
    torch.set_num_threads(1)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if RESULT_FILE.exists():
        raise FileExistsError(RESULT_FILE)

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    test_records = split["splits"]["test"]

    assert len(split["classes"]) == 8
    assert len(test_records) == 752

    dataset = KatherDataset(test_records)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = models.resnet18(
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions = []
    labels_all = []

    print("=" * 60, flush=True)
    print("KATHER 2016 | DOCKER CPU TEST", flush=True)
    print("Test samples:", len(dataset), flush=True)
    print("Checkpoint epoch:", checkpoint["epoch"], flush=True)
    print("=" * 60, flush=True)

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)

            predictions.extend(
                outputs.argmax(1).tolist()
            )

            labels_all.extend(labels.tolist())

    accuracy = accuracy_score(
        labels_all,
        predictions,
    )

    macro_f1 = f1_score(
        labels_all,
        predictions,
        labels=list(range(8)),
        average="macro",
        zero_division=0,
    )

    metrics = {
        "dataset": "Kather 2016",
        "model": "ResNet18",
        "device": "cpu",
        "split": "test",
        "samples": len(dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classes": split["classes"],
        "confusion_matrix": confusion_matrix(
            labels_all,
            predictions,
            labels=list(range(8)),
        ).tolist(),
    }

    RESULT_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"MACRO-F1: {macro_f1:.4f}")
    print("RESULT:", RESULT_FILE)
    print("FINAL STATUS: PASS")


if __name__ == "__main__":
    main()
