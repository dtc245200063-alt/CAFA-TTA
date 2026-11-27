
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Read the exact same shift configuration as TENT
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    config = json.load(f)

assert config["split"] == "val"
shift = config["shift"]

normalize = transforms.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3
)

def shifted_transform(image):
    image = TF.adjust_brightness(
        image, shift["brightness"]
    )
    image = TF.adjust_contrast(
        image, shift["contrast"]
    )
    image = TF.adjust_saturation(
        image, shift["saturation"]
    )
    return normalize(TF.to_tensor(image))

dataset = PathMNIST(
    split="test",
    root=str(ROOT / "data/raw/medmnist"),
    transform=shifted_transform,
    download=False,
    size=28
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

checkpoint = torch.load(
    ROOT / "checkpoints/resnet18_pathmnist_best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)

# Enable current-batch BatchNorm statistics
# without allowing any parameter updates.
model.train()
model.requires_grad_(False)

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

# Evaluation only: no optimizer, no backward.
y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)

        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.view(-1).tolist())

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro",
    zero_division=0
)

print("=" * 50)
print("CAFA-TTA | BN-ONLY BASELINE")
print("Device:", DEVICE)
print("Shift:", shift)
print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))
print("=" * 50)

output = ROOT / "experiments/bn_only_test"
output.mkdir(parents=True, exist_ok=True)

with open(output / "metrics.json", "w") as f:
    json.dump({
        "split": "test",
        "batch_size": 64,
        "shift": shift,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }, f, indent=2)

print("BN-ONLY EVALUATION COMPLETED")
