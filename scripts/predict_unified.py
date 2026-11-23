"""Inference for the unified ONE-checkpoint 17-label ResNet18 model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from train_unified import build_model

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/cafa_tta_unified_best.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = state["labels"]
    model = build_model(len(labels))
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    size = int(state["input_size"])
    norm = state["normalization"]
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                              transforms.Normalize(norm["mean"], norm["std"])])
    with Image.open(args.image) as raw:
        x = tfm(raw.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(x).softmax(1)[0].cpu()
    top = torch.topk(probabilities, k=min(5, len(labels)))
    print(json.dumps({"checkpoint": str(args.checkpoint), "predictions": [
        {"label": labels[int(i)], "probability": round(float(v), 6)}
        for v, i in zip(top.values, top.indices)]}, indent=2, ensure_ascii=False))
    print("RESEARCH ONLY: not for clinical diagnosis.")

if __name__ == "__main__":
    main()
