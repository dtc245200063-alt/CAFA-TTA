
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetFeatureExtractor(nn.Module):
    """
    Returns (embeddings, logits) for a torchvision classifier.

    Supports ResNet (conv1..layer4, fc) and DenseNet (features, classifier).
    Embedding size: 512 for ResNet-18, 2048 for ResNet-50, 1024 for DenseNet-121.
    The name is kept for the many existing callers.
    """

    def __init__(self, backbone):
        super().__init__()
        if all(hasattr(backbone, name) for name in ("conv1", "layer4", "fc")):
            self.kind = "resnet"
        elif all(hasattr(backbone, name) for name in ("features", "classifier")):
            self.kind = "densenet"
        else:
            raise ValueError(
                f"Unsupported backbone {type(backbone).__name__}: "
                "expected a torchvision ResNet or DenseNet"
            )
        self.backbone = backbone

    def forward(self, x):
        model = self.backbone

        if self.kind == "densenet":
            # Mirrors torchvision DenseNet.forward.
            x = F.relu(model.features(x), inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            features = torch.flatten(x, 1)
            return features, model.classifier(features)

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)

        features = torch.flatten(x, 1)
        logits = model.fc(features)

        return features, logits
