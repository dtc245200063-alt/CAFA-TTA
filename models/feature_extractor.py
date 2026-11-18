
import torch
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    """
    Returns ResNet-18 embeddings and logits.

    Embeddings: [B, 512]
    Logits: [B, 9] for PathMNIST
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        model = self.backbone

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
