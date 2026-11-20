
import torch
import torch.nn as nn


def configure_tent(model):
    """Configure ResNet for Tent adaptation."""

    model.train()

    # Freeze all model parameters.
    model.requires_grad_(False)

    parameters = []

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)

            # Use current target-batch statistics.
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

            parameters.extend([
                module.weight,
                module.bias,
            ])

    if not parameters:
        raise RuntimeError(
            "No BatchNorm parameters found for Tent"
        )

    return model, parameters


def entropy_loss(logits):
    """Calculate prediction entropy."""

    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)

    entropy = -(
        probabilities * log_probabilities
    ).sum(dim=1)

    return entropy.mean()


class TentAdapter:
    def __init__(self, model, learning_rate=1e-4):
        self.model, params = configure_tent(model)

        self.optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
        )

    def adapt(self, images):
        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(images)

        loss = entropy_loss(logits)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, images):
        # Remain in train mode to use
        # target-batch BatchNorm statistics.
        self.model.train()
        return self.model(images)
