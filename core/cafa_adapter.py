
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.confidence import (
    confidence_score,
    weighted_entropy_loss,
)
from kernel.feature_alignment import (
    confidence_alignment_loss,
)


class CAFAAdapter:
    def __init__(
        self,
        model,
        prototypes,
        lr=1e-4,
        threshold=0.7,
        lambda_align=0.05,
        min_samples=2,
    ):
        self.model = model
        self.threshold = threshold
        self.lambda_align = lambda_align
        self.min_samples = min_samples

        self.prototypes = F.normalize(
            prototypes.detach().to(
                next(model.parameters()).device
            ).float(),
            dim=1,
        )

        # Freeze entire backbone first.
        self.model.requires_grad_(False)
        self.model.train()

        params = []

        # Enable only BatchNorm affine parameters.
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

                if module.affine:
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)
                    params.extend([
                        module.weight,
                        module.bias,
                    ])

        if not params:
            raise RuntimeError(
                "No BatchNorm affine parameters found"
            )

        self.params = params
        self.optimizer = torch.optim.Adam(
            params, lr=lr
        )

    @torch.no_grad()
    def predict(self, images):
        # Keep target-batch BN statistics active.
        self.model.train()
        features, logits = self.model(images)
        return logits

    def adapt(self, images):
        self.model.train()

        # Estimate confidence using current weights.
        with torch.no_grad():
            _, initial_logits = self.model(images)
            scores = confidence_score(initial_logits)

            selected = scores >= self.threshold
            selected_count = int(selected.sum().item())

        if selected_count < self.min_samples:
            return {
                "updated": False,
                "selected": selected_count,
                "entropy_loss": 0.0,
                "alignment_loss": 0.0,
                "total_loss": 0.0,
            }

        self.optimizer.zero_grad(set_to_none=True)

        features, logits = self.model(images)

        # Confidence-filtered entropy objective.
        loss_entropy = weighted_entropy_loss(
            logits,
            threshold=self.threshold,
        )

        pseudo_labels = (
            initial_logits.argmax(dim=1)
        )

        loss_align = confidence_alignment_loss(
            target_features=features,
            pseudo_labels=pseudo_labels,
            confidence_scores=scores,
            source_prototypes=self.prototypes,
            threshold=self.threshold,
        )

        total_loss = (
            loss_entropy
            + self.lambda_align * loss_align
        )

        if not torch.isfinite(total_loss):
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "updated": False,
                "selected": selected_count,
                "reason": "non_finite_loss",
            }

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.params, max_norm=1.0
        )

        self.optimizer.step()

        return {
            "updated": True,
            "selected": selected_count,
            "entropy_loss": loss_entropy.item(),
            "alignment_loss": loss_align.item(),
            "total_loss": total_loss.item(),
        }
