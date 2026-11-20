
import torch
import torch.nn.functional as F

from core.cafa_adapter import CAFAAdapter
from kernel.confidence import (
    confidence_score,
    normalized_entropy,
)


class CAFASoftAdapter(CAFAAdapter):
    """
    Experimental CAFA-TTA v3:
    Soft confidence weighting for entropy,
    agreement-aware prototype alignment.

    Updates BN affine parameters only.
    No target ground-truth labels are used.
    """

    def __init__(
        self,
        model,
        prototypes,
        lr=1e-4,
        lambda_align=0.05,
        entropy_floor=0.2,
        min_samples=2,
    ):
        super().__init__(
            model=model,
            prototypes=prototypes,
            lr=lr,
            threshold=0.0,
            lambda_align=lambda_align,
            min_samples=min_samples,
        )

        if not (0.0 <= entropy_floor <= 1.0):
            raise ValueError(
                "entropy_floor must be in [0, 1]"
            )

        self.entropy_floor = entropy_floor

    def adapt(self, images):
        self.model.train()

        # Step 1: Calculate pseudo-labels and
        # confidence without gradients.
        with torch.no_grad():
            features, logits = self.model(images)

            scores = confidence_score(logits)
            classifier_labels = logits.argmax(dim=1)

            features_normalized = F.normalize(
                features.float(),
                p=2,
                dim=1,
            )

            similarities = (
                features_normalized
                @ self.prototypes.T
            )

            prototype_labels = similarities.argmax(
                dim=1
            )

            agreement = (
                classifier_labels == prototype_labels
            )

            agreement_count = int(
                agreement.sum().item()
            )

            # All samples contribute to entropy.
            entropy_weights = (
                self.entropy_floor
                + (1.0 - self.entropy_floor) * scores
            )

            # Only agreement samples contribute
            # to alignment, weighted by confidence.
            alignment_weights = (
                scores * agreement.float()
            )

        # Step 2: Forward pass with gradients.
        self.optimizer.zero_grad(set_to_none=True)

        features, logits = self.model(images)

        # Step 3: Soft-weighted entropy loss.
        entropy = normalized_entropy(logits)

        loss_entropy = (
            entropy * entropy_weights
        ).sum() / entropy_weights.sum().clamp_min(1e-8)

        # Step 4: Agreement-weighted alignment.
        if agreement_count >= self.min_samples:
            target_features = F.normalize(
                features.float(),
                p=2,
                dim=1,
            )

            selected_prototypes = self.prototypes[
                classifier_labels
            ]

            distances = (
                target_features - selected_prototypes
            ).pow(2).sum(dim=1)

            loss_align = (
                distances * alignment_weights
            ).sum() / alignment_weights.sum().clamp_min(1e-8)

        else:
            loss_align = features.sum() * 0.0

        # Step 5: Combined objective.
        total_loss = (
            loss_entropy
            + self.lambda_align * loss_align
        )

        if not torch.isfinite(total_loss).item():
            self.optimizer.zero_grad(set_to_none=True)

            return {
                "updated": False,
                "selected": agreement_count,
                "entropy_loss": 0.0,
                "alignment_loss": 0.0,
                "total_loss": 0.0,
                "reason": "non_finite_loss",
            }

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.params,
            max_norm=1.0,
        )

        self.optimizer.step()

        return {
            "updated": True,
            "selected": agreement_count,
            "entropy_loss": float(
                loss_entropy.detach().item()
            ),
            "alignment_loss": float(
                loss_align.detach().item()
            ),
            "total_loss": float(
                total_loss.detach().item()
            ),
            "mean_confidence": float(
                scores.mean().item()
            ),
            "entropy_weight_mean": float(
                entropy_weights.mean().item()
            ),
        }
