
import torch
import torch.nn.functional as F

from core.cafa_adapter import CAFAAdapter
from kernel.confidence import (
    confidence_score,
    normalized_entropy,
)
from kernel.feature_alignment import (
    confidence_alignment_loss,
)


class CAFAAgreementAdapter(CAFAAdapter):
    """
    CAFA-TTA v2:
    Confidence + classifier/prototype agreement.

    Updates only BN affine parameters.
    Uses no target ground-truth labels.
    """

    def adapt(self, images):
        self.model.train()

        # Step 1: Build selection mask without gradient.
        with torch.no_grad():
            features, logits = self.model(images)

            scores = confidence_score(logits)
            classifier_labels = logits.argmax(dim=1)

            normalized_features = F.normalize(
                features,
                dim=1
            )

            similarities = (
                normalized_features
                @ self.prototypes.T
            )

            prototype_labels = (
                similarities.argmax(dim=1)
            )

            agreement = (
                classifier_labels == prototype_labels
            )

            selected = (
                (scores >= self.threshold)
                & agreement
            )

            selected_count = int(selected.sum().item())

        if selected_count < self.min_samples:
            return {
                "updated": False,
                "selected": selected_count,
                "entropy_loss": 0.0,
                "alignment_loss": 0.0,
                "total_loss": 0.0,
            }

        # Step 2: Forward with gradient.
        self.optimizer.zero_grad(set_to_none=True)

        features, logits = self.model(images)

        # Step 3: Entropy loss only on agreed samples.
        entropy = normalized_entropy(logits)

        loss_entropy = (
            entropy[selected].mean()
        )

        # Step 4: Alignment uses the SAME joint mask.
        # Non-selected samples receive zero weight.
        filtered_scores = (
            scores * selected.float()
        )

        loss_align = confidence_alignment_loss(
            target_features=features,
            pseudo_labels=classifier_labels,
            confidence_scores=filtered_scores,
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

        # Step 5: Update only BN affine parameters.
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.params,
            max_norm=1.0
        )

        self.optimizer.step()

        return {
            "updated": True,
            "selected": selected_count,
            "entropy_loss": loss_entropy.item(),
            "alignment_loss": loss_align.item(),
            "total_loss": total_loss.item(),
        }
