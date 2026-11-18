
import torch
import torch.nn.functional as F


def normalize_features(features):
    return F.normalize(
        features,
        p=2,
        dim=1,
        eps=1e-8
    )


@torch.no_grad()
def compute_prototypes(features, labels, num_classes):
    """
    Compute class prototypes from labeled
    source-domain embeddings.
    """

    features = normalize_features(features)
    labels = labels.view(-1).long()

    prototypes = torch.zeros(
        num_classes,
        features.shape[1],
        device=features.device,
        dtype=features.dtype
    )

    counts = torch.zeros(
        num_classes,
        device=features.device,
        dtype=features.dtype
    )

    prototypes.index_add_(
        0, labels, features
    )

    counts.index_add_(
        0,
        labels,
        torch.ones_like(
            labels,
            dtype=features.dtype
        )
    )

    prototypes = prototypes / counts.clamp_min(1).unsqueeze(1)

    return normalize_features(prototypes), counts


def confidence_alignment_loss(
    target_features,
    pseudo_labels,
    confidence_scores,
    source_prototypes,
    threshold=0.7,
):
    """
    Align target embeddings with corresponding
    source class prototypes.
    """

    target_features = normalize_features(
        target_features
    )

    source_prototypes = normalize_features(
        source_prototypes
    )

    pseudo_labels = pseudo_labels.view(-1).long()

    selected_prototypes = source_prototypes[
        pseudo_labels
    ]

    distances = (
        target_features - selected_prototypes
    ).pow(2).sum(dim=1)

    with torch.no_grad():
        weights = torch.where(
            confidence_scores >= threshold,
            confidence_scores,
            torch.zeros_like(confidence_scores)
        )

    return (
        distances * weights
    ).sum() / weights.sum().clamp_min(1e-8)
