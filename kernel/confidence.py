
import math

import torch
import torch.nn.functional as F


def normalized_entropy(logits):
    """
    Returns entropy normalized to [0, 1].
    Shape: [batch_size]
    """
    num_classes = logits.shape[1]

    if num_classes < 2:
        raise ValueError("Need at least two classes")

    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)

    entropy = -(
        probabilities * log_probabilities
    ).sum(dim=1)

    return entropy / math.log(num_classes)


def confidence_score(logits):
    """
    Higher values indicate more concentrated
    predicted class distributions.

    This is a heuristic confidence score,
    not a calibrated probability of correctness.
    """
    entropy = normalized_entropy(logits)

    return (1.0 - entropy).clamp(0.0, 1.0)


def confidence_mask(logits, threshold=0.7):
    """
    Select samples with confidence >= threshold.
    """
    scores = confidence_score(logits)
    mask = scores >= threshold

    return mask, scores


def weighted_entropy_loss(logits, threshold=0.7):
    """
    Confidence-filtered entropy objective.

    Samples failing the threshold do not
    contribute to the loss.
    """
    entropy = normalized_entropy(logits)

    with torch.no_grad():
        scores = confidence_score(logits)
        weights = (
            scores >= threshold
        ).to(dtype=entropy.dtype)

    if weights.sum().item() == 0:
        # Return a valid zero loss connected
        # to the computation graph.
        return entropy.sum() * 0.0

    return (
        entropy * weights
    ).sum() / weights.sum()
