from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()


def prediction_consistency(
    reference_logits: torch.Tensor, transformed_logits: torch.Tensor
) -> torch.Tensor:
    """Fraction retaining the reference model prediction."""
    reference_predictions = reference_logits.argmax(dim=1)
    transformed_predictions = transformed_logits.argmax(dim=1)
    return (reference_predictions == transformed_predictions).float().mean()


def representation_similarity(
    reference_features: torch.Tensor, transformed_features: torch.Tensor
) -> torch.Tensor:
    """Mean cosine similarity between paired latent vectors."""
    # TODO: implement this metric using F.cosine_similarity.
    return F.cosine_similarity(
        reference_features, transformed_features, dim=1
    ).mean()
