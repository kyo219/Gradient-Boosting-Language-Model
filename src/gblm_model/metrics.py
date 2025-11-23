"""
Evaluation metrics for GBLM model.
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list]


def compute_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute simple classification accuracy.

    Args:
        y_true: True labels (shape: (N,))
        y_pred: Predicted labels (shape: (N,))

    Returns:
        Accuracy (0.0-1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    return float((y_true == y_pred).mean())


def compute_multi_logloss(y_true: ArrayLike, proba: ArrayLike, eps: float = 1e-15) -> float:
    """
    Compute multiclass negative log-likelihood (logloss).

    Args:
        y_true: True labels (shape: (N,))
        proba: Predicted probabilities (shape: (N, C)), each row is probability per class
        eps: Small value for numerical stability in log

    Returns:
        Average logloss
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)

    N, C = proba.shape
    assert y_true.shape[0] == N, f"Sample count mismatch: {y_true.shape[0]} vs {N}"
    assert np.all(y_true >= 0) and np.all(y_true < C), "Invalid class labels"

    # Clip probabilities for numerical stability
    clipped = np.clip(proba, eps, 1.0 - eps)

    # Normalize probabilities (safety check)
    row_sums = clipped.sum(axis=1, keepdims=True)
    clipped = clipped / row_sums

    # Extract true class probabilities using advanced indexing
    p_true = clipped[np.arange(N), y_true]
    logloss = -np.log(p_true).mean()
    return float(logloss)


def compute_perplexity(y_true: ArrayLike, proba: ArrayLike, eps: float = 1e-15) -> float:
    """
    Compute perplexity = exp(average negative log-likelihood).

    Args:
        y_true: True labels (shape: (N,))
        proba: Predicted probabilities (shape: (N, C))
        eps: Small value for numerical stability

    Returns:
        Perplexity value
    """
    logloss = compute_multi_logloss(y_true, proba, eps=eps)
    return float(np.exp(logloss))