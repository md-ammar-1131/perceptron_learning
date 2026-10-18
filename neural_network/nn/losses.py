"""Loss functions."""
import numpy as np


def cross_entropy(probs, y_onehot):
    """Mean cross-entropy loss over a batch.

    probs    : (m, k) predicted class probabilities (softmax output)
    y_onehot : (m, k) one-hot true labels
    """
    eps = 1e-12  # avoid log(0)
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


def one_hot(y, n_classes):
    """Convert integer labels, shape (m,), to one-hot, shape (m, n_classes)."""
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out
