import numpy as np


def softmax(x, axis=0):
    """Compute softmax along the specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def masked_softmax(x, axis=0, ignore_zeros=True):
    """Compute softmax along the specified axis, ignoring zero values but keeping them in place.
    If all values in a row/column are zero, keep them zero."""

    mask = x != 0  # Mask of nonzero elements
    all_zeros = np.all(
        ~mask, axis=axis, keepdims=True
    )  # Check if all values are zero along the axis

    if ignore_zeros:
        if np.any(~all_zeros):  # If at least one row/column is nonzero
            masked_x = np.where(
                mask, x, -np.inf
            )  # Replace zeros with -inf to exclude them
            exp_x = (
                np.exp(masked_x - np.max(masked_x, axis=axis, keepdims=True)) * mask
            )  # Compute exp only for nonzero
            softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        else:
            softmax_x = np.zeros_like(x)  # If all values are zero, return zeros
        return np.where(
            all_zeros, 0, softmax_x
        )  # Ensure all-zero rows/columns stay zero

    masked_x = np.where(mask, x, -np.inf)  # Set zeros to -inf to exclude from softmax

    # Compute softmax only for nonzero elements
    exp_x = np.exp(masked_x - np.max(masked_x, axis=axis, keepdims=True))
    exp_x *= mask  # Keep original zeros

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
