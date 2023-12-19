import numpy as np


def softmax(arr: np.ndarray) -> np.ndarray:
    assert len(np.shape(arr)) == 1, "The input array is not 1-dim."
    softmax_arr = np.exp(arr - np.max(arr))
    softmax_arr = softmax_arr / np.sum(softmax_arr)
    return softmax_arr


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
