import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    x1 = np.matmul(x,W1)+b1
    x1 = np.where(x1>0,x1,0)
    x2 = np.matmul(x1,W2)+b2

    return x2