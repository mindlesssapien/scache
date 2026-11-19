import numpy as np
import numpy.typing as npt


def cosine_similarity(
    vec_a: npt.NDArray[np.float32], vec_b: npt.NDArray[np.float32]
) -> float:
    """
    Computes cosine similarity between two vectors.
    If vectors are already L2-normalized, this is equivalent to their dot product.
    """
    a = vec_a.flatten()
    b = vec_b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    score = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, score))


def batch_cosine_similarity(
    vectors_a: npt.NDArray[np.float32], vectors_b: npt.NDArray[np.float32]
) -> np.ndarray:
    """
    Computes element-wise cosine similarity between two batches of L2-normalized vectors.
    Shape of both inputs: (N, D). Returns: (N,)
    """
    assert vectors_a.shape == vectors_b.shape
    dots = np.sum(vectors_a * vectors_b, axis=1)
    return np.clip(dots, -1.0, 1.0)
