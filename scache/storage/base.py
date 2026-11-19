from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseVectorStore(ABC):
    """
    Abstract interface for vector storage engines (FAISS, Redis VSS, etc.)
    """

    @abstractmethod
    def add(
        self,
        embedding: npt.NDArray[np.float32],
        prompt: str,
        response: str,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> Any:
        """
        Store a vector along with its text prompt and response payload.
        """

    @abstractmethod
    def search(
        self,
        query_vector: npt.NDArray[np.float32],
        top_k: int = 1,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> tuple[float, dict] | None:
        """
        Search for nearest vector match.
        Returns (similarity_score, metadata_dict) or None if no match.
        """
