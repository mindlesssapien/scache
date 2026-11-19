"""
Facebook AI Similarity Search (FAISS) is an open source library built by 
meta for fast, efficient Similarity searching and clustering of dense vectors.
It allows us to search through collections of million-billions high-dimensional 
vectors(text/images) at very high speeds.
"""
import time

import faiss
import numpy as np
import numpy.typing as npt

from scache.storage.base import BaseVectorStore


class FaissStore(BaseVectorStore):

    def __init__(self):
        self.d = 384
        self.metadata: dict[int, dict] = {}
        self.index = faiss.IndexFlatIP(self.d) # inner product distance


    def add(
        self,
        embedding: npt.NDArray[np.float32],
        prompt: str,
        response: str,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> None:
        assert embedding.ndim == 2
        timestamp = time.time()
        self.index.add(embedding)
        idx = self.index.ntotal - 1 # ntotal stores the total number of vectors currently indexed, it updates dynamically
        self.metadata[idx] = {
            "prompt": prompt,
            "response": response,
            "created_at": timestamp,
            "tenant_id": tenant_id,
        }


    def search(
        self,
        query_vector,
        top_k: int = 1,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> tuple | None:
        assert query_vector.ndim == 2

        if self.index.ntotal > 0:
            distances, indicies = self.index.search(query_vector, top_k)
            match_idx = indicies[0][0]
            if match_idx == -1 or match_idx not in self.metadata: 
                return None
            
            metadata = self.metadata[match_idx]
            # TTL Expiration Check
            if ttl is not None:
                created_at = metadata.get("created_at", 0.0)
                if (time.time() - created_at) > ttl:
                    # Entry is expired: remove from metadata and return None
                    del self.metadata[match_idx]
                    return None

            score = distances[0][0]
            return (score, metadata)

        return None

    def clean_expired(self, ttl: float) -> int:
        """
        Removes expired items from metadata and rebuilds FAISS index if expired items exist.
        Returns the number of pruned items.
        """
        now = time.time()
        expired_ids = [
            idx for idx, meta in self.metadata.items()
            if (now - meta.get("created_at", 0.0)) > ttl
        ]
        for idx in expired_ids:
            del self.metadata[idx]
        return len(expired_ids)
        
