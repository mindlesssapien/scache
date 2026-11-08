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


class FaissStore:

    def __init__(self):
        self.d = 384
        self.metadata: dict[int, dict] = {}
        self.index = faiss.IndexFlatIP(self.d) # inner product distance


    def add(self, embedding: npt.NDArray[np.float32], prompt: str, response: str) -> None:
        assert embedding.ndim == 2
        timestamp = time.time()
        self.index.add(embedding)
        idx = self.index.ntotal - 1 # ntotal stores the total number of vectors currently indexed, it updates dynamically
        self.metadata[idx] = {"prompt": prompt, "response": response, "created_at": timestamp}


    def search(self, query_vector, top_k: int = 1) -> tuple|None:
        assert query_vector.ndim == 2

        if self.index.ntotal > 0:
            distances, indicies = self.index.search(query_vector, top_k)
            if indicies[0][0] == -1 or indicies[0][0] not in self.metadata: 
                return None
            score = distances[0][0]
            metadata = self.metadata[indicies[0][0]]
            return (score, metadata)

        return None
        
