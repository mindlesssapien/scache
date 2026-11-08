from ..core.embedding import encoded
from ..storage.faiss_store import FaissStore


class SemanticCache:

    def __init__(self, threshold: float = 0.85, store = None) -> None:
        self.threshold = threshold
        self.store = FaissStore()

    def lookup(self, prompt: str) -> tuple[str, float] | None:
        prompt_embedding = encoded(prompt)
        data = self.store.search(prompt_embedding)
        if data is not None:
            score, metadata = data
            if score >= self.threshold:
                return (metadata["response"], score)
            if score < self.threshold:
                return None
        return None
    
    def update(self, prompt: str, response: str) -> None:
        prompt_embedding = encoded(prompt)
        self.store.add(prompt_embedding,prompt,response)

    def check_health(self) -> None:
        print(f"the cache size is {self.store.index.ntotal}")
