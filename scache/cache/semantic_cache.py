from scache.core.guardrail import EntityGuardrail

from ..core.embedding import encoded
from ..storage.faiss_store import FaissStore


class SemanticCache:

    def __init__(self, threshold: float = 0.85, store = None, guardrail = None) -> None:
        self.threshold = threshold
        self.store = store if store is not None else FaissStore()
        self.guardrail = guardrail if guardrail is not None else EntityGuardrail()

    def lookup(self, prompt: str) -> tuple[str, float] | None:
        prompt_embedding = encoded(prompt)
        data = self.store.search(prompt_embedding)
        if data is not None:
            score, metadata = data
            if score >= self.threshold:
                if self.guardrail:
                    cached_prompt = metadata.get("prompt", "")
                    if not self.guardrail.verify(prompt, cached_prompt):
                        return None

                return metadata["response"], score
            if score < self.threshold:
                return None
        return None
    
    def update(self, prompt: str, response: str) -> None:
        prompt_embedding = encoded(prompt)
        self.store.add(prompt_embedding,prompt,response)

    def check_health(self) -> None:
        print(f"the cache size is {self.store.index.ntotal}")
