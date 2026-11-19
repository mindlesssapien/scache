import inspect

from scache.core.embedding import encoded
from scache.core.guardrail import EntityGuardrail
from scache.storage.base import BaseVectorStore
from scache.storage.faiss_store import FaissStore
from scache.storage.redis_store import RedisStore


class SemanticCache:
    """
    High-level Semantic Cache facade supporting both local FAISS and distributed Redis VSS.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        ttl: float | None = 86400.0,
        store: BaseVectorStore | None = None,
        guardrail = None,
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.guardrail = guardrail if guardrail is not None else EntityGuardrail()
        # If a custom store (e.g. RedisStore) is provided, use it across all tenants
        self.custom_store = store
        # Otherwise, maintain tenant-partitioned FAISS stores
        self.faiss_stores: dict[str, FaissStore] = {}

    def _get_store(self, tenant_id: str = "default") -> BaseVectorStore:
        if self.custom_store is not None:
            return self.custom_store
        if tenant_id not in self.faiss_stores:
            self.faiss_stores[tenant_id] = FaissStore()
        return self.faiss_stores[tenant_id]

    @property
    def store(self) -> BaseVectorStore:
        """Convenience property for accessing the default tenant store."""
        return self._get_store("default")

    def _process_search_result(
        self, prompt: str, data: tuple | None, effective_threshold: float
    ) -> tuple[str, float] | None:
        if data is not None:
            score, metadata = data
            if score >= effective_threshold:
                if self.guardrail:
                    cached_prompt = metadata.get("prompt", "")
                    if not self.guardrail.verify(prompt, cached_prompt):
                        return None
                return metadata["response"], score
        return None

    def lookup(
        self,
        prompt: str,
        tenant_id: str = "default",
        threshold: float | None = None,
        ttl: float | None = None,
    ) -> tuple[str, float] | None:
        """
        Synchronous lookup for in-memory FAISS store.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        effective_ttl = ttl if ttl is not None else self.ttl
        store = self._get_store(tenant_id)

        prompt_embedding = encoded(prompt)
        data = store.search(prompt_embedding, tenant_id=tenant_id, ttl=effective_ttl)
        if inspect.isawaitable(data):
            raise RuntimeError(
                "Async store (RedisStore) detected. Please call 'await cache.alookup(...)' instead."
            )
        return self._process_search_result(prompt, data, effective_threshold)

    async def alookup(
        self,
        prompt: str,
        tenant_id: str = "default",
        threshold: float | None = None,
        ttl: float | None = None,
    ) -> tuple[str, float] | None:
        """
        Asynchronous lookup supporting both synchronous FAISS and asynchronous Redis VSS.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        effective_ttl = ttl if ttl is not None else self.ttl
        store = self._get_store(tenant_id)

        prompt_embedding = encoded(prompt)
        data = store.search(prompt_embedding, tenant_id=tenant_id, ttl=effective_ttl)
        if inspect.isawaitable(data):
            data = await data
        return self._process_search_result(prompt, data, effective_threshold)

    def update(self, prompt: str, response: str, tenant_id: str = "default") -> None:
        """
        Synchronous update for in-memory FAISS store.
        """
        store = self._get_store(tenant_id)
        prompt_embedding = encoded(prompt)
        res = store.add(prompt_embedding, prompt, response, tenant_id=tenant_id, ttl=self.ttl)
        if inspect.isawaitable(res):
            raise RuntimeError(
                "Async store (RedisStore) detected. Please call 'await cache.aupdate(...)' instead."
            )

    async def aupdate(self, prompt: str, response: str, tenant_id: str = "default") -> None:
        """
        Asynchronous update supporting both synchronous FAISS and asynchronous Redis VSS.
        """
        store = self._get_store(tenant_id)
        prompt_embedding = encoded(prompt)
        res = store.add(prompt_embedding, prompt, response, tenant_id=tenant_id, ttl=self.ttl)
        if inspect.isawaitable(res):
            await res

    def check_health(self, tenant_id: str = "default") -> None:
        store = self._get_store(tenant_id)
        if hasattr(store, "index") and hasattr(store.index, "ntotal"):
            print(f"the cache size for tenant '{tenant_id}' is {store.index.ntotal}")
        elif isinstance(store, RedisStore):
            print(f"RedisStore connected at {store.redis_url} with index '{store.index_name}'")
        else:
            print(f"Active vector store: {type(store).__name__}")
