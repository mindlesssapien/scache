import time
import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import redis.asyncio as redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from scache.storage.base import BaseVectorStore


class RedisStore(BaseVectorStore):
    """
    Distributed Vector Storage engine using Redis VSS (Vector Similarity Search).
    Requires Redis Stack or Redis 8+ with RediSearch module enabled.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        index_name: str = "idx:scache",
        dim: int = 384,
        distance_metric: str = "COSINE",
    ) -> None:
        self.redis_url = redis_url
        self.index_name = index_name
        self.dim = dim
        self.distance_metric = distance_metric
        # Use decode_responses=False so binary vector blobs can be written safely
        self.client = redis.from_url(redis_url, decode_responses=False)
        self._index_initialized = False

    async def init_index(self) -> None:
        """
        Creates the RediSearch index if it does not already exist.
        """
        if self._index_initialized:
            return

        try:
            # Check if index already exists
            await self.client.ft(self.index_name).info()
            self._index_initialized = True
        except Exception:
            # Index does not exist; create it
            schema = [
                TagField("tenant_id"),
                TextField("prompt"),
                TextField("response"),
                VectorField(
                    "vector",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dim,
                        "DISTANCE_METRIC": self.distance_metric,
                    },
                ),
            ]
            definition = IndexDefinition(prefix=["scache:doc:"], index_type=IndexType.HASH)
            await self.client.ft(self.index_name).create_index(schema, definition=definition)
            self._index_initialized = True

    async def add(
        self,
        embedding: npt.NDArray[np.float32],
        prompt: str,
        response: str,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> str:
        """
        Stores prompt, response, and vector embedding in a Redis Hash.
        If ttl is specified, sets native key expiration.
        """
        await self.init_index()

        doc_id = f"scache:doc:{tenant_id}:{uuid.uuid4().hex[:12]}"
        vector_bytes = embedding.astype(np.float32).tobytes()

        mapping = {
            b"tenant_id": tenant_id.encode("utf-8"),
            b"prompt": prompt.encode("utf-8"),
            b"response": response.encode("utf-8"),
            b"vector": vector_bytes,
            b"created_at": str(time.time()).encode("utf-8"),
        }

        await self.client.hset(doc_id, mapping=mapping)

        # Native Redis TTL: automatically deletes key and prunes vector from index!
        if ttl is not None and ttl > 0:
            await self.client.expire(doc_id, int(ttl))

        return doc_id

    async def search(
        self,
        query_vector: npt.NDArray[np.float32],
        top_k: int = 1,
        tenant_id: str = "default",
        ttl: float | None = None,
    ) -> tuple[float, dict] | None:
        """
        Executes a pre-filtered KNN search within the specified tenant's partition.
        RediSearch COSINE distance returns d in [0, 2], where d = 1 - cos(theta).
        Similarity score is computed as: similarity = 1.0 - distance.
        """
        await self.init_index()

        vector_bytes = query_vector.astype(np.float32).tobytes()
        query_str = f"(@tenant_id:{{{tenant_id}}})=>[KNN {top_k} @vector $vec AS score]"

        q = (
            Query(query_str)
            .sort_by("score")
            .return_fields("prompt", "response", "score", "created_at", "tenant_id")
            .dialect(2)
        )
        params = {"vec": vector_bytes}

        try:
            res = await self.client.ft(self.index_name).search(q, query_params=params)
        except Exception:
            return None

        if not res or res.total == 0:
            return None

        top_doc = res.docs[0]
        # RediSearch COSINE distance: score = 1 - cos(theta)
        cosine_distance = float(top_doc.score)
        similarity = max(-1.0, min(1.0, 1.0 - cosine_distance))

        prompt_text = (
            top_doc.prompt.decode("utf-8") if isinstance(top_doc.prompt, bytes) else top_doc.prompt
        )
        response_text = (
            top_doc.response.decode("utf-8")
            if isinstance(top_doc.response, bytes) else top_doc.response
        )
        created_at = float(
            top_doc.created_at.decode("utf-8")
            if isinstance(getattr(top_doc, "created_at", b"0"), bytes)
            else getattr(top_doc, "created_at", 0)
        )

        metadata = {
            "prompt": prompt_text,
            "response": response_text,
            "created_at": created_at,
            "tenant_id": tenant_id,
        }

        return (similarity, metadata)
