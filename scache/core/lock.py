import asyncio
import hashlib
import time
import uuid

import redis.asyncio as redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def generate_prompt_hash(prompt: str) ->  str:
    normalized = prompt.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


async def acquire_lock(prompt_hash: str, tenant_id: str = "default") -> tuple[bool, str]:
    token = str(uuid.uuid4())
    try:
        is_acquired = await r.set(f"lock:{tenant_id}:{prompt_hash}", token, nx=True, ex=10)
        if is_acquired:
            return (True, token)
        return (False, "")
    except (redis.ConnectionError, Exception):
        return (True, "")



async def release_lock(prompt_hash: str, token: str, tenant_id: str = "default") -> None:
    if not token: 
        return
    try:
        current_token = await r.get(f"lock:{tenant_id}:{prompt_hash}")
        if current_token == token:
            await r.delete(f"lock:{tenant_id}:{prompt_hash}")
    except Exception:
        pass


async def wait_for_cache(
    prompt: str,
    cache,
    tenant_id: str = "default",
    max_wait_sec: float = 10.0,
    poll_interval: float = 0.05,
    threshold: float | None = None,
) -> tuple[str, float] | None:
    start = time.time()
    while time.time() - start < max_wait_sec:
        if hasattr(cache, "alookup"):
            cached = await cache.alookup(prompt, tenant_id=tenant_id, threshold=threshold)
        else:
            cached = cache.lookup(prompt, tenant_id=tenant_id, threshold=threshold)
        if cached is not None:
            return cached

        await asyncio.sleep(poll_interval)

    return None

