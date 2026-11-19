import time
import uuid

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from scache.api.streaming import (
    UPSTREAM_API_KEY,
    forward_non_streaming,
    stream_cached_response,
    stream_upstream_and_cache,
)
from scache.cache.semantic_cache import SemanticCache
from scache.core.lock import (
    acquire_lock,
    generate_prompt_hash,
    release_lock,
    wait_for_cache,
)

router = APIRouter()
scache = SemanticCache()

class ChatMessage(BaseModel):
    role : str #Annotated[Literal["user", "system", "assistant"], Field(..., description="role indicator")]
    content: str

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow") #allows extra fields from openAI sdk rather than showing 422 error

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None # the = None makes this truly optional parameter to fill otherwise it will be required and would later raise errors
    max_tokens: int | None = None


    def get_last_user_prompt(self) -> str | None:
        """
            traverse the messages in reverse to get 
            latest message from user.
        """
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content.strip()
        return None


def build_cache_response(content: str, model: str, score: float, elapsed_ms: float) -> JSONResponse:
    payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", # fixed type required by openAI sdk
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    headers = {
            "X-Cache": "HIT",
            "X-Cache-Score": f"{score:.4f}",
            "X-Response-Time": f"{elapsed_ms:.2f}ms",
        }

    return JSONResponse(content=payload, headers=headers)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
    x_cache_threshold: float | None = Header(default=None, alias="X-Cache-Threshold"),
    x_cache_ttl: float | None = Header(default=None, alias="X-Cache-TTL"),
):
    start_time = time.perf_counter()
    prompt = request.get_last_user_prompt()

    headers = {}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    if not prompt:
        payload = request.model_dump(exclude_none=True)
        return await forward_non_streaming(
            payload, headers, prompt="", elapsed_start=start_time, SemanticCache=scache, tenant_id=x_tenant_id
        )

    cached = await scache.alookup(prompt, tenant_id=x_tenant_id, threshold=x_cache_threshold, ttl=x_cache_ttl)
    if cached is None:
        prompt_hash = generate_prompt_hash(prompt)
        acquired, lock_token = await acquire_lock(prompt_hash, tenant_id=x_tenant_id)

        if not acquired:
            # FOLLOWER: Wait for Leader to complete and populate the cache
            cached_result = await wait_for_cache(
                prompt, scache, tenant_id=x_tenant_id, threshold=x_cache_threshold
            )
            if cached_result is not None:
                response, score = cached_result
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if request.stream:
                    return StreamingResponse(
                        stream_cached_response(response, request.model),
                        media_type="text/event-stream",
                        headers={
                            "X-Cache": "HIT",
                            "X-Cache-Score": f"{score:.4f}",
                            "X-Response-Time": f"{elapsed_ms:.2f}ms",
                        },
                    )
                return build_cache_response(response, request.model, score, elapsed_ms)

        # LEADER (or fallback if wait timed out): query upstream
        payload = request.model_dump(exclude_none=True)
        if request.stream:
            return StreamingResponse(
                stream_upstream_and_cache(
                    payload,
                    headers,
                    prompt,
                    SemanticCache=scache,
                    lock_info=(prompt_hash, lock_token) if acquired else None,
                    tenant_id=x_tenant_id,
                ),
                media_type="text/event-stream",
                headers={"X-Cache": "MISS"},
            )
        else:
            try:
                return await forward_non_streaming(
                    payload, headers, prompt, start_time, SemanticCache=scache, tenant_id=x_tenant_id
                )
            finally:
                if acquired:
                    await release_lock(prompt_hash, lock_token, tenant_id=x_tenant_id)
    response, score = cached
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if request.stream:
        return StreamingResponse(
                stream_cached_response(response, request.model),
                media_type="text/event-stream",
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Score": f"{score:.4f}",
                    "X-Response-Time": f"{elapsed_ms:.2f}ms"
                    }
                )
    return build_cache_response(response, request.model, score, elapsed_ms) 
    



