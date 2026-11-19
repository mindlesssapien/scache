import json
import os
import time
import uuid

import httpx
from fastapi.responses import JSONResponse

from scache.core.lock import release_lock

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")


async def forward_non_streaming(
        request_payload: dict, 
        headers: dict,
        prompt: str,
        elapsed_start: float,
        SemanticCache = None,
        tenant_id: str = "default",
        ) -> JSONResponse:
    
    async with httpx.AsyncClient(base_url=UPSTREAM_BASE_URL, timeout=60.0) as client:
        upstream_resp = await client.post(
                "/chat/completions",
                json=request_payload,
                headers=headers,
        )
        data = upstream_resp.json()

    if upstream_resp.status_code == 200:
        choices = data.get("choices", [])
        if choices:
            assistant_reply = choices[0]["message"]["content"]
            if prompt and SemanticCache:
                if hasattr(SemanticCache, "aupdate"):
                    await SemanticCache.aupdate(prompt, assistant_reply, tenant_id=tenant_id)
                else:
                    SemanticCache.update(prompt, assistant_reply, tenant_id=tenant_id)

    elapsed_ms = (time.perf_counter() - elapsed_start) * 1000
    # diagnostic headers
    diag_headers = {
            "X-Cache": "MISS",
            "X-Response-Time": f"{elapsed_ms:.2f}ms",
    }
    return JSONResponse(content=data, status_code=upstream_resp.status_code, headers=diag_headers)

async def stream_upstream_and_cache(
    request_payload: dict,
    headers: dict,
    prompt: str,
    SemanticCache = None,
    lock_info: tuple[str, str] | None = None,
    tenant_id: str = "default",
):
    """
        Network Format: The server keeps the HTTP connection open and yields discrete chunks formatted as:
        data: {"choices": [{"delta": {"content": "The"}}]}
    """
    accumulated_content = []  # acts as buffer
    try:
        async with httpx.AsyncClient(base_url=UPSTREAM_BASE_URL, timeout=60.0) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=request_payload,
                headers=headers,
            ) as upstream_stream:

                async for raw_line in upstream_stream.aiter_lines():
                    if not raw_line:
                        continue

                    yield f"{raw_line}\n\n"

                    if raw_line.startswith("data: ") and raw_line != "data: [DONE]":
                        try:
                            chunk = json.loads(raw_line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                accumulated_content.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

        # Stream is complete! Write full accumulated response to cache
        full_text = "".join(accumulated_content)
        if full_text and SemanticCache:
            if hasattr(SemanticCache, "aupdate"):
                await SemanticCache.aupdate(prompt, full_text, tenant_id=tenant_id)
            else:
                SemanticCache.update(prompt, full_text, tenant_id=tenant_id)
    finally:
        if lock_info:
            prompt_hash, token = lock_info
            await release_lock(prompt_hash, token, tenant_id=tenant_id)

async def stream_cached_response(content: str, model: str):
    chunk_id = f"chat-cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model, 
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # 2. Yield finish chunk + DONE marker
    finish_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


