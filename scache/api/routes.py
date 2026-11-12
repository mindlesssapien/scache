import os
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, ConfigDict

from scache.cache.semantic_cache import SemanticCache


router = APIRouter()
scache = SemanticCache()

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")


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


async def forward_non_streaming(
        request_payload: dict,
        headers: dict, 
        prompt: str, 
        elapsed_start: float,
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
                if prompt:
                    scache.update(prompt, assistant_reply)

    # Attach diagnostic headers
    elapsed_ms = (time.perf_counter() - elapsed_start) * 1000
    diag_headers = {
        "X-Cache": "MISS",
        "X-Response-Time": f"{elapsed_ms:.2f}ms",
    }
    return JSONResponse(content=data, status_code=upstream_resp.status_code, headers=diag_headers)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_time = time.perf_counter()
    prompt = request.get_last_user_prompt()
    if not prompt:
        headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}"}
        payload = request.model_dump(exclude_none=True)
        return await forward_non_streaming(payload, headers, prompt="", elapsed_start=start_time)
    cached = scache.lookup(prompt)
    if cached is None:
        if request.stream:
            pass
        else:
            headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}"}
            return await forward_non_streaming(request.model_dump(exclude_none=True), headers, prompt, start_time)
    response, score = cached
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return build_cache_response(response, request.model, score, elapsed_ms) 
    



