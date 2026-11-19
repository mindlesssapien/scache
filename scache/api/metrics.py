import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

# Metrics Definitions
CACHE_REQUESTS = Counter(
    "scache_requests_total",
    "Total number of requests processed by semantic cache",
    ["result", "tenant_id"]  # result: 'hit' or 'miss'
)

CACHE_LATENCY = Histogram(
    "scache_request_duration_seconds",
    "Request latency in seconds",
    ["result"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

TOKENS_SAVED = Counter(
    "scache_tokens_saved_total",
    "Estimated tokens saved from upstream LLM generation due to cache hits",
    ["tenant_id"]
)

COST_SAVED_USD = Counter(
    "scache_cost_saved_usd_total",
    "Estimated USD saved based on upstream token pricing ($0.00015 / 1K tokens)",
    ["tenant_id"]
)


def record_cache_hit(duration: float, tenant_id: str = "default", estimated_tokens: int = 150):
    CACHE_REQUESTS.labels(result="hit", tenant_id=tenant_id).inc()
    CACHE_LATENCY.labels(result="hit").observe(duration)
    TOKENS_SAVED.labels(tenant_id=tenant_id).inc(estimated_tokens)
    # Estimate savings: ~$0.15 per million tokens for gpt-4o-mini / gpt-3.5 input/output mix
    cost_savings = (estimated_tokens / 1000.0) * 0.00015
    COST_SAVED_USD.labels(tenant_id=tenant_id).inc(cost_savings)


def record_cache_miss(duration: float, tenant_id: str = "default"):
    CACHE_REQUESTS.labels(result="miss", tenant_id=tenant_id).inc()
    CACHE_LATENCY.labels(result="miss").observe(duration)


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
