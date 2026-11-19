import asyncio
import time
import pytest

from scache.cache.semantic_cache import SemanticCache


def test_cache_miss_on_empty():
    """Empty cache should return None on lookup."""
    cache = SemanticCache()
    assert cache.lookup("Any question?") is None


def test_cache_exact_hit():
    """Exact prompt should return cached response with score ~1.0."""
    cache = SemanticCache(threshold=0.85)
    prompt = "What is the boiling point of water?"
    response = "100 degrees Celsius at standard atmospheric pressure."

    cache.update(prompt, response)
    result = cache.lookup(prompt)

    assert result is not None
    cached_text, score = result
    assert cached_text == response
    assert pytest.approx(score, abs=1e-3) == 1.0


def test_cache_paraphrase_hit():
    """Semantically equivalent prompt should hit the cache."""
    cache = SemanticCache(threshold=0.85)
    cache.update("What is the capital of France?", "Paris")

    result = cache.lookup("Which city serves as France's capital?")
    assert result is not None
    cached_text, score = result
    assert cached_text == "Paris"
    assert score >= 0.85


def test_cache_guardrail_swapped_entity():
    """Guardrail should reject queries with swapped proper nouns."""
    cache = SemanticCache(threshold=0.85)
    cache.update("What is the capital of France?", "Paris")

    # Swapped entity (France -> Germany) must be rejected
    result = cache.lookup("What is the capital of Germany?")
    assert result is None


def test_multi_tenant_isolation():
    """Entries in one tenant namespace must never leak into another tenant."""
    cache = SemanticCache()
    secret_prompt = "What is the secret API key?"
    secret_response = "sk-12345-secret"

    # Store under tenant 'acme'
    cache.update(secret_prompt, secret_response, tenant_id="acme")

    # Query as tenant 'acme' -> HIT
    acme_result = cache.lookup(secret_prompt, tenant_id="acme")
    assert acme_result is not None
    assert acme_result[0] == secret_response

    # Query as tenant 'globex' -> MISS
    globex_result = cache.lookup(secret_prompt, tenant_id="globex")
    assert globex_result is None

    # Query under default tenant -> MISS
    default_result = cache.lookup(secret_prompt, tenant_id="default")
    assert default_result is None


def test_threshold_override():
    """Custom threshold passed to lookup should override default threshold."""
    cache = SemanticCache(threshold=0.80)
    cache.update("How to reverse a string in Python?", "s[::-1]")

    # Paraphrase at default threshold (0.80) -> HIT
    assert cache.lookup("Python code to reverse string") is not None

    # Paraphrase with strict override (0.99) -> MISS
    assert cache.lookup("Python code to reverse string", threshold=0.99) is None


def test_ttl_expiration():
    """Entries should expire and return None after TTL seconds."""
    # 0.15s TTL
    cache = SemanticCache(ttl=0.15)
    cache.update("Temporary token", "token_xyz")

    # Immediate lookup -> HIT
    assert cache.lookup("Temporary token") is not None

    # Wait for expiration
    time.sleep(0.2)

    # Expired lookup -> MISS
    assert cache.lookup("Temporary token") is None


def test_async_alookup_and_aupdate():
    """Verify asynchronous alookup and aupdate methods."""
    async def _run():
        cache = SemanticCache(threshold=0.85)
        prompt = "Explain recursion simply"
        response = "A function that calls itself until a base condition is met."

        await cache.aupdate(prompt, response, tenant_id="async_test")
        res = await cache.alookup("Explain recursion simply", tenant_id="async_test")

        assert res is not None
        assert res[0] == response

    asyncio.run(_run())
