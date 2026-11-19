import pytest
from fastapi.testclient import TestClient

from scache.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "scache_requests_total" in content
    assert "scache_request_duration_seconds" in content
    assert "scache_tokens_saved_total" in content
    assert "scache_cost_saved_usd_total" in content


def test_chat_completions_cache_hit_flow(client):
    # Prime cache via testclient or internal cache
    from scache.api.routes import scache

    scache.update("What is semantic caching?", "It stores responses based on meaning.")

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is semantic caching?"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Cache") == "HIT"
    assert resp.json()["choices"][0]["message"]["content"] == "It stores responses based on meaning."

    # Check that metrics recorded the hit
    metrics_resp = client.get("/metrics")
    assert 'scache_requests_total{result="hit"' in metrics_resp.text
