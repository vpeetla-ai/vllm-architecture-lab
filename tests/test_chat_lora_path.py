"""vLLM Lab chat + adapter registry (ADR-022 educational Path B)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_vllm_chat_completions_returns_triage_shape():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "domainforge-triage-v0",
            "messages": [{"role": "user", "content": "I forgot my password"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "domainforge-triage-v0"
    assert body["adapter_swapped"] is True
    content = body["choices"][0]["message"]["content"]
    assert "intent" in content
    assert "suggested_action" in content


def test_vllm_adapters_list():
    resp = client.get("/v1/adapters")
    assert resp.status_code == 200
    ids = {a["id"] for a in resp.json()["data"]}
    assert "domainforge-triage-v0" in ids
