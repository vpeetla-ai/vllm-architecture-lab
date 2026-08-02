"""Ops / observability honesty — teaching engine, not GPU-backed vLLM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_declares_pure_python_simulator() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "vllm-architecture-lab"
    assert body["engine"] == "pure_python_simulator"
    assert body["wall_clock_latency"] is False
    assert body["gpu_backed"] is False
    assert body["adapters_registered"] >= 1
    assert "step_count" in body


def test_observability_status_planes() -> None:
    resp = client.get("/v1/observability/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "EngineTrace" in {e["name"] for e in body["exporters"]}
    assert body["planes"]["cuda_kernels"] is False
    assert body["planes"]["multi_lora_path_b"] is True


def test_ops_metrics_null_p95_and_path_b() -> None:
    resp = client.get("/v1/ops/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "vllm-architecture-lab"
    assert body["p95_latency_ms"] is None
    extra = body["extra"]
    assert extra["wall_clock_latency"] is False
    assert extra["gpu_backed"] is False
    assert extra["multi_lora"]["path"] == "educational_b"
    assert extra["multi_lora"]["cuda_kernels"] is False
    assert "domainforge-triage-v0" in extra["multi_lora"]["adapters"]
