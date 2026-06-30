import pytest
from fastapi.testclient import TestClient

from app.main import app
from vllm_lab.config import EngineConfig
from vllm_lab.kv_cache.block_manager import BlockSpaceManager
from vllm_lab.kv_cache.formulas import LLAMA3_8B, compute_memory_budget, kv_bytes_per_token
from vllm_lab.engine.llm_engine import LLMEngine
from vllm_lab.types import make_group


def test_kv_bytes_per_token_llama3_8b():
    # 2 × 32 × 128 × 2 × 32 layers = 524288
    assert kv_bytes_per_token(LLAMA3_8B) == 524_288


def test_memory_budget_h100_8b():
    result = compute_memory_budget(LLAMA3_8B, gpu_memory_gb=80, model_weights_gb=16)
    assert result.kv_budget_gb > 50
    assert result.num_gpu_blocks > 1000


def test_block_allocate_and_free():
    mgr = BlockSpaceManager(block_size=16, num_gpu_blocks=8)
    blocks = mgr.allocate("seq-a", 20)
    assert blocks is not None
    assert len(blocks) == 2
    mgr.free_sequence("seq-a")
    assert len(mgr.free_gpu) == 8


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="block_size"):
        EngineConfig(block_size=0)
    with pytest.raises(ValueError, match="preemption_mode"):
        EngineConfig(preemption_mode="invalid")


def test_prefix_sharing_increments_refcount():
    mgr = BlockSpaceManager(block_size=16, num_gpu_blocks=8)
    shared = mgr.allocate("seq-a", 16)
    assert shared
    mgr.share_prefix_blocks("seq-b", shared)
    assert mgr.blocks[shared[0]].ref_count == 2
    assert mgr.blocks[shared[0]].state.value == "shared"


def test_append_to_shared_tail_uses_copy_on_write():
    mgr = BlockSpaceManager(block_size=4, num_gpu_blocks=8)
    shared = mgr.allocate("seq-a", 4)
    assert shared

    table = mgr.share_prefix_blocks("seq-b", shared)
    assert table == shared

    updated = mgr.append_token("seq-b", 4)
    assert updated is not None
    assert updated[0] != shared[0]
    assert mgr.blocks[shared[0]].ref_count == 1
    assert mgr.blocks[updated[0]].owner_seq_id == "seq-b"


def test_engine_runs_until_finish():
    engine = LLMEngine(EngineConfig(num_gpu_blocks=32, max_num_seqs=8))
    group = make_group("test prompt", max_tokens=3)
    engine.add_request(group)
    outputs = engine.run_until_idle()
    assert len(outputs) >= 3
    assert not engine.scheduler.running


def test_continuous_batching_multiple_requests():
    engine = LLMEngine(EngineConfig(num_gpu_blocks=64, max_num_seqs=8))
    engine.add_request(make_group("short", max_tokens=2))
    engine.add_request(make_group("medium length prompt here", max_tokens=4))
    outputs = engine.run_until_idle()
    assert len(outputs) >= 4
    snap = engine.snapshot()
    assert snap["queues"]["waiting"] == 0


def test_scheduler_preempts_when_blocks_are_exhausted():
    engine = LLMEngine(EngineConfig(block_size=1, num_gpu_blocks=2, num_cpu_blocks=2, max_num_seqs=2))
    engine.add_request(make_group("a", max_tokens=2))
    engine.add_request(make_group("b", max_tokens=2))

    first = engine.step()

    assert first is not None
    assert first.scheduler.preempted_seq_ids
    assert any(event.event == "preempt" for event in first.trace)
    assert engine.scheduler.swapped


def test_stale_prefix_cache_entry_is_not_reused():
    engine = LLMEngine(EngineConfig(block_size=4, num_gpu_blocks=8, max_num_seqs=4))
    engine.add_request(make_group("same prefix", max_tokens=1))
    engine.run_until_idle()

    engine.add_request(make_group("same prefix", max_tokens=2))
    out = engine.step()

    assert out is not None
    assert any(event.event == "cache_stale" for event in out.trace)
    assert all(engine.block_manager.has_live_blocks(blocks) for blocks in engine.prefix_cache.entries.values())


def test_step_trace_is_structured():
    engine = LLMEngine(EngineConfig(num_gpu_blocks=16, max_num_seqs=4))
    engine.add_request(make_group("trace me", max_tokens=1))

    out = engine.step()

    assert out is not None
    assert out.trace[0].as_dict()["event"] == "step"
    assert any(event.event == "sample" for event in out.trace)


def test_api_simulate_returns_trace_events():
    client = TestClient(app)
    response = client.post("/api/simulate", json={"prompt": "api trace", "max_tokens": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["steps"]
    assert body["steps"][0]["trace"][0]["event"] == "step"


def test_openai_completion_does_not_mutate_interactive_engine():
    client = TestClient(app)
    client.post("/api/reset")
    before = client.get("/api/snapshot").json()["step_count"]

    response = client.post("/v1/completions", json={"prompt": "isolated", "max_tokens": 2})
    after = client.get("/api/snapshot").json()["step_count"]

    assert response.status_code == 200
    assert before == 0
    assert after == 0
