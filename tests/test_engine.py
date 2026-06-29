import pytest

from vllm_lab.kv_cache.block_manager import BlockSpaceManager
from vllm_lab.kv_cache.formulas import LLAMA3_8B, compute_memory_budget, kv_bytes_per_token
from vllm_lab.config import EngineConfig
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


def test_prefix_sharing_increments_refcount():
    mgr = BlockSpaceManager(block_size=16, num_gpu_blocks=8)
    shared = mgr.allocate("seq-a", 16)
    assert shared
    mgr.share_prefix_blocks("seq-b", shared)
    assert mgr.blocks[shared[0]].ref_count == 2
    assert mgr.blocks[shared[0]].state.value == "shared"


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
