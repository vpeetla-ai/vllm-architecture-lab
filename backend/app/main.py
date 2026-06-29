from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vllm_lab.config import EngineConfig
from vllm_lab.engine.llm_engine import LLMEngine
from vllm_lab.kv_cache.formulas import LLAMA3_8B, LLAMA3_70B, ModelSpec, compute_memory_budget
from vllm_lab.types import make_group

app = FastAPI(
    title="vLLM Architecture Lab API",
    description="Educational simulator — PagedAttention, continuous batching, KV budget",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = LLMEngine(EngineConfig(num_gpu_blocks=128, max_num_seqs=64))


class PromptRequest(BaseModel):
    prompt: str = "Hello from vLLM lab"
    max_tokens: int = Field(default=16, ge=1, le=256)


class MemoryBudgetRequest(BaseModel):
    model: str = "llama-3-8b"
    gpu_memory_gb: float = 80.0
    gpu_memory_utilization: float = 0.9
    model_weights_gb: float = 16.0
    block_size: int = 16
    tensor_parallel_size: int = 1
    quantization: str | None = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.7
    stream: bool = False


MODELS = {"llama-3-8b": LLAMA3_8B, "llama-3-70b": LLAMA3_70B}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "vllm-architecture-lab", "version": "0.1.0"}


@app.post("/api/reset")
def reset_engine() -> dict:
    global _engine
    _engine = LLMEngine(EngineConfig(num_gpu_blocks=128, max_num_seqs=64))
    return {"reset": True}


@app.get("/api/snapshot")
def snapshot() -> dict:
    return _engine.snapshot()


@app.post("/api/step")
def step_once() -> dict:
    out = _engine.step()
    if out is None:
        return {"idle": True}
    return {
        "idle": False,
        "step": out.step,
        "generated_tokens": out.generated_tokens,
        "finished_seq_ids": out.finished_seq_ids,
        "gpu_utilization_pct": out.gpu_utilization_pct,
        "scheduler": {
            "scheduled": out.scheduler.scheduled_seq_ids,
            "preempted": out.scheduler.preempted_seq_ids,
            "swapped_in": out.scheduler.swapped_in_seq_ids,
            "batched_tokens": out.scheduler.num_batched_tokens,
            "queues": out.scheduler.metadata,
        },
    }


@app.post("/api/simulate")
def simulate(req: PromptRequest) -> dict:
    global _engine
    _engine = LLMEngine(EngineConfig(num_gpu_blocks=128, max_num_seqs=64))
    group = make_group(req.prompt, max_tokens=req.max_tokens)
    _engine.add_request(group)
    steps = []
    for out in _engine.run_until_idle(max_steps=req.max_tokens + 10):
        steps.append(
            {
                "step": out.step,
                "generated_tokens": out.generated_tokens,
                "finished_seq_ids": out.finished_seq_ids,
                "gpu_utilization_pct": out.gpu_utilization_pct,
                "queues": out.scheduler.metadata,
            }
        )
    return {"steps": steps, "snapshot": _engine.snapshot()}


@app.post("/api/memory/budget")
def memory_budget(req: MemoryBudgetRequest) -> dict:
    spec = MODELS.get(req.model, LLAMA3_8B)
    result = compute_memory_budget(
        spec,
        gpu_memory_gb=req.gpu_memory_gb,
        gpu_memory_utilization=req.gpu_memory_utilization,
        model_weights_gb=req.model_weights_gb,
        block_size=req.block_size,
        tensor_parallel_size=req.tensor_parallel_size,
        quantization=req.quantization,
    )
    return {
        "model_name": result.model_name,
        "gpu_memory_gb": result.gpu_memory_gb,
        "gpu_memory_utilization": result.gpu_memory_utilization,
        "model_weights_gb": result.model_weights_gb,
        "kv_budget_gb": round(result.kv_budget_gb, 2),
        "block_size": result.block_size,
        "num_gpu_blocks": result.num_gpu_blocks,
        "max_concurrent_tokens": result.max_concurrent_tokens,
        "formula_notes": result.formula_notes,
    }


@app.post("/v1/completions")
def openai_completions(req: CompletionRequest) -> dict:
    """OpenAI-compatible stub — runs educational simulator, not real LLM."""
    group = make_group(req.prompt, max_tokens=req.max_tokens)
    _engine.add_request(group)
    tokens: list[int] = []
    for out in _engine.run_until_idle(max_steps=req.max_tokens + 5):
        for tid in out.generated_tokens.values():
            tokens.append(tid)
    text = "".join(chr(t % 128) if 32 <= (t % 128) < 127 else "?" for t in tokens)
    return {
        "id": f"cmpl-{group.group_id}",
        "object": "text_completion",
        "model": "vllm-lab-simulator",
        "choices": [{"text": text or "[simulated decode]", "index": 0, "finish_reason": "length"}],
        "usage": {"prompt_tokens": group.sequences[0].num_prompt_tokens, "completion_tokens": len(tokens), "total_tokens": group.sequences[0].num_prompt_tokens + len(tokens)},
    }
