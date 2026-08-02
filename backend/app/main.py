from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vllm_lab.config import EngineConfig
from vllm_lab.engine.llm_engine import LLMEngine
from vllm_lab.kv_cache.formulas import LLAMA3_8B, LLAMA3_70B, ModelSpec, compute_memory_budget
from vllm_lab.types import StepOutput, make_group

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
    gpu_memory_gb: float = Field(default=80.0, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    model_weights_gb: float = Field(default=16.0, ge=0)
    block_size: int = Field(default=16, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    quantization: str | None = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "domainforge-triage-v0"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0)


# Educational multi-LoRA registry (ADR-022 Path B demo slice — not CUDA LoRA kernels).
ADAPTER_REGISTRY = [
    {"id": "domainforge-triage-v0", "status": "promoted", "base_model": "vllm-lab-simulator"},
    {"id": "domainforge-triage-dpo-v0", "status": "registered", "base_model": "vllm-lab-simulator"},
]


MODELS = {"llama-3-8b": LLAMA3_8B, "llama-3-70b": LLAMA3_70B}


def _step_to_dict(out: StepOutput) -> dict:
    return {
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
        "trace": [event.as_dict() for event in out.trace],
    }


@app.get("/health")
def health() -> dict:
    snap = _engine.snapshot()
    return {
        "status": "ok",
        "service": "vllm-architecture-lab",
        "version": "0.1.0",
        "engine": "pure_python_simulator",
        "wall_clock_latency": False,
        "gpu_backed": False,
        "adapters_registered": len(ADAPTER_REGISTRY),
        "step_count": int(snap.get("step_count", 0)),
    }


@app.get("/v1/observability/status")
def observability_status() -> dict:
    return {
        "source_of_truth": "In-process educational LLMEngine snapshot (/api/snapshot)",
        "exporters": [
            {
                "name": "EngineTrace",
                "state": "live",
                "detail": "steps[].trace events from POST /api/simulate — not SSE",
            },
            {
                "name": "OpsMetrics",
                "state": "live",
                "detail": "Queue + KV block counts; p95_latency_ms always null (no wall clock)",
            },
        ],
        "planes": {
            "engine": "pure_python_simulator",
            "gpu_utilization_pct": "scheduler_heuristic_35_plus_running_times_12",
            "multi_lora_path_b": True,
            "cuda_kernels": False,
        },
        "recommendation": "Use for teaching PagedAttention/continuous batching; use upstream vLLM for production serving.",
    }


@app.get("/v1/ops/metrics")
def ops_metrics() -> dict:
    from datetime import datetime, timezone

    snap = _engine.snapshot()
    kv = snap.get("kv_blocks") or {}
    allocated = kv.get("num_allocated_blocks", 0) if isinstance(kv, dict) else 0
    return {
        "service": "vllm-architecture-lab",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": int(snap.get("step_count", 0)),
        "success_rate_pct": 100.0,
        "p95_latency_ms": None,
        "active_entities": int(allocated),
        "slo": {"target_uptime_pct": 99.5, "success_target_pct": 95.0},
        "extra": {
            "queues": snap.get("queues"),
            "kv_blocks": kv,
            "engine": "pure_python_simulator",
            "wall_clock_latency": False,
            "gpu_backed": False,
            "gpu_utilization_pct": "heuristic_not_nvml",
            "multi_lora": {
                "path": "educational_b",
                "adapters": [a["id"] for a in ADAPTER_REGISTRY],
                "cuda_kernels": False,
            },
            "openai_compat_stubs": ["/v1/completions", "/v1/chat/completions"],
        },
    }


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
    return {"idle": False, **_step_to_dict(out)}


@app.post("/api/simulate")
def simulate(req: PromptRequest) -> dict:
    global _engine
    _engine = LLMEngine(EngineConfig(num_gpu_blocks=128, max_num_seqs=64))
    group = make_group(req.prompt, max_tokens=req.max_tokens)
    _engine.add_request(group)
    steps = []
    for out in _engine.run_until_idle(max_steps=req.max_tokens + 10):
        step = _step_to_dict(out)
        step["queues"] = out.scheduler.metadata
        steps.append(step)
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
    engine = LLMEngine(EngineConfig(num_gpu_blocks=128, max_num_seqs=64))
    group = make_group(req.prompt, max_tokens=req.max_tokens)
    engine.add_request(group)
    tokens: list[int] = []
    for out in engine.run_until_idle(max_steps=req.max_tokens + 5):
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


@app.get("/v1/adapters")
def list_adapters() -> dict:
    """Mock multi-LoRA adapter registry for DomainForge Path B demos (ADR-022)."""
    return {"object": "list", "data": ADAPTER_REGISTRY}


@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat stub with adapter/model id — educational multi-LoRA swap."""
    adapter_ids = {a["id"] for a in ADAPTER_REGISTRY}
    model = req.model if req.model in adapter_ids or req.model == "base" else "domainforge-triage-v0"
    user_bits = [m.content for m in req.messages if m.role == "user"]
    prompt = user_bits[-1] if user_bits else "triage"
    # Deterministic structured stub DomainForge can validate as triage JSON.
    triage = {
        "intent": "password_reset" if "password" in prompt.lower() else "general_inquiry",
        "category": "account_access" if "password" in prompt.lower() else "general",
        "priority": "medium",
        "entities": {},
        "suggested_action": "verify_identity_then_send_reset_link"
        if "password" in prompt.lower()
        else "gather_more_details",
        "cite_faq_ids": [],
        "confidence": 0.81,
        "summary": f"[vllm-lab adapter={model}] educational multi-LoRA decode",
    }
    import json as _json

    content = _json.dumps(triage)
    return {
        "id": f"chatcmpl-{model}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": max(2, (len(prompt) + len(content)) // 4),
        },
        "adapter_swapped": model in adapter_ids,
    }
