from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Simplified model dimensions for KV budget math."""

    name: str
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: int = 2  # FP16/BF16


# Llama-3 8B approximate
LLAMA3_8B = ModelSpec("llama-3-8b", num_layers=32, num_heads=32, head_dim=128, dtype_bytes=2)
LLAMA3_70B = ModelSpec("llama-3-70b", num_layers=80, num_heads=64, head_dim=128, dtype_bytes=2)


def kv_bytes_per_token_per_layer(spec: ModelSpec) -> int:
    """2 × num_heads × head_dim × dtype_bytes (K + V)."""
    return 2 * spec.num_heads * spec.head_dim * spec.dtype_bytes


def kv_bytes_per_token(spec: ModelSpec) -> int:
    return kv_bytes_per_token_per_layer(spec) * spec.num_layers


def kv_mb_per_token(spec: ModelSpec) -> float:
    return kv_bytes_per_token(spec) / (1024 * 1024)


@dataclass
class MemoryBudgetResult:
    model_name: str
    gpu_memory_gb: float
    gpu_memory_utilization: float
    model_weights_gb: float
    kv_budget_gb: float
    block_size: int
    kv_bytes_per_block: int
    num_gpu_blocks: int
    max_concurrent_tokens: int
    formula_notes: list[str]


def compute_memory_budget(
    spec: ModelSpec,
    *,
    gpu_memory_gb: float = 80.0,
    gpu_memory_utilization: float = 0.9,
    model_weights_gb: float = 16.0,
    block_size: int = 16,
    tensor_parallel_size: int = 1,
    quantization: str | None = None,
) -> MemoryBudgetResult:
    """Estimate num_gpu_blocks from the vLLM startup formula."""
    notes: list[str] = []
    weights = model_weights_gb
    if quantization == "awq_int4":
        weights = model_weights_gb / 4
        notes.append("AWQ INT4: weights ÷ 4")
    if tensor_parallel_size > 1:
        weights = weights / tensor_parallel_size
        notes.append(f"TP={tensor_parallel_size}: weight shard per GPU")

    avail = gpu_memory_gb * gpu_memory_utilization
    kv_budget = max(0.0, avail - weights)
    kv_per_block_bytes = kv_bytes_per_token(spec) * block_size
    num_blocks = int((kv_budget * 1024**3) // kv_per_block_bytes) if kv_per_block_bytes else 0

    notes.extend(
        [
            f"avail = {gpu_memory_gb}GB × {gpu_memory_utilization}",
            f"kv_budget = avail − weights ({weights:.1f}GB)",
            f"num_blocks = kv_budget // (block_size × kv_per_token)",
        ]
    )

    return MemoryBudgetResult(
        model_name=spec.name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_utilization=gpu_memory_utilization,
        model_weights_gb=weights,
        kv_budget_gb=kv_budget,
        block_size=block_size,
        kv_bytes_per_block=kv_per_block_bytes,
        num_gpu_blocks=num_blocks,
        max_concurrent_tokens=num_blocks * block_size,
        formula_notes=notes,
    )
