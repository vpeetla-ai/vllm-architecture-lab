from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    block_size: int = 16
    num_gpu_blocks: int = 128
    num_cpu_blocks: int = 64
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    preemption_mode: str = "swap"  # swap | recompute
    gpu_memory_utilization: float = 0.9
