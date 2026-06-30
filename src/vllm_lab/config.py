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

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.num_gpu_blocks < 1:
            raise ValueError("num_gpu_blocks must be >= 1")
        if self.num_cpu_blocks < 0:
            raise ValueError("num_cpu_blocks must be >= 0")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.preemption_mode not in {"swap", "recompute"}:
            raise ValueError("preemption_mode must be 'swap' or 'recompute'")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
