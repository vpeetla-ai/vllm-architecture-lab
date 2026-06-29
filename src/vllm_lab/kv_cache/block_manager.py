from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib


class BlockState(str, Enum):
    FREE = "free"
    USED = "used"
    SHARED = "shared"
    EVICTED = "evicted"


@dataclass
class PhysicalBlock:
    block_id: int
    state: BlockState = BlockState.FREE
    ref_count: int = 0
    owner_seq_id: str | None = None
    on_cpu: bool = False


@dataclass
class BlockSpaceManager:
    """PagedAttention-style block allocator (educational simulator)."""

    block_size: int
    num_gpu_blocks: int
    num_cpu_blocks: int = 64
    blocks: list[PhysicalBlock] = field(default_factory=list)
    free_gpu: list[int] = field(default_factory=list)
    free_cpu: list[int] = field(default_factory=list)
    seq_block_table: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.blocks:
            self.blocks = [PhysicalBlock(i) for i in range(self.num_gpu_blocks + self.num_cpu_blocks)]
            self.free_gpu = list(range(self.num_gpu_blocks))
            self.free_cpu = list(range(self.num_gpu_blocks, self.num_gpu_blocks + self.num_cpu_blocks))

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return len(self.free_gpu) >= self.blocks_needed(num_tokens)

    def allocate(self, seq_id: str, num_tokens: int) -> list[int] | None:
        needed = self.blocks_needed(num_tokens)
        if len(self.free_gpu) < needed:
            return None
        block_ids = [self.free_gpu.pop(0) for _ in range(needed)]
        for bid in block_ids:
            b = self.blocks[bid]
            b.state = BlockState.USED
            b.owner_seq_id = seq_id
            b.ref_count = 1
            b.on_cpu = False
        self.seq_block_table[seq_id] = block_ids
        return block_ids

    def append_token(self, seq_id: str, num_tokens: int) -> list[int] | None:
        table = self.seq_block_table.get(seq_id, [])
        needed = self.blocks_needed(num_tokens)
        if len(table) >= needed:
            return table
        if not self.free_gpu:
            return None
        bid = self.free_gpu.pop(0)
        b = self.blocks[bid]
        b.state = BlockState.USED
        b.owner_seq_id = seq_id
        b.ref_count = 1
        table.append(bid)
        self.seq_block_table[seq_id] = table
        return table

    def free_sequence(self, seq_id: str) -> None:
        for bid in self.seq_block_table.pop(seq_id, []):
            b = self.blocks[bid]
            b.ref_count -= 1
            if b.ref_count <= 0:
                b.state = BlockState.FREE
                b.owner_seq_id = None
                b.on_cpu = False
                if bid < self.num_gpu_blocks:
                    self.free_gpu.append(bid)
                else:
                    self.free_cpu.append(bid)

    def share_prefix_blocks(self, seq_id: str, shared_block_ids: list[int]) -> list[int]:
        """Copy-on-write prefix sharing."""
        table: list[int] = []
        for bid in shared_block_ids:
            b = self.blocks[bid]
            b.ref_count += 1
            b.state = BlockState.SHARED
            table.append(bid)
        self.seq_block_table[seq_id] = table
        return table

    def swap_out(self, seq_id: str) -> bool:
        table = self.seq_block_table.get(seq_id, [])
        if not table:
            return False
        for bid in table:
            if bid >= self.num_gpu_blocks:
                continue
            if not self.free_cpu:
                return False
        new_table: list[int] = []
        for bid in table:
            if bid < self.num_gpu_blocks:
                cpu_bid = self.free_cpu.pop(0)
                self.blocks[bid].state = BlockState.FREE
                self.blocks[bid].owner_seq_id = None
                self.free_gpu.append(bid)
                cb = self.blocks[cpu_bid]
                cb.state = BlockState.EVICTED
                cb.owner_seq_id = seq_id
                cb.on_cpu = True
                new_table.append(cpu_bid)
            else:
                new_table.append(bid)
        self.seq_block_table[seq_id] = new_table
        return True

    def swap_in(self, seq_id: str) -> bool:
        table = self.seq_block_table.get(seq_id, [])
        needed_gpu = sum(1 for bid in table if self.blocks[bid].on_cpu)
        if len(self.free_gpu) < needed_gpu:
            return False
        new_table: list[int] = []
        for bid in table:
            b = self.blocks[bid]
            if b.on_cpu:
                gpu_bid = self.free_gpu.pop(0)
                self.free_cpu.append(bid)
                b.state = BlockState.FREE
                b.on_cpu = False
                gb = self.blocks[gpu_bid]
                gb.state = BlockState.USED
                gb.owner_seq_id = seq_id
                gb.on_cpu = False
                new_table.append(gpu_bid)
            else:
                new_table.append(bid)
        self.seq_block_table[seq_id] = new_table
        return True

    def snapshot(self) -> dict:
        return {
            "block_size": self.block_size,
            "num_gpu_blocks": self.num_gpu_blocks,
            "free_gpu": len(self.free_gpu),
            "free_cpu": len(self.free_cpu),
            "blocks": [
                {
                    "id": b.block_id,
                    "state": b.state.value,
                    "owner": b.owner_seq_id,
                    "on_cpu": b.on_cpu,
                    "ref_count": b.ref_count,
                }
                for b in self.blocks
            ],
            "sequences": self.seq_block_table,
        }


def hash_prefix(token_ids: list[int], block_size: int) -> str:
    data = ",".join(str(t) for t in token_ids[: block_size * 2])
    return hashlib.sha256(data.encode()).hexdigest()[:12]
