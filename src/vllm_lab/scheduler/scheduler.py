from __future__ import annotations

from dataclasses import dataclass, field

from vllm_lab.kv_cache.block_manager import BlockSpaceManager
from vllm_lab.kv_cache.prefix_cache import PrefixCache
from vllm_lab.kv_cache.block_manager import hash_prefix
from vllm_lab.types import SchedulerOutput, Sequence, SequenceGroup, SequenceStatus


@dataclass
class Scheduler:
    """FCFS scheduler with waiting / running / swapped queues."""

    block_manager: BlockSpaceManager
    prefix_cache: PrefixCache
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    waiting: list[SequenceGroup] = field(default_factory=list)
    running: list[Sequence] = field(default_factory=list)
    swapped: list[Sequence] = field(default_factory=list)
    groups: dict[str, SequenceGroup] = field(default_factory=dict)
    seq_to_group: dict[str, str] = field(default_factory=dict)

    def add_request(self, group: SequenceGroup) -> None:
        self.groups[group.group_id] = group
        for seq in group.sequences:
            self.seq_to_group[seq.seq_id] = group.group_id
        self.waiting.append(group)

    def abort(self, seq_id: str) -> None:
        seq = self._find_seq(seq_id)
        if not seq:
            return
        seq.status = SequenceStatus.ABORTED
        self._remove_from_queues(seq)
        self.block_manager.free_sequence(seq_id)

    def _find_seq(self, seq_id: str) -> Sequence | None:
        for seq in self.running + self.swapped:
            if seq.seq_id == seq_id:
                return seq
        for group in self.waiting:
            for seq in group.sequences:
                if seq.seq_id == seq_id:
                    return seq
        return None

    def _remove_from_queues(self, seq: Sequence) -> None:
        self.running = [s for s in self.running if s.seq_id != seq.seq_id]
        self.swapped = [s for s in self.swapped if s.seq_id != seq.seq_id]
        self.waiting = [g for g in self.waiting if all(s.seq_id != seq.seq_id for s in g.sequences)]

    def schedule(self) -> SchedulerOutput:
        scheduled: list[str] = []
        preempted: list[str] = []
        swapped_in: list[str] = []
        batched_tokens = 0

        # Promote swapped sequences when GPU blocks free up
        for seq in list(self.swapped):
            if len(self.running) >= self.max_num_seqs:
                break
            if self.block_manager.swap_in(seq.seq_id):
                self.swapped.remove(seq)
                self.running.append(seq)
                seq.status = SequenceStatus.RUNNING
                swapped_in.append(seq.seq_id)
                scheduled.append(seq.seq_id)

        # Admit waiting sequences (FCFS)
        admitted: list[SequenceGroup] = []
        for group in self.waiting:
            if len(self.running) >= self.max_num_seqs:
                break
            seq = group.sequences[0]
            prefix_hash = hash_prefix(seq.prompt_token_ids, self.block_manager.block_size)
            cached = self.prefix_cache.lookup(prefix_hash)
            if cached:
                seq.shared_prefix_hash = prefix_hash
                seq.block_ids = self.block_manager.share_prefix_blocks(seq.seq_id, cached)
            else:
                blocks = self.block_manager.allocate(seq.seq_id, seq.num_prompt_tokens)
                if blocks is None:
                    break
                seq.block_ids = blocks
                self.prefix_cache.register(prefix_hash, blocks[: max(1, len(blocks) // 2)])
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq.seq_id)
            batched_tokens += seq.num_total_tokens
            admitted.append(group)

        for g in admitted:
            self.waiting.remove(g)

        # Grow running sequences; preempt if out of blocks
        for seq in list(self.running):
            if batched_tokens >= self.max_num_batched_tokens:
                break
            group = self.groups[self.seq_to_group[seq.seq_id]]
            new_total = seq.num_total_tokens + 1
            blocks = self.block_manager.append_token(seq.seq_id, new_total)
            if blocks is None:
                victim = self._pick_preemption_victim(seq)
                if victim and self.block_manager.swap_out(victim.seq_id):
                    victim.status = SequenceStatus.SWAPPED
                    self.running.remove(victim)
                    self.swapped.append(victim)
                    preempted.append(victim.seq_id)
                    blocks = self.block_manager.append_token(seq.seq_id, new_total)
            if blocks:
                batched_tokens += 1

        return SchedulerOutput(
            scheduled_seq_ids=scheduled,
            preempted_seq_ids=preempted,
            swapped_in_seq_ids=swapped_in,
            num_batched_tokens=batched_tokens,
            metadata={
                "waiting": len(self.waiting),
                "running": len(self.running),
                "swapped": len(self.swapped),
            },
        )

    def _pick_preemption_victim(self, protected: Sequence) -> Sequence | None:
        candidates = [s for s in self.running if s.seq_id != protected.seq_id]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.priority)

    def finish_sequence(self, seq_id: str) -> None:
        seq = self._find_seq(seq_id)
        if not seq:
            return
        seq.status = SequenceStatus.FINISHED
        self._remove_from_queues(seq)
        self.block_manager.free_sequence(seq_id)

    def queue_stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "prefix_cache": self.prefix_cache.stats(),
        }
