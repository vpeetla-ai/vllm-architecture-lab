from __future__ import annotations

from dataclasses import dataclass, field

from vllm_lab.config import EngineConfig
from vllm_lab.kv_cache.block_manager import BlockSpaceManager
from vllm_lab.kv_cache.prefix_cache import PrefixCache
from vllm_lab.sampling.sampler import Sampler
from vllm_lab.scheduler.scheduler import Scheduler
from vllm_lab.types import SequenceGroup, StepOutput


@dataclass
class LLMEngine:
    """Synchronous vLLM-style engine: add_request → step() loop."""

    config: EngineConfig
    block_manager: BlockSpaceManager = field(init=False)
    prefix_cache: PrefixCache = field(init=False)
    scheduler: Scheduler = field(init=False)
    sampler: Sampler = field(init=False)
    step_count: int = 0
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.block_manager = BlockSpaceManager(
            block_size=self.config.block_size,
            num_gpu_blocks=self.config.num_gpu_blocks,
            num_cpu_blocks=self.config.num_cpu_blocks,
        )
        self.prefix_cache = PrefixCache(block_size=self.config.block_size)
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            prefix_cache=self.prefix_cache,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
        )
        self.sampler = Sampler()

    def add_request(self, group: SequenceGroup) -> str:
        self.scheduler.add_request(group)
        self.trace.append(f"add_request group={group.group_id} prompt_tokens={group.sequences[0].num_prompt_tokens}")
        return group.group_id

    def abort_request(self, seq_id: str) -> None:
        self.scheduler.abort(seq_id)
        self.trace.append(f"abort_request seq={seq_id}")

    def step(self) -> StepOutput | None:
        if not self.scheduler.running and not self.scheduler.waiting:
            return None

        sched_out = self.scheduler.schedule()
        generated: dict[str, int] = {}
        finished: list[str] = []

        for seq in list(self.scheduler.running):
            group = self.scheduler.groups[self.scheduler.seq_to_group[seq.seq_id]]
            params = group.sampling_params
            token_id = self.sampler.sample([0.1, 0.3, 0.6], params)
            seq.output_token_ids.append(token_id)
            generated[seq.seq_id] = token_id

            if seq.is_finished(params):
                finished.append(seq.seq_id)
                self.scheduler.finish_sequence(seq.seq_id)
                self.trace.append(f"finish seq={seq.seq_id} output_len={seq.num_output_tokens}")

        self.step_count += 1
        running = len(self.scheduler.running)
        waiting = len(self.scheduler.waiting)
        util = min(95.0, 35.0 + running * 12.0) if running else 10.0

        return StepOutput(
            step=self.step_count,
            generated_tokens=generated,
            finished_seq_ids=finished,
            scheduler=sched_out,
            gpu_utilization_pct=util,
            trace=[f"step={self.step_count} batched={sched_out.num_batched_tokens}"],
        )

    def run_until_idle(self, max_steps: int = 500) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        for _ in range(max_steps):
            out = self.step()
            if out is None:
                break
            outputs.append(out)
            if not self.scheduler.running and not self.scheduler.waiting:
                break
        return outputs

    def snapshot(self) -> dict:
        return {
            "step_count": self.step_count,
            "queues": self.scheduler.queue_stats(),
            "kv_blocks": self.block_manager.snapshot(),
            "config": {
                "block_size": self.config.block_size,
                "max_num_seqs": self.config.max_num_seqs,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "enable_chunked_prefill": self.config.enable_chunked_prefill,
                "preemption_mode": self.config.preemption_mode,
            },
        }
