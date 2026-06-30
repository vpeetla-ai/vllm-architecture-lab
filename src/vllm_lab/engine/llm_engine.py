from __future__ import annotations

from dataclasses import dataclass, field

from vllm_lab.config import EngineConfig
from vllm_lab.kv_cache.block_manager import BlockSpaceManager
from vllm_lab.kv_cache.prefix_cache import PrefixCache
from vllm_lab.sampling.sampler import Sampler
from vllm_lab.scheduler.scheduler import Scheduler
from vllm_lab.types import SequenceGroup, StepOutput, TraceEvent


@dataclass
class LLMEngine:
    """Synchronous vLLM-style engine: add_request → step() loop."""

    config: EngineConfig
    block_manager: BlockSpaceManager = field(init=False)
    prefix_cache: PrefixCache = field(init=False)
    scheduler: Scheduler = field(init=False)
    sampler: Sampler = field(init=False)
    step_count: int = 0
    trace: list[TraceEvent] = field(default_factory=list)

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
        self.trace.append(
            TraceEvent(
                event="add_request",
                group_id=group.group_id,
                seq_id=group.sequences[0].seq_id,
                message="Added request to scheduler waiting queue.",
                data={"prompt_tokens": group.sequences[0].num_prompt_tokens},
            )
        )
        return group.group_id

    def abort_request(self, seq_id: str) -> None:
        self.scheduler.abort(seq_id)
        self.trace.append(
            TraceEvent(
                event="abort_request",
                seq_id=seq_id,
                message="Aborted request and freed associated KV blocks.",
            )
        )

    def step(self) -> StepOutput | None:
        if not self.scheduler.running and not self.scheduler.waiting and not self.scheduler.swapped:
            return None

        sched_out = self.scheduler.schedule()
        generated: dict[str, int] = {}
        finished: list[str] = []
        trace_events = list(sched_out.trace_events)

        for seq in list(self.scheduler.running):
            group = self.scheduler.groups[self.scheduler.seq_to_group[seq.seq_id]]
            params = group.sampling_params
            token_id = self.sampler.sample([0.1, 0.3, 0.6], params)
            seq.output_token_ids.append(token_id)
            generated[seq.seq_id] = token_id
            trace_events.append(
                TraceEvent(
                    event="sample",
                    seq_id=seq.seq_id,
                    group_id=group.group_id,
                    message="Sampled one educational stub token.",
                    data={"token_id": token_id, "output_tokens": seq.num_output_tokens},
                )
            )

            if seq.is_finished(params):
                finished.append(seq.seq_id)
                self.scheduler.finish_sequence(seq.seq_id)
                trace_events.append(
                    TraceEvent(
                        event="finish",
                        seq_id=seq.seq_id,
                        group_id=group.group_id,
                        message="Sequence reached max_tokens and freed KV blocks.",
                        data={"output_tokens": seq.num_output_tokens},
                    )
                )

        self.step_count += 1
        running = len(self.scheduler.running)
        waiting = len(self.scheduler.waiting)
        util = min(95.0, 35.0 + running * 12.0) if running else 10.0
        trace_events.insert(
            0,
            TraceEvent(
                event="step",
                message="Completed one scheduler/decode step.",
                data={"step": self.step_count, "batched_tokens": sched_out.num_batched_tokens},
            ),
        )
        self.trace.extend(trace_events)

        return StepOutput(
            step=self.step_count,
            generated_tokens=generated,
            finished_seq_ids=finished,
            scheduler=sched_out,
            gpu_utilization_pct=util,
            trace=trace_events,
        )

    def run_until_idle(self, max_steps: int = 500) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        for _ in range(max_steps):
            out = self.step()
            if out is None:
                break
            outputs.append(out)
            if not self.scheduler.running and not self.scheduler.waiting and not self.scheduler.swapped:
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
            "trace": [event.as_dict() for event in self.trace[-50:]],
        }
