from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class Sequence:
    seq_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_ids: list[int] = field(default_factory=list)
    priority: int = 0
    shared_prefix_hash: str | None = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def is_finished(self, params: SamplingParams) -> bool:
        return self.num_output_tokens >= params.max_tokens


@dataclass
class SequenceGroup:
    group_id: str
    sequences: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = 0.0


def new_sequence(prompt: str | list[int]) -> Sequence:
    if isinstance(prompt, str):
        token_ids = [ord(c) % 256 for c in prompt[:64]]
    else:
        token_ids = list(prompt)
    return Sequence(
        seq_id=str(uuid.uuid4())[:8],
        prompt_token_ids=token_ids,
    )


def make_group(prompt: str, max_tokens: int = 32) -> SequenceGroup:
    seq = Sequence(
        seq_id=str(uuid.uuid4())[:8],
        prompt_token_ids=[ord(c) % 256 for c in prompt[:64]],
    )
    return SequenceGroup(
        group_id=str(uuid.uuid4())[:8],
        sequences=[seq],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


@dataclass
class SchedulerOutput:
    scheduled_seq_ids: list[str]
    preempted_seq_ids: list[str]
    swapped_in_seq_ids: list[str]
    num_batched_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutput:
    step: int
    generated_tokens: dict[str, int]
    finished_seq_ids: list[str]
    scheduler: SchedulerOutput
    gpu_utilization_pct: float
    trace: list[str] = field(default_factory=list)
