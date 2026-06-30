from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from vllm_lab.engine.llm_engine import LLMEngine
from vllm_lab.types import SequenceGroup, StepOutput


class AsyncLLMEngine:
    """Async wrapper around LLMEngine — mirrors vLLM AsyncLLMEngine pattern."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._queue: asyncio.Queue[StepOutput | None] = asyncio.Queue()

    async def add_request(self, group: SequenceGroup) -> str:
        return self.engine.add_request(group)

    async def step_async(self) -> StepOutput | None:
        return await asyncio.to_thread(self.engine.step)

    async def run_loop(self, max_steps: int = 500) -> AsyncGenerator[StepOutput, None]:
        for _ in range(max_steps):
            out = await self.step_async()
            if out is None:
                break
            yield out
            if not self.engine.scheduler.running and not self.engine.scheduler.waiting and not self.engine.scheduler.swapped:
                break
