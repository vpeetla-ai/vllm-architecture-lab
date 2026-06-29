from __future__ import annotations

import random
from dataclasses import dataclass

from vllm_lab.types import SamplingParams


@dataclass
class Sampler:
    """Applies temperature / top-k / top-p to logits (educational stub)."""

    def sample(self, logits: list[float], params: SamplingParams) -> int:
        if not logits:
            return 0
        if params.temperature <= 0:
            return max(range(len(logits)), key=lambda i: logits[i])

        scaled = [l / max(params.temperature, 1e-6) for l in logits]
        probs = _softmax(scaled)

        if params.top_k > 0:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            keep = {i for i, _ in indexed[: params.top_k]}
            probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]

        if params.top_p < 1.0:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            cumulative = 0.0
            keep: set[int] = set()
            for i, p in indexed:
                cumulative += p
                keep.add(i)
                if cumulative >= params.top_p:
                    break
            probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]

        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [pow(2.718281828, x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]
