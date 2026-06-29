from __future__ import annotations

import argparse

from vllm_lab.config import EngineConfig
from vllm_lab.engine.llm_engine import LLMEngine
from vllm_lab.kv_cache.formulas import LLAMA3_8B, compute_memory_budget
from vllm_lab.types import make_group


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM architecture lab CLI")
    parser.add_argument("--prompt", default="Explain PagedAttention in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--budget", action="store_true", help="Print KV memory budget only")
    args = parser.parse_args()

    if args.budget:
        result = compute_memory_budget(LLAMA3_8B, gpu_memory_gb=80, model_weights_gb=16)
        print(f"Model: {result.model_name}")
        print(f"KV budget: {result.kv_budget_gb:.1f} GB → {result.num_gpu_blocks} blocks")
        print(f"Max tokens: {result.max_concurrent_tokens:,}")
        return

    config = EngineConfig(num_gpu_blocks=64, max_num_seqs=32)
    engine = LLMEngine(config=config)
    group = make_group(args.prompt, max_tokens=args.max_tokens)
    engine.add_request(group)

    print(f"Running continuous batching sim: prompt={args.prompt!r}")
    for out in engine.run_until_idle():
        tokens = ", ".join(f"{sid}:{tid}" for sid, tid in out.generated_tokens.items())
        print(
            f"  step {out.step}: tokens=[{tokens}] "
            f"util={out.gpu_utilization_pct:.0f}% "
            f"queues={out.scheduler.metadata}"
        )
        if out.finished_seq_ids:
            print(f"  finished: {out.finished_seq_ids}")

    snap = engine.snapshot()
    print(f"\nPrefix cache hit rate: {snap['queues']['prefix_cache']['hit_rate_pct']}%")


if __name__ == "__main__":
    main()
