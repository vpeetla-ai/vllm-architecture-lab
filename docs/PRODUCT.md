# Product framing — vLLM Architecture Lab

## Who we serve

| Persona | Pain today | What they get |
|---------|------------|---------------|
| **Principal / staff engineer** | Hard to explain inference trade-offs in interviews | Interactive PagedAttention + batching demo |
| **Platform team** | vLLM docs are dense before first GPU cluster | Simulator + memory budget API without GPUs |
| **FDE / solutions architect** | Customers ask "why not just batch size 1?" | Side-by-side scheduler visualization |

## Job-to-be-done

> "Understand how high-throughput LLM serving works under the hood before tuning production inference."

## What we are NOT

- A production vLLM deployment
- A hosted inference API with real model weights
- A substitute for VAP's model router (application layer)

## Trade-offs

| Choice | Why | Cost |
|--------|-----|------|
| Python simulator | Readable, testable, free-tier friendly | Not kernel-accurate |
| Educational `/v1/completions` stub | Shows API contract | No real tokens |
| Vercel + Render split | Portfolio demo pattern | Cold starts on API |

## Related

- [Case study](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/case-studies/vllm-architecture-lab.md)
- [VAP INFERENCE.md](https://github.com/vpeetla-ai/venkat-ai-platform/blob/main/docs/INFERENCE.md)
