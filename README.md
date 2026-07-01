# vLLM Architecture Lab

## Agent skills (Cursor + Codex)

Org skills: [vpeetla-ai-skills](https://github.com/vpeetla-ai/vpeetla-ai-skills). This repo includes `.cursor/skills/`, `AGENTS.md`, and `CONTEXT.md`.

```bash
git clone https://github.com/vpeetla-ai/vpeetla-ai-skills.git
./vpeetla-ai-skills/scripts/install.sh --cursor --codex --project .
```

---

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://vllm-architecture-lab.vercel.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7)](https://vllm-architecture-lab-api.onrender.com/health)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Org](https://img.shields.io/badge/GitHub-vpeetla--ai-blue)](https://github.com/vpeetla-ai)

**Educational vLLM inference engine** — PagedAttention block allocator, continuous batching scheduler, KV cache budget formulas, and interactive architecture explorer.

> Not a production vLLM fork. A **reference lab** for Principal AI Architect / FDE interviews: understand *why* vLLM achieves 3–4× throughput vs static batching.

[▶ Live demo](https://vllm-architecture-lab.vercel.app) · [API](https://vllm-architecture-lab-api.onrender.com/health) · [Architecture tabs](demo/index.html) · [venkat-ai.com/work](https://venkat-ai.com/work)

**Portfolio:** [Case study](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/case-studies/vllm-architecture-lab.md) · [Architecture](docs/ARCHITECTURE.md) · [Product / tradeoffs](docs/PRODUCT.md)

## What you'll learn

| Concept | Implementation |
|---------|----------------|
| **PagedAttention** | `BlockSpaceManager` — logical → physical blocks, CoW prefix sharing |
| **Continuous batching** | `Scheduler` — waiting / running / swapped queues, per-step slot reuse |
| **Prefix cache** | Hash-based KV block reuse (60–80% hit in chat workloads) |
| **KV budget formula** | `compute_memory_budget()` — H100 + Llama-3 sizing |
| **AsyncLLMEngine** | `LLMEngine.step()` + async wrapper |
| **OpenAI API shape** | `POST /v1/completions` stub on educational simulator |

---

## 60-second architecture

```text
Client → FastAPI (/v1/completions, /api/simulate)
      → AsyncLLMEngine
      → Scheduler (FCFS, preemption, swap)
      → BlockSpaceManager (PagedAttention pages)
      → Worker/Sampler (stub — no CUDA required)
      → token stream
```

Interactive explorer: **5 tabs** — Architecture · KV Cache · Batching · Memory · FDE Relevance.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# CLI simulator
python -m vllm_lab --prompt "Explain continuous batching" --max-tokens 8

# KV budget calculator
python -m vllm_lab --budget

# API server
uvicorn backend.app.main:app --reload --port 8000

# Tests (no GPU, no API keys)
pytest -q
```

Open `demo/index.html` locally or deploy to Vercel. Set `demo/config.js` → `VLLM_LAB_API` to your Render API URL for live sim.

---

## Implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| BlockSpaceManager (paging) | **Implemented** | allocate, free, swap, CoW |
| Scheduler (3 queues) | **Implemented** | FCFS, preemption, continuous batch |
| Prefix cache | **Implemented** | hash lookup + stats |
| KV budget formulas | **Implemented** | Llama-3 8B/70B, AWQ, TP |
| LLMEngine step loop | **Implemented** | educational token stub |
| FastAPI + OpenAI stub | **Implemented** | `/v1/completions`, `/api/simulate` |
| Interactive HTML explorer | **Implemented** | 5-tab architecture UI |
| PagedAttention CUDA kernel | **Conceptual** | documented, not implemented |
| FlashAttention / real model | **Conceptual** | use upstream vLLM for prod |
| Tensor parallel / Ray | **Conceptual** | topology in Memory tab |
| Speculative decoding | **Conceptual** | batching tab explains mechanics |

---

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health check |
| `POST /api/simulate` | Run full request through engine, return step trace |
| `POST /api/step` | Single `step()` call |
| `GET /api/snapshot` | KV blocks + queue state |
| `POST /api/memory/budget` | GPU KV budget calculator |
| `POST /v1/completions` | OpenAI-compatible stub |

---

## Deploy

| Layer | Host | URL |
|-------|------|-----|
| UI | Vercel | https://vllm-architecture-lab.vercel.app |
| API | Render | https://vllm-architecture-lab-api.onrender.com |

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/vpeetla-ai/vllm-architecture-lab)

See [docs/DEPLOY.md](docs/DEPLOY.md) for full steps.

---

## Repo layout

```text
src/vllm_lab/
  kv_cache/       # BlockSpaceManager, PrefixCache, formulas
  scheduler/      # FCFS continuous batching
  engine/         # LLMEngine, AsyncLLMEngine
  sampling/       # top-p, top-k, temperature
backend/app/      # FastAPI
demo/             # Interactive architecture HTML
tests/            # pytest (deterministic, no GPU)
docs/ARCHITECTURE.md
```

---

## Connect

- Part of [vpeetla-ai](https://github.com/vpeetla-ai) reference stack
- Essay: [From Multi-Agent OS to Agent Governance](https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/case-studies/from-multi-agent-os-to-agent-governance.md)
- Production inference: use [vLLM](https://github.com/vllm-project/vllm) upstream

MIT License
