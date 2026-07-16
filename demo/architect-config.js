window.VLLM_LAB_API = window.VLLM_LAB_API || "https://vllm-architecture-lab-api.onrender.com";
window.ARCHITECT_CONFIG = {
  // Product already has Architecture / KV / Batching tabs — don't inject a second Architecture workbench tab.
  skipWorkbenchTabs: true,
  tagline:
    "Glass-box vLLM inference lab — replay the real engine trace (schedule → KV cache → decode → sample → finish) from a pure-Python PagedAttention + continuous-batching simulator.",
  metricsUrl: window.VLLM_LAB_API + "/v1/ops/metrics",
  metricsPath: "/v1/ops/metrics",
  metricLabels: { runs: "Engine steps", entities: "KV blocks allocated", latency: "P95 latency" },
  layers: [
    { tier: "L1", name: "Explorer UI", role: "Interactive teaching", components: ["Live simulator", "KV / Batching / Memory tabs", "Memory budget"] },
    { tier: "L2", name: "Engine core", role: "Scheduling semantics", components: ["Block manager", "Continuous batching", "Preemption"] },
    { tier: "L3", name: "KV cache", role: "Memory math", components: ["PagedAttention", "Block tables", "Prefix cache"] },
    { tier: "L4", name: "Ops", role: "Portfolio proof", components: ["/v1/ops/metrics", "Golden eval CI", "SLO.md"] },
  ],
  tradeoffs: [
    { decision: "Pure-Python simulator vs CUDA kernels", gain: "Inspectable state for interviews and CI", trade: "Not production throughput numbers" },
    { decision: "Educational tabs vs single demo page", gain: "Maps to FDE/customer conversations", trade: "More UI than minimal hello-world" },
    { decision: "Pairs with DomainForge ADR-022", gain: "Multi-LoRA serving story end-to-end", trade: "Two repos to keep aligned" },
    { decision: "Render API + Vercel static UI", gain: "Free portfolio hosting", trade: "Cold start on first simulate call" },
  ],
  adrLinks: [
    { title: "ADR-022 — Multi-LoRA serving target", href: "https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/adr/ADR-022-domainforge-vllm-multi-lora-serving.md" },
    { title: "Case study — vLLM Architecture Lab", href: "https://github.com/vpeetla-ai/ai-architecture-portfolio/blob/main/case-studies/vllm-architecture-lab.md" },
  ],
  docsLinks: [
    { title: "Architecture", href: "https://github.com/vpeetla-ai/vllm-architecture-lab/blob/main/docs/ARCHITECTURE.md" },
    { title: "SLO targets", href: "https://github.com/vpeetla-ai/vllm-architecture-lab/blob/main/docs/SLO.md" },
  ],
};
