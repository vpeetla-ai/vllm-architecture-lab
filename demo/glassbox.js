/**
 * vLLM Architecture Lab glass-box center — honest engine trace replay.
 *
 * Maps the real `steps[].trace` events returned by POST /api/simulate onto a
 * five-phase serving pipeline. Counts are real simulator events; there is NO
 * wall-clock latency because this is a pure-Python educational engine — the
 * ops strip surfaces engine steps, tokens, batched-token budget and queues,
 * and the util figure is labelled a heuristic (35 + running*12) not a GPU read.
 */
(function () {
  // event -> phase mapping (see src/vllm_lab/{engine,scheduler})
  const EVENT_PHASE = {
    add_request: "admit",
    admit: "admit",
    swap_in: "admit",
    preempt: "admit",
    abort_request: "admit",
    cache_hit: "kv",
    cache_miss: "kv",
    cache_stale: "kv",
    decode_slot: "kv",
    step: "decode",
    sample: "sample",
    finish: "finish",
  };

  const CANONICAL = [
    { id: "admit", label: "Schedule", detail: "waiting → running", accent: "var(--sched)" },
    { id: "kv", label: "KV cache", detail: "PagedAttention blocks", accent: "var(--kv2)" },
    { id: "decode", label: "Decode", detail: "batched forward step", accent: "var(--batch)" },
    { id: "sample", label: "Sample", detail: "next-token logits", accent: "var(--gpu)" },
    { id: "finish", label: "Finish", detail: "free KV blocks", accent: "var(--api)" },
  ];

  const els = {
    pipeline: () => document.getElementById("gbPipeline"),
    gate: () => document.getElementById("gbGate"),
    log: () => document.getElementById("gbEventLog"),
    ops: () => document.getElementById("gbOpsStrip"),
    badge: () => document.getElementById("gbSourceBadge"),
  };

  let timer = null;
  let activeId = null;
  let done = new Set();

  function setBadge(source) {
    const b = els.badge();
    if (!b) return;
    b.className = "gb-source-badge";
    if (source === "live") {
      b.classList.add("live");
      b.textContent = "engine trace";
    } else if (source === "running") {
      b.classList.add("running");
      b.textContent = "running…";
    } else {
      b.textContent = "awaiting run";
    }
  }

  function setGate(text) {
    const g = els.gate();
    if (g) g.innerHTML = text;
  }

  function clearLog() {
    const log = els.log();
    if (log) log.innerHTML = "";
  }

  function appendLog(line) {
    const log = els.log();
    if (!log) return;
    if (log.querySelector(".muted")) log.innerHTML = "";
    const row = document.createElement("div");
    row.className = "ev-live";
    row.textContent = line;
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  function setOps(meta) {
    const ops = els.ops();
    if (!ops) return;
    ops.innerHTML =
      "<span><strong>steps</strong> " +
      (meta.steps ?? "—") +
      "</span><span><strong>tokens</strong> " +
      (meta.tokens ?? "—") +
      "</span><span><strong>util*</strong> " +
      (meta.util ?? "—") +
      "</span><span><strong>queues</strong> " +
      (meta.queues ?? "—") +
      "</span><span><strong>latency</strong> " +
      (meta.latency ?? "n/a") +
      "</span>";
  }

  function renderNodes(counts) {
    const root = els.pipeline();
    if (!root) return;
    counts = counts || {};
    root.innerHTML = CANONICAL.map((p, i) => {
      const cls =
        activeId === p.id ? " gb-active" : done.has(p.id) ? " gb-done" : "";
      const c = counts[p.id];
      const badge = c != null ? c + " ev" : "—";
      return (
        (i > 0 ? '<span class="gb-agent-arrow" aria-hidden="true">→</span>' : "") +
        '<div class="gb-agent-node' +
        cls +
        '" data-phase-id="' +
        p.id +
        '" style="--node-accent:' +
        p.accent +
        '">' +
        '<span class="gb-agent-idx">' +
        String(i + 1).padStart(2, "0") +
        "</span>" +
        "<div><strong>" +
        p.label +
        "</strong><small>" +
        p.detail +
        "</small></div><em>" +
        badge +
        "</em></div>"
      );
    }).join("");
  }

  function highlight(id) {
    activeId = id;
    document.querySelectorAll(".gb-agent-node").forEach((n) => {
      const pid = n.getAttribute("data-phase-id");
      n.classList.toggle("gb-active", pid === activeId);
      n.classList.toggle("gb-done", done.has(pid) && pid !== activeId);
    });
  }

  function clearTimer() {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  }

  /** Aggregate real trace events from every step (+ snapshot tail) into phase counts. */
  function aggregate(data) {
    const counts = {};
    const traces = [];
    (data.steps || []).forEach((s) => {
      (s.trace || []).forEach((e) => traces.push(e));
    });
    // snapshot.trace holds engine-level events (add_request) not always in per-step trace
    if (data.snapshot && Array.isArray(data.snapshot.trace)) {
      data.snapshot.trace.forEach((e) => {
        if (e && e.event === "add_request") traces.push(e);
      });
    }
    traces.forEach((e) => {
      const phase = EVENT_PHASE[e && e.event];
      if (!phase) return;
      counts[phase] = (counts[phase] || 0) + 1;
    });
    return counts;
  }

  function summarize(data) {
    const steps = data.steps || [];
    const stepCount =
      (data.snapshot && data.snapshot.step_count) || steps.length;
    let tokens = 0;
    let lastUtil = null;
    steps.forEach((s) => {
      tokens += Object.keys(s.generated_tokens || {}).length;
      if (s.gpu_utilization_pct != null) lastUtil = s.gpu_utilization_pct;
    });
    const q = (steps[steps.length - 1] || {}).queues ||
      (data.snapshot && data.snapshot.queues) || {};
    const queues =
      "w" + (q.waiting ?? 0) + " r" + (q.running ?? 0) + " s" + (q.swapped ?? 0);
    return { steps: stepCount, tokens, util: lastUtil != null ? lastUtil + "%" : "—", queues };
  }

  function replay(data) {
    clearTimer();
    done = new Set();
    activeId = null;
    clearLog();

    const counts = aggregate(data);
    const summary = summarize(data);
    renderNodes(counts);
    setBadge("live");
    setOps({
      steps: summary.steps,
      tokens: summary.tokens,
      util: summary.util,
      queues: summary.queues,
      latency: "n/a (sim)",
    });

    // Only replay phases that actually fired at least one event.
    const order = CANONICAL.filter((p) => counts[p.id]).map((p) => p.id);
    if (!order.length) {
      setGate("No trace events returned — engine idled. Try a longer prompt or more tokens.");
      return;
    }

    let i = 0;
    let prev = null;
    const tick = () => {
      if (i >= order.length) {
        if (prev) done.add(prev);
        activeId = null;
        highlight(null);
        setGate(
          "Replayed <strong>" +
            order.length +
            "</strong> serving phases from real engine trace · <strong>" +
            summary.steps +
            "</strong> decode steps, <strong>" +
            summary.tokens +
            "</strong> tokens. No wall-clock ms — pure-Python simulator."
        );
        if (typeof window.VllmRefreshMetrics === "function") window.VllmRefreshMetrics();
        return;
      }
      const id = order[i];
      if (prev) done.add(prev);
      highlight(id);
      prev = id;
      const meta = CANONICAL.find((c) => c.id === id);
      setGate(
        "<strong>" + meta.label + "</strong> — " + meta.detail +
          " · " + counts[id] + " events"
      );
      appendLog("▸ " + id + " · " + counts[id] + " events");
      i += 1;
      timer = setTimeout(tick, 420);
    };
    tick();
  }

  window.GlassBox = {
    reset() {
      clearTimer();
      done = new Set();
      activeId = null;
      renderNodes({});
      const log = els.log();
      if (log) {
        log.innerHTML =
          '<div class="muted" style="font-style:italic">No trace yet — run a simulation to replay engine phases.</div>';
      }
      setBadge("idle");
      setGate(
        "Schedule → KV cache → Decode → Sample → Finish. Replay maps real <code>steps[].trace</code> events from <code>POST /api/simulate</code>."
      );
      setOps({});
    },

    setRunning() {
      clearTimer();
      done = new Set();
      activeId = null;
      renderNodes({});
      setBadge("running");
      setGate("Running simulation — waking API (cold start may take ~30s on free tier)…");
      clearLog();
      appendLog("▸ POST /api/simulate");
      setOps({ steps: "…", tokens: "…", util: "—", queues: "—", latency: "running" });
    },

    /** From POST /api/simulate response. */
    onSimResult(data) {
      if (!data || !Array.isArray(data.steps)) {
        this.reset();
        return;
      }
      replay(data);
    },

    onError(msg) {
      clearTimer();
      setBadge("idle");
      setGate("API error — " + (msg || "could not reach simulator") + ".");
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.GlassBox.reset());
  } else {
    window.GlassBox.reset();
  }
})();
