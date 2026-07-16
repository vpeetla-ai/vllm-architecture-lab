/** Compact architecture rail for #gb-rail (glass-box left column). */
(function () {
  const cfg = window.ARCHITECT_CONFIG;
  if (!cfg) return;

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function normalize(data) {
    return {
      total_runs: data.total_runs ?? data.sample_size ?? data.total ?? 0,
      success_rate_pct: data.success_rate_pct ?? 100 - (data.failure_rate_pct || 0),
      p95_latency_ms: data.p95_latency_ms ?? data.p95_ms ?? null,
      active_entities: data.active_entities ?? 0,
    };
  }

  function renderMetrics(root, data) {
    root.innerHTML = "";
    const labels = cfg.metricLabels || {};
    const grid = el("div", "gb-metrics");
    [
      [labels.runs || "Runs", data.total_runs],
      ["Success", data.success_rate_pct + "%"],
      [labels.latency || "P95", data.p95_latency_ms != null ? data.p95_latency_ms + "ms" : "—"],
      [labels.entities || "Entities", data.active_entities],
    ].forEach(([label, value]) => {
      const card = el("div", "gb-metric");
      card.innerHTML = "<span>" + label + "</span><strong>" + value + "</strong>";
      grid.appendChild(card);
    });
    root.appendChild(grid);
  }

  function renderFailed(root, retry) {
    root.innerHTML = "";
    const wrap = el("div", "gb-metrics-failed");
    wrap.appendChild(el("p", "muted", "API waking (~30s)…"));
    const btn = el("button", "gb-retry", "Retry");
    btn.type = "button";
    btn.addEventListener("click", retry);
    wrap.appendChild(btn);
    root.appendChild(wrap);
  }

  function buildRail() {
    const root = el("div", "gb-rail-inner");

    root.appendChild(el("p", "gb-rail-lede", cfg.tagline || ""));

    root.appendChild(el("h2", "gb-rail-title", "Stack"));
    const stack = el("div", "gb-stack");
    (cfg.layers || []).forEach((layer) => {
      const row = el("div", "gb-stack-layer");
      row.appendChild(el("div", "gb-stack-tier", layer.tier));
      row.appendChild(el("div", "gb-stack-name", layer.name));
      row.appendChild(el("div", "gb-stack-role", layer.role));
      stack.appendChild(row);
    });
    root.appendChild(stack);

    root.appendChild(el("h2", "gb-rail-title", "Live metrics"));
    const slot = el("div", "gb-metrics-slot");
    slot.appendChild(el("p", "muted", "Loading…"));
    root.appendChild(slot);

    function load() {
      if (!cfg.metricsUrl) return;
      slot.innerHTML = "";
      slot.appendChild(el("p", "muted", "Loading…"));
      fetch(cfg.metricsUrl, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((data) => renderMetrics(slot, normalize(data)))
        .catch(() => renderFailed(slot, load));
    }
    load();
    window.VllmRefreshMetrics = load;

    root.appendChild(el("h2", "gb-rail-title", "Tradeoffs"));
    (cfg.tradeoffs || []).slice(0, 3).forEach((t) => {
      const card = el("div", "gb-tradeoff");
      card.innerHTML = "<strong>" + t.decision + "</strong><p>" + t.gain + "</p>";
      root.appendChild(card);
    });

    const links = [].concat(cfg.adrLinks || [], cfg.docsLinks || []).slice(0, 4);
    if (links.length) {
      root.appendChild(el("h2", "gb-rail-title", "ADRs & docs"));
      const ul = el("ul", "gb-adr-links");
      links.forEach((link) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = link.href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = link.title + " →";
        li.appendChild(a);
        ul.appendChild(li);
      });
      root.appendChild(ul);
    }
    return root;
  }

  function mount() {
    const rail = document.getElementById("gb-rail");
    if (!rail) return;
    rail.innerHTML = "";
    rail.appendChild(buildRail());
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
