/** Tabbed workbench for static demos — product first, architecture second.
 * Principal UX: section jump links + metrics Loading/Live/Failed+Retry.
 */
(function () {
  const cfg = window.ARCHITECT_CONFIG;
  if (!cfg) return;

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function renderLayers(root) {
    const stack = el("div", "arch-layers");
    (cfg.layers || []).forEach((layer) => {
      const row = el("div", "arch-layer");
      row.appendChild(el("span", "arch-tier", layer.tier));
      const mid = el("div", "arch-mid");
      mid.appendChild(el("strong", "ao-layer-name", layer.name));
      mid.appendChild(el("span", "muted", layer.role));
      row.appendChild(mid);
      const chips = el("div", "arch-chips");
      (layer.components || []).forEach((c) => chips.appendChild(el("span", "arch-chip", c)));
      row.appendChild(chips);
      stack.appendChild(row);
    });
    root.appendChild(stack);
  }

  function renderTradeoffs(root) {
    const grid = el("div", "arch-tradeoffs");
    (cfg.tradeoffs || []).forEach((t) => {
      const card = el("div", "arch-tradeoff");
      card.innerHTML =
        '<strong class="ao-trade-title">' +
        t.decision +
        '</strong><p><span class="gain">Gain</span> — ' +
        t.gain +
        "</p><p><span class=\"trade\">Trade</span> — " +
        t.trade +
        "</p>";
      grid.appendChild(card);
    });
    root.appendChild(grid);
  }

  function renderDocLinks(root) {
    const links = [].concat(cfg.adrLinks || [], cfg.docsLinks || []);
    if (!links.length) return;
    const wrap = el("div", "");
    wrap.id = "ao-adrs";
    wrap.appendChild(el("h2", "ao-title", "Architecture record"));
    const ul = el("ul", "arch-doc-links");
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
    wrap.appendChild(ul);
    root.appendChild(wrap);
  }

  function renderMetrics(root, data) {
    root.innerHTML = "";
    const labels = cfg.metricLabels || {};
    const grid = el("div", "arch-metrics");
    const cards = [
      [labels.runs || "Runs", data.total_runs],
      ["Success rate", data.success_rate_pct + "%"],
      [labels.latency || "P95", data.p95_latency_ms != null ? data.p95_latency_ms + "ms" : "—"],
      [labels.entities || "Entities", data.active_entities],
    ];
    cards.forEach(([label, value]) => {
      const card = el("div", "arch-metric");
      card.innerHTML = "<span>" + label + "</span><strong>" + value + "</strong>";
      grid.appendChild(card);
    });
    root.appendChild(grid);
    root.appendChild(
      el(
        "p",
        "muted api-hint",
        "Live from <code>" + (cfg.metricsPath || "/ops/metrics") + "</code>"
      )
    );
  }

  function renderMetricsFailed(root, retry) {
    root.innerHTML = "";
    const wrap = el("div", "ao-metrics-failed");
    wrap.appendChild(
      el("p", "muted", "Metrics unavailable — API may be waking from idle (~30s on free tier).")
    );
    const btn = el("button", "secondary", "Retry");
    btn.type = "button";
    btn.addEventListener("click", retry);
    wrap.appendChild(btn);
    root.appendChild(wrap);
  }

  function normalize(data) {
    return {
      total_runs: data.total_runs ?? data.sample_size ?? data.total ?? 0,
      success_rate_pct: data.success_rate_pct ?? 100 - (data.failure_rate_pct || 0),
      p95_latency_ms: data.p95_latency_ms ?? data.p95_node_latency_ms ?? data.p95_ms ?? null,
      active_entities: data.active_entities ?? data.invited_users ?? 0,
    };
  }

  function buildArchitecturePanel() {
    const panel = el("section", "panel architect-panel workbench-arch-panel");
    panel.hidden = true;

    const hasDocs = (cfg.adrLinks || []).length + (cfg.docsLinks || []).length > 0;
    const jump = el("nav", "ao-jump");
    jump.setAttribute("aria-label", "Architecture sections");
    [
      ["#ao-stack", "Stack"],
      ["#ao-tradeoffs", "Tradeoffs"],
      ...(hasDocs ? [["#ao-adrs", "ADRs"]] : []),
      ["#ao-metrics", "Metrics"],
    ].forEach(([href, label]) => {
      const a = document.createElement("a");
      a.href = href;
      a.textContent = label;
      jump.appendChild(a);
    });
    panel.appendChild(jump);

    const stack = el("div", "");
    stack.id = "ao-stack";
    stack.appendChild(el("p", "eyebrow", "Eagle-eye architecture"));
    stack.appendChild(el("h2", "ao-title", "How the system is wired"));
    stack.appendChild(el("p", "lede", cfg.tagline));
    renderLayers(stack);
    panel.appendChild(stack);

    const trade = el("div", "");
    trade.id = "ao-tradeoffs";
    trade.appendChild(el("h2", "ao-title", "Principal tradeoffs"));
    renderTradeoffs(trade);
    panel.appendChild(trade);

    renderDocLinks(panel);

    const metricsWrap = el("div", "");
    metricsWrap.id = "ao-metrics";
    metricsWrap.appendChild(el("h2", "ao-title", "Production metrics"));
    const metricsSlot = el("div", "arch-metrics-slot");
    metricsSlot.appendChild(el("p", "muted", "Loading live metrics…"));
    metricsWrap.appendChild(metricsSlot);
    panel.appendChild(metricsWrap);

    function loadMetrics() {
      if (!cfg.metricsUrl) {
        metricsSlot.innerHTML = "";
        metricsSlot.appendChild(el("p", "muted", "No metrics URL configured."));
        return;
      }
      metricsSlot.innerHTML = "";
      metricsSlot.appendChild(el("p", "muted", "Loading live metrics…"));
      fetch(cfg.metricsUrl, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((data) => renderMetrics(metricsSlot, normalize(data)))
        .catch(() => renderMetricsFailed(metricsSlot, loadMetrics));
    }
    loadMetrics();

    return panel;
  }

  function mount() {
    const productRoot = document.getElementById("workbench-product");
    const main =
      document.querySelector("main.shell, main.app-main, main") ||
      document.querySelector(".app-main, .shell") ||
      (productRoot && productRoot.parentElement) ||
      document.body;
    if (!main) return;

    const hero =
      main.querySelector(".page-hero") ||
      main.querySelector(".hdr") ||
      main.querySelector(".app-header");
    const tabs = el("nav", "workbench-tabs");
    tabs.setAttribute("role", "tablist");

    const btnProduct = el("button", "workbench-tab is-active", "");
    btnProduct.type = "button";
    btnProduct.innerHTML =
      '<span class="workbench-tab__label">Live product</span><span class="workbench-tab__hint">Run the demo</span>';

    const btnArch = el("button", "workbench-tab", "");
    btnArch.type = "button";
    btnArch.innerHTML =
      '<span class="workbench-tab__label">Architecture & metrics</span><span class="workbench-tab__hint">Stack, tradeoffs, SLOs</span>';

    tabs.appendChild(btnProduct);
    tabs.appendChild(btnArch);

    const archPanel = buildArchitecturePanel();

    if (productRoot) {
      if (hero && hero.nextSibling) main.insertBefore(tabs, hero.nextSibling);
      else main.insertBefore(tabs, main.firstChild);
      main.appendChild(archPanel);

      function show(tab) {
        const isProduct = tab === "product";
        productRoot.hidden = !isProduct;
        archPanel.hidden = isProduct;
        btnProduct.classList.toggle("is-active", isProduct);
        btnArch.classList.toggle("is-active", !isProduct);
      }
      btnProduct.addEventListener("click", () => show("product"));
      btnArch.addEventListener("click", () => show("architecture"));
      show("product");
      return;
    }

    const legacyRoot = document.getElementById("architect-root");
    if (legacyRoot) legacyRoot.remove();
    if (hero) hero.parentNode.insertBefore(tabs, hero.nextSibling);
    main.appendChild(archPanel);

    const panels = Array.from(main.querySelectorAll("section.panel, .live-panel, .command-center, #view-loop"));
    const productWrap = el("div", "workbench-product-legacy");
    panels.forEach((p) => {
      if (p !== archPanel && !p.closest(".workbench-arch-panel")) productWrap.appendChild(p);
    });
    tabs.after(productWrap);

    function show(tab) {
      const isProduct = tab === "product";
      productWrap.hidden = !isProduct;
      archPanel.hidden = isProduct;
      btnProduct.classList.toggle("is-active", isProduct);
      btnArch.classList.toggle("is-active", !isProduct);
    }
    btnProduct.addEventListener("click", () => show("product"));
    btnArch.addEventListener("click", () => show("architecture"));
    show("product");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
