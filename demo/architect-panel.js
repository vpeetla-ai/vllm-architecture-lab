/** Tabbed workbench for static demos — product first, architecture second. */
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
    root.appendChild(el("h2", "ao-title", "Architecture record"));
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
    root.appendChild(ul);
  }

  function renderMetrics(root, data) {
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
    panel.appendChild(el("p", "eyebrow", "Eagle-eye architecture"));
    panel.appendChild(el("h2", "ao-title", "How the system is wired"));
    panel.appendChild(el("p", "lede", cfg.tagline));
    renderLayers(panel);
    panel.appendChild(el("h2", "ao-title", "Principal tradeoffs"));
    renderTradeoffs(panel);
    renderDocLinks(panel);
    panel.appendChild(el("h2", "ao-title", "Production metrics"));
    const metricsSlot = el("div", "arch-metrics-slot");
    panel.appendChild(metricsSlot);
    if (cfg.metricsUrl) {
      fetch(cfg.metricsUrl, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => data && renderMetrics(metricsSlot, normalize(data)))
        .catch(() => null);
    }
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
    btnProduct.innerHTML = '<span class="workbench-tab__label">Live product</span><span class="workbench-tab__hint">Run the demo</span>';

    const btnArch = el("button", "workbench-tab", "");
    btnArch.type = "button";
    btnArch.innerHTML =
      '<span class="workbench-tab__label">Architecture & metrics</span><span class="workbench-tab__hint">Stack, tradeoffs, SLOs</span>';

    tabs.appendChild(btnProduct);
    tabs.appendChild(btnArch);

    const archPanel = buildArchitecturePanel();

    if (productRoot) {
      const mountPoint = hero ? hero.nextSibling : main.firstChild;
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

    // Legacy: wrap panels after architect-root
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
