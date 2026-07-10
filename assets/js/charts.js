/* Grouped bar charts (SVG, direct-labeled, tooltip on hover) + the small
   signal-strip diagrams in the Section-2 comparison cards. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const tooltip = document.getElementById('viz-tooltip');

  const SERIES_COLORS = {
    'Action-only': '#eda100',
    'Thoughts + Actions': '#1baf7a',
    'Action-only BC': '#eda100',
    'Frontier-trace SFT': '#1baf7a',
    'Act-PRM SFT (ours)': '#2a78d6',
  };

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function showTip(evt, html) {
    tooltip.innerHTML = html;
    tooltip.style.left = evt.clientX + 'px';
    tooltip.style.top = evt.clientY + 'px';
    tooltip.style.opacity = '1';
  }
  function hideTip() { tooltip.style.opacity = '0'; }

  /* ---------- grouped bar chart ---------- */
  function groupedBars(rootId, cfg) {
    const root = document.querySelector('#' + rootId + ' .chart-body');
    const legendEl = document.querySelector('#' + rootId + ' .legend');
    if (!root) return;

    const { groups, series, yMax } = cfg;
    const W = 720, H = 300;
    const m = { top: 14, right: 10, bottom: 40, left: 44 };
    const iw = W - m.left - m.right, ih = H - m.top - m.bottom;

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.aria || '' }, root);

    // y gridlines + ticks (clean 0.2 steps)
    const ticks = Math.max(2, Math.round(yMax / 0.2));
    for (let i = 0; i <= ticks; i++) {
      const v = (yMax / ticks) * i;
      const y = m.top + ih - (v / yMax) * ih;
      el('line', { x1: m.left, x2: m.left + iw, y1: y, y2: y, stroke: '#e1e0d9', 'stroke-width': 1 }, svg);
      const t = el('text', { x: m.left - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#898781' }, svg);
      t.textContent = v.toFixed(1);
    }

    const bandW = iw / groups.length;
    const barW = Math.min(24, (bandW * 0.72) / series.length - 2);
    const groupInnerW = series.length * barW + (series.length - 1) * 2;

    groups.forEach((g, gi) => {
      const bx0 = m.left + bandW * gi + (bandW - groupInnerW) / 2;
      // x label (may be 2 lines)
      const lines = g.label.split('\n');
      lines.forEach((ln, li) => {
        const t = el('text', {
          x: m.left + bandW * gi + bandW / 2,
          y: m.top + ih + 16 + li * 12,
          'text-anchor': 'middle', 'font-size': 11.5, fill: '#52514e',
        }, svg);
        t.textContent = ln;
      });

      series.forEach((s, si) => {
        const v = g.values[si];
        if (v == null) return;
        const h = (v / yMax) * ih;
        const x = bx0 + si * (barW + 2);
        const y = m.top + ih - h;
        const color = SERIES_COLORS[s] || '#2a78d6';
        const bar = el('path', {
          d: `M ${x} ${m.top + ih}
              L ${x} ${y + 4}
              Q ${x} ${y} ${x + 4} ${y}
              L ${x + barW - 4} ${y}
              Q ${x + barW} ${y} ${x + barW} ${y + 4}
              L ${x + barW} ${m.top + ih} Z`,
          fill: color,
        }, svg);
        bar.style.cursor = 'default';
        bar.addEventListener('mousemove', (e) =>
          showTip(e, `<strong>${s}</strong><br>${g.label.replace('\n', ' ')} · ${cfg.metric}: ${v.toFixed(3)}`));
        bar.addEventListener('mouseleave', hideTip);

        // direct label on the cap
        const t = el('text', {
          x: x + barW / 2, y: y - 5, 'text-anchor': 'middle',
          'font-size': 10.5, fill: '#52514e',
          'font-variant-numeric': 'tabular-nums',
        }, svg);
        t.textContent = ('' + v).replace(/^0\./, '.');
      });
    });

    // baseline
    el('line', { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih, stroke: '#c3c2b7', 'stroke-width': 1 }, svg);

    // legend
    series.forEach((s) => {
      const key = document.createElement('span');
      key.className = 'key';
      key.innerHTML = `<span class="swatch" style="background:${SERIES_COLORS[s]}"></span>${s}`;
      legendEl.appendChild(key);
    });
  }

  groupedBars('chart-motivating', {
    metric: 'action-token accuracy',
    aria: 'Bar chart: thoughts plus actions SFT beats action-only SFT on all three tasks',
    yMax: 0.8,
    series: ['Action-only', 'Thoughts + Actions'],
    groups: [
      { label: 'Treasure Hunter\n(TextWorld)', values: [0.563, 0.661] },
      { label: 'Coin Collector\n(TextWorld)', values: [0.531, 0.674] },
      { label: 'Snorkel Finance\nReasoning', values: [0.053, 0.144] },
    ],
  });

  groupedBars('chart-main', {
    metric: 'action-token accuracy',
    aria: 'Bar chart: Act-PRM SFT beats action-only BC everywhere and matches or exceeds frontier-trace SFT on several tasks',
    yMax: 1.0,
    series: ['Action-only BC', 'Frontier-trace SFT', 'Act-PRM SFT (ours)'],
    groups: [
      { label: 'τ²-Airline', values: [0.155, 0.255, 0.253] },
      { label: 'τ²-Retail', values: [0.249, 0.453, 0.376] },
      { label: 'Snorkel\nFinance', values: [0.053, 0.144, 0.299] },
      { label: 'TW Coin', values: [0.531, 0.674, 0.929] },
      { label: 'TW Treasure', values: [0.563, 0.661, 0.739] },
    ],
  });

  /* ---------- signal strips (s → z → x chains) ---------- */
  // kinds: rl (z+x generated, sparse reward at end), full (z+x supervised),
  // actiononly (z missing), actprm (z inferred + reward arrows)
  function strip(kind, root) {
    const W = 200, H = 44;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}` }, root);
    const y = 22;
    const xs = [18, 74, 130, 182];

    function node(x, type, dashed) {
      if (type === 's') {
        el('circle', { cx: x, cy: y, r: 9, fill: '#efeeea', stroke: '#898781', 'stroke-width': 1.3 }, svg);
      } else if (type === 'z') {
        const a = { cx: x, cy: y, r: 9, fill: '#fff', stroke: '#2a78d6', 'stroke-width': 1.4 };
        if (dashed) a['stroke-dasharray'] = '3 2.4';
        el('circle', a, svg);
      } else if (type === 'zmiss') {
        el('circle', { cx: x, cy: y, r: 9, fill: 'none', stroke: '#c3c2b7', 'stroke-width': 1.2, 'stroke-dasharray': '2 3' }, svg);
        const t = el('text', { x: x, y: y + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#c3c2b7' }, svg);
        t.textContent = '?';
      } else if (type === 'x') {
        el('circle', { cx: x, cy: y, r: 9, fill: 'rgba(27,175,122,0.18)', stroke: '#1baf7a', 'stroke-width': 1.4 }, svg);
      }
      const labels = { s: 's', z: 'z', x: 'x', zmiss: '' };
      if (labels[type]) {
        const t = el('text', { x: x, y: y + 3.6, 'text-anchor': 'middle', 'font-size': 10, 'font-style': 'italic',
          fill: type === 's' ? '#52514e' : type === 'z' ? '#2a78d6' : '#0e7a54' }, svg);
        t.textContent = labels[type];
      }
    }
    function edge(x1, x2) {
      el('line', { x1: x1 + 11, x2: x2 - 11, y1: y, y2: y, stroke: '#c3c2b7', 'stroke-width': 1.1 }, svg);
    }

    const zKind = kind === 'actiononly' ? 'zmiss' : 'z';
    node(xs[0], 's');
    edge(xs[0], xs[1]);
    node(xs[1], zKind, kind === 'actprm' || kind === 'rl');
    edge(xs[1], xs[2]);
    node(xs[2], 'x');
    edge(xs[2], xs[3]);
    const t = el('text', { x: xs[3], y: y + 4, 'text-anchor': 'middle', 'font-size': 12, fill: '#898781' }, svg);
    t.textContent = '…';

    if (kind === 'actprm') {
      // curved reward arrow from x back to z
      el('path', {
        d: `M ${xs[2]} ${y - 12} C ${xs[2] - 14} ${y - 26}, ${xs[1] + 14} ${y - 26}, ${xs[1] + 2} ${y - 13}`,
        fill: 'none', stroke: '#eda100', 'stroke-width': 1.4,
      }, svg);
      el('path', { d: `M ${xs[1] + 7} ${y - 17} L ${xs[1] + 2} ${y - 13} L ${xs[1] + 8.5} ${y - 11.5} Z`, fill: '#eda100' }, svg);
      const rt = el('text', { x: (xs[1] + xs[2]) / 2, y: y - 29, 'text-anchor': 'middle', 'font-size': 8.6, fill: '#9a6900' }, svg);
      rt.textContent = 'p(x | s, z)';
    }
    if (kind === 'rl') {
      const rt = el('text', { x: xs[3], y: y - 16, 'text-anchor': 'middle', 'font-size': 9, fill: '#9a6900' }, svg);
      rt.textContent = 'r = 1?';
    }
  }

  document.querySelectorAll('#signal-grid [data-strip]').forEach((elx) => {
    strip(elx.getAttribute('data-strip'), elx);
  });
})();
