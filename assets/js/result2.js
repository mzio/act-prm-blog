/* Result 2 interactive: the reward curve of the FIRST step of a chosen task
   (10 samples: 2 held-out + 8 train), hover to preview a training iteration,
   CLICK to inspect — the panels below show the prompt, the ground-truth action,
   and the thought the model generated at that iteration.
   Data: assets/js/lenpen-curve-data.js */
(function () {
  const root = document.getElementById('result2-viz');
  const data = window.LENPEN_CURVE;
  if (!root || !data) return;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 720, H = 250, m = { top: 16, right: 14, bottom: 34, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const maxIter = Math.max(...data.eval_mean.map((d) => d.i));

  let taskIdx = 0;
  let curIter = data.final_iter;      // committed (clicked) iteration
  let hoverIter = null;               // previewed (hovered) iteration

  root.innerHTML = `
    <div class="chart-title">Watching a thought evolve over training</div>
    <div class="chart-sub">Reward p(x | s, ẑ) of the selected first-step thought for 10
      <strong>held-out</strong> tasks, re-sampled from every saved checkpoint of the λ = 0.15 run.
      <strong>Hover to scrub, click a point to inspect it.</strong> Gray line: mean held-out reward
      across <em>all</em> steps during training — it starts much higher because later steps are far
      easier than step 1 (their context already contains prior tool calls).</div>
    <div class="r2-toggles">
      <span class="r2-toggle-label">held-out task</span><span class="r2-eval"></span>
    </div>
    <div class="r2-chart"></div>
    <div class="r2-stats">
      <div class="r2-stat"><div class="lbl">Inspecting iteration</div><div class="val r2-s-iter"></div></div>
      <div class="r2-stat"><div class="lbl">p(x | s, ẑ)</div><div class="val r2-s-p"></div></div>
      <div class="r2-stat"><div class="lbl">Thought length</div><div class="val r2-s-tok"></div></div>
    </div>
    <div class="r2-grid">
      <div class="r2-col">
        <div class="r2-msg r2-obs"><div class="who">prompt</div><span class="body"></span></div>
        <div class="r2-msg r2-action"><div class="who">ground-truth action (from the log)</div><pre class="body"></pre></div>
      </div>
      <div class="r2-col">
        <div class="r2-msg r2-thought"><div class="who"></div><span class="body"></span></div>
      </div>
    </div>`;

  const chartEl = root.querySelector('.r2-chart');

  function pill(label, active, onclick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'r2-pill' + (active ? ' active' : '');
    b.textContent = label;
    b.addEventListener('click', onclick);
    return b;
  }

  function renderToggles() {
    const evalEl = root.querySelector('.r2-eval');
    evalEl.innerHTML = '';
    data.tasks.forEach((t, i) => {
      evalEl.appendChild(pill(String(i + 1), i === taskIdx, () => {
        taskIdx = i;
        const series = data.tasks[i].per_iter;
        if (!series.some((d) => d.i === curIter)) curIter = series[series.length - 1].i;
        render();
      }));
    });
  }

  const x = (i) => m.left + (i / maxIter) * iw;
  const y = (p) => m.top + ih - p * ih;

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function renderChart() {
    chartEl.innerHTML = '';
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': 'Reward of the selected thought across training iterations; click a point to inspect it' });
    chartEl.appendChild(svg);

    for (let v = 0; v <= 1.0001; v += 0.25) {
      el('line', { x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v),
                   stroke: '#e1e0d9', 'stroke-width': 1 }, svg);
      const t = el('text', { x: m.left - 8, y: y(v) + 4, 'text-anchor': 'end',
                             'font-size': 11, fill: '#898781' }, svg);
      t.textContent = v.toFixed(2);
    }
    for (let i = 0; i <= maxIter; i += 10) {
      const t = el('text', { x: x(i), y: m.top + ih + 20, 'text-anchor': 'middle',
                             'font-size': 11, fill: '#898781' }, svg);
      t.textContent = i;
    }
    const xl = el('text', { x: m.left + iw / 2, y: H - 2, 'text-anchor': 'middle',
                            'font-size': 11, fill: '#52514e' }, svg);
    xl.textContent = 'EM iteration';

    el('path', {
      d: data.eval_mean.map((d, k) => `${k ? 'L' : 'M'} ${x(d.i)} ${y(d.p)}`).join(' '),
      fill: 'none', stroke: '#c3c2b7', 'stroke-width': 1.4,
    }, svg);

    const series = data.tasks[taskIdx].per_iter;
    el('path', {
      d: series.map((d, k) => `${k ? 'L' : 'M'} ${x(d.i)} ${y(d.p)}`).join(' '),
      fill: 'none', stroke: '#2a78d6', 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }, svg);
    series.forEach((d) => {
      el('circle', { cx: x(d.i), cy: y(d.p), r: 4, fill: '#2a78d6',
                     stroke: '#fff', 'stroke-width': 1.5, 'data-iter': d.i }, svg);
    });

    // hover preview: crosshair + floating hint
    const hoverLine = el('line', { x1: 0, x2: 0, y1: m.top, y2: m.top + ih,
                                   stroke: 'rgba(137,135,129,0.35)', 'stroke-width': 1,
                                   'stroke-dasharray': '3 3', opacity: 0 }, svg);
    const hint = el('text', { x: 0, y: m.top + 10, 'text-anchor': 'middle',
                              'font-size': 10, fill: '#898781', opacity: 0 }, svg);
    // committed marker: ring + solid crosshair
    const selLine = el('line', { x1: x(curIter), x2: x(curIter), y1: m.top, y2: m.top + ih,
                                 stroke: 'rgba(42,120,214,0.3)', 'stroke-width': 1 }, svg);
    const selDot = el('circle', { cx: 0, cy: 0, r: 5.5, fill: '#2a78d6',
                                  stroke: '#fff', 'stroke-width': 2 }, svg);
    const selRing = el('circle', { cx: 0, cy: 0, r: 9, fill: 'none',
                                   stroke: 'rgba(42,120,214,0.4)', 'stroke-width': 1.5 }, svg);

    function place(iter) {
      const d = series.find((q) => q.i === iter) || series[series.length - 1];
      selLine.setAttribute('x1', x(d.i)); selLine.setAttribute('x2', x(d.i));
      selDot.setAttribute('cx', x(d.i)); selDot.setAttribute('cy', y(d.p));
      selRing.setAttribute('cx', x(d.i)); selRing.setAttribute('cy', y(d.p));
    }
    place(curIter);

    function nearest(clientX) {
      const r = svg.getBoundingClientRect();
      const px = ((clientX - r.left) / r.width) * W;
      let bestD = Infinity, best = series[0];
      series.forEach((d) => {
        const dd = Math.abs(x(d.i) - px);
        if (dd < bestD) { bestD = dd; best = d; }
      });
      return best;
    }

    svg.addEventListener('mousemove', (e) => {
      const d = nearest(e.clientX);
      hoverIter = d.i;
      hoverLine.setAttribute('x1', x(d.i)); hoverLine.setAttribute('x2', x(d.i));
      hoverLine.setAttribute('opacity', 1);
      hint.setAttribute('x', Math.min(Math.max(x(d.i), m.left + 56), m.left + iw - 56));
      hint.textContent = `iter ${d.i} · p=${d.p.toFixed(2)} — click to inspect`;
      hint.setAttribute('opacity', 1);
    });
    svg.addEventListener('mouseleave', () => {
      hoverLine.setAttribute('opacity', 0);
      hint.setAttribute('opacity', 0);
    });
    svg.addEventListener('click', (e) => {
      const d = nearest(e.clientX);
      curIter = d.i;
      place(curIter);
      renderPanel();
    });
    svg.style.cursor = 'pointer';
  }

  function renderPanel() {
    const task = data.tasks[taskIdx];
    const pt = task.per_iter.find((d) => d.i === curIter) || task.per_iter[task.per_iter.length - 1];
    root.querySelector('.r2-s-iter').textContent = pt.i;
    root.querySelector('.r2-s-p').textContent = pt.p.toFixed(3);
    root.querySelector('.r2-s-tok').textContent = `${pt.tok} tok`;
    root.querySelector('.r2-obs .who').textContent =
      `prompt — held-out task ${taskIdx + 1}`;
    root.querySelector('.r2-obs .body').textContent = task.question;
    root.querySelector('.r2-action .body').textContent = task.action;
    root.querySelector('.r2-thought .who').textContent =
      `generated thought ẑ @ iteration ${pt.i}`;
    root.querySelector('.r2-thought .body').textContent = pt.z;
  }

  function render() {
    renderToggles();
    renderChart();
    renderPanel();
  }
  render();
})();
