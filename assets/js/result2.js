/* Result 2 interactive: the eval reward curve for a chosen held-out task + step,
   with hover showing the observation, ground-truth action, and the thought the
   model generated at that training iteration. Data: assets/js/lenpen-curve-data.js */
(function () {
  const root = document.getElementById('result2-viz');
  const data = window.LENPEN_CURVE;
  if (!root || !data) return;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 720, H = 250, m = { top: 16, right: 14, bottom: 34, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const maxIter = Math.max(...data.eval_mean.map((d) => d.i));

  let taskIdx = 0, stepIdx = 0, curIter = data.final_iter;

  // ---------- scaffold ----------
  root.innerHTML = `
    <div class="chart-title">Watching a thought evolve over training</div>
    <div class="chart-sub">Held-out reward p(x | s, ẑ) of the selected thought at every EM iteration
      (λ = 0.15 run) — <strong>hover the curve</strong> to see what the model was thinking at that
      point in training. Gray line: mean over all eval steps.</div>
    <div class="r2-toggles">
      <span class="r2-toggle-label">task</span><span class="r2-tasks"></span>
      <span class="r2-toggle-label" style="margin-left:0.9rem">step</span><span class="r2-steps"></span>
    </div>
    <div class="r2-chart"></div>
    <div class="r2-panel">
      <div class="r2-msg r2-obs"><div class="who"></div><span class="body"></span></div>
      <div class="r2-msg r2-action"><div class="who">ground-truth action (from the log)</div><pre class="body"></pre></div>
      <div class="r2-msg r2-thought"><div class="who"></div><span class="body"></span></div>
    </div>`;

  const chartEl = root.querySelector('.r2-chart');
  const tasksEl = root.querySelector('.r2-tasks');
  const stepsEl = root.querySelector('.r2-steps');

  function pill(label, active, onclick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'r2-pill' + (active ? ' active' : '');
    b.textContent = label;
    b.addEventListener('click', onclick);
    return b;
  }

  function renderToggles() {
    tasksEl.innerHTML = '';
    stepsEl.innerHTML = '';
    data.tasks.forEach((t, i) =>
      tasksEl.appendChild(pill(String(i + 1), i === taskIdx, () => {
        taskIdx = i;
        stepIdx = Math.min(stepIdx, data.tasks[i].steps.length - 1);
        render();
      })));
    data.tasks[taskIdx].steps.forEach((s, i) =>
      stepsEl.appendChild(pill(String(i + 1), i === stepIdx, () => { stepIdx = i; render(); })));
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
      'aria-label': 'Reward of the selected thought across training iterations' });
    chartEl.appendChild(svg);

    // grid + axis ticks
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

    // context: mean over all eval steps
    el('path', {
      d: data.eval_mean.map((d, k) => `${k ? 'L' : 'M'} ${x(d.i)} ${y(d.p)}`).join(' '),
      fill: 'none', stroke: '#c3c2b7', 'stroke-width': 1.4,
    }, svg);

    // the selected (task, step) series
    const series = data.tasks[taskIdx].steps[stepIdx].per_iter;
    el('path', {
      d: series.map((d, k) => `${k ? 'L' : 'M'} ${x(d.i)} ${y(d.p)}`).join(' '),
      fill: 'none', stroke: '#2a78d6', 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }, svg);
    series.forEach((d) => {
      el('circle', { cx: x(d.i), cy: y(d.p), r: d.i === curIter ? 5.5 : 2.6,
                     fill: '#2a78d6', stroke: '#fff',
                     'stroke-width': d.i === curIter ? 2 : 1,
                     'data-iter': d.i }, svg);
    });

    // crosshair
    const cross = el('line', { x1: x(curIter), x2: x(curIter), y1: m.top, y2: m.top + ih,
                               stroke: 'rgba(42,120,214,0.25)', 'stroke-width': 1 }, svg);

    const dots = Array.from(svg.querySelectorAll('circle[data-iter]'));
    function setHover(iter) {
      curIter = iter;
      cross.setAttribute('x1', x(iter));
      cross.setAttribute('x2', x(iter));
      dots.forEach((c) => {
        const active = Number(c.getAttribute('data-iter')) === iter;
        c.setAttribute('r', active ? 5.5 : 2.6);
        c.setAttribute('stroke-width', active ? 2 : 1);
      });
      renderPanel();
    }

    // hover (and touch-drag): snap to nearest recorded iteration
    function snap(clientX) {
      const r = svg.getBoundingClientRect();
      const px = ((clientX - r.left) / r.width) * W;
      let bestD = Infinity, bestI = curIter;
      series.forEach((d) => {
        const dd = Math.abs(x(d.i) - px);
        if (dd < bestD) { bestD = dd; bestI = d.i; }
      });
      if (bestI !== curIter) setHover(bestI);
    }
    svg.addEventListener('mousemove', (e) => snap(e.clientX));
    svg.addEventListener('touchmove', (e) => {
      if (e.touches.length) { snap(e.touches[0].clientX); e.preventDefault(); }
    }, { passive: false });
    svg.style.cursor = 'crosshair';
  }

  function renderPanel() {
    const step = data.tasks[taskIdx].steps[stepIdx];
    const pt = step.per_iter.find((d) => d.i === curIter) || step.per_iter[step.per_iter.length - 1];
    root.querySelector('.r2-obs .who').textContent =
      `observation (${step.obs_kind}) — step ${stepIdx + 1}, ${data.tasks[taskIdx].label.toLowerCase()}`;
    root.querySelector('.r2-obs .body').textContent = step.obs;
    root.querySelector('.r2-action .body').textContent = step.action;
    root.querySelector('.r2-thought .who').textContent =
      `generated thought ẑ @ iteration ${pt.i} — p(x|s,ẑ) = ${pt.p.toFixed(3)}, ${pt.tok} tokens`;
    root.querySelector('.r2-thought .body').textContent = pt.z;
  }

  function render() {
    renderToggles();
    renderChart();
    renderPanel();
  }
  render();
})();
