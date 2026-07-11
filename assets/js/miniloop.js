/* Mini Act-PRM animation ("at a glance"), two-step story with a gentle pan:
   (1) the prompt node, the LOGGED action node, and the next observation node
       are laid out from the start (the chain is given by the log);
   (2) candidate thoughts are sampled between o and x, typed, scored with
       p(x | s, z), and the best is selected;
   (3) the action resolves and the stage pans one step to the next
       observation / logged-action pair (already in place);
   (4) the sampling replays there — then the whole thing fades and restarts
       with fresh candidates. Scales itself to fit narrow/mobile viewports. */
(function () {
  const canvas = document.getElementById('miniloop-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const COL = {
    state: '#898781', stateFill: '#efeeea',
    thought: '#2a78d6', thoughtSoft: 'rgba(42,120,214,0.35)',
    action: '#1baf7a', actionFill: 'rgba(27,175,122,0.16)',
    reward: '#eda100', ink: '#52514e', ghost: 'rgba(137,135,129,0.4)',
    label: '#898781',
  };
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const DESIGN_W = 400;           // visible stage width (design coordinates)
  const DESIGN_H = 250;
  const SEG = 340;                // horizontal distance between chain steps
  const O_X = 28, CAND_X = 100, X_X = 268;   // node anchors within a segment
  const CAND_W = 96, CAND_H = 34;
  const G = 3;
  const CYCLE = 16000;            // ms for the full two-step story + fade

  let W = 0, H = 0, dpr = 1, scale = 1, offX = 0;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth;
    scale = Math.min(1, W / DESIGN_W);
    canvas.style.height = Math.round(DESIGN_H * scale) + 'px';
    H = canvas.clientHeight;
    offX = (W / scale - DESIGN_W) / 2;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    if (reduceMotion) draw(0.42, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  function prand(i, j) {
    const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  const easeOut = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
  const easeInOut = (t) => {
    t = Math.min(Math.max(t, 0), 1);
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  };
  const clamp01 = (t) => Math.min(Math.max(t, 0), 1);

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function label(text, x, y) {
    ctx.fillStyle = COL.label;
    ctx.font = '10.5px "Helvetica Neue", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
  }

  function obsNode(x, cy, text) {
    ctx.lineWidth = 1.5;
    ctx.fillStyle = COL.stateFill;
    ctx.strokeStyle = COL.state;
    ctx.beginPath(); ctx.arc(x, cy, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COL.ink;
    ctx.font = 'italic 12px Georgia, serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('o', x, cy + 1);
    if (text) label(text, x, cy + 32);
  }

  function actionNode(x, cy, solid, text) {
    ctx.save();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = COL.action;
    ctx.fillStyle = solid ? COL.actionFill : 'rgba(27,175,122,0.05)';
    if (!solid) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(x, cy, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#0e7a54';
    ctx.font = 'italic 12px Georgia, serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('x', x, cy + 1);
    ctx.restore();
    if (text) label(text, x, cy + 32);
  }

  function bezierPartial(x1, y1, x2, y2, t) {
    ctx.beginPath();
    const steps = 20;
    for (let i = 0; i <= steps * t; i++) {
      const u = i / steps;
      const bx = (1 - u) * (1 - u) * x1 + 2 * (1 - u) * u * ((x1 + x2) / 2) + u * u * x2;
      const by = (1 - u) * (1 - u) * y1 + 2 * (1 - u) * u * ((y1 + y2) / 2) + u * u * y2;
      if (i === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  /* the sampling animation for one segment; q ∈ [0,1] local phase, iter varies
     candidates; alpha = global fade; labeled = show the score annotation */
  function drawSampling(baseX, q, iter, alpha, labeled) {
    const cy = DESIGN_H / 2 - 8;
    const sx = baseX + O_X, candX = baseX + CAND_X, ax = baseX + X_X;
    const winner = Math.floor(prand(iter, 9) * G);

    const pFan = easeOut(q / 0.14);
    const pType = clamp01((q - 0.10) / 0.26);
    const pScore = clamp01((q - 0.40) / 0.16);
    const pPick = clamp01((q - 0.62) / 0.10);   // pick by ~0.72, hold to 0.85
    const pAct = clamp01((q - 0.85) / 0.12);

    const band = Math.min(DESIGN_H - 58, 152);
    const spread = band / (G - 1);
    const bandTop = cy - band / 2 - CAND_H / 2 + 6;

    for (let g = 0; g < G; g++) {
      const ty = bandTop + spread * g;
      const isWin = g === winner;
      const fade = pPick > 0 && !isWin ? 1 - 0.75 * pPick : 1;

      const t = clamp01(pFan * (1 - 0.08 * g));
      if (t > 0.01) {
        ctx.save();
        ctx.globalAlpha = alpha * fade * 0.9;
        ctx.strokeStyle = isWin && pPick > 0 ? COL.thought : COL.ghost;
        ctx.lineWidth = isWin && pPick > 0 ? 1.6 : 1;
        bezierPartial(sx + 15, cy, candX - 6, ty + CAND_H / 2, t);
        ctx.restore();
      }
      if (pType <= 0) continue;

      ctx.save();
      ctx.globalAlpha = alpha * fade;
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = isWin && pPick > 0 ? COL.thought : COL.thoughtSoft;
      ctx.lineWidth = isWin && pPick > 0 ? 1.6 : 1.1;
      if (!(isWin && pPick > 0)) ctx.setLineDash([3.5, 2.5]);
      roundRect(candX, ty, CAND_W, CAND_H, 8);
      ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);

      if (isWin && pPick > 0) {
        ctx.save();
        ctx.globalAlpha = alpha * 0.3 * pPick * fade;
        ctx.strokeStyle = COL.thought;
        ctx.lineWidth = 4;
        roundRect(candX - 2, ty - 2, CAND_W + 4, CAND_H + 4, 10);
        ctx.stroke();
        ctx.restore();
      }

      for (let l = 0; l < 2; l++) {
        const full = CAND_W - 34 - prand(iter, g * 7 + l) * 22;
        const lineP = clamp01(pType * 2.6 - l);
        if (lineP <= 0) continue;
        ctx.strokeStyle = isWin && pPick > 0 ? 'rgba(42,120,214,0.55)' : 'rgba(137,135,129,0.45)';
        ctx.lineWidth = 2.4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(candX + 22, ty + 12 + l * 9.5);
        ctx.lineTo(candX + 22 + full * lineP, ty + 12 + l * 9.5);
        ctx.stroke();
      }
      ctx.fillStyle = isWin && pPick > 0 ? COL.thought : 'rgba(42,120,214,0.55)';
      ctx.font = 'italic 11px Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('z', candX + 8, ty + 17);

      if (pScore > 0) {
        const rw = 0.22 + 0.75 * prand(iter, g * 3 + 1);
        const rewardW = (CAND_W - 16) * (isWin ? Math.max(rw, 0.85) : Math.min(rw, 0.55));
        ctx.strokeStyle = 'rgba(0,0,0,0.07)';
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(candX + 8, ty + CAND_H + 6);
        ctx.lineTo(candX + CAND_W - 8, ty + CAND_H + 6);
        ctx.stroke();
        ctx.strokeStyle = isWin && pPick > 0.3 ? COL.thought : COL.reward;
        ctx.beginPath();
        ctx.moveTo(candX + 8, ty + CAND_H + 6);
        ctx.lineTo(candX + 8 + rewardW * easeOut(pScore), ty + CAND_H + 6);
        ctx.stroke();
      }
      ctx.restore();
    }

    if (labeled && pScore > 0.4) {
      ctx.save();
      ctx.globalAlpha = alpha * clamp01((pScore - 0.4) / 0.4);
      ctx.fillStyle = '#9a6900';
      ctx.font = '10.5px "Helvetica Neue", Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('score: p(x | s, z)', candX + CAND_W / 2, bandTop + band + CAND_H + 15);
      ctx.restore();
    }

    // winner connects to the logged action
    if (pAct > 0) {
      const wy = bandTop + spread * winner + CAND_H / 2;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = COL.thought;
      ctx.lineWidth = 1.6;
      bezierPartial(candX + CAND_W, wy, ax - 17, cy, easeOut(pAct));
      ctx.restore();
    }
    return pAct;
  }

  /* one full frame: p ∈ [0,1] global phase, iter varies both segments */
  function draw(p, iter) {
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    ctx.clearRect(0, 0, W / scale, H / scale);

    let alpha = 1;
    if (p < 0.02) alpha = easeOut(p / 0.02);
    else if (p > 0.97) alpha = 1 - easeOut((p - 0.97) / 0.03);

    // story timeline: segment 0 → pan → segment 1 → fade
    const q0 = clamp01(p / 0.44);
    const pPan = easeInOut((p - 0.45) / 0.08);
    const q1 = clamp01((p - 0.54) / 0.42);
    const cam = SEG * pPan;

    ctx.translate(offX - cam, 0);
    ctx.globalAlpha = alpha;

    const cy = DESIGN_H / 2 - 8;

    // the chain is given from the start: o1 … x1 → o2 (and o2 … x2 → o3 after the pan)
    const seg0Solid = q0 >= 0.91;
    const seg1Solid = q1 >= 0.91;

    // connectors x_k → o_{k+1}, drawn once each segment's action resolves
    ctx.strokeStyle = COL.ghost;
    ctx.lineWidth = 1;
    if (seg0Solid) {
      ctx.beginPath(); ctx.moveTo(X_X + 15, cy); ctx.lineTo(SEG + O_X - 15, cy); ctx.stroke();
    }
    if (seg1Solid) {
      ctx.beginPath(); ctx.moveTo(SEG + X_X + 15, cy); ctx.lineTo(2 * SEG + O_X - 15, cy); ctx.stroke();
    }

    // segment 0 nodes + sampling
    obsNode(O_X, cy, 'prompt');
    actionNode(X_X, cy, seg0Solid, 'logged action');
    drawSampling(0, q0, iter * 2, alpha, pPan < 0.5);

    // segment 1 nodes (visible from the start — "already there") + sampling after pan
    obsNode(SEG + O_X, cy, 'next obs');
    actionNode(SEG + X_X, cy, seg1Solid, pPan > 0.5 ? 'logged action' : null);
    if (q1 > 0) drawSampling(SEG, q1, iter * 2 + 1, alpha, pPan >= 0.5);

    // the observation after segment 1 (chain continues …)
    obsNode(2 * SEG + O_X, cy, null);

    ctx.globalAlpha = 1;
  }

  let running = false, rafId = null;
  function frame(now) {
    if (!running) return;
    const iter = Math.floor(now / CYCLE);
    const p = (now % CYCLE) / CYCLE;
    draw(p, iter % 1000);
    rafId = requestAnimationFrame(frame);
  }

  if (reduceMotion) {
    draw(0.42, 0);   // segment 0 resolved, pre-pan, annotations visible
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting && !running) {
        running = true;
        rafId = requestAnimationFrame(frame);
      } else if (!e.isIntersecting && running) {
        running = false;
        if (rafId) cancelAnimationFrame(rafId);
      }
    });
  }, { threshold: 0.15 });
  io.observe(canvas);
})();
