/* Mini Act-PRM animation ("at a glance"): a FIXED stage that replays the loop —
   the prompt/observation and the LOGGED action are always visible (spaced apart);
   each cycle samples 3 candidate thoughts between them, scores them with
   p(x | s, z), keeps the best, resolves the action, holds, then fades and replays
   with fresh candidates (a carousel of iterations, no scrolling). Scales itself
   down to fit narrow/mobile viewports. */
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

  const DESIGN_W = 340;           // fixed design width of the stage (one step)
  const DESIGN_H = 250;           // design height (matches the canvas CSS height)
  const CYCLE = 9500;             // ms per full replay (incl. holds + fade)
  const G = 3;
  const CAND_W = 96, CAND_H = 34;

  let W = 0, H = 0, dpr = 1, scale = 1, offX = 0;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth;
    scale = Math.min(1, W / DESIGN_W);
    // shrink the canvas itself on narrow screens so nothing is clipped
    canvas.style.height = Math.round(DESIGN_H * scale) + 'px';
    H = canvas.clientHeight;
    offX = (W / scale - DESIGN_W) / 2;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    if (reduceMotion) draw(0.9, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  function prand(i, j) {
    const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  const easeOut = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
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

  /* one full frame of the replay: p ∈ [0,1] is the cycle phase; iter varies the
     sampled candidates. Everything is drawn in DESIGN_W × DESIGN_H coordinates,
     scaled + centered to the actual canvas. */
  function draw(p, iter) {
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    ctx.clearRect(0, 0, W / scale, H / scale);
    ctx.translate(offX, 0);

    // fade in/out at the cycle boundaries (the carousel transition)
    let alpha = 1;
    if (p < 0.03) alpha = easeOut(p / 0.03);
    else if (p > 0.96) alpha = 1 - easeOut((p - 0.96) / 0.04);
    ctx.globalAlpha = alpha;

    // sub-phases: fan → type → score → pick → (hold) → resolve → (hold) → fade
    const pFan = easeOut(p / 0.10);
    const pType = clamp01((p - 0.07) / 0.20);
    const pScore = clamp01((p - 0.29) / 0.13);
    const pPick = clamp01((p - 0.46) / 0.08);     // winner chosen by ~0.54
    const pAct = clamp01((p - 0.70) / 0.10);      // ~1.5s hold, then action resolves

    const cy = DESIGN_H / 2;
    const sx = 30, candX = 118, ax = 292;
    const winner = Math.floor(prand(iter, 9) * G);

    // --- observation node ---
    ctx.lineWidth = 1.5;
    ctx.fillStyle = COL.stateFill;
    ctx.strokeStyle = COL.state;
    ctx.beginPath(); ctx.arc(sx, cy, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COL.ink;
    ctx.font = 'italic 12px Georgia, serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('o', sx, cy + 1);
    label('prompt', sx, cy + 32);

    // --- logged action node (always visible; dashed until resolved) ---
    const solid = pAct > 0.5;
    ctx.save();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = COL.action;
    ctx.fillStyle = solid ? COL.actionFill : 'rgba(27,175,122,0.05)';
    if (!solid) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(ax, cy, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#0e7a54';
    ctx.font = 'italic 12px Georgia, serif';
    ctx.fillText('x', ax, cy + 1);
    ctx.restore();
    label('logged action', ax, cy + 32);

    // --- candidate thoughts ---
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
        ctx.beginPath();
        const x1 = sx + 15, y1 = cy, x2 = candX - 6, y2 = ty + CAND_H / 2;
        const steps = 20;
        for (let i = 0; i <= steps * t; i++) {
          const u = i / steps;
          const bx = (1 - u) * (1 - u) * x1 + 2 * (1 - u) * u * ((x1 + x2) / 2) + u * u * x2;
          const by = (1 - u) * (1 - u) * y1 + 2 * (1 - u) * u * ((y1 + y2) / 2) + u * u * y2;
          if (i === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
        }
        ctx.stroke();
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

      // typing text lines (vary with the carousel iteration)
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

      // reward bar
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

    // scoring annotation
    if (pScore > 0.4) {
      ctx.save();
      ctx.globalAlpha = alpha * clamp01((pScore - 0.4) / 0.4);
      ctx.fillStyle = '#9a6900';
      ctx.font = '10.5px "Helvetica Neue", Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('score: p(x | s, z)', candX + CAND_W / 2, bandTop + band + CAND_H + 18);
      ctx.restore();
    }

    // winner connects to the logged action
    if (pAct > 0) {
      const wy = bandTop + spread * winner + CAND_H / 2;
      ctx.strokeStyle = COL.thought;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      const x1 = candX + CAND_W, y1 = wy, x2 = ax - 17, y2 = cy;
      const steps = 20, tt = easeOut(pAct);
      for (let i = 0; i <= steps * tt; i++) {
        const u = i / steps;
        const bx = (1 - u) * (1 - u) * x1 + 2 * (1 - u) * u * ((x1 + x2) / 2) + u * u * x2;
        const by = (1 - u) * (1 - u) * y1 + 2 * (1 - u) * u * ((y1 + y2) / 2) + u * u * y2;
        if (i === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }
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
    draw(0.9, 0);   // fully-resolved static frame with annotations
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
