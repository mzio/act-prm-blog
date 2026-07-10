/* Mini looping Act-PRM animation ("at a glance"): for each step, the prompt/state
   and the LOGGED action are shown first (spaced apart), then 3 candidate thoughts
   are sampled between them, scored by p(x | s, z), and the best is kept — then the
   chain advances to the next logged action. Companion to the hero background, but
   inline, compact, and with the ground-truth action visible before sampling. */
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

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', () => { resize(); if (reduceMotion) drawStatic(); });
  resize();

  const G = 3;                    // candidate thoughts per step
  const STEP_W = 360;             // horizontal span per timestep
  const PERIOD = 7200;            // ms per timestep (incl. ~1.7s hold after the pick)
  const CAND_W = 96, CAND_H = 34;

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

  /* one timestep anchored at x0; p ∈ [0,1] is its phase; labeled = draw captions */
  function drawStep(stepIdx, x0, p, labeled) {
    const cy = H / 2;
    const sx = x0, candX = x0 + 96, ax = x0 + 262;
    const winner = Math.floor(prand(stepIdx, 9) * G);

    const pFan = easeOut(p / 0.12);
    const pType = clamp01((p - 0.08) / 0.22);
    const pScore = clamp01((p - 0.32) / 0.14);
    const pPick = clamp01((p - 0.48) / 0.09);
    // hold: winner stays highlighted from ~0.57 to 0.80 before the action resolves
    const pAct = clamp01((p - 0.80) / 0.12);

    // --- state node (visible immediately) ---
    ctx.lineWidth = 1.5;
    ctx.fillStyle = COL.stateFill;
    ctx.strokeStyle = COL.state;
    ctx.beginPath(); ctx.arc(sx, cy, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COL.ink;
    ctx.font = 'italic 12px Georgia, serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('o', sx, cy + 1);
    if (labeled) label(stepIdx === 0 ? 'prompt' : 'observation', sx, cy + 32);

    // --- LOGGED action node: dashed outline from the start (it's given!) ---
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
    if (labeled) label('logged action', ax, cy + 32);

    // --- candidate thoughts fan out between s and x ---
    const band = Math.min(H - 58, 150);
    const spread = band / (G - 1);
    const bandTop = cy - band / 2 - CAND_H / 2 + 6;
    for (let g = 0; g < G; g++) {
      const ty = bandTop + spread * g;
      const isWin = g === winner;
      const fade = pPick > 0 && !isWin ? 1 - 0.75 * pPick : 1;

      const t = clamp01(pFan * (1 - 0.08 * g));
      if (t > 0.01) {
        ctx.save();
        ctx.globalAlpha = fade * 0.9;
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
      ctx.globalAlpha = fade;
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = isWin && pPick > 0 ? COL.thought : COL.thoughtSoft;
      ctx.lineWidth = isWin && pPick > 0 ? 1.6 : 1.1;
      if (!(isWin && pPick > 0)) ctx.setLineDash([3.5, 2.5]);
      roundRect(candX, ty, CAND_W, CAND_H, 8);
      ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);

      if (isWin && pPick > 0) {
        ctx.save();
        ctx.globalAlpha = 0.3 * pPick * fade;
        ctx.strokeStyle = COL.thought;
        ctx.lineWidth = 4;
        roundRect(candX - 2, ty - 2, CAND_W + 4, CAND_H + 4, 10);
        ctx.stroke();
        ctx.restore();
      }

      // typing text lines
      for (let l = 0; l < 2; l++) {
        const full = CAND_W - 34 - prand(stepIdx, g * 7 + l) * 22;
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

      // reward bar: p(x | s, z)
      if (pScore > 0) {
        const rw = 0.22 + 0.75 * prand(stepIdx, g * 3 + 1);
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
      ctx.globalAlpha = clamp01((pScore - 0.4) / 0.4);
      ctx.fillStyle = '#9a6900';
      ctx.font = '10.5px "Helvetica Neue", Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('score: p(x | s, z)', candX + CAND_W / 2, bandTop + band + CAND_H + 18);
      ctx.restore();
    }

    // --- winner connects to the logged action ---
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
      // connector to the next state
      if (pAct > 0.8) {
        ctx.strokeStyle = COL.ghost;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax + 15, cy);
        ctx.lineTo(ax + 15 + (STEP_W - 292) * easeOut((pAct - 0.8) / 0.2), cy);
        ctx.stroke();
      }
    }
  }

  let running = false, rafId = null;
  function frame(now) {
    if (!running) return;
    ctx.clearRect(0, 0, W, H);
    const tGlobal = now / PERIOD;
    const camX = Math.max(0, (tGlobal - 1)) * STEP_W;   // hold still for step 0
    const first = Math.floor(camX / STEP_W) - 1;
    const count = Math.ceil(W / STEP_W) + 2;
    for (let k = 0; k < count; k++) {
      const stepIdx = Math.max(0, first + k);
      const x0 = stepIdx * STEP_W - camX + 28;
      const p = clamp01(tGlobal - stepIdx + 0.16);
      if (x0 > -STEP_W && x0 < W + 40) {
        drawStep(stepIdx % 1000, x0, p, stepIdx === 0 && camX < STEP_W * 0.4);
      }
    }
    // edge fades
    const fadeW = 46;
    let grd = ctx.createLinearGradient(0, 0, fadeW, 0);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grd; ctx.fillRect(0, 0, fadeW, H);
    grd = ctx.createLinearGradient(W - fadeW, 0, W, 0);
    grd.addColorStop(0, 'rgba(255,255,255,0)');
    grd.addColorStop(1, 'rgba(255,255,255,1)');
    ctx.fillStyle = grd; ctx.fillRect(W - fadeW, 0, fadeW, H);

    rafId = requestAnimationFrame(frame);
  }

  function drawStatic() {
    ctx.clearRect(0, 0, W, H);
    drawStep(0, 28, 1, true);
    if (W > 640) drawStep(1, 28 + STEP_W, 1, false);
  }

  if (reduceMotion) {
    drawStatic();
    return;
  }
  // animate only while visible
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
