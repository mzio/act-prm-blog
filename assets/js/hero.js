/* Hero banner: an animated chain of states → sampled thoughts → actions.
   Gray nodes = states/observations, hollow blue = candidate latent thoughts
   (several sampled, best one selected), green = logged explicit actions. */
(function () {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const COL = {
    state: '#898781',
    stateFill: '#efeeea',
    thought: '#2a78d6',
    thoughtSoft: 'rgba(42,120,214,0.35)',
    action: '#1baf7a',
    actionFill: 'rgba(27,175,122,0.16)',
    reward: '#eda100',
    ink: '#52514e',
    ghost: 'rgba(137,135,129,0.4)',
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

  // ---- layout constants ----
  const STEP_W = 380;            // horizontal distance per timestep
  const G = 4;                   // candidate thoughts per step
  const PERIOD = 5200;           // ms per timestep cycle
  const CAND_W = 118, CAND_H = 44;

  // deterministic pseudo-random per (step, candidate)
  function prand(i, j) {
    const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
    return x - Math.floor(x);
  }

  function easeOut(t) { return 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3); }
  function clamp01(t) { return Math.min(Math.max(t, 0), 1); }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // draw one timestep anchored at x0 (center-line cy), with local phase p ∈ [0, 1]
  function drawStep(stepIdx, x0, p) {
    const cy = H / 2;
    const sx = x0;               // state node x
    const candX = x0 + 128;      // candidates x
    const ax = x0 + 300;         // action node x

    // pick the "winner" pseudo-randomly per step
    const winner = Math.floor(prand(stepIdx, 9) * G);

    // phases within the cycle
    const pFan = easeOut(p / 0.18);                       // edges fan out
    const pType = clamp01((p - 0.12) / 0.30);             // thoughts "type"
    const pScore = clamp01((p - 0.42) / 0.18);            // reward bars fill
    const pPick = clamp01((p - 0.60) / 0.14);             // winner selected
    const pAct = clamp01((p - 0.72) / 0.16);              // edge to action + action appears

    // --- state node ---
    ctx.lineWidth = 1.6;
    ctx.fillStyle = COL.stateFill;
    ctx.strokeStyle = COL.state;
    ctx.beginPath();
    ctx.arc(sx, cy, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = COL.ink;
    ctx.font = 'italic 15px Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('s', sx, cy + 1);

    // --- candidate thoughts (kept in a band around the center line) ---
    const band = Math.min(Math.max(H - 140, 220), 430);
    const spread = band / (G - 1);
    const bandTop = cy - band / 2 - CAND_H / 2;
    for (let g = 0; g < G; g++) {
      const ty = bandTop + spread * g;
      const isWin = g === winner;
      const fade = pPick > 0 && !isWin ? 1 - 0.72 * pPick : 1;

      // edge from state to candidate (quadratic)
      const t = clamp01(pFan * (1 - 0.08 * g));
      if (t > 0.01) {
        ctx.save();
        ctx.globalAlpha = fade * 0.9;
        ctx.strokeStyle = isWin && pPick > 0 ? COL.thought : COL.ghost;
        ctx.lineWidth = isWin && pPick > 0 ? 1.8 : 1.1;
        ctx.beginPath();
        const x1 = sx + 20, y1 = cy;
        const x2 = candX - 8, y2 = ty + CAND_H / 2;
        const mx = (x1 + x2) / 2;
        // partial quadratic curve
        const steps = 24;
        for (let i = 0; i <= steps * t; i++) {
          const u = i / steps;
          const bx = (1 - u) * (1 - u) * x1 + 2 * (1 - u) * u * mx + u * u * x2;
          const by = (1 - u) * (1 - u) * y1 + 2 * (1 - u) * u * ((y1 + y2) / 2) + u * u * y2;
          if (i === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
        }
        ctx.stroke();
        ctx.restore();
      }

      if (pType <= 0) continue;

      // bubble
      ctx.save();
      ctx.globalAlpha = fade;
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = isWin && pPick > 0 ? COL.thought : COL.thoughtSoft;
      ctx.lineWidth = isWin && pPick > 0 ? 1.8 : 1.2;
      if (!(isWin && pPick > 0)) ctx.setLineDash([4, 3]);
      roundRect(candX, ty, CAND_W, CAND_H, 9);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      // winner glow
      if (isWin && pPick > 0) {
        ctx.save();
        ctx.globalAlpha = 0.28 * pPick * fade;
        ctx.strokeStyle = COL.thought;
        ctx.lineWidth = 5;
        roundRect(candX - 2, ty - 2, CAND_W + 4, CAND_H + 4, 11);
        ctx.stroke();
        ctx.restore();
      }

      // "typing" text lines inside bubble
      const lines = 2;
      for (let l = 0; l < lines; l++) {
        const full = CAND_W - 40 - prand(stepIdx, g * 7 + l) * 26;
        const lineP = clamp01(pType * (lines + 0.6) - l);
        if (lineP <= 0) continue;
        ctx.strokeStyle = isWin && pPick > 0 ? 'rgba(42,120,214,0.55)' : 'rgba(137,135,129,0.45)';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(candX + 26, ty + 15 + l * 11);
        ctx.lineTo(candX + 26 + full * lineP, ty + 15 + l * 11);
        ctx.stroke();
      }
      // z glyph
      ctx.fillStyle = isWin && pPick > 0 ? COL.thought : 'rgba(42,120,214,0.55)';
      ctx.font = 'italic 13px Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('z', candX + 10, ty + 21);

      // reward bar under bubble
      if (pScore > 0) {
        const rw = 0.25 + 0.75 * prand(stepIdx, g * 3 + 1);
        const rewardW = (CAND_W - 20) * (isWin ? Math.max(rw, 0.86) : Math.min(rw, 0.62));
        ctx.strokeStyle = 'rgba(0,0,0,0.07)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(candX + 10, ty + CAND_H + 7);
        ctx.lineTo(candX + CAND_W - 10, ty + CAND_H + 7);
        ctx.stroke();
        ctx.strokeStyle = isWin && pPick > 0.3 ? COL.thought : COL.reward;
        ctx.beginPath();
        ctx.moveTo(candX + 10, ty + CAND_H + 7);
        ctx.lineTo(candX + 10 + rewardW * easeOut(pScore), ty + CAND_H + 7);
        ctx.stroke();
      }
      ctx.restore();
    }

    // --- edge winner → action + action node ---
    if (pAct > 0) {
      const wy = bandTop + spread * winner + CAND_H / 2;
      ctx.strokeStyle = COL.thought;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      const x1 = candX + CAND_W, y1 = wy, x2 = ax - 22, y2 = cy;
      const steps = 24, tt = easeOut(pAct);
      for (let i = 0; i <= steps * tt; i++) {
        const u = i / steps;
        const bx = (1 - u) * (1 - u) * x1 + 2 * (1 - u) * u * ((x1 + x2) / 2) + u * u * x2;
        const by = (1 - u) * (1 - u) * y1 + 2 * (1 - u) * u * ((y1 + y2) / 2) + u * u * y2;
        if (i === 0) ctx.moveTo(bx, by); else ctx.lineTo(bx, by);
      }
      ctx.stroke();

      if (pAct > 0.55) {
        const ap = easeOut((pAct - 0.55) / 0.45);
        ctx.save();
        ctx.globalAlpha = ap;
        ctx.fillStyle = COL.actionFill;
        ctx.strokeStyle = COL.action;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(ax, cy, 20 * (0.6 + 0.4 * ap), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#0e7a54';
        ctx.font = 'italic 15px Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText('x', ax, cy + 1);
        ctx.restore();

        // connector to next state
        if (pAct > 0.85) {
          ctx.strokeStyle = COL.ghost;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(ax + 20, cy);
          ctx.lineTo(ax + 20 + (STEP_W - 320) * easeOut((pAct - 0.85) / 0.15), cy);
          ctx.stroke();
        }
      }
    }
  }

  function frame(now) {
    ctx.clearRect(0, 0, W, H);
    // global time → camera position; one step completes each PERIOD
    const tGlobal = now / PERIOD;
    const camX = tGlobal * STEP_W;

    const first = Math.floor(camX / STEP_W) - 1;
    const count = Math.ceil(W / STEP_W) + 3;
    for (let k = 0; k < count; k++) {
      const stepIdx = first + k;
      const x0 = stepIdx * STEP_W - camX + W * 0.12;
      // step's own phase: completes when camera reaches it
      const p = clamp01(tGlobal - stepIdx + 0.2);
      if (x0 > -STEP_W && x0 < W + 60) drawStep(((stepIdx % 1000) + 1000) % 1000, x0, p);
    }

    // edge fades at both sides
    const fadeW = Math.min(120, W * 0.14);
    let grd = ctx.createLinearGradient(0, 0, fadeW, 0);
    grd.addColorStop(0, 'rgba(252,252,251,1)');
    grd.addColorStop(1, 'rgba(252,252,251,0)');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, fadeW, H);
    grd = ctx.createLinearGradient(W - fadeW, 0, W, 0);
    grd.addColorStop(0, 'rgba(252,252,251,0)');
    grd.addColorStop(1, 'rgba(252,252,251,1)');
    ctx.fillStyle = grd;
    ctx.fillRect(W - fadeW, 0, fadeW, H);

    requestAnimationFrame(frame);
  }

  function drawStatic() {
    ctx.clearRect(0, 0, W, H);
    // three fully-resolved steps
    for (let k = 0; k < Math.ceil(W / STEP_W) + 1; k++) {
      drawStep(k, k * STEP_W + 40, 1);
    }
  }

  if (reduceMotion) drawStatic();
  else requestAnimationFrame(frame);
})();
