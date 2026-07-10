/* KaTeX rendering, hoverable/clickable equation terms with a floating tooltip,
   collapsible-section handling, bibtex copy buttons. */
(function () {
  const TERM_INFO = {
    'term-post': {
      color: '#4a3aa7',
      title: 'Posterior over thoughts',
      html: 'p<sub>θ⁽ⁿ⁾</sub>(z | x, s) — the distribution over latent thoughts <em>given</em> the action we already know was taken. The E-step wants expectations under it, but we can\'t sample from it directly.',
    },
    'term-z': {
      color: '#2a78d6',
      title: 'Thought policy (sampleable)',
      html: 'p<sub>θ⁽ⁿ⁾</sub>(z | s) — the LLM\'s own distribution over thoughts given only the current state. This we <em>can</em> sample — it\'s just the model generating. Bayes\' rule lets us swap the posterior for this, leaving a reward behind.',
    },
    'term-x': {
      color: '#0e7a54',
      title: 'Action likelihood → the reward',
      html: 'p<sub>θ⁽ⁿ⁾</sub>(x | z, s) — how probable the logged action is once we condition on a candidate thought. A thought is rewarded exactly to the extent it makes the human\'s actual next action more likely.',
    },
    'term-marg': {
      color: '#9a6900',
      title: 'Marginal action likelihood (intractable)',
      html: 'p<sub>θ⁽ⁿ⁾</sub>(x | s) — an integral over <em>all</em> possible thoughts, so intractable. We estimate it with the mean unnormalized reward of the same G sampled thoughts — which is what makes the final reward group-normalized.',
    },
    'term-joint': {
      color: '#52514e',
      title: 'Complete-data log-likelihood',
      html: 'log p<sub>θ</sub>(x, z | s) — thought and action together, the quantity the M-step maximizes. Note it carries the free θ (everything else is frozen at θ⁽ⁿ⁾) — this is the term the gradient flows through.',
    },
  };
  const ALL_TERMS = Object.keys(TERM_INFO);

  /* ---------- KaTeX ---------- */
  function renderMath() {
    if (typeof renderMathInElement !== 'function') return setTimeout(renderMath, 60);
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      trust: (ctx) => ctx.command === '\\htmlClass',
      strict: false,
      throwOnError: false,
    });
    bindTerms();
  }

  /* ---------- equation terms: hover tooltip + cross-highlight + click-to-pin ---------- */
  function bindTerms() {
    const explain = document.getElementById('eq-explain');
    const defaultText = explain ? explain.textContent : '';
    let pinned = null;

    // floating tooltip
    const tip = document.createElement('div');
    tip.className = 'eq-tooltip';
    document.body.appendChild(tip);

    function termOf(el) {
      const span = el.closest && el.closest('.eq-term');
      if (!span) return null;
      return ALL_TERMS.find((t) => span.classList.contains(t)) || null;
    }

    function light(term, on) {
      document.querySelectorAll('.eq-term.' + term).forEach((el) => el.classList.toggle('lit', on));
      document.querySelectorAll('#eq-legend button').forEach((b) =>
        b.classList.toggle('active', on && b.dataset.term === term));
      if (explain) {
        if (on && TERM_INFO[term]) {
          explain.innerHTML =
            `<strong style="color:${TERM_INFO[term].color}">${TERM_INFO[term].title}</strong> — ${TERM_INFO[term].html}`;
          explain.style.borderColor = TERM_INFO[term].color;
        } else if (!pinned) {
          explain.textContent = defaultText;
          explain.style.borderColor = '';
        }
      }
    }

    function showTip(term, x, y) {
      const info = TERM_INFO[term];
      tip.innerHTML = `<strong style="color:${info.color}">${info.title}</strong><br>${info.html}`;
      tip.classList.add('show');
      const pad = 14;
      const w = Math.min(340, window.innerWidth - 24);
      let left = Math.min(x + pad, window.innerWidth - w - 12);
      tip.style.left = Math.max(12, left) + 'px';
      // above the cursor if there's room, else below
      const h = tip.offsetHeight || 90;
      tip.style.top = (y - h - pad > 8 ? y - h - pad : y + pad) + 'px';
    }
    function hideTip() { tip.classList.remove('show'); }

    // delegated events cover every KaTeX-rendered span, whenever it was rendered
    document.addEventListener('mouseover', (e) => {
      const term = termOf(e.target);
      if (!term) return;
      if (!pinned) light(term, true);
      showTip(term, e.clientX, e.clientY);
    });
    document.addEventListener('mousemove', (e) => {
      const term = termOf(e.target);
      if (term) showTip(term, e.clientX, e.clientY);
    });
    document.addEventListener('mouseout', (e) => {
      const term = termOf(e.target);
      if (!term) return;
      if (!pinned) light(term, false);
      hideTip();
    });
    document.addEventListener('click', (e) => {
      const term = termOf(e.target);
      if (term) {                    // click a term: pin/unpin its highlight
        if (pinned && pinned !== term) light(pinned, false);
        pinned = pinned === term ? null : term;
        light(term, pinned === term);
        return;
      }
      if (pinned && !e.target.closest('#eq-legend')) {   // click away: unpin
        light(pinned, false);
        pinned = null;
      }
    });

    document.querySelectorAll('#eq-legend button').forEach((btn) => {
      const term = btn.dataset.term;
      btn.addEventListener('mouseenter', () => { if (!pinned) light(term, true); });
      btn.addEventListener('mouseleave', () => { if (!pinned) light(term, false); });
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (pinned && pinned !== term) light(pinned, false);
        pinned = pinned === term ? null : term;
        light(term, pinned === term);
      });
    });
  }

  /* ---------- collapsibles: auto-open ancestors when a hash points inside one ---------- */
  function openCollapseForHash() {
    const id = location.hash.slice(1);
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    let el = target;
    while (el) {
      if (el.tagName === 'DETAILS') el.open = true;
      el = el.parentElement;
    }
  }
  window.addEventListener('hashchange', openCollapseForHash);
  openCollapseForHash();

  // hint text flips with the block's state
  document.querySelectorAll('details.collapse').forEach((d) => {
    const hint = d.querySelector('.collapse-hint');
    if (!hint) return;
    const update = () => { hint.textContent = d.open ? 'click to collapse' : 'click to expand'; };
    d.addEventListener('toggle', update);
    update();
  });

  /* ---------- quick navigation: close the dropdown after choosing a section ---------- */
  const quicknav = document.getElementById('quicknav');
  if (quicknav) {
    quicknav.querySelectorAll('a').forEach((a) =>
      a.addEventListener('click', () => { quicknav.open = false; }));
    document.addEventListener('click', (e) => {
      if (quicknav.open && !quicknav.contains(e.target)) quicknav.open = false;
    });
  }

  /* ---------- bibtex copy ---------- */
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const pre = document.getElementById(btn.dataset.copy);
      navigator.clipboard.writeText(pre.textContent).then(() => {
        btn.textContent = 'copied!';
        setTimeout(() => (btn.textContent = 'copy'), 1400);
      });
    });
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMath);
  } else {
    renderMath();
  }
})();
