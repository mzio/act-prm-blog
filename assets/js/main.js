/* KaTeX rendering, hoverable equation terms, bibtex copy buttons. */
(function () {
  const TERM_INFO = {
    'term-post': {
      color: '#4a3aa7',
      html: '<strong style="color:#4a3aa7">Posterior over thoughts</strong> — p<sub>θ⁽ⁿ⁾</sub>(z | x, s): the distribution over latent thoughts <em>given</em> the action we already know was taken. The E-step wants expectations under it, but we can\'t sample from it directly.',
    },
    'term-z': {
      color: '#2a78d6',
      html: '<strong style="color:#2a78d6">Thought policy</strong> — p<sub>θ⁽ⁿ⁾</sub>(z | s): the LLM\'s own distribution over thoughts given only the current state. This we <em>can</em> sample — it\'s just the model generating. Bayes\' rule lets us swap the posterior for this, leaving a reward behind.',
    },
    'term-x': {
      color: '#0e7a54',
      html: '<strong style="color:#0e7a54">Action likelihood → the reward</strong> — p<sub>θ⁽ⁿ⁾</sub>(x | z, s): how probable the logged action is once we condition on a candidate thought. A thought is rewarded exactly to the extent it makes the human\'s actual next action more likely.',
    },
    'term-marg': {
      color: '#9a6900',
      html: '<strong style="color:#9a6900">Marginal action likelihood</strong> — p<sub>θ⁽ⁿ⁾</sub>(x | s): an integral over <em>all</em> possible thoughts, so intractable. We estimate it with the mean unnormalized reward of the same G sampled thoughts — which is what makes the final reward group-normalized.',
    },
    'term-joint': {
      color: '#52514e',
      html: '<strong>Complete-data log-likelihood</strong> — log p<sub>θ</sub>(x, z | s): thought and action together, the quantity the M-step maximizes. Note it carries the free θ (everything else is frozen at θ⁽ⁿ⁾) — this is the term the gradient flows through.',
    },
  };

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

  function bindTerms() {
    const explain = document.getElementById('eq-explain');
    if (!explain) return;
    const defaultText = explain.textContent;
    let pinned = null;

    function light(term, on) {
      document.querySelectorAll('.eq-term.' + term).forEach((el) => el.classList.toggle('lit', on));
      document.querySelectorAll('#eq-legend button').forEach((b) =>
        b.classList.toggle('active', on && b.dataset.term === term));
      if (on && TERM_INFO[term]) {
        explain.innerHTML = TERM_INFO[term].html;
        explain.style.borderColor = TERM_INFO[term].color;
      } else if (!pinned) {
        explain.textContent = defaultText;
        explain.style.borderColor = '';
      }
    }

    const allTerms = Object.keys(TERM_INFO);
    document.querySelectorAll('.eq-term').forEach((el) => {
      const term = allTerms.find((t) => el.classList.contains(t));
      if (!term) return;
      el.addEventListener('mouseenter', () => { if (!pinned) light(term, true); });
      el.addEventListener('mouseleave', () => { if (!pinned) light(term, false); });
    });

    document.querySelectorAll('#eq-legend button').forEach((btn) => {
      const term = btn.dataset.term;
      btn.addEventListener('mouseenter', () => { if (!pinned) light(term, true); });
      btn.addEventListener('mouseleave', () => { if (!pinned) light(term, false); });
      btn.addEventListener('click', () => {
        if (pinned === term) {
          pinned = null;
          light(term, false);
        } else {
          if (pinned) light(pinned, false);
          pinned = term;
          light(term, true);
        }
      });
    });
  }

  // bibtex copy
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
