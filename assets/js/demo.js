/* Interactive playground: sample candidate thoughts character-by-character,
   score each by the likelihood it induces on the logged next action, select
   the winner, commit it to context, advance to the next step. */
(function () {
  const transcript = document.getElementById('demo-transcript');
  const candsEl = document.getElementById('demo-cands');
  const noteEl = document.getElementById('demo-note');
  const runBtn = document.getElementById('demo-run');
  const resetBtn = document.getElementById('demo-reset');
  if (!transcript) return;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Abridged from a real Snorkel Finance Reasoning trajectory (see Results §2).
  // rTilde = length-normalized p(x | s, z) for each candidate thought.
  const STEPS = [
    {
      context: [
        { kind: 'state', who: 'state s₁ — user', text: 'Here is the question: What is the company’s lease financing strategy and how heavily does it rely on operating leases versus finance leases? The company to query in the database: meta' },
      ],
      action: '<tool_call>{"name": "get_descriptions", "arguments": {"company_name": "meta"}}</tool_call>',
      cands: [
        { text: 'I need to retrieve information about Meta’s lease financing strategy to determine its reliance on operating versus finance leases. The provided tool lets me get descriptions of the tables available for the company “meta” — I should start there.', rTilde: 0.34 },
        { text: 'The question is about operating vs. finance leases, so I’ll write a SQL query against the lease balance-sheet table right away.', rTilde: 0.07 },
        { text: 'Meta is a technology company that primarily leases data centers, offices, and network infrastructure under long-term agreements.', rTilde: 0.02 },
        { text: 'Before answering I should check what data exists. Listing the available table descriptions for this company seems like a reasonable first move.', rTilde: 0.21 },
      ],
    },
    {
      context: [
        { kind: 'obs', who: 'observation o₂ — tool', text: 'Result: 62 tables for “meta”, including “meta_LeaseBalanceSheetInformationTableTextBlock”: lease assets and liabilities reported on the balance sheet…' },
      ],
      action: '<tool_call>{"name": "get_table_info", "arguments": {"company_name": "meta", "table_name": "meta_LeaseBalanceSheetInformationTableTextBlock"}}</tool_call>',
      cands: [
        { text: 'The query result includes a table named “meta_LeaseBalanceSheetInformationTableTextBlock”, which likely contains lease assets and liabilities split by operating and finance leases. I’ll retrieve details from this specific table next.', rTilde: 0.41 },
        { text: 'I have the table list. I’ll answer the question now based on Meta’s well-known preference for operating leases.', rTilde: 0.015 },
        { text: 'There are many tables here. The income-statement lease-cost table might be relevant, so let me inspect the lease cost breakdown first.', rTilde: 0.09 },
        { text: 'To compare operating and finance leases I need the balance-sheet lease table’s schema before querying it.', rTilde: 0.26 },
      ],
    },
  ];

  let stepIdx = 0;
  let busy = false;
  let timers = [];

  function later(fn, ms) {
    if (reduceMotion) ms = 0;
    const t = setTimeout(fn, ms);
    timers.push(t);
    return t;
  }

  function addMsg(kind, who, html) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + kind;
    div.innerHTML = `<div class="who">${who}</div>${html}`;
    transcript.appendChild(div);
    return div;
  }

  function norm(cands) {
    const sum = cands.reduce((a, c) => a + c.rTilde, 0);
    return cands.map((c) => c.rTilde / sum);
  }

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function reset() {
    timers.forEach(clearTimeout);
    timers = [];
    busy = false;
    stepIdx = 0;
    transcript.innerHTML = '';
    candsEl.innerHTML = '';
    noteEl.textContent = '';
    STEPS[0].context.forEach((c) => addMsg(c.kind, c.who, `<span>${esc(c.text)}</span>`));
    noteEl.textContent = 'The next logged action is hidden from the model. Press “Sample thoughts”.';
    runBtn.disabled = false;
    runBtn.textContent = '▶ Sample thoughts';
  }

  function runStep() {
    if (busy || stepIdx >= STEPS.length) return;
    busy = true;
    runBtn.disabled = true;
    const step = STEPS[stepIdx];
    const rewards = norm(step.cands);
    const winner = rewards.indexOf(Math.max.apply(null, rewards));

    candsEl.innerHTML = '';
    noteEl.textContent = 'E-step: sampling G = 4 thoughts z⁽ᵍ⁾ ~ p(z | s) …';

    // build candidate cards
    const cards = step.cands.map((c, i) => {
      const card = document.createElement('div');
      card.className = 'cand';
      card.innerHTML =
        `<div class="cand-label"><span>thought z⁽${i + 1}⁾</span><span class="winner-badge">selected ẑ</span></div>` +
        `<div class="cand-text"><span class="typed"></span><span class="caret"></span></div>` +
        `<div class="rbar-wrap" style="visibility:hidden"><div class="rbar-track"><div class="rbar"></div></div><div class="rval"></div></div>`;
      candsEl.appendChild(card);
      return card;
    });

    // 1) type all candidates concurrently, char by char
    let done = 0;
    step.cands.forEach((c, i) => {
      const typed = cards[i].querySelector('.typed');
      const caret = cards[i].querySelector('.caret');
      const chars = [...c.text];
      const cps = 55 + (i % 2) * 18; // chars per second, slightly desynced
      let j = 0;
      function tick() {
        const burst = reduceMotion ? chars.length : 2 + Math.floor(Math.random() * 3);
        j = Math.min(chars.length, j + burst);
        typed.textContent = chars.slice(0, j).join('');
        if (j < chars.length) later(tick, 1000 * (burst / cps));
        else {
          caret.remove();
          if (++done === step.cands.length) scorePhase();
        }
      }
      later(tick, 120 * i);
    });

    // 2) score
    function scorePhase() {
      noteEl.textContent = 'Scoring: computing p(x | s, z⁽ᵍ⁾) over the logged action’s tokens, then normalizing within the group…';
      later(() => {
        step.cands.forEach((c, i) => {
          const wrap = cards[i].querySelector('.rbar-wrap');
          wrap.style.visibility = 'visible';
          const bar = cards[i].querySelector('.rbar');
          const val = cards[i].querySelector('.rval');
          later(() => { bar.style.width = (rewards[i] * 100).toFixed(1) + '%'; }, 60);
          val.textContent = `r̄ = ${rewards[i].toFixed(2)}`;
        });
        later(pickPhase, 1100);
      }, 350);
    }

    // 3) pick winner
    function pickPhase() {
      cards.forEach((card, i) => card.classList.add(i === winner ? 'winner' : 'loser'));
      noteEl.textContent = 'The highest-reward thought ẑ is committed to context; every sampled thought and its reward go into the policy-gradient batch.';
      later(commitPhase, 1400);
    }

    // 4) commit thought + reveal action
    function commitPhase() {
      addMsg('thought-committed', `committed thought ẑ${stepIdx + 1} — model`, `<span>${esc(step.cands[winner].text)}</span>`);
      later(() => {
        addMsg('action', `logged action x${stepIdx + 1} — from the process log`, `<span>${esc(step.action)}</span>`);
        candsEl.innerHTML = '';
        stepIdx += 1;
        busy = false;
        if (stepIdx < STEPS.length) {
          STEPS[stepIdx].context.forEach((c) => addMsg(c.kind, c.who, `<span>${esc(c.text)}</span>`));
          noteEl.textContent = 'New observation appended to the state. Sample thoughts for the next logged action.';
          runBtn.disabled = false;
          runBtn.textContent = '▶ Sample thoughts (step 2)';
        } else {
          noteEl.textContent = 'End of the abridged trajectory — the relabelled (state, thought, action) tuples are now SFT-ready. Reset to replay.';
          runBtn.textContent = '▶ Sample thoughts';
        }
        transcript.lastElementChild.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'nearest' });
      }, 450);
    }
  }

  runBtn.addEventListener('click', runStep);
  resetBtn.addEventListener('click', reset);
  reset();
})();
