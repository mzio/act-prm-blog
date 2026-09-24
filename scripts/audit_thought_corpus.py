#!/usr/bin/env python3
"""Audit a thought corpus for few-shot contamination.

    uv run python scripts/audit_thought_corpus.py data/sft_corpus/snorkel_insurance/policy_fs2
    ... --gate            # exit 1 if the corpus looks contaminated (for use in a chain)

WHY: Stage-1's reversal prompt is seeded with ONE worked example. Until 2026-09-23 that
example was the finance/`acme` task for every domain, and it poisoned insurance -- which
shares its task modality (SQL over tables) but not its subject. Measured on the shipped
insurance corpora: 77.7% of `policy` thoughts carried finance vocabulary, `base` had 308
verbatim "I need revenue figures for acme" openers and 57.5% duplicate prefixes. Sample,
on an appetite question about a Maine B&B: "I need to find revenue growth in 2024..."
followed by get_table_data_dictionary(small_business_insurance_appetite) -- the thought
contradicts its own action.

Retail/airline scored 0.0% on the structural tests because they are CRUD/API domains with
a tool vocabulary disjoint from the seed's: nothing to copy, so the model had to ground.
Style transfers regardless (the "I need ..." opener runs 78-82% everywhere), which is the
few-shot doing its job -- it should teach FORMAT, not content. So `starts_i_need` is
reported but NOT gated on.

Gate thresholds are set above the clean corpora (retail/airline: 0.0% n-gram overlap,
1.3-3.8% duplicate prefixes) and below the poisoned ones (insurance base: 18.8% / 57.5%).
"""
import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# Gate: fail if a corpus exceeds any of these.
MAX_SHARED_4GRAM = 3.0   # % of thoughts sharing a 4-gram with their own seed
MAX_DUP_PREFIX = 15.0    # % of thoughts whose first 60 chars repeat another's
MIN_N = 20               # below this, the percentages are noise


def _words(s):
    return re.findall(r"[a-z']+", s.lower())


def thoughts_of(path):
    out = []
    for split in ("train", "eval"):
        fp = os.path.join(path, f"{split}.json")
        if not os.path.exists(fp):
            continue
        for traj in json.load(open(fp)):
            for m in traj.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                c = m.get("content") or ""
                i = c.find("<tool_call>")
                if i < 0:
                    continue
                z = c[:i].strip()
                if len(z) >= 10:
                    out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="directory holding train.json / eval.json")
    ap.add_argument("--domain", default=None,
                    help="few-shot domain to compare against; default inferred from the path")
    ap.add_argument("--gate", action="store_true", help="exit 1 when the corpus looks contaminated")
    args = ap.parse_args()

    from act_prm.generator.act_prm.prompts import (
        FEWSHOT_FINANCE, THOUGHT_BOS, THOUGHT_EOS, resolve_fewshot,
    )

    def _seed_words(fs):
        t = fs[1]["content"]
        if THOUGHT_BOS in t:
            t = t.split(THOUGHT_BOS, 1)[1].replace(THOUGHT_EOS, "")
        return _words(t)

    dom = args.domain or args.corpus
    seed = resolve_fewshot(dom)
    # Compare against BOTH this domain's seed and the LEGACY shared finance seed. A corpus
    # generated before 2026-09-23 was produced under the finance seed regardless of domain,
    # so checking only the resolved seed reports a clean 0.0% on exactly the corpora that
    # are known to be poisoned.
    SEEDS = [("domain", seed), ("legacy-finance", FEWSHOT_FINANCE)]
    grams = {}
    for lbl, fs in SEEDS:
        sw = _seed_words(fs)
        grams[lbl] = {n: {tuple(sw[i:i + n]) for i in range(len(sw) - n + 1)} for n in (4, 5)}

    z = thoughts_of(args.corpus)
    if not z:
        print(f"  no thoughts found in {args.corpus}")
        return 1 if args.gate else 0

    n = len(z)
    hits = {lbl: [0, 0] for lbl, _ in SEEDS}
    for x in z:
        w = _words(x)
        for lbl, _ in SEEDS:
            if any(tuple(w[i:i + 4]) in grams[lbl][4] for i in range(len(w) - 3)):
                hits[lbl][0] += 1
            if any(tuple(w[i:i + 5]) in grams[lbl][5] for i in range(len(w) - 4)):
                hits[lbl][1] += 1
    heads = collections.Counter(x[:60] for x in z)
    dup = sum(c for c in heads.values() if c > 1)
    ineed = sum(1 for x in z if x.lower().lstrip("*# ").startswith("i need"))
    med = sorted(len(x) for x in z)[n // 2]

    pdup = 100 * dup / n
    p4 = max(100 * hits[lbl][0] / n for lbl, _ in SEEDS)
    print(f"  corpus        : {args.corpus}")
    print(f"  thoughts      : {n}   median {med} chars")
    for lbl, _ in SEEDS:
        print(f"  shared 4/5-gram vs {lbl:14s}: {100*hits[lbl][0]/n:5.1f}% / {100*hits[lbl][1]/n:5.1f}%"
              f"{'   (gate < %.1f)' % MAX_SHARED_4GRAM if lbl == 'domain' else ''}")
    print(f"  dup prefix    : {pdup:5.1f}%   (gate < {MAX_DUP_PREFIX})")
    print(f"  starts 'I need': {100*ineed/n:4.1f}%   (style, not gated)")
    for s, c in heads.most_common(3):
        if c > 1:
            print(f"     {c:4d}x  {s!r}")

    bad = n >= MIN_N and (p4 >= MAX_SHARED_4GRAM or pdup >= MAX_DUP_PREFIX)
    print(f"  VERDICT       : {'CONTAMINATED' if bad else 'clean'}")
    return 1 if (bad and args.gate) else 0


if __name__ == "__main__":
    sys.exit(main())
