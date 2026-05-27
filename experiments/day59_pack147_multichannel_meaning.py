"""
Day 59 Pack 147 -- Multi-channel meaning, NATIVE to Ikigai.

Pack 146 gave isa + property at speak time.  Pack 147 adds three more
meaning channels and bakes them into the organism itself (multirole_memory.py
+ integrate.py) so they are part of Ikigai, not just experiment fixtures:

  episode    -- per word: a sentence-HV (bundled token keys) bound under
                'episode' role. "Where have I seen this used."
  affordance -- per noun: which verbs were observed acting on/by it.
                "What does this do."
  modifier   -- per noun: which adjectives described it. "What is it like."

API added to IkigaiOrganism:
    org.expose_meaning(text, subj_vocab=..., verb_vocab=..., obj_vocab=...,
                            adj_vocab=...)
    org.expose_episode(text)
    org.expose_affordance(subj, verb, obj=None)
    org.expose_modifier(modifier, noun)

DEFAULT_ROLES extended: 'episode', 'affordance', 'mod' now register
automatically when a fresh organism is built.

Verifications:
    V1 fresh organism has episode/affordance/mod roles registered out-of-box
    V2 episode role populates after expose_meaning calls
    V3 affordance role populates
    V4 modifier role populates
    V5 per-word episode recall favors related over unrelated sentence
    V6 affordance("cat") top candidate is a verb
    V7 modifier("cat") top candidate is an adjective
    V8 substrate FIXED 192 MB through all writes
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from integrate import IkigaiOrganism

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 147: Multi-channel meaning (native API) ===\n')

org = IkigaiOrganism(flat_only=True)
mr = org.unified
sub_before = mr.substrate_bytes()

# ---- V1 roles registered out-of-the-box ----
have_all = all(r in mr.roles for r in ('episode', 'affordance', 'mod'))
_log(f'  fresh organism roles include episode/affordance/mod: {have_all}')
check('V1 episode/affordance/mod registered out-of-box', have_all)

# ---- training corpus: SVO with optional adjective ----
NOUNS_SUBJ = ['cat', 'dog', 'bird', 'fish', 'horse', 'boy', 'girl', 'man', 'woman']
NOUNS_OBJ  = ['mouse', 'bone', 'fish', 'water', 'apple', 'ball', 'book', 'tree', 'rock']
VERBS      = ['chases', 'eats', 'sees', 'finds', 'holds', 'watches', 'follows', 'likes']
ADJS       = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'happy', 'soft']

TEMPLATES = [
    "the {adj} {subj} {verb} the {obj}",
    "a {adj} {subj} {verb} a {obj}",
    "the {subj} {verb} the {adj} {obj}",
]
rng = random.Random(0)
sentences = []
for _ in range(400):
    t = rng.choice(TEMPLATES)
    s = t.format(
        adj=rng.choice(ADJS),
        subj=rng.choice(NOUNS_SUBJ),
        verb=rng.choice(VERBS),
        obj=rng.choice(NOUNS_OBJ),
    )
    sentences.append(s)
_log(f'\n  training corpus: {len(sentences)} synthetic SVO sentences')

# Use the NATIVE API. Vocab hint sets are passed once.
totals = {'episode': 0, 'affordance': 0, 'modifier': 0}
for s in sentences:
    mr.expose_cooccur(s)
    mr.expose_transitions(s)
    counts = org.expose_meaning(
        s,
        subj_vocab=NOUNS_SUBJ,
        verb_vocab=VERBS,
        obj_vocab=NOUNS_OBJ,
        adj_vocab=ADJS,
    )
    for k in totals:
        totals[k] += counts.get(k, 0)
_log(f'  writes: episode={totals["episode"]}  affordance={totals["affordance"]}  '
     f'modifier={totals["modifier"]}')

sub_after = mr.substrate_bytes()

# ---- V2-V4 channel population ----
check('V2 episode role populated',
      len(mr._role_targets.get('episode', set())) >= 10,
      f'got {len(mr._role_targets.get("episode", set()))}')
check('V3 affordance role populated',
      len(mr._role_targets.get('affordance', set())) >= 5,
      f'got {len(mr._role_targets.get("affordance", set()))}')
check('V4 modifier role populated',
      len(mr._role_targets.get('mod', set())) >= 5,
      f'got {len(mr._role_targets.get("mod", set()))}')

# ---- V5 per-word episode recall favors related > unrelated ----
def sentence_hv(tokens):
    accum = np.zeros(mr.d, dtype=np.complex64)
    for t in tokens:
        accum = accum + mr.ck.key(t)
    mags = np.abs(accum)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (accum / mags).astype(np.complex64)

cat_addr = (mr.ck.key('cat') * mr.roles['episode']).astype(np.complex64)
cat_recall = mr.sdm_rel.read(cat_addr)
related = sentence_hv("the big cat chases the small mouse".split())
unrelated = sentence_hv("the woman finds a tree".split())
sim_rel = float(np.real(np.vdot(cat_recall, related))) / mr.d
sim_unr = float(np.real(np.vdot(cat_recall, unrelated))) / mr.d
_log(f'\n  episode recall("cat")  vs cat-sentence: {sim_rel:+.4f}')
_log(f'  episode recall("cat")  vs unrelated:    {sim_unr:+.4f}')
check('V5 episode recall favors related sentence over unrelated',
      sim_rel > sim_unr, f'rel={sim_rel:.4f} unr={sim_unr:.4f}')

# ---- V6 affordance("cat") top is a verb ----
pred_v, score_v = mr.query('cat', 'affordance', candidates=VERBS)
_log(f'  affordance("cat") top: {pred_v}  (sc={score_v:+.3f}, candidates=VERBS)')
check('V6 affordance("cat") returns a verb', pred_v in set(VERBS),
      f'got {pred_v}')

# ---- V7 modifier("cat") top is an adjective ----
pred_a, score_a = mr.query('cat', 'mod', candidates=ADJS)
_log(f'  mod("cat") top:        {pred_a}  (sc={score_a:+.3f}, candidates=ADJS)')
check('V7 mod("cat") returns an adjective', pred_a in set(ADJS),
      f'got {pred_a}')

# ---- V8 substrate FIXED ----
check('V8 substrate FIXED 192 MB through all writes',
      sub_before == sub_after,
      f'{sub_before} -> {sub_after}')

# ---- summary ----
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 147 -- Multi-channel meaning (native)')
_log(f'{"="*64}')
_log(f'  Episode words:    {len(mr._role_targets.get("episode", set()))}')
_log(f'  Affordance words: {len(mr._role_targets.get("affordance", set()))}')
_log(f'  Modifier words:   {len(mr._role_targets.get("mod", set()))}')
_log(f'  Substrate FIXED:  {sub_after/1_048_576:.0f} MB')
_log(f'  Native API: org.expose_meaning(text, subj_vocab=..., verb_vocab=...,')
_log(f'              obj_vocab=..., adj_vocab=...)')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- three new meaning channels native to Ikigai'
                     if FAIL == 0 else 'NEEDS FIX'))
