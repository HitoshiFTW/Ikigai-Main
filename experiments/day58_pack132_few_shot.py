"""
Day 58 Pack 132b -- Few-shot pattern learning in Ikigai.

Demonstrates org.few_shot_learn / org.few_shot_apply: 5 novel pattern mappings
learned from a single set of examples, then queried after distractor floods.
The structural no-forgetting claim applied to the "in-context learning" use
case (where LLMs do this via prompt; we do it via flat-memory write).

Verifications:
    V1  org.few_shot_learn / apply API wired
    V2  5 novel patterns recovered exactly after learning (baseline)
    V3  same 5 patterns recovered after 20 distractor pattern injections
    V4  same 5 patterns recovered after 5000 text co-occurrence writes
    V5  same 5 patterns recovered after 100 vision classifications
    V6  fresh few-shot pattern learned AFTER distractors still works
    V7  substrate FIXED bytes throughout
    V8  unrelated input (not in examples) returns LOW-confidence answer
"""

import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from integrate import IkigaiOrganism

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 132b: Few-shot learning in Ikigai ===\n')

org = IkigaiOrganism(flat_only=True)
sub0 = org.unified.substrate_bytes()
_log(f'  organism: substrate={sub0/1_048_576:.0f} MB FIXED')

check('V1 few_shot_learn / apply wired',
      hasattr(org, 'few_shot_learn') and hasattr(org, 'few_shot_apply'))

#  5 novel pattern mappings (made-up words to colors)
PATTERNS = [
    ('flompet',  'red'),
    ('zarble',   'blue'),
    ('quintex',  'green'),
    ('mibvor',   'yellow'),
    ('pradnar',  'purple'),
]
def check_all(after=''):
    ok = 0; got = {}
    for inp, exp in PATTERNS:
        pred, score = org.few_shot_apply(inp)
        got[inp] = pred
        if pred == exp: ok += 1
    return ok, got

#  learn the 5 patterns (Phase 0)
_log('\nLearning 5 novel patterns...')
org.few_shot_learn(PATTERNS, n_reinforce=30)
ok, got = check_all()
_log(f'  baseline: {ok}/5 -- {got}')
check('V2 5 patterns recovered exactly (baseline)', ok == 5)

#  Phase A: flood with 20 unrelated patterns
_log('\n--- Phase A: +20 unrelated pattern distractors ---')
DISTR_PATTERNS = [(f'distractor_{i}', f'noise_{i % 7}') for i in range(20)]
org.few_shot_learn(DISTR_PATTERNS, n_reinforce=30)
ok_a, got_a = check_all('A')
_log(f'  originals: {ok_a}/5 -- {got_a}')
check('V3 5 patterns survive +20 pattern distractors', ok_a == 5)

#  Phase B: flood with 5000 text co-occurrence writes
_log('\n--- Phase B: +5000 text co-occurrence writes ---')
TEXT_BANK = ["the cat sat on the mat", "the dog ran fast",
             "a small bird sang loud", "the boy played in the park",
             "she ate the sweet fruit", "the king wore a crown"]
rng = random.Random(132)
for _ in range(5000):
    org.unified.expose_cooccur(rng.choice(TEXT_BANK))
ok_b, got_b = check_all('B')
_log(f'  originals: {ok_b}/5 -- {got_b}')
check('V4 5 patterns survive +5K text co-occurrence', ok_b == 5)

#  Phase C: flood with 100 vision classifications
_log('\n--- Phase C: +100 vision (digit) classifications ---')
from sklearn.datasets import load_digits
digits = load_digits()
for i in rng.sample(range(len(digits.data)), 100):
    org.expose_image(digits.data[i].astype(np.float32) / 16.0, int(digits.target[i]))
ok_c, got_c = check_all('C')
_log(f'  originals: {ok_c}/5 -- {got_c}')
check('V5 5 patterns survive +100 vision writes', ok_c == 5)

#  Phase D: NEW few-shot pattern after all distractors
_log('\n--- Phase D: learn NEW pattern after all distractors ---')
NEW = [('flibbergast', 'cyan'), ('whorblix', 'magenta')]
org.few_shot_learn(NEW, n_reinforce=30)
new_ok = sum(org.few_shot_apply(i)[0] == e for i, e in NEW)
ok_orig_d, _ = check_all('D')
_log(f'  new patterns: {new_ok}/2   originals after new+distractors: {ok_orig_d}/5')
check('V6 new few-shot pattern works AFTER distractors',
      new_ok == 2 and ok_orig_d == 5,
      f'new={new_ok}/2 orig={ok_orig_d}/5')

#  Phase E: substrate flat
sub_final = org.unified.substrate_bytes()
check('V7 substrate FIXED throughout', sub_final == sub0,
      f'{sub0} -> {sub_final}')

#  Phase F: unknown input -> low confidence
unk_pred, unk_score = org.few_shot_apply('flarbanitz')
_log(f'\n  unknown input "flarbanitz" -> {unk_pred} (score={unk_score:.3f})')
# A known input scores much higher
known_pred, known_score = org.few_shot_apply('flompet')
_log(f'  known input "flompet" -> {known_pred} (score={known_score:.3f})')
check('V8 unknown input has lower confidence than known',
      known_score > unk_score + 0.1,
      f'known={known_score:.3f} unk={unk_score:.3f}')

#  summary
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 132b -- Few-shot learning in Ikigai')
_log(f'{"="*64}')
_log(f'  baseline           : {ok}/5')
_log(f'  after +20 patterns : {ok_a}/5')
_log(f'  after +5K text     : {ok_b}/5')
_log(f'  after +100 vision  : {ok_c}/5')
_log(f'  new patterns       : {new_ok}/2')
_log(f'  originals at end   : {ok_orig_d}/5')
_log(f'  substrate FIXED    : {sub_final/1_048_576:.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- few-shot learning + no-forgetting' if FAIL == 0 else 'NEEDS FIX'))
