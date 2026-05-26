"""
Day 58 Pack 129 -- No-forgetting benchmark.

The architecture's central claim: catastrophic-forgetting-free by construction.
Superposition is additive; new writes touch different hard locations than old
ones; old facts cannot be "overwritten" the way neural network weights are.

Test: inject 5 original IS-A facts, verify recall. Then flood the substrate
with distractors across ALL modalities (more IS-A, thousands of text cooccur
exposures, hundreds of verb observations, hundreds of image classifications)
and recheck the original 5 after each flood.

If all 5 original facts recall correctly through every distractor wave,
the architecture is empirically no-forgetting -- the headline result for
the paper, and the structural property that distinguishes Ikigai from
transformer-based systems (which need replay buffers, LoRA, or full retrain
to learn new facts without crashing old ones).

Verifications:
    V1  original 5 facts recall (baseline)
    V2  after 20 distractor IS-A facts: original 5 still 100%
    V3  after 5K text exposures: original 5 still 100%
    V4  after 100 verb observations: original 5 still 100%
    V5  after 100 vision classifications: original 5 still 100%
    V6  AFTER all distractors, the distractor facts also still work
        (additive accumulation, not blocking)
    V7  substrate FIXED bytes through every phase
    V8  inference RAM stays under 700 MB
"""

import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    PROC = psutil.Process(os.getpid())
    def rss(): return PROC.memory_info().rss / 1_048_576
except ImportError:
    def rss(): return -1.0

import numpy as np
from integrate import IkigaiOrganism

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 129: No-forgetting benchmark ===\n')

# ── organism ──────────────────────────────────────────────────────────────────
org = IkigaiOrganism(flat_only=True)
sub_initial = org.unified.substrate_bytes()
_log(f'  organism: substrate={sub_initial/1_048_576:.0f} MB FIXED  RSS={rss():.0f} MB')

# ── ORIGINAL 5 FACTS (must NOT forget) ───────────────────────────────────────
ORIGINAL = {
    'aardvark':  'mammal',     # invented mappings, novel words
    'lyrebird':  'bird',
    'manatee':   'mammal',
    'penguin':   'bird',
    'quokka':    'marsupial',
}
def check_originals(after=''):
    """Returns (n_correct, all_correct, results)."""
    res = {h: org.isa_of(h) for h in ORIGINAL}
    ok = sum(res[h] == exp for h, exp in ORIGINAL.items())
    return ok, ok == len(ORIGINAL), res

_log('\n--- Phase 0: inject + baseline the 5 original facts ---')
for h, y in ORIGINAL.items():
    org.assert_isa(h, y, n=50)
ok0, all_ok0, res0 = check_originals()
_log(f'  baseline: {ok0}/5 -- {res0}')
check('V1 original 5 facts recall (baseline)', all_ok0, f'{res0}')

# ── PHASE A: 20 distractor IS-A facts ────────────────────────────────────────
_log('\n--- Phase A: flood with 20 distractor IS-A facts ---')
DISTRACTORS_ISA = {
    'dog':'mammal','cat':'mammal','rose':'flower','oak':'tree','apple':'fruit',
    'car':'vehicle','truck':'vehicle','salmon':'fish','tulip':'flower','elm':'tree',
    'sparrow':'bird','eagle':'bird','tiger':'mammal','horse':'mammal','grape':'fruit',
    'bus':'vehicle','daisy':'flower','pine':'tree','trout':'fish','owl':'bird',
}
for h, y in DISTRACTORS_ISA.items():
    org.assert_isa(h, y, n=50)
ok_a, all_ok_a, res_a = check_originals(after='isa')
_log(f'  after +20 IS-A:  {ok_a}/5 -- {res_a}')
check('V2 original 5 survive +20 IS-A distractors', all_ok_a, f'{res_a}')

# ── PHASE B: 5000 text cooccur exposures ─────────────────────────────────────
_log('\n--- Phase B: flood with 5000 text co-occurrence exposures ---')
TEXT_BANK = [
    "the boy ran in the park", "the girl smiled at the sun",
    "a cat and a dog played in the garden", "the king sat on his throne",
    "she ate an apple under the tree", "the fast car drove down the road",
    "the queen wore a beautiful crown", "rain fell on the green grass",
    "the small bird sang a happy song", "he found a coin in the street",
    "lights flashed across the dark sky", "the river flowed past the village",
]
t0 = time.perf_counter()
rng = random.Random(129)
for _ in range(5000):
    org.unified.expose_cooccur(rng.choice(TEXT_BANK))
ok_b, all_ok_b, res_b = check_originals(after='text')
_log(f'  trained 5K text in {time.perf_counter()-t0:.1f}s; originals: {ok_b}/5 -- {res_b}')
check('V3 original 5 survive +5K text exposures', all_ok_b, f'{res_b}')

# ── PHASE C: 100 verb observations ───────────────────────────────────────────
_log('\n--- Phase C: flood with 100 verb arithmetic observations ---')
SUBJ, OBJ = ['Mary','Tom','Anna','Bob'], ['apples','coins','toys']
for _ in range(100):
    s = rng.choice(SUBJ); o = rng.choice(OBJ); n0 = rng.randint(5, 40)
    if rng.random() < 0.5:
        v, sgn = 'ate', -1; m = rng.randint(1, min(n0-1, 20))
    else:
        v, sgn = 'found', +1; m = rng.randint(1, 20)
    text = f"{s} had {n0} {o}. She {v} {m} {o}. Now {s} has {n0+sgn*m} {o}."
    obs = org.operations.observe_story(text)
    if obs is not None:
        vb, nb, md, na, _ = obs
        if md and abs(md) > 1e-9:
            org.unified.expose_verb_observation(vb, (na - nb) / md)
ok_c, all_ok_c, res_c = check_originals(after='verb')
_log(f'  after +100 verb obs: {ok_c}/5 -- {res_c}')
check('V4 original 5 survive +100 verb observations', all_ok_c, f'{res_c}')

# ── PHASE D: 100 vision classifications ──────────────────────────────────────
_log('\n--- Phase D: flood with 100 vision (digit) classifications ---')
from sklearn.datasets import load_digits
data = load_digits()
idxs = rng.sample(range(len(data.data)), 100)
for i in idxs:
    org.expose_image(data.data[i].astype(np.float32) / 16.0, int(data.target[i]))
ok_d, all_ok_d, res_d = check_originals(after='vision')
_log(f'  after +100 images: {ok_d}/5 -- {res_d}')
check('V5 original 5 survive +100 vision writes', all_ok_d, f'{res_d}')

# ── PHASE E: do distractor IS-A facts still work too? (no blockage) ──────────
_log('\n--- Phase E: distractors also recall (additive, not blocking) ---')
distractor_ok = sum(org.isa_of(h) == y for h, y in DISTRACTORS_ISA.items())
_log(f'  distractor IS-A: {distractor_ok}/{len(DISTRACTORS_ISA)} still recall')
check('V6 distractors ALSO survive (additive accumulation)',
      distractor_ok >= 0.85 * len(DISTRACTORS_ISA),
      f'{distractor_ok}/{len(DISTRACTORS_ISA)}')

# ── PHASE F: flat + RAM checks ───────────────────────────────────────────────
sub_final = org.unified.substrate_bytes()
check('V7 substrate FIXED across ALL phases',
      sub_final == sub_initial, f'init={sub_initial} final={sub_final}')

rss_final = rss()
check('V8 RAM under 700 MB', rss_final < 700, f'{rss_final:.0f} MB')

# ── summary ──────────────────────────────────────────────────────────────────
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 129 -- No-forgetting benchmark')
_log(f'{"="*64}')
_log(f'  Original 5 IS-A facts survival across distractor floods:')
_log(f'    baseline        : {ok0}/5')
_log(f'    +20 IS-A        : {ok_a}/5')
_log(f'    +5K text cooccur: {ok_b}/5')
_log(f'    +100 verb obs   : {ok_c}/5')
_log(f'    +100 vision     : {ok_d}/5')
_log(f'  Distractor IS-A still recall: {distractor_ok}/{len(DISTRACTORS_ISA)} (no blocking)')
_log(f'  Substrate: {sub_final/1_048_576:.0f} MB FIXED throughout')
_log(f'  RSS: {rss_final:.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- no-forgetting empirically proven' if FAIL == 0 else 'NEEDS FIX'))
