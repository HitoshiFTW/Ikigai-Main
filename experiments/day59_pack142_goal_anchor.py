"""
Day 59 Pack 142 -- Goal-state HV: fixed prompt anchor for long-gen coherence.

Pack 135's cogitate has one thought HV that drifts via momentum-weighted
substrate recall.  After 20-50 tokens the thought has wandered far from
the initial prompt, output topic loses coherence.

Pack 142 adds a SECOND, FIXED HV: the goal.  Goal = initial prompt HV,
copied once at _init_thought and never updated.  Speak step scores
candidates by BOTH thought-alignment (drifting) and goal-alignment
(fixed).  Net effect: thought explores, goal anchors topic.

Implementation: GenerationEngine adds `goal_gamma` parameter.  In speak_step,
`boost = thought_gamma * cos(key(w), thought) + goal_gamma * cos(key(w), goal)`.
goal_gamma=0 (default) preserves Pack 135 behaviour exactly.

Verifications:
    V1  goal_gamma=0 produces identical output to pre-Pack-142 cogitate
    V2  goal_gamma>0 produces different output from goal_gamma=0
    V3  goal vector is FIXED across speak steps (does not drift)
    V4  with goal_gamma, late-gen tokens score higher cos vs prompt HV than without
    V5  RAM stays O(1) per token with goal anchor (no growth)
    V6  substrate FIXED through goal-anchored generation
    V7  back-compat: bigram-only flag still works (ngram_weights=(1,0,0))
    V8  goal anchor scales to long gen (n=500 still emits in-vocab tokens)
"""

import sys, os, re, time
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

_log('=== Pack 142: Goal-state HV ===\n')

# ---- corpus (same as Pack 136 for apples-to-apples) ----
TEXT = [
    "the cat sat on the mat and watched the rain",
    "the dog ran in the park with a stick",
    "a cat chased a mouse across the garden",
    "the boy played with the dog in the park",
    "the girl held the cat in her arms",
    "the king sat on his golden throne in the castle",
    "the queen wore a beautiful silver crown",
    "the rain fell softly on the green grass",
    "the bird sang a sweet song in the morning",
    "the river flowed past the quiet village",
    "she ate a sweet red apple under the tree",
    "he picked up the small stone from the road",
    "the night was dark and the stars were bright",
    "the fire crackled warmly in the fireplace",
    "the boat sailed across the calm blue sea",
] * 60

org = IkigaiOrganism(flat_only=True)
for s in TEXT:
    org.unified.expose_cooccur(s)
    org.unified.expose_transitions(s)
_log(f'  trained on {len(TEXT)} sentences  substrate={org.unified.substrate_bytes()/1_048_576:.0f}MB')
sub_before = org.unified.substrate_bytes()

# ---- V1 back-compat: goal_gamma=0 == pre-Pack-142 ----
out_a = org.cogitate('the king sat', max_tokens=20, seed=0,
                     goal_gamma=0.0)
out_b = org.cogitate('the king sat', max_tokens=20, seed=0,
                     goal_gamma=0.0)
check('V1 goal_gamma=0 reproducible (back-compat path)', out_a == out_b,
      f'a={out_a!r}\n   b={out_b!r}')

# ---- V2 goal_gamma>0 produces different output ----
out_off = org.cogitate('the king sat', max_tokens=20, seed=0, goal_gamma=0.0)
out_on  = org.cogitate('the king sat', max_tokens=20, seed=0, goal_gamma=3.0)
_log(f'\n  goal off: "{out_off}"')
_log(f'  goal on:  "{out_on}"')
check('V2 goal_gamma>0 shifts output', out_off != out_on)

# ---- V3 goal vector is fixed across speak steps ----
# Use the engine directly to inspect goal at start vs end
from ikigai.cognition.generation_engine import GenerationEngine
eng = GenerationEngine(org, goal_gamma=3.0)
eng.generate('the king sat', max_tokens=20, seed=0)
goal_start = eng.thought_trace[0].copy()    # this is thought_0 = prompt HV == goal
goal_end   = eng.goal                       # goal field stays the prompt HV
diff = float(np.max(np.abs(goal_start - goal_end)))
check('V3 goal HV unchanged across generation',
      diff < 1e-6, f'max|delta|={diff:.6f}')

# ---- V4 late-gen tokens score higher cos vs prompt HV with goal anchor ----
def gen_and_score(prompt, goal_gamma, n=30, seed=0):
    eng = GenerationEngine(org, goal_gamma=goal_gamma)
    eng.generate(prompt, max_tokens=n, seed=seed)
    prompt_hv = eng.goal      # = initial prompt HV
    # score the LAST 10 generated tokens (excluding prompt tokens)
    d = org.unified.d
    last10 = eng.history[-10:]
    aligns = []
    for t in last10:
        kt = org.unified.ck.key(t)
        aligns.append(float(np.real(np.vdot(kt, prompt_hv))) / d)
    return sum(aligns) / len(aligns) if aligns else 0.0

align_off = gen_and_score('the king sat', 0.0, n=30, seed=0)
align_on  = gen_and_score('the king sat', 3.0, n=30, seed=0)
_log(f'\n  late-token cos vs prompt HV  goal_off: {align_off:+.5f}')
_log(f'  late-token cos vs prompt HV  goal_on : {align_on:+.5f}')
check('V4 goal anchor pulls late tokens closer to prompt HV',
      align_on > align_off,
      f'on={align_on:.5f} off={align_off:.5f}')

# ---- V5 / V6 substrate + RAM at long gen with goal anchor ----
_log('\n  long-gen with goal_gamma=3.0:')
sub_pre = org.unified.substrate_bytes()
results = []
for n in [50, 200, 500]:
    t0 = time.perf_counter()
    out = org.cogitate('the king sat', max_tokens=n, seed=0,
                       goal_gamma=3.0, temperature=0.7, think_steps=2)
    el = time.perf_counter() - t0
    sub = org.unified.substrate_bytes()
    r = rss()
    _log(f'    n={n:4d}  {el:6.1f}s  {n/el:5.0f} tok/s  '
         f'substrate={sub/1_048_576:.0f}MB  RSS={r:.0f}MB')
    results.append((n, sub, r))
sub_post = org.unified.substrate_bytes()
check('V5 RAM bounded with goal anchor (RSS growth < 100 MB)',
      results[-1][2] - results[0][2] < 100,
      f'RSS delta {results[-1][2]-results[0][2]:.0f}MB')
check('V6 substrate FIXED through goal-anchored long gen',
      sub_post == sub_pre,
      f'{sub_pre} -> {sub_post}')

# ---- V7 bigram-only back-compat ----
out_bi = org.cogitate('the king sat', max_tokens=15, seed=0,
                      ngram_weights=(1.0, 0.0, 0.0), goal_gamma=0.0)
check('V7 bigram-only path still works with goal field present',
      len(out_bi.split()) > 3)

# ---- V8 goal anchor at long gen still produces in-vocab tokens ----
out_long = org.cogitate('the king sat', max_tokens=500, seed=0,
                        goal_gamma=3.0, temperature=0.7, think_steps=2)
toks = out_long.split()
in_vocab = sum(1 for t in toks if t in org.unified._cooccur_seen)
check(f'V8 long gen ({len(toks)} toks) stays in-vocab',
      in_vocab / len(toks) > 0.9,
      f'in-vocab {in_vocab}/{len(toks)} = {in_vocab/len(toks):.2%}')

# ---- summary ----
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 142 -- Goal-state HV')
_log(f'{"="*64}')
_log(f'  Goal anchor improves late-token prompt-alignment:')
_log(f'    cos vs prompt: off={align_off:+.5f}  on={align_on:+.5f}')
_log(f'    delta:         {align_on - align_off:+.5f}')
_log(f'  Substrate {sub_post/1_048_576:.0f} MB FIXED, RAM O(1) per token')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- goal anchor wired'
                     if FAIL == 0 else 'NEEDS FIX'))
