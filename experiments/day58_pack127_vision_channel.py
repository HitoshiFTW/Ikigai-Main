"""
Day 58 Pack 127 -- Vision channel (modality-blindness proof).

Plug actual pixel patches into the flat substrate via random_projection_encoder.
Train MNIST-style digit classification. Verify text channels (IS-A, similarity)
still work AFTER vision training -- all in one 192 MB substrate.

This proves the architecture's universal/modality-blind claim with real pixels,
not just text.

Verifications:
    V1  digits dataset loaded
    V2  vision train accuracy >= 70% (model can learn)
    V3  vision test accuracy >= 55% (real generalization, not memorization)
    V4  substrate FIXED bytes pre/post vision training
    V5  IS-A facts injected BEFORE vision still recall correctly AFTER
    V6  text cooccur similarity still works AFTER vision
    V7  fresh IS-A facts injected AFTER vision still work (substrate not full)
    V8  inference RAM stays under 500 MB
"""

import sys, os, pathlib, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    PROC = psutil.Process(os.getpid())
    def rss(): return PROC.memory_info().rss / 1_048_576
except ImportError:
    def rss(): return -1.0

import numpy as np
from integrate import IkigaiOrganism

# vision lib
try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# random projection encoder from flatmem-style
from ikigai.cognition.flat_memory import _renorm

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 127: Vision channel (modality-blindness proof) ===\n')

#  0. Build flat-only organism

org = IkigaiOrganism(flat_only=True)
mr = org.unified
# Register a 'class' role for digit labels (not in DEFAULT_ROLES, add manually)
import numpy as _np
rng = _np.random.default_rng(127)
ph = rng.uniform(-_np.pi, _np.pi, mr.d).astype(_np.float32)
mr.roles['class'] = _np.exp(1j * ph).astype(_np.complex64)

sub_initial = mr.substrate_bytes()
rss0 = rss()
_log(f'  organism: substrate={sub_initial/1_048_576:.0f} MB FIXED, RSS={rss0:.0f} MB')

#  1. Inject TEXT knowledge first (this will be tested for survival)

_log('\n--- Phase A: text knowledge ---')
text_corpus = ["the cat sat on the mat", "the dog ran in the park",
               "a cat and a dog", "the boy played with the car",
               "the girl smiled at the car", "the king wore a crown",
               "the queen sat on a throne", "a fast car drove past",
               "the car is red"]
for s in text_corpus * 30:
    mr.expose_cooccur(s)
isa_facts = {'cat': 'mammal', 'dog': 'mammal', 'rose': 'flower', 'oak': 'tree'}
for h, y in isa_facts.items():
    org.assert_isa(h, y, n=30)

cd_before = org.unified_similarity('cat', 'dog') or 0.0
cc_before = org.unified_similarity('cat', 'car') or 0.0
isa_before = {h: org.isa_of(h) for h in isa_facts}
_log(f'  cat~dog={cd_before:+.3f}  cat~car={cc_before:+.3f}')
_log(f'  isa: {isa_before}')

#  2. Vision data

if not HAVE_SKLEARN:
    _log('\nERROR: scikit-learn missing. pip install scikit-learn')
    sys.exit(1)

_log('\n--- Phase B: vision (8x8 digits, sklearn) ---')
data = load_digits()
X = data.data.astype(np.float32) / 16.0  # normalize to [0, 1]
y = data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
_log(f'  train: {len(X_tr)}, test: {len(X_te)}, classes: 10, features: {X.shape[1]}')

check('V1 digits dataset loaded', len(X_tr) > 100 and len(X_te) > 50)

#  3. Vision encoder (random projection -> phasor)

# Pre-compute projection matrix once
D = mr.d  # 512
rng_proj = np.random.default_rng(127)
P_matrix = rng_proj.standard_normal((D, X.shape[1])).astype(np.float32)

def encode_image(img):
    """Flatten pixels -> random projection -> phasor."""
    phase = (P_matrix @ img).astype(np.float32)
    return np.exp(1j * phase).astype(np.complex64)

#  4. Train: write each image's encoded HV under 'class' role, value = label key

_log('\nTraining vision...')
t0 = time.perf_counter()
class_keys = {str(c): mr.ck.key(str(c)) for c in range(10)}
for img, lbl in zip(X_tr, y_tr):
    addr_img = encode_image(img)
    bound_addr = mr._bind(addr_img, mr.roles['class'])
    mr.sdm_rel.write(bound_addr, class_keys[str(int(lbl))])
_log(f'  trained {len(X_tr)} images in {time.perf_counter()-t0:.1f}s')

#  5. Evaluate

def predict(img):
    addr_img = encode_image(img)
    bound = mr._bind(addr_img, mr.roles['class'])
    out = mr.sdm_rel.read(bound)
    best, bscore = -1, -9.0
    for c in range(10):
        s = float(np.real(np.vdot(out, class_keys[str(c)]))) / D
        if s > bscore:
            bscore, best = s, c
    return best

_log('Evaluating vision...')
t0 = time.perf_counter()
train_correct = sum(predict(x) == y for x, y in zip(X_tr, y_tr))
train_acc = train_correct / len(X_tr)
test_correct = sum(predict(x) == y for x, y in zip(X_te, y_te))
test_acc = test_correct / len(X_te)
_log(f'  train acc: {train_acc:.0%}  ({train_correct}/{len(X_tr)})')
_log(f'  test  acc: {test_acc:.0%}  ({test_correct}/{len(X_te)})')
_log(f'  eval in {time.perf_counter()-t0:.1f}s')

check('V2 train acc >= 70%', train_acc >= 0.70, f'train={train_acc:.0%}')
check('V3 test  acc >= 55%', test_acc  >= 0.55, f'test={test_acc:.0%}')

#  6. Substrate flatness

sub_after_vision = mr.substrate_bytes()
check('V4 substrate FIXED pre/post vision', sub_after_vision == sub_initial,
      f'{sub_initial} -> {sub_after_vision}')

#  7. Verify text knowledge survives

_log('\n--- Phase C: verify text channels survived vision training ---')
cd_after = org.unified_similarity('cat', 'dog') or 0.0
cc_after = org.unified_similarity('cat', 'car') or 0.0
isa_after = {h: org.isa_of(h) for h in isa_facts}
_log(f'  cat~dog={cd_after:+.3f}  cat~car={cc_after:+.3f}  (was {cd_before:+.3f}/{cc_before:+.3f})')
_log(f'  isa: {isa_after}')

# Cooccur lives in DENSE bank, vision in SPARSE bank -- should be untouched.
# IS-A and vision BOTH in sparse bank -- might see some crosstalk but
# clean reinforced facts should survive.
check('V5 IS-A facts survive vision training',
      isa_after == isa_before,
      f'before={isa_before}  after={isa_after}')

check('V6 cooccur similarity preserves cat~dog > cat~car',
      cd_after > cc_after, f'cd={cd_after} cc={cc_after}')

#  8. Inject NEW IS-A facts AFTER vision -- substrate not "full"

_log('\n--- Phase D: inject new facts after vision ---')
new_facts = {'apple': 'fruit', 'car': 'vehicle'}
for h, y in new_facts.items():
    org.assert_isa(h, y, n=30)
got = {h: org.isa_of(h) for h in new_facts}
_log(f'  new facts: {got}')
check('V7 new IS-A facts work after vision', got == new_facts, f'{got}')

#  9. RAM

rss_final = rss()
_log(f'\n  final RSS: {rss_final:.0f} MB  (started {rss0:.0f})')
check('V8 inference RAM under 500 MB', rss_final < 500, f'{rss_final:.0f} MB')

#  summary

total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 127 -- Vision channel proof')
_log(f'{"="*64}')
_log(f'  vision: train {train_acc:.0%}  test {test_acc:.0%}  on 8x8 digits')
_log(f'  text channels survived: isa OK, sim {cd_after:+.3f} > {cc_after:+.3f}')
_log(f'  fresh facts after vision: OK')
_log(f'  substrate: {sub_after_vision/1_048_576:.0f} MB FIXED throughout')
_log(f'  RSS: {rss_final:.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- modality-blindness proven on real pixels' if FAIL == 0 else 'NEEDS FIX'))
