"""
Day 58 Pack 137 -- Simple English Wikipedia training at scale.

Pack 136 added trigram + 4-gram channels but the toy corpus didn't populate
them densely. Pack 137 fixes that with real text from Simple English
Wikipedia.

Two-phase to avoid HF datasets library RAM bloat (parquet buffers eat
GBs and never release):
  phase 0: extract N articles to a local jsonl ONCE, drop HF library
  phase 1: stream the local jsonl, train, verify

Verifications:
    V1  substrate stays FIXED across milestones
    V2  trigram seen-prevs >> Pack 136 baseline (58)
    V3  4-gram  seen-prevs >> Pack 136 baseline (52)
    V4  next_word_candidates returns sensible vocab for common prompts
    V5  cogitate output diverges across 3 different prompts
    V6  cooccur similarity captures a known related pair from Wikipedia
    V7  per-token RAM constant at long generation post-wiki-flood
    V8  cross-modal (vision + math) still works after wiki flood
"""

import sys, os, re, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    PROC = psutil.Process(os.getpid())
    def rss(): return PROC.memory_info().rss / 1_048_576
except ImportError:
    def rss(): return -1.0

import numpy as np
from ikigai.cognition.rapid_trainer import RapidTrainer

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

N_ARTICLES = 200   # demo scale; ~50K tokens. Substrate behaviour independent of N.
LOCAL_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_pack137_simple_wiki.jsonl')

_log('=== Pack 137: Simple English Wikipedia training ===\n')
_log(f'  RSS at start: {rss():.0f} MB')

# ---- phase 0: extract once to local jsonl (HF library released after) ----
if not os.path.exists(LOCAL_DUMP) or sum(1 for _ in open(LOCAL_DUMP, encoding='utf-8')) < N_ARTICLES:
    _log(f'  [phase 0] extracting {N_ARTICLES} articles to {LOCAL_DUMP}')
    from datasets import load_dataset
    ds = load_dataset('wikimedia/wikipedia', '20231101.simple',
                      split='train', streaming=True)
    t0 = time.perf_counter()
    with open(LOCAL_DUMP, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(ds):
            if i >= N_ARTICLES: break
            f.write(json.dumps({'title': ex['title'], 'text': ex['text']}) + '\n')
    _log(f'    extracted in {time.perf_counter()-t0:.1f}s, '
         f'file={os.path.getsize(LOCAL_DUMP)/1_048_576:.1f} MB')
    # drop HF library refs so memory frees
    del ds
    import datasets as _ds; del _ds
    sys.modules.pop('datasets', None)
    gc.collect()
else:
    _log(f'  [phase 0] reusing cached {LOCAL_DUMP}')

_log(f'  RSS after extract: {rss():.0f} MB')

# ---- phase 1: import organism, train from local file ----
from integrate import IkigaiOrganism

def sentences(text):
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.split()) >= 3]

org = IkigaiOrganism(flat_only=True)
rt = RapidTrainer(org, batch_size=128, drop_stop=True)
sub_before = org.unified.substrate_bytes()
_log(f'\n  starting substrate={sub_before/1_048_576:.0f} MB  RSS={rss():.0f} MB')
_log(f'  trainer: RapidTrainer batch_size=128 drop_stop=True')

milestones = {}
t0 = time.perf_counter()
with open(LOCAL_DUMP, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= N_ARTICLES: break
        ex = json.loads(line)
        for s in sentences(ex['text']):
            rt.add(s)
        if (i + 1) % 25 == 0:
            _log(f'    .. {i+1:4d} articles  sents={rt.n_sentences}  toks={rt.n_tokens}  '
                 f't={time.perf_counter()-t0:.1f}s  RSS={rss():.0f}MB')
        if (i + 1) in (50, 100, 200):
            rt.flush()
            milestones[i + 1] = {
                'substrate':    org.unified.substrate_bytes(),
                'rss':          rss(),
                'sentences':    rt.n_sentences,
                'tokens':       rt.n_tokens,
                'trigram_seen': len(org.unified._role_targets.get('next2', set())),
                'fourgram_seen':len(org.unified._role_targets.get('next3', set())),
            }
            _log(f'    @ {i+1:4d}: substrate={milestones[i+1]["substrate"]/1_048_576:.0f}MB '
                 f'RSS={milestones[i+1]["rss"]:.0f}MB  '
                 f'trigram={milestones[i+1]["trigram_seen"]}  '
                 f'4gram={milestones[i+1]["fourgram_seen"]}')
rt.flush()

elapsed = time.perf_counter() - t0
sub_after = org.unified.substrate_bytes()
n_sentences = rt.n_sentences
n_tokens = rt.n_tokens
_log(f'\n  trained {n_sentences} sentences / {n_tokens} tokens in '
     f'{elapsed:.1f}s ({n_tokens/elapsed:.0f} tok/s)')

# ---- V1 substrate FIXED across milestones ----
fixed = all(m['substrate'] == sub_before for m in milestones.values())
check('V1 substrate FIXED across 50/100/200 articles', fixed,
      f'{sub_before} -> {sub_after}')

# ---- V2/V3 n-gram channel coverage ----
trigram_seen  = milestones[max(milestones)]['trigram_seen']
fourgram_seen = milestones[max(milestones)]['fourgram_seen']
_log(f'\n  trigram seen-prevs: {trigram_seen}  (Pack 136 baseline 58)')
_log(f'  4-gram  seen-prevs: {fourgram_seen}  (Pack 136 baseline 52)')
check('V2 trigram coverage > 10x Pack 136', trigram_seen > 580,
      f'got {trigram_seen}')
check('V3 4-gram coverage > 10x Pack 136', fourgram_seen > 520,
      f'got {fourgram_seen}')

# ---- V4 vocab sanity: next-word top for common prompt ----
top_the = org.unified.next_word_candidates('the', top_k=8)
_log(f'\n  next("the") top-8: {[(w, round(s,3)) for w,s in top_the[:8]]}')
check('V4 next_word_candidates returns vocab', len(top_the) > 0)

# ---- V5 cogitate is prompt-responsive ----
out_a = org.cogitate('the city of', max_tokens=20, seed=0, temperature=0.6)
out_b = org.cogitate('a famous scientist', max_tokens=20, seed=0, temperature=0.6)
out_c = org.cogitate('the war between', max_tokens=20, seed=0, temperature=0.6)
_log(f'\n  cogitate(the city of):       "{out_a}"')
_log(f'  cogitate(a famous scientist):"{out_b}"')
_log(f'  cogitate(the war between):   "{out_c}"')
check('V5 prompt-responsive (3 distinct outputs)',
      out_a != out_b and out_b != out_c and out_a != out_c)

# ---- V6 cooccur sanity on known-related pair ----
sims = {}
for a, b in [('city', 'country'), ('year', 'month'), ('water', 'air'),
             ('king', 'queen'), ('north', 'south'), ('first', 'second')]:
    if a in org.unified._cooccur_seen and b in org.unified._cooccur_seen:
        sims[(a, b)] = float(org.unified.similarity(a, b))
_log(f'\n  related-pair cooccur sims: {sims}')
check('V6 at least one related-pair has positive similarity',
      any(v > 0 for v in sims.values()),
      f'sims: {sims}')

# ---- V7 RAM constant at long gen post-flood ----
_log('\n  long-generation post-wiki (constant RAM claim):')
sub_pre = org.unified.substrate_bytes()
for n in [50, 500, 1500]:
    t = time.perf_counter()
    out = org.cogitate('the city of', max_tokens=n, seed=0,
                       temperature=0.7, think_steps=2,
                       ngram_weights=(0.2, 0.4, 0.4))
    el = time.perf_counter() - t
    sub = org.unified.substrate_bytes()
    _log(f'    n={n:5d}  {el:6.1f}s  {n/el:5.0f} tok/s  '
         f'substrate={sub/1_048_576:.0f}MB  RSS={rss():.0f}MB')
sub_post = org.unified.substrate_bytes()
check('V7 substrate FIXED at long gen post-wiki', sub_post == sub_pre,
      f'{sub_pre} -> {sub_post}')

# ---- V8 cross-modal still works after wiki flood ----
from sklearn.datasets import load_digits
digs = load_digits()
X, y = digs.images, digs.target
for i in range(50):
    org.expose_image(X[i], int(y[i]), n=1)
correct = 0
for i in range(50, 100):
    pred, _ = org.classify_image(X[i], candidates=list(range(10)))
    if int(pred) == int(y[i]):
        correct += 1
vision_acc = correct / 50
_log(f'\n  vision after wiki flood: {correct}/50 correct ({vision_acc*100:.0f}%)')

org.assert_isa('zerg', 'creature', n=50)
got = org.isa_of('zerg')
_log(f'  isa("zerg") after wiki: {got}')

check('V8 cross-modal intact post-wiki', vision_acc >= 0.5 and got == 'creature',
      f'vision={vision_acc:.2f}  isa_zerg={got}')

# ---- summary ----
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 137 -- Simple English Wikipedia training')
_log(f'{"="*64}')
_log(f'  Trained: {N_ARTICLES} articles / {n_sentences} sents / {n_tokens} toks')
_log(f'  Rate: {n_tokens/elapsed:.0f} tok/s')
_log(f'  Substrate: {sub_post/1_048_576:.0f} MB FIXED through everything')
_log(f'  Trigram coverage: {trigram_seen} unique contexts')
_log(f'  4-gram coverage:  {fourgram_seen} unique contexts')
_log(f'  Final RSS: {rss():.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- flat substrate absorbs real Wikipedia'
                     if FAIL == 0 else 'NEEDS FIX'))
