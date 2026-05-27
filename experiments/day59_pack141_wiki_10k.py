"""
Day 59 Pack 141 -- Scale Pack 137 to 10K Simple English Wikipedia articles.

Pack 137 trained 200 articles in 67.7s with RapidTrainer.  Extrapolating
to 10K -> ~57 min on CPU.  Goal: dense trigram + 4-gram coverage (~500K
contexts each) so cogitate output starts being coherent.

The trained substrate is saved to checkpoints/pack141_wiki10k.pkl so
Packs 142+ can load it without re-running training.

Verifications:
    V1  substrate stays FIXED across 1K / 5K / 10K milestones
    V2  trigram seen-prevs > 100K
    V3  4-gram seen-prevs  > 100K
    V4  next_word_candidates returns sensible vocab for common prompts
    V5  cooccur sims POSITIVE for well-known related pairs from Wikipedia
    V6  cogitate output uses vocab tokens (basic sanity)
    V7  save/load round-trip preserves substrate exactly
    V8  saved checkpoint < 500 MB on disk (Pack 125 baseline 161 MB at 100K stories)
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

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

N_ARTICLES = 10000
LOCAL_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_pack141_simple_wiki_10k.jsonl')
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'checkpoints', 'pack141_wiki10k.pkl')

_log('=== Pack 141: Wikipedia training at 10K-article scale ===\n')
_log(f'  RSS at start: {rss():.0f} MB')

# ---- phase 0: extract once to local jsonl (HF library released after) ----
have_n = 0
if os.path.exists(LOCAL_DUMP):
    have_n = sum(1 for _ in open(LOCAL_DUMP, encoding='utf-8'))
if have_n < N_ARTICLES:
    _log(f'  [phase 0] extracting {N_ARTICLES} articles to {LOCAL_DUMP} '
         f'(have {have_n})')
    from datasets import load_dataset
    ds = load_dataset('wikimedia/wikipedia', '20231101.simple',
                      split='train', streaming=True)
    t0 = time.perf_counter()
    written = 0
    with open(LOCAL_DUMP, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(ds):
            if i >= N_ARTICLES: break
            f.write(json.dumps({'title': ex['title'], 'text': ex['text']}) + '\n')
            written += 1
            if (i + 1) % 1000 == 0:
                _log(f'    extracted {i+1}/{N_ARTICLES}  '
                     f'({time.perf_counter()-t0:.1f}s, file='
                     f'{os.path.getsize(LOCAL_DUMP)/1_048_576:.1f}MB)')
    _log(f'    extracted {written} articles in '
         f'{time.perf_counter()-t0:.1f}s, file='
         f'{os.path.getsize(LOCAL_DUMP)/1_048_576:.1f}MB')
    del ds
    import datasets as _ds; del _ds
    sys.modules.pop('datasets', None)
    gc.collect()
else:
    _log(f'  [phase 0] reusing cached {LOCAL_DUMP} ({have_n} articles)')

_log(f'  RSS after extract: {rss():.0f} MB')

# ---- phase 1: train via RapidTrainer ----
from integrate import IkigaiOrganism
from ikigai.cognition.flat_trainer import FlatTrainer

def sentences(text):
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.split()) >= 3]

org = IkigaiOrganism(flat_only=True)
rt = FlatTrainer(org, batch_size=128, drop_stop=True,
                 compact_threshold=8000)
sub_before = org.unified.substrate_bytes()
rss_peak = rss()
_log(f'\n  starting substrate={sub_before/1_048_576:.0f} MB  RSS={rss():.0f} MB')
_log(f'  trainer: FlatTrainer batch_size=256 cache_cap=8192 flush_keys=20000')

milestones = {}
t0 = time.perf_counter()
with open(LOCAL_DUMP, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= N_ARTICLES: break
        ex = json.loads(line)
        for s in sentences(ex['text']):
            rt.add(s)
        if (i + 1) % 250 == 0:
            r_now = rss()
            if r_now > rss_peak: rss_peak = r_now
            _log(f'    .. {i+1:5d}/{N_ARTICLES}  sents={rt.n_sentences}  '
                 f'toks={rt.n_tokens}  '
                 f't={time.perf_counter()-t0:.1f}s  RSS={r_now:.0f}MB  '
                 f'peak={rss_peak:.0f}MB')
        if (i + 1) in (1000, 5000, 10000):
            rt.flush()
            milestones[i + 1] = {
                'substrate':    org.unified.substrate_bytes(),
                'rss':          rss(),
                'sentences':    rt.n_sentences,
                'tokens':       rt.n_tokens,
                'trigram_seen': len(org.unified._role_targets.get('next2', set())),
                'fourgram_seen':len(org.unified._role_targets.get('next3', set())),
            }
            _log(f'    @ {i+1:5d}: substrate={milestones[i+1]["substrate"]/1_048_576:.0f}MB '
                 f'RSS={milestones[i+1]["rss"]:.0f}MB  '
                 f'trigram={milestones[i+1]["trigram_seen"]}  '
                 f'4gram={milestones[i+1]["fourgram_seen"]}')
rt.flush()
rt.restore_caches()

elapsed = time.perf_counter() - t0
sub_after = org.unified.substrate_bytes()
n_sentences = rt.n_sentences
n_tokens = rt.n_tokens
_log(f'\n  trained {n_sentences} sents / {n_tokens} toks in '
     f'{elapsed:.1f}s ({n_tokens/elapsed:.0f} tok/s)')

# ---- V1 substrate FIXED ----
fixed = all(m['substrate'] == sub_before for m in milestones.values())
check('V1 substrate FIXED across 1K/5K/10K milestones', fixed,
      f'{sub_before} -> {sub_after}')

# ---- V2 / V3 channel coverage ----
trigram_seen  = milestones[max(milestones)]['trigram_seen']
fourgram_seen = milestones[max(milestones)]['fourgram_seen']
_log(f'\n  trigram seen-prevs: {trigram_seen}  (Pack 137 N=200 baseline 11,873)')
_log(f'  4-gram  seen-prevs: {fourgram_seen}  (Pack 137 N=200 baseline 11,093)')
check('V2 trigram coverage > 100K', trigram_seen > 100_000,
      f'got {trigram_seen}')
check('V3 4-gram coverage > 100K', fourgram_seen > 100_000,
      f'got {fourgram_seen}')

# ---- V4 vocab sanity ----
top_the = org.unified.next_word_candidates('the', top_k=10)
_log(f'\n  next("the") top-10: {[(w, round(s,3)) for w,s in top_the[:10]]}')
check('V4 next_word_candidates returns vocab', len(top_the) > 0)

# ---- V5 cooccur sims on common pairs ----
pairs = [('city', 'country'), ('year', 'month'), ('water', 'air'),
         ('king', 'queen'), ('north', 'south'), ('first', 'second'),
         ('book', 'author'), ('country', 'capital'), ('film', 'movie')]
sims = {}
for a, b in pairs:
    if a in org.unified._cooccur_seen and b in org.unified._cooccur_seen:
        sims[(a, b)] = float(org.unified.similarity(a, b))
_log(f'\n  related-pair cooccur sims:')
for k, v in sims.items():
    _log(f'    {k[0]:10s} ~ {k[1]:10s} : {v:+.3f}')
n_pos = sum(1 for v in sims.values() if v > 0)
check(f'V5 most pairs positive ({n_pos}/{len(sims)})',
      n_pos >= len(sims) * 0.6,
      f'positives {n_pos}/{len(sims)}')

# ---- V6 cogitate basic ----
out = org.cogitate('the city of', max_tokens=25, seed=0, temperature=0.7)
_log(f'\n  cogitate(the city of): "{out}"')
out_toks = out.split()
in_vocab = sum(1 for t in out_toks if t in org.unified._cooccur_seen)
check('V6 cogitate emits in-vocab tokens',
      in_vocab / len(out_toks) > 0.5,
      f'in-vocab {in_vocab}/{len(out_toks)}')

# ---- V7 save / load round trip ----
os.makedirs(os.path.dirname(CKPT), exist_ok=True)
org.save(CKPT)
ckpt_size = os.path.getsize(CKPT) / 1_048_576
_log(f'\n  checkpoint saved: {ckpt_size:.1f} MB  ({CKPT})')

org2 = IkigaiOrganism.load(CKPT, flat_only=True)
sub_loaded = org2.unified.substrate_bytes()
sim_a = float(org.unified.similarity('king', 'queen'))
sim_b = float(org2.unified.similarity('king', 'queen'))
check('V7 save/load preserves substrate', sub_loaded == sub_after and abs(sim_a - sim_b) < 1e-4,
      f'sub {sub_loaded} sim_a {sim_a:.4f} sim_b {sim_b:.4f}')

check('V8 checkpoint under 500 MB on disk', ckpt_size < 500.0,
      f'{ckpt_size:.1f} MB')

# ---- summary ----
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 141 -- Wikipedia 10K-scale training')
_log(f'{"="*64}')
_log(f'  Trained: {N_ARTICLES} articles / {n_sentences} sents / {n_tokens} toks')
_log(f'  Rate: {n_tokens/elapsed:.0f} tok/s  ({elapsed/60:.1f} min total)')
_log(f'  Substrate: {sub_after/1_048_576:.0f} MB FIXED through everything')
_log(f'  Trigram coverage: {trigram_seen:,} contexts')
_log(f'  4-gram coverage:  {fourgram_seen:,} contexts')
_log(f'  Checkpoint:       {ckpt_size:.1f} MB on disk')
_log(f'  Final RSS: {rss():.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- 10K Wikipedia absorbed flat'
                     if FAIL == 0 else 'NEEDS FIX'))
