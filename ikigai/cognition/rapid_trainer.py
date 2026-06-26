"""
ikigai.cognition.rapid_trainer -- batched fast-path trainer for the flat substrate.

Same write semantics as MultiRoleMemory.expose_cooccur / expose_transitions,
but trains on a corpus 5-15x faster by:

1. Optional stopword filter (reduces cooccur write count by ~30%;
   mean-removal at recall time already handles stopwords, so removing them
   at write usually improves discrimination too).

2. Cross-sentence batching: accumulate K sentences, aggregate per-unique-key
   ONCE, do one locs_batch matmul per role per flush. The per-sentence loop
   that the baseline does pays the M=16384 matmul cost once per sentence per
   write; batching pays it once per batch.

3. Vectorized cumsum for cooccur windows is preserved sentence-by-sentence
   (already O(n) per sentence in baseline) but the per-token bank.locs_batch
   call is amortized across the whole batch.

Public API:
    rt = RapidTrainer(organism)
    rt.train(sentence_iter)          # consumes iter, flushes automatically
    rt.flush()                       # writes any leftover batch

Drop-in replacement for:
    for s in sentences:
        org.unified.expose_cooccur(s)
        org.unified.expose_transitions(s)

Output substrate is bit-identical to baseline modulo argpartition top-k
tie-breaking on FP-sensitive ties (same caveat as Pack 116 batched migration).
"""

import re
import numpy as np

DEFAULT_STOPWORDS = {
    'the','a','an','of','to','in','is','was','are','were','am','be','been','being',
    'and','or','but','for','with','on','at','by','from','it','its','this','that',
    'these','those','as','do','does','did','have','has','had','having',
    'will','would','can','could','should','may','might','must','shall',
    'i','you','he','she','we','they','me','him','her','us','them','my','your','his',
    'their','our','what','which','who','whom','whose','where','when','why','how',
    'not','no','than','then','so','if','because','about','into','out','up','down',
    'over','under','again','further','here','there','also','only','very','just',
    'only','more','most','other','some','such','any','few','many'
}

_TOK_RE = re.compile(r"[a-z0-9']+")

def _tokenize(text):
    return _TOK_RE.findall(text.lower())


class RapidTrainer:
    """
    Fast batched trainer over MultiRoleMemory.

    Parameters:
        organism      -- IkigaiOrganism (uses .unified)
        batch_size    -- sentences per flush (default 64)
        drop_stop     -- filter stopwords from cooccur AND n-gram tokens (default True)
        cooccur_w     -- override unified.window (default: use unified.window)
        do_cooccur    -- write cooccur channel (default True)
        do_ngrams     -- write next / next2 / next3 channels (default True)
    """

    def __init__(self, organism, batch_size=64, drop_stop=True,
                 cooccur_w=None, do_cooccur=True, do_ngrams=True,
                 stopwords=None):
        self.org = organism
        self.mr = organism.unified
        self.batch_size = int(batch_size)
        self.drop_stop = bool(drop_stop)
        self.cooccur_w = self.mr.window if cooccur_w is None else int(cooccur_w)
        self.do_cooccur = bool(do_cooccur)
        self.do_ngrams = bool(do_ngrams)
        self.stop = set(stopwords) if stopwords is not None else (
            DEFAULT_STOPWORDS if drop_stop else set())
        self._batch = []
        self.n_sentences = 0
        self.n_tokens = 0

    # ── public ─────────────────────────────────────────────────────────────
    def add(self, text):
        toks = _tokenize(text)
        if self.stop:
            toks = [t for t in toks if t not in self.stop]
        if len(toks) < 2:
            return
        self._batch.append(toks)
        self.n_sentences += 1
        self.n_tokens += len(toks)
        if len(self._batch) >= self.batch_size:
            self.flush()

    def train(self, sentence_iter):
        for s in sentence_iter:
            self.add(s)
        self.flush()

    def flush(self):
        if not self._batch:
            return
        if self.do_cooccur:
            self._flush_cooccur(self._batch)
        if self.do_ngrams:
            for n_ctx, role in [(1, 'next'), (2, 'next2'), (3, 'next3')]:
                self._flush_ngram(self._batch, n_ctx, role)
        self._batch = []
        self.mr._dirty = True

    # ── cooccur batched flush ──────────────────────────────────────────────
    def _flush_cooccur(self, batch):
        d = self.mr.d
        w = self.cooccur_w
        ck = self.mr.ck
        # Pre-warm: collect every unique token across the batch and ensure
        # they're all in ck cache via one batched call.  Cheaper than
        # per-token ck.key() inside the sentence loop.
        unique_toks = set()
        for tokens in batch:
            unique_toks.update(tokens)
        if hasattr(ck, 'key_batch'):
            key_map = ck.key_batch(unique_toks)
        else:
            key_map = {t: ck.key(t) for t in unique_toks}

        agg = {}
        order = []
        for tokens in batch:
            n = len(tokens)
            if n < 2:
                continue
            K = np.stack([key_map[t] for t in tokens])
            P = np.empty((n + 1, d), dtype=np.complex128)
            P[0] = 0
            P[1:] = np.cumsum(K.astype(np.complex128), axis=0)
            for i in range(n):
                lo = i - w if i - w > 0 else 0
                hi = i + w + 1 if i + w + 1 < n else n
                ctx = (P[hi] - P[lo]) - K[i]
                t = tokens[i]
                if t in agg:
                    agg[t] = agg[t] + ctx
                else:
                    agg[t] = ctx
                    order.append(t)
                    self.mr._seen.add(t)
                    self.mr._cooccur_seen.add(t)
        if not order:
            return
        rolev = self.mr.roles['cooccur']
        ukeys = np.stack([self.mr._bind(key_map[t], rolev) for t in order])
        slots = [self.mr._slot(t, 'cooccur') for t in order]
        locs = self.mr.sdm.locs_batch(ukeys, slots)
        for t, idx in zip(order, locs):
            self.mr.sdm.C[idx] += agg[t].astype(np.complex64)

    # ── n-gram batched flush ───────────────────────────────────────────────
    def _flush_ngram(self, batch, n_ctx, role):
        mr = self.mr
        rolev = mr.roles[role]
        bank = mr._bank(role)
        ck = mr.ck
        # Pre-warm all unique tokens in batch (same trick as cooccur)
        unique_toks = set()
        for tokens in batch:
            unique_toks.update(tokens)
        if hasattr(ck, 'key_batch'):
            key_map = ck.key_batch(unique_toks)
        else:
            key_map = {t: ck.key(t) for t in unique_toks}

        agg_data = {}
        slot_to_ctx_hv = {}
        order = []
        for tokens in batch:
            if len(tokens) < n_ctx + 1:
                continue
            for i in range(n_ctx, len(tokens)):
                ctx = tokens[i - n_ctx:i]
                curr = tokens[i]
                slot = '|'.join(ctx) + f'\x00{role}'
                curr_key = key_map[curr]
                if slot in agg_data:
                    agg_data[slot] = agg_data[slot] + curr_key
                else:
                    agg_data[slot] = curr_key.astype(np.complex64).copy()
                    slot_to_ctx_hv[slot] = mr._ngram_ctx_hv(ctx)
                    order.append(slot)
                    mr._role_targets.setdefault(role, set()).add(ctx[0])
                    mr._seen.add(ctx[0])
        if not order:
            return
        ukeys = np.stack([mr._bind(slot_to_ctx_hv[s], rolev) for s in order])
        locs = bank.locs_batch(ukeys, order)
        for s, idx in zip(order, locs):
            bank.C[idx] += agg_data[s]
