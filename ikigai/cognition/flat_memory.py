"""
ikigai.cognition.flat_memory -- Flat lifelong memory (VSA-addressed SDM).

Day 57 Packs 114-115. The central scaling invention.

Biological memory is FLAT: fixed substrate, superposed knowledge, computed
identity, lossy reconstructive recall. The brain does not add bytes per
experience -- it changes synaptic strengths in a fixed matrix. This module
replaces the unbounded {word: HV} dict (grows O(vocab), Heap's law) with a
constant-size substrate whose footprint never grows, no matter how much data
flows in.

Three pillars:
    1. ComputedKey  -- word identity = seeded phasor projection of char
                       trigrams. ZERO storage per word, infinite vocabulary.
    2. VSASDM       -- Sparse Distributed Memory (Kanerva 1988, cerebellum
                       model). M fixed hard locations + counter bank. Sparse
                       write (top-k nearest). FIXED size forever.
    3. Adaptive     -- recall = read counters at activated locations, then
       recall          remove top-r common directions ("all-but-the-top",
                       Arora 2017). Sensory adaptation kills the shared
                       stopword baseline. Read-time adaptation > write-time
                       gating (Pack 115 finding: IDF write-weighting HURT).

Universal interface: write(addr, data) / read(addr). Modality-blind --
any ALife organism (text, sensorimotor, chemical) plugs into the same
substrate.

Pack 114: flat proven (vocab 0->14,429, substrate +0 bytes).
Pack 115: discrimination sep 0.026 -> 0.371 via mean-removal.
"""

import re
import hashlib

import numpy as np


def tokenize(text):
    # Pack 199 -- Unicode-aware: any letter from any script (Latin / Cyrillic /
    # Greek / CJK / Arabic / Devanagari ...), lowercased. No ASCII hardcode.
    # Drop digits, underscores, punctuation, control chars.
    return [t for t in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE) if t]


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


def _cos(a, b, d):
    return float(np.real(np.vdot(a, b))) / d


# ── Pillar 1: computed (zero-storage) word identity ───────────────────────────

class ComputedKey:
    """
    Word identity GENERATED from characters, never stored.
        key(w) = renorm( sum over char-trigrams of seeded phasor projection )
    Deterministic + cross-platform (hashlib, not Python's salted hash()).
    Infinite vocabulary at zero marginal bytes.

    _cache is a regenerable accelerator (memoization), NOT stored knowledge --
    every key is recomputable from the word string alone.
    """
    def __init__(self, d=512, seed=114, word_weight=4.0):
        """
        word_weight: scale of the whole-word random phasor mixed into the key
        (in addition to char-trigram components). >0 ensures distinct words get
        distinguishable keys regardless of character overlap. Pack 120 fix:
        at word_weight=0, key('little')~key('litttle')=0.85 -> cleanup
        confuses typos and morphological variants. word_weight=4 pushes that
        cosine well below 0.5 while keeping some OOV trigram signal.
        """
        self.d = int(d)
        self.seed = int(seed)
        self.word_weight = float(word_weight)
        self._cache = {}
        # trigram cache: char-trigrams form a tiny finite set (~5K-15K seen
        # in any English corpus).  Caching them separately means whole-word
        # cache eviction is cheap -- trigrams stay memoized, recomputing a
        # whole word becomes ~5 dict lookups + sum, not 5 SHA256+rng+exp.
        # Bounded by alphabet^3 in the worst case (~50K for [a-z0-9']),
        # roughly 200 MB upper bound but typically <30 MB.
        self._trigram_cache = {}

    def _seeded_phasor(self, tag):
        h = hashlib.sha256(f'{tag}:{self.seed}'.encode()).digest()
        s = int.from_bytes(h[:8], 'little')
        rng = np.random.default_rng(s)
        ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    def _trigram_vec(self, tg):
        v = self._trigram_cache.get(tg)
        if v is None:
            v = self._seeded_phasor(tg)
            self._trigram_cache[tg] = v
        return v

    def _word_vec(self, word):
        return self._seeded_phasor(f'WORD:{word}')

    def key(self, word):
        if word in self._cache:
            return self._cache[word]
        w = f'#{word}#'
        trigrams = [w[i:i+3] for i in range(len(w) - 2)] or [w]
        accum = np.zeros(self.d, dtype=np.complex64)
        for tg in trigrams:
            accum += self._trigram_vec(tg)
        if self.word_weight:
            accum += self.word_weight * self._word_vec(word)
        k = _renorm(accum)
        self._cache[word] = k
        return k

    def key_batch(self, words):
        """
        Vectorized key computation across many words.  Returns a dict
        {word: hv}.  Cheaper than per-word key() because:
          - cache hits skipped immediately,
          - trigrams unique-ified across the batch (one _trigram_vec call
            per unique trigram, not per word),
          - whole-word phasors still per-word (SHA256+rng is unique seed),
            but only computed for misses.
        Hot path for FlatRapidTrainer.
        """
        out = {}
        miss = []
        for w in words:
            v = self._cache.get(w)
            if v is None:
                miss.append(w)
            else:
                out[w] = v
        if not miss:
            return out

        # Collect unique trigrams across misses
        word_to_tgs = {}
        seen_tgs = set()
        for w in miss:
            ww = f'#{w}#'
            tgs = [ww[i:i+3] for i in range(len(ww) - 2)] or [ww]
            word_to_tgs[w] = tgs
            seen_tgs.update(tgs)
        # Force trigram cache to absorb all unique trigrams once
        for tg in seen_tgs:
            if tg not in self._trigram_cache:
                self._trigram_cache[tg] = self._seeded_phasor(tg)

        # Now compose each missing word from cached trigrams + word phasor
        for w in miss:
            accum = np.zeros(self.d, dtype=np.complex64)
            for tg in word_to_tgs[w]:
                accum += self._trigram_cache[tg]
            if self.word_weight:
                accum += self.word_weight * self._word_vec(w)
            k = _renorm(accum)
            self._cache[w] = k
            out[w] = k
        return out

    def __getstate__(self):
        # Drop both caches from pickle; both are regenerable accelerators.
        return {'d': self.d, 'seed': self.seed, 'word_weight': self.word_weight}

    def __setstate__(self, s):
        self.d = s['d']; self.seed = s['seed']
        # Backward-compat: older pickles lack word_weight (Pack <120) -- those
        # were built with word_weight=0 implicitly, so restore as such.
        self.word_weight = float(s.get('word_weight', 0.0))
        self._cache = {}
        self._trigram_cache = {}
        self._cache = {}


# ── Pillar 2: fixed sparse distributed memory substrate ───────────────────────

class VSASDM:
    """
    Fixed bank of M hard locations (random phasor addresses) + counter bank.
        write(addr, data): activate top-k nearest hard locations, add data.
        read(addr):        sum counters at activated locations, renorm.

    Substrate = (H + C). FIXED. Reading more text fills counters; never adds
    locations. H is regenerable from seed (dropped from pickle to save space).
    """
    def __init__(self, d=512, M=16384, k=64, seed=114, consolidate_every=0):
        self.d = int(d); self.M = int(M); self.k = int(k); self.seed = int(seed)
        self.consolidate_every = int(consolidate_every)   # 0 = off
        self._writes_since = 0
        self.n_consolidations = 0
        self.Hconj = self._make_Hconj()         # conj of hard-location addresses
        self.C = np.zeros((self.M, self.d), dtype=np.complex64)
        self._loc_cache = {}

    def _make_Hconj(self):
        # We only ever need conj(H) for similarity, so store that directly.
        rng = np.random.default_rng(self.seed + 7)
        ph = rng.uniform(-np.pi, np.pi, (self.M, self.d)).astype(np.float32)
        return np.exp(-1j * ph).astype(np.complex64)   # conj(exp(i*ph))

    def _activate(self, addr):
        sims = (self.Hconj @ addr).real             # <H[m], addr> for all m
        return np.argpartition(-sims, self.k)[:self.k]

    def locs(self, addr, word=None):
        if word is not None and word in self._loc_cache:
            return self._loc_cache[word]
        idx = self._activate(addr)
        if word is not None:
            self._loc_cache[word] = idx
        return idx

    def locs_batch(self, addrs, words):
        """
        Activate many addresses in ONE matmul (the real speed lever, Pack 116):
        cold words cost O(M*d) each via _activate; batching turns u separate
        mat-vecs into a single (u,d)@(d,M) BLAS call.
        """
        out = [None] * len(words)
        need_rows, need_i = [], []
        for i, wd in enumerate(words):
            cached = self._loc_cache.get(wd)
            if cached is not None:
                out[i] = cached
            else:
                need_rows.append(addrs[i]); need_i.append(i)
        if need_rows:
            A = np.stack(need_rows)                  # (m, d)
            sims = (A @ self.Hconj.T).real           # (m, M)
            for r, i in enumerate(need_i):
                idx = np.argpartition(-sims[r], self.k)[:self.k]
                self._loc_cache[words[i]] = idx
                out[i] = idx
        return out

    def write(self, addr, data, word=None):
        self.C[self.locs(addr, word)] += data
        if self.consolidate_every:
            self._writes_since += 1
            if self._writes_since >= self.consolidate_every:
                self.consolidate()
                self._writes_since = 0
                self.n_consolidations += 1

    def read(self, addr, word=None):
        return _renorm(self.C[self.locs(addr, word)].sum(axis=0))

    def consolidate(self):
        """Per-location L2 renorm -- bounds saturation, keeps recall sharp.
        Loc cache stays valid: only counter MAGNITUDES change, not the H
        addresses, so word->locs mapping is unaffected."""
        nrm = np.linalg.norm(self.C, axis=1, keepdims=True)
        nrm = np.where(nrm > 1e-9, nrm, 1.0)
        self.C = (self.C / nrm).astype(np.complex64)

    def substrate_bytes(self):
        return self.Hconj.nbytes + self.C.nbytes

    def __getstate__(self):
        s = self.__dict__.copy()
        s['Hconj'] = None        # regenerable from seed
        s['_loc_cache'] = {}     # regenerable accelerator
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        if getattr(self, 'Hconj', None) is None:
            self.Hconj = self._make_Hconj()
        if not hasattr(self, '_loc_cache') or self._loc_cache is None:
            self._loc_cache = {}


# ── Pillar 3: the flat-memory channel (wraps key + substrate + recall) ────────

class FlatMemory:
    """
    Flat lifelong co-occurrence memory. Constant RAM regardless of vocabulary.

    expose(text)        write co-occurrence into the fixed substrate
    recall(word)        adaptive reconstructive readout (mean-removed)
    similarity(a, b)    cosine of two recalls
    neighbors(word, k)  nearest seen words
    consolidate()       bound saturation, keep recall sharp

    remove_r: number of common directions stripped at read (Pack 115: r=1
    gave the biggest jump, sep 0.026 -> 0.371). Set 0 to disable adaptation.
    """
    def __init__(self, d=512, M=16384, k=64, seed=114, window=3,
                 remove_r=1, svd_sample=2500):
        self.d = int(d); self.window = int(window)
        self.remove_r = int(remove_r); self.svd_sample = int(svd_sample)
        self.ck = ComputedKey(d=d, seed=seed)
        self.sdm = VSASDM(d=d, M=M, k=k, seed=seed)
        self._seen = set()       # word strings seen (SVD sample pool); regenerable
        self._dirs = None        # cached common-direction basis
        self._dirty = True
        self.n_exposures = 0

    # ── learning ──────────────────────────────────────────────────────────
    def expose(self, text):
        """
        Write windowed co-occurrence into the substrate.

        Vectorized (Pack 116): window sums via prefix-sum (kills the inner
        neighbor loop), then aggregate per unique word and scatter ONCE each
        (linear: sum-of-writes == write-of-sum). Identical result to the naive
        double loop, much faster.
        """
        tokens = tokenize(text)
        if not tokens:
            return 0
        n = len(tokens)
        K = np.stack([self.ck.key(t) for t in tokens])        # (n, d) complex64
        w = self.window
        # complex128 prefix sums -- avoids catastrophic cancellation when
        # subtracting two large cumulative sums to recover a small window.
        P = np.empty((n + 1, self.d), dtype=np.complex128)
        P[0] = 0
        P[1:] = np.cumsum(K.astype(np.complex128), axis=0)
        agg = {}
        order = []
        for i in range(n):
            lo = i - w if i - w > 0 else 0
            hi = i + w + 1 if i + w + 1 < n else n
            ctx = (P[hi] - P[lo]) - K[i]                       # window sum minus self
            t = tokens[i]
            if t in agg:
                agg[t] = agg[t] + ctx
            else:
                agg[t] = ctx
                order.append(t)
                self._seen.add(t)
        # batch-activate unique words (one matmul), then scatter once each
        ukeys = np.stack([self.ck.key(t) for t in order])
        locs_list = self.sdm.locs_batch(ukeys, order)
        for t, idx in zip(order, locs_list):
            self.sdm.C[idx] += agg[t].astype(np.complex64)
        self.n_exposures += 1
        self._dirty = True
        return n

    # ── adaptive recall ─────────────────────────────────────────────────────
    def _refresh_dirs(self):
        if not self._dirty and self._dirs is not None:
            return
        if self.remove_r <= 0 or not self._seen:
            self._dirs = np.zeros((0, self.d), dtype=np.complex64)
            self._dirty = False
            return
        words = list(self._seen)
        if len(words) > self.svd_sample:
            rng = np.random.default_rng(0)
            words = [words[i] for i in rng.choice(len(words), self.svd_sample,
                                                  replace=False)]
        Mtx = np.stack([self.sdm.read(self.ck.key(w), w) for w in words])
        _, _, Vh = np.linalg.svd(Mtx, full_matrices=False)
        self._dirs = Vh[:self.remove_r].astype(np.complex64)
        self._dirty = False

    def recall(self, word):
        self._refresh_dirs()
        m = self.sdm.read(self.ck.key(word), word)
        for v in self._dirs:
            m = m - np.vdot(v, m) * v       # remove projection onto common dir
        return _renorm(m)

    def similarity(self, w1, w2):
        if w1 not in self._seen or w2 not in self._seen:
            return None
        return _cos(self.recall(w1), self.recall(w2), self.d)

    def neighbors(self, word, k=10):
        if word not in self._seen:
            return []
        target = self.recall(word)
        out = [(w, _cos(target, self.recall(w), self.d))
               for w in self._seen if w != word]
        return sorted(out, key=lambda x: -x[1])[:k]

    def consolidate(self):
        self.sdm.consolidate()
        self._dirty = True

    # ── introspection ───────────────────────────────────────────────────────
    @property
    def vocab_size(self):
        return len(self._seen)

    def substrate_bytes(self):
        return self.sdm.substrate_bytes()

    def status(self):
        return {
            'vocab':           len(self._seen),
            'exposures':       self.n_exposures,
            'substrate_mb':    round(self.substrate_bytes() / 1_048_576, 1),
            'remove_r':        self.remove_r,
            'flat':            True,   # substrate size is independent of vocab
        }
