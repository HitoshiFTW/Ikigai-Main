"""
ikigai.cognition.factored_meaning -- Day 91.  FACTORED distributional meaning (RHC for vectors).

WHY THIS EXISTS
---------------
The shared-superposition co-occurrence store (FlatMemory / unified 'cooccur' role) has a hard
capacity knee: all V words pile into ONE constant-RAM bank, so past ~20k vocab each word's signal
drowns in crosstalk and the similarity readout collapses to noise (measured Day 91: cooccur
pool-rank ~= chance at 210k, whitening + resonator cleanup cannot recover -- the per-item signal is
destroyed, not hidden).

RHC (Pack 257) breaks the analogous wall for INTEGERS: instead of one saturating bundle, encode
across residue factors -> capacity = PRODUCT of factor sizes.  This applies the same idea to
distributional MEANING: represent each word's co-occurrence vector by K*b random-hyperplane bits
(LSH over the substrate's own random phasors).  Capacity 2^(K*b) >> any vocab, so words never
collide; similarity is recovered from the codes by bit-agreement (Charikar 2002: P(bit agree) =
1 - angle/pi, so agreement -> cosine).  Storage is a few BYTES PER WORD (never saturates), the same
trade the exact cache makes -- readable at scale instead of constant-RAM-but-blind.

Validated (experiments/audit/day91_factored_meaning.py): superposition separation collapses
+0.027 -> 0.000 as V grows 5k -> 200k; factored holds +0.51 -> +0.57 flat, at 12.8 MB for 200k
words.  Readable distributional meaning 10x past the knee, to billions, at bytes/word.

HONEST SCOPE
------------
* LSH gives an APPROXIMATE cosine (agreement ~ cosine) -- ideal for a ranking critic, not exact
  decode.  A meaning critic only ranks candidates, so this is the right tool.
* It is NOT constant-RAM: codes cost K*b bits/word.  Tiny (64 B/word here) but grows with vocab.
* The codes are only as good as the embedding fed in; the embedding is built by Random Indexing
  over CO-OCCURRENCE (the substrate's own mechanism), never from labels or word lists.

NATIVE
------
Pure substrate: random phasor hyperplanes (regenerable from a seed, not stored knowledge),
projection, and popcount agreement.  No backprop, no learned codebook, no python-over-a-dict
similarity table.
"""

import numpy as np


class FactoredMeaning:
    """Factored (LSH/residue-coded) distributional-meaning store.

    observe_cooccur(tokens)  accumulate windowed co-occurrence into per-word embeddings
    encode(word)/consolidate() project accumulated embeddings -> factored bit codes (durable)
    sim(a, b)                cosine estimate from the codes (bit agreement)
    neighbors(word, cands)   rank candidate words by coded similarity
    """

    def __init__(self, ck, bits=512, window=3, seed=91):
        """ck: a ComputedKey (shared word-identity space with the rest of the organism).
        bits: K*b total code bits/word (capacity 2^bits).  window: co-occurrence radius."""
        self.ck = ck
        self.d = int(ck.d)
        self.bits = int(bits)
        self.window = int(window)
        self.seed = int(seed)
        self.H = self._make_H()      # random phasor hyperplanes (regenerable from seed)
        self._acc = {}        # word -> running co-occurrence embedding (training-time, transient)
        self.codes = {}       # word -> packed bit code (bits//8 uint8)  [durable, bytes/word]
        self.n_obs = 0

    def _make_H(self):
        """Fixed, regenerable hyperplane codebook (substrate primitive, not stored knowledge)."""
        rng = np.random.default_rng(self.seed)
        ph = rng.uniform(-np.pi, np.pi, (self.bits, self.d)).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    # ── persistence (codes are word-keyed + ck-independent; H regenerates from seed) ──
    def state_dict(self):
        """Durable state for .ikg: params + the bytes/word code store.  H is NOT stored
        (regenerated from seed); _acc is transient and dropped."""
        return {'bits': self.bits, 'window': self.window, 'seed': self.seed,
                'codes': self.codes}

    def load_state(self, st):
        """Restore codes + params from a state_dict; rebuild H from the saved seed."""
        self.bits = int(st.get('bits', self.bits))
        self.window = int(st.get('window', self.window))
        self.seed = int(st.get('seed', self.seed))
        self.H = self._make_H()
        self.codes = st.get('codes', {}) or {}
        return len(self.codes)

    # ── learning ─────────────────────────────────────────────────────────────
    def observe_cooccur(self, tokens):
        """Random Indexing: each token's embedding += sum of the OTHER tokens' keys within the
        window.  Same-context words converge in embedding space (distributional meaning).  Kept
        per-word (transient) until encode/consolidate projects it to a durable code."""
        toks = [str(t).strip().lower() for t in tokens if str(t).strip()]
        n = len(toks)
        if n < 2:
            return 0
        keys = {t: self.ck.key(t) for t in set(toks)}
        w = self.window
        for i, t in enumerate(toks):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            ctx = np.zeros(self.d, dtype=np.complex64)
            for j in range(lo, hi):
                if j != i:
                    ctx = ctx + keys[toks[j]]
            acc = self._acc.get(t)
            self._acc[t] = ctx if acc is None else acc + ctx
        self.n_obs += 1
        return n

    def _code_of(self, emb):
        """project an embedding onto the hyperplanes -> sign bits -> packed bytes."""
        proj = np.real(self.H @ np.conj(emb))          # (bits,)
        return np.packbits(proj > 0)

    def encode(self, word):
        """Project a word's accumulated embedding to its durable factored code."""
        w = str(word).strip().lower()
        emb = self._acc.get(w)
        if emb is None:
            return False
        self.codes[w] = self._code_of(emb)
        return True

    # ── Day 101: direct vector injection (whole-item HVs, not co-occurrence) ──
    def observe_vector(self, key, hv):
        """Register an already-built HV (e.g. cat4's bound sentence-state) under
        `key`, bypassing observe_cooccur's co-occurrence accumulation. The rest
        of the pipeline (encode/consolidate/sim/neighbors) doesn't care how the
        embedding arose -- this lets FactoredMeaning serve as the chunked/coded
        identification layer for WHOLE-ITEM vectors, the Day-101 capacity-wall
        target, not just single-word distributional meaning."""
        self._acc[str(key).strip().lower()] = np.asarray(hv, dtype=np.complex64)

    def encode_once(self, key, hv):
        """Day 101: observe + encode + immediately drop the transient embedding
        for `key`.  Word-level co-occurrence legitimately accumulates over many
        documents before a single consolidate() drops _acc in bulk; a one-shot
        whole-item vector (cat4's anchors) never gets a second observation, so
        leaving it in _acc would silently retain a full d-dim complex64 copy
        (3.2 KB/anchor at d=400) forever alongside the 64 B/item durable code --
        defeating the bytes-not-vectors point of this store. Use this instead
        of observe_vector+encode for one-shot injections."""
        self.observe_vector(key, hv)
        self.encode(key)
        self._acc.pop(str(key).strip().lower(), None)

    def _codes_matrix(self):
        """Stack all durable codes into one (n, bits//8) uint8 array + the
        matching key list, cached until the code count changes (append-only
        in the intended write pattern -- rebuilt cheaply if it shrinks)."""
        cache = getattr(self, '_codes_mat_cache', None)
        if cache is not None and cache[0] == len(self.codes):
            return cache[1], cache[2]
        keys = list(self.codes.keys())
        mat = np.stack([self.codes[k] for k in keys]) if keys else \
            np.zeros((0, self.bits // 8), dtype=np.uint8)
        self._codes_mat_cache = (len(self.codes), keys, mat)
        return keys, mat

    # ── Day 102: SPANN-style inverted index over the code store ──────────────
    def build_index(self, n_clusters=None, closure_r=4, iters=12,
                    fit_sample=25000, seed=102, min_n=2000):
        """Day 102 -- build a SPANN-style (Chen et al. 2021, arXiv 2111.08566)
        inverted index over the code store so nearest_by_hv is SUBLINEAR instead
        of O(N).  SELF-CONTAINED in the codes: clusters the unpacked +-1 sign
        codes (LSH preserves cosine, so code-space cosine ~ address cosine), so
        it needs no raw address vectors (which this store deliberately drops).

        centroids (~2*sqrt(N), RAM-resident) route a query to a few POSTING
        lists; only the union of those lists is Hamming-scanned.  Each item joins
        its top-`closure_r` centroids (SPANN closure augmentation -> boundary
        recall).  Measured (experiments/audit/day102_*): exact recall@1 = 1.0000
        at ~1.6-3.4% of N (vs the full scan's 100%), candidate% FALLS as N grows
        (sublinear), centroids ~0.2 GB projected at N=1e9 with postings+codes on
        SSD.  PARAPHRASE routing is weaker in code space (~0.87-0.96 vs the
        oracle) -- so nearest_by_hv only TRUSTS the index for near-exact hits and
        FALLS BACK to the full O(N) scan otherwise (the scan is KEPT as the
        ground-truth oracle; results are never worse than it).  Fuzzy-robust
        routing wants a graph-navigated index (DiskANN/SPTAG) -- the next step.

        Skips the IVF (index=None) below min_n, where the full scan is already
        instant -- but ALWAYS builds the Tier-0 exact hashmap (code -> key), which
        is cheap and gives O(1) exact re-lookup at any N (see recall_cascade).
        Returns the number of items indexed by the IVF (0 if skipped)."""
        keys, mat = self._codes_matrix()
        N = len(keys)
        # Tier 0 (always): exact code -> key hashmap for O(1) exact re-lookup.
        self._exact_map = {}
        for i, k in enumerate(keys):
            self._exact_map.setdefault(mat[i].tobytes(), k)     # first key wins on collision
        if N < int(min_n):
            self._index = None
            return 0
        rng = np.random.default_rng(seed)
        P = np.unpackbits(mat, axis=1).astype(np.float32) * 2.0 - 1.0   # (N, bits) +-1
        P /= np.sqrt(P.shape[1])                                        # unit (const norm)
        if n_clusters is None:
            n_clusters = max(1, int(round(2.0 * np.sqrt(N))))
        fit = P[rng.choice(N, size=min(int(fit_sample), N), replace=False)]
        cent = P[rng.choice(N, size=n_clusters, replace=False)].copy()
        for _ in range(int(iters)):
            assign = np.argmax(fit @ cent.T, axis=1)
            moved = 0
            for c in range(n_clusters):
                m = fit[assign == c]
                if len(m) == 0:
                    cent[c] = fit[rng.integers(fit.shape[0])]
                    continue
                v = m.sum(axis=0); v /= (np.linalg.norm(v) + 1e-9)
                moved += float(np.linalg.norm(v - cent[c]) > 1e-4)
                cent[c] = v
            if moved == 0:
                break
        r = int(closure_r)
        buckets = [[] for _ in range(n_clusters)]
        for s in range(0, N, 20000):                       # chunked -> bounded peak RAM
            sims = P[s:s+20000] @ cent.T
            topr = np.argpartition(-sims, r - 1, axis=1)[:, :r]
            for row in range(topr.shape[0]):
                gi = s + row
                for c in topr[row]:
                    buckets[c].append(gi)
        self._index = {'N': N, 'keys': keys, 'centroids': cent.astype(np.float32),
                       'postings': [np.asarray(b, dtype=np.int32) for b in buckets]}
        return N

    # ── Day 102: conformal abstention over the index (correct-or-abstain) ────
    def _random_hv(self, rng, n_terms=6):
        """A fresh random unit-phasor bundle in this store's space -- an
        OUT-OF-STORE query (nothing was written for it)."""
        ph = rng.uniform(-np.pi, np.pi, (n_terms, self.d)).astype(np.float32)
        acc = np.exp(1j * ph).sum(axis=0)
        mag = np.abs(acc); mag = np.where(mag > 1e-9, mag, 1.0)
        return (acc / mag).astype(np.complex64)

    def calibrate_abstain(self, absent_hvs=None, n_absent=2000, alpha=0.05, seed=102):
        """Calibrate a conformal abstention threshold on the code store so a
        recall is only trusted when its agreement clears what an OUT-OF-STORE
        query would score.  Verified: experiments/audit/day102_conformal_gate.py
        (false-answer rate on unknowns <= alpha, exact finite-sample).

        absent_hvs: a caller-supplied set of should-abstain query HVs.  STRONGLY
        preferred to be STRUCTURED held-out absents (real queries whose fact is
        not stored), which are exchangeable with live unknowns.  If None, random
        phasor bundles are used -- convenient but OPTIMISTIC: a populated store's
        real absent read is CROSSTALK from stored items and runs hotter than pure
        random (see calibration.empirical_boundary's crosstalk note), so the
        random-calibrated tau under-rejects structured unknowns.  Returns tau."""
        from .calibration import ConformalAbstain
        rng = np.random.default_rng(seed)
        if absent_hvs is None:
            absent_hvs = [self._random_hv(rng) for _ in range(int(n_absent))]
        scores = []
        for hv in absent_hvs:
            res = self.nearest_by_hv(hv, top_k=1)
            if res:
                scores.append(res[0][1])
        ab = ConformalAbstain(alpha=alpha)
        ab.calibrate(scores, [False] * len(scores))     # all should-abstain
        self._abstain = ab
        return ab.tau

    def recall_or_abstain(self, hv, top_k=1):
        """nearest_by_hv, but returns None (ABSTAIN) when the top agreement does
        not clear the conformal threshold -- i.e. when the query is
        indistinguishable from an out-of-store one.  Falls back to plain recall
        if calibrate_abstain was never called (inert until calibrated, so callers
        built before this see no change)."""
        res = self.nearest_by_hv(hv, top_k=top_k)
        if not res:
            return None
        ab = getattr(self, '_abstain', None)
        if ab is not None and not ab.accept(res[0][1]):
            return None
        return res

    # ── Day 102: UNIFIED TIERED RECALL -- the organism's free-energy principle
    #    (many faculties, cheapest-confident wins) applied to memory lookup.
    #    Each tier covers the previous one's blind spot; a query descends only as
    #    far as it must, so average cost is the cheap tier and worst case is exact.
    def set_fuzzy_router(self, fn):
        """Install the Tier-2 fuzzy router: a callable hv -> (key, agree) or None
        for the fuzzy tail (Day-103 PQ-on-SSD / AiSAQ).  None = socket empty; the
        cascade then skips Tier 2 and the O(N) oracle (Tier 3) backstops fuzzy."""
        self._fuzzy_router = fn

    def recall_cascade(self, hv, probe_p=4, trust_index_fuzzy=False):
        """Tiered recall.  Returns {'key', 'agree', 'tier'} or None (abstain):
          Tier 0  exact hashmap        -- O(1) exact re-lookup.
          Tier 1  IVF index            -- sublinear; trusted only on an EXACT hit
                                          by default (safe: the index's fuzzy pick
                                          can be less similar than the oracle, the
                                          confident-wrong error a no-hallucination
                                          store must not make).  trust_index_fuzzy
                                          also accepts conformal-cleared fuzzy hits
                                          (approximate mode -- only sound once a
                                          high-recall router bounds Tier-1 mis-ID).
          Tier 2  fuzzy router socket  -- Day-103 PQ-on-SSD; skipped if unset.
          Tier 3  O(N) code oracle     -- exact-perfect backstop on the residual;
                                          so the DEFAULT cascade is never less
                                          similar than the exact scan.
          abstain conformal gate       -- None when the best clears no tier's floor
                                          (an out-of-store query).
        Each tier only fires when the cheaper ones passed the query down."""
        keys, mat = self._codes_matrix()
        if not keys:
            return None
        code = self._code_of(np.asarray(hv, dtype=np.complex64))
        ab = getattr(self, '_abstain', None)

        # Tier 0 -- exact hashmap
        em = getattr(self, '_exact_map', None)
        if em is not None:
            k = em.get(code.tobytes())
            if k is not None:
                return {'key': k, 'agree': 1.0, 'tier': 0}

        # Tier 1 -- IVF fast path (auto-build if warranted)
        idx = getattr(self, '_index', None)
        if idx is None and len(keys) >= 2000:
            self.build_index()
            idx = getattr(self, '_index', None)
        if idx is not None and idx['N'] == len(keys):
            cand = self._route_candidates(code, idx, probe_p)
            if len(cand):
                ham = np.unpackbits(np.bitwise_xor(mat[cand], code[np.newaxis, :]),
                                    axis=1).sum(axis=1)
                b = int(np.argmin(ham))
                agree = 1.0 - ham[b] / self.bits
                key = keys[int(cand[b])]
                # exact-safe by default; opt-in fuzzy trust needs conformal clearance
                if agree >= 1.0 or (trust_index_fuzzy and ab is not None and ab.accept(agree)):
                    return {'key': key, 'agree': float(agree), 'tier': 1}

        # Tier 2 -- fuzzy router socket (Day-103); consulted only for the tail
        fr = getattr(self, '_fuzzy_router', None)
        if fr is not None:
            res = fr(hv)
            if res is not None and (ab is None or ab.accept(res[1])):
                return {'key': res[0], 'agree': float(res[1]), 'tier': 2}

        # Tier 3 -- O(N) exact oracle backstop
        ham = np.unpackbits(np.bitwise_xor(mat, code[np.newaxis, :]),
                            axis=1).sum(axis=1)
        b = int(np.argmin(ham))
        agree = 1.0 - ham[b] / self.bits
        if ab is not None and not ab.accept(agree):
            return None                                  # abstain: out of store
        return {'key': keys[b], 'agree': float(agree), 'tier': 3}

    def recall(self, hv, **kw):
        """Day 102 -- HOT-QUERY CACHE in front of the tiered cascade (the 'cache'
        of 'sdm + chunked vsa + cache'). A verbatim-repeat query -- the bulk of
        real, Zipfian traffic (a few hot facts asked constantly) -- skips ALL
        tiers and returns in one dict lookup: O(1), a few nanojoules, no 512-bit
        hashing or candidate scan. Cold/new queries fall through to recall_cascade
        and their answer (including an abstain) is cached. The cache is keyed on
        the store size and CLEARS when the store grows, so a newly taught fact can
        never be shadowed by a stale cached answer (write-invalidation).
        Returns {'key','agree','tier'} (tier=-1 on a cache hit) or None (abstain)."""
        from collections import OrderedDict
        keys, _ = self._codes_matrix()
        code = self._code_of(np.asarray(hv, dtype=np.complex64))
        hb = code.tobytes()
        hot = getattr(self, '_hot', None)
        if hot is None or hot['N'] != len(keys):
            hot = {'N': len(keys), 'd': OrderedDict(), 'cap': 100000}
            self._hot = hot
        d = hot['d']
        if hb in d:                                      # HIT -- O(1), skip all tiers
            d.move_to_end(hb)
            c = d[hb]
            return None if c is None else {**c, 'tier': -1}
        out = self.recall_cascade(hv, **kw)              # MISS -- run the tiers, then cache
        d[hb] = out
        if len(d) > hot['cap']:
            d.popitem(last=False)                        # LRU evict
        return out

    def _route_candidates(self, code, idx, probe_p):
        """Route a query code to its top-`probe_p` centroids -> union of their
        posting lists (the sublinear candidate set)."""
        q = np.unpackbits(code[np.newaxis, :], axis=1).astype(np.float32)[0] * 2.0 - 1.0
        cs = idx['centroids'] @ q                           # O(C)=O(sqrt N)
        p = min(int(probe_p), cs.shape[0])
        probe = np.argpartition(-cs, p - 1)[:p]
        return np.unique(np.concatenate([idx['postings'][c] for c in probe]))

    # ── Day 102: SAFE SUBLINEAR FUZZY retrieval -- multi-table LSH + exact verify ──
    #    The Tier-2 fuzzy router the cascade socket wanted, finally safe. Prior
    #    tries failed: a graph over the sign codes was WORSE than flat-IVF
    #    (distance-concentrated); IVFPQ routed fuzzy but its mis-IDs were
    #    unscreenable (0.25-0.30 confident-wrong). This one works because it
    #    (1) AMPLIFIES containment with R independent LSH tables (miss ~ miss1^R,
    #    measured independent) so the true item is in the candidate union with
    #    tunable probability -> 1, then (2) EXACT sign-Hamming verifies the union
    #    (so the score is the true agreement), and (3) leaves the rare containment
    #    miss to the conformal floor (abstain) / Tier-3 (exact). Build is INSTANT
    #    (bit-slicing the codes we already store -- no k-means, no vectors, no
    #    extra bytes/item), and candidate% ∝ 1/sqrt(N) -> <1% at a billion items.
    def build_lsh(self, R=16, b=12, seed=102, min_n=2000):
        """Build R independent LSH tables by slicing the sign-code bits into R
        disjoint b-bit hashes. Self-contained in the code store. Skips below
        min_n (full scan already instant). Returns #tables (0 if skipped)."""
        keys, mat = self._codes_matrix()
        N = len(keys)
        if N < int(min_n) or R * b > self.bits:
            self._lsh = None
            return 0
        rng = np.random.default_rng(seed)
        allbits = np.unpackbits(mat, axis=1)                 # (N, bits)
        perm = rng.permutation(self.bits)
        slices = [perm[r * b:(r + 1) * b] for r in range(R)]
        w = (1 << np.arange(b - 1, -1, -1)).astype(np.int64)
        tables = []
        for r in range(R):
            key = (allbits[:, slices[r]].astype(np.int64) * w).sum(1)
            tab = {}
            for i in range(N):
                tab.setdefault(int(key[i]), []).append(i)
            tables.append({k: np.asarray(v, np.int32) for k, v in tab.items()})
        self._lsh = {'N': N, 'keys': keys, 'R': R, 'b': b, 'slices': slices,
                     'w': w, 'tables': tables}
        self.set_fuzzy_router(self.recall_fuzzy)             # auto-wire cascade Tier 2
        return R

    def _lsh_candidates(self, code, multiprobe=1):
        """Union of the query's bucket (+ multiprobe-Hamming buckets) across all
        R tables -- the sublinear candidate set for a fuzzy query."""
        from itertools import combinations
        lx = self._lsh
        qb = np.unpackbits(code[np.newaxis, :], axis=1)[0]
        cands = []
        for r in range(lx['R']):
            qs = qb[lx['slices'][r]]
            base = int((qs.astype(np.int64) * lx['w']).sum())
            buckets = [base]
            for t in range(1, int(multiprobe) + 1):
                for flip in combinations(range(lx['b']), t):
                    bb = base
                    for f in flip:
                        bb ^= (1 << (lx['b'] - 1 - f))
                    buckets.append(bb)
            tab = lx['tables'][r]
            for bk in buckets:
                v = tab.get(bk)
                if v is not None:
                    cands.append(v)
        return np.unique(np.concatenate(cands)) if cands else np.empty(0, np.int32)

    def recall_fuzzy(self, hv, multiprobe=1):
        """SAFE sublinear fuzzy recall: LSH candidate union -> EXACT sign-Hamming
        verify -> (key, agree). Returns None if the store has no LSH index or no
        candidate. The caller (recall_cascade Tier 2) applies the conformal floor,
        so an out-of-store query whose best agreement is at the noise floor is
        abstained (no-hallucination bound holds)."""
        lx = getattr(self, '_lsh', None)
        keys, mat = self._codes_matrix()
        if lx is None or lx['N'] != len(keys):
            return None
        code = self._code_of(np.asarray(hv, dtype=np.complex64))
        cand = self._lsh_candidates(code, multiprobe)
        if len(cand) == 0:
            return None
        ham = np.unpackbits(np.bitwise_xor(mat[cand], code[np.newaxis, :]),
                            axis=1).sum(axis=1)
        b = int(np.argmin(ham))
        return keys[int(cand[b])], float(1.0 - ham[b] / self.bits)

    def nearest_by_hv(self, hv, top_k=1, probe_p=4, exact_fallback=True,
                      fast_agree=1.0, autobuild=True):
        """Day 101 -- identify which stored KEY a fresh (unregistered) vector
        is nearest to, by bit-agreement of its on-the-fly code against the
        durable codes (Hamming popcount).  Returns [(key, agree_fraction), ...]
        sorted descending, best first.

        Day 102 -- SUBLINEAR fast path: if a code index is built (auto-built once
        the store crosses min_n, so this stays default-on with no API change),
        route the query to a few posting lists and Hamming-scan only their union.

        SAFETY (measured, day102_index_gate): flat-IVF code routing finds EXACT
        matches perfectly (recall@1=1.0) but on fuzzy queries it returns a
        marginally-less-similar item ~1.5% of the time -- a CONFIDENT-WRONG error,
        the one kind a no-hallucination oracle must not make.  There is no
        provably-safe trust bar below an exact match, so the DEFAULT fast_agree=1.0
        trusts the fast path ONLY on an exact code hit (agree==1.0, Hamming 0 ->
        provably the global nearest); anything less FALLS BACK to the full O(N)
        scan (the kept oracle).  So the answer is NEVER less similar than the exact
        scan; the sublinear win lands on exact re-lookups.  Fuzzy-robust speedup
        needs a graph-navigated index (DiskANN/SPTAG) precise enough to trust --
        the next step.  fast_agree<1.0 trades that safety for fuzzy speed;
        exact_fallback=False is pure-ANN (benchmark only)."""
        keys, mat = self._codes_matrix()
        if not keys:
            return []
        code = self._code_of(np.asarray(hv, dtype=np.complex64))

        idx = getattr(self, '_index', None)
        if idx is None and autobuild and len(keys) >= 2000:
            self.build_index()
            idx = getattr(self, '_index', None)
        if idx is not None and idx['N'] == len(keys):
            cand = self._route_candidates(code, idx, probe_p)
            if len(cand):
                xor = np.bitwise_xor(mat[cand], code[np.newaxis, :])
                ham = np.unpackbits(xor, axis=1).sum(axis=1)
                agree = 1.0 - ham / self.bits
                k = min(top_k, len(cand))
                if len(cand) > k:
                    part = np.argpartition(-agree, k - 1)[:k]
                    order = part[np.argsort(-agree[part])]
                else:
                    order = np.argsort(-agree)
                if (not exact_fallback) or float(agree[order[0]]) >= fast_agree:
                    return [(keys[int(cand[i])], float(agree[i])) for i in order]
                # uncertain -> fall through to the exact O(N) oracle scan

        # ── full O(N) oracle scan (no index, stale index, or low-confidence) ──
        xor = np.bitwise_xor(mat, code[np.newaxis, :])
        ham = np.unpackbits(xor, axis=1).sum(axis=1)
        agree = 1.0 - ham / self.bits
        if top_k >= len(keys):
            order = np.argsort(-agree)
        else:
            part = np.argpartition(-agree, top_k)[:top_k]
            order = part[np.argsort(-agree[part])]
        return [(keys[i], float(agree[i])) for i in order]

    def _common_dirs(self, r):
        """Top-r common directions of the embeddings (the 'everything' modes that frequent words
        share) -- SVD over a sample.  Removing them de-swamps retrieval so RARE informative words
        drive similarity, not high-frequency function words."""
        if r <= 0 or not self._acc:
            return np.zeros((0, self.d), dtype=np.complex64)
        words = list(self._acc)
        if len(words) > 2500:
            rng = np.random.default_rng(0)
            words = [words[i] for i in rng.choice(len(words), 2500, replace=False)]
        M = np.stack([self._acc[w] / (np.linalg.norm(self._acc[w]) + 1e-9) for w in words])
        _, _, Vh = np.linalg.svd(M, full_matrices=False)
        return Vh[:r].astype(np.complex64)

    def _whiten(self, emb, dirs):
        for v in dirs:
            emb = emb - np.vdot(v, emb) * v
        return emb

    def consolidate(self, drop_acc=True, remove_r=0):
        """Project every accumulated embedding to a durable code (biology: consolidation).
        remove_r>0 strips that many common-mode directions before coding; measured on Tatoeba it
        OVER-removed and hurt retrieval, so default is 0 (available for denser corpora).  Drops the
        transient embeddings by default, leaving only the bytes/word code store."""
        self._dirs = self._common_dirs(remove_r)
        n = 0
        for w in list(self._acc):
            self.codes[w] = self._code_of(self._whiten(self._acc[w].copy(), self._dirs))
            n += 1
        if drop_acc:
            self._acc = {}
        return n

    # ── readout ──────────────────────────────────────────────────────────────
    def _bits_of(self, word):
        w = str(word).strip().lower()
        c = self.codes.get(w)
        if c is None and w in self._acc:            # not yet consolidated -> code on the fly
            c = self._code_of(self._acc[w])
        return c

    def sim(self, a, b):
        """Cosine estimate from the codes via bit agreement (Charikar).  None if either unseen."""
        ca, cb = self._bits_of(a), self._bits_of(b)
        if ca is None or cb is None:
            return None
        ham = np.unpackbits(np.bitwise_xor(ca, cb)).sum()
        agree = 1.0 - ham / self.bits
        return float(np.cos(np.pi * (1.0 - agree)))

    def neighbors(self, word, candidates, k=5):
        """Rank candidate words by coded similarity to `word` (the meaning-critic primitive)."""
        out = []
        for c in candidates:
            s = self.sim(word, c)
            if s is not None:
                out.append((c, s))
        return sorted(out, key=lambda x: -x[1])[:k]

    # ── decodable meaning (Day 91 #1): locality-preserving PQ factors + code-space emit ──
    def build_decodable(self, n_factors=6, per_factor=32, iters=10):
        """Learn K locality-preserving factor codebooks (cosine k-means per embedding SUBSPACE)
        over the accumulated co-occurrence embeddings, and assign each word a K-index factor CODE.
        Capacity = per_factor ** n_factors distinct meanings; similar words share indices; EMISSION
        composes in code space (no HV-decode -> no resonator d-bound).  Call after observe_cooccur,
        before consolidate drops the embeddings.  Returns #words coded."""
        if not self._acc:
            return 0
        words = list(self._acc)
        E = np.stack([self._acc[w] / (np.linalg.norm(self._acc[w]) + 1e-9) for w in words])
        K = int(n_factors); C = int(per_factor); sub = self.d // K
        rng = np.random.default_rng(self.seed)
        self._dK, self._dC, self._dsub = K, C, sub
        self._centroids = []
        codes = np.zeros((len(words), K), dtype=np.int32)
        for k in range(K):
            Xs = E[:, k*sub:(k+1)*sub]
            Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
            cent = Xs[rng.choice(len(words), min(C, len(words)), replace=False)].copy()
            for _ in range(int(iters)):
                assign = np.argmax(np.real(Xs @ np.conj(cent).T), axis=1)
                for c in range(len(cent)):
                    m = Xs[assign == c]
                    if len(m):
                        v = m.sum(axis=0); cent[c] = v / (np.linalg.norm(v) + 1e-9)
            codes[:, k] = np.argmax(np.real(Xs @ np.conj(cent).T), axis=1)
            self._centroids.append(cent)
        self.dcode = {words[i]: tuple(int(c) for c in codes[i]) for i in range(len(words))}
        self._cell = {}
        for wd, cd in self.dcode.items():
            self._cell.setdefault(cd, []).append(wd)
        # JET ENGINE: block-partitioned random-phasor codebooks -> the meaning code is an HV that
        # decodes INDEPENDENTLY per block (no joint-resonator d-bound), so it is decodable at C**K.
        self._blockcb = [np.exp(1j*rng.uniform(-np.pi, np.pi, (C, sub))).astype(np.complex64)
                         for _ in range(K)]
        self.decodable_capacity = C ** K
        return len(self.dcode)

    def meaning_hv(self, word_or_code):
        """The decodable meaning HV of a word (or a code tuple): block-partitioned concat of the
        per-factor phasors.  Decodes independently per block -> readable at C**K."""
        code = word_or_code if isinstance(word_or_code, tuple) else self.dcode.get(
            str(word_or_code).strip().lower())
        if code is None:
            return None
        return np.concatenate([self._blockcb[k][code[k]] for k in range(self._dK)]).astype(np.complex64)

    def decode_hv(self, hv, k=3):
        """Recover the K factor indices from a meaning HV by INDEPENDENT per-block argmax (no joint
        d-bound), then the word(s) in that cell.  The generative read: meaning HV -> word."""
        K, sub = self._dK, self._dsub
        code = tuple(int(np.argmax(np.real(self._blockcb[i] @ np.conj(hv[i*sub:(i+1)*sub]))))
                     for i in range(K))
        got = self._cell.get(code)
        if got:
            return got[:k]
        best = max(self._cell, key=lambda cd: sum(cd[i] == code[i] for i in range(K)))
        return self._cell[best][:k]

    def emit(self, tokens, k=3):
        """EMISSION: blend the meaning of `tokens` in CODE space (per-factor nearest learned
        centroid over their embeddings, or majority code if embeddings are gone) -> the words in
        that cell.  The generative primitive: compose a target meaning -> grounded word(s)."""
        if not getattr(self, 'dcode', None):
            return []
        K, sub = self._dK, self._dsub
        toks = [str(t).strip().lower() for t in tokens]
        embs = [self._acc[t] for t in toks if t in self._acc]
        if embs:                                          # blend embeddings -> nearest centroids
            blend = np.sum(embs, axis=0); blend = blend / (np.linalg.norm(blend) + 1e-9)
            code = []
            for kk in range(K):
                xs = blend[kk*sub:(kk+1)*sub]; xs = xs / (np.linalg.norm(xs) + 1e-9)
                code.append(int(np.argmax(np.real(self._centroids[kk] @ np.conj(xs)))))
            code = tuple(code)
        else:                                             # code-space majority over token codes
            cds = [self.dcode[t] for t in toks if t in self.dcode]
            if not cds:
                return []
            code = tuple(int(np.bincount([c[kk] for c in cds]).argmax()) for kk in range(K))
        exact = self._cell.get(code, [])
        if exact:
            return exact[:k]
        # nearest cell by shared-index fraction
        scored = sorted(self._cell.items(),
                        key=lambda kv: -sum(kv[0][i] == code[i] for i in range(K)))
        return [w for cd, ws in scored[:1] for w in ws][:k]

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def vocab_size(self):
        return len(set(self.codes) | set(self._acc))

    def substrate_bytes(self):
        """durable code store = bits/8 per word (+ the fixed hyperplane codebook)."""
        return len(self.codes) * (self.bits // 8) + self.H.nbytes

    def status(self):
        return {'vocab': self.vocab_size, 'coded': len(self.codes),
                'bits_per_word': self.bits, 'bytes_per_word': self.bits // 8,
                'capacity': f'2^{self.bits}',
                'store_mb': round(self.substrate_bytes() / 1_048_576, 3)}
