"""
ikigai.cognition.rhc_stores -- Day 91.  RHC-addressed EXACT stores at 1e9 scale, fixed substrate.

Two stores, both using Residue HDC (Pack 257) block-partitioned addressing so decode is independent
per modulus (sidesteps the joint-resonator d-bound) and scales to ~1e9 at a few bytes/item:

  FactStore  -- knowledge as address-tuples.  A fact (subj, rel, obj) is three lexicon indices ->
                one composite integer; stored as its residue tuple (bytes/fact); recovered exactly
                by CRT.  Consequences derived, not stored.
  AssocCache -- the anchor cache IN the substrate.  A key -> an RHC phasor ADDRESS over a 1e9
                collision-free space (no external hash dict); value stored by sparse SDM addressing.

Validated: experiments/audit/day91_rhc_fact_store.py, day91_rhc_insubstrate_cache.py.

HONEST SCOPE (Day 95 red-team correction, F0b):
  * AssocCache is genuinely substrate: key -> RHC phasor ADDRESS -> sparse SDM store/cleanup
    (put/get run phasor encode + argmax cleanup; no dict for the value store).
  * FactStore, by contrast, stores facts in a PYTHON `set` of (s,r,o) int-tuples plus a
    `dict` adjacency index (`_by_subj`); has()/objects_of()/derive_chain() are set-membership,
    dict-lookup, and BFS-over-the-dict. The `_BlockRHC` CRT machinery is instantiated but is
    used ONLY by encode_slot/decode_slot (recovering a slot index that arrives as a noisy HV);
    it does NOT back the store, recall, or derivation. So ">= 1e9, a few bytes/fact" describes
    the RHC ADDRESS RANGE (product of coprime moduli), NOT the storage, which is a python
    tuple-set held in RAM. Do not describe FactStore as "holographic exact storage".
"""

import hashlib
import numpy as np

_DEFAULT_MODULI = (7, 11, 13, 17, 19, 23, 29, 31)      # pairwise coprime, range ~6.69e9


def _crt(residues, moduli, prod):
    x = 0
    for r, m in zip(residues, moduli):
        Mk = prod // m
        x += r * Mk * pow(Mk % m, -1, m)
    return x % prod


class _BlockRHC:
    """Block-partitioned Residue HDC: each modulus owns a disjoint slice of `d` dims, so each
    residue is read INDEPENDENTLY (small argmax) then CRT-recombined -- no joint-decode d-bound."""

    def __init__(self, d=256, moduli=_DEFAULT_MODULI, seed=257):
        self.d = int(d)
        self.moduli = tuple(int(m) for m in moduli)
        self.K = len(self.moduli)
        self.blk = self.d // self.K
        self.range = 1
        for m in self.moduli:
            self.range *= m
        rng = np.random.default_rng(int(seed))
        self.bases = [rng.integers(1, self.moduli[k], self.blk).astype(np.float32)
                      for k in range(self.K)]
        self.codebooks = [np.stack([self._res_hv(k, r) for r in range(self.moduli[k])])
                          for k in range(self.K)]

    def _res_hv(self, k, r):
        theta = (2.0 * np.pi * (int(r) % self.moduli[k]) / self.moduli[k]) * self.bases[k]
        return np.exp(1j * theta).astype(np.complex64)

    def residues(self, x):
        return [int(x) % m for m in self.moduli]

    def encode(self, x):
        return np.concatenate([self._res_hv(k, int(x) % self.moduli[k])
                               for k in range(self.K)]).astype(np.complex64)

    def decode(self, hv):
        res = []
        for k in range(self.K):
            blk = hv[k*self.blk:(k+1)*self.blk]
            res.append(int(np.argmax(np.real(self.codebooks[k] @ np.conj(blk)))))
        return _crt(res, self.moduli, self.range)


class FactStore:
    """Knowledge as address-tuples: a fact (subj, rel, obj) is three RHC-addressable indices,
    stored as the tuple itself (a few bytes/fact).  Each slot spans the full RHC range (>= 1e9),
    so lexicon per slot = range (no composite L^3 limit).  Consequences derived, not stored; an
    index arriving as a noisy HV is recovered exactly by encode_slot/decode_slot (block-CRT)."""

    def __init__(self, d=256, moduli=_DEFAULT_MODULI, seed=257):
        self.rhc = _BlockRHC(d=d, moduli=moduli, seed=seed)
        self.L = self.rhc.range                           # lexicon capacity per slot
        self._facts = set()                               # {(s, r, o)}  (durable, tiny)
        self._by_subj = {}                                # s -> [(r, o)]  (derive index)

    def add(self, s, r, o):
        t = (int(s), int(r), int(o))
        if t not in self._facts:
            self._facts.add(t)
            self._by_subj.setdefault(t[0], []).append((t[1], t[2]))
        return t

    def has(self, s, r, o):
        return (int(s), int(r), int(o)) in self._facts

    def encode_slot(self, x):
        """RHC HV for a lexicon index (for perception of a slot as a phasor)."""
        return self.rhc.encode(x)

    def decode_slot(self, hv):
        """Recover a lexicon index from its HV exactly (block-partitioned per-modulus + CRT)."""
        return self.rhc.decode(hv)

    def objects_of(self, s, r=None):
        return [oo for (rr, oo) in self._by_subj.get(int(s), []) if r is None or rr == int(r)]

    def derive_chain(self, s, r, max_hops=8):
        """transitive closure over relation r from s (consequences derived, stored zero)."""
        seen, frontier, chain = {int(s)}, [int(s)], []
        for _ in range(max_hops):
            nxt = []
            for x in frontier:
                for o in self.objects_of(x, r):
                    if o not in seen:
                        seen.add(o); nxt.append(o); chain.append((x, o))
            if not nxt:
                break
            frontier = nxt
        return chain

    @property
    def n_facts(self):
        return len(self._facts)

    def capacity(self):
        """ADDRESS-SPACE size (product of coprime moduli, >= 1e9) -- the range of DISTINCT
        lexicon indices a slot can name WITHOUT collision. This is NOT the storage capacity:
        the store holds every fact as a python (s,r,o) int-tuple in `self._facts`, so real
        capacity is RAM-bounded and memory grows LINEARLY (~a tuple of 3 interned ids per
        fact), not held in a fixed substrate. Use `n_facts` for what is actually stored."""
        return self.rhc.range

    def storage_bytes(self):
        """MEASURED in-RAM footprint of the actual fact store (the honest scaling number):
        the python set of (s,r,o) tuples + the _by_subj adjacency index. Grows LINEARLY with
        n_facts -- there is no fixed-size holographic compression of the exact store."""
        import sys
        b = sys.getsizeof(self._facts) + sys.getsizeof(self._by_subj)
        for t in self._facts:
            b += sys.getsizeof(t)
        for k, v in self._by_subj.items():
            b += sys.getsizeof(v)
        return b

    def state_dict(self):
        return {'d': self.rhc.d, 'moduli': list(self.rhc.moduli),
                'facts': list(self._facts)}

    def load_state(self, st):
        self.rhc = _BlockRHC(d=int(st['d']), moduli=tuple(st['moduli']))
        self.L = self.rhc.range
        self._facts = set(); self._by_subj = {}
        for (s, r, o) in st['facts']:
            self.add(s, r, o)
        return len(self._facts)


class AssocCache:
    """The anchor cache IN the substrate: key -> RHC phasor address over a 1e9 collision-free
    space (no external hash dict); value stored by sparse SDM addressing.  Store capacity is the
    SDM's (M locations); address space is RHC's (>= 1e9)."""

    def __init__(self, ck, d=256, M=20000, kw=20, moduli=_DEFAULT_MODULI, seed=258):
        self.ck = ck
        self.d = int(d)
        self.M = int(M)
        self.kw = int(kw)
        self.rhc = _BlockRHC(d=d, moduli=moduli, seed=seed)
        rng = np.random.default_rng(seed)
        self.loc = np.exp(1j * rng.uniform(-np.pi, np.pi, (self.M, self.d))).astype(np.complex64)
        self.C = np.zeros((self.M, self.d), dtype=np.complex64)
        self.n = 0

    def _addr_int(self, key):
        return int.from_bytes(hashlib.blake2b(str(key).encode(), digest_size=8).digest(),
                              'little') % self.rhc.range

    def _locs(self, addr_hv):
        sims = np.real(self.loc @ np.conj(addr_hv))
        return np.argpartition(-sims, self.kw)[:self.kw]

    def put(self, key, value):
        addr = self.rhc.encode(self._addr_int(key))
        self.C[self._locs(addr)] += self.ck.key(str(value).strip().lower())
        self.n += 1

    def get(self, key, candidates):
        addr = self.rhc.encode(self._addr_int(key))
        v = self.C[self._locs(addr)].sum(axis=0)
        cand = list(candidates)
        Kc = np.stack([self.ck.key(str(c).strip().lower()) for c in cand])
        return cand[int(np.argmax(np.real(Kc @ np.conj(v))))]

    def capacity(self):
        """ADDRESS-SPACE range (>= 1e9): the number of distinct KEYS that map to
        collision-free phasor addresses. NOT the STORAGE capacity -- values live in the
        SDM's M hard locations (self.M, default 20000) with k-of-M sparse superposition,
        so reliable recall is bounded by the SDM's crosstalk floor (~M/k order), NOT 1e9.
        This is the honest split: 1e9 addressing, SDM-bounded storage."""
        return self.rhc.range

    def store_capacity(self):
        """The store's real reliable-recall bound (SDM hard-location count), not the
        address range. Recall degrades past the SDM crosstalk floor."""
        return self.M


class HashedReachStore:
    """Day 95 -- SUBSTRATE-NATIVE transitive membership at flat compute (the biological answer to
    F0: reasoning on the substrate, not a dict walk).

    An item (x is a transitive descendant of c) is addressed by its bound phasor addr = key(x) *
    key(c). Instead of scanning M hard locations (brute SDM = O(M*d)), g cheap phasor-projection
    HASHES map addr to g of B fixed distributed counter buckets in O(g*d) -- no keyed dict, no
    M-scan. CONSOLIDATE writes the derived closure once (like sleep); MEMBER reads one bank and
    gates on an EMPIRICALLY-calibrated boundary -> correct-or-abstain (below boundary = abstain,
    never a confident lie). The bucket index is GEOMETRIC (phase of a projection), so the pair is
    never a dict key -- this is genuine HDC, a fixed-size distributed store.

    Measured (experiments/nl/day95_biological_reach.py): 100% recall / 1% false-accept to ~18k
    pairs at ~113 us/query, FLAT in store size. Capacity ~O(B); B is a design knob. RAM = B*d.
    """

    def __init__(self, ck=None, d=1024, B=8192, g=16, seed=95):
        # Reach bank runs in its OWN dimension (default d=1024), DECOUPLED from the production
        # substrate dim (often 400): it maps entity NAMES -> its own deterministic keys, so its
        # capacity/SNR is not throttled by the reasoning substrate's smaller d. Pass a ck only to
        # share an identity space explicitly.
        if ck is not None:
            self.ck = ck
            self.d = int(ck.d)
        else:
            from ikigai.cognition.flat_memory import ComputedKey
            self.d = int(d)
            self.ck = ComputedKey(d=self.d, seed=int(seed))
        self.B = int(B)
        self.g = int(g)
        r = np.random.default_rng(int(seed))
        self.R = np.exp(1j * r.uniform(-np.pi, np.pi, (self.g, self.d))).astype(np.complex64)
        self.C = np.zeros((self.B, self.d), dtype=np.complex64)
        self.boundary = 0.0
        self.n = 0
        self._shift = max(1, self.d // 3)      # permutation offset -> DIRECTED bind (see _addr)
        self._kcache = {}

    def _key(self, w):
        w = str(w).strip().lower()
        k = self._kcache.get(w)
        if k is None:
            k = np.asarray(self.ck.key(w), np.complex64).reshape(-1)
            self._kcache[w] = k
        return k

    def _addr(self, x, c):
        # DIRECTED edge (x -> ancestor c). bind (phase-add) is COMMUTATIVE, so key(x)*key(c) can't
        # tell (x is-ancestor-of c) from (c is-ancestor-of x). Permute the ancestor key first
        # (np.roll = a fixed permutation, self-inverse-free), so addr(x,c) != addr(c,x). Standard
        # VSA directed binding (Plate/Kanerva use permutation for order/role).
        return (self._key(x) * np.roll(self._key(c), self._shift)).astype(np.complex64)

    def _buckets(self, addr):
        ph = np.angle(self.R @ addr)                                  # g phases, O(g*d)
        return (((ph + np.pi) / (2.0 * np.pi)) * self.B).astype(np.int64) % self.B

    def _read(self, addr):
        v = self.C[self._buckets(addr)].sum(axis=0)
        nv = float(np.linalg.norm(v))
        return 0.0 if nv < 1e-9 else float(np.real(np.vdot(addr, v)) / (nv * np.sqrt(self.d)))

    def write_pair(self, x, c):
        addr = self._addr(x, c)
        self.C[self._buckets(addr)] += addr
        self.n += 1

    def calibrate(self, absent_pairs, pct=99.5, margin=1.15):
        """Set the abstain boundary from a sample of ABSENT (non-member) pairs. The IN-SAMPLE
        pct-th percentile underestimates the held-out tail at scale, so we take pct=99.5 and a
        small multiplicative margin -> held-out false-accept stays ~1% instead of drifting to 2%.
        Present pairs read ~1.0 vs a boundary ~0.05, so this costs ~no recall. Raise pct/margin for
        stricter abstention (less lying, more abstaining); lower for more recall."""
        sims = [self._read(self._addr(x, c)) for (x, c) in absent_pairs]
        self.boundary = (float(np.percentile(sims, pct)) * float(margin)) if sims else 0.0
        return self.boundary

    def score(self, x, c):
        return self._read(self._addr(x, c))

    def member(self, x, c):
        """True iff (c is a transitive ancestor of x) reads above the calibrated boundary.
        Below the boundary -> False (= abstain: statistically indistinguishable from not-stored)."""
        return self._read(self._addr(x, c)) >= self.boundary

    def substrate_bytes(self):
        return self.R.nbytes + self.C.nbytes
