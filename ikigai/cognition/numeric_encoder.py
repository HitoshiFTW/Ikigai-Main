"""
ikigai.cognition.numeric_encoder -- Pack 252 Fractional Power Encoding.

Day 73. Magnitude-aware numeric primitive for the FHRR substrate.

Problem
-------
Today the substrate maps every token (incl. number words "five", "fifty")
to a random orthogonal phasor via ConceptKeys. cos(key("five"), key("six"))
~ 0; cos(key("five"), key("fifty")) ~ 0. No magnitude topology = no
arithmetic.

Fix
---
Fractional Power Encoding (Plate 1995, Komer + Eliasmith 2019). Fix a
base random phasor Z (d-dim, unit complex). For scalar x define:
    Z^x[k] = exp(i * x * phase_k)   for k in 0..d-1
Properties:
    Z^0           = 1               (identity)
    Z^x * Z^y     = Z^(x+y)         (group homomorphism, Hadamard product)
    cos(Z^x, Z^y) is a function of (x - y) only, peaks at 0 and falls
        smoothly with |x - y|  ->  natural magnitude topology

Numbers close to each other have high HV similarity. Numbers far apart
have low. Arithmetic becomes substrate-readable.

Integration with NeuroSeed
--------------------------
This encoder is independent of ConceptKeys. Number tokens still get
their random phasor via ck.key (so word-level absorb keeps working).
The numeric magnitude HV is wired through a NEW relation role
'magnitude': mr.relate("five", "magnitude", <numeric HV>).

When math-aware queries are needed:
    five_mag = mr.recall("five", "magnitude")
    five_decoded = num_enc.decode(five_mag) -> ~5

Or directly: num_enc.encode(5) yields the canonical magnitude HV.

The encoder's `phases` vector IS the model parameter. Save/load via
seed (deterministic) or explicit array.

NO hardcoded word -> int mapping
--------------------------------
parse_number ONLY recognizes digit strings ('5', '42', '-7'). Number
WORDS ('five', 'twenty-five') do NOT get magnitude from a built-in
table. They acquire magnitude EMERGENTLY through cat-3 reasoning-chain
absorb, where the LLM teacher emits chains containing both digit and
word forms in equivalent positions. The substrate learns 'five <-> 5'
via co-occurrence + parallel-context binding -- not from a curated
lookup.
"""

import numpy as np


def parse_number(tok: str):
    """Parse a single token as an integer. Returns int or None.

    Recognizes digit strings only: '5', '42', '-7'. Returns None for
    any non-digit token, including number words. Number words gain
    magnitude through substrate absorb, not via a curated table.
    """
    t = str(tok).strip()
    if not t:
        return None
    try:
        if t.isdigit() or (t.startswith('-') and t[1:].isdigit()):
            return int(t)
    except ValueError:
        pass
    return None


class NumericEncoder:
    """FPE encoder. Maps scalar x -> d-dim complex64 phasor."""

    def __init__(self, d=400, scale=10.0, seed=2520):
        """
        Args:
            d     -- substrate dim (match mr.d)
            scale -- divisor on x to keep phases unwrapped in working range.
                      With scale=10 and d=400 random phases in [-pi,pi],
                      cos(Z^x, Z^y) falls below 0.1 around |x-y| ~ 30,
                      reaches noise floor by |x-y| ~ 100. Good span for
                      integer arithmetic up to ~1e4.
            seed  -- rng for the base phase vector. Deterministic.
        """
        self.d = int(d)
        self.scale = float(scale)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.phases = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)

    # ---- encode / decode ----------------------------------------------

    def encode(self, x: float) -> np.ndarray:
        """Z^x as a complex64 phasor."""
        ph = (float(x) / self.scale) * self.phases
        return np.exp(1j * ph).astype(np.complex64)

    def encode_token(self, tok: str):
        """tok -> numeric HV if parseable, else None."""
        n = parse_number(tok)
        if n is None:
            return None
        return self.encode(float(n))

    def similarity(self, x: float, y: float) -> float:
        """Analytic cos(Z^x, Z^y). Useful for sanity checks."""
        a = self.encode(x); b = self.encode(y)
        return float((np.conj(a) * b).mean().real)

    def decode(self, hv: np.ndarray, x_min=-50, x_max=200, step=1) -> float:
        """Find x that maximizes cos(hv, Z^x). Brute force on grid."""
        hv = np.asarray(hv, dtype=np.complex64)
        xs = np.arange(x_min, x_max + step, step, dtype=np.float32)
        best_x = xs[0]; best_s = -1e9
        # vectorize: build all candidates as (N, d) matrix
        cands = np.exp(1j * (xs[:, None] / self.scale) * self.phases[None, :])
        cands = cands.astype(np.complex64)
        sims = (np.conj(cands) @ hv).real / self.d
        idx = int(np.argmax(sims))
        return float(xs[idx]), float(sims[idx])

    # ---- arithmetic on phasors ----------------------------------------

    def add(self, hv_a, hv_b):
        """Z^a * Z^b = Z^(a+b). Hadamard product."""
        return (np.asarray(hv_a, dtype=np.complex64)
                 * np.asarray(hv_b, dtype=np.complex64)).astype(np.complex64)

    def sub(self, hv_a, hv_b):
        """Z^a * conj(Z^b) = Z^(a-b)."""
        return (np.asarray(hv_a, dtype=np.complex64)
                 * np.conj(np.asarray(hv_b, dtype=np.complex64))
                 ).astype(np.complex64)

    def neg(self, hv):
        """Z^(-x) = conj(Z^x)."""
        return np.conj(np.asarray(hv, dtype=np.complex64)).astype(np.complex64)

    # ---- persistence --------------------------------------------------

    def __getstate__(self):
        return {'d': self.d, 'scale': self.scale, 'seed': self.seed,
                 'phases': self.phases}

    def __setstate__(self, st):
        self.d = int(st['d'])
        self.scale = float(st['scale'])
        self.seed = int(st['seed'])
        self.phases = np.asarray(st['phases'], dtype=np.float32)


class SpatialScene:
    """Day 94 -- a SCENE on the FHRR substrate via Spatial Semantic Pointers (Komer & Eliasmith).

    The SAME substrate that holds symbolic taxonomy holds CONTINUOUS 2-D position: an object at
    (x,y) is the phasor SSP(x,y) = X^x (.) Y^y (Hadamard of two Fractional-Power-Encoded axes).  A
    scene is ONE holographic memory  M = SUM_o  name_o (.) SSP(x_o,y_o).  Position is read back by
    unbinding (resonance); PROXIMITY is substrate similarity (no coordinates compared); and the
    base spatial relations feed the organism's existing derive engine, which composes them
    (right-of / above / between) exactly as it composes is-a.  Generalization off the text axis --
    zero new mechanism.

        sc = SpatialScene(); sc.add('a', 3, 5).add('b', 9, 2)
        sc.where('a')                 -> (3, 5)       (decode from the scene memory)
        sc.nearest(4, 6)              -> 'a'          (SSP resonance)
        org.ingest_triples(sc.base_facts(), discover=True)   -> derive-engine spatial composition
    """

    def __init__(self, d=1024, scale=6.0, seed=101):
        self.d = int(d)
        self.encX = NumericEncoder(d=d, scale=scale, seed=seed)
        self.encY = NumericEncoder(d=d, scale=scale, seed=seed + 101)
        self.objs = {}            # name -> (x, y)
        self._ssp = {}            # name -> SSP phasor

    def ssp(self, x, y):
        """Spatial Semantic Pointer for a point: bind the two axis codes (Hadamard product)."""
        return (self.encX.encode(float(x)) * self.encY.encode(float(y))).astype(np.complex64)

    def add(self, name, x, y):
        name = str(name).strip().lower()
        self.objs[name] = (float(x), float(y))
        self._ssp[name] = self.ssp(x, y)
        return self

    def _name_hv(self, name):
        r = np.random.default_rng(abs(hash(('ssp:name', name))) % (2 ** 31))
        return np.exp(1j * r.uniform(-np.pi, np.pi, self.d)).astype(np.complex64)

    def memory(self):
        """The whole scene as ONE holographic phasor memory."""
        M = np.zeros(self.d, dtype=np.complex64)
        for n in self.objs:
            M += self._name_hv(n) * self._ssp[n]
        return M

    def where(self, name, x_max=None, y_max=None):
        """Recover an object's (x,y) by unbinding the scene memory and decoding on a grid."""
        name = str(name).strip().lower()
        if name not in self.objs:
            return None
        M = self.memory()
        rec = M * np.conj(self._name_hv(name))
        xm = int(x_max if x_max is not None else max((v[0] for v in self.objs.values()), default=16)) + 1
        ym = int(y_max if y_max is not None else max((v[1] for v in self.objs.values()), default=16)) + 1
        best, bxy = -1e9, (0, 0)
        for x in range(xm + 1):
            cx = self.encX.encode(x)
            for y in range(ym + 1):
                s = float((np.conj(cx * self.encY.encode(y)) * rec).mean().real)
                if s > best:
                    best, bxy = s, (x, y)
        return bxy

    def nearest(self, x, y):
        """Nearest object to a point by SSP RESONANCE (proximity = substrate similarity)."""
        if not self.objs:
            return None
        q = self.ssp(x, y)
        return max(self.objs, key=lambda n: float((np.conj(self._ssp[n]) * q).mean().real))

    def base_facts(self):
        """Immediate 'rightof' (x) and 'above' (y) neighbor chains -- feed to the derive engine,
        whose transitive closure + composition then answers right-of / above / between queries."""
        names = list(self.objs)
        bx = sorted(names, key=lambda n: self.objs[n][0])
        by = sorted(names, key=lambda n: self.objs[n][1])
        return ([(bx[i], 'rightof', bx[i + 1]) for i in range(len(bx) - 1)] +
                [(by[i], 'above', by[i + 1]) for i in range(len(by) - 1)])
