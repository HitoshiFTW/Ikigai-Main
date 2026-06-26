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
