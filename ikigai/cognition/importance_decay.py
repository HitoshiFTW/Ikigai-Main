"""
ikigai.cognition.importance_decay -- Importance-Weighted Decay Lattice.

Day 55 Pack 66 -- invention #6: bio-realistic forgetting curve.
Half-life ∝ surprise × frequency. Important items persist; routine items fade.

Ebbinghaus forgetting curve: memory strength s(t) = s_0 * exp(-t / tau)
Standard half-life tau is fixed. Bio reality: tau scales with importance.

Importance model:
    imp_i = alpha * surprise_i + beta * log(1 + freq_i)
    tau_i = tau_0 * (1 + imp_i)
    strength_i(t) = strength_0_i * exp(-(t - last_access_i) / tau_i)

Each access:
    - increments freq
    - refreshes last_access
    - if surprise > prev_surprise: bump surprise (EMA)
    - recompute tau

Lattice:
    Items live in priority queue by current_strength().
    decay(now) prunes items below threshold.
    Free-energy budget: total strength bounded.

Bio analog:
    Hippocampus -> cortex consolidation. Repeated novel events become permanent.
    Routine events compete for space; surprising events get protected slots.

vs LLM: no forgetting curve. Context window forgets uniformly by token position.
        Decay lattice: surprise + frequency drive retention. Permanent for important.
"""

import math
import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


class DecayItem:
    """One lattice node: HV + freq + surprise + last_access."""

    __slots__ = (
        'name', 'hv', 'freq', 'surprise', 'last_access', 'strength_0',
    )

    def __init__(self, name, hv, surprise, t):
        self.name        = name
        self.hv          = hv
        self.freq        = 1
        self.surprise    = float(surprise)
        self.last_access = int(t)
        self.strength_0  = 1.0

    def importance(self, alpha=1.0, beta=0.5):
        """imp = alpha * surprise + beta * log(1 + freq)."""
        return alpha * self.surprise + beta * math.log1p(self.freq)

    def tau(self, tau_0=100.0, alpha=1.0, beta=0.5):
        """Half-life in ticks. Scales with importance."""
        return tau_0 * (1.0 + self.importance(alpha, beta))

    def strength(self, now, tau_0=100.0, alpha=1.0, beta=0.5):
        """exp(-(now - last_access) / tau)."""
        dt = max(0, now - self.last_access)
        return self.strength_0 * math.exp(-dt / self.tau(tau_0, alpha, beta))


class ImportanceDecayLattice:
    """
    Importance-weighted memory lattice with exponential forgetting.

    record(name, tokens, surprise=0.0, now=None)
        Insert / refresh an item. Surprise EMA-merged.

    strength(name, now) -> float
        Current strength in [0, 1].

    prune(now, threshold=0.1) -> list of pruned names
        Drop items below strength threshold.

    rank(now, top_k=None) -> [(name, strength), ...]
        Items ordered by current strength.

    half_lives() -> {name: tau}
        Per-item tau snapshot.

    budget() -> float
        Sum of current strengths (proxy for 'memory load').
    """

    def __init__(self, d=400, tau_0=100.0, alpha=1.0, beta=0.5, surprise_ema=0.3):
        self.d            = d
        self.tau_0        = float(tau_0)
        self.alpha        = float(alpha)
        self.beta         = float(beta)
        self.surprise_ema = float(surprise_ema)
        self._items       = {}    # name -> DecayItem
        self._tick        = 0

    #  record / refresh

    def record(self, name, tokens, surprise=0.0, now=None):
        """Insert or refresh. Subsequent calls: freq++, surprise EMA-merge."""
        if now is None:
            now = self._tick
        if name in self._items:
            it           = self._items[name]
            it.freq     += 1
            it.last_access = int(now)
            # Surprise EMA: new = old + ema * (now_surprise - old)
            it.surprise = (
                it.surprise + self.surprise_ema * (float(surprise) - it.surprise)
            )
        else:
            hv = _encode(tokens, self.d)
            self._items[name] = DecayItem(name, hv, surprise, now)
        return self._items[name]

    def advance(self, n=1):
        """Advance internal tick counter."""
        self._tick += int(n)
        return self._tick

    #  strength + ranking

    def strength(self, name, now=None):
        if now is None:
            now = self._tick
        it = self._items.get(name)
        if it is None:
            return 0.0
        return it.strength(now, self.tau_0, self.alpha, self.beta)

    def rank(self, now=None, top_k=None):
        if now is None:
            now = self._tick
        results = [(n, self.strength(n, now)) for n in self._items]
        results.sort(key=lambda x: -x[1])
        if top_k is not None:
            results = results[:top_k]
        return results

    #  pruning

    def prune(self, now=None, threshold=0.1):
        """Drop items below strength threshold. Returns list of pruned names."""
        if now is None:
            now = self._tick
        pruned = []
        for name in list(self._items.keys()):
            if self.strength(name, now) < threshold:
                pruned.append(name)
                del self._items[name]
        return pruned

    #  introspection

    def half_lives(self):
        return {
            n: it.tau(self.tau_0, self.alpha, self.beta)
            for n, it in self._items.items()
        }

    def importances(self):
        return {n: it.importance(self.alpha, self.beta) for n, it in self._items.items()}

    def budget(self, now=None):
        if now is None:
            now = self._tick
        return float(sum(self.strength(n, now) for n in self._items))

    @property
    def n_items(self):
        return len(self._items)

    @property
    def tick(self):
        return self._tick

    def item(self, name):
        return self._items.get(name)
