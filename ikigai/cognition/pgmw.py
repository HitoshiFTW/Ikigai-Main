"""
ikigai.cognition.pgmw -- Persona Grid Metric Warp.

Day 56 Pack 86 -- Pillar 3 of UHE.

Same memory matrix M. Different lens p (persona) -> different retrieval.

Mathematical core:
    persona p ∈ ℂ^d (unit norm)
    warped metric: M(p) = I + lambda * (p p^H - I/d)
    warped inner product: <x, y>_p = x^H M(p) y
                                   = x^H y + lambda * (<x,p><p,y> - <x,y>/d)
    score(x, y; p) = <x, y> + lambda * (<x,p> * <y,p> - <x,y>/d)
                   = (1 - lambda/d) * <x, y> + lambda * <x,p> * conj(<y,p>)

Effect:
    - lambda = 0: identity metric (no persona effect)
    - lambda > 0: emphasize subspace aligned with p
    - lambda < 0: suppress subspace aligned with p
    - rank-k extension: M(p_1, ..., p_k) = I + sum_i lambda_i * (p_i p_i^H - I/d)

Combines with Pack 66 ImportanceDecayLattice:
    final_score(q, m_i, t, p) = sigma_i(t) * <q, m_i>_p
"""

import numpy as np


def _cdot(a, b):
    """Complex inner product Re<a, b>."""
    return float(np.real(np.vdot(a, b)))


def _normalize(v):
    # Component-wise unit phasor normalization -- matches lexicon HV convention.
    # Each component gets magnitude 1 (not whole-vector L2=1).
    mags = np.abs(v)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (v / mags).astype(np.complex64)


class PersonaGrid:
    """
    Rank-k metric warp over memory retrieval.

    add_persona(name, vector, lam=0.5)
        Add a persona with mixing weight.
    set_active(*names)
        Activate one or more personas (compose their metric perturbations).
    warped_cosine(x, y)
        Cosine in the warped inner product space.
    retrieve(query, memory_keys, top_k=5)
        Top-k retrieval scored under current active persona metric.
    persona_from_examples(name, examples_hvs, lam=0.5)
        Compute persona vector as mean-normalized of example HVs.
    """

    def __init__(self, d=2048):
        self.d = int(d)
        self._personas = {}  # name -> (vector, lam)
        self._active   = []  # list of names

    #  persona definition

    def add_persona(self, name, vector, lam=0.5):
        v = np.asarray(vector)
        if v.shape != (self.d,):
            raise ValueError(f'persona dim mismatch: got {v.shape}, expected ({self.d},)')
        v = _normalize(v.astype(np.complex64))
        self._personas[name] = (v, float(lam))
        return v

    def persona_from_examples(self, name, examples_hvs, lam=0.5):
        """Mean of examples (then normalize)."""
        if not examples_hvs:
            raise ValueError('need at least one example')
        mean = np.zeros(self.d, dtype=np.complex64)
        for h in examples_hvs:
            mean = mean + np.asarray(h, dtype=np.complex64)
        return self.add_persona(name, mean, lam)

    def remove_persona(self, name):
        self._personas.pop(name, None)
        if name in self._active:
            self._active.remove(name)

    #  activation

    def set_active(self, *names):
        for n in names:
            if n not in self._personas:
                raise ValueError(f'unknown persona: {n}')
        self._active = list(names)
        return self._active

    def clear_active(self):
        self._active = []

    @property
    def active(self):
        return list(self._active)

    #  warped inner product

    def warped_inner(self, x, y):
        """
        <x, y>_M = <x, y> + sum_i lam_i * (<x, p_i> * conj(<y, p_i>) - <x, y>/d)
        Real-valued (returns Re part).
        """
        x = np.asarray(x, dtype=np.complex64)
        y = np.asarray(y, dtype=np.complex64)
        base = _cdot(x, y)
        if not self._active:
            return base
        bonus = 0.0
        total_lam = 0.0
        for name in self._active:
            p, lam = self._personas[name]
            xp = np.vdot(p, x)   # <p, x> = sum p_conj * x  (complex)
            yp = np.vdot(p, y)
            bonus += lam * float(np.real(xp * np.conj(yp)))
            total_lam += lam
        return (1.0 - total_lam / self.d) * base + bonus

    def warped_cosine(self, x, y):
        """Warped inner product normalized by warped norms."""
        sxx = self.warped_inner(x, x)
        syy = self.warped_inner(y, y)
        sxy = self.warped_inner(x, y)
        if sxx <= 0 or syy <= 0:
            return 0.0
        return sxy / np.sqrt(sxx * syy)

    #  retrieval

    def retrieve(self, query, memory_items, top_k=5):
        """
        memory_items: list of (name, hv) tuples
        Returns [(name, warped_cos), ...] sorted descending.
        """
        results = []
        for name, m in memory_items:
            sim = self.warped_cosine(query, m)
            results.append((name, float(sim)))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    #  introspection

    @property
    def n_personas(self):
        return len(self._personas)

    def persona_vector(self, name):
        e = self._personas.get(name)
        return e[0] if e else None

    def persona_names(self):
        return list(self._personas.keys())
