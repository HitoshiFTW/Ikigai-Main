"""
ikigai.cognition.sac_field -- Self-Assembling Convergence Field.

Day 56 Pack 87 -- Pillar 4 of UHE.

Energy landscape inference over hypervector manifold.
Each task category = basin attractor. Inference = gradient descent on energy.

Mathematical core:
    State s ∈ ℂ^d (phasor)
    Basin attractors: B_1, ..., B_K (one per task category)
    Pairwise field tensor per basin:  F_k^(2) = Σ_{i ∈ basin_k} q_i q_i^H

    Energy:
        E_local_k(s) = -Re<s, F_k^(2) s>     (pull toward basin k)
        E(s) = -Σ_k softmax(<s, B_k>).E_local_k(s)   (basin-conditional)

    Gradient (closed-form, no autograd):
        ∂E_local_k / ∂s = -2 . F_k^(2) . s

    Dynamics:
        s_{t+1} = normalize(s_t - η . ∇E(s_t))

Algorithm:
    1. Encode query -> s_0
    2. Step k=1..N: descend energy field
    3. Trajectory enters a basin; basin acts as attractor
    4. Final s_∞ decoded against codebook
"""

import numpy as np


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / max(n, eps) * np.sqrt(len(v))


def _cdot(a, b):
    return float(np.real(np.vdot(a, b)))


class BasinField:
    """
    Single basin: attractor center + rank-r outer-product field tensor (low-memory).

    Stored as a list of memory vectors {q_i, weight_i}. Field never materialized
    in full (d² could be 16 MB at d=2048). Instead computed on the fly:
        F . s = Σ_i w_i . q_i . <q_i, s>
    This is O(N . d) per matrix-vector product, far cheaper than O(d²).
    """

    def __init__(self, name, d):
        self.name      = name
        self.d         = d
        self._mems     = []          # list of (q, weight)
        self._center   = None        # mean attractor

    def add(self, q, weight=1.0):
        q = np.asarray(q, dtype=np.complex64)
        self._mems.append((q.copy(), float(weight)))
        # Update running center
        if self._center is None:
            self._center = q.copy()
        else:
            self._center = self._center + q
        return self

    def center(self):
        """Normalized attractor center."""
        if self._center is None:
            return None
        return _normalize(self._center.astype(np.complex64))

    def apply(self, s):
        """Compute F . s = Σ_i w_i . q_i . <q_i, s>."""
        if not self._mems:
            return np.zeros(self.d, dtype=np.complex64)
        out = np.zeros(self.d, dtype=np.complex64)
        for q, w in self._mems:
            coef = w * np.vdot(q, s)         # <q, s>  (complex)
            out  = out + coef * q
        return out

    def energy(self, s):
        """E = -Re<s, F.s>."""
        return -_cdot(s, self.apply(s))

    @property
    def n_examples(self):
        return len(self._mems)


class SACField:
    """
    Multi-basin energy landscape with self-assembling convergence dynamics.

    add_basin(name)              -> create new basin
    populate(name, examples)     -> add example HVs to basin
    classify(s, top_k=1)         -> top-k basins by attractor alignment
    descend(s, steps=20, eta=...) -> gradient-descend in current basin-weighted field
    converge(query)              -> full pipeline: classify + descend -> s_∞
    """

    def __init__(self, d=2048, temperature=1.0):
        self.d           = int(d)
        self.temperature = float(temperature)
        self._basins     = {}    # name -> BasinField

    #  basin construction

    def add_basin(self, name):
        if name not in self._basins:
            self._basins[name] = BasinField(name, self.d)
        return self._basins[name]

    def populate(self, name, examples, weight=1.0):
        b = self.add_basin(name)
        for ex in examples:
            b.add(ex, weight=weight)
        return b

    @property
    def n_basins(self):
        return len(self._basins)

    def basin_names(self):
        return list(self._basins.keys())

    def basin(self, name):
        return self._basins.get(name)

    #  classification by attractor cosine

    def basin_scores(self, s):
        """Return {basin_name: cosine to basin center}."""
        out = {}
        s_norm = float(np.linalg.norm(s))
        for name, b in self._basins.items():
            c = b.center()
            if c is None:
                out[name] = 0.0
                continue
            cn = float(np.linalg.norm(c))
            num = _cdot(s, c)
            out[name] = num / (s_norm * cn) if s_norm * cn > 0 else 0.0
        return out

    def classify(self, s, top_k=1):
        scores = self.basin_scores(s)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    #  softmax weights over basins (for joint field)

    def basin_weights(self, s):
        scores = self.basin_scores(s)
        names = list(scores.keys())
        vals = np.array([scores[n] for n in names], dtype=np.float32)
        # Softmax over basin alignments
        z = vals / max(self.temperature, 1e-6)
        z = z - z.max()    # stability
        exp_z = np.exp(z)
        w = exp_z / exp_z.sum()
        return dict(zip(names, w.tolist()))

    #  energy gradient

    def gradient(self, s):
        """
        ∇E(s) = -Σ_k w_k(s) . (F_k . s)
        Where w_k(s) is softmax over basin alignments.
        """
        weights = self.basin_weights(s)
        grad = np.zeros(self.d, dtype=np.complex64)
        for name, w in weights.items():
            if w < 1e-6:
                continue
            grad = grad - w * self._basins[name].apply(s)
        return grad

    def energy(self, s):
        weights = self.basin_weights(s)
        return -sum(w * (-self._basins[n].energy(s))
                    for n, w in weights.items() if w > 0)

    #  dynamics

    def descend(self, s, steps=20, eta=0.1, renormalize=True):
        """Gradient descent. Returns trajectory list of states."""
        traj = [s.copy()]
        cur = s.copy()
        for _ in range(steps):
            g = self.gradient(cur)
            cur = cur - eta * g
            if renormalize:
                # Project back onto phasor manifold (unit-magnitude each component)
                mags = np.abs(cur)
                mags = np.where(mags > 1e-9, mags, 1.0)
                cur = (cur / mags).astype(np.complex64)
            traj.append(cur.copy())
        return cur, traj

    def converge(self, s, steps=30, eta=0.1):
        """Full convergence: descend until stable or steps exhausted."""
        s_final, _ = self.descend(s, steps=steps, eta=eta)
        cls = self.classify(s_final, top_k=1)
        return s_final, cls[0] if cls else (None, 0.0)
