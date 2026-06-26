"""
ikigai.cognition.qfhrr -- Pack 259 quaternion FHRR base ring.

Day 74. Quaternion-valued FHRR variant. Each location holds a unit
quaternion (4 reals) instead of a complex64 phasor (2 reals). Binding
is the Hamilton (non-commutative) product; unbinding is bind with
the conjugate.

WHY THIS EXISTS
---------------
Pack 252 NumericEncoder + the substrate use complex64 FHRR. Capacity
per location ~ 2 angular dimensions. Quaternion FHRR carries 4
angular dimensions per location. Per Schlegel et al. (2022) "A
Comparison of Vector Symbolic Architectures", quaternion VSA has
strictly higher binding capacity per real-number budget than complex
VSA when codebook size is large.

Per Day 73 close inventions roadmap, Pack 259 is a candidate base
ring for Invention 1 HPE (Hierarchical Phase Encoding). HPE recurses
phase nesting; qFHRR can provide a 2x density floor before HPE
recursion compounds.

BIND IS NON-COMMUTATIVE
-----------------------
Hamilton product is associative but NOT commutative: bind(a, b) !=
bind(b, a) in general. This is a FEATURE -- enables ordered binding
without permutation tricks. For symmetric semantics use the symmetric
product 0.5*(bind(a,b) + bind(b,a)).

UNITS
-----
We store unit quaternions throughout. Unit quaternions form a group
under Hamilton product (the 3-sphere S^3). All operations preserve
the unit norm; renormalize after sums.

NOT A NEW SUBSTRATE
-------------------
Pack 259 is a primitive in its own right but is meant to plug into
HPE later. Standalone use: binding+unbinding+similarity over toy
codebooks for capacity benchmarks vs complex FHRR.
"""

import numpy as np


def _hamilton(q1, q2):
    """Quaternion Hamilton product. Inputs shape (..., 4); output (..., 4)."""
    a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        a*e - b*f - c*g - d*h,
        a*f + b*e + c*h - d*g,
        a*g - b*h + c*e + d*f,
        a*h + b*g - c*f + d*e,
    ], axis=-1).astype(np.float32)


def _conjugate(q):
    """Quaternion conjugate (w, -x, -y, -z)."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]],
                     axis=-1).astype(np.float32)


def _renorm(q):
    """Renormalize each quaternion to unit norm."""
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return (q / n).astype(np.float32)


def _cosine(q1, q2):
    """Mean inner product per location. For unit quaternions in [-1, 1].
    Cos=1 = same. Cos=0 = orthogonal."""
    return float(np.mean(np.sum(q1 * q2, axis=-1)))


class QFHRR:
    """Pack 259 quaternion FHRR encoder + ops.

    USAGE
    -----
        q = QFHRR(d=400, seed=259)
        A = q.random_unit()
        B = q.random_unit()
        C = q.bind(A, B)          # Hamilton product per location
        A_hat = q.unbind(C, B)    # Hamilton product with conj(B)
        sim = q.cosine(A_hat, A)  # ~= 1.0
    """

    def __init__(self, d, seed=259):
        self.d = int(d)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    # ---- ring ops ----------------------------------------------------

    def random_unit(self):
        """Sample d random unit quaternions (uniform on S^3 per location)."""
        # Gaussian then renormalize -> uniform on S^3
        x = self._rng.standard_normal((self.d, 4)).astype(np.float32)
        return _renorm(x)

    def bind(self, q1, q2):
        return _hamilton(q1, q2)

    def unbind(self, q_bound, q_factor):
        """Recover the OTHER factor: bind(unbind(c, b), b) ~= c."""
        return _hamilton(q_bound, _conjugate(q_factor))

    def conjugate(self, q):
        return _conjugate(q)

    def renorm(self, q):
        return _renorm(q)

    def cosine(self, q1, q2):
        return _cosine(q1, q2)

    def superpose(self, *qs):
        """Element-wise sum then renormalize per location. Approximate
        superposition; loses info as #items grows."""
        if not qs:
            return np.zeros((self.d, 4), dtype=np.float32)
        s = np.stack(qs).sum(axis=0)
        return _renorm(s)

    # ---- capacity diagnostic ----------------------------------------

    def cleanup_decode(self, target_q, codebook):
        """Snap target to closest entry in codebook. codebook = dict
        {name: q(d,4)}. Returns (name, cos).
        """
        if not codebook:
            return None, -1.0
        names = list(codebook.keys())
        K = np.stack([codebook[n] for n in names])
        # Cosine per candidate
        sims = np.mean(np.sum(K * target_q[None, :, :], axis=-1), axis=-1)
        idx = int(np.argmax(sims))
        return names[idx], float(sims[idx])
