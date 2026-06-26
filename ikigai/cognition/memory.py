"""
ikigai.cognition.memory — VSA associative memory primitives.

Houses:
    TransitionMemory  — Hebbian outer-product associative matrix
                        (Day 54 Pack 21, Rank 3 invention).
                        Stores (prev -> curr) transitions as a single 400x400
                        int32 matrix. Constant memory regardless of N records.
                        Supports analogical query via bundled inputs.
"""

import random
import numpy as np


HV_DIM = 400


def _cosine(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TransitionMemory:
    """
    Outer-product associative memory.
      record(prev, curr): M += outer(prev_hv, curr_hv)
      predict(prev):      query_hv @ M  -> nearest vocab entry
    """

    def __init__(self, dim=HV_DIM):
        self.dim = dim
        self.M = np.zeros((dim, dim), dtype=np.int32)
        self.vocab = {}        # state_key -> bipolar HV (int8, +-1)
        self.n_records = 0

    def _make_hv(self, key):
        rng = random.Random(hash(f'__state__{key}') & 0x7FFFFFFF)
        return np.array(
            [1 if rng.randint(0, 1) else -1 for _ in range(self.dim)],
            dtype=np.int8,
        )

    def add_state(self, key):
        if key not in self.vocab:
            self.vocab[key] = self._make_hv(key)
        return self.vocab[key]

    def record(self, prev_key, curr_key):
        """Hebbian outer product."""
        p = self.add_state(prev_key).astype(np.int32)
        c = self.add_state(curr_key).astype(np.int32)
        self.M += np.outer(p, c)
        self.n_records += 1

    def predict_hv(self, query_hv):
        """Vector-matrix product."""
        return query_hv.astype(np.int32) @ self.M

    def predict(self, prev_key, exclude=None):
        if prev_key not in self.vocab:
            return None, 0.0
        q = self.predict_hv(self.vocab[prev_key])
        excl = set(exclude or [])
        best_k, best_sim = None, -2.0
        for k, h in self.vocab.items():
            if k in excl:
                continue
            s = _cosine(q, h)
            if s > best_sim:
                best_sim = s; best_k = k
        return best_k, best_sim

    def predict_from_hv(self, query_hv, exclude=None):
        q = self.predict_hv(query_hv)
        excl = set(exclude or [])
        best_k, best_sim = None, -2.0
        for k, h in self.vocab.items():
            if k in excl:
                continue
            s = _cosine(q, h)
            if s > best_sim:
                best_sim = s; best_k = k
        return best_k, best_sim

    def matrix_bytes(self):
        return int(self.M.nbytes)
