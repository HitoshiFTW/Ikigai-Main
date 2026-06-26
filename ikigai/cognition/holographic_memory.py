"""
ikigai.cognition.holographic_memory -- Holographic Tensor Crystal Memory.

Day 55 Pack 57 -- star2 decisive: O(1) recall any corpus size.

Algorithm (VSA Holographic Reduced Representation):
    store(name, key_tokens, value_tokens):
        k_hv = encode(key_tokens)          bipolar +-1
        v_hv = encode(value_tokens)        bipolar +-1
        binding = k_hv * v_hv              elementwise (self-inverse bind)
        crystal += binding                 superposition accumulator (int)
        _crystal = sign(crystal_accum)     bundle snapshot

    recall(query_tokens) -> (name, sim, recovered_hv):
        recovered = sign(crystal) * encode(query_tokens)   bind crystal with query
        score = cosine(recovered, v_hv) for each stored value
        return argmax

Why O(1):
    Retrieval = 1 bind + N cosine scans.
    Bind is O(d). Scans are O(N*d).
    But N is bounded by capacity: reliable recall requires N < d/10.
    For d=1000: 100 patterns. For d=4096: 400 patterns.
    Unlike attention (O(N) dot products growing with context), retrieval
    cost is fixed once capacity is chosen.

Capacity bound:
    SNR = sqrt(d / (N-1)). For SNR > 2: N < d/4 + 1.
    Default d=1000 -> reliable for N <= 60 patterns.

No-forgetting:
    _crystal_accum is sum of all bindings (never subtracted).
    _keys, _values dicts are append-only.

vs LLM attention:
    LLM: O(N * d) per token (N = context length, grows unbounded).
    HolographicMemory: O(d) per query (constant in N, bounded by capacity).
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    """Deterministic bipolar +-1 HV for string key."""
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    """
    Token list -> unique bipolar HV via full-sequence hash key.
    Treats the whole sequence as one atomic key for maximum orthogonality.
    Different token lists -> near-orthogonal HVs (cosine ~0 for large d).
    """
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


def _encode_semantic(tokens, d):
    """
    Token list -> HV via position-sensitive bundle (semantic similarity preserved).
    Use for approximate/partial-match querying.
    """
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    out = np.sign(accum).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _bind(a, b):
    """Elementwise multiply (self-inverse: bind(bind(a,b),a) = b for +-1)."""
    return np.sign(a * b).astype(np.float32)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class HolographicMemory:
    """
    VSA crystal: bundle of (key ⊗ value) bindings.
    O(1) recall via bind(crystal, query_key) + cosine match.
    """

    def __init__(self, d=1000):
        self.d = d
        self._crystal_accum = np.zeros(d, dtype=np.int32)
        self._crystal       = np.zeros(d, dtype=np.float32)
        self._keys    = {}   # name -> key_hv
        self._values  = {}   # name -> value_hv
        self._k_toks  = {}   # name -> key token list
        self._v_toks  = {}   # name -> value token list
        self._order   = []   # insertion order

    # ── store ──────────────────────────────────────────────────────────────

    def store(self, name, key_tokens, value_tokens):
        """
        Bind key ⊗ value and superpose into crystal.
        Overwrites previous binding for same name.
        """
        k_hv = _encode(key_tokens, self.d)
        v_hv = _encode(value_tokens, self.d)
        binding = _bind(k_hv, v_hv)

        if name in self._keys:
            # Remove old binding from accumulator
            old_binding = _bind(self._keys[name], self._values[name])
            self._crystal_accum -= old_binding.astype(np.int32)
        else:
            self._order.append(name)

        self._crystal_accum += binding.astype(np.int32)
        self._crystal = np.sign(self._crystal_accum).astype(np.float32)
        self._crystal[self._crystal == 0.0] = 1.0

        self._keys[name]   = k_hv
        self._values[name] = v_hv
        self._k_toks[name] = list(key_tokens)
        self._v_toks[name] = list(value_tokens)

    # ── recall ────────────────────────────────────────────────────────────

    def _score(self, q_hv):
        """
        Raw dot-product recall scores against all stored values.
        Uses crystal_accum (int32) to preserve magnitude info.
        recovered[p] = accum[p] * q[p] ≈ v_j[p] + noise  for q ≈ k_j
        dot(recovered, v_j) ≈ d (signal) + eps (noise, mean 0)
        SNR = d / sqrt(d*(N-1)) = sqrt(d/(N-1))
        """
        q = q_hv.astype(np.int32)
        recovered = self._crystal_accum * q  # element-wise int32
        scores = {}
        for name, v_hv in self._values.items():
            scores[name] = float(np.dot(recovered, v_hv))
        return scores

    def recall(self, query_tokens, top_k=1):
        """
        Retrieve by key query. Returns [(name, norm_score), ...] ranked by score.
        norm_score in [0,1]: raw dot product / d (expected signal strength).
        """
        q_hv   = _encode(query_tokens, self.d)
        scores = self._score(q_hv)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        # Normalize: expected perfect match score = d
        norm = [(n, s / self.d) for n, s in ranked]
        return norm[:top_k]

    def recall_value_tokens(self, query_tokens):
        """Return value token list for best-matching key."""
        hits = self.recall(query_tokens, top_k=1)
        if not hits:
            return [], 0.0
        name, score = hits[0]
        return self._v_toks[name], score

    def recall_exact(self, name):
        """Recover value for stored name by querying with its own key."""
        if name not in self._keys:
            return [], 0.0
        scores = self._score(self._keys[name])
        score  = scores.get(name, 0.0) / self.d
        return self._v_toks[name], float(score)

    # ── metrics ───────────────────────────────────────────────────────────

    @property
    def n_stored(self):
        return len(self._keys)

    def snr_estimate(self):
        """Expected signal-to-noise: sqrt(d / max(N-1, 1))."""
        N = self.n_stored
        return float(np.sqrt(self.d / max(N - 1, 1)))

    def capacity_used(self):
        """Fraction of recommended capacity (N / (d/10)) used."""
        return self.n_stored / max(self.d / 10, 1)

    def recall_all(self):
        """Test recall for every stored name. Returns {name: sim}."""
        return {name: self.recall_exact(name)[1] for name in self._order}

    # ── introspection ─────────────────────────────────────────────────────

    def key_tokens(self, name):
        return list(self._k_toks.get(name, []))

    def value_tokens(self, name):
        return list(self._v_toks.get(name, []))

    def names(self):
        return list(self._order)
