"""
ikigai.cognition.adversarial_immune -- Adversarial Immune System.

Day 55 Pack 67 -- invention #12: antibody HVs vs prompt injection / jailbreak.

Biological immune system:
    - B-cells produce antibodies for specific antigens
    - When antibody matches antigen -> immune response
    - Memory cells retain known threats; faster response on re-exposure
    - Self/non-self discrimination

Ikigai mapping:
    - Antibody HV = encoded threat pattern (jailbreak prompt template)
    - Antigen     = incoming user query
    - cosine(antibody, query) > threshold -> threat detected
    - threats logged with strength + last_seen
    - Memory cell = high-strength antibody that survived encounters

API:
    register_threat(name, tokens)          -> add antibody
    scan(query_tokens, threshold)          -> [(name, score), ...] of matches
    is_safe(query_tokens, threshold)       -> bool
    quarantine(query_tokens, reason)       -> log + return safe-fallback HV
    immunize(threat_name)                  -> bump antibody strength
    forget(threat_name)                    -> remove antibody (test only)

Memory cells:
    Each antibody has a 'strength' that grows with detected exposures.
    Strong antibodies trigger at lower cosine thresholds (more sensitive).

vs LLM: prompt-injection defense = expensive RLHF + filtering.
        Ikigai: O(N_threats) cosine scan. Zero training. Bio-aligned.
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode_semantic(tokens, d):
    """Position-sensitive bundle (recognizes partial prompt-injection patterns)."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    s = np.sign(accum).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _encode_bag(tokens, d):
    """Position-insensitive bundle (recognizes scrambled injection attempts)."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for tok in tokens:
        accum += _hv_for(f'{tok}', d).astype(np.int32)
    s = np.sign(accum).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Antibody:
    """Single immune cell: HV + strength + last-seen tick."""

    __slots__ = ('name', 'pos_hv', 'bag_hv', 'strength', 'exposures', 'last_seen')

    def __init__(self, name, pos_hv, bag_hv):
        self.name      = name
        self.pos_hv    = pos_hv
        self.bag_hv    = bag_hv
        self.strength  = 1.0
        self.exposures = 0
        self.last_seen = -1


class AdversarialImmune:
    """
    Antibody registry + scan + quarantine.

    register_threat(name, tokens)
        Encode threat as (pos, bag) HV pair. Position-sensitive AND scrambled.

    scan(query_tokens, threshold=0.4)
        Returns [(name, max_score), ...] for antibodies matching query.

    is_safe(query_tokens, threshold)
        True iff no antibody scores >= threshold.

    quarantine(query_tokens, reason='auto')
        Log + return safe-fallback HV (zeros, signaling rejection).

    immunize(threat_name) / forget(threat_name)
        Adjust antibody strength / remove.
    """

    SAFE_FALLBACK_TOKENS = ['__quarantined__']

    def __init__(self, d=400):
        self.d           = d
        self._antibodies = {}      # name -> Antibody
        self._log        = []      # list of (tick, query_tokens, hits)
        self._tick       = 0

    # ── registration ──────────────────────────────────────────────────────

    def register_threat(self, name, tokens):
        """Encode a known jailbreak / injection template as antibody."""
        pos = _encode_semantic(tokens, self.d)
        bag = _encode_bag(tokens, self.d)
        self._antibodies[name] = Antibody(name, pos, bag)
        return self._antibodies[name]

    def forget(self, threat_name):
        """Remove antibody (test/admin only)."""
        return self._antibodies.pop(threat_name, None) is not None

    # ── scanning ──────────────────────────────────────────────────────────

    def scan(self, query_tokens, threshold=0.4):
        """
        Cosine-match query against every antibody (both pos and bag).
        Returns [(name, score), ...] for matches >= threshold (adjusted by strength).
        """
        q_pos = _encode_semantic(query_tokens, self.d)
        q_bag = _encode_bag(query_tokens, self.d)
        hits  = []
        for name, ab in self._antibodies.items():
            s_pos = _cosine(q_pos, ab.pos_hv)
            s_bag = _cosine(q_bag, ab.bag_hv)
            score = max(s_pos, s_bag)
            effective_threshold = threshold / max(1.0, ab.strength)
            if score >= effective_threshold:
                hits.append((name, float(score)))
        hits.sort(key=lambda x: -x[1])
        return hits

    def is_safe(self, query_tokens, threshold=0.4):
        return len(self.scan(query_tokens, threshold)) == 0

    # ── response ──────────────────────────────────────────────────────────

    def quarantine(self, query_tokens, reason='auto'):
        """
        Log threat, immunize matched antibodies, return safe fallback HV.
        """
        hits = self.scan(query_tokens, threshold=0.0)
        self._log.append({
            'tick':     self._tick,
            'query':    list(query_tokens),
            'hits':     hits[:5],
            'reason':   reason,
        })
        # Strengthen matched antibodies (immune memory)
        for name, _ in hits[:3]:
            self.immunize(name)
        self._tick += 1
        return _encode_semantic(self.SAFE_FALLBACK_TOKENS, self.d)

    def immunize(self, threat_name):
        """Bump antibody strength + log exposure."""
        ab = self._antibodies.get(threat_name)
        if ab is None:
            return False
        ab.strength  += 0.5
        ab.exposures += 1
        ab.last_seen  = self._tick
        return True

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def n_threats(self):
        return len(self._antibodies)

    def threat_names(self):
        return list(self._antibodies.keys())

    def antibody(self, name):
        return self._antibodies.get(name)

    def log(self):
        return list(self._log)

    def strongest(self, top_k=5):
        items = sorted(
            self._antibodies.values(),
            key=lambda ab: -ab.strength,
        )
        return [(ab.name, ab.strength, ab.exposures) for ab in items[:top_k]]
