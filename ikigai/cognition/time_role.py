"""
ikigai.cognition.time_role -- Kill Stack invention #8.

Time-As-A-Role: every fact bound with a timestamp role. Native temporal
indexing on the substrate.  Query "what was X like at time T?" via role
unbinding.  Transformers have zero native temporal structure -- they
have to put a date string in the prompt and hope the model uses it.

Mechanism:
    For each bucket B (e.g. year), pre-generate a phasor ROLE_B from a
    seeded deterministic generator. To write a TIMED fact (x, role, y, t):
        addr = key(x) * ROLE_role * ROLE_t   (triple bind, t = time bucket)
        value = key(y)
        substrate.sdm_rel.write(addr, value)

    To query "what was (x, role) at time t?":
        addr = key(x) * ROLE_role * ROLE_t
        slot = substrate.sdm_rel.read(addr)
        nearest_neighbour(slot, candidates)

VSA is associative so the order of binds doesn't matter; the time role
is just another phasor factor in the address. Substrate FIXED.
"""

import time
import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class TimeRole:
    """
    Temporal indexing as a substrate role. Time buckets are deterministic
    phasor HVs generated from a seeded RNG -- bucket "2026" always maps
    to the same HV. Buckets compose multiplicatively into the address.
    """

    def __init__(self, mr, seed=14802):
        self.mr = mr
        self.d = mr.d
        self.seed = int(seed)
        self._bucket_cache = {}

    def bucket_hv(self, bucket):
        """Return the phasor HV for a time bucket (cached deterministic)."""
        if bucket not in self._bucket_cache:
            rng = np.random.default_rng(hash((self.seed, str(bucket)))
                                         & 0xFFFFFFFF)
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            self._bucket_cache[bucket] = np.exp(1j * ph).astype(np.complex64)
        return self._bucket_cache[bucket]

    # ── write a timed fact ────────────────────────────────────────────────
    def assert_at(self, word, role, target_word, bucket, n=20):
        """
        Write (word, role) -> target_word, BOUND to time bucket.
        n = reinforcement count (same convention as relate).
        """
        if role not in self.mr.roles:
            return 0
        role_v = self.mr.roles[role]
        time_v = self.bucket_hv(bucket)
        addr = (self.mr.ck.key(word) * role_v * time_v).astype(np.complex64)
        target = self.mr.ck.key(target_word)
        bank = self.mr._bank(role)
        slot = f'{word}\x00{role}\x00{bucket}'
        for _ in range(int(n)):
            bank.write(addr, target, word=slot)
        self.mr._role_targets.setdefault(role, set()).add(word)
        # also track which words are time-indexed
        self.mr._role_targets.setdefault(f'{role}@time', set()).add(word)
        return n

    # ── query a timed fact ────────────────────────────────────────────────
    def query_at(self, word, role, bucket, candidates=None):
        """
        Recall (word, role) at the given time bucket. Returns (best, score).
        """
        if role not in self.mr.roles:
            return None, 0.0
        role_v = self.mr.roles[role]
        time_v = self.bucket_hv(bucket)
        addr = (self.mr.ck.key(word) * role_v * time_v).astype(np.complex64)
        bank = self.mr._bank(role)
        slot = f'{word}\x00{role}\x00{bucket}'
        recalled = bank.read(addr, word=slot)
        if candidates is None:
            candidates = list(self.mr._role_targets.get(role, set()))
        if not candidates:
            return None, 0.0
        best = None; best_s = -9.0
        for c in candidates:
            kv = self.mr.ck.key(c)
            s = float(np.real(np.vdot(recalled, kv))) / self.d
            if s > best_s:
                best_s = s; best = c
        return best, best_s

    # ── compare two time buckets ──────────────────────────────────────────
    def diff(self, word, role, bucket_a, bucket_b, candidates=None):
        """
        Return (target_a, target_b) -- what `word`'s `role` resolved to at
        each bucket. Useful for "what changed between 2024 and 2026?".
        """
        a, _ = self.query_at(word, role, bucket_a, candidates=candidates)
        b, _ = self.query_at(word, role, bucket_b, candidates=candidates)
        return a, b
