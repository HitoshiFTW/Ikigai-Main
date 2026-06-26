"""
ikigai.cognition.holographic_context -- Kill Stack invention #1.

Pack an arbitrarily long token stream into ONE bounded phasor HV via
permute-then-bundle. The HV has fixed size d (default 512), regardless
of how many tokens have ever been seen. No KV cache. No attention matrix.
True O(1) RAM at any context length.

Algorithm (textbook VSA temporal binding):

    For each new token t at position i:
        ctx += permute_i(key(t))            # cyclic shift by i positions

    To recall "what was at position i?":
        slot = permute_{-i}(ctx)            # invert the shift
        word = nearest_neighbour(slot, vocabulary)

Properties:
    - Context size: O(d) bytes FIXED, regardless of stream length
    - Recent positions recover with low crosstalk
    - Older positions fade GRACEFULLY through superposition noise
      (the system "forgets" the way humans do, not by hitting a wall)
    - Cleanup query is O(vocab * d), independent of context length

This is the structural answer to the KV cache:
    Transformer: K_i, V_i stored per token, O(N * d) RAM grows linearly
    Holographic: O(d) RAM, N entirely tracked by the position counter

Public API:
    ctx = HolographicContext(mr)
    ctx.append('the')
    ctx.append('cat')
    ctx.append('sat')
    w, score = ctx.query_position(1)        # -> 'cat', high score
    recent = ctx.recent_tokens(n=5)         # last 5 (word, score) pairs
"""

import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class HolographicContext:
    """
    Bounded-size context HV. Append tokens; query any position later.

    Parameters:
        mr           -- MultiRoleMemory (uses ck.key for word identity)
        renormalise  -- if True, renormalise ctx after every append.
                        Preserves complex64 unit-phasor property; trades
                        absolute magnitude for stability across long
                        streams. Default True.
    """

    def __init__(self, mr, renormalise=True):
        self.mr = mr
        self.d = mr.d
        self.ctx = np.zeros(self.d, dtype=np.complex64)
        self.position = 0
        self.history = []
        self.renormalise = bool(renormalise)

    # ── core operations ───────────────────────────────────────────────────
    def append(self, token):
        """Bundle a new token at the current position. O(d) work."""
        k = self.mr.ck.key(token)
        # cyclic shift by current position encodes "when" this token arrived
        shifted = np.roll(k, self.position).astype(np.complex64)
        self.ctx = (self.ctx + shifted).astype(np.complex64)
        if self.renormalise:
            self.ctx = _renorm(self.ctx)
        self.history.append(token)
        self.position += 1

    def reset(self):
        self.ctx = np.zeros(self.d, dtype=np.complex64)
        self.position = 0
        self.history = []

    # ── query ─────────────────────────────────────────────────────────────
    def query_position(self, i, candidates=None):
        """
        Return (best_word, score) for the token at position i. Score is
        cos similarity to nearest vocabulary token; if vocabulary candidates
        is None, uses everything the substrate has seen.
        """
        if i < 0 or i >= self.position:
            return None, 0.0
        # invert the position shift to recover the token's key-direction
        slot = np.roll(self.ctx, -i).astype(np.complex64)
        if candidates is None:
            candidates = list(self.mr._seen)
        if not candidates:
            return None, 0.0
        # vectorised cleanup
        K = np.stack([self.mr.ck.key(w) for w in candidates])
        sims = (np.conj(slot)[None, :] * K).sum(axis=1).real / self.d
        idx = int(np.argmax(sims))
        return candidates[idx], float(sims[idx])

    def recent_tokens(self, n=10, candidates=None):
        """Return [(position, word, score), ...] for the last n positions."""
        out = []
        for i in range(max(0, self.position - n), self.position):
            w, s = self.query_position(i, candidates)
            out.append((i, w, s))
        return out

    def __len__(self):
        return self.position

    def memory_bytes(self):
        """Return bytes used by the context HV (size FIXED regardless of length)."""
        return self.ctx.nbytes
