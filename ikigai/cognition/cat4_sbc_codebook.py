"""
ikigai.cognition.cat4_sbc_codebook -- Pack 285.5.

Pack 285 v0 (Day 77 morning) shipped the SBC encoder primitive
showing 10,000-item discrimination at d=400 (FHRR ceiling ~20) in
O(d) per-block argmax cleanup.  The production action vocab is
~9,620 tokens, sitting right on the FHRR cleanup wall.

This module provides:

  * `SBCActionCodebook(action_vocab, d, B, L)` -- parallel SBC
    encoding of the cat-4 action vocabulary, kept alongside (NOT
    instead of) the existing FHRR `action_codebook()`.

  * `cleanup_token(state_tokens)` -- deterministic SBC lookup using
    the same blake2b indices.  Returns the action token whose
    B-tuple matches the cleanup of the state-tokens HV.  Falls back
    to None when the B-tuple does not match any registered vocab
    token; caller routes the miss to the existing FHRR cleanup or
    Pack 280 vectorized recall.

Production safety gate:
    SBC's structural weakness (additive-noise brittleness, k>1
    superposition failure, Hadamard-binding mismatch) means we
    DO NOT replace the FHRR substrate.  SBC sits as a SECOND
    CHECK after Pack 273 cache lookup and before the FHRR recall
    fallback.  If both cache + SBC miss, the FHRR path runs as
    today.  Any regression on the 25/25 bench disables SBC at
    runtime via the `enabled` flag.

When SBC genuinely fires:
    Pre-grounded queries where state_tokens deterministically
    hash to a known action.  This duplicates Pack 273 cache but
    with constant-time O(d) lookup, useful when the cache lives
    on slow disk (Pack 282 LMDB latency ~50us vs SBC ~5us).
"""
import numpy as np

from ikigai.cognition.sparse_block_codes import (
    SBCEncoder, _stable_block_indices)


class SBCActionCodebook:
    """SBC-encoded view of an FHRR action vocabulary.

    Builds incrementally as vocabulary grows.  Same `enabled`
    runtime kill-switch principle used in Pack 251 OPV.
    """

    def __init__(self, d=400, B=40, L=10, seed=285):
        self.encoder = SBCEncoder(d=d, B=B, L=L, seed=seed)
        self.enabled = True
        self.stats = {
            'registered': 0, 'hits': 0, 'misses': 0,
            'cleanup_calls': 0,
        }

    # ---- vocab registration -----------------------------------------

    def register_vocab(self, tokens):
        """Bulk-register action vocabulary tokens.  Idempotent --
        re-registering an already-known token is a no-op."""
        added = 0
        for t in tokens:
            if t not in self.encoder._token_to_tuple:
                self.encoder.block_tuple(t)
                added += 1
        self.stats['registered'] = len(self.encoder._token_to_tuple)
        return added

    # ---- runtime cleanup --------------------------------------------

    def cleanup_token(self, state_tokens, token_filter=None):
        """Deterministic SBC lookup from a state-token sequence.

        Builds the B-tuple for the state_tokens via the same
        `_stable_block_indices` hash used at vocab encode time.
        If the resulting tuple matches a registered vocab token,
        return it; else None.

        `token_filter` (optional callable) -- restrict valid hits
        to tokens passing the predicate (e.g. valid action class).
        """
        if not self.enabled:
            return None
        self.stats['cleanup_calls'] += 1
        # Use the state-token sequence as a single hashable key.
        # blake2b indices reproduce across processes so the lookup
        # is deterministic regardless of PYTHONHASHSEED.
        key = '|'.join(map(str, state_tokens))
        ix = _stable_block_indices(
            key, self.encoder.B, self.encoder.L, self.encoder.seed)
        tok = self.encoder._tuple_to_token.get(ix)
        if tok is None:
            self.stats['misses'] += 1
            return None
        if token_filter and not token_filter(tok):
            self.stats['misses'] += 1
            return None
        self.stats['hits'] += 1
        return tok

    # ---- vocab persistence ------------------------------------------

    def vocab_size(self):
        return len(self.encoder._token_to_tuple)

    def disable(self, reason=None):
        """Kill switch -- subsequent cleanup_token calls return None.
        Caller logs `reason` upstream."""
        self.enabled = False
        self.stats['disabled_reason'] = str(reason or '')
