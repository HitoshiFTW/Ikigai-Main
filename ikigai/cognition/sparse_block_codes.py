"""
ikigai.cognition.sparse_block_codes -- Pack 285 v0.

Hersche, Vasilache & Sebastian (NAI-240713 2025) "Sparse Block Codes for
Efficient In-Context Learning"; building on Frady, Kleyko & Sommer 2021
"Variable Binding for Sparse Distributed Representations".

The Frady-Sommer cleanup bound at fixed d=400 caps the discrimination
ceiling at K_max ~= sqrt(d / n_factors) ~= 20 distinguishable items per
factor.  Sparse Block Codes break this by reshaping the d-dim phasor
substrate into B blocks of L phasors each (B * L = d).  Encoding a
token assigns it a B-tuple of within-block slot indices (s_1, ..., s_B)
where each s_k in [0, L).  The encoded HV has nonzero phasor at slot
s_k within block k for every k, zero elsewhere.

Capacity:
    Distinct tokens encodable    : L^B
    For d=400, B=40, L=10        : 10^40 (vastly more than needed)
    For d=400, B=20, L=20        : 20^20 = ~10^26

Cleanup:
    Per block k, argmax over L slots gives s_k -- O(B * L) total
    instead of O(N) cosine cleanups against the full codebook.
    For d=400, B=40, L=10: cleanup cost = 400 ops (constant in vocab N).

Binding:
    Hadamard (component-wise multiply) still composes B-tuples by
    per-block addition of slot indices mod L.  Cleanup recovers the
    composed tuple block-independently.

This module ships v0 = the encoder + cleanup demo.  Production wiring
into cat4_absorb (substituting focus_hv) is Pack 285.5 follow-up
gated on a 25/25 bench gate showing no regression vs the current
trailing-weighted focus_hv.
"""
import hashlib
import numpy as np


def _stable_block_indices(token, B, L, seed):
    """Deterministic B-tuple of within-block slot indices in [0, L).

    Uses blake2b(token + block_index) -> int -> mod L per block.
    Reproducible across processes (PYTHONHASHSEED-independent), so a
    sidecar codebook can be regenerated on reload without persisting
    the index map alongside organism.ikg.
    """
    indices = []
    for k in range(B):
        h = hashlib.blake2b(
            f'{seed}|{token}|{k}'.encode('utf-8'),
            digest_size=4).digest()
        indices.append(int.from_bytes(h, 'big') % L)
    return tuple(indices)


class SBCEncoder:
    """Sparse Block Code encoder for d-dim phasor substrate.

    d == B * L.  Each block holds L phasor slots; encoding a token
    activates exactly one slot per block.  Active slot phasor is a
    random unit-phasor drawn from a per-block fixed RNG so binding
    via Hadamard product composes meaningfully.
    """

    def __init__(self, d=400, B=40, L=10, seed=285):
        if int(B) * int(L) != int(d):
            raise ValueError(
                f'd ({d}) must equal B*L ({B}*{L}={B*L})')
        self.d = int(d)
        self.B = int(B)
        self.L = int(L)
        self.seed = int(seed)
        # Per-block fixed phasor codebook: shape (B, L) complex64.
        # Slot l within block k is exp(i * theta_kl) with theta_kl
        # drawn uniform on [-pi, pi] from a deterministic RNG.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-np.pi, np.pi, size=(self.B, self.L))
        self._slot_phasors = np.exp(1j * theta).astype(np.complex64)
        # tuple(s_1..s_B) -> token index for inverse cleanup
        self._tuple_to_token = {}
        # token -> tuple cache
        self._token_to_tuple = {}

    # ---- encoding ---------------------------------------------------

    def block_tuple(self, token):
        """Return the deterministic B-tuple for `token` and cache it."""
        t = str(token)
        cached = self._token_to_tuple.get(t)
        if cached is not None:
            return cached
        ix = _stable_block_indices(t, self.B, self.L, self.seed)
        self._token_to_tuple[t] = ix
        self._tuple_to_token[ix] = t
        return ix

    def encode(self, token):
        """Build the d-dim SBC HV for `token`.  Active slot per block;
        zeros elsewhere."""
        ix = self.block_tuple(token)
        hv = np.zeros(self.d, dtype=np.complex64)
        for k, s in enumerate(ix):
            hv[k * self.L + s] = self._slot_phasors[k, s]
        return hv

    def encode_dense(self, token):
        """Variant that fills the inactive slots with low-magnitude
        noise instead of strict zeros.  Useful for sanity-checking
        cleanup against a baseline that allows magnitude leakage."""
        hv = self.encode(token)
        rng = np.random.default_rng(
            int(hashlib.blake2b(
                f'noise|{token}'.encode('utf-8'),
                digest_size=4).digest().hex(), 16))
        noise = (0.01 *
                  (rng.standard_normal(self.d).astype(np.float32)
                   + 1j * rng.standard_normal(self.d).astype(np.float32))
                  ).astype(np.complex64)
        return hv + noise

    def register_tokens(self, tokens):
        """Pre-populate the tuple->token map for a vocabulary.
        Required before cleanup() can recover tokens from bundled HVs."""
        for t in tokens:
            self.block_tuple(t)
        return len(self._tuple_to_token)

    # ---- cleanup ----------------------------------------------------

    def cleanup_tuple(self, hv):
        """Decode the B-tuple of slot indices from an HV.

        For each block k, take argmax(|hv[k*L : k*L + L]|).  O(d) total.
        """
        result = []
        for k in range(self.B):
            block = hv[k * self.L:(k + 1) * self.L]
            result.append(int(np.argmax(np.abs(block))))
        return tuple(result)

    def cleanup(self, hv):
        """Return the token whose tuple matches the cleanup of `hv`.
        None if no registered token has that exact tuple."""
        ix = self.cleanup_tuple(hv)
        return self._tuple_to_token.get(ix)

    # ---- binding ----------------------------------------------------
    # Pack 285.6: SBC binding is per-block slot-index ADDITION mod L (the
    # group operation on Z_L^B), NOT Hadamard.  Two SBC HVs are sparse
    # (one active slot per block); their Hadamard product shares a slot
    # only by chance and zeros every mismatched block -- which is why the
    # Pack 285 v0 Hadamard bind was broken (kept as bind_hadamard below
    # for the regression that exposed it).  Slot-add binding is exact and
    # invertible: bind(a, b) then unbind(., a) recovers b.

    def encode_tuple(self, ix):
        """Build the d-dim SBC HV from an explicit B-tuple of slot
        indices (each reduced mod L).  Generalizes encode() to composite
        (bound) tuples that need not match any registered token."""
        hv = np.zeros(self.d, dtype=np.complex64)
        for k, s in enumerate(ix):
            s = int(s) % self.L
            hv[k * self.L + s] = self._slot_phasors[k, s]
        return hv

    def bind_tuple(self, ta, tb):
        """SBC binding on block-tuples: per-block (s_a + s_b) mod L."""
        L = self.L
        return tuple((int(a) + int(b)) % L for a, b in zip(ta, tb))

    def unbind_tuple(self, tbound, ta):
        """Inverse of bind_tuple: per-block (s_bound - s_a) mod L."""
        L = self.L
        return tuple((int(c) - int(a)) % L for c, a in zip(tbound, ta))

    def bind(self, a, b):
        """Bind two SBC HVs via slot-index addition mod L, re-encoded to
        a valid one-slot-per-block HV.  Exact and invertible (see
        unbind)."""
        return self.encode_tuple(
            self.bind_tuple(self.cleanup_tuple(a), self.cleanup_tuple(b)))

    def unbind(self, bound, a):
        """Recover b's HV from bind(a, b) and a."""
        return self.encode_tuple(
            self.unbind_tuple(self.cleanup_tuple(bound), self.cleanup_tuple(a)))

    @staticmethod
    def bind_hadamard(a, b):
        """DEPRECATED (Pack 285 v0).  Hadamard product -- WRONG for SBC:
        sparse one-slot-per-block HVs coincide on a slot only by chance,
        so the product zeros most blocks.  Use bind()/bind_tuple()."""
        return (a * b).astype(np.complex64)

    @staticmethod
    def bundle(hvs):
        """Sum of SBC HVs = superposition.  Cleanup recovers top match
        per block (loses individual items past ~L superposed)."""
        if not hvs:
            return None
        out = np.zeros_like(hvs[0])
        for h in hvs:
            out = out + h
        return out.astype(np.complex64)
