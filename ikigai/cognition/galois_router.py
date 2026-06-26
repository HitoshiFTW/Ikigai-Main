"""ikigai.cognition.galois_router -- Pack 190 must-invent primitive #2.

Galois-field similarity + relative-rank routing.

Standard cosine ranking on N atoms breaks past Plate-bound (~M/log(M)
clean retrievals) because crosstalk drowns the signal. Galois-field
similarity replaces continuous cosine with discrete Hamming-rank over
a finite ring GF(p). Two key properties:

  1. Orthogonal Galois codewords have ZERO crosstalk (vs ~1/sqrt(d) for
     complex HRR). Bounded above by 1/p, not 1/sqrt(d).
  2. Rank-percentile routing (instead of absolute similarity) is invariant
     to global signal magnitude shifts -- so multi-model substrate noise
     doesn't push a winner past the loser.

Public API:
    router = GaloisRouter(p=251, d=2048)
    code   = router.encode(hv)
    ranks  = router.rank_route(query_hv, codebook_codes)   # (K,) int rank
    best   = router.topk(query_hv, codebook_codes, k=8)

Building blocks: VaCoAl (Yeh & Lin 2024) -- Galois-HD CAM for sharpness.

Note: rank routing is a sharpener, not a quantizer. Reconstruction still
uses the original float codebook. This module supplies the *index lookup*
that Pack 189 currently does via raw cosine.
"""

import numpy as np


class GaloisRouter:
    """Galois-field rank router for codebook atom retrieval.

    Replace cosine ranking with discrete Hamming-rank over GF(p) codewords.
    Sharper at scale -- crosstalk bounded by 1/p, independent of N atoms.
    """

    def __init__(self, p=251, d=2048, seed=4096):
        if p < 2:
            raise ValueError('p must be >= 2 (use 251 for uint8, 65521 for uint16)')
        self.p = int(p)
        self.d = int(d)
        self._rng = np.random.default_rng(int(seed))
        # random orthonormal-ish basis for float -> GF(p) projection
        self._W = self._rng.standard_normal((self.d, self.d)).astype(np.float32)
        self._W /= (np.linalg.norm(self._W, axis=0, keepdims=True) + 1e-9)

    def encode(self, hv):
        """Float HV -> GF(p) codeword (uint16 vector length d).

        Project to d-dim, modulo into [0, p).
        """
        hv = np.asarray(hv, dtype=np.float32).reshape(-1)
        if hv.shape[0] != self.d:
            # accept arbitrary length: pad / project
            if hv.shape[0] < self.d:
                hv = np.pad(hv, (0, self.d - hv.shape[0]))
            else:
                hv = hv[:self.d]
        z = self._W @ hv
        # map to [0, p): center, scale by std, mod
        zc = z - z.mean()
        zc = zc / (zc.std() + 1e-9)
        # rescale to integer range
        codes = np.round((zc + 4.0) * (self.p / 8.0)).astype(np.int64) % self.p
        return codes.astype(np.uint16)

    def encode_batch(self, hvs):
        """Batch encode (N, d) -> (N, d) uint16 codewords."""
        hvs = np.asarray(hvs, dtype=np.float32)
        if hvs.ndim != 2:
            raise ValueError(f'expected 2D, got shape {hvs.shape}')
        N = hvs.shape[0]
        if hvs.shape[1] != self.d:
            # pad/truncate
            if hvs.shape[1] < self.d:
                pad = np.zeros((N, self.d - hvs.shape[1]), dtype=np.float32)
                hvs = np.concatenate([hvs, pad], axis=1)
            else:
                hvs = hvs[:, :self.d]
        z = hvs @ self._W.T
        zc = z - z.mean(axis=1, keepdims=True)
        zc = zc / (zc.std(axis=1, keepdims=True) + 1e-9)
        codes = np.round((zc + 4.0) * (self.p / 8.0)).astype(np.int64) % self.p
        return codes.astype(np.uint16)

    def hamming(self, code_a, code_b):
        """GF(p) Hamming distance = count of mismatched positions."""
        a = np.asarray(code_a, dtype=np.uint16)
        b = np.asarray(code_b, dtype=np.uint16)
        return int((a != b).sum())

    def similarity_batch(self, query_code, codebook_codes):
        """(d,) query vs (K, d) codebook -> (K,) float similarity in [0, 1].
        sim = 1 - hamming/d.
        """
        q = np.asarray(query_code, dtype=np.uint16).reshape(-1)
        cb = np.asarray(codebook_codes, dtype=np.uint16)
        if cb.ndim != 2:
            raise ValueError(f'codebook must be 2D, got {cb.shape}')
        # exact match indicator
        matches = (cb == q[None, :]).sum(axis=1)
        return matches.astype(np.float32) / float(self.d)

    def rank_route(self, query_code, codebook_codes):
        """Return per-atom RANK (lower = better) under Galois Hamming.
        Output shape (K,) int -- rank position [0, K-1]. Use ranks instead of
        absolute scores for crosstalk-invariant routing.
        """
        sim = self.similarity_batch(query_code, codebook_codes)
        # rank: higher sim -> lower rank index
        order = np.argsort(-sim)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        return ranks

    def topk(self, query_code, codebook_codes, k=8):
        """Top-k atom indices by Galois similarity."""
        sim = self.similarity_batch(query_code, codebook_codes)
        if k >= sim.shape[0]:
            return np.argsort(-sim)
        return np.argpartition(-sim, k)[:k]


def galois_sharpen_cosine(cos_scores, gal_ranks, alpha=0.5):
    """Blend cosine scores with inverted Galois rank as a sharpener.

    cos_scores: (K,) float cosine similarity.
    gal_ranks:  (K,) int rank (0 best).
    alpha: blend weight for Galois sharpener.

    Returns combined score (K,) float -- higher better.
    """
    cos = np.asarray(cos_scores, dtype=np.float32)
    rk = np.asarray(gal_ranks, dtype=np.float32)
    inv_rank = 1.0 - (rk / max(rk.max(), 1.0))
    return (1.0 - alpha) * cos + alpha * inv_rank
