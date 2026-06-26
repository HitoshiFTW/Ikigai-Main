"""
ikigai.cognition.vsa_attention -- Kill Stack invention #2.

Sub-quadratic attention on the flat substrate. Replaces the transformer
Q.K^T softmax block.

In a transformer with N tokens, each layer computes Q @ K^T -- O(N^2 * d)
work plus O(N * d) memory per head. With H heads and L layers that's
H*L*O(N^2 * d).

In Ikigai, attention is one matmul against a FIXED number of substrate
hard locations:

    activations = query @ Hconj.T          # (1, M) -- M=16384 FIXED
    top-k highest activations select the M_k contributing rows
    output = sum_{i in top-k} C[i]         # weighted by activation

Total work: O(M * d) per head per query, NOT O(N^2). The model never sees
"all token pairs"; it queries the SUBSTRATE which has already absorbed
all relevant context.

Multi-head VSA-Attention: run K parallel queries with different ROLE
bindings (e.g. cooccur, isa, episode), concatenate or sum results.
Equivalent to transformer multi-head but with named semantic roles
instead of learned linear projections.

Public API:
    att = VSAAttention(organism, roles=('cooccur', 'isa', 'property'))
    output = att.attend(query_hv)          # multi-head substrate lookup
"""

import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class VSAAttention:
    """
    Multi-head substrate attention. Each "head" is a different role.
    Query a thought HV through K roles in parallel, sum the recalls.

    Properties:
        - O(M * d) per head, M FIXED, regardless of context length
        - K heads = K substrate lookups, embarrassingly parallel
        - Output is a phasor HV directly usable as next thought / token cleanup target
    """

    def __init__(self, organism, roles=None):
        self.org = organism
        self.mr = organism.unified
        self.d = self.mr.d
        self.roles = list(roles or ('cooccur',))

    # ── one-head ──────────────────────────────────────────────────────────
    def head(self, query, role):
        """
        Single attention head: query the substrate under the named role.
        Returns the renormalised recall HV.
        """
        if role not in self.mr.roles:
            return np.zeros(self.d, dtype=np.complex64)
        role_v = self.mr.roles[role]
        addr = (query * role_v).astype(np.complex64)
        bank = self.mr._bank(role)
        return _renorm(bank.read(addr))

    # ── multi-head ────────────────────────────────────────────────────────
    def attend(self, query, weights=None):
        """
        Multi-head attention: query all configured roles in parallel,
        weighted sum the outputs into a single HV.
        """
        if weights is None:
            weights = [1.0 / len(self.roles)] * len(self.roles)
        if len(weights) != len(self.roles):
            raise ValueError("weights length must equal roles length")
        accum = np.zeros(self.d, dtype=np.complex64)
        for role, w in zip(self.roles, weights):
            accum = accum + (w * self.head(query, role)).astype(np.complex64)
        return _renorm(accum)

    # ── softmax-style cleanup over a candidate vocabulary ────────────────
    def cleanup(self, query, candidates, weights=None, temperature=1.0):
        """
        Multi-head attend + cleanup against candidate words.
        Returns [(word, prob)] sorted by descending probability.
        """
        out = self.attend(query, weights=weights)
        sims = []
        for c in candidates:
            kv = self.mr.ck.key(c)
            s = float(np.real(np.vdot(out, kv))) / self.d
            sims.append((c, s))
        # softmax over similarities
        vals = np.array([s for _, s in sims], dtype=np.float64)
        vals = vals / max(temperature, 1e-6)
        vals = vals - vals.max()
        ev = np.exp(vals)
        probs = ev / max(ev.sum(), 1e-9)
        return sorted([(c, float(p)) for (c, _), p in zip(sims, probs)],
                       key=lambda x: -x[1])
