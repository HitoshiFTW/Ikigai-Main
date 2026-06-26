"""
ikigai.cognition.inverse_gen -- Kill Stack invention #9.

Inverse Generation: substrate is bidirectional. Given an output HV,
find candidate INPUTS that could have produced it.

Transformers are forward-only. Once they generate a token sequence, you
can't ask "what prompt would have produced this?" without trying many
prompts and watching the model's output.

Substrate inverse: given a sentence HV (bundle of token keys),
    - Unbinding by each token's key reveals contributions
    - Querying each role on the bundle reveals "what facts went into this"
    - The substrate's full memory of which addresses are populated lets us
      back-trace which writers contributed

Public API:
    inv = InverseGenerator(organism)
    inputs = inv.candidates(output_hv, top_k=10)
    inputs = inv.from_text("the cat sat", top_k=10)
"""

import re
import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class InverseGenerator:
    """
    Given an output HV, return ranked candidate INPUT tokens that could
    have contributed to it. Operates by cleanup against the vocabulary.
    """

    def __init__(self, organism):
        self.org = organism
        self.mr = organism.unified
        self.d = self.mr.d

    def from_text(self, text, top_k=10, vocab=None):
        """Decompose a sentence into ranked plausible input tokens."""
        toks = re.findall(r"[a-z0-9']+", text.lower())
        if not toks:
            return []
        accum = np.zeros(self.d, dtype=np.complex64)
        for t in toks:
            accum = accum + self.mr.ck.key(t)
        return self.candidates(_renorm(accum), top_k=top_k, vocab=vocab,
                                exclude=set(toks))

    def candidates(self, hv, top_k=10, vocab=None, exclude=None):
        """
        Rank vocabulary tokens by their similarity to the HV. The most
        similar ones are the most plausible inputs.
        """
        if vocab is None:
            vocab = list(self.mr._seen | self.mr._cooccur_seen)
        if not vocab:
            return []
        exclude = exclude or set()
        sims = []
        for w in vocab:
            if w in exclude: continue
            kv = self.mr.ck.key(w)
            s = float(np.real(np.vdot(hv, kv))) / self.d
            sims.append((w, s))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]
