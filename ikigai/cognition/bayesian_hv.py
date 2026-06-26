"""
ikigai.cognition.bayesian_hv -- Kill Stack invention #10.

Bayesian HV Magnitudes: confidence falls out of substrate math, not a
learned softmax head. Transformers fake confidence via the model's own
softmax (which is famously miscalibrated). Substrate gives it natively:
the magnitude of a recall HV directly encodes how strongly the answer
is "anchored" in the substrate.

Operations:
    recall_magnitude(word, role) -> float  -- avg |C[i]| at activated locations
    confidence(word, role, target, candidates=None) -> float in [0, 1]
    is_known(word, role) -> bool  -- True if substrate has strong response

The confidence metric:
    cleanup against candidates -> top-1 vs top-2 margin (softmax-margin)
    PLUS the absolute magnitude of the recall HV (anchored vs noisy)
"""

import numpy as np


class BayesianHV:
    """
    Calibrated confidence from substrate magnitudes.
    """

    def __init__(self, organism):
        self.org = organism
        self.mr = organism.unified
        self.d = self.mr.d

    def recall_magnitude(self, word, role):
        """
        Pre-renorm magnitude of the recall HV. Reads the counter bank
        directly, summing activated rows BEFORE the standard renorm in
        VSASDM.read. High value = substrate is strongly anchored on this
        (word, role).
        """
        if role not in self.mr.roles:
            return 0.0
        bank = self.mr._bank(role)
        addr = self.mr._addr(word, role)
        slot = self.mr._slot(word, role)
        locs = bank.locs(addr, slot)
        raw = bank.C[locs].sum(axis=0)
        return float(np.mean(np.abs(raw)))

    def confidence(self, word, role, candidates=None):
        """
        Return (best_target, confidence in [0,1]). Confidence = raw top-1
        cleanup score against candidates, clamped to [0,1]. For known
        facts this is close to 1.0; for unknown words it tends toward 0.
        """
        if role not in self.mr.roles:
            return None, 0.0
        if candidates is None:
            candidates = list(self.mr._role_targets.get(role, set()))
        if not candidates:
            return None, 0.0
        hv = self.mr.recall(word, role)
        sims = []
        for c in candidates:
            kv = self.mr.ck.key(c)
            s = float(np.real(np.vdot(hv, kv))) / self.d
            sims.append((c, s))
        sims.sort(key=lambda x: -x[1])
        best, best_s = sims[0]
        # raw cleanup similarity is already in [-1, 1]; clamp to [0, 1]
        confidence = max(0.0, min(1.0, best_s))
        return best, float(confidence)

    def is_known(self, word, role, threshold=0.05):
        """True if substrate has any non-trivial recall mass for this address."""
        return self.recall_magnitude(word, role) > threshold
