"""
ikigai.cognition.hebbian_tuner -- Hebbian Online Vocabulary Tuner.

Day 55 Pack 38 -- online coupling of T_couple to user vocabulary.

Problem: PSSTC emit_vec starts random. Decoder cosine(emit, word_hv) is noise.
Fix: after each turn, run hebbian_update(S, G_id, B_U) so T_couple[:, G_id, :]
     learns to map S -> B_U. emit_vec converges to B_U direction over k turns.

Since d_sem == d_emit == d_bspm (all 64 in default), B_U is a valid target
for the emit space. After k steps, cosine(emit(S, G_id), B_U) -> ~1.

Each grammar node (G_id) learns independently:
    T[:, G_id_GREET, :]    tuned on GREET-context turns
    T[:, G_id_QUERY, :]    tuned on QUERY-context turns
    etc.

Invariant: Hebbian update is additive (monotone). No weight decay unless clipping.
           Locality: update to T[:, G_id, :] does not touch other slices.
"""

import numpy as np


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class HebbianVocabTuner:
    """
    Online Hebbian tuner: T_couple[:, G_id, :] += eta * outer(S, B_U).

    After k updates for (S, G_id) pairs, emit(S, G_id) converges toward B_U.
    Grammar locality: each G_id slice trained independently.
    """

    def __init__(self, psstc, eta=0.05):
        """
        psstc: ParallelSemSynCoupling instance (shared with kernel).
        eta:   Hebbian learning rate. Smaller = slower, more stable.
        """
        self.psstc = psstc
        self.eta = eta
        self.update_count = 0
        self.cosine_log = []        # [(cos_before, cos_after)] per update
        self.node_update_counts = {}  # G_id -> n_updates

    def tune(self, S, G_id, target):
        """
        One Hebbian step: T[:, G_id, :] += eta * outer(S, target).

        S:      semantic HV (d_sem,)  -- encoded utterance
        G_id:   grammar node id
        target: target vector (d_emit,) -- use B_U from BSPM

        Returns (emit_after, cos_before, cos_after).
        """
        emit_before = self.psstc.emit(S, G_id)
        cos_before  = _cosine(emit_before, target)

        self.psstc.hebbian_update(S, G_id, target)

        emit_after = self.psstc.emit(S, G_id)
        cos_after  = _cosine(emit_after, target)

        self.cosine_log.append((cos_before, cos_after))
        self.node_update_counts[G_id] = self.node_update_counts.get(G_id, 0) + 1
        self.update_count += 1
        return emit_after, cos_before, cos_after

    def cosine_gain(self):
        """Mean (cos_after - cos_before) across all updates."""
        if not self.cosine_log:
            return 0.0
        return float(np.mean([a - b for b, a in self.cosine_log]))

    def cosine_final(self):
        """Mean cos_after across all updates."""
        if not self.cosine_log:
            return 0.0
        return float(np.mean([a for _, a in self.cosine_log]))

    def node_coverage(self):
        """Number of distinct grammar nodes that have been tuned."""
        return len(self.node_update_counts)

    def reset_stats(self):
        self.update_count = 0
        self.cosine_log.clear()
        self.node_update_counts.clear()
