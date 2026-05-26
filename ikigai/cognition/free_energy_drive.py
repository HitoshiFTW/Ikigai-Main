"""
ikigai.cognition.free_energy_drive -- Conversational Variational Free Energy Field (CVFEF).

Day 55 Pack 35 -- conversational substrate primitive #4.
Replaces rule-based response selection with active inference.

Free energy functional:
    F_t = KL_surprise + w_k * contradiction_score + w_g * gap_score

    KL_surprise       = 1 - cosine(B_current, B_prior)         in [0, 1]
    contradiction_score = frac(recent belief deltas < 0)        in [0, 1]
    gap_score         = 1 - mean alignment of recent utterances  in [0, 1]

Action set A (7):
    respond, clarify, challenge, summarize, redirect, terminate, volunteer

Action selection:
    a* = argmin_a E[F | a]   where E[F|a] = coeff[a] * F_t

    Coefficients (lower = action reduces F more):
        terminate: 0.20   clarify: 0.30   summarize: 0.40
        respond:   0.50   volunteer: 0.60  challenge: 0.70   redirect: 0.80

Invariant: F_t >= 0. select_action always returns element of ACTIONS.
"""

import numpy as np

ACTIONS = ['respond', 'clarify', 'challenge', 'summarize', 'redirect', 'terminate', 'volunteer']

_EF_COEFF = {
    'respond':   0.50,
    'clarify':   0.30,
    'challenge': 0.70,
    'summarize': 0.40,
    'redirect':  0.80,
    'terminate': 0.20,
    'volunteer': 0.60,
}


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


class ConversationalVariationalFreeEnergyField:
    """
    F_t = KL_surprise(B_current || B_prior)
        + w_k * contradiction_score(window)
        + w_g * gap_score(window)

    KL_surprise: cosine distance from prior belief.
    contradiction_score: fraction of window turns where belief moved away
                         from established (delta_cosine < 0).
    gap_score: 1 - mean(cosine(utterance_hv, B_prior)) over window,
               normalized to [0,1].

    Action selection: argmin_a _EF_COEFF[a] * F_t.
    """

    def __init__(self, d=64, w_k=0.4, w_g=0.3, window=8, seed=42):
        self.d = d
        self.w_k = w_k
        self.w_g = w_g
        self.window = window

        rng = np.random.RandomState(seed)
        self.B_prior   = _l2_normalize(rng.standard_normal(d).astype(np.float32))
        self.B_current = _l2_normalize(rng.standard_normal(d).astype(np.float32))

        self._belief_deltas    = []   # cosine(B_t, B_{t-1}) per turn
        self._utterance_cosines = []  # cosine(u_t, B_prior) per turn

        self.F_log      = []
        self.action_log = []
        self.turn_count = 0

    # ─── free energy components ────────────────────────────────────

    def kl_surprise(self):
        return float(np.clip(1.0 - _cosine(self.B_current, self.B_prior), 0.0, 1.0))

    def contradiction_score(self):
        if not self._belief_deltas:
            return 0.0
        recent = self._belief_deltas[-self.window:]
        return float(sum(1 for d in recent if d < 0)) / len(recent)

    def gap_score(self):
        if not self._utterance_cosines:
            return 0.5
        recent = self._utterance_cosines[-self.window:]
        mean_cos = float(np.mean(recent))
        return float(np.clip(1.0 - (mean_cos + 1.0) / 2.0, 0.0, 1.0))

    def free_energy(self):
        return float(np.clip(
            self.kl_surprise()
            + self.w_k * self.contradiction_score()
            + self.w_g * self.gap_score(),
            0.0, None
        ))

    # ─── action selection ──────────────────────────────────────────

    def expected_free_energy(self, action, F=None):
        if F is None:
            F = self.free_energy()
        return _EF_COEFF[action] * F

    def select_action(self):
        F = self.free_energy()
        best = min(ACTIONS, key=lambda a: _EF_COEFF[a] * F)
        self.action_log.append(best)
        return best

    def action_distribution(self):
        F = self.free_energy()
        return {a: _EF_COEFF[a] * F for a in ACTIONS}

    # ─── state update ─────────────────────────────────────────────

    def ingest(self, B_new, utterance_hv=None):
        """
        Update with new belief B_new (e.g. from BeliefProjectionManifold.B_U).
        utterance_hv: raw encode_utterance result for gap scoring.
        Returns F_t after update.
        """
        prev = self.B_current.copy()
        self.B_current = _l2_normalize(B_new.copy())

        self._belief_deltas.append(_cosine(self.B_current, prev))
        if utterance_hv is not None:
            self._utterance_cosines.append(_cosine(utterance_hv, self.B_prior))
        else:
            self._utterance_cosines.append(0.0)

        F = self.free_energy()
        self.F_log.append(F)
        self.turn_count += 1
        return F

    def update_prior(self, B_new_prior):
        self.B_prior = _l2_normalize(B_new_prior.copy())

    # ─── stats ────────────────────────────────────────────────────

    def free_energy_stats(self):
        if not self.F_log:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        arr = np.array(self.F_log)
        return {'mean': float(np.mean(arr)), 'min': float(np.min(arr)),
                'max': float(np.max(arr)), 'std': float(np.std(arr))}

    def action_counts(self):
        counts = {a: 0 for a in ACTIONS}
        for a in self.action_log:
            counts[a] += 1
        return counts

    def reset(self):
        self._belief_deltas.clear()
        self._utterance_cosines.clear()
        self.F_log.clear()
        self.action_log.clear()
        self.turn_count = 0
