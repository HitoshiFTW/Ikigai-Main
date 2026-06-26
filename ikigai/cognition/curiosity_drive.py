"""
ikigai.cognition.curiosity_drive -- Curiosity Drive (intrinsic motivation).

Day 55 Pack 74 -- Phase B: explore high-PE regions of state space.

Intrinsic reward = prediction error.
    PE(state) = 1 - cos(observed_outcome, predicted_outcome)

High PE -> action was surprising -> visit again to learn (curiosity).
Low PE -> action was predictable -> system can ignore (boredom).

Algorithm:
    record_observation(state, action, observed_next_state)
        Stores PE for this transition.
    novelty(state) = mean PE over recent visits to this state
    curiosity_bonus(action_candidates) -> +reward for high-PE actions
    next_action(state, candidates) = argmax(curiosity_bonus + exploit_value)

Bio analogs:
    - Dopamine bursts on prediction error (RPE)
    - Novelty drives hippocampal replay
    - Berlyne 1960: arousal peaks at intermediate complexity

vs LLM: no intrinsic motivation. Always optimizes log-likelihood.
        Curiosity: actively seeks unexplored / surprising regions.
"""

import math
import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CuriosityDrive:
    """
    Intrinsic motivation via prediction-error tracking.

    record_pe(state_tokens, action_tokens, predicted_outcome, observed_outcome)
        Logs |1 - cos(pred, obs)| at this state-action.

    pe(state_tokens, action_tokens) -> EMA prediction error.
    novelty(state_tokens) -> max-PE across actions at this state.
    boredom(state_tokens) -> 1.0 - mean(PE).
    curiosity_bonus(state, action) = self.beta * pe + self.gamma * novelty.
    visit_count(state) -> n times this state visited.
    decay()                  -> decay all PE values toward zero (boredom over time).

    next_action(state_tokens, action_candidates, exploit_scores)
        Returns argmax(exploit_score + curiosity_bonus).
    """

    def __init__(self, d=400, beta=0.5, gamma=0.3, ema_alpha=0.3, decay_rate=0.01):
        self.d           = d
        self.beta        = float(beta)        # weight on per-(state,action) PE
        self.gamma       = float(gamma)       # weight on state-level novelty
        self.ema_alpha   = float(ema_alpha)   # EMA smoothing for repeated visits
        self.decay_rate  = float(decay_rate)  # global PE decay per tick
        self._pe         = {}    # (state_key, action_key) -> running PE
        self._visits     = {}    # state_key -> count

    # ── observation logging ──────────────────────────────────────────────

    def record_pe(self, state_tokens, action_tokens, predicted_outcome_hv, observed_outcome_hv):
        """Log PE = 1 - cos(predicted, observed). EMA-merge into existing PE."""
        sk = tuple(state_tokens)
        ak = tuple(action_tokens)
        pe_step = 1.0 - _cosine(predicted_outcome_hv, observed_outcome_hv)
        pe_step = max(0.0, min(2.0, pe_step))   # clamp [0, 2]

        if (sk, ak) in self._pe:
            self._pe[(sk, ak)] = (
                self._pe[(sk, ak)] * (1.0 - self.ema_alpha)
                + pe_step * self.ema_alpha
            )
        else:
            self._pe[(sk, ak)] = pe_step

        self._visits[sk] = self._visits.get(sk, 0) + 1
        return pe_step

    # ── PE queries ────────────────────────────────────────────────────────

    def pe(self, state_tokens, action_tokens):
        return float(self._pe.get((tuple(state_tokens), tuple(action_tokens)), 0.0))

    def novelty(self, state_tokens):
        """Max PE across all actions at this state. High = unfamiliar region."""
        sk = tuple(state_tokens)
        pe_vals = [v for (s, _), v in self._pe.items() if s == sk]
        if not pe_vals:
            return 1.0   # unvisited = max novelty
        return float(max(pe_vals))

    def boredom(self, state_tokens):
        """1 - mean PE. High = predictable / overlearned."""
        sk = tuple(state_tokens)
        pe_vals = [v for (s, _), v in self._pe.items() if s == sk]
        if not pe_vals:
            return 0.0
        return float(max(0.0, 1.0 - float(np.mean(pe_vals))))

    def visit_count(self, state_tokens):
        return int(self._visits.get(tuple(state_tokens), 0))

    # ── curiosity-augmented action selection ──────────────────────────────

    def curiosity_bonus(self, state_tokens, action_tokens):
        """beta * PE(s,a) + gamma * novelty(s). Higher = more curious.
        Unseen (state, action) pair: treat per-action PE as max (1.0)
        so untried actions get explored even from familiar states."""
        sk = tuple(state_tokens)
        ak = tuple(action_tokens)
        nv = self.novelty(state_tokens)
        if (sk, ak) not in self._pe:
            # Unseen state-action -> max epistemic PE bonus
            return self.beta * 1.0 + self.gamma * nv
        pe = self._pe[(sk, ak)]
        return self.beta * pe + self.gamma * nv

    def next_action(self, state_tokens, action_candidates, exploit_scores=None):
        """
        Pick argmax over candidates: exploit_score + curiosity_bonus.
        action_candidates: list of action_tokens.
        exploit_scores: dict mapping str(action_tokens) -> exploit value.
        Returns (best_action_tokens, total_score).
        """
        best_action, best_score = None, -float('inf')
        for action_tokens in action_candidates:
            cb = self.curiosity_bonus(state_tokens, action_tokens)
            key = '|'.join(str(t) for t in action_tokens)
            ev = 0.0
            if exploit_scores is not None:
                ev = float(exploit_scores.get(key, 0.0))
            score = ev + cb
            if score > best_score:
                best_score = score
                best_action = action_tokens
        return best_action, float(best_score)

    # ── decay (boredom) ──────────────────────────────────────────────────

    def decay(self):
        """Globally decay all PE values toward 0. Models habituation."""
        for k in list(self._pe.keys()):
            self._pe[k] = max(0.0, self._pe[k] * (1.0 - self.decay_rate))

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def n_logged(self):
        return len(self._pe)

    @property
    def n_visited_states(self):
        return len(self._visits)

    def top_curious(self, top_k=5):
        """Top-k (state, action) pairs by PE (most surprising)."""
        items = sorted(self._pe.items(), key=lambda x: -x[1])
        return [(s, a, v) for (s, a), v in items[:top_k]]

    def all_pe(self):
        return dict(self._pe)
