"""
ikigai.cognition.counterfactual_sim -- Counterfactual Simulator.

Day 55 Pack 64 -- invention #8: N parallel futures in single HV superposition.

Architecture:
    Each scenario: (action_hv, outcome_hv) bound -> action ⊕ outcome
    Superposition: bundle(all scenario binds) -> single counterfactual_hv
    Query "what if action A?" -> bind(counterfactual_hv, A) -> recover outcome A

Key property:
    Querying with action A recovers its outcome with cos ~ 1/N >> noise floor.
    N parallel futures live in O(d) space, not O(N*d).
    Zero rollout cost: all futures evaluated in one bind.

Planning use:
    1. Generate candidate actions
    2. Bind each with predicted outcome (via world model)
    3. Bundle into counterfactual field
    4. Query with goal_hv -> action that maximally produces goal

vs LLM: chain-of-thought rollouts each cost N tokens -> O(N) inference cost.
        Counterfactual sim: all N branches in one HV, O(1) query.
        Free-energy gradient over branches in O(d) work.
"""

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


def _bind(a, b):
    """Bipolar bind = elementwise multiply (self-inverse)."""
    return np.sign(a * b).astype(np.float32)


def _bsign(x):
    s = np.sign(x).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CounterfactualField:
    """
    Superposition of N parallel (action, outcome) bindings.

    add_scenario(action_tokens, outcome_tokens, weight=1)
        Append (action ⊕ outcome) to field.

    query_outcome(action_tokens, candidates_tokens) -> [(name, sim), ...]
        Predict outcome of given action.
        Returns ranked candidate outcomes by recovery cosine.

    query_action(outcome_tokens, candidates_tokens) -> [(name, sim), ...]
        Reverse: which action produces this outcome?

    free_energy(goal_tokens) -> [(scenario_idx, fe), ...]
        Rank scenarios by -cos(predicted_outcome, goal). Lower FE = better aligned.

    best_action(goal_tokens, action_candidates) -> (best_name, score)
        Argmax over candidate actions for maximizing alignment with goal.
    """

    def __init__(self, d=400):
        self.d        = d
        self._field   = np.zeros(d, dtype=np.float32)
        self._scenarios = []   # list of (name, action_hv, outcome_hv, weight)
        self._raw_accum = np.zeros(d, dtype=np.int32)   # un-normalized for ranking

    # ── construction ──────────────────────────────────────────────────────

    def add_scenario(self, name, action_tokens, outcome_tokens, weight=1.0):
        """Add (action ⊕ outcome) binding to field."""
        a_hv = _encode(action_tokens,  self.d)
        o_hv = _encode(outcome_tokens, self.d)
        binding = _bind(a_hv, o_hv)
        self._raw_accum += (weight * binding).astype(np.int32)
        self._field      = _bsign(self._raw_accum.astype(np.float32))
        self._scenarios.append((name, a_hv, o_hv, float(weight)))
        return binding

    def n_scenarios(self):
        return len(self._scenarios)

    def field_hv(self):
        return self._field.copy()

    # ── queries ───────────────────────────────────────────────────────────

    def query_outcome(self, action_tokens, candidate_outcome_tokens):
        """
        Given action, rank candidate outcomes by recovery cosine.
        Recovery: field ⊕ action_hv ~= outcome (for matching scenario).
        """
        a_hv      = _encode(action_tokens, self.d)
        # Use raw_accum for higher-fidelity recovery (preserves signal magnitude)
        recovered = self._raw_accum.astype(np.float32) * a_hv
        results = []
        for (name, cand_tokens) in candidate_outcome_tokens:
            cand_hv = _encode(cand_tokens, self.d)
            results.append((name, float(np.dot(recovered, cand_hv)) / self.d))
        results.sort(key=lambda x: -x[1])
        return results

    def query_action(self, outcome_tokens, candidate_action_tokens):
        """Reverse: given outcome, rank candidate actions."""
        o_hv      = _encode(outcome_tokens, self.d)
        recovered = self._raw_accum.astype(np.float32) * o_hv
        results = []
        for (name, cand_tokens) in candidate_action_tokens:
            cand_hv = _encode(cand_tokens, self.d)
            results.append((name, float(np.dot(recovered, cand_hv)) / self.d))
        results.sort(key=lambda x: -x[1])
        return results

    # ── planning ──────────────────────────────────────────────────────────

    def free_energy(self, goal_tokens):
        """
        For each scenario, FE = -cos(outcome, goal). Lower FE = better aligned.
        Returns list of (scenario_name, fe).
        """
        goal_hv = _encode(goal_tokens, self.d)
        result = []
        for (name, _, o_hv, _) in self._scenarios:
            fe = -_cosine(o_hv, goal_hv)
            result.append((name, float(fe)))
        result.sort(key=lambda x: x[1])  # ascending FE = best first
        return result

    def best_action(self, goal_tokens, action_candidates):
        """
        Find action whose predicted outcome best matches goal.
        action_candidates: list of (name, tokens).
        Returns (best_name, score) or (None, 0.0).
        """
        goal_hv = _encode(goal_tokens, self.d)
        best_name, best_score = None, -2.0
        for (name, action_tokens) in action_candidates:
            # Predicted outcome HV from binding: field ⊕ action
            a_hv = _encode(action_tokens, self.d)
            predicted = self._raw_accum.astype(np.float32) * a_hv
            # Normalize predicted and compute cos with goal
            norm = float(np.linalg.norm(predicted))
            if norm == 0.0:
                score = 0.0
            else:
                score = float(np.dot(predicted, goal_hv) / (norm * np.linalg.norm(goal_hv)))
            if score > best_score:
                best_score = score
                best_name  = name
        return best_name, float(best_score)

    # ── perturbation / counterfactual edit ────────────────────────────────

    def perturb(self, scenario_name, new_outcome_tokens):
        """
        Counterfactual edit: replace scenario's outcome.
        Useful for "what if X had a different outcome?" reasoning.
        """
        for i, (name, a_hv, o_hv, w) in enumerate(self._scenarios):
            if name != scenario_name:
                continue
            # Remove old binding
            old_binding = _bind(a_hv, o_hv)
            self._raw_accum -= (w * old_binding).astype(np.int32)
            # Add new binding
            new_o_hv = _encode(new_outcome_tokens, self.d)
            new_binding = _bind(a_hv, new_o_hv)
            self._raw_accum += (w * new_binding).astype(np.int32)
            self._field = _bsign(self._raw_accum.astype(np.float32))
            self._scenarios[i] = (name, a_hv, new_o_hv, w)
            return True
        return False

    def scenario_names(self):
        return [s[0] for s in self._scenarios]
