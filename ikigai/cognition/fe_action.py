"""
ikigai.cognition.fe_action -- Free-Energy Action Selection.

Day 55 Pack 71 -- Phase B: policy = argmin expected_free_energy(action).

Friston Free-Energy Principle:
    Agents act to minimize expected surprise (variational free energy).
    Free energy = KL(belief, prior) + expected -log p(observation | belief)

In VSA terms:
    FE(action, goal) = -cos(predicted_outcome(action), goal_hv) + lambda * surprise(action)
    predicted_outcome = bind(world_model_hv, action_hv) [via CounterfactualField]
    surprise(action) = -cos(action_hv, belief_field) [pragmatic value]

Policy:
    pi*(state, goal) = argmin_a [ FE(a, goal) ]
                     = argmax_a [ cos(predicted_outcome(a), goal) - lambda * surprise(a) ]

Two-term tradeoff:
    - Pragmatic (exploit): match goal directly
    - Epistemic (explore): reduce uncertainty about unfamiliar actions

Bio: dopamine = reward prediction error = -gradFE. Action selection in BG.

vs LLM: action selection = autoregressive next-token. No goal-conditioning.
        FEActionSelector: explicit FE gradient on belief field. Plan-aware.
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
    return np.sign(a * b).astype(np.float32)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FreeEnergyActionSelector:
    """
    Action selection via expected free-energy minimization.

    Inputs:
        - World model: action_tokens -> predicted_outcome_tokens
        - Goal: target outcome HV
        - Belief field: current state HV (from BeliefField)

    register_action(name, action_tokens, outcome_tokens, prior_weight=1.0)
        Add an action to the policy library.

    expected_fe(action_name, goal_tokens, belief_field_hv=None, lam=0.3)
        Returns FE = -cos(predicted, goal) - lam * cos(action, belief_field).
        Lower FE = better action.

    select(goal_tokens, belief_field_hv=None, lam=0.3, top_k=1)
        Returns top-k actions by ascending FE.

    pragmatic_value(action, goal)  -- positive = goal-aligned
    epistemic_value(action, belief) -- positive = explores unfamiliar
    """

    def __init__(self, d=400):
        self.d         = d
        self._actions  = {}     # name -> (action_hv, outcome_hv, prior_weight)

    #  registration

    def register_action(self, name, action_tokens, outcome_tokens, prior_weight=1.0):
        a_hv = _encode(action_tokens,  self.d)
        o_hv = _encode(outcome_tokens, self.d)
        self._actions[name] = (a_hv, o_hv, float(prior_weight))
        return a_hv

    def action_hv(self, name):
        e = self._actions.get(name)
        return e[0] if e else None

    def outcome_hv(self, name):
        e = self._actions.get(name)
        return e[1] if e else None

    #  free-energy components

    def pragmatic_value(self, action_name, goal_tokens):
        """+cos(predicted_outcome, goal). Higher = action achieves goal."""
        e = self._actions.get(action_name)
        if e is None:
            return 0.0
        _, o_hv, _ = e
        return _cosine(o_hv, _encode(goal_tokens, self.d))

    def epistemic_value(self, action_name, belief_field_hv):
        """+ (1 - cos(action, belief_field)). Higher = action explores unknown."""
        e = self._actions.get(action_name)
        if e is None or belief_field_hv is None:
            return 0.0
        a_hv, _, _ = e
        return 1.0 - _cosine(a_hv, belief_field_hv)

    def expected_fe(self, action_name, goal_tokens, belief_field_hv=None, lam=0.3):
        """
        FE = - pragmatic - lam * epistemic
        Lower = better. Minimization target.
        """
        pv = self.pragmatic_value(action_name, goal_tokens)
        ev = self.epistemic_value(action_name, belief_field_hv) if belief_field_hv is not None else 0.0
        e  = self._actions.get(action_name)
        prior = float(e[2]) if e else 1.0
        fe = -pv - lam * ev - 0.1 * np.log(prior + 1e-6)
        return float(fe)

    #  selection

    def select(self, goal_tokens, belief_field_hv=None, lam=0.3, top_k=1):
        """Returns [(action_name, fe), ...] ascending FE (best first)."""
        results = []
        for name in self._actions:
            fe = self.expected_fe(name, goal_tokens, belief_field_hv, lam)
            results.append((name, fe))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def policy(self, goal_tokens, belief_field_hv=None, lam=0.3):
        """Returns single best action name."""
        sel = self.select(goal_tokens, belief_field_hv, lam, top_k=1)
        return sel[0][0] if sel else None

    #  batch evaluation

    def fe_landscape(self, goal_tokens, belief_field_hv=None, lam=0.3):
        """Return dict {action_name: fe} over all actions."""
        return {
            name: self.expected_fe(name, goal_tokens, belief_field_hv, lam)
            for name in self._actions
        }

    #  introspection

    @property
    def n_actions(self):
        return len(self._actions)

    def action_names(self):
        return list(self._actions.keys())
