"""
ikigai.cognition.causal_world_model -- Temporal Causal World Model.

Day 55 Pack 72 -- Phase B: causal graph in HV. State + action -> next state.

Architecture:
    Transitions: (state, action) -> next_state stored as HV binding.
        edge_hv = bind(state_hv, action_hv)
        transition: store edge_hv -> next_state_hv in HolographicMemory
    Predict: predict(state, action) -> next_state via bind-and-recall
    Rollout: simulate N steps via repeated predict
    Causal chain: (S0, A0) -> S1 -> (S1, A1) -> S2 -> ...

Query:
    explain(s_end, s_start, depth) -> shortest action sequence linking them.
    Symbolic causal inference via VSA superposition + cosine ranking.

vs LLM: world model = unstated, statistical. Counterfactuals = sample many tokens.
        Causal model: explicit HV graph. Edit any node, propagate effects.
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


class CausalWorldModel:
    """
    Transition graph in HV space. (state, action) -> next_state.

    add_transition(state_tokens, action_tokens, next_state_tokens)
        Register one edge. Multiple edges accumulate as superposition.

    predict(state_tokens, action_tokens, candidate_states) -> [(state_name, sim), ...]
        Given state + action, predict next state from candidates.

    rollout(state_tokens, action_seq, candidate_states) -> [trajectory]
        Multi-step prediction.

    explain(target_state_tokens, current_state_tokens, candidate_actions, depth=3)
        Search for action sequence linking current -> target.

    add_state / add_action -- registers vocabulary HVs.
    """

    def __init__(self, d=400, perm_shift=1):
        self.d                  = d
        # Superposed transition memory: sum of bind(bind(state, action), permute(next_state))
        self._trans_accum       = np.zeros(d, dtype=np.int32)
        # Vocabulary
        self._states            = {}     # name -> hv
        self._actions           = {}     # name -> hv
        # Cyclic permutation shift breaks bind-symmetry between source/target positions.
        # Without permute: predict(s_k, next) leaks to s_{k-1} in chains
        # because bind(s_{k-1}, next, s_k) * bind(s_k, next) = s_{k-1} (self-inverse).
        # With permute: bind(s_{k-1}, next, perm(s_k)) * bind(s_k, next)
        # = s_{k-1} * perm(s_k) * s_k (NOT a clean recovery of s_{k-1}).
        self._perm_shift        = int(perm_shift)
        # Transition log for explain/inspect
        self._edges             = []     # list of (state, action, next_state)

    def _perm(self, hv):
        return np.roll(hv, self._perm_shift)

    #  vocabulary

    def add_state(self, name, tokens=None):
        if tokens is None:
            tokens = [name]
        hv = _encode(tokens, self.d)
        self._states[name] = hv
        return hv

    def add_action(self, name, tokens=None):
        if tokens is None:
            tokens = [name]
        hv = _encode(tokens, self.d)
        self._actions[name] = hv
        return hv

    def state_hv(self, name):
        return self._states.get(name)

    def action_hv(self, name):
        return self._actions.get(name)

    #  transitions

    def add_transition(self, state_name, action_name, next_state_name):
        """
        Register one transition. (state, action) -> next_state.
        Auto-creates vocab entries if missing.
        """
        if state_name not in self._states:
            self.add_state(state_name)
        if action_name not in self._actions:
            self.add_action(action_name)
        if next_state_name not in self._states:
            self.add_state(next_state_name)

        s_hv  = self._states[state_name]
        a_hv  = self._actions[action_name]
        ns_hv = self._perm(self._states[next_state_name])   # permute next-state

        edge_hv = _bind(s_hv, a_hv)
        bound   = _bind(edge_hv, ns_hv)
        self._trans_accum += bound.astype(np.int32)
        self._edges.append((state_name, action_name, next_state_name))
        return bound

    @property
    def n_transitions(self):
        return len(self._edges)

    #  prediction

    def predict(self, state_name, action_name, candidate_states=None, top_k=3):
        """
        Predict next_state given (state, action).
        Recovery: bind(bind(state, action), trans_accum) ~= next_state_hv.

        Returns ranked candidate states by cosine similarity.
        candidate_states defaults to all registered states.
        """
        s_hv  = self._states.get(state_name)
        a_hv  = self._actions.get(action_name)
        if s_hv is None or a_hv is None:
            return []

        edge_hv = _bind(s_hv, a_hv)
        # Recover via superposition unbind (use raw accum for SNR)
        recovered = self._trans_accum.astype(np.float32) * edge_hv

        if candidate_states is None:
            candidate_states = list(self._states.keys())

        results = []
        for cand in candidate_states:
            cand_hv = self._states.get(cand)
            if cand_hv is None:
                continue
            # Compare against PERMUTED candidate (next-state was stored permuted)
            cand_perm = self._perm(cand_hv)
            sim = float(np.dot(recovered, cand_perm)) / self.d
            results.append((cand, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    #  rollout

    def rollout(self, start_state, action_seq, top_k=1):
        """
        Multi-step prediction. Starting from start_state, apply action_seq.
        Returns trajectory: [(state, action, next_state, score), ...]
        """
        trajectory = []
        current = start_state
        for action in action_seq:
            pred = self.predict(current, action, top_k=top_k)
            if not pred:
                break
            next_state, score = pred[0]
            trajectory.append((current, action, next_state, score))
            current = next_state
        return trajectory

    #  causal explanation

    def explain(self, target_state, current_state, candidate_actions=None, depth=3):
        """
        Greedy BFS: find action sequence from current to target within `depth`.
        Returns ([actions], success_bool).
        """
        if candidate_actions is None:
            candidate_actions = list(self._actions.keys())

        from collections import deque
        queue   = deque([(current_state, [])])
        visited = {current_state}

        while queue:
            state, path = queue.popleft()
            if state == target_state:
                return path, True
            if len(path) >= depth:
                continue
            for action in candidate_actions:
                pred = self.predict(state, action, top_k=1)
                if not pred:
                    continue
                next_state, score = pred[0]
                if score < 0.1:
                    continue
                if next_state in visited:
                    continue
                visited.add(next_state)
                queue.append((next_state, path + [action]))
        return [], False

    #  introspection

    @property
    def n_states(self):
        return len(self._states)

    @property
    def n_actions(self):
        return len(self._actions)

    def edges(self):
        return list(self._edges)
