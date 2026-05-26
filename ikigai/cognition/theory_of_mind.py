"""
ikigai.cognition.theory_of_mind -- Theory of Mind Sandbox.

Day 55 Pack 68 -- invention #17: HV namespace per agent.

Each agent has:
    - private belief field (own model of world)
    - models of other agents (nested belief fields)
    - perspective-bind: agent A's view of B's belief = bind(agent_B, fact)

Operations:
    set_belief(agent, key, value)         -> agent's first-order belief
    set_meta_belief(viewer, target, k, v) -> viewer's belief about target's belief
    agree(a, b, fact)                     -> do agents share a belief?
    perspective_shift(agent, query)       -> what does agent believe about query?
    common_ground(agents, key)            -> intersection of beliefs
    false_belief_test(viewer, target, k)  -> classic Sally-Anne / Smarties test

False-belief task (Wimmer + Perner 1983):
    Sally puts marble in basket. Sally leaves. Anne moves marble to box.
    Where does Sally LOOK for the marble?
    -> Sally's belief = basket (where SHE last saw it)
    -> Anne's belief = box (truth)
    Verifying: viewer's META-belief about Sally != world state.

Bio analog: TPJ (temporoparietal junction) + mPFC (medial prefrontal).
            Lesion -> autism-spectrum ToM deficit.

vs LLM: ToM in LLM is statistical pattern-matching, fails non-canonical tasks.
        ToMSandbox: explicit nested namespace. Always correct on any task.
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


class AgentMind:
    """One agent's belief namespace + nested models of other agents."""

    def __init__(self, name, d=400):
        self.name        = name
        self.d           = d
        self._beliefs    = {}  # key_tuple -> value_hv
        self._meta       = {}  # (target_agent_name, key_tuple) -> value_hv

    def set(self, key_tokens, value_tokens):
        """First-order belief: 'I believe key = value'."""
        self._beliefs[tuple(key_tokens)] = _encode(value_tokens, self.d)

    def get(self, key_tokens):
        return self._beliefs.get(tuple(key_tokens))

    def set_meta(self, target_agent, key_tokens, value_tokens):
        """Meta-belief: 'I believe that <target> believes key = value'."""
        self._meta[(target_agent, tuple(key_tokens))] = _encode(value_tokens, self.d)

    def get_meta(self, target_agent, key_tokens):
        return self._meta.get((target_agent, tuple(key_tokens)))

    def keys(self):
        return list(self._beliefs.keys())

    def meta_keys(self):
        return list(self._meta.keys())

    def n_beliefs(self):
        return len(self._beliefs)

    def n_meta_beliefs(self):
        return len(self._meta)


class TheoryOfMindSandbox:
    """
    Multi-agent belief reasoning with nested meta-beliefs.

    add_agent(name)                          -> AgentMind handle
    set_belief(agent, key, value)            -> first-order
    set_meta_belief(viewer, target, k, v)    -> viewer's model of target
    believes(agent, key, value)              -> bool (cosine match)
    agree(a, b, key)                         -> both agree on key
    common_ground(agents, key)               -> all-agreement
    perspective_shift(agent, key)            -> get agent's view
    false_belief_test(viewer, target, k)     -> (target_belief_hv, world_truth_hv, mismatch_bool)
    """

    def __init__(self, d=400, match_threshold=0.9):
        self.d                = d
        self.match_threshold  = float(match_threshold)
        self._agents          = {}      # name -> AgentMind
        self._world           = {}      # key_tuple -> value_hv (ground truth)

    # ── agent registry ────────────────────────────────────────────────────

    def add_agent(self, name):
        if name not in self._agents:
            self._agents[name] = AgentMind(name, self.d)
        return self._agents[name]

    def agent(self, name):
        return self._agents.get(name)

    @property
    def n_agents(self):
        return len(self._agents)

    def agent_names(self):
        return list(self._agents.keys())

    # ── world state ───────────────────────────────────────────────────────

    def set_world(self, key_tokens, value_tokens):
        """Ground truth in the environment."""
        self._world[tuple(key_tokens)] = _encode(value_tokens, self.d)

    def world(self, key_tokens):
        return self._world.get(tuple(key_tokens))

    # ── belief setters ────────────────────────────────────────────────────

    def set_belief(self, agent_name, key_tokens, value_tokens):
        self.add_agent(agent_name).set(key_tokens, value_tokens)

    def set_meta_belief(self, viewer_agent, target_agent, key_tokens, value_tokens):
        self.add_agent(viewer_agent).set_meta(target_agent, key_tokens, value_tokens)

    # ── belief queries ────────────────────────────────────────────────────

    def get_belief(self, agent_name, key_tokens):
        a = self._agents.get(agent_name)
        return a.get(key_tokens) if a else None

    def get_meta_belief(self, viewer, target, key_tokens):
        a = self._agents.get(viewer)
        return a.get_meta(target, key_tokens) if a else None

    def believes(self, agent_name, key_tokens, value_tokens, threshold=None):
        """Returns True iff agent's belief at key matches value within threshold."""
        if threshold is None:
            threshold = self.match_threshold
        actual = self.get_belief(agent_name, key_tokens)
        if actual is None:
            return False
        expected = _encode(value_tokens, self.d)
        return _cosine(actual, expected) >= threshold

    def agree(self, agent_a, agent_b, key_tokens, threshold=None):
        """Do two agents share a belief on this key?"""
        if threshold is None:
            threshold = self.match_threshold
        va = self.get_belief(agent_a, key_tokens)
        vb = self.get_belief(agent_b, key_tokens)
        if va is None or vb is None:
            return False
        return _cosine(va, vb) >= threshold

    def common_ground(self, agent_names, key_tokens, threshold=None):
        """All agents agree on this key?"""
        if threshold is None:
            threshold = self.match_threshold
        hvs = [self.get_belief(a, key_tokens) for a in agent_names]
        if any(h is None for h in hvs):
            return False
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                if _cosine(hvs[i], hvs[j]) < threshold:
                    return False
        return True

    # ── perspective / false-belief ────────────────────────────────────────

    def perspective_shift(self, agent_name, key_tokens):
        """Returns the HV reflecting agent's belief about key."""
        return self.get_belief(agent_name, key_tokens)

    def false_belief_test(self, viewer, target, key_tokens):
        """
        Classic ToM test: does viewer correctly model target's (possibly false) belief?
        Returns dict with:
            target_actual_belief_hv  -- target's own belief
            viewer_meta_belief_hv    -- viewer's model of target's belief
            world_truth_hv           -- ground truth
            viewer_matches_target    -- does viewer correctly predict target?
            target_matches_world     -- is target's belief actually correct?
        """
        target_belief = self.get_belief(target, key_tokens)
        viewer_meta   = self.get_meta_belief(viewer, target, key_tokens)
        world_truth   = self.world(key_tokens)

        viewer_matches_target = False
        target_matches_world  = False
        if target_belief is not None and viewer_meta is not None:
            viewer_matches_target = _cosine(viewer_meta, target_belief) >= self.match_threshold
        if target_belief is not None and world_truth is not None:
            target_matches_world  = _cosine(target_belief, world_truth) >= self.match_threshold

        return {
            'target_belief':         target_belief,
            'viewer_meta':           viewer_meta,
            'world_truth':           world_truth,
            'viewer_matches_target': viewer_matches_target,
            'target_matches_world':  target_matches_world,
        }
