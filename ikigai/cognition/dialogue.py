"""
ikigai.cognition.dialogue -- Multi-turn dialogue layer.

Day 56 Pack 106 -- Phase 3 of curriculum.

Wraps IkigaiOrganism with conversation state:
    - turn_history: ordered list of (role, text, turn_hv)
    - current_persona: active PGMW persona (Pack 86)
    - context_hv: running HV summary of conversation
    - role-binding via Pi_k (Pack 85): ROLE_USER vs ROLE_AGENT asymmetric

Each turn:
    1. Mint turn HV from text (CGPSP-encoded via being)
    2. Bind with role HV using Pi_k chain
    3. Add to history
    4. Update context (running average)
    5. Standard `org.read()` still updates 5 channels w/ text

Recall via:
    - relevant_turns(query): cosine on turn HVs
    - persona-modulated retrieval (PGMW lens)

Conversation persona basins: aggregate turn HVs by tagged persona
(friendly, formal, technical, empathetic, etc.) -> SACField attractors.
Conversation drifts into nearest basin -> defines voice.
"""

import re
import numpy as np

from ikigai.cognition.being import IkigaiBeing, _renormalize_phasor
from ikigai.cognition.pi_k_algebra import PiK


ROLE_USER  = '__role_user__'
ROLE_AGENT = '__role_agent__'


def _random_role_hv(name, d):
    """Deterministic role HV from string key."""
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    ph = rng.uniform(-np.pi, np.pi, size=d).astype(np.float32)
    return np.exp(1j * ph).astype(np.complex64)


class Turn:
    """One conversation turn."""
    __slots__ = ('role', 'text', 'hv', 'idx', 'meta')

    def __init__(self, role, text, hv, idx, meta=None):
        self.role  = role
        self.text  = text
        self.hv    = hv
        self.idx   = idx
        self.meta  = meta or {}

    def __repr__(self):
        return f"<Turn #{self.idx} {self.role}: {self.text[:40]!r}...>"


class DialogueLoop:
    """
    Multi-turn dialogue wrapper around IkigaiOrganism.

    Public:
        start(persona_name=None)
        user_says(text) -> Turn
        agent_says(text) -> Turn
        context_hv()
        recall_turns(query, top_k=3)
        n_turns
        history()
        reset()
        switch_persona(name)
    """

    def __init__(self, organism, d=2048):
        self.org = organism
        self.d   = int(d)
        self.pi  = PiK(d=self.d, n_primes=32)

        # Role HVs (fixed seeded random phasors)
        self._role_user  = _random_role_hv(ROLE_USER,  self.d)
        self._role_agent = _random_role_hv(ROLE_AGENT, self.d)

        # Conversation state
        self._turns          = []
        self._context_hv     = None
        self._active_persona = None
        self._next_idx       = 0

    #  conversation lifecycle

    def start(self, persona_name=None):
        """Begin a fresh conversation. Optionally activate a persona basin."""
        self._turns       = []
        self._context_hv  = None
        self._next_idx    = 0
        if persona_name is not None and persona_name in self.org.persona._personas:
            self.org.persona.set_active(persona_name)
            self._active_persona = persona_name
        else:
            self.org.persona.clear_active()
            self._active_persona = None
        return self

    def reset(self):
        return self.start()

    def switch_persona(self, name):
        if name in self.org.persona._personas:
            self.org.persona.set_active(name)
            self._active_persona = name
            return True
        return False

    #  turn helpers

    def _encode_text(self, text):
        """Get a phasor HV for the turn from being's encoder (CGPSP byte-trajectory)."""
        return self.org.encoder.encode(text)

    def _semantic_text_hv(self, text):
        """
        Semantic bag-of-lexicon HV. Average of token HVs from being's lexicon.
        Used for RECALL only (catches synonyms via learned Hebbian similarity).
        """
        import re as _re
        tokens = _re.findall(r"[a-z0-9']+", text.lower())
        if not tokens:
            return self._encode_text(text)
        accum = np.zeros(self.d, dtype=np.complex64)
        n_in = 0
        for tok in tokens:
            if tok in self.org.being.lexicon:
                accum = accum + self.org.being.lexicon[tok]
                n_in += 1
        if n_in == 0:
            return self._encode_text(text)
        return _renormalize_phasor(accum / n_in)

    def _make_turn_hv(self, text, role_hv):
        """Turn HV = bind(role, text_encoding) via Pi_k for asymmetry."""
        text_hv = self._encode_text(text)
        # Use index-0 of Pi_k family (cyclic shift by 2)
        return self.pi.bind(0, role_hv, text_hv)

    def _add_turn(self, role, text, hv):
        turn = Turn(role=role, text=text, hv=hv, idx=self._next_idx)
        self._next_idx += 1
        self._turns.append(turn)
        # Update running context (EMA)
        if self._context_hv is None:
            self._context_hv = hv.copy()
        else:
            alpha = 0.3
            self._context_hv = _renormalize_phasor(
                (1 - alpha) * self._context_hv + alpha * hv
            )
        return turn

    #  primary interface

    def user_says(self, text):
        """Process a user turn: read into organism + bind w/ USER role."""
        # 1. Organism reads (all 5 channels update)
        self.org.read(text)
        # 2. Turn HV
        hv = self._make_turn_hv(text, self._role_user)
        # 3. Append + update context
        return self._add_turn('user', text, hv)

    def agent_says(self, text):
        """Process an agent turn."""
        self.org.read(text)
        hv = self._make_turn_hv(text, self._role_agent)
        return self._add_turn('agent', text, hv)

    #  retrieval

    def context_hv(self):
        return self._context_hv if self._context_hv is not None \
               else np.zeros(self.d, dtype=np.complex64)

    def recall_turns(self, query_text, top_k=3, role_filter=None):
        """
        Return top-k past turns most similar to query (semantic bag-of-lexicon).
        Catches synonyms via learned Hebbian similarity (e.g. dogs ~ puppies).
        role_filter: None | 'user' | 'agent'
        """
        if not self._turns:
            return []
        q_hv = self._semantic_text_hv(query_text)
        results = []
        for turn in self._turns:
            if role_filter and turn.role != role_filter:
                continue
            text_hv = self._semantic_text_hv(turn.text)
            sim = float(np.real(np.vdot(q_hv, text_hv))) / self.d
            results.append((turn, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    #  persona inspection

    def active_persona(self):
        return self._active_persona

    def conversation_persona_score(self):
        """
        For each registered persona, score how aligned the current context is.
        Returns dict {persona_name: score}.
        """
        if self._context_hv is None:
            return {}
        scores = {}
        for name in self.org.persona._personas:
            p_hv, _ = self.org.persona._personas[name]
            sim = float(np.real(np.vdot(self._context_hv, p_hv))) / self.d
            scores[name] = sim
        return scores

    #  introspection

    @property
    def n_turns(self):
        return len(self._turns)

    def history(self):
        return list(self._turns)

    def last_turn(self):
        return self._turns[-1] if self._turns else None

    def __repr__(self):
        return (f"<DialogueLoop turns={self.n_turns} "
                f"persona={self._active_persona}>")
