"""
ikigai.cognition.persona_fe_coupling -- Persona × Free-Energy Coupling.

Day 55 Pack 75 -- Phase B: voice/style depends on internal FE state.

Maps internal state (free energy, surprise, conflict) -> persona dimensions:
    high FE     -> uncertain, hedging, formal language
    low FE      -> confident, direct, casual
    high conflict -> apologetic, qualifying
    low conflict  -> assertive
    high surprise -> curious, exclamatory
    low surprise  -> matter-of-fact

5 manifold dimensions (matches BeliefProjectionManifold):
    VALENCE     -- positive vs negative tone
    AROUSAL     -- calm vs excited
    CERTAINTY   -- hedged vs confident
    FORMALITY   -- casual vs formal
    TECHNICALITY -- plain vs jargon

Coupling matrix:
    persona_dim = sum_i (fe_metric_i * coupling_weight_{i, dim})

Output: a 5-dim persona vector applied to response generation.

Bio: limbic system (mood) + prefrontal (deliberation) jointly shape behavior.
     Anxiety (high cortisol) -> more cautious responses.

vs LLM: persona = prompt-engineered, brittle.
        Persona×FE: emergent from internal state. Real-time adaptive.
"""

import numpy as np


# Persona dimension indices (matches BeliefProjectionManifold)
VALENCE     = 0
AROUSAL     = 1
CERTAINTY   = 2
FORMALITY   = 3
TECHNICALITY = 4
N_DIMS      = 5

DIM_NAMES = ['valence', 'arousal', 'certainty', 'formality', 'technicality']


class PersonaFEC:
    """
    Couples internal FE metrics -> persona manifold dimensions.

    set_fe_metric(name, value)
        Set a free-energy metric (e.g. 'belief_conflict', 'novelty', 'surprise').

    set_coupling(metric_name, dim, weight)
        Set how this metric affects this persona dimension.

    persona() -> dict {dim_name: value} in [-1, 1]
        Compute current persona vector.

    style_descriptor() -> str
        Human-readable summary of current persona.

    update_from_organism(organism)
        Pull metrics from a UnifiedOrganism's status and update persona.
    """

    def __init__(self, decay=0.05):
        self.decay        = float(decay)
        self._metrics     = {}    # name -> current value
        self._couplings   = {}    # name -> {dim_idx: weight}
        self._persona     = np.zeros(N_DIMS, dtype=np.float32)

        # Default couplings -- biologically motivated
        # belief_conflict: high = uncertain (low certainty), apologetic (low arousal)
        self.set_coupling('belief_conflict', CERTAINTY, -1.0)
        self.set_coupling('belief_conflict', AROUSAL,   -0.3)
        self.set_coupling('belief_conflict', FORMALITY, +0.4)
        # novelty: high = curious (high arousal, casual)
        self.set_coupling('novelty',         AROUSAL,    +0.7)
        self.set_coupling('novelty',         FORMALITY,  -0.3)
        # surprise: high = curious, exclamatory
        self.set_coupling('surprise',        AROUSAL,    +0.6)
        self.set_coupling('surprise',        VALENCE,    +0.2)
        # cortisol / threat: defensive
        self.set_coupling('threat',          VALENCE,    -0.5)
        self.set_coupling('threat',          CERTAINTY,  -0.2)
        self.set_coupling('threat',          FORMALITY,  +0.5)
        # technicality marker: high = jargon
        self.set_coupling('technicality_pref', TECHNICALITY, +1.0)
        # confidence (low FE): increases certainty
        self.set_coupling('confidence',      CERTAINTY,  +1.0)
        self.set_coupling('confidence',      VALENCE,    +0.3)

    # ── metric updates ────────────────────────────────────────────────────

    def set_fe_metric(self, name, value):
        self._metrics[name] = float(value)

    def get_metric(self, name):
        return float(self._metrics.get(name, 0.0))

    def set_coupling(self, metric_name, dim_idx, weight):
        if metric_name not in self._couplings:
            self._couplings[metric_name] = {}
        self._couplings[metric_name][int(dim_idx)] = float(weight)

    # ── persona computation ──────────────────────────────────────────────

    def compute(self):
        """Recompute persona vector from current metrics + couplings."""
        new_p = np.zeros(N_DIMS, dtype=np.float32)
        for metric_name, dim_weights in self._couplings.items():
            m_val = self._metrics.get(metric_name, 0.0)
            for dim_idx, weight in dim_weights.items():
                new_p[dim_idx] += weight * m_val
        # Smooth with previous (EMA-style)
        self._persona = (1.0 - self.decay) * self._persona + self.decay * new_p
        # Clip to [-1, 1]
        self._persona = np.clip(self._persona, -1.0, 1.0)
        return self._persona.copy()

    def persona(self):
        """Returns dict {dim_name: value}."""
        return {name: float(self._persona[i]) for i, name in enumerate(DIM_NAMES)}

    def persona_vector(self):
        return self._persona.copy()

    def style_descriptor(self):
        """Human-readable summary."""
        p = self._persona
        adjectives = []
        if p[CERTAINTY] > 0.3: adjectives.append('confident')
        if p[CERTAINTY] < -0.3: adjectives.append('hedging')
        if p[VALENCE] > 0.3: adjectives.append('positive')
        if p[VALENCE] < -0.3: adjectives.append('cautious')
        if p[AROUSAL] > 0.3: adjectives.append('engaged')
        if p[AROUSAL] < -0.3: adjectives.append('calm')
        if p[FORMALITY] > 0.3: adjectives.append('formal')
        if p[FORMALITY] < -0.3: adjectives.append('casual')
        if p[TECHNICALITY] > 0.3: adjectives.append('technical')
        if p[TECHNICALITY] < -0.3: adjectives.append('plain')
        return ', '.join(adjectives) if adjectives else 'neutral'

    # ── integration with organism ─────────────────────────────────────────

    def update_from_organism(self, organism):
        """
        Pull FE-style metrics from a UnifiedOrganism status and update persona.
        Maps:
          belief_conflicts -> 'belief_conflict' (normalized)
          recent novelty   -> 'novelty'
          safety_blocks    -> 'threat'
        """
        status = organism.status() if hasattr(organism, 'status') else {}
        n_beliefs = max(1, status.get('n_beliefs', 1))
        conflicts = status.get('belief_conflicts', 0)
        self.set_fe_metric('belief_conflict', conflicts / n_beliefs)

        # Novelty from atomizer episode load
        n_eps = status.get('n_episodes', 0)
        n_atoms = max(1, status.get('n_atoms', 1))
        self.set_fe_metric('novelty', min(1.0, n_eps / (n_atoms * 5.0)))

        # Threat from safety blocks
        n_proc = max(1, status.get('n_processed', 1))
        threat_rate = status.get('safety_blocks', 0) / n_proc
        self.set_fe_metric('threat', min(1.0, threat_rate * 5.0))

        # Confidence = 1 - normalized conflicts
        self.set_fe_metric('confidence', max(0.0, 1.0 - conflicts / n_beliefs))

        return self.compute()

    @property
    def n_metrics(self):
        return len(self._metrics)

    def all_metrics(self):
        return dict(self._metrics)
