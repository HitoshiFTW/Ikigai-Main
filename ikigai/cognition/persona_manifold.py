"""
ikigai.cognition.persona_manifold -- Belief-State Projection Manifold (BSPM).

Day 55 Pack 34 -- conversational substrate primitive #3.
Replaces stateless prompt engineering with geometric belief tracking.

Three parallel representations:
    B_U in R^d        per-user belief manifold (EMA of utterance HVs)
    C_U in R^5        cognitive axes [valence, arousal, certainty, formality, technicality]
    P_self in R^d     rigid persona HV (frozen at init, immutable)

Output alignment:
    resp_aligned = normalize(P_self + gamma * proj(resp_base, B_U))
    where proj(a, b) = (a.b / |b|^2) * b

Persuasion tracking:
    delta_certainty = C_U[certainty]_t - C_U[certainty]_{t-1}
    positive = system persuading user; negative = user challenging

Invariant: B_U always L2-normalized. P_self never mutated after init.
"""

import numpy as np

VALENCE      = 0
AROUSAL      = 1
CERTAINTY    = 2
FORMALITY    = 3
TECHNICALITY = 4
N_AXES       = 5


def _token_hv(token, d):
    seed = hash(f'bspm::{token}') & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=d) * 2 - 1).astype(np.float32)


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _proj(a, b):
    """Orthogonal projection of a onto direction of b."""
    nb2 = float(np.dot(b, b))
    if nb2 < 1e-12:
        return np.zeros_like(a)
    return (float(np.dot(a, b)) / nb2) * b


class BeliefProjectionManifold:
    """
    Per-user belief manifold B_U + cognitive axes C_U + frozen persona P_self.

    B_U EMA: B_U_t = normalize((1 - alpha)*B_U_{t-1} + alpha*encode(tokens))
    C_U EMA: C_U_t = (1 - alpha)*C_U_{t-1} + alpha*extract_axes(tokens)
    align:   normalize(P_self + gamma * proj(resp_base, B_U))
    """

    _POSITIVE = {'yes', 'sure', 'agree', 'correct', 'exactly', 'certainly',
                 'true', 'right', 'good', 'great', 'perfect', 'definitely'}
    _NEGATIVE  = {'no', 'not', 'never', 'wrong', 'incorrect', 'disagree',
                  'false', 'bad', 'uncertain', 'maybe', 'doubt', 'hmm'}
    _AROUSAL   = {'urgent', 'emergency', 'critical', 'immediately', 'now',
                  'asap', 'fast', 'quick', 'help', 'problem', 'error', 'crash'}
    _FORMAL    = {'please', 'kindly', 'respectfully', 'regards', 'sincerely',
                  'however', 'therefore', 'thus', 'furthermore', 'moreover'}
    _TECH      = {'code', 'function', 'class', 'api', 'error', 'debug',
                  'algorithm', 'variable', 'import', 'module', 'type', 'compile'}

    def __init__(self, d=64, alpha=0.3, gamma=0.5, seed=42):
        self.d = d
        self.alpha = alpha
        self.gamma = gamma

        rng = np.random.RandomState(seed)
        self.P_self = _l2_normalize(rng.standard_normal(d).astype(np.float32))
        self._P_self_hash = self.P_self.tobytes()   # for immutability check

        self.B_U = _l2_normalize(rng.standard_normal(d).astype(np.float32))
        self.C_U = np.zeros(N_AXES, dtype=np.float32)
        self._C_U_prev = np.zeros(N_AXES, dtype=np.float32)

        self.turn_count = 0
        self.persuasion_log = []
        self.belief_log = []

    def encode_utterance(self, tokens):
        if not tokens:
            return np.zeros(self.d, dtype=np.float32)
        s = np.zeros(self.d, dtype=np.float32)
        for t in tokens:
            s = s + _token_hv(t, self.d)
        return _l2_normalize(s)

    def _extract_axes(self, tokens):
        toks = [t.lower() for t in tokens]
        n = max(1, len(toks))
        pos  = sum(1 for t in toks if t in self._POSITIVE)
        neg  = sum(1 for t in toks if t in self._NEGATIVE)
        arou = sum(1 for t in toks if t in self._AROUSAL)
        form = sum(1 for t in toks if t in self._FORMAL)
        tech = sum(1 for t in toks if t in self._TECH)

        return np.array([
            float(np.clip((pos - neg) / n * 4, -1, 1)),       # valence
            float(np.clip(arou / n * 6, -1, 1)),               # arousal
            float(np.clip((pos - neg * 0.5) / n * 5, -1, 1)), # certainty
            float(np.clip(form / n * 8, -1, 1)),               # formality
            float(np.clip(tech / n * 6, -1, 1)),               # technicality
        ], dtype=np.float32)

    def update(self, tokens):
        """Ingest utterance. Returns (B_U, C_U, delta_certainty)."""
        u_t = self.encode_utterance(tokens)
        axes_t = self._extract_axes(tokens)

        B_prev = self.B_U.copy()
        self._C_U_prev = self.C_U.copy()

        self.B_U = _l2_normalize((1.0 - self.alpha) * self.B_U + self.alpha * u_t)
        self.C_U = (1.0 - self.alpha) * self.C_U + self.alpha * axes_t

        self.belief_log.append(_cosine(self.B_U, B_prev))
        delta_cert = float(self.C_U[CERTAINTY] - self._C_U_prev[CERTAINTY])
        self.persuasion_log.append(delta_cert)
        self.turn_count += 1
        return self.B_U.copy(), self.C_U.copy(), delta_cert

    def align(self, resp_base):
        """resp_aligned = normalize(P_self + gamma * proj(resp_base, B_U))."""
        projection = _proj(resp_base, self.B_U)
        return _l2_normalize(self.P_self + self.gamma * projection)

    def persona_immutable(self):
        return self.P_self.tobytes() == self._P_self_hash

    def persuasion_score(self):
        if not self.persuasion_log:
            return 0.0
        return float(np.mean(self.persuasion_log))

    def belief_drift(self):
        if not self.belief_log:
            return 1.0
        return float(np.mean(self.belief_log))

    def persona_alignment(self, resp_hv):
        return _cosine(resp_hv, self.P_self)

    def user_alignment(self, resp_hv):
        return _cosine(resp_hv, self.B_U)

    def cognitive_state(self):
        return {
            'valence':      float(self.C_U[VALENCE]),
            'arousal':      float(self.C_U[AROUSAL]),
            'certainty':    float(self.C_U[CERTAINTY]),
            'formality':    float(self.C_U[FORMALITY]),
            'technicality': float(self.C_U[TECHNICALITY]),
        }

    def reset(self, seed=42):
        rng = np.random.RandomState(seed + 99)
        self.B_U = _l2_normalize(rng.standard_normal(self.d).astype(np.float32))
        self.C_U[:] = 0.0
        self._C_U_prev[:] = 0.0
        self.turn_count = 0
        self.persuasion_log.clear()
        self.belief_log.clear()
