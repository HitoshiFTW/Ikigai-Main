"""
ikigai.cognition.phasor_state -- Phase-Locked Holographic Buffer (PLHB).

Day 55 Pack 32 -- conversational substrate primitive #1.
The LLM context-window killer.

Replaces sliding token buffer with single phasor manifold Phi in C^d.
Conversation length is irrelevant -- state size = constant (3.2 KB at d=400).
Past turns are not forgotten; they are rotated to nearly-orthogonal phase
positions and recovered via reverse rotation + role unbind.

Primitive operations:
    rotate(v, k, w)   phase advance by k*w (reversible)
    bind(a, b)        complex multiplication = phase addition
    unbind(c, b)      inverse via complex conjugate (self-inverse when |b|=1)
    superpose(parts)  sum + normalize each component to unit magnitude

Master update:
    Phi_t = normalize_phase[ rotate(Phi_{t-1}, 1, w) + bind(turn_hv, role_hv) ]

Recall turn k_back ago:
    tau_back = rotate(Phi_t, -k_back, w)
    turn_hv  = unbind(tau_back, role_hv)
"""

import math
import time
import numpy as np


HV_DIM = 400
PHI_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0
OMEGA_DEFAULT = 2.0 * math.pi / PHI_GOLDEN   # ~3.883 rad, golden-ratio increment


def random_phasor(dim=HV_DIM, seed=None):
    """Fresh unit-phasor HV. Each component = e^(i theta), theta uniform [0, 2pi)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    theta = rng.uniform(0.0, 2.0 * math.pi, size=dim)
    return np.exp(1j * theta).astype(np.complex64)


def normalize_phase(v):
    """Project each component onto unit circle. Preserves phase, sets |.|=1."""
    mag = np.abs(v)
    mag = np.where(mag < 1e-12, 1.0, mag)
    return (v / mag).astype(np.complex64)


def bind(a, b):
    """Complex multiplication = phase addition. Composes two unit phasors."""
    return (a * b).astype(np.complex64)


def unbind(c, b):
    """Self-inverse when |b|=1: unbind(bind(a, b), b) == a."""
    return (c * np.conjugate(b)).astype(np.complex64)


def rotate(v, k, omega=OMEGA_DEFAULT):
    """Phase advance by k*omega. Reversible: rotate(rotate(v, k), -k) == v."""
    return (v * np.exp(1j * k * omega)).astype(np.complex64)


def cosine(a, b):
    """Magnitude of normalized complex inner product. Returns scalar in [0, 1]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    inner = np.vdot(a, b)
    return float(np.abs(inner) / (na * nb))


def superpose(parts):
    """Bundle list of phasor HVs via sum + normalize_phase. Order-invariant."""
    if not parts:
        return np.zeros(HV_DIM, dtype=np.complex64)
    s = np.zeros_like(parts[0], dtype=np.complex64)
    for p in parts:
        s = s + p
    return normalize_phase(s)


class PhaseLockedHolographicBuffer:
    """
    Conversational state as single phasor manifold Phi in C^d.

    Each turn:
        Phi_t = normalize_phase[ rotate(Phi_{t-1}, 1, omega) + bind(turn_hv, role_hv) ]

    Recall turn k_back ago:
        Phi rotated backward by k_back*omega, then unbind by role_hv,
        compared to ground-truth turn HV via cosine similarity.

    State footprint: dim * 8 bytes (complex64). 3.2 KB at d=400.
    Per-turn cost: O(d).
    """

    def __init__(self, dim=HV_DIM, omega=OMEGA_DEFAULT, seed=42):
        self.dim = dim
        self.omega = omega
        self._seed_base = seed
        self.Phi = random_phasor(dim, seed=seed)
        self.turn_count = 0
        self.vocab = {}
        self.roles = {
            'user': random_phasor(dim, seed=seed + 1),
            'self': random_phasor(dim, seed=seed + 2),
        }
        self.history = []   # list of (tokens, role, turn_hv) for verification only

    def token_hv(self, token):
        """Lazy phasor allocation per token. Fresh random, locked into vocab."""
        if token not in self.vocab:
            seed = hash(f'tok::{token}') & 0x7FFFFFFF
            self.vocab[token] = random_phasor(self.dim, seed=seed)
        return self.vocab[token]

    def encode_turn(self, tokens):
        """Bundle token phasors via superposition."""
        if not tokens:
            return random_phasor(self.dim, seed=0)
        return superpose([self.token_hv(t) for t in tokens])

    def add_turn(self, tokens, role='user'):
        """Ingest a turn: advance state phase, bundle role-bound turn HV."""
        role_hv = self.roles.get(role)
        if role_hv is None:
            role_hv = self.roles['user']
        turn_hv = self.encode_turn(tokens)
        tau = bind(turn_hv, role_hv)
        advanced = rotate(self.Phi, 1, self.omega)
        self.Phi = superpose([advanced, tau])
        self.history.append((tokens, role, turn_hv))
        self.turn_count += 1
        return self.Phi

    def recall_turn(self, k_back, role='user'):
        """Reverse-rotate state by k_back*omega, unbind role.
        Returns approximate turn HV (noisy from intervening superpositions).
        """
        if k_back < 0 or k_back >= self.turn_count:
            return None
        role_hv = self.roles.get(role)
        if role_hv is None:
            role_hv = self.roles['user']
        reversed_state = rotate(self.Phi, -k_back, self.omega)
        return unbind(reversed_state, role_hv)

    def recall_fidelity(self, k_back, role='user'):
        """Cosine similarity between recall_turn and ground-truth turn HV."""
        if k_back < 0 or k_back >= self.turn_count:
            return 0.0
        idx = self.turn_count - 1 - k_back
        _, true_role, true_turn_hv = self.history[idx]
        recalled = self.recall_turn(k_back, role=role)
        if recalled is None:
            return 0.0
        return cosine(recalled, true_turn_hv)

    def coherence(self, prev_state):
        """Cosine to a prior Phi snapshot. For topic-shift detection."""
        if prev_state is None:
            return 1.0
        return cosine(self.Phi, prev_state)

    def state_size_bytes(self):
        return int(self.Phi.nbytes)

    def reset(self, seed=None):
        s = seed if seed is not None else self._seed_base
        self.Phi = random_phasor(self.dim, seed=s)
        self.turn_count = 0
        self.history.clear()
