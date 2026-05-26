"""
ikigai.cognition.cgpsp_encoder -- Continuous Geometric Phase Space Parser.

Day 56 Pack 84 -- Pillar 1 of UHE.
Byte stream -> phasor state via state-dependent rotation flow.

Foundation:
    s ∈ ℂ^d, each component on unit circle.
    Per-byte phase vector θ_c ∈ ℝ^d (random, fixed).
    Per-byte coupling matrix Π_c (sparse cyclic shift with byte-specific stride).

Flow:
    α_t      = (1 + Re⟨s_t, R_phase⟩) / 2          // state projection [0, 1]
    s_t+1    = Π_c . (s_t ⊙ exp(i.θ_c.(1 + γ.α_t)))   // rotate then shift

Properties:
    - Same byte at different states -> different rotation magnitude (non-linear)
    - Π_c shifts -> couples components (no longer elementwise)
    - State always on unit-phasor torus (unitary evolution)
    - Byte stream -> continuous trajectory in ℂ^d

vs standard VSA:
    Discrete codebook lookup => continuous integration
    Commutative bundle => non-commutative flow (word order matters by construction)
    OOV problem => none (bytes only)
"""

import numpy as np


def _hv_for(key, d, dtype):
    """Deterministic random vector keyed by (key, d, dtype)."""
    rng  = np.random.default_rng(abs(hash((key, dtype))) % (2**31))
    if dtype == 'phase':
        # Random phases ∈ [-π, π]
        return rng.uniform(-np.pi, np.pi, size=d).astype(np.float32)
    elif dtype == 'shift':
        # Coprime-with-d stride
        return int(rng.integers(1, d - 1)) | 1   # force odd
    elif dtype == 'role':
        # Random phasor (unit norm)
        ph = rng.uniform(-np.pi, np.pi, size=d).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)
    elif dtype == 'bipolar':
        return (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    raise ValueError(dtype)


def _byte_class(byte_val):
    """Map byte -> character class for shared phase basis."""
    if 48 <= byte_val <= 57:       # digit
        return 'digit'
    if 65 <= byte_val <= 90:       # uppercase
        return 'letter'
    if 97 <= byte_val <= 122:      # lowercase
        return 'letter'
    if byte_val in (32, 9, 10, 13):  # whitespace
        return 'ws'
    if 33 <= byte_val <= 47 or 58 <= byte_val <= 64:  # punctuation
        return 'punct'
    if byte_val in (40, 41, 91, 93, 123, 125, 60, 62):  # brackets
        return 'bracket'
    return 'other'


def _byte_normalize(byte_val):
    """Pure deterministic mapping: case-fold + collapse whitespace classes."""
    if 65 <= byte_val <= 90:   # uppercase -> lowercase
        return byte_val + 32
    if byte_val in (9, 10, 13):  # tab/LF/CR -> space
        return 32
    return byte_val


class CGPSPEncoder:
    """
    Continuous Geometric Phase Space Parser.

    d                : hypervector dimension (default 2048)
    gamma            : state-feedback gain (default 0.4)
    couple_mode      : 'shift' uses byte-specific cyclic shifts; 'none' = pure rotation

    encode(text) -> ℂ^d phasor state (numpy complex64)
    encode_bytes(b) -> same, from raw bytes
    cosine(a, b) -> real cosine similarity in ℂ^d (uses Re⟨a,b̄⟩ / (||a|| ||b||))
    """

    def __init__(self, d=2048, gamma=0.4, couple_mode='shift',
                 normalize_bytes=True, class_basis_weight=0.6):
        self.d                  = int(d)
        self.gamma              = float(gamma)
        self.couple_mode        = couple_mode
        self.normalize_bytes    = bool(normalize_bytes)
        self.class_basis_weight = float(class_basis_weight)
        # Per-class base phase vectors (shared across bytes in same class)
        self._class_phase = {
            cls: _hv_for(f'class_{cls}', self.d, 'phase')
            for cls in ('digit', 'letter', 'ws', 'punct', 'bracket', 'other')
        }
        # Per-byte phase vectors θ_c = (1-w) * class_base + w * individual
        # This means letters share most of their direction; differences are individual.
        individual = np.stack([
            _hv_for(f'theta_indiv_{c}', self.d, 'phase') for c in range(256)
        ])
        class_basis = np.stack([
            self._class_phase[_byte_class(c)] for c in range(256)
        ])
        w = self.class_basis_weight
        self._theta = (w * class_basis + (1 - w) * individual).astype(np.float32)
        # Per-byte cyclic shift strides (only depend on class, not individual)
        # This ensures within-class consistency
        class_strides = {
            cls: _hv_for(f'stride_class_{cls}', self.d, 'shift')
            for cls in ('digit', 'letter', 'ws', 'punct', 'bracket', 'other')
        }
        self._strides = np.array([
            class_strides[_byte_class(c)] for c in range(256)
        ], dtype=np.int32)
        # State-projection role (fixed random phasor)
        self._R_phase = _hv_for('R_phase', self.d, 'role')
        # Initial state: uniform phase
        self._init_state = np.ones(self.d, dtype=np.complex64)

    def reset(self):
        return self._init_state.copy()

    def _step(self, s, byte_val):
        """One CGPSP integration step: rotate then couple."""
        # State projection α ∈ [0, 1]
        alpha = (1.0 + float(np.real(np.vdot(self._R_phase, s))) / self.d) / 2.0
        # Per-byte phase rotation, scaled by (1 + gamma.alpha)
        scale = 1.0 + self.gamma * alpha
        theta = self._theta[byte_val] * scale
        rot   = np.exp(1j * theta).astype(np.complex64)
        s_rot = s * rot
        # Couple via cyclic shift (byte-specific stride)
        if self.couple_mode == 'shift':
            s_out = np.roll(s_rot, self._strides[byte_val])
        else:
            s_out = s_rot
        return s_out

    def encode_bytes(self, b):
        """Encode raw bytes -> final phasor state."""
        if isinstance(b, str):
            b = b.encode('utf-8')
        s = self._init_state.copy()
        for byte_val in b:
            byte_val = int(byte_val)
            if self.normalize_bytes:
                byte_val = _byte_normalize(byte_val)
            s = self._step(s, byte_val)
        return s

    def encode(self, text):
        return self.encode_bytes(text)

    def cosine(self, a, b):
        """Real-valued cosine for complex phasor HVs."""
        num = float(np.real(np.vdot(a, b)))
        den = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return num / den if den > 0 else 0.0

    #  reverse for sanity (NOT exact decoder, but trajectory invertible at byte granularity)

    def _step_inverse(self, s, byte_val):
        """Inverse step: undo coupling then derotate."""
        # Undo shift
        if self.couple_mode == 'shift':
            s_rot = np.roll(s, -int(self._strides[byte_val]))
        else:
            s_rot = s
        # Undo rotation -- need same alpha that was used FORWARD.
        # Forward used alpha computed from PRE-step state. Recovering requires
        # iterating. For now we approximate using current-state alpha as
        # close approximation (works when γ is small).
        alpha = (1.0 + float(np.real(np.vdot(self._R_phase, s_rot))) / self.d) / 2.0
        scale = 1.0 + self.gamma * alpha
        theta = self._theta[byte_val] * scale
        rot   = np.exp(-1j * theta).astype(np.complex64)
        return s_rot * rot

    def decode_bytes(self, s_final, byte_sequence):
        """
        Approximate inverse: given final state and the byte sequence,
        backtrack to recover initial state. Tests reversibility.
        """
        s = s_final.copy()
        for byte_val in reversed(byte_sequence):
            s = self._step_inverse(s, int(byte_val))
        return s
