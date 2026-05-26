"""
ikigai.cognition.metacognitive_mirror -- Metacognitive HV Mirror.

Day 55 Pack 50 -- *6 Decisive stack: self-model via HV binding.

Problem: system has no model of its own belief state across turns.
         No way to detect when own output diverges from belief.
         LLMs have no explicit self-tracking at inference time.

Fix: MetacognitiveHVMirror binds belief + emission each turn.
     self_hv = bind(B_U_bipolar, emit_hv)
     drift = 1 - cosine(self_hv_t, self_hv_{t-1})
     high_drift() -> request clarification / flag inconsistency

Binding algebra:
     B_U_bipolar = sign(B_U)         # float belief -> +-1
     emit_hv = encode(emit_tokens)   # tokens -> +-1 HV
     self_hv = B_U_bipolar * emit_hv # component product (bind)

     recover_emit: bind(self_hv, B_U_bipolar) ~= emit_hv (self-inverse)

No forgetting: _self_hv_history and _drift_log append-only.
               All turns recoverable. Monotone n_turns.

vs LLM: LLM has no persistent self-model between forward passes.
        Mirror: exact +-1 self_hv per turn, drift computed analytically.
"""

import numpy as np


def _hv(word, d):
    seed = hash(f'mirror::{word}') & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=d) * 2 - 1).astype(np.float32)


def _encode(tokens, d):
    s = np.zeros(d, dtype=np.float32)
    for t in tokens:
        s += _hv(t, d)
    out = np.sign(s).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _bind(a, b):
    return a * b


def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _to_bipolar(v):
    """Convert float vector to +-1 bipolar."""
    out = np.sign(v).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


class MetacognitiveHVMirror:
    """
    Self-model via per-turn HV binding of belief + emission.

    update(B_U, emit_tokens) -> (self_hv, drift)
        -- bind belief + emission, compute drift from prev turn

    recover_emit(B_U, self_hv) -> emit_hv_approx
        -- unbind B_U from self_hv to recover emission HV

    high_drift(threshold=0.3) -> bool
        -- True when last drift exceeds threshold

    drift_log, self_hv_history -- append-only, no forgetting
    """

    def __init__(self, d=64, drift_threshold=0.3):
        self.d = d
        self.drift_threshold = drift_threshold
        self._self_hv_history = []   # list of +-1 self_hv per turn
        self._drift_log       = []   # float drift per turn
        self._turn            = 0

    #  per-turn update

    def update(self, B_U, emit_tokens):
        """
        Compute self_hv = bind(sign(B_U), encode(emit_tokens)).
        Drift = 1 - cosine(self_hv, self_hv_prev). 0 on first turn.
        Returns (self_hv, drift).
        """
        d_actual   = B_U.shape[0] if hasattr(B_U, 'shape') else self.d
        B_bipolar  = _to_bipolar(np.asarray(B_U, dtype=np.float32))
        emit_hv    = _encode(emit_tokens, d_actual)
        self_hv    = _bind(B_bipolar, emit_hv)

        if self._self_hv_history:
            prev  = self._self_hv_history[-1]
            drift = 1.0 - _cosine(self_hv, prev)
        else:
            drift = 0.0

        self._self_hv_history.append(self_hv)
        self._drift_log.append(float(drift))
        self._turn += 1
        return self_hv, float(drift)

    #  recovery

    def recover_emit(self, B_U, self_hv):
        """
        Unbind B_U from self_hv to recover emit_hv approximation.
        Exact for bipolar +-1 (bind is self-inverse).
        """
        B_bipolar = _to_bipolar(np.asarray(B_U, dtype=np.float32))
        return _bind(B_bipolar, self_hv)

    #  drift queries

    def drift(self):
        """Last turn drift. 0.0 if no turns yet."""
        return self._drift_log[-1] if self._drift_log else 0.0

    def mean_drift(self, last_n=None):
        """Mean drift over all (or last N) turns."""
        log = self._drift_log if not last_n else self._drift_log[-last_n:]
        return float(np.mean(log)) if log else 0.0

    def high_drift(self, threshold=None):
        """True if last drift > threshold (default: self.drift_threshold)."""
        thr = threshold if threshold is not None else self.drift_threshold
        return self.drift() > thr

    def drift_trend(self):
        """Slope of drift over time (positive = increasing instability)."""
        if len(self._drift_log) < 2:
            return 0.0
        x = np.arange(len(self._drift_log), dtype=np.float32)
        y = np.array(self._drift_log, dtype=np.float32)
        return float(np.polyfit(x, y, 1)[0])

    #  stats

    @property
    def n_turns(self):
        return self._turn

    @property
    def drift_log(self):
        return list(self._drift_log)

    @property
    def self_hv_history(self):
        return list(self._self_hv_history)

    def summary(self):
        return {
            'n_turns':    self._turn,
            'last_drift': self.drift(),
            'mean_drift': self.mean_drift(),
            'high_drift': self.high_drift(),
            'drift_trend': self.drift_trend(),
        }
