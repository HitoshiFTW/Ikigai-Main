"""
ikigai.cognition.cross_time_resonance -- Cross-Time Resonance Memory.

Day 55 Pack 60 -- complete #5: nested phasor oscillators, beat-frequency replay.

Architecture:
    N oscillators with periods T_0 < T_1 < ... < T_{N-1} (ticks per cycle).
    Each oscillator i has angular frequency omega_i = 2*pi / T_i.
    Phase at tick t: phi_i(t) = omega_i * t  (mod 2*pi)

Beat frequency between oscillators i and j:
    beat_period(i,j) = T_i * T_j / |T_j - T_i|
    beat_strength(i,j,t) = 0.5 * (1 + cos(phi_i(t) - phi_j(t)))  in [0,1]
    1.0 = perfect alignment (resonance), 0.0 = anti-phase

Encoding:
    encode(name, tokens, t): store event in ALL oscillator buffers at tick t.
    Each buffer indexed by phase-bin at encoding time.

Replay at resonance:
    detect_resonance(t, threshold) -> pairs (i,j) with beat_strength > threshold
    recall_near_phase(osc_idx, t) -> events stored near current phase in buffer
    replay(t) -> {(i,j): recalled events} for all resonant pairs

Biological analogy:
    Short-period osc = hippocampal gamma (working memory, fast replay)
    Long-period osc  = cortical slow wave (long-term consolidation)
    Resonance        = cross-frequency coupling -> memory consolidation trigger

vs LLM: LLM has no temporal structure. All tokens equally weighted.
        CrossTimeResonator: events at aligned timescales automatically surface.
        Replay = biological-style memory consolidation. Zero gradient.
"""

import math
import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CrossTimeResonator:
    """
    Nested oscillator hierarchy with beat-frequency event replay.
    Periods are in ticks; events accumulate in per-oscillator phase buffers.
    """

    def __init__(self, d=400, periods=None, n_bins=16):
        if periods is None:
            periods = [10, 100, 1000]
        self.d       = d
        self.periods = list(periods)
        self.n_osc   = len(periods)
        self.n_bins  = n_bins
        self.omegas  = [2.0 * math.pi / p for p in periods]
        self._tick   = 0

        # per-oscillator, per-phase-bin: list of (tick, name, tokens, hv)
        self._buffers = [{} for _ in periods]
        self._resonance_log = []

    #  phase geometry

    def phase(self, osc_idx, tick=None):
        """Phase in [0, 2*pi) for oscillator i at given tick."""
        if tick is None:
            tick = self._tick
        return (self.omegas[osc_idx] * tick) % (2.0 * math.pi)

    def phase_bin(self, osc_idx, tick=None):
        """Discretized phase bin index in [0, n_bins)."""
        phi = self.phase(osc_idx, tick)
        return int(phi / (2.0 * math.pi) * self.n_bins) % self.n_bins

    def beat_strength(self, i, j, tick=None):
        """
        Alignment between oscillators i and j at tick.
        Returns 1.0 (perfect resonance) to 0.0 (anti-phase).
        beat_period(i,j) = T_i * T_j / |T_j - T_i|
        """
        if tick is None:
            tick = self._tick
        phi_i = self.phase(i, tick)
        phi_j = self.phase(j, tick)
        return 0.5 * (1.0 + math.cos(phi_i - phi_j))

    def beat_period(self, i, j):
        """Theoretical beat period (ticks) between oscillators i and j."""
        t_i, t_j = self.periods[i], self.periods[j]
        return (t_i * t_j) / abs(t_j - t_i) if t_i != t_j else float('inf')

    def beat_matrix(self, tick=None):
        """n_osc x n_osc matrix of all pairwise beat strengths."""
        n = self.n_osc
        m = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                m[i, j] = self.beat_strength(i, j, tick)
        return m

    #  encode

    def encode(self, name, tokens, tick=None):
        """
        Store event in all oscillator phase buffers at tick.
        Returns event HV.
        """
        if tick is None:
            tick = self._tick
        hv = _encode(tokens, self.d)
        for i in range(self.n_osc):
            b = self.phase_bin(i, tick)
            if b not in self._buffers[i]:
                self._buffers[i][b] = []
            self._buffers[i][b].append((tick, name, list(tokens), hv))
        return hv

    #  resonance detection

    def detect_resonance(self, tick=None, threshold=0.85):
        """
        Find oscillator pairs in near-alignment at tick.
        Returns [(i, j, strength), ...].
        """
        if tick is None:
            tick = self._tick
        result = []
        for i in range(self.n_osc):
            for j in range(i + 1, self.n_osc):
                s = self.beat_strength(i, j, tick)
                if s >= threshold:
                    result.append((i, j, float(s)))
        return result

    def find_resonance_ticks(self, n_ticks, i=0, j=1, threshold=0.85):
        """Scan ticks 0..n_ticks-1, return those where (i,j) resonate."""
        return [t for t in range(n_ticks)
                if self.beat_strength(i, j, t) >= threshold]

    #  recall + replay

    def recall_near_phase(self, osc_idx, tick=None, top_k=3):
        """
        Retrieve events from oscillator buffer at bins near current phase.
        Returns [(name, tokens, phase_sim), ...] sorted by phase similarity.
        """
        if tick is None:
            tick = self._tick
        phi = self.phase(osc_idx, tick)
        events = []
        for b_idx, entries in self._buffers[osc_idx].items():
            # Phase of bin center
            bin_phi = (b_idx + 0.5) / self.n_bins * 2.0 * math.pi
            sim = 0.5 * (1.0 + math.cos(phi - bin_phi))
            for (t, name, tokens, hv) in entries:
                events.append((sim, name, tokens))
        events.sort(key=lambda x: -x[0])
        return [(name, tokens, float(sim)) for (sim, name, tokens) in events[:top_k]]

    def replay(self, tick=None, threshold=0.85):
        """
        At resonance ticks: retrieve events from aligned oscillator buffers.
        Returns {(i, j): [(name, tokens, beat_strength), ...]} for each resonant pair.
        """
        if tick is None:
            tick = self._tick
        pairs = self.detect_resonance(tick, threshold)
        result = {}
        for (i, j, strength) in pairs:
            recalled = self.recall_near_phase(j, tick, top_k=1)  # slower osc = longer memory
            if recalled:
                result[(i, j)] = [(n, tok, strength) for (n, tok, _) in recalled]
        if result:
            self._resonance_log.append((tick, result))
        return result

    #  tick advance

    def advance(self, n=1):
        """Advance internal tick counter by n steps. Returns new tick."""
        self._tick += n
        return self._tick

    #  introspection

    @property
    def tick(self):
        return self._tick

    def n_events(self, osc_idx=0):
        """Total events stored in oscillator buffer."""
        return sum(len(v) for v in self._buffers[osc_idx].values())

    def phase_all(self, tick=None):
        return [self.phase(i, tick) for i in range(self.n_osc)]

    def resonance_history(self):
        return list(self._resonance_log)
