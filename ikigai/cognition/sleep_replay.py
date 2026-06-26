"""
ikigai.cognition.sleep_replay -- Sleep-replay consolidation on the flat substrate.

Invention #6 from the Kill Stack (2026-05-27). The organism logs its
exposures during waking; during sleep it replays them with optional
amplification + decay, then rebuilds concept HVs. The substrate counter
accumulates the high-frequency patterns more strongly; cold rows fade.

Public API:
    buf = ExposureBuffer(maxlen=10_000)
    buf.log(text)                 # during waking
    buf.snapshot()                # list of entries
    consolidator = SleepConsolidator(organism, buf)
    consolidator.consolidate(replay_factor=3, decay=0.99,
                              build_concepts=True, preset='analogy')

Replay reinforcement deepens the substrate's representation of patterns
the organism actually saw. Decay (multiplicative on counter rows) makes
inactive locations slowly fade, mimicking biological synaptic homeostasis.
After consolidation, concept HVs are rebuilt to reflect the refreshed
substrate.

This is the cleanest "low-hanging revolution" in the kill stack -- Ikigai
already had sleep + dream systems in ikigai.py, just never wired to the
substrate. Pack 159 finally connects them.
"""

import time
import random
import numpy as np
from collections import deque


class ExposureBuffer:
    """
    Bounded ring buffer of recent exposures the organism processed during
    waking. Each entry is (text, timestamp, meta). Snapshot can be replayed
    during sleep.
    """

    def __init__(self, maxlen=10_000):
        self.buffer = deque(maxlen=int(maxlen))
        self.total_logged = 0

    def log(self, text, meta=None):
        if not text: return
        self.buffer.append((text, time.time(), meta or {}))
        self.total_logged += 1

    def snapshot(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class SleepConsolidator:
    """
    Sleep-replay consolidation. Replays the exposure buffer through the
    organism with optional amplification + decay, then rebuilds concepts.
    """

    def __init__(self, organism, exposure_buffer):
        self.org = organism
        self.buf = exposure_buffer
        self.last_stats = {}

    def consolidate(self, replay_factor=3, decay=None,
                    build_concepts=True, concept_words=None,
                    concept_preset=None, shuffle=True, verbose=False):
        """
        Run a sleep cycle.

        replay_factor: how many times to replay each buffered exposure
                       through expose_cooccur + expose_transitions +
                       expose_meaning. >=1 = reinforce, 0 = skip replay.
        decay: optional multiplicative scalar (0..1) applied to the
               cooccur counter bank C. Simulates biological synaptic
               homeostasis -- cold rows fade. None = no decay.
        build_concepts: rebuild concept HVs after replay.
        concept_words: optional vocabulary for concept rebuild.
        concept_preset: ConceptSynthesizer preset ('analogy', 'general', ...)
        shuffle: replay in random order, not log order.

        Returns a stats dict.
        """
        t0 = time.perf_counter()
        snapshot = self.buf.snapshot()
        n = len(snapshot)
        stats = {
            'n_entries': n,
            'replay_factor': int(replay_factor),
            'decay': decay,
            'replayed_writes': 0,
            'pre_substrate_bytes': self.org.unified.substrate_bytes(),
        }
        if n == 0:
            stats['elapsed'] = 0.0
            self.last_stats = stats
            return stats

        # 1. Replay phase
        if replay_factor and replay_factor > 0:
            order = list(range(n))
            if shuffle:
                random.shuffle(order)
            for _ in range(int(replay_factor)):
                for i in order:
                    text, _, meta = snapshot[i]
                    self.org.unified.expose_cooccur(text)
                    self.org.unified.expose_transitions(text)
                    stats['replayed_writes'] += 1
                    # if meta carries vocab hints, hand to expose_meaning
                    if meta:
                        self.org.expose_meaning(text, **meta)
            if verbose:
                print(f'  [sleep] replayed {stats["replayed_writes"]} writes '
                      f'({time.perf_counter()-t0:.2f}s)')

        # 2. Decay phase (multiplicative homeostasis on cooccur bank)
        if decay is not None and 0 < decay < 1:
            self.org.unified.sdm.C *= np.complex64(decay)
            if verbose:
                print(f'  [sleep] applied decay {decay} to cooccur bank')

        # 3. Rebuild concepts to reflect the refreshed substrate
        if build_concepts:
            self.org.build_concepts(words=concept_words,
                                    preset=concept_preset,
                                    write_to_substrate=False,
                                    verbose=False)
            stats['concept_words'] = len(self.org._concepts.concepts)

        stats['post_substrate_bytes'] = self.org.unified.substrate_bytes()
        stats['substrate_fixed'] = (stats['pre_substrate_bytes']
                                     == stats['post_substrate_bytes'])
        stats['elapsed'] = time.perf_counter() - t0
        self.last_stats = stats
        return stats
