"""
ikigai.cognition.wake_sleep -- Wake-Sleep Codebook Factorization (Day 54 Pack 23)

Rank 5 invention from research roadmap (+8% HumanEval projected).
DreamCoder analog for VSA: during sleep, scan accumulated transitions,
identify high-frequency patterns, bundle them into NEW atomic HVs that
become first-class vocabulary items.

Behavior:
    record(prev, curr)      -- log transition into matrix + frequency counter
    sleep(top_k, min_count) -- extract top-K frequent (prev,curr) -> create
                              compound HVs via bipolar bind (p * c)
    find_compound(p, c)     -- look up cached compound for a pair
    encode_via_compounds(seq)
                            -- encode sequence using available compounds first,
                              fallback to atomic HVs (lossy compression of
                              long sequences into shorter bundles)

Result: future queries match cached compounds directly, bypassing per-step
inference. Models the wake-sleep cycle in cortical replay (Stickgold 2005,
Walker 2009) and the abstraction-discovery loop of DreamCoder (Ellis 2021).
"""

from collections import Counter
import numpy as np


class WakeSleepCompressor:
    """
    Wraps a TransitionMemory + maintains its own transition log so that
    sleep() can identify recurring patterns by frequency (matrix loses
    per-transition frequency information after superposition).
    """

    def __init__(self, transition_memory):
        self.tm = transition_memory
        self.transition_log = []
        self.compound_vocab = {}      # compound_key -> bipolar int8 HV
        self.compound_origins = {}    # compound_key -> [(prev, curr)]
        self.n_sleep_cycles = 0

    def record(self, prev_key, curr_key):
        """Log transition into matrix AND frequency log."""
        self.tm.record(prev_key, curr_key)
        self.transition_log.append((prev_key, curr_key))

    def sleep(self, top_k=10, min_count=3):
        """
        Find the top-K most frequent (prev, curr) pairs occurring >= min_count
        times. Create compound HV for each via bipolar bind (element-wise mul).
        Returns list of newly created compound keys.
        """
        self.n_sleep_cycles += 1
        counts = Counter(self.transition_log)
        new_compounds = []
        for (prev, curr), n in counts.most_common(top_k):
            if n < min_count:
                break
            key = f'__compound__{prev}__{curr}'
            if key in self.compound_vocab:
                continue
            p = self.tm.vocab[prev].astype(np.int32)
            c = self.tm.vocab[curr].astype(np.int32)
            compound_hv = (p * c).astype(np.int8)   # bipolar XOR-equivalent
            self.compound_vocab[key] = compound_hv
            self.compound_origins[key] = [(prev, curr)]
            new_compounds.append(key)
        return new_compounds

    def n_compounds(self):
        return len(self.compound_vocab)

    def find_compound(self, prev_key, curr_key):
        return self.compound_vocab.get(f'__compound__{prev_key}__{curr_key}')

    def encode_via_compounds(self, seq):
        """
        Greedy 2-gram compression: walk seq, if (seq[i], seq[i+1]) is a
        compound use it (one HV), else use atomic (one HV per step).
        Returns the bundled (summed) int32 HV.
        """
        i = 0
        atoms = []
        n_compound_hits = 0
        while i < len(seq):
            if i + 1 < len(seq):
                comp = self.find_compound(seq[i], seq[i + 1])
                if comp is not None:
                    atoms.append(comp.astype(np.int32))
                    n_compound_hits += 1
                    i += 2
                    continue
            atomic = self.tm.vocab.get(seq[i])
            if atomic is not None:
                atoms.append(atomic.astype(np.int32))
            i += 1
        if not atoms:
            return np.zeros(self.tm.dim, dtype=np.int32), 0
        s = np.zeros(self.tm.dim, dtype=np.int32)
        for a in atoms:
            s += a
        return s, n_compound_hits
