"""
ikigai.cognition.operational_grounding -- Verb-as-Rotor learning.

Day 56 Pack 97 -- Operational Channel of grounding.

Words like ate, gives, gets, loses don't just live in co-occurrence space.
They CAUSE structural change in the organism's mind. Each verb is a ROTOR
in quantity space.

Phase encoding:
    quantity_hv(N) = exp(i * N * omega * q_axis)   for fixed q_axis
    rotor(c, M)    = exp(i * c * M * omega * q_axis)
    apply:         result_hv = quantity_hv(N) * rotor(c, M)
                            = quantity_hv(N + c*M)

Per-verb learning:
    See (N_before, verb, M, N_after) tuple from text.
    Observed delta = N_after - N_before
    Inferred c for this example = delta / M
    Hebbian-update verb's coefficient c.

Over time:
    "ate" -> c ~= -1   (subtraction)
    "got" -> c ~= +1   (addition)
    "lost" -> c ~= -1
    "buys" -> c ~= +1

This is GROUNDED meaning -- verb hypervector tells you what to DO with quantity,
not just which other words it co-occurs with.

Pure HDC. No gradient. Hebbian only.
"""

import re
import numpy as np


def tokenize(text):
    cleaned = re.sub(r"[^a-z0-9'\s]", ' ', text.lower())
    return [t for t in cleaned.split() if t]


def extract_numbers_in_order(text):
    return [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)]


class OperationalGrounding:
    """
    Verb-as-rotor lexicon.

    Each verb has:
        c     : learned coefficient (signed real). c*M = effect on quantity.
        rotor : phase-encoded HV representation of c (for HV interop).
        n_obs : number of observations (Hebbian moving average count).

    Methods:
        observe_tuple(verb, n_before, modifier, n_after)
            Update verb's c via running average.

        observe_story(text)
            Extract (N, verb, M, K) from sentences using simple pattern.

        predict(n_before, verb, modifier)
            Return predicted post-quantity N + c*M.

        coefficient(verb)
            Get learned c (or None).

        rotor(verb)
            Get verb's rotor HV.
    """

    def __init__(self, d=2048, omega=0.05, learning_rate=0.2):
        self.d         = int(d)
        self.omega     = float(omega)
        self.lr        = float(learning_rate)

        # Fixed quantity-axis direction (bipolar +-1)
        rng_axis = np.random.default_rng(seed=12345)
        self.q_axis    = (rng_axis.integers(0, 2, size=self.d) * 2 - 1).astype(np.float32)

        # Per-verb learned state
        self._c        = {}   # verb -> running coefficient estimate
        self._n_obs    = {}   # verb -> observation count
        self._delta_hist = {} # verb -> list of (n_before, m, n_after) observations

    #  quantity encoding/decoding

    def encode_quantity(self, n):
        """N -> phasor with phase N*omega along q_axis."""
        phase = float(n) * self.omega * self.q_axis
        return np.exp(1j * phase).astype(np.complex64)

    def decode_quantity(self, hv):
        """Phasor -> recover N (median across components for noise tolerance)."""
        phases = np.angle(hv).astype(np.float32)
        ns = phases / (self.omega * self.q_axis + 1e-9)
        return float(np.median(ns))

    def rotor(self, verb, modifier=1.0):
        """Verb's rotor for given modifier. Returns phasor HV."""
        c = self._c.get(verb, 0.0)
        return self.encode_quantity(c * modifier)

    def coefficient(self, verb):
        return self._c.get(verb)

    #  observation / learning

    def observe_tuple(self, verb, n_before, modifier, n_after):
        """
        See: subject had n_before of object; verb modifier; now has n_after.
        Learn verb's coefficient via running average.
        """
        if modifier is None or abs(modifier) < 1e-9:
            return None
        delta = n_after - n_before
        c_est = delta / modifier
        # Running average update
        n = self._n_obs.get(verb, 0) + 1
        prev_c = self._c.get(verb, 0.0)
        # Weighted average toward this estimate
        new_c = prev_c + (c_est - prev_c) / n
        self._c[verb]    = new_c
        self._n_obs[verb] = n
        self._delta_hist.setdefault(verb, []).append((n_before, modifier, n_after))
        return new_c

    def observe_story(self, text, subject_obj_hint=None):
        """
        Parse a multi-sentence story w/ 3 numbers: before, modifier, after.
        Verb extracted from the sentence containing the MODIFIER (middle number).

        Format expected (loose):
            "X had N <obj>. Y <verb> M <obj>. Now there are K <obj>."

        Returns (verb, n_before, modifier, n_after, c) or None.
        """
        sentences = [s.strip() for s in re.split(r'[\.\!\?]+', text) if s.strip()]
        if len(sentences) < 2:
            return None

        all_nums = extract_numbers_in_order(text)
        if len(all_nums) < 3:
            return None

        n_before, modifier, n_after = all_nums[0], all_nums[1], all_nums[-1]

        # Find ACTION sentence: the one containing the modifier (2nd number).
        action_sentence = None
        for sent in sentences:
            nums_in_sent = extract_numbers_in_order(sent)
            if modifier in nums_in_sent and n_before not in nums_in_sent:
                action_sentence = sent
                break
        if action_sentence is None:
            # Fallback to second sentence
            action_sentence = sentences[1] if len(sentences) > 1 else sentences[0]

        tokens = tokenize(action_sentence)
        STOP = {'a', 'an', 'the', 'and', 'or', 'but', 'so', 'now', 'then', 'at',
                'in', 'on', 'of', 'to', 'from', 'has', 'have', 'had', 'is', 'are',
                'was', 'were', 'with', 'by', 'for', 'she', 'he', 'they', 'her',
                'him', 'them', 'i', 'you', 'we', 'it', 'this', 'that',
                'his', 'hers', 'their', 'theirs', 'away', 'more'}
        verb = None
        for t in tokens:
            if t in STOP: continue
            if t.replace('.', '').isdigit(): continue
            if subject_obj_hint and t in subject_obj_hint: continue
            if len(t) >= 3:
                verb = t
                break

        if verb is None:
            return None

        c = self.observe_tuple(verb, n_before, modifier, n_after)
        return (verb, n_before, modifier, n_after, c)

    #  prediction

    def predict(self, n_before, verb, modifier):
        """Predict post-state quantity N + c*M."""
        c = self._c.get(verb)
        if c is None:
            return None
        return n_before + c * modifier

    def apply_in_hv(self, n_before, verb, modifier):
        """HV-level apply: quantity_hv(N) * rotor(c, M) -> result_hv."""
        return self.encode_quantity(n_before) * self.rotor(verb, modifier)

    #  introspection

    def vocab(self):
        return list(self._c.keys())

    def coefficients(self):
        """Returns sorted (verb, c, n_obs) list."""
        rows = [(v, self._c[v], self._n_obs[v]) for v in self._c]
        rows.sort(key=lambda x: -abs(x[1]))
        return rows
