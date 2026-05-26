"""
ikigai.cognition.sensory_grounding -- Sensory/Experiential Channel.

Day 56 Pack 98 -- Channel 3 of grounding.

Property words (red, heavy, sad) cannot be learned from co-occurrence alone.
They need REFERENCE VECTORS pointing to stable ambient properties.

Architecture:
    Fixed anchor HVs at substrate level:
        R_VISUAL_RED, R_VISUAL_BLUE, R_VISUAL_GREEN, R_VISUAL_YELLOW
        R_WEIGHT_HEAVY, R_WEIGHT_LIGHT
        R_EMOTION_HAPPY, R_EMOTION_SAD, R_EMOTION_ANGRY, R_EMOTION_AFRAID
        R_TEMPERATURE_HOT, R_TEMPERATURE_COLD
        R_SIZE_BIG, R_SIZE_SMALL
        R_SOUND_LOUD, R_SOUND_QUIET
        R_TASTE_SWEET, R_TASTE_BITTER

    During text exposure:
        Sentence "The sky is blue" -> word 'blue' drifts toward R_VISUAL_BLUE
        Sentence "She was crying" -> word 'crying' drifts toward R_EMOTION_SAD
                                     (via supervised seed-word mapping or context)

    Seed mapping: anchor name -> canonical word list
        R_VISUAL_RED -> ['red', 'crimson', 'scarlet']
        R_EMOTION_SAD -> ['sad', 'crying', 'tears', 'sorrow']

    Hebbian rule when seed word seen:
        word_hv = renormalize(word_hv + drift_rate * R_anchor)

    Propagation: words co-occurring with anchored words inherit partial anchor.

Pure HDC. Hand-defined ANCHORS (the senses are not learnable, they're given).
Words learn association via exposure.
"""

import re
import numpy as np


def _random_phasor_seed(seed, d):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(-np.pi, np.pi, size=d).astype(np.float32)
    return np.exp(1j * ph).astype(np.complex64)


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


# Default anchor registry: (anchor_name, seed_id, canonical_seed_words)
DEFAULT_ANCHORS = [
    # Color (visual)
    ('R_VISUAL_RED',      1001, ['red', 'crimson', 'scarlet', 'rose']),
    ('R_VISUAL_BLUE',     1002, ['blue', 'azure', 'navy', 'cyan']),
    ('R_VISUAL_GREEN',    1003, ['green', 'emerald', 'lime', 'verdant']),
    ('R_VISUAL_YELLOW',   1004, ['yellow', 'gold', 'golden', 'amber']),
    ('R_VISUAL_BLACK',    1005, ['black', 'dark', 'ebony', 'obsidian']),
    ('R_VISUAL_WHITE',    1006, ['white', 'snow', 'ivory', 'pale']),
    # Weight
    ('R_WEIGHT_HEAVY',    2001, ['heavy', 'weighty', 'massive', 'dense']),
    ('R_WEIGHT_LIGHT',    2002, ['light', 'feather', 'airy', 'weightless']),
    # Emotion
    ('R_EMOTION_HAPPY',   3001, ['happy', 'joyful', 'glad', 'cheerful', 'smiling', 'laughing']),
    ('R_EMOTION_SAD',     3002, ['sad', 'crying', 'tears', 'sorrow', 'mournful', 'weeping']),
    ('R_EMOTION_ANGRY',   3003, ['angry', 'furious', 'rage', 'mad', 'irate']),
    ('R_EMOTION_AFRAID',  3004, ['afraid', 'scared', 'fear', 'terrified', 'frightened']),
    # Temperature
    ('R_TEMP_HOT',        4001, ['hot', 'warm', 'burning', 'scorching', 'fire']),
    ('R_TEMP_COLD',       4002, ['cold', 'freezing', 'icy', 'frosty', 'chilly']),
    # Size
    ('R_SIZE_BIG',        5001, ['big', 'large', 'huge', 'giant', 'enormous', 'vast']),
    ('R_SIZE_SMALL',      5002, ['small', 'tiny', 'little', 'mini', 'wee']),
    # Sound
    ('R_SOUND_LOUD',      6001, ['loud', 'booming', 'thunderous', 'roaring']),
    ('R_SOUND_QUIET',     6002, ['quiet', 'silent', 'whisper', 'hushed']),
    # Taste
    ('R_TASTE_SWEET',     7001, ['sweet', 'sugary', 'honey', 'syrupy']),
    ('R_TASTE_BITTER',    7002, ['bitter', 'sour', 'tart']),
    # Speed
    ('R_SPEED_FAST',      8001, ['fast', 'quick', 'rapid', 'speedy', 'swift']),
    ('R_SPEED_SLOW',      8002, ['slow', 'sluggish', 'crawling']),
]


def tokenize(text):
    cleaned = re.sub(r"[^a-z0-9'\s]", ' ', text.lower())
    return [t for t in cleaned.split() if t]


class SensoryGrounding:
    """
    Sensory/Experiential channel: words bind to fixed property anchors.

    register_anchor(name, seed, canonical_words)
        Add an anchor (fixed HV) + canonical seed words.
    expose(text, lexicon, drift_rate, context_drift, window)
        Update word HVs in given lexicon based on seed-word anchoring +
        contextual propagation.
    nearest_anchor(word, lexicon)
        Which anchor is this word most aligned with?
    """

    def __init__(self, d=2048, anchors=None):
        self.d = int(d)
        self._anchors = {}     # name -> HV
        self._seeds   = {}     # word -> anchor_name (reverse map)

        # Load defaults (or custom)
        if anchors is None:
            anchors = DEFAULT_ANCHORS
        for name, seed, words in anchors:
            self.register_anchor(name, seed, words)

    def register_anchor(self, name, seed, canonical_words):
        self._anchors[name] = _random_phasor_seed(seed, self.d)
        for w in canonical_words:
            self._seeds[w.lower()] = name
        return self._anchors[name]

    def anchor(self, name):
        return self._anchors.get(name)

    @property
    def n_anchors(self):
        return len(self._anchors)

    def anchor_names(self):
        return list(self._anchors.keys())

    # ── exposure: drift lexicon words toward anchors ────────────────────────

    def expose(self, text, lexicon, drift_rate=0.15, context_drift=0.04, window=3):
        """
        For each token, if it's a SEED for an anchor -> drift its HV toward anchor.
        Also drift nearby context words by smaller amount (anchor propagation).

        Modifies `lexicon` in place. Assumes lexicon is dict of word->phasor HV.
        Words not in lexicon are minted with random phasor.
        """
        tokens = tokenize(text)
        if not tokens:
            return

        # Mint missing words (seeded for determinism)
        for t in tokens:
            if t not in lexicon:
                lexicon[t] = _random_phasor_seed(abs(hash(t)) % (2**31), self.d)

        # Find seed-positions: tokens that map to anchor
        seed_positions = []
        for i, tok in enumerate(tokens):
            if tok in self._seeds:
                seed_positions.append((i, tok, self._seeds[tok]))

        # Direct anchoring: seed word drifts toward its anchor
        for i, tok, anchor_name in seed_positions:
            anchor_hv = self._anchors[anchor_name]
            lexicon[tok] = _renorm(lexicon[tok] + drift_rate * anchor_hv)

        # Contextual propagation: words near seed positions get partial drift
        for i, tok, anchor_name in seed_positions:
            anchor_hv = self._anchors[anchor_name]
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j == i:
                    continue
                neighbor = tokens[j]
                if neighbor in self._seeds:    # skip other seed words
                    continue
                distance = abs(i - j)
                strength = context_drift / distance
                lexicon[neighbor] = _renorm(lexicon[neighbor] + strength * anchor_hv)

    # ── inspection ───────────────────────────────────────────────────────────

    def nearest_anchor(self, word_hv):
        """Which anchor best aligns with this word HV?"""
        best_name, best_score = None, -2.0
        for name, anchor_hv in self._anchors.items():
            sim = float(np.real(np.vdot(anchor_hv, word_hv))) / self.d
            if sim > best_score:
                best_score = sim
                best_name = name
        return best_name, best_score

    def all_anchor_scores(self, word_hv):
        return {name: float(np.real(np.vdot(a, word_hv))) / self.d
                for name, a in self._anchors.items()}
