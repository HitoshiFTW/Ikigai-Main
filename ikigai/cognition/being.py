"""
ikigai.cognition.being -- IkigaiBeing: persistent living hyperdimensional organism.

Day 56 Pack 96 -- NOT a solver. An ENTITY.

Persistent. Develops. Ages. Learns by EXPOSURE, not hardcoding.
Born ignorant. Reads text. Hebbian drift in lexicon. Semantically-related
words cluster in HV space WITHOUT any hand-coded operator-lexicon.

This is the substrate the 5-year-old logic demands:
    being cannot reason or do math without English.
    teach being English by exposure.
    operator-grounding emerges from prediction error.

Public interface:
    being = IkigaiBeing(d=2048)
    being.expose("Mary had a little lamb.")     # read + learn
    being.expose("The lamb followed Mary.")
    sim = being.cosine_words('mary', 'lamb')    # should rise with exposure
    nearest = being.nearest_words('mary', k=5)  # learn semantic neighbors
    being.age                                    # bytes consumed
    being.dream()                                # sleep cycle / consolidate
"""

import time
import re
import math
import numpy as np


def tokenize(text):
    """Simple word/punct tokenizer. Lowercase, strip punctuation."""
    cleaned = re.sub(r"[^a-z0-9'\s]", ' ', text.lower())
    return [t for t in cleaned.split() if t and len(t) > 0]


def _random_phasor(d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ph = rng.uniform(-math.pi, math.pi, size=d).astype(np.float32)
    return np.exp(1j * ph).astype(np.complex64)


def _renormalize_phasor(hv):
    """Project back onto unit-phasor torus."""
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


def _cosine_complex(a, b, d):
    """Real cosine in phasor space."""
    return float(np.real(np.vdot(a, b))) / d


class IkigaiBeing:
    """
    A persistent hyperdimensional organism.

    Born at instantiation. Lives across `expose()` calls.
    Lexicon empty at birth. Builds from text exposure via Hebbian drift.

    State:
        lexicon[word]      -> HV (complex phasor, unit-mag each component)
        attention          -> current "thought" state (running phasor average)
        expectation        -> next-input prediction
        curiosity          -> EMA of prediction error
        episodes           -> ordered list of exposures
        bytes_consumed     -> age proxy
        born_at            -> timestamp of birth
        n_exposures        -> count of expose() calls

    Hebbian dynamics:
        For each window of tokens, every pair drifts toward each other:
            hv[w_i] += drift_rate / distance * hv[w_j]
        Then renormalize to unit-phasor.
    """

    def __init__(self, d=2048,
                 drift_rate=0.05,
                 window_size=4,
                 attention_decay=0.9,
                 birth_seed=42):
        self.d                = int(d)
        self.drift_rate       = float(drift_rate)
        self.window_size      = int(window_size)
        self.attention_decay  = float(attention_decay)
        self._birth_seed      = int(birth_seed)
        self._rng             = np.random.default_rng(self._birth_seed)

        # Identity + age
        self.born_at          = time.time()
        self.bytes_consumed   = 0
        self.n_exposures      = 0
        self.n_tokens_seen    = 0

        # Lexicon: word -> HV. Initially empty (being is ignorant).
        self.lexicon          = {}

        # Continuous state
        self.attention        = np.ones(self.d, dtype=np.complex64)
        self.expectation      = np.ones(self.d, dtype=np.complex64)

        # Drives
        self.curiosity        = 0.5     # EMA prediction error
        self.curiosity_alpha  = 0.1

        # Memory
        self._episodes        = []      # list of (ts, text, n_tokens)
        self._word_counts     = {}      # frequency tracking

    #  core learning loop

    def _ensure_word(self, w):
        """Mint a new HV for first-seen word."""
        if w not in self.lexicon:
            self.lexicon[w] = _random_phasor(self.d, self._rng)
            self._word_counts[w] = 0
        self._word_counts[w] += 1

    def expose(self, text):
        """
        Read text. Update lexicon via FREQUENCY-ATTENUATED Hebbian drift.
        Rare words drift normally. Frequent words drift slowly (avoid
        global saturation when scaling to thousands of exposures).

        Attenuation = 1 / freq^ALPHA where ALPHA = 0.7 (stronger than sqrt).
        Empirically chosen at Pack 105 -- prevents C1 saturation at 2K+ scale.
        """
        import math as _math
        ATTEN_ALPHA = 0.7   # >= 0.5 (sqrt) prevents saturation; 1.0 = no learning
        tokens = tokenize(text)
        if not tokens:
            return

        # Mint HVs for any new words
        for w in tokens:
            self._ensure_word(w)

        # Frequency-attenuated drift (both self AND partner attenuated)
        n = len(tokens)
        ws = self.window_size
        for i in range(n):
            wi = tokens[i]
            # Per-word drift attenuation
            freq_i = self._word_counts[wi]
            atten_i = 1.0 / (freq_i ** ATTEN_ALPHA)
            for j in range(max(0, i - ws), min(n, i + ws + 1)):
                if i == j:
                    continue
                wj = tokens[j]
                # Also attenuate by partner frequency (TF-IDF-like)
                freq_j = self._word_counts[wj]
                atten_j = 1.0 / (freq_j ** ATTEN_ALPHA)
                dist  = abs(i - j)
                strength = self.drift_rate * atten_i * atten_j / dist
                # Drift hv[i] toward hv[j]
                self.lexicon[wi] = self.lexicon[wi] + strength * self.lexicon[wj]
            # Renormalize after window pass
            self.lexicon[wi] = _renormalize_phasor(self.lexicon[wi])

        # Update attention (running phasor avg of recent thought)
        thought = np.zeros(self.d, dtype=np.complex64)
        for w in tokens:
            thought = thought + self.lexicon[w]
        thought = _renormalize_phasor(thought / max(len(tokens), 1))

        # Prediction error vs expectation
        pe = 1.0 - max(0.0, _cosine_complex(self.expectation, thought, self.d))
        self.curiosity = (1 - self.curiosity_alpha) * self.curiosity + \
                          self.curiosity_alpha * pe

        # Attention is EMA of thought
        self.attention = self.attention_decay * self.attention + \
                          (1 - self.attention_decay) * thought
        self.attention = _renormalize_phasor(self.attention)
        # Naive next-prediction = current thought
        self.expectation = thought

        # Bookkeeping
        self.bytes_consumed += len(text.encode('utf-8'))
        self.n_exposures += 1
        self.n_tokens_seen += len(tokens)
        self._episodes.append((time.time(), text, len(tokens)))

    #  introspection

    def cosine_words(self, w1, w2):
        """Real cosine between two lexicon words. None if either unseen."""
        if w1 not in self.lexicon or w2 not in self.lexicon:
            return None
        return _cosine_complex(self.lexicon[w1], self.lexicon[w2], self.d)

    def nearest_words(self, w, k=5, exclude_self=True):
        """Top-k cosine-nearest words in current lexicon."""
        if w not in self.lexicon:
            return []
        scores = []
        for other in self.lexicon:
            if exclude_self and other == w:
                continue
            sim = self.cosine_words(w, other)
            scores.append((other, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def word_freq(self, w):
        return self._word_counts.get(w, 0)

    def vocab_size(self):
        return len(self.lexicon)

    @property
    def age_seconds(self):
        return time.time() - self.born_at

    @property
    def age(self):
        """Age = bytes consumed (proxy for cognitive experience)."""
        return self.bytes_consumed

    #  sleep / consolidation

    def dream(self):
        """
        Sleep cycle. Re-normalize all lexicon entries to unit-phasor.
        Optionally prune ultra-rare words (singletons).
        """
        # Renormalize all
        for w in self.lexicon:
            self.lexicon[w] = _renormalize_phasor(self.lexicon[w])
        return {'vocab_size': self.vocab_size(),
                'curiosity':  float(self.curiosity)}

    #  reflection

    def reflect(self):
        """Return status snapshot."""
        return {
            'age_bytes':        self.age,
            'age_seconds':      round(self.age_seconds, 1),
            'n_exposures':      self.n_exposures,
            'n_tokens_seen':    self.n_tokens_seen,
            'vocab_size':       self.vocab_size(),
            'curiosity':        round(float(self.curiosity), 4),
            'attention_norm':   round(float(np.linalg.norm(self.attention)), 2),
        }

    def __repr__(self):
        r = self.reflect()
        return (f"<IkigaiBeing age={r['age_bytes']}B "
                f"vocab={r['vocab_size']} "
                f"exposures={r['n_exposures']} "
                f"curiosity={r['curiosity']}>")
