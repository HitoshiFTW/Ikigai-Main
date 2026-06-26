"""
ikigai.cognition.grammar_grounding -- Channel 5: distributional grammar.

Day 56 Pack 101 -- emergent POS without hardcoding.

For each word, maintain TWO context vectors:
    left_ctx[w]   = bundle of HVs of words appearing IMMEDIATELY before w
    right_ctx[w]  = bundle of HVs of words appearing IMMEDIATELY after w

Words sharing left/right context vectors have same grammatical role.
This is distributional POS induction (Brown 1992, Schutze 1995) in HDC.

Also tracks bigram statistics for sentence boundary + phrase cohesion:
    bigram_count[(w_prev, w_next)] -- raw counts
    surprise(prev, curr) = -log(p(curr | prev))

POS classes EMERGE from clustering. No tags. No corpus. Pure statistics.

Public API:
    expose(text, lexicon)              update left/right ctx + bigrams
    pos_similarity(w1, w2)             cosine of context fingerprints
    pos_neighbors(word, k=5)           k grammatical-twins
    bigram_prob(prev, curr)            P(curr | prev)
    surprise(prev, curr)               -log of above
    is_phrase_boundary(prev, curr, threshold)
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    cleaned = re.sub(r"[^a-z0-9'\s]", ' ', text.lower())
    return [t for t in cleaned.split() if t]


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


def _cos_c(a, b, d):
    return float(np.real(np.vdot(a, b))) / d


class GrammarGrounding:
    """
    Distributional grammar via left/right context bundles per word.

    Maintains:
        _left_ctx[w]   :  cumulative bundle of preceding-word HVs
        _right_ctx[w]  :  cumulative bundle of following-word HVs
        _n_left[w]     :  count of left-context observations
        _n_right[w]    :  count of right-context observations
        _bigram[(p,c)] :  raw count of (p, c) bigram
        _word_count[w] :  total occurrences
        SENT_BOS, SENT_EOS : sentence boundary markers
    """

    SENT_BOS = '__bos__'
    SENT_EOS = '__eos__'

    def __init__(self, d=2048):
        self.d           = int(d)
        self._left_ctx   = {}    # word -> complex64 d-dim
        self._right_ctx  = {}    # word -> complex64 d-dim
        self._n_left     = Counter()
        self._n_right    = Counter()
        self._bigram     = Counter()
        self._word_count = Counter()

    # ── exposure ──────────────────────────────────────────────────────────

    def expose(self, text, lexicon):
        """
        For each sentence in text:
            Add BOS/EOS markers.
            For each adjacent pair (prev, curr):
                update bigram count
                update right_ctx[prev] with lexicon[curr] (TF-IDF weighted)
                update left_ctx[curr] with lexicon[prev] (TF-IDF weighted)
        """
        sentences = re.split(r'[\.\!\?]+', text)
        for sent in sentences:
            tokens = tokenize(sent)
            if not tokens:
                continue
            full = [self.SENT_BOS] + tokens + [self.SENT_EOS]
            for w in full:
                self._word_count[w] += 1
            for i in range(len(full) - 1):
                prev, curr = full[i], full[i + 1]
                self._bigram[(prev, curr)] += 1
                # Update right_ctx of prev w/ curr's HV (weighted by 1/log of curr's freq)
                if curr in lexicon:
                    self._update_ctx(self._right_ctx, self._n_right, prev,
                                     lexicon[curr], curr)
                # Update left_ctx of curr w/ prev's HV (weighted by 1/log of prev's freq)
                if prev in lexicon:
                    self._update_ctx(self._left_ctx, self._n_left, curr,
                                     lexicon[prev], prev)

    def _update_ctx(self, ctx_dict, n_dict, word, partner_hv, partner_word):
        """
        Update word's context bundle with weighted partner contribution.
        Weight = 1 / (1 + log1p(partner_freq)). Damps common partners ('the', 'a').
        """
        partner_freq = self._word_count.get(partner_word, 1)
        idf_weight   = 1.0 / (1.0 + math.log1p(partner_freq))
        weighted     = idf_weight * partner_hv

        if word not in ctx_dict:
            ctx_dict[word] = weighted.astype(partner_hv.dtype)
            n_dict[word] = 1
        else:
            n = n_dict[word]
            # Weighted running update (TF-IDF style)
            ctx_dict[word] = _renorm(
                ctx_dict[word] + idf_weight * (partner_hv - ctx_dict[word]) / (n + 1)
            )
            n_dict[word] += 1

    # ── similarity in grammatical role ────────────────────────────────────

    def pos_similarity(self, w1, w2, weight=(0.5, 0.5)):
        """
        Cosine of combined left+right context fingerprints.
        weight = (left_weight, right_weight).
        """
        lw, rw = weight
        scores = []
        if w1 in self._left_ctx and w2 in self._left_ctx:
            scores.append(lw * _cos_c(self._left_ctx[w1],
                                       self._left_ctx[w2], self.d))
        if w1 in self._right_ctx and w2 in self._right_ctx:
            scores.append(rw * _cos_c(self._right_ctx[w1],
                                       self._right_ctx[w2], self.d))
        if not scores:
            return 0.0
        return float(sum(scores) / max(1, len(scores) * max(lw, rw)))

    def pos_neighbors(self, word, k=5):
        """
        Return k words with most similar grammatical role.
        """
        if word not in self._left_ctx and word not in self._right_ctx:
            return []
        candidates = set(self._left_ctx.keys()) | set(self._right_ctx.keys())
        candidates.discard(word)
        candidates.discard(self.SENT_BOS)
        candidates.discard(self.SENT_EOS)
        scores = [(w, self.pos_similarity(word, w)) for w in candidates]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    # ── bigram statistics ─────────────────────────────────────────────────

    def bigram_count(self, prev, curr):
        return self._bigram.get((prev, curr), 0)

    def bigram_prob(self, prev, curr):
        n_prev = sum(c for (p, _), c in self._bigram.items() if p == prev)
        if n_prev == 0:
            return 0.0
        return self.bigram_count(prev, curr) / n_prev

    def surprise(self, prev, curr):
        """Bits of surprise: -log2 P(curr | prev)."""
        p = self.bigram_prob(prev, curr)
        if p <= 0:
            return 30.0   # cap for unseen
        return -math.log2(p)

    def is_phrase_boundary(self, prev, curr, threshold=5.0):
        """High-surprise boundary = likely end of phrase / start of new constituent."""
        return self.surprise(prev, curr) >= threshold

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def vocab_size(self):
        return len(self._word_count)

    def most_common_words(self, k=10):
        return self._word_count.most_common(k)

    def context_size(self, word):
        return (self._n_left.get(word, 0), self._n_right.get(word, 0))

    def left_context(self, word):
        return self._left_ctx.get(word)

    def right_context(self, word):
        return self._right_ctx.get(word)
