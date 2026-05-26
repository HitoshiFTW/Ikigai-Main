"""
ikigai.cognition.belief_expander -- Belief-Conditioned Phrase Scorer.

Day 55 Pack 41 -- replaces pure argmax n-gram with belief-steered selection.

Problem: NGramExpander picks argmax(count) next word. Ignores semantic intent.
         Same n-gram prefix always produces the same word regardless of topic.

Fix: at each expansion step, take top-k n-gram candidates, score each by
     cosine(phrase_hv(result + [candidate]), B_U), pick highest.

     phrase_hv = L2-normalize(sum(word_hvs))  same space as B_U.

Result: expansion path steered toward user's current belief direction.
        Different B_U values from the same prefix -> different continuations.

Invariant: falls back to argmax count when n-gram has only 1 candidate.
           B_U is never written to (read-only reference from BSPM).
"""

import numpy as np

from ikigai.cognition.ngram_expander import NGramExpander


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


class BeliefConditionedExpander(NGramExpander):
    """
    N-gram expander with belief-steered candidate scoring.

    expand_belief(seed, B_U) -> word list steered toward B_U direction.

    Scoring per step:
        for each candidate word w in top_k n-gram predictions:
            phrase_v = L2-normalize(sum(word_hv(w) for w in result + [w]))
            score    = cosine(phrase_v, B_U)
        pick w with highest score.

    vocab_hv_fn: callable word -> HV (must be same space as B_U).
    """

    def __init__(self, vocab_hv_fn, d, n=3, max_expand=10, top_candidates=5):
        super().__init__(n=n, max_expand=max_expand)
        self._hv   = vocab_hv_fn
        self.d     = d
        self.top_k = top_candidates

        # Stats
        self.belief_wins  = 0   # times belief choice != argmax choice
        self.total_choices = 0

    def _phrase_hv(self, words):
        """L2-normalized superposition of word HVs."""
        if not words:
            return np.zeros(self.d, dtype=np.float32)
        s = np.zeros(self.d, dtype=np.float32)
        for w in words:
            s = s + self._hv(w)
        return _l2_normalize(s)

    def expand_belief(self, seed_words, B_U, max_len=None):
        """
        Expand seed via belief-steered n-gram.

        Selection criterion: cosine(word_hv(candidate), B_U).
        O(d * top_k) per step -- independent of result length.
        Falls back to base expand() if n-gram is empty.
        """
        if not self._counts:
            return list(seed_words)

        limit  = max_len if max_len is not None else self.max_expand
        result = list(seed_words)
        ctx    = self.n - 1

        for _ in range(limit - len(result)):
            if len(result) < ctx:
                break
            prefix = tuple(result[-ctx:])
            if prefix not in self._counts:
                break

            # Top-k candidates from n-gram (by count)
            candidates = sorted(
                self._counts[prefix].items(), key=lambda x: -x[1]
            )[:self.top_k]

            if len(candidates) == 1:
                word = candidates[0][0]
                if word in result[-3:]:
                    break
                result.append(word)
                self.total_choices += 1
                continue

            argmax_word = candidates[0][0]

            # Belief-score: cosine(word_hv(candidate), B_U) -- O(d) per candidate
            best_word  = None
            best_score = -2.0
            for word, _ in candidates:
                if word in result[-3:]:
                    continue
                score = _cosine(self._hv(word), B_U)
                if score > best_score:
                    best_score = score
                    best_word  = word

            if best_word is None:
                break

            if best_word != argmax_word:
                self.belief_wins += 1
            self.total_choices += 1
            result.append(best_word)

        return result

    def belief_override_rate(self):
        """Fraction of steps where belief steered away from argmax count."""
        if self.total_choices == 0:
            return 0.0
        return self.belief_wins / self.total_choices

    def compare_expansions(self, seed, B_U, max_len=None):
        """
        Return (base_expansion, belief_expansion, cos_base, cos_belief).
        Useful for direct comparison.
        """
        base   = self.expand(seed, max_len=max_len)
        belief = self.expand_belief(seed, B_U, max_len=max_len)
        cos_base   = _cosine(self._phrase_hv(base), B_U)
        cos_belief = _cosine(self._phrase_hv(belief), B_U)
        return base, belief, cos_base, cos_belief
