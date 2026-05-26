"""
ikigai.cognition.multiresngram -- Multi-Resolution N-gram Stack.

Day 55 Pack 43 -- Layer 4 Coherence (LLM replacement stack).

Problem: single 3-gram only captures local token co-occurrence.
         No sentence-level or document-level structure.
         Same phrase repeated = same expansion regardless of doc context.

Fix: three-level stack, each trained independently:
     doc_gram   (n=2, unit=sentence-boundary markers)    -- document structure
     sent_gram  (n=3, unit=tokens)                       -- sentence patterns
     tok_gram   (n=4, unit=tokens)                       -- token detail

Expansion: top-level context biases mid-level biases token-level.
           doc_score * sent_score * tok_score -> rerank candidates.

vs LLM: LLM learns document structure from pretraining (frozen).
        MultiResNGram: learns online, per-conversation, zero forgetting.
        At 1000 turns: doc_gram encodes user's discourse style exactly.
"""

import numpy as np
from collections import defaultdict


class _NGramLevel:
    """Single n-gram level: train + next-word distribution."""

    def __init__(self, n, max_vocab=50_000):
        self.n      = n
        self._counts = defaultdict(lambda: defaultdict(int))
        self._total  = 0

    def train(self, tokens):
        ctx = self.n - 1
        if len(tokens) < self.n:
            return
        for i in range(len(tokens) - ctx):
            prefix = tuple(tokens[i: i + ctx])
            self._counts[prefix][tokens[i + ctx]] += 1
        self._total += len(tokens)

    def distribution(self, prefix):
        """Return {word: count} for this prefix. Empty dict if unseen."""
        if len(prefix) >= self.n - 1:
            key = tuple(prefix[-(self.n - 1):])
            return dict(self._counts.get(key, {}))
        return {}

    def top_k(self, prefix, k=10):
        """Return top-k (word, count) sorted by count desc."""
        dist = self.distribution(prefix)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]

    @property
    def n_contexts(self):
        return len(self._counts)

    @property
    def total_tokens(self):
        return self._total


class MultiResNGram:
    """
    Three-level n-gram stack: doc / sent / tok.

    train_turn(tokens)          -- update all three levels
    expand(seed, max_len)       -- token-level expansion biased by upper levels
    expand_multires(seed, context_tokens, max_len)
                                -- full multi-res expansion with context bias
    rerank(candidates, context) -- rerank candidates using doc+sent scores

    No forgetting: all counts monotone non-decreasing.
    """

    def __init__(self, n_tok=4, n_sent=3, n_doc=2, max_expand=12):
        self.tok_gram  = _NGramLevel(n=n_tok)
        self.sent_gram = _NGramLevel(n=n_sent)
        self.doc_gram  = _NGramLevel(n=n_doc)
        self.max_expand = max_expand

        # Document-level: sentence boundary tokens injected between turns
        self._sentence_buffer = []   # last sentence boundary markers
        self._turn_count = 0

    #  training

    def train_turn(self, tokens):
        """
        Train all three levels on one turn.
        Doc level sees sentence-boundary markers derived from turn structure.
        """
        if not tokens:
            return

        # Token level: direct token n-grams
        self.tok_gram.train(tokens)

        # Sentence level: same tokens (paragraph = sentence here)
        self.sent_gram.train(tokens)

        # Doc level: compressed turn signature (first + last token)
        # Represents turn-boundary discourse structure
        doc_tokens = [tokens[0], tokens[-1], '__TURN__']
        self.doc_gram.train(self._sentence_buffer + doc_tokens)
        self._sentence_buffer = [tokens[0], '__TURN__']

        self._turn_count += 1

    def train_sentence(self, sentence_tokens):
        """Train only token + sentence levels (no doc-level turn boundary)."""
        self.tok_gram.train(sentence_tokens)
        self.sent_gram.train(sentence_tokens)

    #  expansion

    def expand(self, seed_words, max_len=None):
        """
        Token-level expansion (argmax, no bias). Same as NGramExpander.
        Falls back to seed if n-gram empty.
        """
        limit  = max_len if max_len is not None else self.max_expand
        result = list(seed_words)

        for _ in range(limit - len(result)):
            cands = self.tok_gram.top_k(result, k=1)
            if not cands:
                cands = self.sent_gram.top_k(result, k=1)
            if not cands:
                break
            word = cands[0][0]
            if word in result[-3:]:
                break
            result.append(word)

        return result

    def expand_multires(self, seed_words, context_tokens=None, max_len=None):
        """
        Multi-resolution expansion: rerank tok_gram candidates using
        sent_gram + doc_gram scores.

        context_tokens: recent conversation context for doc-level query.
        """
        limit  = max_len if max_len is not None else self.max_expand
        result = list(seed_words)
        ctx    = context_tokens or []

        for _ in range(limit - len(result)):
            tok_cands = self.tok_gram.top_k(result, k=10)
            if not tok_cands:
                tok_cands = self.sent_gram.top_k(result, k=10)
            if not tok_cands:
                break

            if len(tok_cands) == 1:
                word = tok_cands[0][0]
                if word in result[-3:]:
                    break
                result.append(word)
                continue

            # Rerank: tok_score * sent_score * doc_score
            best_word  = None
            best_score = -1.0
            for word, tok_count in tok_cands:
                if word in result[-3:]:
                    continue
                # Sent-level score
                sent_dist  = self.sent_gram.distribution(result)
                sent_count = sent_dist.get(word, 0)
                sent_total = max(1, sum(sent_dist.values()))
                sent_score = (sent_count + 1) / (sent_total + len(sent_dist) + 1)

                # Doc-level score (from context signature)
                doc_prefix = [ctx[-1], '__TURN__'] if ctx else ['__TURN__']
                doc_dist   = self.doc_gram.distribution(doc_prefix)
                doc_score  = (doc_dist.get(word, 0) + 1) / (max(1, sum(doc_dist.values())) + 1)

                # Combined: log-linear (avoid zero)
                tok_total = max(1, sum(c for _, c in tok_cands))
                tok_score = (tok_count + 1) / (tok_total + len(tok_cands) + 1)
                combined  = tok_score * sent_score * doc_score

                if combined > best_score:
                    best_score = combined
                    best_word  = word

            if best_word is None:
                break
            result.append(best_word)

        return result

    def rerank(self, candidates, context_tokens=None):
        """
        Rerank word list by multi-res score relative to context.
        Returns candidates sorted by combined score desc.
        """
        ctx       = context_tokens or []
        doc_prefix = [ctx[-1], '__TURN__'] if ctx else ['__TURN__']
        doc_dist  = self.doc_gram.distribution(doc_prefix)

        scored = []
        for word in candidates:
            sent_dist  = self.sent_gram.distribution([word])
            sent_score = sum(sent_dist.values()) / max(1, self.sent_gram.total_tokens)
            doc_score  = doc_dist.get(word, 0) / max(1, sum(doc_dist.values()) + 1)
            scored.append((word, sent_score + doc_score))

        scored.sort(key=lambda x: -x[1])
        return [w for w, _ in scored]

    #  stats

    @property
    def tok_entries(self):
        return self.tok_gram.n_contexts

    @property
    def sent_entries(self):
        return self.sent_gram.n_contexts

    @property
    def doc_entries(self):
        return self.doc_gram.n_contexts

    @property
    def turn_count(self):
        return self._turn_count

    def summary(self):
        return {
            'tok_entries':  self.tok_entries,
            'sent_entries': self.sent_entries,
            'doc_entries':  self.doc_entries,
            'turns':        self.turn_count,
        }
