"""
ikigai.cognition.ngram_expander -- N-gram Online Response Expander.

Day 55 Pack 39 -- expands 3-word geometric seed into a fluent phrase.

Problem: geometric decoder gives top-k words (e.g. ['neural', 'network', 'vector']).
         These are semantically correct but syntactically flat.

Fix: train an n-gram model online from user speech. On response, take the
     geometric seed words as prefix and continue with argmax n-gram prediction.

Training:
    train_on_turn([w1, w2, w3, ...]) -> updates n-gram counts (online, no forgetting)

Expansion:
    expand(['neural', 'network'], max_len=8)
        -> step 1: bigram (neural, network) -> argmax next = 'architecture'
        -> step 2: bigram (network, architecture) -> argmax next = 'system'
        -> result: ['neural', 'network', 'architecture', 'system']

N chosen at init (default 3 = trigram model, bigram context window).
Fallback: if no n-gram known for current prefix, stop expansion.
Anti-repeat: skip next word if it appeared in last 2 positions of result.

Monotone: counts only increment. Zero forgetting. Online, single-pass.
"""


class NGramExpander:
    """
    Online n-gram model + geometric seed expander.

    n=3 (default): uses bigram (2-word) prefix to predict next word.
    counts[prefix] = {next_word: count}  -- all monotone increments.
    """

    def __init__(self, n=3, max_expand=10):
        self.n = n
        self.max_expand = max_expand
        self._counts = {}        # tuple(prefix) -> {next_word: count}
        self._unigram = {}       # word -> total count
        self._total_tokens = 0

    # ─── training ────────────────────────────────────────────────

    def train_on_turn(self, tokens):
        """Online n-gram update. Monotone: counts never decrease."""
        for t in tokens:
            self._unigram[t] = self._unigram.get(t, 0) + 1
        self._total_tokens += len(tokens)

        if len(tokens) < self.n:
            return

        context_size = self.n - 1
        for i in range(len(tokens) - context_size):
            prefix = tuple(tokens[i: i + context_size])
            next_w = tokens[i + context_size]
            if prefix not in self._counts:
                self._counts[prefix] = {}
            self._counts[prefix][next_w] = self._counts[prefix].get(next_w, 0) + 1

    # ─── expansion ───────────────────────────────────────────────

    def expand(self, seed_words, max_len=None):
        """
        Continue seed_words via argmax n-gram prediction.
        Stops when: no known continuation, or max_len reached, or repetition detected.
        Returns extended word list.
        """
        limit = max_len if max_len is not None else self.max_expand
        result = list(seed_words)
        ctx = self.n - 1

        for _ in range(limit - len(result)):
            if len(result) < ctx:
                break
            prefix = tuple(result[-ctx:])
            if prefix not in self._counts:
                break
            # argmax next word
            best = max(self._counts[prefix], key=self._counts[prefix].get)
            # anti-repeat: skip if word appeared in last 3 positions
            if best in result[-3:]:
                # try second-best
                candidates = sorted(self._counts[prefix].items(), key=lambda x: -x[1])
                chosen = None
                for w, _ in candidates[1:4]:
                    if w not in result[-3:]:
                        chosen = w
                        break
                if chosen is None:
                    break
                best = chosen
            result.append(best)

        return result

    # ─── fallback unigram expansion ──────────────────────────────

    def top_unigrams(self, k=5, exclude=None):
        """Top-k unigram words, optionally excluding a set."""
        excl = set(exclude or [])
        ranked = sorted(
            ((w, c) for w, c in self._unigram.items() if w not in excl),
            key=lambda x: -x[1]
        )
        return [w for w, _ in ranked[:k]]

    # ─── stats ───────────────────────────────────────────────────

    @property
    def vocab_size(self):
        return len(self._unigram)

    @property
    def ngram_entries(self):
        return sum(len(v) for v in self._counts.values())

    @property
    def total_tokens(self):
        return self._total_tokens

    def reset(self):
        self._counts.clear()
        self._unigram.clear()
        self._total_tokens = 0
