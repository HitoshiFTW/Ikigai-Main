"""ikigai.cognition.frame_induction -- Day 92.  NO TEMPLATES.  Induce the frame inventory
from raw sentences and REINFORCE it with more data, so generation needs no hand-authored
scaffold.

The organism generator (realize_message) fills a FRAME (slot types) by meaning.  Until now the
frame was hand-given -- a template.  Prince: kill the template; induce it.  Two unsupervised,
data-derived steps (Harris' distributional hypothesis -- structure from context, nothing
authored):

  TYPE INDUCTION -- a word's syntactic type is its DISTRIBUTIONAL context: which words come
  just before and just after it.  Build a context signature per word (left-neighbour counts ++
  right-neighbour counts over the vocab), cluster the signatures (k-means) -> word -> type.
  Determiners cluster because they precede nouns; verbs cluster because nouns precede them; no
  POS list is authored.  MORE sentences -> sharper signatures -> cleaner clusters (reinforcement).

  FRAME INDUCTION -- map every real sentence to its induced TYPE sequence; the frequent
  type-sequences ARE the frame inventory (the grammar, learned).  A type's word set is the
  induced type-lexicon.

Generation then: pick a frequent induced frame, fill each slot by MEANING (realize_message) with
the induced type-lexicon.  Novel sentence, induced structure, zero template.  All derived from
data; reinforced by more of it.
"""
import numpy as np
from collections import Counter, defaultdict


class FrameInducer:
    def __init__(self, seed=92):
        self.seed = int(seed)
        self._vocab = []
        self._idx = {}
        self._L = defaultdict(Counter)      # word -> Counter(left neighbour)
        self._R = defaultdict(Counter)      # word -> Counter(right neighbour)
        self._count = Counter()
        self.word2type = {}
        self.frames = Counter()             # type-tuple -> count
        self._types = {}                    # type -> [words]
        self.n_sentences = 0

    def _id(self, w):
        i = self._idx.get(w)
        if i is None:
            i = len(self._vocab); self._idx[w] = i; self._vocab.append(w)
        return i

    def observe(self, sentences):
        """Accumulate left/right neighbour statistics (reinforcement: call repeatedly)."""
        for s in sentences:
            s = [str(t) for t in s]
            for w in s:
                self._id(w); self._count[w] += 1
            for i, w in enumerate(s):
                l = s[i - 1] if i > 0 else '<s>'
                r = s[i + 1] if i < len(s) - 1 else '</s>'
                self._L[w][l] += 1
                self._R[w][r] += 1
            self.n_sentences += 1
        return self.n_sentences

    def _context_matrix(self):
        V = len(self._vocab)
        bound = {'<s>': V, '</s>': V + 1}
        M = np.zeros((V, 2 * (V + 2)), dtype=np.float32)
        for w, i in self._idx.items():
            for nb, c in self._L[w].items():
                j = bound.get(nb, self._idx.get(nb))
                if j is not None:
                    M[i, j] += c
            for nb, c in self._R[w].items():
                j = bound.get(nb, self._idx.get(nb))
                if j is not None:
                    M[i, (V + 2) + j] += c
        # row-normalise each half (left dist, right dist) so frequency doesn't dominate type
        left, right = M[:, :V + 2], M[:, V + 2:]
        left /= (left.sum(1, keepdims=True) + 1e-9)
        right /= (right.sum(1, keepdims=True) + 1e-9)
        return M

    def induce_types(self, k=6, iters=50):
        """Cluster words by distributional context (k-means) -> word2type.  Unsupervised POS."""
        M = self._context_matrix()
        V = M.shape[0]
        k = min(k, V)
        rng = np.random.RandomState(self.seed)
        # k-means++ init
        cent = [M[rng.randint(V)]]
        for _ in range(1, k):
            d = np.min([((M - c) ** 2).sum(1) for c in cent], axis=0)
            p = d / (d.sum() + 1e-9)
            cent.append(M[rng.choice(V, p=p)])
        C = np.stack(cent)
        assign = np.zeros(V, dtype=int)
        for _ in range(iters):
            d = ((M[:, None, :] - C[None, :, :]) ** 2).sum(2)
            new = d.argmin(1)
            if (new == assign).all():
                break
            assign = new
            for j in range(k):
                m = M[assign == j]
                if len(m):
                    C[j] = m.mean(0)
        self.word2type = {self._vocab[i]: f'T{int(assign[i])}' for i in range(V)}
        self._types = defaultdict(list)
        for w, t in self.word2type.items():
            self._types[t].append(w)
        self._types = dict(self._types)
        return self.word2type

    def induce_frames(self, sentences, min_count=2, top=None):
        """Map sentences to induced TYPE sequences; frequent ones = the frame inventory."""
        self.frames = Counter()
        for s in sentences:
            seq = tuple(self.word2type.get(str(w)) for w in s)
            if all(seq):
                self.frames[seq] += 1
        inv = [(f, c) for f, c in self.frames.items() if c >= min_count]
        inv.sort(key=lambda x: -x[1])
        if top:
            inv = inv[:top]
        self.frame_inventory = [f for f, _ in inv]
        self.frame_weights = np.array([c for _, c in inv], dtype=float)
        return self.frame_inventory

    def type_lexicon(self):
        return dict(self._types)

    def pick_frame(self, rng):
        """Weighted pick of an induced frame (rng = a python random.Random)."""
        if not getattr(self, 'frame_inventory', None):
            return None
        tot = float(self.frame_weights.sum())
        r = rng.random() * tot
        acc = 0.0
        for f, w in zip(self.frame_inventory, self.frame_weights):
            acc += w
            if acc >= r:
                return f
        return self.frame_inventory[-1]

    def type_purity(self, gold):
        """Cluster purity vs a gold word->type map (majority-gold per induced cluster)."""
        by = defaultdict(list)
        for w, t in self.word2type.items():
            if w in gold:
                by[t].append(gold[w])
        correct = total = 0
        for t, gs in by.items():
            if gs:
                correct += Counter(gs).most_common(1)[0][1]
                total += len(gs)
        return correct / max(1, total)
