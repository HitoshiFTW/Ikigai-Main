"""CoherentGenerator -- open-ended, grounded, COHERENT sentence generation on the VSA substrate.

Day-106 (2026-07-24) generation program. Proven on real Tatoeba text: 100% grounded-coherent,
100% novel, complete sentences (`experiments/nl/day106f_ceiling.py`), where a plain order-1
rollout gives 9%.

The mechanism (no backprop, CPU, nJ-scale):
  * PROPOSE, grounded: the next token is drawn only from tokens ACTUALLY OBSERVED after the current
    context (order-2 with order-1 backoff) -> no fabricated transitions.
  * SELECT, global: those grounded candidates are RE-RANKED by a COMPOSED WHOLE-PREFIX STATE --
    recency-positional binding of emergent TYPE signatures + a global topic bundle -- scored by
    attention over the training positions. This carries coherence the local window cannot; it is
    what lifts 9% -> ~92% before any sentence-level critic.
  * (optional) a sentence-level shape critic picks the most coherent NOVEL candidate from a small
    pool ("safest novel") -> ~100% coherent AND novel, with diversity.

This is grounded COMPOSITIONAL generation: novel sentences stitched from learned pieces, globally
re-ranked. It is fluent, not fact-checked -- distinct from the organism's correct-or-abstain
`answer`/`speak` path, which states only derived facts. Use this to GENERATE; use those to ANSWER.
"""
import numpy as np
from collections import defaultdict, Counter

S, E = '<s>', '</s>'


class CoherentGenerator:
    def __init__(self, d=512, W=4, alpha=0.5, beta=12.0, max_keys=12000, manifold=1500, seed=114):
        self.d = int(d); self.W = int(W); self.alpha = float(alpha); self.beta = float(beta)
        self.max_keys = int(max_keys); self.manifold_n = int(manifold)
        self.seed = int(seed)
        self._fitted = False

    # ---- fit from example token-sequences (no backprop) ----
    def fit(self, sequences):
        rs = np.random.RandomState(self.seed)
        seqs = [[S] + [str(t) for t in s] + [E] for s in sequences if len(s) >= 2]
        if not seqs:
            self._fitted = False
            return self
        self.vocab = sorted(set(t for s in seqs for t in s))
        self.vidx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)
        # random atoms + emergent distributional TYPE signatures (neighbour bundle)
        atom = rs.randn(n, self.d).astype(np.float32); atom /= (np.linalg.norm(atom, axis=1, keepdims=True) + 1e-9)
        sig = np.zeros((n, self.d), np.float32)
        for s in seqs:
            for i, t in enumerate(s):
                if i > 0: sig[self.vidx[t]] += atom[self.vidx[s[i - 1]]]
                if i < len(s) - 1: sig[self.vidx[t]] += atom[self.vidx[s[i + 1]]]
        sig /= (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-9)
        self.sig = sig
        # grounded successor index (attested transitions only)
        self.succ2 = defaultdict(set); self.succ1 = defaultdict(set)
        for s in seqs:
            for i in range(len(s) - 1):
                self.succ1[s[i]].add(s[i + 1])
                if i >= 1: self.succ2[(s[i - 1], s[i])].add(s[i + 1])
        # position roles + composed-prefix attention memory (state -> next)
        self.R = rs.choice([-1.0, 1.0], size=(self.W, self.d)).astype(np.float32)
        self.POS = rs.choice([-1.0, 1.0], size=(14, self.d)).astype(np.float32)
        K, V = [], []
        for s in seqs:
            for t in range(len(s) - 1):
                K.append(self._state(s[:t + 1])); V.append(self.vidx[s[t + 1]])
        K = np.stack(K).astype(np.float32); V = np.array(V)
        if len(K) > self.max_keys:
            keep = rs.choice(len(K), size=self.max_keys, replace=False); K = K[keep]; V = V[keep]
        self.K = K; self.V = V
        # sentence-shape manifold for the coherence critic
        man = [self._shape(s) for s in (seqs if len(seqs) <= self.manifold_n
                                        else [seqs[i] for i in rs.choice(len(seqs), self.manifold_n, replace=False)])]
        self.MAN = np.stack(man).astype(np.float32)
        self.trained = set(tuple(s[1:-1]) for s in seqs)
        self._fitted = True
        return self

    def _sig_of(self, tok):
        i = self.vidx.get(tok)
        return self.sig[i] if i is not None else None

    def _state(self, prefix):
        v = np.zeros(self.d, np.float32)
        for j in range(min(self.W, len(prefix))):
            sg = self._sig_of(prefix[-1 - j])
            if sg is not None: v += self.R[j] * sg
        g = np.zeros(self.d, np.float32)
        for t in prefix:
            sg = self._sig_of(t)
            if sg is not None: g += sg
        ng = np.linalg.norm(g)
        if ng > 0: v += self.alpha * (g / ng)
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else v

    def _shape(self, seq):
        body = [t for t in seq if t != S]
        v = np.zeros(self.d, np.float32)
        for i, t in enumerate(body[:14]):
            sg = self._sig_of(t)
            if sg is not None: v += self.POS[i] * sg
        nv = np.linalg.norm(v)
        return v / nv if nv > 0 else v

    def _hist_weight(self, prefix):
        q = self._state(prefix)
        sc = self.K @ q; sc -= sc.max()
        w = np.exp(self.beta * sc)
        dist = np.zeros(len(self.vocab), np.float32)
        np.add.at(dist, self.V, w)
        return dist

    def coherence(self, seq):
        """resonance of a sentence's type-shape with the taught manifold (a coherence score)."""
        return float((self.MAN @ self._shape(seq)).max()) if self._fitted else 0.0

    def _rollout(self, rs, max_steps=14):
        prefix = [S]
        for _ in range(max_steps):
            ctx = tuple(prefix[-2:]) if len(prefix) >= 2 else None
            cand = self.succ2.get(ctx) if ctx else None
            if not cand: cand = self.succ1.get(prefix[-1])
            if not cand: break
            cand = list(cand)
            dist = self._hist_weight(prefix)                 # composed-prefix re-rank
            wv = np.array([dist[self.vidx[c]] for c in cand], np.float64)
            if wv.sum() <= 0:
                nxt = cand[rs.randint(len(cand))]
            else:
                nxt = cand[int(rs.choice(len(cand), p=wv / wv.sum()))]
            prefix.append(nxt)
            if nxt == E: break
        return prefix

    def generate(self, n=1, pool=8, novel=True, max_steps=14, seed=None):
        """Generate `n` coherent sentences. For each: propose a small POOL of grounded,
        composed-re-ranked rollouts, then pick the highest-coherence NOVEL one (safest-novel).
        Returns list of strings (or one string if n==1)."""
        if not self._fitted:
            return [] if n != 1 else ''
        rs = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        out = []
        for _ in range(n):
            cands = [self._rollout(rs, max_steps) for _ in range(max(1, pool))]
            scored = sorted(cands, key=lambda c: -self.coherence(c))
            pick = None
            if novel:
                for c in scored:
                    if tuple(c[1:-1] if c[-1] == E else c[1:]) not in self.trained:
                        pick = c; break
            pick = pick or scored[0]
            out.append(' '.join(t for t in pick if t not in (S, E)))
        return out[0] if n == 1 else out
