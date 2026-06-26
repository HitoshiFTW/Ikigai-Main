"""
ikigai.cognition.hsp -- Hyper-dimensional Spatial Pooling.

Pack 247g (Day 71). Day 70 research-recommended primitive for continuous
semantic topology over the VSA substrate. Each vocab word maps to a
fixed k-active SDR over M columns. Similar substrate HVs share columns;
unrelated HVs share few. This gives EVERY word a class signal without
hand-seeded taxonomy.

Replaces the 6.6% coverage cap of Pack 247d hand-seeded SEED dict with
discovered topology covering 100% of vocab.

Algorithm (HTM Spatial Pooler adapted for complex-phasor VSA):

  1. M column prototypes W[i] in C^d, init random phasor.
  2. encode(hv): cosine(hv, W[i]) for all i. Top-k indices = active SDR.
  3. learn(hv, sdr): Hebbian winner-take-most.
       For i in sdr:  W[i] = renorm((1-lr) W[i] + lr * hv)
  4. fit_vocab(vocab, ck, n_epochs): repeat encode+learn per word.

Properties:
  - Every word gets a non-empty SDR (no coverage gap).
  - Similar words share more active columns (Jaccard > 0).
  - Backprop-free, online-learnable.
  - Storage: M x d complex64 = M * d * 8 bytes (M=256, d=400 -> 800 KB).
  - SDR per word: set of k ints, ~k * 8 bytes (k=20 -> 160 bytes / word).
"""

import numpy as np


def _renorm_phasor(hv):
    mag = np.abs(hv)
    mag = np.where(mag > 1e-9, mag, 1.0)
    return (hv / mag).astype(np.complex64)


class HSPColumnPooler:
    """Hyper-dimensional Spatial Pooler.

    Public API:
        hsp = HSPColumnPooler(d=400, M=256, k_active=20)
        hsp.fit_vocab(vocab_list, ck)
        sdr = hsp.encode_word('cat', ck)            -> set of col idx
        sim = hsp.jaccard('cat', 'dog')             -> [0, 1] topology score
        neighbors = hsp.nearest('cat', k=5)         -> top-k closest words
    """

    def __init__(self, d=400, M=256, k_active=20, lr=0.05, seed=42,
                  boost_strength=2.0, common_mode=True):
        """
        boost_strength: HTM boost factor exponent. Higher = more aggressive
            re-balancing of column activation rates. 0 = no boost.
        common_mode: subtract global mean from inputs before encode/learn.
            Required when inputs share a strong DC / discourse component
            (e.g., behavioral recall vectors).
        """
        self.d = int(d)
        self.M = int(M)
        self.k = int(k_active)
        self.lr = float(lr)
        self.boost_strength = float(boost_strength)
        self.common_mode = bool(common_mode)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(-np.pi, np.pi, (self.M, self.d)).astype(np.float32)
        self.W = np.exp(1j * ph).astype(np.complex64)
        self.word_sdr = {}   # word -> frozenset of int col idx
        self.col_words = [set() for _ in range(self.M)]  # col -> set of words
        self._activation = np.zeros(self.M, dtype=np.float32)  # cumulative
        self._common_hv = None
        self._fit_epochs = 0
        self._fit_n_words = 0
        # Pack 247g Phase 4-A: HTM Temporal Memory successor matrix.
        # P[i, j] = count(col j active at t+1 | col i active at t)
        self.P = np.zeros((self.M, self.M), dtype=np.float32)
        self._P_row_sums = np.zeros(self.M, dtype=np.float32)
        self._transitions_fit = 0

    # ── encode / learn ────────────────────────────────────────────
    def _prep_input(self, hv):
        """Common-mode removal + magnitude normalize."""
        h = hv
        if self.common_mode and self._common_hv is not None:
            h = h - self._common_hv
        h_mag = float(np.abs(h).mean()) + 1e-9
        return (h / h_mag).astype(np.complex64)

    def _cos_columns(self, h_n):
        """Cosine of prepped input against all M column prototypes."""
        W_mag = np.abs(self.W).mean(axis=1) + 1e-9
        W_n = self.W / W_mag[:, None]
        sims = np.real(W_n.conj() @ h_n).astype(np.float32) / self.d
        return sims

    def _column_boost(self):
        """HTM boost: columns with below-average activation get a multiplicative
        bonus, above-average get penalized. Returns (M,) float32 multiplier."""
        if self.boost_strength <= 0 or self._activation.sum() == 0:
            return np.ones(self.M, dtype=np.float32)
        target = self._activation.mean() + 1e-6
        # Boost = exp(beta * (target - actual)/target). Range > 0.
        ratio = (target - self._activation) / target
        boost = np.exp(self.boost_strength * ratio).astype(np.float32)
        return boost

    def encode(self, hv, apply_boost=True, update_activation=True):
        """Return (sdr_set, sims). Optionally apply column-balance boost."""
        h_n = self._prep_input(hv)
        sims = self._cos_columns(h_n)
        if apply_boost:
            sims = sims * self._column_boost()
        top = np.argpartition(-sims, self.k)[:self.k]
        top = top[np.argsort(-sims[top])]
        sdr = frozenset(int(i) for i in top)
        if update_activation:
            for i in sdr:
                self._activation[i] += 1.0
        return sdr, sims

    def learn(self, hv, sdr):
        """Hebbian winner-take-most. Active columns drift toward (prepped) input."""
        h_n = self._prep_input(hv)
        lr = self.lr
        for i in sdr:
            new = (1.0 - lr) * self.W[i] + lr * h_n
            self.W[i] = _renorm_phasor(new)

    # ── batch fit over vocab ──────────────────────────────────────
    def fit_vocab(self, vocab, ck, n_epochs=3, verbose=False):
        """Repeatedly encode+learn each vocab word for n_epochs.

        Step 0: compute common-mode HV (global mean) once if enabled.
        Steps 1..n_epochs: encode (with boost) -> learn (with input prep).
        Final pass: record word_sdr without boost / without activation
        update so SDRs reflect converged column geometry deterministically.
        """
        vocab = list(vocab)
        # Step 0: common-mode HV
        if self.common_mode:
            accum = np.zeros(self.d, dtype=np.complex64)
            for w in vocab:
                accum = accum + ck.key(w)
            self._common_hv = (accum / len(vocab)).astype(np.complex64)
            if verbose:
                cm_mag = float(np.abs(self._common_hv).mean())
                print(f'  HSP common-mode HV magnitude: {cm_mag:.4f}')

        for ep in range(n_epochs):
            for w in vocab:
                hv = ck.key(w)
                sdr, _ = self.encode(hv, apply_boost=True, update_activation=True)
                self.learn(hv, sdr)
            if verbose:
                act_std = float(self._activation.std())
                act_mean = float(self._activation.mean())
                print(f'  HSP epoch {ep+1}/{n_epochs}: activation mean={act_mean:.1f} '
                       f'std={act_std:.1f}')

        # Lock-in pass: deterministic SDR per word, no boost, no activation
        self.word_sdr.clear()
        for i in range(self.M):
            self.col_words[i].clear()
        for w in vocab:
            hv = ck.key(w)
            sdr, _ = self.encode(hv, apply_boost=False, update_activation=False)
            self.word_sdr[w] = sdr
            for i in sdr:
                self.col_words[i].add(w)
        self._fit_epochs = n_epochs
        self._fit_n_words = len(vocab)

    # ── queries ───────────────────────────────────────────────────
    def encode_word(self, word, ck):
        sdr = self.word_sdr.get(word)
        if sdr is not None:
            return sdr
        # Cold word: encode but do not store
        sdr, _ = self.encode(ck.key(word))
        return sdr

    def jaccard(self, w1, w2):
        s1 = self.word_sdr.get(w1)
        s2 = self.word_sdr.get(w2)
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    def nearest(self, word, k=5):
        """Return [(other_word, jaccard), ...] top-k by SDR overlap.
        Linear scan over word_sdr; fast for vocab <= 100K."""
        s1 = self.word_sdr.get(word)
        if not s1:
            return []
        out = []
        for w, s2 in self.word_sdr.items():
            if w == word: continue
            inter = len(s1 & s2)
            if inter == 0: continue
            union = len(s1 | s2)
            out.append((w, inter / union if union else 0.0))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    def coverage(self):
        """Fraction of words with non-empty SDR (should always be 1.0
        after fit_vocab)."""
        if not self.word_sdr: return 0.0
        nz = sum(1 for s in self.word_sdr.values() if s)
        return nz / len(self.word_sdr)

    def column_usage(self):
        """Stats over col_words: how many words per column."""
        sizes = [len(s) for s in self.col_words]
        return {
            'min': min(sizes), 'max': max(sizes),
            'mean': float(np.mean(sizes)),
            'median': float(np.median(sizes)),
            'n_empty': sum(1 for s in sizes if s == 0),
        }

    # ── Pack 247g Phase 4-A: HTM Temporal Memory successor matrix ───
    def fit_transitions(self, token_pairs, normalize='ppmi', alpha=0.75):
        """Build P[i, j] = count col j active at t+1 when col i active at t.

        Pack 247g Phase 4-B (Day 71 research-recommended): PPMI normalization
        with frequency-biased smoothing (alpha=0.75). Suppresses globally
        frequent successor columns (the april/august leakage).

        normalize:
          'ppmi'    -> PPMI[i,j] = max(0, log2(P[i,j] * N_alpha /
                                            (C_src[i] * C_tgt[j]^alpha)))
          'rowsum'  -> P[i,j] / sum_j P[i,j]   (Phase 4-A baseline)
          None / False -> raw counts
        alpha: smoothing exponent on target marginal. 0.75 = standard.
        """
        n_pairs = 0
        for cur, nxt in token_pairs:
            sdr_c = self.word_sdr.get(cur)
            sdr_n = self.word_sdr.get(nxt)
            if not sdr_c or not sdr_n:
                continue
            for i in sdr_c:
                for j in sdr_n:
                    self.P[i, j] += 1.0
            n_pairs += 1
        self._transitions_fit += n_pairs
        if normalize == 'ppmi':
            C_src = self.P.sum(axis=1).astype(np.float64)   # (M,)
            C_tgt = self.P.sum(axis=0).astype(np.float64)   # (M,)
            C_tgt_a = np.power(C_tgt, alpha)
            N_alpha = C_tgt_a.sum()
            denom = np.outer(C_src, C_tgt_a) + 1e-9
            ratio = (self.P.astype(np.float64) * N_alpha) / denom
            # PMI -> log2 of ratio (only where P>0 to avoid log(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                pmi = np.log2(np.where(ratio > 0, ratio, 1.0))
            ppmi = np.maximum(0.0, pmi)
            ppmi[self.P == 0] = 0.0  # mask non-observed pairs
            self.P = ppmi.astype(np.float32)
            self._P_row_sums = self.P.sum(axis=1).astype(np.float32)
        elif normalize == 'rowsum' or normalize is True:
            row_sums = self.P.sum(axis=1)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            self.P = (self.P / row_sums[:, None]).astype(np.float32)
            self._P_row_sums = row_sums.astype(np.float32)
        return n_pairs

    def expected_next_sdr(self, current_word, top_k=None):
        """Given current word, predict the SDR most likely to follow.
        Returns (sdr_set, scores) where sdr_set is top-k columns of
        the expected successor pattern.

        Math: scores[j] = sum_{i in sdr(current)} P[i, j].
        Then top-k columns of scores form the expected next-SDR.
        """
        if top_k is None: top_k = self.k
        sdr_c = self.word_sdr.get(current_word)
        if not sdr_c:
            return frozenset(), np.zeros(self.M, dtype=np.float32)
        idx_c = np.fromiter(sdr_c, dtype=np.int32)
        scores = self.P[idx_c].sum(axis=0).astype(np.float32)
        if scores.sum() <= 0:
            return frozenset(), scores
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
        return frozenset(int(i) for i in top), scores

    def status(self):
        return {
            'd': self.d, 'M': self.M, 'k_active': self.k, 'lr': self.lr,
            'fit_epochs': self._fit_epochs, 'fit_n_words': self._fit_n_words,
            'words_with_sdr': len(self.word_sdr),
            'coverage': self.coverage(),
            'col_usage': self.column_usage(),
            'transitions_fit': self._transitions_fit,
            'P_nonzero_rows': int((self.P.sum(axis=1) > 0).sum()),
            'storage_kb': self.W.nbytes / 1024.0,
            'P_storage_kb': self.P.nbytes / 1024.0,
        }
