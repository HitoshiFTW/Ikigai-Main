"""
ikigai.cognition.frame_field -- Pack 197 Resonance Frame Field.

THE missing primitive: context-conditioned VSA-SDM binding so the organism
can hold N languages and N domains in one substrate without cross-talk.

A human trilingual speaker doesn't blend French and English -- their whole
substrate enters a FRENCH FRAME when hearing French function-word density
(le/la/un/est). Pack 197 builds that mechanism in flat memory.

Design:
  - K frame anchors (phasor HVs in C^d) initialized random.
  - Per-passage assign: choose frame argmax of cosine to anchors + cosine
    to per-frame function-word fingerprint.
  - Per-passage update: leader-follower drift of chosen anchor toward
    passage HV (online k-means).
  - Bindings conditioned: addr = ck.key(word) * (frame_hv * role_hv).
    Back-compat: frame_hv = None -> identical to pre-Pack-197.

Composition with the rest of the substrate:
  - MultiRoleMemory.current_frame_hv = frame_hv before each write/read.
  - At absorb time: per-text compute frame, set, write, clear.
  - At cogitate time: passage frame from prompt, set, generate, clear.

Storage: K=8 x d=400 complex64 anchors = 25 KB. Plus top-N=64 fingerprint
centroids = 2 KB. Plus token frequency dict (transient, ~5 MB during absorb).
Substrate stays 192 MB.
"""

import numpy as np


def _renorm_phasor(hv):
    """Renormalize a complex HV so each component has unit magnitude (phasor)."""
    mag = np.abs(hv)
    mag = np.where(mag > 1e-9, mag, 1.0)
    return (hv / mag).astype(np.complex64)


def _bundle_phasor(hvs):
    """Bundle a list of phasor HVs into a normalized phasor."""
    if not hvs:
        return None
    accum = np.zeros_like(hvs[0], dtype=np.complex64)
    for hv in hvs:
        accum = accum + hv
    return _renorm_phasor(accum)


class FrameField:
    """Adaptive Resonance Frame Field with function-word fingerprinting.

    K frame anchors (phasor HVs) compete to claim each passage. The winner
    drifts toward the passage HV (leader-follower). Function-word
    fingerprint augments the routing signal so languages disambiguate even
    when their semantic HVs look similar.
    """

    def __init__(self, d=400, K=8, top_n=96, seed=42, alpha=0.2,
                 lock_after_unique_tokens=400, lr=0.05):
        """
        d: phasor dimension (must match MultiRoleMemory.d)
        K: number of frame attractors
        top_n: number of top-frequency tokens forming the fingerprint (Pack 199 -> 96)
        seed: RNG seed for initial anchor phases
        alpha: weight for semantic-HV vs fingerprint in assign() routing
               (Pack 199 -> 0.2 = fingerprint dominant, since fp IS the language signal)
        lock_after_unique_tokens: lock the top-N once this many unique tokens seen
        lr: leader-follower learning rate
        """
        self.d = int(d)
        self.K = int(K)
        self.top_n = int(top_n)
        self.alpha = float(alpha)
        self.lock_threshold = int(lock_after_unique_tokens)
        self.lr = float(lr)

        rng = np.random.default_rng(seed)
        # Pack 199 -- orthogonal phasor init via QR on complex Gaussian.
        # Random phases collide in geometry; QR-orthogonalized columns give
        # K mutually-distant anchors -> frames separate faster.
        A = (rng.normal(size=(d, K)) + 1j * rng.normal(size=(d, K))).astype(np.complex128)
        Q, _ = np.linalg.qr(A)            # Q: (d, K), orthonormal columns
        ortho = Q.T                         # (K, d)
        ph = np.angle(ortho).astype(np.float32)
        self.anchors = np.exp(1j * ph).astype(np.complex64)

        # function-word discovery
        self.token_freq = {}                    # token -> count
        self.top_tokens = []                    # locked top-N tokens
        self.top_idx = {}                       # token -> dim index
        self.locked = False

        # per-frame fingerprint centroid (online running mean)
        self.fp_centroids = np.zeros((self.K, self.top_n), dtype=np.float32)
        self.fp_counts = np.zeros(self.K, dtype=np.int64)
        self.fp_total_seen = np.zeros(self.K, dtype=np.float64)

        # assignment stats
        self.assigns_per_frame = np.zeros(self.K, dtype=np.int64)
        self.last_assigned = -1

        # Pack 198: per-frame vocab. Each frame remembers which tokens were
        # absorbed under it. Used by frame-scoped cogitate to keep generations
        # in-language.
        self.frame_vocab = [set() for _ in range(self.K)]
        # Pack 199 NEW1: grammar-state machine via frame transitions.
        # word -> most-recent-frame mapping + (frame_a, frame_b) bigram counts.
        self.word_to_frame = {}
        self.frame_bigram = np.zeros((self.K, self.K), dtype=np.int64)
        self._last_word_frame = -1

    # ── frequency / fingerprint ──────────────────────────────────────────
    def observe_tokens(self, tokens):
        """Bump frequency counter. Auto-locks once threshold passed."""
        for t in tokens:
            self.token_freq[t] = self.token_freq.get(t, 0) + 1
        if not self.locked and len(self.token_freq) >= self.lock_threshold:
            self.lock_anchors()

    def lock_anchors(self):
        """Pick top-N most-frequent tokens as fingerprint dimensions."""
        items = sorted(self.token_freq.items(), key=lambda kv: -kv[1])
        self.top_tokens = [t for t, _ in items[:self.top_n]]
        self.top_idx = {t: i for i, t in enumerate(self.top_tokens)}
        self.locked = True

    def fingerprint(self, tokens):
        """Histogram of top-N tokens in passage. L1-normalized.
        Returns None if not locked yet."""
        if not self.locked:
            return None
        fp = np.zeros(self.top_n, dtype=np.float32)
        for t in tokens:
            i = self.top_idx.get(t)
            if i is not None:
                fp[i] += 1
        s = fp.sum()
        if s > 0:
            fp = fp / s
        return fp

    # ── passage HV ───────────────────────────────────────────────────────
    def passage_hv(self, tokens, ck):
        """Bundled phasor HV of passage using a ComputedKey instance."""
        if not tokens:
            return None
        accum = np.zeros(self.d, dtype=np.complex64)
        for t in tokens:
            accum = accum + ck.key(t)
        return _renorm_phasor(accum)

    # ── frame routing ────────────────────────────────────────────────────
    def assign(self, p_hv, fp=None):
        """Return (frame_idx, score). Argmax of combined cosine."""
        # Semantic HV cosine (real part of inner product / d)
        sims_hv = np.real(self.anchors @ np.conj(p_hv)) / self.d
        sims_hv = sims_hv.astype(np.float32)
        if fp is not None and self.fp_counts.sum() > 0:
            cn = np.linalg.norm(self.fp_centroids, axis=1) + 1e-9
            fn = np.linalg.norm(fp) + 1e-9
            sims_fp = (self.fp_centroids @ fp) / (cn * fn)
            sims = self.alpha * sims_hv + (1.0 - self.alpha) * sims_fp
        else:
            sims = sims_hv
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def update(self, idx, p_hv, fp=None, lr=None, tokens=None):
        """Leader-follower drift toward the passage. Tokens populate per-frame
        vocab (Pack 198) so cogitate can stay in-language. Pack 199 NEW1 also
        builds frame-bigram transition counts for the grammar-state machine."""
        if lr is None:
            lr = self.lr
        new = (1.0 - lr) * self.anchors[idx] + lr * p_hv
        self.anchors[idx] = _renorm_phasor(new)
        if fp is not None:
            self.fp_centroids[idx] = (1.0 - lr) * self.fp_centroids[idx] + lr * fp
        self.fp_counts[idx] += 1
        self.fp_total_seen[idx] += 1.0
        self.assigns_per_frame[idx] += 1
        self.last_assigned = idx
        # Pack 198 + 199 NEW1: per-frame vocab + word->frame + frame transitions
        if tokens:
            self.frame_vocab[idx].update(tokens)
            prev_frame = -1
            for t in tokens:
                t_prev_frame = self.word_to_frame.get(t, -1)
                self.word_to_frame[t] = idx
                if prev_frame >= 0 and t_prev_frame >= 0:
                    self.frame_bigram[prev_frame, t_prev_frame] += 1
                prev_frame = t_prev_frame if t_prev_frame >= 0 else idx

    def get_anchor(self, idx):
        """Return frame HV for binding into MultiRoleMemory."""
        return self.anchors[int(idx)].copy()

    # ── one-shot per-passage routine ─────────────────────────────────────
    def route_passage(self, tokens, ck, observe=True, learn=True):
        """Full per-passage pipeline. Returns (frame_idx, frame_hv, score).
        If `observe`, updates frequency counter (locks if threshold hit).
        If `learn`, drifts anchor toward passage.
        Returns (None, None, 0.0) if passage HV not computable.
        """
        if not tokens:
            return None, None, 0.0
        if observe:
            self.observe_tokens(tokens)
        p_hv = self.passage_hv(tokens, ck)
        if p_hv is None:
            return None, None, 0.0
        fp = self.fingerprint(tokens) if self.locked else None
        idx, score = self.assign(p_hv, fp)
        if learn:
            self.update(idx, p_hv, fp, tokens=tokens)
        return idx, self.get_anchor(idx), score

    # Pack 199 NEW1: grammar-state queries
    def next_frame_probs(self, prev_frame):
        """Given a previous-token-frame, return softmax over likely next-frames."""
        if prev_frame < 0 or prev_frame >= self.K:
            return np.ones(self.K, dtype=np.float32) / self.K
        row = self.frame_bigram[prev_frame].astype(np.float32)
        s = row.sum()
        if s <= 0:
            return np.ones(self.K, dtype=np.float32) / self.K
        return row / s

    def frame_of_word(self, word):
        return self.word_to_frame.get(word, -1)

    # ── inspection ───────────────────────────────────────────────────────
    def status(self):
        return {
            'd': self.d, 'K': self.K, 'top_n': self.top_n,
            'locked': self.locked,
            'unique_tokens_seen': len(self.token_freq),
            'top_tokens': list(self.top_tokens[:16]),
            'assigns_per_frame': self.assigns_per_frame.tolist(),
            'last_assigned': self.last_assigned,
            'frame_bigram_total': int(self.frame_bigram.sum()),
        }

    # ── Pack 198: prompt-based frame routing for inference ───────────────
    def route_prompt(self, tokens, ck):
        """Inference-time route: assign frame from prompt, NO observe/learn.
        Returns (frame_idx, frame_hv, score). Used by cogitate to pick frame
        before generation without polluting the field with the prompt itself."""
        if not tokens:
            return None, None, 0.0
        p_hv = self.passage_hv(tokens, ck)
        if p_hv is None:
            return None, None, 0.0
        fp = self.fingerprint(tokens) if self.locked else None
        idx, score = self.assign(p_hv, fp)
        return idx, self.get_anchor(idx), score

    # ── persistence ──────────────────────────────────────────────────────
    def to_dict(self):
        """Return numpy-savez-friendly state dict."""
        tf_items = list(self.token_freq.items())
        # Pack 198: serialize per-frame vocab as parallel arrays
        fv_idx, fv_tok = [], []
        for k, vocab in enumerate(self.frame_vocab):
            for t in vocab:
                fv_idx.append(k); fv_tok.append(t)
        return {
            'frame_d': np.int32(self.d),
            'frame_K': np.int32(self.K),
            'frame_top_n': np.int32(self.top_n),
            'frame_alpha': np.float32(self.alpha),
            'frame_lr': np.float32(self.lr),
            'frame_anchors': self.anchors,
            'frame_fp_centroids': self.fp_centroids,
            'frame_fp_counts': self.fp_counts,
            'frame_fp_total': self.fp_total_seen,
            'frame_assigns': self.assigns_per_frame,
            'frame_locked': np.int32(1 if self.locked else 0),
            'frame_top_tokens': np.array(self.top_tokens, dtype=object),
            'frame_tf_keys': np.array([k for k, _ in tf_items], dtype=object),
            'frame_tf_vals': np.array([v for _, v in tf_items], dtype=np.int64),
            'frame_vocab_idx': np.array(fv_idx, dtype=np.int32),
            'frame_vocab_tok': np.array(fv_tok, dtype=object),
            # Pack 199 NEW1: grammar FSM state
            'frame_w2f_keys': np.array(list(self.word_to_frame.keys()), dtype=object),
            'frame_w2f_vals': np.array(list(self.word_to_frame.values()),
                                          dtype=np.int32),
            'frame_bigram': self.frame_bigram,
        }

    @classmethod
    def from_dict(cls, z):
        """Reconstruct from a numpy-savez archive's keys."""
        ff = cls(d=int(z['frame_d']), K=int(z['frame_K']),
                  top_n=int(z['frame_top_n']),
                  alpha=float(z['frame_alpha']),
                  lr=float(z['frame_lr']))
        ff.anchors = z['frame_anchors'].astype(np.complex64)
        ff.fp_centroids = z['frame_fp_centroids'].astype(np.float32)
        ff.fp_counts = z['frame_fp_counts'].astype(np.int64)
        ff.fp_total_seen = z['frame_fp_total'].astype(np.float64)
        ff.assigns_per_frame = z['frame_assigns'].astype(np.int64)
        ff.locked = bool(int(z['frame_locked']))
        ff.top_tokens = list(z['frame_top_tokens'].tolist())
        ff.top_idx = {t: i for i, t in enumerate(ff.top_tokens)}
        tf_keys = z['frame_tf_keys'].tolist()
        tf_vals = z['frame_tf_vals'].tolist()
        ff.token_freq = {k: int(v) for k, v in zip(tf_keys, tf_vals)}
        # Pack 198 frame_vocab restore. `z` may be a dict (pending state) or
        # a NpzFile -- handle both.
        keys = z.files if hasattr(z, 'files') else z.keys()
        if 'frame_vocab_idx' in keys and 'frame_vocab_tok' in keys:
            fv_idx = z['frame_vocab_idx'].tolist()
            fv_tok = z['frame_vocab_tok'].tolist()
            for k, t in zip(fv_idx, fv_tok):
                if 0 <= int(k) < ff.K:
                    ff.frame_vocab[int(k)].add(str(t))
        # Pack 199 NEW1 restore: grammar FSM
        if 'frame_w2f_keys' in keys and 'frame_w2f_vals' in keys:
            w_keys = z['frame_w2f_keys'].tolist()
            w_vals = z['frame_w2f_vals'].tolist()
            ff.word_to_frame = {str(k): int(v) for k, v in zip(w_keys, w_vals)}
        if 'frame_bigram' in keys:
            fb = z['frame_bigram']
            if fb.shape == (ff.K, ff.K):
                ff.frame_bigram = fb.astype(np.int64)
        return ff
