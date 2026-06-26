"""
ikigai.cognition.concept_synthesizer -- Self-defining concepts on flat memory.

The missing piece in the substrate: each word's meaning is currently scattered
across many roles (isa, property, affordance, episode, mod, cooccur, antonym).
Recall returns a slice. There is no UNIFIED concept HV per word.

ConceptSynthesizer iteratively condenses every channel's facts about each
word into ONE renormalized HV per word -- a stable fixed point that lives
in the substrate's address space and IS the word's meaning.

Algorithm (one pass over the vocabulary):

    new_concept[w] = renorm(
        key(w)                                  # token identity
      + sum over roles R of  gamma_R * recall(w, R)
      - gamma_ant * recall(w, 'antonym')        # pushed away from opposites
    )

Iterating to fixed point typically takes 5-10 passes; deltas between
iterations decrease monotonically. The resulting concept HVs:

  - cluster semantically (cat ~ dog > cat ~ apple)
  - support concept arithmetic (king - man + woman ~ queen-like word)
  - reveal emergent dimensions via PCA on the concept matrix
  - power generation: cogitate uses concept[prompt-word] as goal anchor
    instead of raw key(prompt-word)

The concept matrix occupies O(vocab * d) bytes during build, but only the
top-K concept neighbours need to be persisted. Concept HVs can be written
back to the substrate under the 'concept' role for permanent storage.
"""

import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class ConceptSynthesizer:
    """
    Build per-word concept HVs by iterative multi-channel condensation.

    Public API:
        cs = ConceptSynthesizer(mr)
        cs.build(words=None, iterations=8, verbose=False) -> list[float]
            iteratively refines concept HVs; returns list of mean deltas
            (convergence trace).
        cs.concept_of(word) -> complex64[d]
        cs.similarity(a, b) -> float in [-1, 1]
        cs.neighbors(word, top_k=10) -> list[(word, score)]
        cs.arithmetic(plus_words=[a,c], minus_words=[b], top_k=5) -> list
            concept arithmetic: A - B + C -> top-K closest concepts
        cs.write_to_substrate() -> int
            persist concept HVs under 'concept' role in the relational bank.
    """

    DEFAULT_WEIGHTS = {
        'isa':        1.0,
        'property':   0.8,
        'affordance': 0.7,
        'episode':    0.5,
        'mod':        0.4,
        'cooccur':    0.3,
        'next':       0.2,
    }
    ANTONYM_WEIGHT = -0.5    # antonym recall is subtracted

    # Pack 157: named presets discovered via gamma sweep on the 1000-noun
    # WordNet-overlay substrate.  Each preset trades off readout sharpness
    # for one axis of meaning.  Use:
    #   general       -- default; balanced across all channels
    #   analogy       -- property-dominant; cleanest king-man+woman style
    #                    arithmetic (Pack 157 winner: 78% top-3)
    #   categorical   -- isa-dominant; best for "what category is X?" queries
    #   distributional-- cooccur-dominant; word2vec-flavoured neighbours
    #   broad         -- all channels equally weighted; useful when you don't
    #                    know what you're querying for yet
    WEIGHT_PRESETS = {
        'general': {
            'isa': 1.0, 'property': 0.8, 'affordance': 0.7,
            'episode': 0.5, 'mod': 0.4, 'cooccur': 0.3, 'next': 0.2,
        },
        'analogy': {
            'isa': 0.1, 'property': 4.0, 'affordance': 0.0,
            'episode': 0.0, 'mod': 0.0, 'cooccur': 0.0, 'next': 0.0,
        },
        'categorical': {
            'isa': 4.0, 'property': 0.5, 'affordance': 0.0,
            'episode': 0.0, 'mod': 0.0, 'cooccur': 0.0, 'next': 0.0,
        },
        'distributional': {
            'isa': 0.0, 'property': 0.0, 'affordance': 0.0,
            'episode': 0.0, 'mod': 0.0, 'cooccur': 4.0, 'next': 1.0,
        },
        'broad': {
            'isa': 1.0, 'property': 1.0, 'affordance': 1.0,
            'episode': 1.0, 'mod': 1.0, 'cooccur': 1.0, 'next': 1.0,
        },
    }

    def __init__(self, mr, weights=None, antonym_weight=None, preset=None):
        self.mr = mr
        self.d = mr.d
        # Resolve weights: explicit weights override preset overrides default.
        if weights is not None:
            self.weights = dict(weights)
        elif preset is not None:
            if preset not in self.WEIGHT_PRESETS:
                raise ValueError(
                    f"preset {preset!r} not in {list(self.WEIGHT_PRESETS)}")
            self.weights = dict(self.WEIGHT_PRESETS[preset])
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)
        self.preset_name = preset
        self.antonym_weight = (antonym_weight if antonym_weight is not None
                                else self.ANTONYM_WEIGHT)
        self.concepts = {}        # word -> complex64[d]
        self._words = []

    # ── build the concept HVs ─────────────────────────────────────────────
    def build(self, words=None, iterations=8, verbose=False, tol=1e-4):
        """
        Iteratively refine concept HVs until convergence (or max iterations).
        Returns a list of mean per-iteration deltas (convergence trace).
        Stops early when delta < tol.
        """
        if words is None:
            words = list(self.mr._seen)
        else:
            words = list(words)
        self._words = words

        # init: concept[w] = key(w)
        for w in words:
            self.concepts[w] = self.mr.ck.key(w).copy()

        deltas = []
        for it in range(iterations):
            new_concepts = {}
            for w in words:
                accum = self.mr.ck.key(w).astype(np.complex64).copy()
                # positive roles
                for role, gamma in self.weights.items():
                    if role not in self.mr.roles: continue
                    if w not in self.mr._role_targets.get(role, set()):
                        continue
                    try:
                        hv = self.mr.recall(w, role)
                    except Exception:
                        continue
                    if hv is None: continue
                    accum = accum + (gamma * hv).astype(np.complex64)
                # antonym subtracted
                if ('antonym' in self.mr.roles
                        and w in self.mr._role_targets.get('antonym', set())):
                    try:
                        hv_a = self.mr.recall(w, 'antonym')
                        if hv_a is not None:
                            accum = accum + (self.antonym_weight * hv_a).astype(np.complex64)
                    except Exception:
                        pass
                new_concepts[w] = _renorm(accum)

            # convergence metric: mean L2 across all word HVs
            tot = 0.0
            for w in words:
                tot += float(np.linalg.norm(new_concepts[w] - self.concepts[w]))
            mean_delta = tot / max(1, len(words))
            deltas.append(mean_delta)
            self.concepts = new_concepts
            if verbose:
                print(f'    iter {it+1}: mean L2 delta = {mean_delta:.5f}')
            if mean_delta < tol:
                if verbose:
                    print(f'    converged at iter {it+1}')
                break
        return deltas

    # ── query methods ─────────────────────────────────────────────────────
    def concept_of(self, word):
        return self.concepts.get(word)

    def similarity(self, a, b):
        ca = self.concept_of(a); cb = self.concept_of(b)
        if ca is None or cb is None: return 0.0
        return float(np.real(np.vdot(ca, cb))) / self.d

    def neighbors(self, word, top_k=10, exclude_self=True):
        c = self.concept_of(word)
        if c is None: return []
        sims = []
        for w, cv in self.concepts.items():
            if exclude_self and w == word: continue
            s = float(np.real(np.vdot(c, cv))) / self.d
            sims.append((w, s))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def arithmetic(self, plus_words=None, minus_words=None, top_k=5):
        """
        Concept arithmetic. Returns top-K nearest concepts to:
            sum(concept[w] for w in plus_words) - sum(concept[w] for w in minus_words)
        Excludes any of the input words from results.
        """
        plus_words = list(plus_words or [])
        minus_words = list(minus_words or [])
        accum = np.zeros(self.d, dtype=np.complex64)
        for w in plus_words:
            c = self.concept_of(w)
            if c is not None:
                accum = accum + c
        for w in minus_words:
            c = self.concept_of(w)
            if c is not None:
                accum = accum - c
        if not np.any(accum):
            return []
        target = _renorm(accum)
        seen_inputs = set(plus_words) | set(minus_words)
        sims = []
        for w, cv in self.concepts.items():
            if w in seen_inputs: continue
            s = float(np.real(np.vdot(target, cv))) / self.d
            sims.append((w, s))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    # ── persist to substrate ──────────────────────────────────────────────
    def write_to_substrate(self):
        """Write all concept HVs to relational bank under 'concept' role."""
        if 'concept' not in self.mr.roles:
            rng = np.random.default_rng(14801)
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            self.mr.roles['concept'] = np.exp(1j * ph).astype(np.complex64)
        bank = self.mr.sdm_rel
        rolev = self.mr.roles['concept']
        n = 0
        for w, c in self.concepts.items():
            addr = (self.mr.ck.key(w) * rolev).astype(np.complex64)
            bank.write(addr, c)
            self.mr._role_targets.setdefault('concept', set()).add(w)
            n += 1
        return n

    # ── Pack 224 -- Resonator-based arithmetic (Frady & Kent 2020) ────────
    def arithmetic_resonator(self, plus_words=None, minus_words=None, top_k=5,
                              n_iters=10, beta=8.0, momentum=0.5,
                              belief_field=True):
        """Same as arithmetic() but uses iterative continuous Hopfield
        decoding instead of single-pass cosine. Each iter softmax-weights
        all concepts by similarity to current target and pulls target
        toward the weighted cluster. Bypasses 1/sqrt(K) cosine ceiling.

        plus_words / minus_words: same as arithmetic().
        n_iters: 0 == fall back to plain arithmetic().
        """
        plus_words = list(plus_words or [])
        minus_words = list(minus_words or [])
        accum = np.zeros(self.d, dtype=np.complex64)
        for w in plus_words:
            c = self.concept_of(w)
            if c is not None: accum = accum + c
        for w in minus_words:
            c = self.concept_of(w)
            if c is not None: accum = accum - c
        if not np.any(accum):
            return []
        target = _renorm(accum)
        seen_inputs = set(plus_words) | set(minus_words)
        cand_words = [w for w in self.concepts.keys() if w not in seen_inputs]
        if not cand_words:
            return []
        K = np.stack([self.concepts[w] for w in cand_words])   # (N, d)
        r = target.astype(np.complex64)
        # Resonator iters operate on the CONCEPT codebook, not key codebook.
        for _ in range(int(n_iters)):
            sims = np.real(K @ np.conj(r)) / self.d
            logits = beta * sims
            logits -= logits.max()
            w = np.exp(logits).astype(np.float32)
            w /= (w.sum() + 1e-12)
            r_new = (w[:, None] * K).sum(axis=0).astype(np.complex64)
            r = (momentum * r + (1.0 - momentum) * r_new).astype(np.complex64)
            mag = float(np.abs(r).mean())
            if mag > 1e-9:
                r = r / mag
        sims = np.real(K @ np.conj(r)) / self.d
        if belief_field and len(cand_words) > 1:
            sims = sims - float(sims.mean())
        order = np.argsort(-sims)[:top_k]
        return [(cand_words[int(i)], float(sims[int(i)])) for i in order]

    # ── PCA over concepts: discover emergent dimensions ───────────────────
    def emergent_axes(self, k=8):
        """
        Top-k principal components over the concept matrix.
        Returns (axes, explained_variance_ratio, axis_extremes) where
        axis_extremes[i] = (most-positive-word, most-negative-word) along axis i.
        Treats concepts as real vectors via concat(real, imag).
        """
        if not self.concepts:
            return None, None, []
        words = list(self.concepts.keys())
        # stack as (n, 2d) real matrix
        M = np.stack([np.concatenate([c.real, c.imag])
                      for c in (self.concepts[w] for w in words)])
        M = M - M.mean(axis=0, keepdims=True)
        # SVD for PCA
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        var_total = float((S ** 2).sum())
        ratios = ((S[:k] ** 2) / max(var_total, 1e-9)).tolist()
        axes = Vt[:k]    # (k, 2d)
        # project each word
        proj = M @ axes.T    # (n, k)
        extremes = []
        for i in range(k):
            col = proj[:, i]
            pos_idx = int(np.argmax(col))
            neg_idx = int(np.argmin(col))
            extremes.append((words[pos_idx], words[neg_idx]))
        return axes, ratios, extremes
