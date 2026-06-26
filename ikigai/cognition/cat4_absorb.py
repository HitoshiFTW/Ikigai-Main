"""
ikigai.cognition.cat4_absorb -- Pack 262 cat-4 ICL pair absorb.

Day 75 / Day 76. Reifies Kanerva (2026) §6.1 focus vector construct.
Detects (state, action) ICL pairs in teacher chains and writes
bind(state_HV, action_HV) into b_self bank under role 'icl_pair'.

Per design doc papers/cat4_focus_vector_design_2026_06_18.md.

WHAT THIS BUILDS
----------------
- focus_HV(t) = Σ_{i in window} Pi_k(t-i, token_HV(i))     [running working memory]
- bind(state_HV, action_HV)                                  [pair representation]
- b_self role 'icl_pair'                                     [storage]
- pair boundary detection                                    [conservative heuristics]
- Pack 251 OPV-style action gating                           [reject junk pairs]

CAT-4 vs CAT-1/2/3
-------------------
cat-1 : next-token (one token of context -> one token of action)
cat-2 : ontological triples (relation graph)
cat-3 : reasoning state graphs (multi-token think chains)
cat-4 : ICL pairs (one MULTI-token state -> one MULTI-token action)

PAIR DETECTION
--------------
Conservative heuristics for v0:
  - blank line (\n\n)
  - sentence-end punctuation [.!?] followed by capital letter
  - "Q:" / "A:" style framing detected emergently (any token that
    repeats at sentence start as a framing marker)

NO hardcoded "Q:"/"A:" lexicon. Pattern detection is positional +
repetitional.

ACTION GATING
-------------
v0: simple rules
  - action length must be 1-40 tokens (skip if too long)
  - action must NOT be identical to state (skip pure echo)
  - action must contain at least one content-bearing token
Later versions: pack 251 OPV-style RPE gating.
"""

import hashlib
import re
import numpy as np

# Same digit-aware tokenizer as cat-3
_TOK_RE = re.compile(r"[a-z]+|-?\d+")


def _stable_anchor(state_toks):
    """Pack 273 -- deterministic anchor hash across processes.
    Python's built-in hash() for tuples uses PYTHONHASHSEED
    randomization (different value per process), which breaks
    the cache lookup at query time vs the bootstrap-time anchor.
    blake2b is stable across processes and platforms.
    """
    h = hashlib.blake2b(
        '|'.join(state_toks).encode('utf-8'),
        digest_size=8,
    ).digest()
    return '__state_' + str(int.from_bytes(h, 'big') % 10**12)


def tokenize_chain(text, min_len=1, max_len=20):
    """Lowercase + digit-aware tokenizer (shared with cat-3)."""
    return [t for t in _TOK_RE.findall(text.lower())
              if min_len <= len(t) <= max_len]


_BOUNDARY_RE = re.compile(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])")


def split_pairs(raw_text):
    """Return list of (raw segment) chunks delimited by pair boundaries.
    Empty / whitespace-only segments dropped."""
    parts = _BOUNDARY_RE.split(raw_text)
    return [p.strip() for p in parts if p and p.strip()]


class Cat4Absorb:
    """Pack 262 cat-4 absorb engine.

    Composes Pack 85 PiK for positional binding + Pack 251 OPV for
    write-gating + the substrate's role-routing.

    USAGE
    -----
        cat4 = Cat4Absorb(mr, num_enc, pik, window=10)
        cat4.absorb_chain(text)
        # b_self bank now has 'icl_pair' role with bound (state, action) HVs
    """

    def __init__(self, mr, num_enc, pik,
                  pair_role='icl_pair',
                  state_role='icl_state',
                  action_token_role='icl_action_token',
                  pair_bank='b_self',
                  window=10,
                  min_action_len=1,
                  max_action_len=40,
                  min_state_len=1,
                  max_state_len=40,
                  opv_enabled=True,
                  opv_known_thresh=0.20,
                  opv_surprise_thresh=0.70,
                  opv_novel_factor=1.0,
                  opv_known_factor=0.30,
                  opv_partial_factor=0.60,
                  opv_surprise_factor=1.0,
                  opv_unlearn_factor=0.50,
                  opv_max_anchors=512,
                  opv_min_state_sim=0.15,
                  opv_seed=265):
        """
        Args:
            mr               -- MultiRoleMemory
            num_enc          -- Pack 252 NumericEncoder (for digit token HVs)
            pik              -- Pack 85 PiK (for positional binding)
            pair_role        -- role name for ICL pair bindings
            pair_bank        -- bank id for that role (default b_self)
            window           -- working-memory window size (Miller 7±2)
            min/max action_len -- gate: skip pairs whose action is out of range
            min/max state_len  -- gate: skip pairs whose state is out of range
        """
        self.mr = mr
        self.num_enc = num_enc
        self.pik = pik
        self.pair_role = str(pair_role)
        self.state_role = str(state_role)
        self.action_token_role = str(action_token_role)
        self.pair_bank = str(pair_bank)
        self.window = int(window)
        self.min_action_len = int(min_action_len)
        self.max_action_len = int(max_action_len)
        self.min_state_len = int(min_state_len)
        self.max_state_len = int(max_state_len)
        # Pack 265 OPV-style RPE write gating knobs
        self.opv_enabled = bool(opv_enabled)
        self.opv_known_thresh = float(opv_known_thresh)
        self.opv_surprise_thresh = float(opv_surprise_thresh)
        self.opv_novel_factor = float(opv_novel_factor)
        self.opv_known_factor = float(opv_known_factor)
        self.opv_partial_factor = float(opv_partial_factor)
        self.opv_surprise_factor = float(opv_surprise_factor)
        self.opv_unlearn_factor = float(opv_unlearn_factor)
        # Sub-sample anchors during the OPV check so the gate stays
        # O(opv_max_anchors) per pair instead of O(|anchors|). With
        # tens of thousands of anchors the full scan dominates wall
        # time and OPV becomes the bottleneck.
        self.opv_max_anchors = int(opv_max_anchors)
        self.opv_min_state_sim = float(opv_min_state_sim)
        self._opv_rng = np.random.default_rng(int(opv_seed))
        self._ensure_role()
        self._ensure_state_role()
        self._ensure_action_token_role()
        self.stats = {
            'chains': 0,
            'segments_seen': 0,
            'pairs_detected': 0,
            'pairs_written': 0,
            'pairs_gated_action_len': 0,
            'pairs_gated_state_len': 0,
            'pairs_gated_echo': 0,
            # Pack 265 OPV bookkeeping
            'opv_novel': 0,
            'opv_known': 0,
            'opv_partial': 0,
            'opv_surprise': 0,
            'opv_unlearned': 0,
            # Pack 273 cache bookkeeping
            'cache_writes': 0,
        }
        # Pack 273 anchor-action cache.  Maps anchor_id (the same
        # deterministic hash used for icl_pair writes) to a list of
        # action token tuples observed at that anchor during absorb.
        # Bypasses unbind + 9k cleanup at query time (Kanerva 2022
        # associative-mapping path).
        # Pack 279: compact int64-keyed bytes-valued cache.  Same API as
        # the Pack 273+274 string-keyed dict via back-compat shim; ~6.8x
        # smaller per-entry (42 bytes vs 288).  1M facts ~40 MB.
        #
        # Pack 282.5 LMDB toggle (Day 77): when env var
        # NEUROSEED_LMDB_CACHE is set to a directory path, swap the
        # in-memory cache for the LMDB-backed variant.  Required for
        # >10M fact organisms; in-memory compact stays default for
        # the current 837-entry production state.
        import os as _os
        lmdb_path = _os.environ.get('NEUROSEED_LMDB_CACHE')
        if lmdb_path:
            try:
                from ikigai.cognition.cat4_lmdb_cache import (
                    LMDBAnchorCache)
                self.anchor_actions = LMDBAnchorCache(
                    lmdb_path, map_size_gb=1.0)
            except Exception as _e:
                # Soft-fail to in-memory compact; log to stats.
                from ikigai.cognition.cat4_compact_cache import (
                    CompactAnchorCache)
                self.anchor_actions = CompactAnchorCache()
                self.stats.setdefault('lmdb_fallback', str(_e))
        else:
            from ikigai.cognition.cat4_compact_cache import (
                CompactAnchorCache)
            self.anchor_actions = CompactAnchorCache()
        # Day 75 v5: action vocab persisted via substrate role
        # `icl_action_token` -- see action_vocab property above.
        # Pack 280 vectorized recall cache.  Stacked (N, d) matrices of
        # stored states + bound HVs replace per-anchor Python loop in
        # recall_action.  Invalidated on absorb_chain.  Built lazily.
        self._pack280_recall_anchors = None
        self._pack280_recall_states = None
        self._pack280_recall_bounds = None

    def _register_role(self, role_name):
        """Generic role registration + bank route to pair_bank."""
        mr = self.mr
        if role_name in mr.roles:
            return
        rng = np.random.default_rng(262_000
                                       + sum(ord(c) for c in role_name))
        ph = rng.uniform(-np.pi, np.pi, mr.d).astype(np.float32)
        mr.roles[role_name] = np.exp(1j * ph).astype(np.complex64)
        if getattr(mr, '_bank_assignment', None) is not None:
            bank_ids = list(mr._bank_assignment.keys())
            target = self.pair_bank if self.pair_bank in bank_ids else bank_ids[0]
            mr._role_to_bank[role_name] = target
            roles_list = mr._bank_assignment[target].setdefault('roles', [])
            if role_name not in roles_list:
                roles_list.append(role_name)
        mr._role_targets.setdefault(role_name, set())

    def _ensure_role(self):
        self._register_role(self.pair_role)

    def _ensure_state_role(self):
        """Store the state HV per anchor under a separate role so
        recall can compute true state similarity (instead of the
        broken self-consistent algebraic identity)."""
        self._register_role(self.state_role)

    def _ensure_action_token_role(self):
        """Day 75 v5 -- persistence fix: each action token is also
        added to a substrate role 'icl_action_token' so the action
        vocabulary survives organism.ikg reload via _role_targets.
        Python-attribute action_vocab is now derived (property)."""
        self._register_role(self.action_token_role)

    @property
    def action_vocab(self):
        """Substrate-persistent action vocabulary. Reconstructed from
        _role_targets[icl_action_token] each access -- survives reloads."""
        return set(self.mr._role_targets.get(self.action_token_role, set()))

    # ---- focus HV construction --------------------------------------

    def focus_hv(self, tokens):
        """Build a focus vector from a token list. Sums Pi_k-permuted
        token HVs over a window with trailing-weight bias.

        Day 75 v7: trailing tokens get EXPONENTIALLY higher weight than
        earlier ones. Without this, common framing tokens (what/is/the/
        capital/of) dominate and ALL 'capital of X' queries collapse
        to the same focus HV. With trailing-bias, the X token (the
        question-specific content at the end) dominates.

        Decay factor = 0.5 per position from end. Last token weight 1.0,
        previous 0.5, 0.25, etc. Effective window ~ 4 trailing tokens.
        """
        if not tokens:
            return np.zeros(self.mr.d, dtype=np.complex64)
        toks = tokens[-self.window:]
        n = len(toks)
        accum = np.zeros(self.mr.d, dtype=np.complex128)
        n_primes = self.pik.n
        for i, t in enumerate(toks):
            k_idx = i % n_primes
            # Position-from-end: last token (i=n-1) gets weight 1.0,
            # earlier tokens decay by 0.5 per step back.
            decay = 0.5 ** (n - 1 - i)
            hv = self.mr.ck.key(t).astype(np.complex128)
            accum += decay * self.pik.pi(k_idx, hv)
        mag = np.abs(accum)
        mag = np.where(mag > 1e-9, mag, 1.0)
        return (accum / mag).astype(np.complex64)

    # ---- absorb ------------------------------------------------------

    def absorb_chain(self, text):
        """Absorb one teacher chain. Detect (state, action) pairs and
        write bind(state_HV, action_HV) into b_self."""
        self.stats['chains'] += 1
        segments = split_pairs(str(text))
        self.stats['segments_seen'] += len(segments)
        if len(segments) < 2:
            return
        # Tokenize each segment
        seg_toks = [tokenize_chain(s) for s in segments]
        # Consecutive (segment_i, segment_{i+1}) treated as (state, action)
        for i in range(len(seg_toks) - 1):
            state_toks = seg_toks[i]
            action_toks = seg_toks[i + 1]
            if not state_toks or not action_toks:
                continue
            self.stats['pairs_detected'] += 1
            # Gate state length
            if (len(state_toks) < self.min_state_len
                    or len(state_toks) > self.max_state_len):
                self.stats['pairs_gated_state_len'] += 1
                continue
            # Gate action length
            if (len(action_toks) < self.min_action_len
                    or len(action_toks) > self.max_action_len):
                self.stats['pairs_gated_action_len'] += 1
                continue
            # Gate echo (state == action exactly)
            if state_toks == action_toks:
                self.stats['pairs_gated_echo'] += 1
                continue
            # Build focus HVs
            state_hv = self.focus_hv(state_toks)
            action_hv = self.focus_hv(action_toks)
            # Pack 265 OPV gate: predict action from state (substrate's
            # current best guess); compare to actual action HV; scale write
            # by prediction error. Skip on first writes (no substrate signal
            # yet -> NOVEL path).
            write_strength = self.opv_novel_factor
            if self.opv_enabled:
                pred_hv = None
                # Pack 271 fix: deterministic anchor first.  The
                # absorb-time anchor for these state_toks is
                # reproducible by hash; if we already wrote it
                # before, it lives in role_targets.  Use IT
                # directly rather than rely on a random subsample
                # to happen to include it.
                #
                # Pack 265 fast path (random subsample) is the
                # fallback when the deterministic anchor is not
                # yet present.
                deterministic_anchor = _stable_anchor(state_toks)
                try:
                    role_targets = self.mr._role_targets.get(
                        self.pair_role, set())
                    if deterministic_anchor in role_targets:
                        anchors = [deterministic_anchor]
                    else:
                        anchors = list(role_targets)
                        if len(anchors) > self.opv_max_anchors:
                            idx = self._opv_rng.choice(
                                len(anchors),
                                size=self.opv_max_anchors,
                                replace=False)
                            anchors = [anchors[int(i)] for i in idx]
                    if anchors:
                        query_state = state_hv  # already focus_hv
                        best_sim = -1e9
                        best_anchor = None
                        best_stored = None
                        for a in anchors:
                            stored = self.mr.recall(a, self.state_role)
                            if stored is None:
                                continue
                            s_n = np.asarray(stored, dtype=np.complex64)
                            s_mag = float(np.abs(s_n).mean()) + 1e-12
                            s_n = s_n / s_mag
                            sim = float(np.real(
                                np.vdot(s_n, query_state)) / self.mr.d)
                            if sim > best_sim:
                                best_sim = sim
                                best_anchor = a
                                best_stored = s_n
                        if (best_anchor is not None
                                and best_sim >= self.opv_min_state_sim):
                            bound = self.mr.recall(
                                best_anchor, self.pair_role)
                            if bound is not None:
                                b_n = np.asarray(bound,
                                                   dtype=np.complex64)
                                action_est = (b_n * np.conj(best_stored)
                                                ).astype(np.complex64)
                                mag = (float(
                                    np.abs(action_est).mean()) + 1e-12)
                                pred_hv = (action_est / mag
                                             ).astype(np.complex64)
                except Exception:
                    pred_hv = None
                if pred_hv is None:
                    write_strength = self.opv_novel_factor
                    self.stats['opv_novel'] += 1
                else:
                    pred_n = np.asarray(pred_hv, dtype=np.complex64)
                    pred_n = pred_n / (float(np.abs(pred_n).mean()) + 1e-12)
                    act_n = action_hv.astype(np.complex64)
                    act_n = act_n / (float(np.abs(act_n).mean()) + 1e-12)
                    cos = float(
                        np.real(np.vdot(pred_n, act_n)) / self.mr.d)
                    err = 1.0 - cos
                    if err < self.opv_known_thresh:
                        # KNOWN: substrate already has it -> weak reinforce
                        write_strength = self.opv_known_factor
                        self.stats['opv_known'] += 1
                    elif err > self.opv_surprise_thresh:
                        # SURPRISE: substrate predicted wrong action ->
                        # unlearn predicted bound + learn actual at full
                        # strength
                        predicted_bound = (state_hv * pred_n).astype(
                            np.complex64)
                        pb_mag = float(
                            np.abs(predicted_bound).mean()) + 1e-12
                        predicted_bound /= pb_mag
                        unlearn_hv = (
                            -self.opv_unlearn_factor * predicted_bound
                        ).astype(np.complex64)
                        unlearn_anchor = _stable_anchor(state_toks)
                        self.mr.write_relation(
                            unlearn_anchor, self.pair_role, unlearn_hv)
                        write_strength = self.opv_surprise_factor
                        self.stats['opv_surprise'] += 1
                        self.stats['opv_unlearned'] += 1
                    else:
                        # PARTIAL: between known and surprise -> standard
                        write_strength = self.opv_partial_factor
                        self.stats['opv_partial'] += 1
            # Bind state -> action under 'icl_pair' role
            bound = state_hv * action_hv
            bound_norm = bound / (float(np.abs(bound).mean()) + 1e-12)
            bound_norm = (write_strength * bound_norm).astype(np.complex64)
            # Use a stable hash of state tokens as the "anchor token"
            # so we can later retrieve by similar state
            anchor = _stable_anchor(state_toks)
            self.mr.write_relation(anchor, self.pair_role, bound_norm)
            self.mr._role_targets[self.pair_role].add(anchor)
            # Also store the raw state HV under state_role for similarity
            # lookup at recall time (state encoding does NOT need RPE
            # gating -- it's the address, not the content)
            self.mr.write_relation(anchor, self.state_role,
                                     state_hv.astype(np.complex64))
            self.mr._role_targets[self.state_role].add(anchor)
            # Persist each action token to substrate via icl_action_token
            # role so action_vocab survives organism.ikg reload.
            for t in action_toks:
                self.mr._role_targets[self.action_token_role].add(t)
            # Pack 273 anchor-action cache: store the raw action token
            # tuple at this anchor.  Bypasses cleanup at query time.
            tok_tuple = tuple(action_toks)
            _av = getattr(self.anchor_actions, 'add_value', None)
            if _av is not None:                       # Pack 330 multi-value
                if _av(anchor, tok_tuple):
                    self.stats['cache_writes'] += 1
            else:                                     # plain-dict fallback
                existing = self.anchor_actions.get(anchor)
                if existing is None:
                    self.anchor_actions[anchor] = [tok_tuple]
                    self.stats['cache_writes'] += 1
                elif tok_tuple not in existing:
                    existing.append(tok_tuple)
                    self.stats['cache_writes'] += 1
            self.stats['pairs_written'] += 1
            # Pack 280 INCREMENTAL: append (or overwrite) this anchor's row in
            # the recall cache so we never re-pay the cold rebuild cost when
            # a single new fact is absorbed.  Only valid when the cache has
            # already been built; otherwise the next recall_action will do
            # the full lazy build anyway.
            if self._pack280_recall_states is not None:
                sn = state_hv.astype(np.complex64)
                sn_mag = float(np.abs(sn).mean()) + 1e-12
                sn = sn / sn_mag
                bn = bound_norm.astype(np.complex64)
                idx = None
                try:
                    idx = self._pack280_recall_anchors.index(anchor)
                except ValueError:
                    pass
                if idx is not None:
                    # Reinforce existing anchor -- overwrite the row
                    self._pack280_recall_states[idx] = sn
                    self._pack280_recall_bounds[idx] = bn
                else:
                    # New anchor -- vstack one new row to each matrix
                    self._pack280_recall_states = np.vstack(
                        [self._pack280_recall_states, sn[None, :]])
                    self._pack280_recall_bounds = np.vstack(
                        [self._pack280_recall_bounds, bn[None, :]])
                    self._pack280_recall_anchors.append(anchor)
        return

    # ---- Pack 273 cache-only populate --------------------------------

    def populate_cache_from_text(self, text):
        """Parse `text` into (state, action) pairs via the same logic
        as absorb_chain, but write ONLY to the anchor_actions cache
        -- no substrate writes, no OPV gate.  Used to backfill the
        Pack 273 cache for an organism that was trained before the
        cache existed.

        Returns int = number of pairs added to cache.
        """
        if not text:
            return 0
        segments = split_pairs(str(text))
        if len(segments) < 2:
            return 0
        seg_toks = [tokenize_chain(s) for s in segments]
        added = 0
        for i in range(len(seg_toks) - 1):
            state_toks = seg_toks[i]
            action_toks = seg_toks[i + 1]
            if not state_toks or not action_toks:
                continue
            if (len(state_toks) < self.min_state_len
                    or len(state_toks) > self.max_state_len):
                continue
            if (len(action_toks) < self.min_action_len
                    or len(action_toks) > self.max_action_len):
                continue
            if state_toks == action_toks:
                continue
            anchor = _stable_anchor(state_toks)
            tok_tuple = tuple(action_toks)
            _av = getattr(self.anchor_actions, 'add_value', None)
            if _av is not None:                       # Pack 330 multi-value
                if _av(anchor, tok_tuple):
                    added += 1
            else:                                     # plain-dict fallback
                existing = self.anchor_actions.get(anchor)
                if existing is None:
                    self.anchor_actions[anchor] = [tok_tuple]
                    added += 1
                elif tok_tuple not in existing:
                    existing.append(tok_tuple)
                    added += 1
        return added

    # ---- recall ------------------------------------------------------

    def _pack280_build_recall_cache(self):
        """Stack all stored state HVs + bound HVs into (N, d) matrices.
        Called lazily on first recall_action after invalidation.

        Cost: N substrate recall calls + 2 np.stack.  Amortized over
        subsequent queries until next absorb_chain.

        Memory: 2 * N * d * 8 bytes complex64.  At N=31K, d=400:
        ~200 MB total.  Acceptable under 1 GB ceiling.
        """
        anchors = sorted(self.mr._role_targets.get(self.pair_role, set()))
        states = []
        bounds = []
        keep = []
        for a in anchors:
            s = self.mr.recall(a, self.state_role)
            if s is None:
                continue
            b = self.mr.recall(a, self.pair_role)
            if b is None:
                continue
            sn = np.asarray(s, dtype=np.complex64)
            mag = float(np.abs(sn).mean()) + 1e-12
            states.append(sn / mag)
            bounds.append(np.asarray(b, dtype=np.complex64))
            keep.append(a)
        if states:
            self._pack280_recall_states = np.stack(states, axis=0)
            self._pack280_recall_bounds = np.stack(bounds, axis=0)
        else:
            self._pack280_recall_states = np.zeros(
                (0, self.mr.d), dtype=np.complex64)
            self._pack280_recall_bounds = np.zeros(
                (0, self.mr.d), dtype=np.complex64)
        self._pack280_recall_anchors = keep

    def recall_action(self, state_tokens, top_k=5):
        """Pack 280 vectorized recall: single matmul over stacked
        (N, d) cached state matrix replaces per-anchor Python loop.

        Returns top_k (anchor, action_hv, state_sim) sorted by sim.

        Speedup target vs Pack 262 v2: 28s -> 50ms per query (~560x).
        Falls back gracefully on empty cache.
        """
        if self._pack280_recall_states is None:
            self._pack280_build_recall_cache()
        if not self._pack280_recall_anchors:
            return []
        query_state = self.focus_hv(state_tokens).astype(np.complex64)
        # Vectorized inner product: vdot(stored_i, query) for each row i.
        # = sum(conj(stored_i) * query) = (stored.conj() @ query) per row
        dots = self._pack280_recall_states.conj() @ query_state
        state_sims = (np.real(dots) / float(self.mr.d)).astype(np.float32)
        N = state_sims.shape[0]
        k = int(min(top_k, N))
        if k <= 0:
            return []
        # argpartition for O(N) top-k, then sort the k winners
        if N > k:
            part = np.argpartition(-state_sims, k - 1)[:k]
            top_idx = part[np.argsort(-state_sims[part])]
        else:
            top_idx = np.argsort(-state_sims)
        results = []
        for i in top_idx:
            sn = self._pack280_recall_states[i]
            bn = self._pack280_recall_bounds[i]
            action_est = bn * np.conj(sn)
            mag = float(np.abs(action_est).mean()) + 1e-12
            action_n = (action_est / mag).astype(np.complex64)
            results.append(
                (self._pack280_recall_anchors[int(i)], action_n,
                 float(state_sims[int(i)])))
        return results

    # ---- Pack 272 cached action codebook (speed-up) ------------------

    def action_codebook(self):
        """Return (sorted_vocab, codebook_matrix). Cached on the
        instance.  Invalidated when action_vocab grows by >= 50
        tokens since last cache.  general_reasoner uses this
        instead of rebuilding focus_hv per cand per query (saves
        ~9k focus_hv calls per reason() call)."""
        vocab = sorted(self.action_vocab)
        cached_vocab = getattr(self, '_pack272_cb_vocab', None)
        if (cached_vocab is not None
                and cached_vocab == vocab):
            return cached_vocab, self._pack272_cb_K
        if not vocab:
            self._pack272_cb_vocab = []
            self._pack272_cb_K = np.zeros((0, self.mr.d),
                                            dtype=np.complex64)
            return [], self._pack272_cb_K
        K = np.stack([self.focus_hv([c])
                       for c in vocab]).astype(np.complex64)
        self._pack272_cb_vocab = list(vocab)
        self._pack272_cb_K = K
        return self._pack272_cb_vocab, self._pack272_cb_K

    def predict_action_hv_efe(self, state_tokens, goal_hv=None,
                                top_k=5, min_state_sim=0.05,
                                epistemic_weight=1.0, pragmatic_weight=1.0):
        """Pack 262 v3 -- EFE-scored action selection over cat-4 recalls.

        Adapts Pack 258 ActiveInferencePlanner's EFE formulation to
        cat-4's ICL pair recall:

            EFE(action) = -[ epistemic + pragmatic ]
            epistemic   = sharpness of the state-similarity distribution
                          over recalls (peaked = high info-gain)
            pragmatic   = cos(action_HV, goal_HV) if goal given, else 0

        Returns weighted superposition of top-k actions by -EFE
        (i.e., favoring actions where the substrate is both confident
        AND goal-aligned).

        Falls back to predict_action_hv (pure similarity weighting)
        when goal_hv is None and epistemic_weight is 0.
        """
        recalls = self.recall_action(state_tokens, top_k=top_k)
        if not recalls:
            return None
        # Filter by min state similarity
        filtered = [(a, h, s) for (a, h, s) in recalls if s >= min_state_sim]
        if not filtered:
            return None
        sims = np.array([s for (_, _, s) in filtered], dtype=np.float64)
        # Per-candidate epistemic value: each candidate's own state
        # similarity scaled into [0, 1]. Higher = more substrate
        # confidence in this specific recall.
        epis_per = np.clip(sims, 0.0, 1.0)
        # Pragmatic value per candidate (cosine to goal HV)
        prag_per = np.zeros(len(filtered), dtype=np.float64)
        if goal_hv is not None:
            goal = np.asarray(goal_hv, dtype=np.complex64)
            gm = float(np.abs(goal).mean()) + 1e-12
            goal = goal / gm
            for i, (_, action_hv, _) in enumerate(filtered):
                am = float(np.abs(action_hv).mean()) + 1e-12
                a_n = action_hv / am
                prag_per[i] = float(np.real(np.vdot(goal, a_n))
                                      / self.mr.d)
        # Per-candidate EFE: -(epistemic_per + pragmatic_per)
        efe_per = -(epistemic_weight * epis_per
                     + pragmatic_weight * prag_per)
        # Day 75 v6: sharp softmax (beta=20) prevents prototype collapse
        # at large absorb scales -- v4 worked at 7 anchors because state
        # similarities were 0.999 vs ~0; at 1500+ chains, multiple
        # archetype-similar anchors get 0.5-0.9 sims and uniform softmax
        # averaged them into mushy 'paris-for-everything'.
        weights = np.exp(-20.0 * (efe_per - efe_per.min()))
        weights /= (weights.sum() + 1e-12)
        # Superpose actions by EFE-weight
        action_sum = np.zeros(self.mr.d, dtype=np.complex64)
        for i, (_, action_hv, _) in enumerate(filtered):
            action_sum += weights[i] * action_hv
        mag = float(np.abs(action_sum).mean()) + 1e-12
        return (action_sum / mag).astype(np.complex64)

    def predict_action_hv(self, state_tokens, top_k=3, min_state_sim=0.05):
        """Convenience: return the SUPERPOSITION of top-k action HVs
        weighted by state similarity. Caller cleans up against own
        codebook. Drops anchors whose state similarity is below
        `min_state_sim` (treats them as unrelated noise).
        """
        recalls = self.recall_action(state_tokens, top_k=top_k)
        if not recalls:
            return None
        action_sum = np.zeros(self.mr.d, dtype=np.complex64)
        n_kept = 0
        for _, action_hv, state_sim in recalls:
            if state_sim < min_state_sim:
                continue
            action_sum += state_sim * action_hv
            n_kept += 1
        if n_kept == 0:
            return None
        mag = float(np.abs(action_sum).mean()) + 1e-12
        return (action_sum / mag).astype(np.complex64)

    # ---- diagnostics -----------------------------------------------

    def summary(self):
        s = dict(self.stats)
        s['pair_targets'] = len(self.mr._role_targets.get(self.pair_role, set()))
        return s
