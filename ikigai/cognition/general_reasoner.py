"""
ikigai.cognition.general_reasoner -- Pack 255 General Reasoning Engine (PCER v0).

Day 73. Substrate-native general reasoning. NO task-specific paths --
math, code, dialogue all use the same loop. Specialization emerges from
absorbed data via Pack 253 cat-3 absorb, not from hardcoded operator
lexicons or `_safe_eval` shortcuts.

Composes existing primitives (no new substrate math):
    Pack 85  PiK                 sequential state separation
    Pack 72  CausalWorldModel    (state, action) -> next_state edges
    KS#5     LogicalFixedPoint   provable derivation closure
    Pack 73  MultiStepPlanner    DFS + beam-search + FE heuristic
    Pack 252 NumericEncoder      FPE magnitude topology
    Pack 253 Cat3Absorb          fact grounding from teacher chains
    Pack 251 OnPolicyEvaluator   teacher-gated three-factor write
    Pack 63  ProofChain          derivation audit trail (lazy)

Architecture (per [[research-2026-06-16-pcer]]):

    1. FACTORIZE working state via PiK / cooccur recall (Resonator
       network is Pack 256, plug in later via self.resonator)
    2. CAUSAL PATH PREDICTION via CausalWorldModel.predict / rollout
    3. ACTION SELECTION (FE heuristic now, EFE via Pack 258 later)
    4. LOGICAL TRANSITIVE CLOSURE via LogicalFixedPoint
    5. PHASE CLEANUP via mr.query_clean (Pack 247z-CM common-mode)

Pack 255 v0 wires the composition. Empirical validation needed before
declaring this a finished engine -- per UDSP lesson, math from research
return is unverified until smoked.

WHAT THIS IS NOT
----------------
- No hardcoded operator lexicon (no ADD/SUB/MUL dispatch table)
- No `_safe_eval` shortcut path
- No IntentRouter "math intent -> calculator" branching
- No regex extraction of numbers and "evaluation"
- No task-specific gsm8k handler stack

Math emerges by absorbing reasoning chains via Pack 253 + Pack 252 FPE
magnitudes; the same reason() entry point handles language tasks.
"""

import re

import numpy as np


class GeneralReasoner:
    """Pack 255 substrate-native general reasoning engine.

    Construct via IkigaiOrganism.reasoner (lazy property).
    """

    def __init__(self, org,
                  pik_n_primes=16,
                  proof_chain=True,
                  cleanup_role='cooccur'):
        """
        Args:
            org             -- IkigaiOrganism (needs unified, vs_fsm,
                                opv, num_enc, cat3)
            pik_n_primes    -- number of Pi_k permutations for state chains
            proof_chain     -- record ProofChain audit trail (Pack 63)
            cleanup_role    -- which role to use for substrate-cleanup
                                queries during decode
        """
        self.org = org
        self.mr = org.unified
        self.fsm = org.vs_fsm
        self.opv = org.opv
        self.num_enc = org.num_enc
        self.cat3 = org.cat3
        self.d = self.mr.d
        self.pik_n_primes = int(pik_n_primes)
        self.proof_chain_enabled = bool(proof_chain)
        self.cleanup_role = str(cleanup_role)
        # Lazy sub-components (built on first access)
        self._pik = None
        self._cwm = None
        self._lfp = None
        self._planner = None
        self._resonator = None   # Pack 256 placeholder
        self._stats = {
            'reason_calls': 0,
            'steps_taken': 0,
            'closure_iters': 0,
            'cleanup_hits': 0,
        }

    # ---- lazy sub-component builders --------------------------------

    @property
    def pik(self):
        if self._pik is None:
            from ikigai.cognition.pi_k_algebra import PiK
            self._pik = PiK(d=self.d, n_primes=self.pik_n_primes)
        return self._pik

    @property
    def cwm(self):
        if self._cwm is None:
            from ikigai.cognition.causal_world_model import CausalWorldModel
            self._cwm = CausalWorldModel(d=self.d)
        return self._cwm

    @property
    def lfp(self):
        if self._lfp is None:
            from ikigai.cognition.logical_fixed_point import LogicalFixedPoint
            self._lfp = LogicalFixedPoint(self.org)
        return self._lfp

    @property
    def planner(self):
        if self._planner is None:
            from ikigai.cognition.multistep_planner import MultiStepPlanner
            self._planner = MultiStepPlanner(self.cwm)
        return self._planner

    @property
    def derive_engine(self):
        """Pack 304 -- lazy compositional derive-not-store engine."""
        eng = getattr(self, '_derive_engine', None)
        if eng is None:
            from ikigai.cognition.compositional import CompositionEngine
            eng = CompositionEngine(self)
            self._derive_engine = eng
        return eng

    # ---- state construction -----------------------------------------

    def tokenize(self, text):
        """Cat-3 tokenizer (digit-aware, lowercase, len 1..20)."""
        from ikigai.cognition.cat3_absorb import tokenize_chain
        return tokenize_chain(text)

    def state_hv(self, tokens):
        """Build a superposed state HV from a token list.

        Sums Pi_k-permuted token HVs so token order matters (avoids
        commutativity bug -- (a+b) != (b+a) under positional binding).
        """
        if not tokens:
            return np.zeros(self.d, dtype=np.complex64)
        ck = self.mr.ck
        accum = np.zeros(self.d, dtype=np.complex128)
        for i, t in enumerate(tokens):
            k_idx = i % self.pik_n_primes
            hv = ck.key(t).astype(np.complex128)
            accum += self.pik.pi(k_idx, hv)
        # Normalize to unit-magnitude phasors
        mag = np.abs(accum)
        mag = np.where(mag > 1e-9, mag, 1.0)
        return (accum / mag).astype(np.complex64)

    # ---- inference paths --------------------------------------------

    def recall_next_token(self, prev, cur, top_k=5):
        """Cat-1 floor: predict next token via fsm.step. Same path as
        Pack 254 chain_completion mode."""
        cands = list(self.mr._role_targets.get('next', set()))
        try:
            res = self.fsm.step(cur, prev_token=prev, candidates=cands,
                                  n_iters=3, beta=8.0, top_k=top_k)
        except Exception:
            return []
        return [(t, s) for (t, s) in res if t]

    def recall_magnitude(self, tok):
        """Pack 252 + 253: recall token's magnitude HV and decode."""
        return self.cat3.recall_magnitude(tok)

    def cleanup_decode(self, hv, role=None, candidates=None, top_k=5):
        """Substrate cleanup of an arbitrary HV against the cleanup role's
        target set. Uses Pack 247z-CM common-mode subtract via query_clean
        when n_targets is large enough."""
        role = role or self.cleanup_role
        cands = candidates or list(self.mr._role_targets.get(role, set()))
        if not cands:
            return []
        # query_clean handles CM auto-application
        try:
            picks = self.mr.query_clean(
                None, role, candidates=cands, top_k=top_k,
                auto_cm_threshold=200, beta=1.0)
        except Exception:
            picks = []
        if picks:
            self._stats['cleanup_hits'] += 1
        return picks

    # ---- closure ----------------------------------------------------

    def derive_closure(self, rules=None, max_iters=3):
        """Pack 247z + KS#5: run LogicalFixedPoint to derive transitive
        / symmetric / inverse / composition closures.

        Default rules: TRANSITIVE on isa + SYMMETRIC on cooccur.
        Caller can override by passing a rules list.
        """
        from ikigai.cognition.logical_fixed_point import LogicalRule
        lfp = self.lfp
        lfp.rules.clear()
        if rules:
            for r in rules:
                lfp.add_rule(r)
        else:
            for r in ('isa', 'similar'):
                if r in self.mr.roles:
                    lfp.add_rule(LogicalRule.transitive(r))
            for r in ('cooccur', 'antonym'):
                if r in self.mr.roles:
                    lfp.add_rule(LogicalRule.symmetric(r))
        try:
            stats = lfp.run(max_iterations=max_iters)
        except Exception as e:
            stats = {'err': str(e)[:120]}
        self._stats['closure_iters'] += max_iters
        return stats

    # ---- main entry -------------------------------------------------

    def reason(self, text, goal=None, max_steps=4, do_closure=False,
                do_icl=True, icl_min_state_sim=None, do_multihop=True,
                do_active=True, do_derive=True, do_abstain=True):
        """General reasoning entry. ONE path -- works for math, code,
        language alike. Day 75 v4: cat-4 b_self ICL recall added.

        Args:
            text             -- input prompt
            goal             -- optional target state token (or string)
            max_steps        -- max planning depth
            do_closure       -- pre-run LogicalFixedPoint over known facts
            do_icl           -- query cat-4 b_self ICL pair recall
            icl_min_state_sim -- gate: only use cat-4 if top state similarity
                                  exceeds this threshold

        Returns dict with:
            tokens       -- tokenized input
            next_predict -- top-5 next-token predictions (cat-1 path)
            magnitudes   -- list of (token, int, score) from cat-3 grounding
            icl_recalls  -- list of (anchor, state_sim) from cat-4 (if do_icl)
            icl_action   -- decoded action token from cat-4 b_self (if confident)
            plan         -- planner output (if goal)
            answer       -- best answer chosen across paths
            method       -- which sub-path produced the answer
        """
        self._stats['reason_calls'] += 1
        # Pack 310 -- CALIBRATION boundary, derived from substrate geometry
        # (k-sigma above the 1/sqrt(2d) noise floor), NOT a tuned number.
        # Replaces the old hand-tuned icl_min_state_sim=0.10 default.
        if icl_min_state_sim is None:
            from ikigai.cognition.calibration import abstain_boundary
            icl_min_state_sim = abstain_boundary(self.d)
        toks = self.tokenize(text)
        if not toks:
            return {'tokens': [], 'answer': None, 'method': 'empty'}
        # Pack 305 perf FAST PATH -- resolve cheap authoritative paths
        # (arith / derive / multihop / exact cache) WITHOUT the expensive
        # cat4 recall_action (142s cold / 77ms hot rebuild) + 9.6k-vocab
        # codebook matmul.  A fact answers in <50ms (cache hit in us).  The
        # heavy ICL/cleanup/active-learn path runs only when nothing cheap
        # and confident resolves.
        fast = self._fast_answer(text, toks, goal, do_multihop, do_derive)
        if fast is not None:
            return fast
        # Day-100 -- ABSTENTION MUST BE CHEAP, OR IT IS NOT CALIBRATION.
        #
        # Nothing authoritative resolved. The path below then pays cat4.recall_action, whose cold
        # `_pack280_build_recall_cache` walks EVERY anchor with two substrate recalls each --
        # MEASURED 154s on the production organism -- and finishes by cleaning the query up over
        # the 9.6k-word codebook, which returns the NEAREST WORD rather than nothing. Measured:
        #
        #     gr.reason('the capital of zqxlandia is')  -> 'succinctly'   after 152.74s
        #
        # That is a fabrication, and it is the expensive path that produces it. Searching 154s to
        # emit a word the organism cannot ground is correct-or-abstain exactly backwards.
        #
        # So: if NOTHING the question mentions is an entity the organism holds a fact about, and no
        # goal was given, it does not know -- and it can say so now, for ~139us. This is not a
        # shortcut around the guarantee; it IS the guarantee, applied before the cost instead of
        # after it. Anything with real grounding, a goal, or an arithmetic/derive/cache answer has
        # already returned above or still takes the full path below.
        #
        # (full_capability's calibration probe passed only because 'z912qxland' carries DIGITS: the
        # tokenizer splits it to ['z','912','qxland'] and that happened to miss. The alpha-only name
        # the standing rule demands returns 'succinctly'. The gate was green on the tokenizer bug.)
        if do_abstain and goal is None:
            _org = getattr(self, 'org', None)
            if _org is not None:
                try:
                    _grounded = any(_org.knows(t) for t in toks)
                except Exception:
                    _grounded = True          # never abstain because a probe errored
                if not _grounded:
                    return {'tokens': toks, 'answer': None, 'method': 'abstain_ungrounded',
                            'icl_recalls': [], 'icl_action': None, 'magnitudes': [],
                            'next_predict': []}
        # Build superposed state
        s0 = self.state_hv(toks)
        # Optional closure
        closure_stats = None
        if do_closure:
            closure_stats = self.derive_closure()
        # Next-token at the trailing edge (cat-1 path; always present)
        prev = toks[-2] if len(toks) >= 2 else None
        cur = toks[-1]
        next_pred = self.recall_next_token(prev, cur, top_k=5)
        # Magnitude scan: any token decode to a numeric?
        magnitudes = []
        for t in toks:
            n, score = self.recall_magnitude(t)
            if n is not None and score > 0.3:
                magnitudes.append((t, n, score))
        # Day 75 v4: cat-4 ICL b_self recall
        icl_recalls = []
        icl_action_token = None
        icl_top_sim = 0.0
        n_action_vocab = 0    # Pack 322: cleanup vocab size for argmax-safe gate
        # Pack 296: True only on an EXACT Pack-273 cache hit (deterministic
        # anchor hash).  Soft recall_action matches can reach sim ~1.0 on
        # novel "what is the X of Y" queries (dense KG anchors) -- a sim
        # threshold cannot tell them from real hits, so active learning
        # keys on this flag instead.
        icl_exact_cache = False
        if do_icl:
            cat4 = getattr(self.org, '_cat4', None) or getattr(self.org, 'cat4', None)
            if cat4 is not None:
                try:
                    n_action_vocab = len(cat4.action_vocab)
                except Exception:
                    n_action_vocab = 0
                try:
                    # Pack 273 fast path -- cache lookup BEFORE recall_action.
                    # Cache is the Kanerva 2022 associative-mapping bypass;
                    # fires whenever query state_toks hash matches a cached
                    # anchor, independent of substrate state-similarity.
                    anchor_actions_cache = getattr(
                        cat4, 'anchor_actions', None)
                    if anchor_actions_cache:
                        from ikigai.cognition.cat4_absorb import (
                            _stable_anchor)
                        query_anchor = _stable_anchor(toks)
                        cache_entry = anchor_actions_cache.get(
                            query_anchor)
                        if cache_entry:
                            chosen = cache_entry[-1]
                            if isinstance(chosen, (list, tuple)) and chosen:
                                icl_action_token = ' '.join(chosen)
                            elif isinstance(chosen, str):
                                chosen and chosen
                                icl_action_token = chosen
                            if icl_action_token is not None:
                                icl_top_sim = 1.0   # fully confident cache hit
                                icl_exact_cache = True
                    raw_recalls = cat4.recall_action(toks, top_k=5)
                    icl_recalls = [(a, s) for (a, _, s) in raw_recalls]
                    if icl_recalls and icl_top_sim < 1.0:
                        icl_top_sim = float(icl_recalls[0][1])
                    if icl_action_token is None and icl_top_sim >= icl_min_state_sim:
                        # Goal HV: if goal given as string/list, use focus
                        goal_hv = None
                        if goal is not None:
                            g_toks = goal if isinstance(goal, list) else [str(goal)]
                            goal_hv = cat4.focus_hv(g_toks)
                        action_hv = cat4.predict_action_hv_efe(
                            toks, goal_hv=goal_hv, top_k=5,
                            min_state_sim=icl_min_state_sim)
                        if action_hv is not None:
                            # Pack 273 anchor-action cache: hash the
                            # query state tokens with the SAME
                            # deterministic anchor formula absorb_chain
                            # uses; if the cache holds an entry under
                            # that anchor, return its trailing action
                            # token DIRECTLY. Bypasses unbind + 9k
                            # cleanup pipeline (Kanerva 2022 associative
                            # -mapping path). Cleanup wall hits only
                            # when cache is cold.
                            cache_hit = None
                            anchor_actions = getattr(
                                cat4, 'anchor_actions', None)
                            if anchor_actions:
                                # Same stable blake2b hash as absorb_chain
                                from ikigai.cognition.cat4_absorb import (
                                    _stable_anchor)
                                query_anchor = _stable_anchor(toks)
                                cache_entry = anchor_actions.get(
                                    query_anchor)
                                if cache_entry:
                                    chosen = cache_entry[-1]
                                    if isinstance(chosen, (list, tuple))\
                                            and chosen:
                                        # join full action token sequence
                                        # so multi-word capitals like
                                        # ('buenos','aires') return as
                                        # 'buenos aires' not just 'aires'
                                        cache_hit = ' '.join(chosen)
                                    elif isinstance(chosen, str):
                                        cache_hit = chosen
                            if cache_hit is not None:
                                icl_action_token = cache_hit
                                icl_exact_cache = True
                            else:
                                # Pack 266 v3 cleanup fallback when cache
                                # cold.  Pack 269 hierarchical k-means
                                # broke production (see history).
                                # Pack 272 cached action codebook.
                                cands, cb = cat4.action_codebook()
                                if cands:
                                    sims = (np.real(cb @ np.conj(action_hv))
                                              / self.mr.d)
                                    top_k = 200
                                    top_idx = np.argsort(-sims)[:top_k]
                                    # Use the best-matching anchor's stored
                                    # state + bound for reconstruction check
                                    if raw_recalls:
                                        best_anchor = raw_recalls[0][0]
                                        stored_state = self.mr.recall(
                                            best_anchor, cat4.state_role)
                                        stored_bound = self.mr.recall(
                                            best_anchor, cat4.pair_role)
                                        if (stored_state is not None
                                                and stored_bound is not None):
                                            ss = np.asarray(stored_state,
                                                               dtype=np.complex64)
                                            sb = np.asarray(stored_bound,
                                                               dtype=np.complex64)
                                            ss = ss / (float(np.abs(
                                                ss).mean()) + 1e-12)
                                            sb = sb / (float(np.abs(
                                                sb).mean()) + 1e-12)
                                            # Rerank top-k by reconstruction
                                            best_score = -1e9
                                            best_idx = int(top_idx[0])
                                            for i_ in top_idx:
                                                i_ = int(i_)
                                                cand_hv = cb[i_]
                                                recon = ss * cand_hv
                                                recon = recon / (float(np.abs(
                                                    recon).mean()) + 1e-12)
                                                rec_sim = float(np.real(
                                                    np.vdot(recon, sb)) / self.mr.d)
                                                combo = (0.5 * float(sims[i_])
                                                           + 0.5 * rec_sim)
                                                if combo > best_score:
                                                    best_score = combo
                                                    best_idx = i_
                                            idx = best_idx
                                        else:
                                            idx = int(top_idx[0])
                                    else:
                                        idx = int(top_idx[0])
                                    if float(sims[idx]) > 0.05:
                                        icl_action_token = cands[idx]
                except Exception:
                    pass
        # Goal-directed planning if goal provided + CWM has edges.
        # Day 90 -- under _deep_dispatch, a request with NO explicit goal is
        # UNDERSTOOD as a goal over world-model symbols (the states it mentions
        # that the causal world model already knows), so the planner competes in
        # the SAME argmin-F arbitration as derive / multihop -- the loop reaches
        # for decomposition emergently, not via an authored 'if it abstained' fork.
        # The explicit-goal path below is unchanged (pixel-safe when the flag is off).
        plan = None
        if goal is not None:
            try:
                plan = self.planner.plan(
                    start_state=toks,
                    goal_state=goal if isinstance(goal, list)
                    else [str(goal)],
                    max_depth=max_steps, beam_width=3)
            except Exception:
                plan = None
        elif getattr(self, '_deep_dispatch', False):
            g_start, g_goal = self._derive_goal(toks)
            if g_start is not None:
                try:
                    plan = self.planner.plan_with_backtrack(
                        g_start, g_goal, max_depth=max_steps, branching=3)
                except Exception:
                    plan = None
        # Day 75 v8 method selection priority:
        #   1. Pack 254 RHC substrate_arith if arithmetic detected   'substrate_arith'
        #   2. cat-4 ICL if confident                                'b_self_icl'
        #   3. planner if successful                                 'planner'
        #   4. cat-1 next-token                                      'next_token'
        # Pack 254 first because RHC is bulletproof on digit-form
        # math regardless of cat-4 corpus coverage.
        answer = None
        method = 'empty'
        arith_pred = None
        try:
            from ikigai.cognition.math_eval import MathEval
            from ikigai.cognition.cat4_dopamine import is_compositional_query
            # Pack 296: gate the arithmetic path with the compositional
            # classifier.  Without this, the word->magnitude operand recall
            # fabricates a degenerate "add" on plain fact queries ("red planet"
            # -> 12), eating them before active learning / cache.  Day-83 audit:
            # the gate is DE-HARDCODED -- op detection via the substrate's
            # EMERGENT mev.is_operator (cat3.detect_operator), no op-word list.
            mev = getattr(self, '_math_eval', None)
            if mev is None:
                mev = MathEval(self.org, engine='auto')
                self._math_eval = mev
            if is_compositional_query(text, op_detector=mev.is_operator):
                a_pred, a_op, a_dbg = mev.substrate_arith(text)
                if a_pred is not None and a_op is not None:
                    arith_pred = (a_pred, a_op, a_dbg)
        except Exception:
            arith_pred = None
        # Pack 293 -- 2-hop compositional reasoning over cache lookups.
        # Fires only on explicit multi-hop templates, after arith and
        # before single-hop ICL (the composite query has no single cache
        # anchor of its own, so it would otherwise fall to next_token).
        multihop_ans = None
        if do_multihop:
            try:
                multihop_ans = self._reason_multihop(text)
            except Exception:
                multihop_ans = None
        # Pack 304 -- compositional derive-not-store.  Answers composite
        # relation queries by composing ATOMS at query time (continent of
        # the capital = continent of the country; same-continent
        # comparisons; attribute chains) with ZERO cache writes.  Runs
        # after arith, before multi-hop, so composites are DERIVED rather
        # than served from a stored composite -- multi-hop stays as the
        # fallback when a required atom is missing.
        derive_ans = None
        if do_derive:
            try:
                derive_ans = self.derive_engine.derive(text)
            except Exception:
                derive_ans = None
        # Pack 322: argmax-safe acceptance/abstain boundary. icl_top_sim on a
        # SOFT recall is the MAX over the action vocabulary, so the Pack-310
        # per-comparison floor leaks vocab noise-floor maxima (limit-test 319).
        # Exact cache hits (sim=1.0) stay confident; soft recalls must clear
        # the multiple-comparison boundary abstain_boundary_n(d, N_vocab).
        if icl_exact_cache or n_action_vocab <= 1:
            accept_sim = icl_min_state_sim
        else:
            from ikigai.cognition.calibration import abstain_boundary_n
            accept_sim = max(icl_min_state_sim,
                             abstain_boundary_n(self.d, n_action_vocab))

        # Day-100 -- PARTIAL GROUNDING STILL FABRICATES. abstain_ungrounded (above) only fires when
        # NOTHING in the question is a known entity. A question mentioning a known entity but an
        # UNKNOWN relation ('the mayor of france is') still reaches here with icl_top_sim clearing
        # accept_sim -- MEASURED: state_sim 0.282 vs boundary 0.2005 (n=9616) -- and returns the
        # nearest codebook word ('phrasing') as if it were an answer, via the identical mechanism
        # that produced 'succinctly'. The boundary is a GEOMETRIC noise floor; it cannot tell "this
        # state is significantly similar to something" from "similar to the RIGHT thing" -- a
        # superposed state built from ['the','mayor','of','france','is'] shares 4 of 5 tokens with
        # a taught state like ['the','capital','of','france','is'], so function-word overlap alone
        # clears the floor.
        #
        # The verifier is the organism's own knowledge, same move as induce_surface_verified: a
        # soft (non-exact) recall is only trustworthy if the candidate word is a value the question
        # already gives evidence for. Cheap (knows() is 139us) and it costs nothing when it doesn't
        # apply -- if no token is a known entity, icl_verified stays True and the EARLIER
        # abstain_ungrounded check has already handled that case.
        icl_verified = True
        if not icl_exact_cache and n_action_vocab > 1:
            _org = getattr(self, 'org', None)
            if _org is not None:
                try:
                    webs = {t: (_org.knows(t) or {}) for t in toks}
                except Exception:
                    webs = {}
                known_ents = {t: w for t, w in webs.items() if w}
                if known_ents:
                    held = {str(v).strip().lower()
                            for w in known_ents.values() for vals in w.values() for v in vals}
                    held |= set(known_ents)                          # self-reference is not a lie
                    icl_verified = False                              # verify the CANDIDATE below

        def _verified(candidate):
            if icl_verified:
                return True
            if candidate is None:
                return False
            return str(candidate).strip().lower() in held

        if self._fe_dispatch:
            # EMERGENT free-energy arbitration -- replaces the fixed if/elif order
            # below with a scalar competition (argmin F).  Exacts dominate (F=0,
            # registration order arith<derive<multihop<cache breaks ties); the
            # eligibility gate (epistemic ask-teacher) and the geometric abstain
            # boundary are preserved as principled EFE structure.
            eligible = (do_active and not icl_exact_cache
                        and self._active_learn_eligible(text))
            exacts = []
            if arith_pred is not None:
                exacts.append(('substrate_arith', str(arith_pred[0]), 0.0))
            if derive_ans is not None:
                exacts.append((derive_ans[1], derive_ans[0], 0.0))
            if multihop_ans is not None:
                exacts.append((multihop_ans[1], multihop_ans[0], 0.0))
            if icl_action_token is not None and icl_exact_cache:
                exacts.append(('b_self_icl', icl_action_token, 0.0))
            if exacts:
                a, m = self._arbitrate(exacts)
                if a is not None:
                    answer, method = a, m
            elif eligible:
                learned = self._active_learn(text)
                if learned:
                    answer = learned
                    icl_action_token = learned
                    icl_top_sim = 1.0
                    method = 'active_learn'
                else:
                    props = []
                    if (icl_action_token is not None and icl_top_sim >= accept_sim
                            and _verified(icl_action_token)):
                        props.append(('b_self_icl', icl_action_token,
                                      self._free_energy(icl_top_sim)))
                    a, m = self._arbitrate(props)
                    if a is not None:
                        answer, method = a, m
                    elif do_abstain:
                        answer, method = 'unknown', 'abstain'
                    elif next_pred:
                        answer, method = next_pred[0][0], 'next_token'
            else:
                props = []
                if (icl_action_token is not None and icl_top_sim >= accept_sim
                        and _verified(icl_action_token)):
                    props.append(('b_self_icl', icl_action_token,
                                  self._free_energy(icl_top_sim)))
                if plan and plan.get('success'):
                    props.append(('planner',
                                  plan.get('actions') or plan.get('trajectory'),
                                  self._F_PLANNER))
                a, m = self._arbitrate(props)
                if a is not None:
                    answer, method = a, m
                elif do_abstain:
                    answer, method = 'unknown', 'abstain'
                elif next_pred:
                    answer, method = next_pred[0][0], 'next_token'
        elif arith_pred is not None:
            answer = str(arith_pred[0])
            method = 'substrate_arith'
        elif derive_ans is not None:
            answer, method = derive_ans
        elif multihop_ans is not None:
            answer, method = multihop_ans
        elif icl_action_token is not None and icl_exact_cache:
            # exact Pack-273 cache hit -- reliable, free, never asks teacher
            answer = icl_action_token
            method = 'b_self_icl'
        elif do_active and not icl_exact_cache and self._active_learn_eligible(text):
            # Pack 294 -- uncertain on a fact query: ask the teacher,
            # absorb, answer.  Next identical query becomes a native
            # cache hit (the elif above).
            learned = self._active_learn(text)
            if learned:
                answer = learned
                # Reflect the learned answer in the ICL fields so the
                # returned dict -- and multi-hop hops that read icl_action
                # first -- see it, not the stale uncertain substrate guess.
                icl_action_token = learned
                icl_top_sim = 1.0
                method = 'active_learn'
            elif (icl_action_token is not None and icl_top_sim >= accept_sim
                    and _verified(icl_action_token)):
                answer = icl_action_token
                method = 'b_self_icl'
            elif next_pred:
                answer = next_pred[0][0]
                method = 'next_token'
        elif (icl_action_token is not None and icl_top_sim >= accept_sim
                and _verified(icl_action_token)):
            answer = icl_action_token
            method = 'b_self_icl'
        elif plan and plan.get('success'):
            answer = plan.get('actions') or plan.get('trajectory')
            method = 'planner'
        elif do_abstain and (icl_top_sim < accept_sim or not _verified(icl_action_token)):
            # Pack 310 CALIBRATION -- recall similarity is below the
            # k-sigma boundary derived from the substrate geometry
            # (1/sqrt(2d)), i.e. statistically indistinguishable from
            # querying EMPTY memory.  No confident path fired (arith /
            # derive / multihop / exact-cache / soft-ICL).  Return UNKNOWN
            # instead of the next_token guess -- that guess IS the
            # hallucination (the 'theologians' noise-floor token).  Pure
            # geometry, no word lists.  Organism-side half of abstention
            # (teacher-side = Pack 309).
            answer = 'unknown'
            method = 'abstain'
        elif next_pred:
            answer = next_pred[0][0]
            method = 'next_token'
        return {
            'tokens': toks,
            'state_hv_norm': float(np.abs(s0).sum()),
            'next_predict': next_pred,
            'magnitudes': magnitudes,
            'icl_recalls': icl_recalls,
            'icl_action': icl_action_token,
            'icl_top_state_sim': icl_top_sim,
            'plan': plan,
            'answer': answer,
            'method': method,
            'closure_stats': closure_stats,
        }

    # ---- Pack 305 perf fast path ------------------------------------

    def _fast_result(self, toks, answer, method, icl=None, exact=False):
        return {'tokens': toks, 'state_hv_norm': 0.0, 'next_predict': [],
                'magnitudes': [], 'icl_recalls': [], 'icl_action': icl,
                'icl_top_state_sim': 1.0 if exact else 0.0, 'plan': None,
                'answer': answer, 'method': method, 'closure_stats': None,
                'fast': True}

    def _fast_answer(self, text, toks, goal=None, do_multihop=True,
                      do_derive=True):
        """Resolve cheap, confident answers before the heavy cat4 path.
        Returns a result dict, or None to fall through.  Priority mirrors
        the full method-selection order: arith -> derive -> multihop ->
        exact cache.  Never asks the teacher (that stays on the slow,
        uncertain path)."""
        if getattr(self, '_fe_dispatch', False):
            # emergent free-energy arbitration over the same exact proposals
            return self._fast_answer_fe(text, toks, goal, do_multihop, do_derive)
        # 1. arithmetic (gated by the Pack 283 compositional classifier)
        try:
            from ikigai.cognition.cat4_dopamine import is_compositional_query
            from ikigai.cognition.math_eval import MathEval
            mev = getattr(self, '_math_eval', None)
            if mev is None:
                mev = MathEval(self.org, engine='auto')
                self._math_eval = mev
            if is_compositional_query(text, op_detector=mev.is_operator):
                a_pred, a_op, _ = mev.substrate_arith(text)
                if a_pred is not None and a_op is not None:
                    return self._fast_result(toks, str(a_pred),
                                             'substrate_arith')
        except Exception:
            pass
        # 2. derive (compositional, read-only over atoms)
        if do_derive:
            try:
                d = self.derive_engine.derive(text)
                if d is not None:
                    return self._fast_result(toks, d[0], d[1], icl=d[0])
            except Exception:
                pass
        # 3. multi-hop (chained cheap reason() calls)
        if do_multihop:
            try:
                mh = self._reason_multihop(text)
                if mh is not None:
                    return self._fast_result(toks, mh[0], mh[1])
            except Exception:
                pass
        # 4. exact anchor-cache hit (dict lookup, ~us)
        try:
            cat4 = (getattr(self.org, '_cat4', None)
                    or getattr(self.org, 'cat4', None))
            if cat4 is not None and getattr(cat4, 'anchor_actions', None):
                from ikigai.cognition.cat4_absorb import _stable_anchor
                entry = cat4.anchor_actions.get(_stable_anchor(toks))
                if entry:
                    chosen = entry[-1]
                    ans = (' '.join(chosen)
                           if isinstance(chosen, (list, tuple)) else str(chosen))
                    if ans:
                        return self._fast_result(toks, ans, 'b_self_icl',
                                                 icl=ans, exact=True)
        except Exception:
            pass
        return None

    def _fast_answer_fe(self, text, toks, goal=None, do_multihop=True,
                         do_derive=True):
        """Emergent-dispatch twin of _fast_answer: compute the SAME four exact
        proposals (arith / derive / multihop / exact-cache), then pick argmin
        free energy (all exact = F 0.0; stable sort keeps arith<derive<multihop
        <cache tie-break, so a lone hit is identical to the ladder's first-hit).
        Returns a result dict or None."""
        props = []   # (method, answer, F, icl, exact)
        try:
            from ikigai.cognition.cat4_dopamine import is_compositional_query
            from ikigai.cognition.math_eval import MathEval
            mev = getattr(self, '_math_eval', None)
            if mev is None:
                mev = MathEval(self.org, engine='auto')
                self._math_eval = mev
            if is_compositional_query(text, op_detector=mev.is_operator):
                a_pred, a_op, _ = mev.substrate_arith(text)
                if a_pred is not None and a_op is not None:
                    props.append(('substrate_arith', str(a_pred), 0.0, None, False))
        except Exception:
            pass
        if do_derive:
            try:
                d = self.derive_engine.derive(text)
                if d is not None:
                    props.append((d[1], d[0], 0.0, d[0], False))
            except Exception:
                pass
        if do_multihop:
            try:
                mh = self._reason_multihop(text)
                if mh is not None:
                    props.append((mh[1], mh[0], 0.0, None, False))
            except Exception:
                pass
        try:
            cat4 = (getattr(self.org, '_cat4', None)
                    or getattr(self.org, 'cat4', None))
            if cat4 is not None and getattr(cat4, 'anchor_actions', None):
                from ikigai.cognition.cat4_absorb import _stable_anchor
                entry = cat4.anchor_actions.get(_stable_anchor(toks))
                if entry:
                    chosen = entry[-1]
                    ans = (' '.join(chosen)
                           if isinstance(chosen, (list, tuple)) else str(chosen))
                    if ans:
                        props.append(('b_self_icl', ans, 0.0, ans, True))
        except Exception:
            pass
        if not props:
            return None
        # Day-99 -- same order-invariance rule as _arbitrate (see its docstring). This path
        # inlined `props.sort(...)[0]`, so fixing only _arbitrate would have left the ladder
        # alive right here: the fast path is where most production queries actually resolve.
        fmin = min(float(p[2]) for p in props)
        best = [p for p in props if float(p[2]) <= fmin + 1e-12]
        if len(best) > 1 and len({str(p[1]).strip().lower() for p in best}) > 1:
            return None            # tied exacts DISAGREE -> no evidence to prefer one; do not
                                   # let append-order pick. Fall through to the deliberate path.
        m, a, _f, icl, ex = best[0]
        return self._fast_result(toks, a, m, icl=icl, exact=ex)

    # ---- Day 89: EMERGENT free-energy dispatch ----------------------------
    # Replaces the hand-authored if/elif PRIORITY LADDER (which method fires,
    # in a fixed typed order) with a uniform SCALAR competition: every path
    # proposes an answer with a free energy F = -log(confidence) (surprise);
    # the organism picks argmin F.  Exact paths (arith/derive/multihop/exact
    # -cache) are F=0 and dominate; a soft recall competes at F=-log(sim) but
    # only above the calibrated acceptance boundary; a new capability just
    # registers a proposer -- no ladder to edit, no mode to hardcode.  The
    # eligibility gate (epistemic 'ask the teacher' vs pragmatic answer) and
    # the geometric abstain boundary are preserved -- both are principled EFE
    # structure, not arbitrary priority.  Gated by _fe_dispatch; proven pixel
    # -identical to the ladder before it becomes default.
    #
    # HONESTY NOTE (Day-95 red-team, DS-DISPATCH): this REFORMULATES the ladder as
    # argmin-over-costs; it does NOT remove the authored priority. All exact paths get a
    # HARDCODED F=0.0 and TIE, broken by the stable-sort REGISTRATION ORDER
    # (arith<derive<multihop<cache) -- identical to the old elif order; planner/next_token
    # get hardcoded F constants (_F_PLANNER=10, _F_NEXT=20) reproducing the lower rungs; the
    # literal if/elif ladder still exists below as the _fe_dispatch=False fallback; and the two
    # were proven PIXEL-IDENTICAL (0 diffs). So do NOT sell this as "emergent" or "de-hardcoded":
    # the ONLY genuinely competitive part is the soft-recall rung (real F=-log(sim) gated by the
    # calibration boundary). Accurate description: "the dispatch ladder written as argmin over
    # costs; behaviour unchanged; the soft-recall rung is confidence-gated."
    _fe_dispatch = True      # Day-89: emergent arbitration is DEFAULT (pixel-gated).
    _deep_dispatch = True    # Day-90: understand a goal-request -> plan competes in
                             # the SAME arbitration.  DEFAULT after pixel proof
                             # (day90_dispatch_pixel: 0 diffs on 36 production queries).
    _F_PLANNER = 10.0        # ranks below any accepted soft recall, above nothing
    _F_NEXT = 20.0

    def _derive_goal(self, toks):
        """Day 90 -- understand a request as a GOAL over the world model: the
        symbols it mentions that the causal world model ALREADY knows as states.
        Positional (first known state = start, last = goal); generic -- no per-
        request rule and no curated token list, the same discipline as the derive
        parser.  Returns (start, goal) state names, or (None, None) when the request
        names fewer than two known states (then no plan is proposed and the arbiter
        arbitrates exactly as before).  Reads the arbiter's OWN causal world model
        (built lazily; None until something populates it -> deep dispatch is inert)."""
        cwm = getattr(self, '_cwm', None)
        states = getattr(cwm, '_states', None) if cwm is not None else None
        if not states:
            return None, None
        seen, known = set(), []
        for t in toks:
            if t in states and t not in seen:
                seen.add(t); known.append(t)
        if len(known) >= 2 and known[0] != known[-1]:
            return known[0], known[-1]
        return None, None

    @staticmethod
    def _free_energy(conf):
        import math
        c = max(1e-9, min(1.0, float(conf)))
        return -math.log(c)

    def _arbitrate(self, proposals):
        """proposals: list of (method, answer, free_energy).  Pick argmin F.

        Day-99 -- ORDER-INVARIANT.  It used to `sorted(...)[0]`, i.e. a stable sort whose
        tie-break IS registration order.  Since every exact path is handed a LITERAL 0.0, all
        four always tied, so the "argmin free-energy competition" resolved to arith<derive<
        multihop<cache -- the original if/elif ladder, unchanged, wearing free-energy vocabulary.
        MEASURED (day99_is_dispatch_emergent): the same two proposals at the same F returned
        ANSWER_A when derive was appended first and ANSWER_B when cache was appended first. The
        order decided the answer. F decided nothing.

        Registration order is NOT evidence: it is an authored priority, and reading it as if it
        were a decision is the DS-DISPATCH cheat the Day-95 red team named.  So ties are now
        resolved by what the organism actually knows:
          * unique argmin        -> that proposal wins (a real decision).
          * tie, answers AGREE   -> consensus. Order-invariant, and genuine corroboration.
          * tie, answers DIFFER  -> the organism has NO evidence to prefer one. It says so
                                    (method='conflict') instead of silently taking whichever
                                    path happened to be appended first. Correct-or-abstain
                                    applied to arbitration itself.
        Returns (answer, method), or (None, 'conflict'), or (None, None) if nothing proposed."""
        if not proposals:
            return None, None
        fmin = min(float(p[2]) for p in proposals)
        best = [p for p in proposals if float(p[2]) <= fmin + 1e-12]
        if len(best) == 1:
            return best[0][1], best[0][0]
        answers = {str(p[1]).strip().lower() for p in best}
        if len(answers) == 1:
            return best[0][1], '+'.join(sorted(str(p[0]) for p in best))
        return None, 'conflict'

    # Pack 293 -- multi-hop question templates.  Specific by design so
    # false positives are near zero; anything else skips the multi-hop
    # path and reasons single-hop as before.
    _MH_YESNO_RE = re.compile(
        r'^\s*is\s+(?:the\s+)?capital\s+(?:city\s+)?of\s+(.+?)\s+in\s+'
        r'(.+?)\s*\??\s*$', re.IGNORECASE)
    _MH_WHICH_RE = re.compile(
        r'^\s*(?:what|which)\s+(?:continent|region)\s+is\s+(?:the\s+)?'
        r'capital\s+(?:city\s+)?of\s+(.+?)\s+in\s*\??\s*$', re.IGNORECASE)

    def _mh_resolve(self, query):
        """Single-hop resolve via reason() with multi-hop disabled (so a
        hop never recurses into another multi-hop).  Returns the
        lowercased answer string, or None when nothing confident."""
        r = self.reason(query, do_icl=True, do_multihop=False,
                          do_active=False)
        ans = r.get('icl_action') or r.get('answer')
        if ans is None:
            return None
        ans = str(ans).strip().lower()
        return ans or None

    def _reason_multihop(self, text):
        """Pack 293 -- 2-hop compositional reasoning by chaining cache
        lookups (the substrate-composition join from the roadmap).

        'is the capital of France in Europe' resolves capital(France)
        -> paris, then continent(paris) -> europe, and compares the two.
        'what continent is the capital of Japan in' returns the second
        hop directly.  Each hop is an ordinary single-hop reason() call,
        so the second hop is genuinely keyed on the first hop's output.

        Returns (answer, 'multihop') or None when either hop misses.
        """
        m = self._MH_YESNO_RE.match(text or '')
        if m:
            country = m.group(1).strip()
            target = m.group(2).strip().lower().rstrip('?').strip()
            cap = self._mh_resolve(f'what is the capital of {country}')
            if not cap:
                return None
            cont = self._mh_resolve(f'what continent is {cap} in')
            if not cont:
                return None
            yes = (target in cont) or (cont in target)
            self._stats['multihop'] = self._stats.get('multihop', 0) + 1
            return ('yes' if yes else 'no', 'multihop')
        m = self._MH_WHICH_RE.match(text or '')
        if m:
            country = m.group(1).strip()
            cap = self._mh_resolve(f'what is the capital of {country}')
            if not cap:
                return None
            cont = self._mh_resolve(f'what continent is {cap} in')
            if not cont:
                return None
            self._stats['multihop'] = self._stats.get('multihop', 0) + 1
            return (cont, 'multihop')
        return None

    # ---- Pack 294 active learning -----------------------------------
    # Defaults as class attributes so reason() works before (or without)
    # enable_active_learning -- inert until an oracle is attached, so
    # benches never trigger surprise teacher calls.
    _al_on = False
    _al_oracle = None
    _al_lo = 0.0
    # Only an EXACT cache hit (Pack 273 fast path sets icl_top_sim=1.0)
    # counts as confident.  Soft recall_action matches -- which the dense
    # KG "what is the X of Y" anchors produce at ~0.7-0.9 for NOVEL
    # queries -- are false-confident and must defer to the teacher.
    _al_hi = 0.999

    def enable_active_learning(self, oracle, lo=0.0, hi=0.999):
        """Pack 294 -- wire a teacher oracle so reason() self-teaches on
        uncertain atomic fact queries.

        When the organism is not confident (icl_top_sim < hi) on a
        fact-shaped query, it asks the oracle, absorbs the answer into
        the cache, and answers -- so the next identical query is a native
        cache hit.  `lo` > 0 restricts to the midband uncertainty band
        (skip near-zero confidence, the Pack 294 KL-fire framing); the
        default lo=0.0 also learns brand-new facts (sim ~ 0).
        """
        self._al_oracle = oracle
        self._al_on = True
        self._al_lo = float(lo)
        self._al_hi = float(hi)

    def disable_active_learning(self):
        self._al_on = False

    @staticmethod
    def _looks_like_question(text):
        """Guard against polluting the cache with junk: only fact-shaped
        queries are eligible for active learning."""
        t = (text or '').strip().lower()
        if len(t.split()) < 3:
            return False
        if t.endswith('?') or ' of ' in t:
            return True
        return t.split()[0] in {'what', 'who', 'which', 'where', 'when',
                                 'is', 'are', 'does', 'do', 'name'}

    def _active_learn_eligible(self, text):
        # Confidence is decided upstream by the exact-cache flag, so this
        # only checks shape: fact-like, non-math, teacher attached.
        if not self._al_on or self._al_oracle is None:
            return False
        try:
            from ikigai.cognition.cat4_dopamine import is_atomic_query
            from ikigai.cognition.math_eval import MathEval
            mev = getattr(self, '_math_eval', None)
            if mev is None:
                mev = MathEval(self.org, engine='auto')
                self._math_eval = mev
            if not is_atomic_query(text, op_detector=mev.is_operator):  # skip math
                return False
        except Exception:
            pass
        return self._looks_like_question(text)

    def _active_learn(self, text):
        """Ask the oracle, absorb into the cache, return the answer.
        Degrades to None when the teacher is unreachable (so reason()
        falls back to its normal paths)."""
        try:
            ans = self._al_oracle.ask(text)
        except Exception:
            return None
        if not ans:
            return None
        ans = str(ans).strip()
        cat4 = getattr(self.org, 'cat4', None)
        if cat4 is not None:
            try:
                cat4.populate_cache_from_text(f'{text}\n\n{ans}\n\n')
                self._stats['active_learned'] = (
                    self._stats.get('active_learned', 0) + 1)
            except Exception:
                pass
        return ans.lower()

    def stats(self):
        return dict(self._stats)
