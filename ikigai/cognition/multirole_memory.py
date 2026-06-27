"""
ikigai.cognition.multirole_memory -- One flat substrate, every channel.

Day 57 Pack 117. The unifier.

VSA role-binding stores different relation types in the SAME fixed counter
bank without interference:

    addr(word, role) = key(word) (*) ROLE      # (*) = Hadamard (complex mult)

Different roles -> orthogonal addresses (since for unit phasors
<x*a, x*b> = <a,b>, and random ROLEs are ~orthogonal) -> different hard
locations -> zero crosstalk between channels. So co-occurrence, IS-A,
sensory, property ALL live in one substrate. The {word: HV} dict can die.

Bonus: cross-channel composition. property(isa(cat)) retrieves "warm" by
chaining two role queries through the same substrate -- the binding the
separate-channel organism never had.

bind(x, r)   = x * r            (r unit phasor)
unbind(c, r) = c * conj(r)      (recovers, r*conj(r) = 1)

Recall is reconstructive + cleaned up against a candidate vocabulary
(resonator-style): read the noisy superposed value, pick the nearest
computed key.
"""

import numpy as np

from ikigai.cognition.flat_memory import (
    ComputedKey, VSASDM, tokenize, _renorm, _cos,
)


class MultiRoleMemory:
    """
    Unified flat memory across relation types via role-binding.

    relate(word, role, target)       store word --role--> target (a word)
    write_relation(word, role, hv)   store an arbitrary value vector
    recall(word, role)               raw reconstructive read
    query(word, role, candidates)    recall + cleanup -> best candidate word
    expose_cooccur(text)             Channel-1 co-occurrence under 'cooccur' role
    similarity(w1, w2)               mean-removed co-occurrence cosine
    """

    DEFAULT_ROLES = ('cooccur', 'next', 'next2', 'next3',
                     'isa', 'sensory', 'property', 'verb', 'class',
                     'episode', 'affordance', 'mod', 'antonym', 'concept',
                     # Pack 195 (Day 64) -- richer semantic relations
                     'similar', 'cause', 'effect', 'qa', 'context',
                     'definition', 'before', 'after', 'refers_to',
                     # Pack 199 NEW2 -- syntax-tree positional binding.
                     # token at position i in a sentence beginning with X is
                     # written as relate(X, f'pos_{i}', token_i).
                     'pos_0', 'pos_1', 'pos_2', 'pos_3',
                     'pos_4', 'pos_5', 'pos_6', 'pos_7',
                     # Pack 331 -- the ASK role: a question cue token binds to
                     # the relation that question is asking for (question ->
                     # relation, the inverse of relation -> question template).
                     # Learned from data; never hardcoded.
                     'ask')

    def __init__(self, d=512, M=16384, k=64, seed=114, window=3, remove_r=1,
                 svd_sample=2000, roles=DEFAULT_ROLES, M_rel=8192,
                 consolidate_every=0, q_omega=0.05, q_seed=12345,
                 bank_assignment=None):
        self.d = int(d); self.window = int(window)
        self.remove_r = int(remove_r); self.svd_sample = int(svd_sample)
        # Pack 242: persist the init seed (drives ComputedKey + role HVs +
        # legacy bank seed). Distinct from per-bank seeds in multi-bank mode.
        self._init_seed = int(seed)
        self.ck = ComputedKey(d=d, seed=seed)
        # Pack 241 multi-bank mode (Day 68, 2026-06-08):
        #   bank_assignment = {bank_id: {'M': int, 'seed': int (optional),
        #                                  'roles': [role names]}}
        #   When set: build N separate banks, one per bank_id; route each role
        #   to its bank. sdm/sdm_rel become aliases for back-compat.
        # When None: legacy 2-bank mode (Pack 117 traffic-class split):
        #   cooccur+dense -> sdm; isa/property/etc -> sdm_rel.
        self._bank_assignment = (dict(bank_assignment)
                                  if bank_assignment is not None else None)
        if self._bank_assignment is not None:
            self.banks = {}
            self._role_to_bank = {}
            for bank_id, cfg in self._bank_assignment.items():
                bank_M = int(cfg.get('M', M))
                bank_seed = int(cfg.get('seed',
                                          seed + (abs(hash(bank_id)) % 10000)))
                self.banks[bank_id] = VSASDM(
                    d=d, M=bank_M, k=k, seed=bank_seed,
                    consolidate_every=consolidate_every)
                for role_name in cfg.get('roles', []):
                    self._role_to_bank[role_name] = bank_id
            # Pick default bank for unassigned roles (first declared bank).
            self._default_bank_id = next(iter(self._bank_assignment))
            # Back-compat aliases: sdm + sdm_rel still required by some code
            # paths. Alias to first two banks (sdm = lang/dense bank;
            # sdm_rel = next bank if any).
            _bank_list = list(self.banks.values())
            self.sdm = _bank_list[0]
            self.sdm_rel = _bank_list[1] if len(_bank_list) > 1 else _bank_list[0]
        else:
            # Legacy 2-bank mode (Pack 117 finding).
            # TWO traffic classes in separate fixed banks:
            #   cooccur = high traffic (millions of writes) -> own bank
            #   relational (isa/sensory/property/verb) = sparse, equal low
            #   traffic; role-binding separates them WITHIN one shared
            #   relational bank. A shared bank across traffic classes lets
            #   co-occurrence mass (~2000) swamp a single relational write
            #   (~22). Separating by traffic fixes it.
            #   Online consolidation on cooccur bank only -- relational bank
            #   must preserve write-count reinforcement.
            self.sdm = VSASDM(d=d, M=M, k=k, seed=seed,
                              consolidate_every=consolidate_every)
            self.sdm_rel = VSASDM(d=d, M=M_rel, k=k, seed=seed + 1)
            self.banks = None
            self._role_to_bank = None
            self._default_bank_id = None
        # Build role HVs. Include any extra roles required by bank_assignment.
        all_roles = set(roles)
        if self._bank_assignment is not None:
            for cfg in self._bank_assignment.values():
                all_roles.update(cfg.get('roles', []))
        rng = np.random.default_rng(seed + 999)
        self.roles = {}
        for name in sorted(all_roles):
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            self.roles[name] = np.exp(1j * ph).astype(np.complex64)
        self._seen = set()        # words with any relation
        self._cooccur_seen = set()
        self._role_targets = {}   # role -> set of target words seen (cleanup candidates)
        self._verb_seen = set()
        self._dirs = None
        self._dirty = True
        # Pack 197: optional frame conditioning. When set, every (word, role)
        # binding is augmented with the frame HV via Hadamard product.
        # current_frame_hv = None -> identical to pre-Pack-197 behavior.
        self.current_frame_hv = None
        # Pack 197 surprise gate: per-token unigram count for write strength.
        self._unigram_count = {}
        self._unigram_total = 0
        # Pack 199 P1: bigram counter for transition surprise.
        self._bigram_count = {}
        self._bigram_total = 0
        # Reference to a FrameField if IkigaiOrganism attaches one (Pack 197).
        self._frame_field_ref = None
        # Loaded frame state pending IkigaiOrganism rehydrate (Pack 197).
        self._pending_frame_state = None
        # Quantity-axis for verb-rotor encoding (Channel 2). Phase encodes a scalar.
        self.q_omega = float(q_omega)
        rng_axis = np.random.default_rng(q_seed)
        self.q_axis = (rng_axis.integers(0, 2, size=self.d) * 2 - 1).astype(np.float32)
        # Pack 247w (Day 72) abductive role induction. Buffer unknown relation
        # labels; mint phasor + bank-route when support recurs. Volatile buffer:
        # sub-threshold triples drop on save by design.
        from ikigai.cognition.role_minter import RoleMinter
        self.role_minter = RoleMinter(self)
        # Pack 247z-CM (Day 72) read-time common-mode HV cache per role.
        # Populated lazily via build_v_common(role) or eagerly via
        # bake_v_common_all(). Persisted in save_ikg.
        self._v_common = {}

    # Roles partition by TRAFFIC CLASS:
    #   dense (high traffic, every token):     cooccur, next
    #   sparse (low traffic, explicit facts):  isa, sensory, property, verb
    # Sparse signal would drown in the dense bank, so they get their own.
    DENSE_ROLES = {'cooccur', 'next', 'next2', 'next3'}

    def _bank(self, role):
        # Pack 241 multi-bank routing.
        if self._bank_assignment is not None:
            bank_id = self._role_to_bank.get(role, self._default_bank_id)
            return self.banks[bank_id]
        # Legacy 2-bank traffic-class routing.
        return self.sdm if role in self.DENSE_ROLES else self.sdm_rel

    def bank_for_role(self, role):
        """Return the bank id (string in multi-bank mode) or 'sdm' / 'sdm_rel'
        in legacy mode for a given role. Introspection helper."""
        if self._bank_assignment is not None:
            return self._role_to_bank.get(role, self._default_bank_id)
        return 'sdm' if role in self.DENSE_ROLES else 'sdm_rel'

    # ── binding ──────────────────────────────────────────────────────────
    def _bind(self, a, r):
        return (a * r).astype(np.complex64)

    def _unbind(self, c, r):
        return (c * np.conj(r)).astype(np.complex64)

    def _addr(self, word, role):
        rolev = self.roles[role]
        # Pack 197: frame conditioning. addr = key(word) * (frame_hv * role_hv).
        if self.current_frame_hv is not None:
            rolev = self._bind(self.current_frame_hv, rolev)
        return self._bind(self.ck.key(word), rolev)

    # Pack 197 frame helpers
    def set_frame(self, frame_hv, frame_tag=None):
        """Activate a frame for subsequent writes/reads.
        `frame_tag` is a short string id (e.g. 'f3'). Used in slot-cache key
        so loc-cache stays frame-distinct.
        """
        if frame_hv is None:
            self.current_frame_hv = None
            self._current_frame_tag = None
            return
        self.current_frame_hv = frame_hv.astype(np.complex64)
        if frame_tag is None:
            # fallback: hash of HV bytes (stable across runs for same hv)
            frame_tag = f'h{abs(hash(self.current_frame_hv.tobytes())) % 100000}'
        self._current_frame_tag = str(frame_tag)

    def clear_frame(self):
        self.current_frame_hv = None
        self._current_frame_tag = None

    class _FrameContext:
        def __init__(self, mr, frame_hv, frame_tag=None):
            self.mr = mr; self.frame_hv = frame_hv
            self.frame_tag = frame_tag
            self.prev_hv = None; self.prev_tag = None
        def __enter__(self):
            self.prev_hv = self.mr.current_frame_hv
            self.prev_tag = getattr(self.mr, '_current_frame_tag', None)
            self.mr.set_frame(self.frame_hv, frame_tag=self.frame_tag)
            return self.mr
        def __exit__(self, exc_type, exc, tb):
            # Restore BOTH hv and tag (Pack 247y fix: was leaking tag)
            self.mr.current_frame_hv = self.prev_hv
            self.mr._current_frame_tag = self.prev_tag

    def in_frame(self, frame_hv, frame_tag=None):
        """Context manager: `with mr.in_frame(hv, tag): ...`"""
        return MultiRoleMemory._FrameContext(self, frame_hv, frame_tag)

    # Pack 197 surprise gate
    def observe_unigrams(self, tokens):
        """Update unigram counters (drives surprise-gated write_strength)."""
        for t in tokens:
            self._unigram_count[t] = self._unigram_count.get(t, 0) + 1
            self._unigram_total += 1

    def write_strength(self, token):
        """Pack 197: unigram surprise -> write strength [0.05, 1.0].
        Frequent tokens -> ~0.05x. Rare -> ~1.0x.
        """
        c = self._unigram_count.get(token, 0)
        N = max(self._unigram_total, 1)
        if c <= 0 or N <= 1:
            return 1.0
        import math
        if N == c:
            return 0.05
        val = math.log(N / max(c, 1)) / math.log(max(N, 2))
        return max(0.05, min(1.0, float(val)))

    def transition_write_strength(self, prev, curr):
        """Pack 199 P1: bigram-conditional surprise -> write strength.
        Combines unigram surprise of curr with bigram-conditional surprise:
            surprise(curr | prev) = -log P(curr | prev)
        For frequent transitions like (the, the) or (of, the), this drops to
        near-zero. For novel transitions, it stays near 1.0.
        """
        ws_uni = self.write_strength(curr)
        bcount = self._bigram_count.get((prev, curr), 0)
        pcount = self._unigram_count.get(prev, 0)
        if pcount <= 0 or bcount <= 0:
            return ws_uni
        # P(curr | prev) = bcount / pcount.  Higher P -> lower write strength.
        # Map P in [0,1] -> [1.0, 0.05] via 1 - 0.95 * P (clamp).
        p_cond = bcount / max(pcount, 1)
        ws_bi = max(0.05, 1.0 - 0.95 * float(p_cond))
        return min(ws_uni, ws_bi)

    # Pack 199 NEW3 -- predictive-coding write strength via substrate query
    def predictive_write_strength(self, prev, curr, role='next', top_k=8):
        """Active-inference: predict curr from prev via current substrate.
        If the substrate already correctly predicts curr (low surprise),
        write_strength drops. If the substrate is WRONG (high surprise),
        write_strength is full -- this is where learning matters.

        Returns scalar in [0.05, 1.0]. Pairs with transition_write_strength
        (count-based) for a richer surprise signal.
        """
        cands = self._role_targets.get(role, None)
        if not cands or curr not in cands:
            return 1.0
        try:
            pred, _ = self.query(prev, role, candidates=cands,
                                   hopfield_iter=0, belief_field=False)
        except Exception:
            return 1.0
        if pred == curr:
            return 0.1   # already known -- low info, weak write
        return 1.0       # wrong -> learn

    # Pack 199 NEW1 -- grammar-state queries (mirror FrameField for parity)
    def next_frame_distribution(self, frame_field, prev_token):
        """Given a previous token, return softmax over likely next-frames
        using FrameField.frame_bigram. Glue for cogitate-time filtering."""
        if frame_field is None:
            return None
        pf = frame_field.frame_of_word(prev_token)
        return frame_field.next_frame_probs(pf)

    def _slot(self, word, role):
        # Pack 197: include frame tag in slot so the loc cache returns
        # frame-distinct locations. Without this, the cache short-circuits
        # the frame-conditioned address and all frames map to the same loc.
        tag = getattr(self, '_current_frame_tag', None)
        if tag:
            return f'{word}\x00{role}\x00{tag}'
        return f'{word}\x00{role}'

    # ── writing ──────────────────────────────────────────────────────────
    def write_relation(self, word, role, value_hv):
        self._bank(role).write(self._addr(word, role),
                               np.asarray(value_hv, dtype=np.complex64),
                               word=self._slot(word, role))
        self._seen.add(word)

    def unrelate(self, word, role, target_word):
        """
        Inverse of relate: subtracts the bound target's HV from this
        (word, role) address. Pair with the same N reinforcement count
        used in the original relate calls to fully neutralise a fact.
        Kill Stack #4 -- reversible substrate writes.
        """
        if role not in self.roles:
            return
        addr = self._addr(word, role)
        value = (-self.ck.key(target_word)).astype(np.complex64)
        self._bank(role).write(addr, value, word=self._slot(word, role))
        # role_targets stays -- "this word was associated with this role
        # at some point" is a separate fact from "target_word is the answer".

    def ensure_role(self, name):
        """Mint a phasor HV for `name` if the role doesn't exist yet, so new
        relational channels (e.g. Pack 331 'ask') can be added at runtime on a
        loaded organism without a re-init.  Idempotent."""
        if name in self.roles:
            return
        rng = np.random.default_rng(self._init_seed + 999 + (abs(hash(name)) % 100000))
        ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        self.roles[name] = np.exp(1j * ph).astype(np.complex64)

    def relate(self, word, role, target_word):
        """word --role--> target_word  (stores the target's computed key)."""
        self.ensure_role(role)
        self.write_relation(word, role, self.ck.key(target_word))
        self._role_targets.setdefault(role, set()).add(target_word)

    def targets(self, role):
        """Cleanup-candidate words ever stored as targets of this role."""
        return self._role_targets.get(role, set())

    def expose_cooccur(self, text):
        """Channel-1 co-occurrence, written under the 'cooccur' role."""
        tokens = tokenize(text)
        if not tokens:
            return 0
        n = len(tokens)
        K = np.stack([self.ck.key(t) for t in tokens])
        w = self.window
        P = np.empty((n + 1, self.d), dtype=np.complex128)
        P[0] = 0
        P[1:] = np.cumsum(K.astype(np.complex128), axis=0)
        agg = {}; order = []
        for i in range(n):
            lo = i - w if i - w > 0 else 0
            hi = i + w + 1 if i + w + 1 < n else n
            ctx = (P[hi] - P[lo]) - K[i]
            t = tokens[i]
            if t in agg:
                agg[t] = agg[t] + ctx
            else:
                agg[t] = ctx; order.append(t)
                self._seen.add(t); self._cooccur_seen.add(t)
        # Pack 197 frame conditioning: use _addr() so current_frame_hv applies.
        ukeys = np.stack([self._addr(t, 'cooccur') for t in order])
        slots = [self._slot(t, 'cooccur') for t in order]
        # Pack 241: route via _bank('cooccur') so multi-bank assignment works.
        cooccur_bank = self._bank('cooccur')
        locs = cooccur_bank.locs_batch(ukeys, slots)
        # Pack 197 surprise gate: scale each token's write by its unigram surprise.
        # Frequent tokens (the/of/and) get ~0.05x writes; rare tokens get ~1.0x.
        # Update unigram counter for downstream tokens.
        self.observe_unigrams(tokens)
        for t, idx in zip(order, locs):
            ws = self.write_strength(t)
            cooccur_bank.C[idx] += (ws * agg[t]).astype(np.complex64)
        self._dirty = True
        return n

    # ── Channel 2 (verb rotor) on flat memory ──────────────────────────────
    def _encode_scalar(self, c):
        """Scalar c -> phasor: phase = c * omega * q_axis (per component)."""
        phase = float(c) * self.q_omega * self.q_axis
        return np.exp(1j * phase).astype(np.complex64)

    def _decode_scalar(self, hv):
        """Phasor -> scalar (median of phase/omega/q_axis across components)."""
        phases = np.angle(hv).astype(np.float32)
        cs = phases / (self.q_omega * self.q_axis + 1e-9)
        return float(np.median(cs))

    def expose_verb_observation(self, verb, c_est):
        """
        Channel-2 write: encode one (verb, c_est) observation as a phasor
        and superpose into (verb, 'verb') in the relational bank. Many obs
        accumulate -> decode of recall yields the mean coefficient.
        """
        self.write_relation(verb, 'verb', self._encode_scalar(c_est))
        self._verb_seen.add(verb)

    def predict_verb_coefficient(self, verb):
        """Recall (verb,'verb'), decode -> learned coefficient. None if unseen."""
        if verb not in self._verb_seen:
            return None
        hv = self.recall(verb, 'verb')
        return self._decode_scalar(hv)

    def _ngram_ctx_hv(self, ctx_tokens):
        """
        Build ordered n-gram context HV via permute-then-bind (Pack 136).
        ctx_tokens listed oldest-first.  Last token gets shift 0; earlier
        tokens get cyclic shifts of (n-1-i). Permutation preserves order;
        Hadamard binding combines into one address. Distinct n-grams ->
        distinct addresses regardless of token overlap.
        """
        n = len(ctx_tokens)
        if n == 0:
            return np.ones(self.d, dtype=np.complex64)
        accum = None
        for i, t in enumerate(ctx_tokens):
            shift = (n - 1) - i
            k = self.ck.key(t)
            if shift > 0:
                k = np.roll(k, shift).astype(np.complex64)
            accum = k if accum is None else (accum * k).astype(np.complex64)
        return accum

    # Pack 199 NEW2 -- syntax-tree positional binding
    def expose_syntax_tree(self, text, max_pos=8):
        """Write sentence-shape positional bindings.
        For sentence S = [t_0, t_1, ..., t_n], write:
            relate(t_0, 'pos_i', t_i)  for i = 1..min(n, max_pos-1)
        At query time, query(first_token, 'pos_3') returns the most-recalled
        word at position 3 in sentences starting with first_token.
        This is a degenerate syntax tree -- positional list -- but gives
        sentence-structure recall without POS labels.
        """
        tokens = tokenize(text)
        if len(tokens) < 2:
            return 0
        first = tokens[0]
        written = 0
        n_pos = int(min(len(tokens), max_pos))
        for i in range(1, n_pos):
            role = f'pos_{i}'
            if role not in self.roles:
                continue
            self.relate(first, role, tokens[i])
            written += 1
        return written

    def expose_transitions(self, text):
        """
        Write n-gram transitions (Pack 120 bigram + Pack 136 trigram + 4-gram).
        Bigram under 'next', trigram under 'next2', 4-gram under 'next3'.
        All go to the dense bank.
        """
        tokens = tokenize(text)
        if len(tokens) < 2:
            return 0
        for n_ctx, role in [(1, 'next'), (2, 'next2'), (3, 'next3')]:
            if len(tokens) < n_ctx + 1:
                continue
            self._expose_ngram_role(tokens, n_ctx, role)
        return len(tokens) - 1

    # ── Pack 147: native multi-channel meaning exposure ────────────────────
    def expose_episode(self, text):
        """
        Bind a sentence-HV (bundled token keys) to each content token under
        the 'episode' role. Per-word recall later returns a "where have I
        seen this used" gist. Written to the relational bank.
        """
        tokens = tokenize(text)
        if len(tokens) < 2:
            return 0
        # sentence HV = renorm sum of token keys (order-free gist)
        accum = np.zeros(self.d, dtype=np.complex64)
        for t in tokens:
            accum = accum + self.ck.key(t)
        mags = np.abs(accum)
        mags = np.where(mags > 1e-9, mags, 1.0)
        shv = (accum / mags).astype(np.complex64)
        # bind under episode role for each token
        rolev = self.roles['episode']
        bank = self._bank('episode')
        written = 0
        for t in set(tokens):
            addr = self._bind(self.ck.key(t), rolev)
            bank.write(addr, shv)
            self._role_targets.setdefault('episode', set()).add(t)
            self._seen.add(t)
            written += 1
        return written

    def expose_affordance(self, subj, verb, obj=None):
        """
        Write a verb affordance: (subj does verb) and optionally (verb does
        obj). Goes to the 'affordance' role in the relational bank. Simple
        wrapper so callers don't have to know the role name.
        """
        n = 0
        if subj and verb:
            self.relate(subj, 'affordance', verb)
            self._role_targets.setdefault('affordance', set()).add(subj)
            n += 1
        if verb and obj:
            self.relate(verb, 'affordance', obj)
            self._role_targets.setdefault('affordance', set()).add(verb)
            n += 1
        return n

    # ── Pack 195 (Day 64) -- additional semantic exposure helpers ────────
    def expose_qa(self, question_text, answer_text):
        """Bind a sentence-HV of question to the bag of answer tokens
        under the 'qa' role. Lets recall: query(q_word, 'qa') -> answer
        token cluster. Cheap, no parsing required."""
        q_tokens = tokenize(question_text)
        a_tokens = tokenize(answer_text)
        if not q_tokens or not a_tokens:
            return 0
        # episode-style sentence HV of the question
        accum = np.zeros(self.d, dtype=np.complex64)
        for t in q_tokens:
            accum = accum + self.ck.key(t)
        mags = np.abs(accum)
        mags = np.where(mags > 1e-9, mags, 1.0)
        q_hv = (accum / mags).astype(np.complex64)
        rolev = self.roles['qa']
        bank = self._bank('qa')
        # bind question-HV under qa role, value = bundled answer keys
        a_bundle = np.zeros(self.d, dtype=np.complex64)
        for t in a_tokens:
            a_bundle = a_bundle + self.ck.key(t)
        a_mags = np.abs(a_bundle)
        a_mags = np.where(a_mags > 1e-9, a_mags, 1.0)
        a_hv = (a_bundle / a_mags).astype(np.complex64)
        addr = self._bind(q_hv, rolev)
        bank.write(addr, a_hv)
        # also bind each Q token to first answer token under 'qa' for
        # token-level recall
        first_a = a_tokens[0]
        for q in set(q_tokens):
            self.relate(q, 'qa', first_a)
            self._role_targets.setdefault('qa', set()).add(q)
        return len(q_tokens)

    def expose_cause(self, cause_text, effect_text):
        """Write cause -> effect binding under 'cause' role.
        Tokens of cause text bound to first content token of effect."""
        c_tokens = tokenize(cause_text)
        e_tokens = tokenize(effect_text)
        if not c_tokens or not e_tokens:
            return 0
        target = e_tokens[0]
        for c in set(c_tokens):
            self.relate(c, 'cause', target)
            self._role_targets.setdefault('cause', set()).add(c)
        # reverse: effect -> cause under 'effect'
        rev_target = c_tokens[0]
        for e in set(e_tokens):
            self.relate(e, 'effect', rev_target)
        return len(c_tokens)

    def expose_definition(self, term, definition_text):
        """Bind term to its definition tokens under 'definition' role."""
        d_tokens = tokenize(definition_text)
        if not term or not d_tokens:
            return 0
        # bundle definition tokens, bind under 'definition'
        bundle = np.zeros(self.d, dtype=np.complex64)
        for t in d_tokens:
            bundle = bundle + self.ck.key(t)
        mags = np.abs(bundle)
        mags = np.where(mags > 1e-9, mags, 1.0)
        d_hv = (bundle / mags).astype(np.complex64)
        rolev = self.roles['definition']
        addr = self._bind(self.ck.key(term), rolev)
        self._bank('definition').write(addr, d_hv)
        # also bind first def-token via relate for cleanup
        self.relate(term, 'definition', d_tokens[0])
        self._role_targets.setdefault('definition', set()).add(term)
        return len(d_tokens)

    def expose_before_after(self, prev_clause, next_clause):
        """Bind temporal order: prev -> next under 'before', next -> prev
        under 'after'. Pattern detector for narrative."""
        p_tokens = tokenize(prev_clause)
        n_tokens = tokenize(next_clause)
        if not p_tokens or not n_tokens:
            return 0
        for p in set(p_tokens):
            self.relate(p, 'before', n_tokens[0])
        for n in set(n_tokens):
            self.relate(n, 'after', p_tokens[0])
        return len(p_tokens) + len(n_tokens)

    def expose_modifier(self, modifier, noun):
        """
        Write that `modifier` was observed as a descriptor of `noun` under
        the 'mod' role.
        """
        if not modifier or not noun:
            return 0
        self.relate(noun, 'mod', modifier)
        self._role_targets.setdefault('mod', set()).add(noun)
        return 1

    def expose_meaning(self, text, pos_classifier=None,
                       subj_vocab=None, verb_vocab=None,
                       obj_vocab=None, adj_vocab=None):
        """
        Single-call meaning exposure: episode + affordance + modifier
        in one pass.

        If `pos_classifier` is a callable returning a POS-tag dict per
        token, we use it for SVO + adjective extraction. Otherwise we
        fall back to the explicit *_vocab sets (Pack 147 convention) or
        skip the affordance / modifier writes if no info is available.

        Returns dict of counts written per channel.
        """
        tokens = tokenize(text)
        out = {'episode': 0, 'affordance': 0, 'modifier': 0}
        out['episode'] = self.expose_episode(text)
        # SVO + adj extraction
        if pos_classifier is not None:
            tags = pos_classifier(tokens)
            nouns = [t for t in tokens if tags.get(t, '').startswith('N')]
            verbs = [t for t in tokens if tags.get(t, '').startswith('V')]
            adjs  = [t for t in tokens if tags.get(t, '').startswith('J')]
            subj = nouns[0] if nouns else None
            verb = verbs[0] if verbs else None
            obj  = nouns[1] if len(nouns) > 1 else None
        else:
            # vocab-set fallback
            subj_vocab = set(subj_vocab or [])
            verb_vocab = set(verb_vocab or [])
            obj_vocab  = set(obj_vocab  or [])
            adj_vocab  = set(adj_vocab  or [])
            subj = next((t for t in tokens if t in subj_vocab), None)
            verb = next((t for t in tokens if t in verb_vocab), None)
            obj  = next((t for t in tokens if t in obj_vocab),  None)
            adjs = [t for t in tokens if t in adj_vocab]
        out['affordance'] = self.expose_affordance(subj, verb, obj)
        if adjs:
            target = subj or obj
            if target:
                for a in adjs[:2]:
                    out['modifier'] += self.expose_modifier(a, target)
        return out

    def _expose_ngram_role(self, tokens, n_ctx, role):
        """Write each (ctx_n_ctx -> curr) under `role`. Aggregates per ctx.

        Pack 199 -- frame-conditioned addresses (current_frame_hv applies),
        surprise-gated write strength per curr token (suppresses frequent
        target dominance), bigram-conditional surprise (Pack 199 P1) so
        common transitions (the->the) don't pile up.
        """
        rolev = self.roles[role]
        # Pack 199: frame conditioning on role vector
        if self.current_frame_hv is not None:
            rolev_use = self._bind(self.current_frame_hv, rolev)
            frame_tag = getattr(self, '_current_frame_tag', None)
        else:
            rolev_use = rolev
            frame_tag = None
        bank = self._bank(role)
        agg_data = {}             # slot -> bundled curr_key sum
        slot_curr = {}            # slot -> curr token (for surprise scaling)
        slot_ctx_first = {}       # slot -> first ctx token (for bigram surprise)
        slot_to_ctx_hv = {}
        order = []
        for i in range(n_ctx, len(tokens)):
            ctx = tokens[i - n_ctx:i]
            curr = tokens[i]
            base = '|'.join(ctx) + f'\x00{role}'
            slot = base + (f'\x00{frame_tag}' if frame_tag else '')
            curr_key = self.ck.key(curr)
            if slot in agg_data:
                agg_data[slot] = agg_data[slot] + curr_key
            else:
                agg_data[slot] = curr_key.astype(np.complex64).copy()
                slot_to_ctx_hv[slot] = self._ngram_ctx_hv(ctx)
                slot_curr[slot] = curr
                slot_ctx_first[slot] = ctx[0]
                order.append(slot)
                self._role_targets.setdefault(role, set()).add(ctx[0])
                # also track curr as a target (cleanup-candidate vocab)
                self._role_targets[role].add(curr)
                self._seen.add(ctx[0])
                self._seen.add(curr)
            # bigram count (Pack 199 P1): track (prev, curr) frequencies
            if n_ctx == 1:
                self._bigram_count[(ctx[0], curr)] = \
                    self._bigram_count.get((ctx[0], curr), 0) + 1
                self._bigram_total += 1
        if not order: return
        ukeys = np.stack([self._bind(slot_to_ctx_hv[s], rolev_use) for s in order])
        locs = bank.locs_batch(ukeys, order)
        for s, idx in zip(order, locs):
            curr_tok = slot_curr[s]
            prev_tok = slot_ctx_first[s]
            # Pack 199 P1: bigram-conditional surprise
            ws = self.transition_write_strength(prev_tok, curr_tok)
            bank.C[idx] += (ws * agg_data[s]).astype(np.complex64)

    def next_word_candidates(self, prev, candidates=None, top_k=20, role=None):
        """
        n-gram cleanup query (Pack 120 bigram + Pack 136 trigram + 4-gram).
        `prev` may be:
          - str: single previous token (bigram, role='next')
          - list/tuple of N tokens: n-gram context (auto role: 1->'next',
            2->'next2', 3->'next3'; if >3, last 3 used as 4-gram).
        Returns [(word, score)] sorted by descending score.
        """
        if isinstance(prev, str):
            ctx = [prev]
        else:
            ctx = list(prev)
        if not ctx:
            return []
        if role is None:
            n = len(ctx)
            role = ('next', 'next2', 'next3')[min(n, 3) - 1]
            if n > 3:
                ctx = ctx[-3:]
        if candidates is None:
            candidates = self._cooccur_seen
        if not candidates:
            return []
        rolev = self.roles[role]
        bank = self._bank(role)
        ctx_hv = self._ngram_ctx_hv(ctx)
        addr = self._bind(ctx_hv, rolev)
        r = bank.read(addr)
        cands = list(candidates)
        K = np.stack([self.ck.key(c) for c in cands])
        sims = (np.conj(r) @ K.T).real / self.d
        if top_k >= len(cands):
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, top_k)[:top_k]
            order = top[np.argsort(-sims[top])]
        return [(cands[i], float(sims[i])) for i in order]

    def combined_ngram_candidates(self, last_tokens, candidates=None,
                                   top_k=20, weights=(0.2, 0.4, 0.4)):
        """
        Combined n-gram backoff scoring (Pack 136). Sums weighted candidate
        scores across bigram (next), trigram (next2), 4-gram (next3) as
        context allows. weights = (w_bi, w_tri, w_4gram).  Default tilts
        higher-order; bigram remains as backoff for sparse n-grams.
        """
        if candidates is None:
            candidates = self._cooccur_seen
        if not candidates:
            return []
        cands = list(candidates)
        scores = {c: 0.0 for c in cands}
        n = len(last_tokens)
        for level in range(min(n, 3)):
            ctx_len = level + 1
            if n < ctx_len: continue
            ctx = last_tokens[-ctx_len:]
            w = weights[level] if level < len(weights) else 0.0
            if w == 0: continue
            ranked = self.next_word_candidates(ctx, candidates=cands,
                                                top_k=len(cands))
            for word, s in ranked:
                scores[word] += w * max(s, 0.0)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]

    def poe_candidates(self, last_tokens, candidates=None, top_k=20,
                       pmi=True, eps=None):
        """Pack 311 (Day 80) -- Product-of-Experts (multiplicative AND-gate)
        fusion of the next/next2/next3 transition recalls, with optional
        substrate-native PMI reweight. The free-fluency next-token chooser.

        WHY (vs combined_ngram_candidates): the additive backoff is an OR-gate
        -- one order's function-word energy survives the sum, so greedy argmax
        collapses into the function-word attractor ("the of the of"). The
        MULTIPLICATIVE consensus is an AND-gate: a candidate's fused score
        survives only if it is supported across ALL available orders. A
        high-frequency function word with no higher-order support has its
        fused score collapse -> longer context overrides the unigram prior.
        This is the load-bearing mechanism (probe-confirmed: in-sentence
        repeat 67%->17%, vocab coverage 15%->38%).

        PMI reweight (polish on top): score = sim / (prior + eps), where the
        prior is the substrate's OWN observed unigram frequency (_unigram_count
        /_unigram_total) and eps is the geometry noise floor 1/sqrt(2d). NO
        stopword list -- the frequency prior is measured, not authored.

        Returns [(word, fused_score)] sorted descending.
        """
        if candidates is None:
            candidates = self._cooccur_seen
        if not candidates:
            return []
        cands = list(candidates)
        if eps is None:
            eps = 1.0 / np.sqrt(2.0 * self.d)   # noise-floor regularizer
        # substrate-native frequency prior (observed unigram counts).
        # Defensive for pre-Pack-197 banks with no unigram state (prior flat
        # -> PMI degrades to a no-op, PoE still applies).
        ucount = getattr(self, '_unigram_count', None) or {}
        N = max(getattr(self, '_unigram_total', 0) or 0, 1)
        prior = np.array(
            [ucount.get(c, 0) / N for c in cands],
            dtype=np.float64)
        n = len(last_tokens)
        fused = np.ones(len(cands), dtype=np.float64)
        used_orders = 0
        for level in range(min(n, 3)):
            ctx_len = level + 1
            if n < ctx_len:
                continue
            ctx = last_tokens[-ctx_len:]
            ranked = self.next_word_candidates(ctx, candidates=cands,
                                               top_k=len(cands))
            sim = {w: s for w, s in ranked}
            # relu the per-order sim profile (negatives carry no support)
            p = np.array([max(sim.get(c, 0.0), 0.0) for c in cands],
                         dtype=np.float64)
            if pmi:
                p = p / (prior + eps)        # divide out frequency prior
            # normalize to a distribution before the product (so no single
            # order dominates the scale; the AND-gate acts on shape)
            ssum = p.sum()
            if ssum > 1e-12:
                p = p / ssum
            fused *= (p + 1e-9)              # multiplicative consensus
            used_orders += 1
        if used_orders == 0:
            return []
        ranked = sorted(zip(cands, fused.tolist()), key=lambda kv: -kv[1])
        return ranked[:int(top_k)]

    # ── reading ──────────────────────────────────────────────────────────
    def recall(self, word, role):
        return self._bank(role).read(self._addr(word, role),
                                     word=self._slot(word, role))

    # Pack 247z-CM (Day 72): per-role read-time common-mode HV.
    # v_common[role] = mean(recall(w, role) for w in sample). Subtracting
    # at read time kills the FHRR crosstalk floor so the structured
    # role signal becomes argmax-visible. Cached lazily, persisted in
    # save_ikg, AUTO-APPLIED in query_clean when n_targets > threshold.
    def bake_v_common_all(self, min_targets=200, sample_size=1000, seed=11):
        """Pre-compute v_common for every role whose target count >=
        min_targets. Idempotent; safe to call repeatedly. Returns list of
        roles baked."""
        baked = []
        for role in list(self.roles.keys()):
            n = len(self._role_targets.get(role, set()))
            if n < int(min_targets):
                continue
            self.build_v_common(role, sample_size=sample_size, seed=seed)
            baked.append(role)
        return baked

    def build_v_common(self, role, sample_size=1000, seed=11):
        """Compute and cache v_common HV for `role`. Returns the HV.
        sample_size = number of vocab words to average recall over."""
        import random as _random
        if not hasattr(self, '_v_common') or self._v_common is None:
            self._v_common = {}
        sample = list(self._seen)
        if len(sample) > sample_size:
            rng = _random.Random(seed)
            rng.shuffle(sample)
            sample = sample[:sample_size]
        if not sample:
            return None
        v_sum = np.zeros(self.d, dtype=np.complex128)
        for w in sample:
            v_sum += self.recall(w, role)
        v_c = (v_sum / max(len(sample), 1)).astype(np.complex64)
        self._v_common[role] = v_c
        return v_c

    @staticmethod
    def make_sense_hv(sense_name, d, seed_offset=24770):
        """Pack 247y deterministic per-sense frame HV from a synset name
        string (e.g. 'dog.n.01'). Phasor on unit circle. Stable across
        runs: same synset name -> same HV."""
        import hashlib
        h = hashlib.blake2b(str(sense_name).encode('utf-8'),
                              digest_size=16).digest()
        seed = int.from_bytes(h, 'little') ^ seed_offset
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        ph = rng.uniform(-np.pi, np.pi, int(d)).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    def query_clean_polysemic(self, word, role, candidates=None,
                                synset_names=None, auto_cm_threshold=200,
                                beta=1.0, top_k=5, agg='max'):
        """Pack 247y per-sense cleanup. Iterates synset frames bound to
        `word`, calls query_clean inside each frame, aggregates picks.

        Args:
            word          -- bare word ('dog')
            role          -- relation role ('isa')
            candidates    -- cleanup pool; None -> all role targets
            synset_names  -- explicit list of synset names ('dog.n.01').
                              None -> wn.synsets(word) lookup at call time.
            agg           -- 'max' (pick sense with top1 highest), 'sum'
                              (sum scores across senses then rerank)

        Restores frame state on exit. Returns list of (cand, score).
        """
        if synset_names is None:
            try:
                from nltk.corpus import wordnet as _wn
                synset_names = [s.name() for s in _wn.synsets(word)]
            except Exception:
                synset_names = []
        if not synset_names:
            return self.query_clean(word, role, candidates,
                                       auto_cm_threshold=auto_cm_threshold,
                                       beta=beta, top_k=top_k)

        # Save frame state, restore on exit (defense-in-depth even though
        # in_frame restores)
        saved_hv = self.current_frame_hv
        saved_tag = getattr(self, '_current_frame_tag', None)
        try:
            if agg == 'max':
                best_picks = []
                best_top1 = -1e18
                for sn in synset_names:
                    sense_hv = MultiRoleMemory.make_sense_hv(sn, self.d)
                    with self.in_frame(sense_hv, frame_tag=sn):
                        picks = self.query_clean(
                            word, role, candidates,
                            auto_cm_threshold=auto_cm_threshold,
                            beta=beta, top_k=top_k)
                    if picks and picks[0][1] > best_top1:
                        best_top1 = picks[0][1]
                        best_picks = picks
                return best_picks
            elif agg == 'sum':
                # Sum per-candidate scores across senses, then rerank
                if candidates is None:
                    candidates = list(self.targets(role))
                cand_list = list(candidates) if candidates else []
                if not cand_list:
                    return []
                acc = {c: 0.0 for c in cand_list}
                for sn in synset_names:
                    sense_hv = MultiRoleMemory.make_sense_hv(sn, self.d)
                    with self.in_frame(sense_hv, frame_tag=sn):
                        picks = self.query_clean(
                            word, role, cand_list,
                            auto_cm_threshold=auto_cm_threshold,
                            beta=beta, top_k=len(cand_list))
                    for c, s in picks:
                        acc[c] = acc.get(c, 0.0) + float(s)
                ordered = sorted(acc.items(), key=lambda kv: -kv[1])
                return ordered[:int(top_k)]
            else:
                raise ValueError(f'unknown agg={agg}')
        finally:
            self.current_frame_hv = saved_hv
            self._current_frame_tag = saved_tag

    def query_clean(self, word, role, candidates=None,
                     auto_cm_threshold=200, beta=1.0,
                     resonator_iters=0, top_k=5):
        """Pack 247z-CM read-time cleanup.

        Auto-applies CM subtraction when len(role_targets) >= threshold
        (CM helps big-target roles, HURTS small ones like `class` n=40).

        Args:
            word              -- subject of (subj, role, ?) query
            role              -- role name
            candidates        -- candidate cleanup pool; None -> all targets
            auto_cm_threshold -- only CM if len(targets) >= this
            beta              -- polynomial sharpen exponent on |scores|
            resonator_iters   -- 0 disables; >0 iterative cleanup via
                                  subtract-best-then-rescore
            top_k             -- how many ranked picks to return

        Returns list of (candidate, score) sorted descending.
        """
        if candidates is None:
            candidates = self.targets(role)
        cand_list = list(candidates) if candidates else []
        if not cand_list:
            return []
        n_tot = len(self._role_targets.get(role, set()))
        do_cm = n_tot >= int(auto_cm_threshold)
        r = self.recall(word, role)
        if do_cm:
            if not hasattr(self, '_v_common') or self._v_common is None:
                self._v_common = {}
            v_c = self._v_common.get(role)
            if v_c is None:
                v_c = self.build_v_common(role)
            if v_c is not None:
                r = r - v_c
        K = np.stack([self.ck.key(c) for c in cand_list])
        scores = np.real(K @ np.conj(r)) / self.d
        if beta != 1.0:
            scores = np.sign(scores) * (np.abs(scores) ** float(beta))

        if resonator_iters > 0:
            picks = []
            scores_work = scores.copy()
            used = np.zeros(len(cand_list), dtype=bool)
            for _ in range(int(resonator_iters)):
                masked = np.where(used, -1e9, scores_work)
                idx = int(np.argmax(masked))
                picks.append((cand_list[idx], float(scores_work[idx])))
                used[idx] = True
                # subtract picked candidate's contribution from r, rescore
                r = r - K[idx]
                scores_work = np.real(K @ np.conj(r)) / self.d
                if beta != 1.0:
                    scores_work = (np.sign(scores_work) *
                                     (np.abs(scores_work) ** float(beta)))
                if len(picks) >= top_k:
                    break
            return picks

        order = np.argsort(-scores)[:int(top_k)]
        return [(cand_list[i], float(scores[i])) for i in order]

    def calibrate_abstain(self, role, candidates=None, k=4.0, n_probe=400,
                          seed=320, belief_field=False):
        """Pack 320 -- EMPIRICAL abstain boundary for (role, candidates),
        measured from THIS bank's own absent-query similarity distribution.

        Probes n_probe random keys that were never stored (guaranteed-absent
        subjects), records the top cleanup sim of each against `candidates`,
        and sets boundary = mean + k*std of those sims (calibration.
        empirical_boundary).  This captures the real SDM CROSSTALK floor that
        the theoretical 1/sqrt(2d) model misses (limit-test 319).  Caches the
        result in self._abstain_boundary[role]; returns the boundary.

        Substrate-native: the threshold is READ OFF the bank, not tuned.
        """
        from ikigai.cognition.calibration import empirical_boundary
        import random as _random
        if candidates is None:
            candidates = self.targets(role)
        cand_list = list(candidates) if candidates else []
        if not cand_list:
            return None
        rng = _random.Random(seed)
        alpha = 'abcdefghijklmnopqrstuvwxyz'
        seen = self._seen
        sims = []
        tries = 0
        while len(sims) < int(n_probe) and tries < int(n_probe) * 4:
            tries += 1
            q = '\x00probe_' + ''.join(rng.choice(alpha) for _ in range(12))
            if q in seen:                 # guarantee absent
                continue
            _, sim = self.query(q, role, candidates=cand_list,
                                belief_field=belief_field)
            sims.append(sim)
        if not sims:
            return None
        b = empirical_boundary(sims, k=k)
        if not hasattr(self, '_abstain_boundary') or \
                self._abstain_boundary is None:
            self._abstain_boundary = {}
        self._abstain_boundary[role] = b
        return b

    def abstain_threshold(self, role):
        """Pack 320 -- cached empirical abstain boundary for `role`, or None
        if calibrate_abstain has not been run for it."""
        d = getattr(self, '_abstain_boundary', None)
        return d.get(role) if d else None

    def query(self, word, role, candidates=None, hopfield_iter=0,
                beta=8.0, momentum=0.5, belief_field=True, bf_alpha=1.0):
        """
        Recall the value, clean up against candidate words. Returns (best, score).
        candidates=None -> use all targets ever stored under this role.

        Pack 193: hopfield_iter > 0 enables continuous Hopfield cleanup
        (Ramsauer 2021). Each iter softmax-weights candidates by current
        similarity, blends r toward weighted cluster, re-scores. Sharpens
        recall past Plate-bound. Brain-like reconstructive recall.

        Pack 194: belief_field=True subtracts the mean candidate-score from
        all candidates BEFORE argmax. Kills dominant-attractor pull (e.g.
        every query collapsing to 'the' or '/API'). bf_alpha controls
        subtraction strength (1.0 = full mean removal).
        """
        if candidates is None:
            candidates = self.targets(role)
        cand_list = list(candidates) if candidates else []
        if not cand_list:
            return None, -9.0
        r = self.recall(word, role)
        K = np.stack([self.ck.key(c) for c in cand_list])   # (n, d) complex64

        # Pack 193: continuous Hopfield iter (no-op if hopfield_iter=0)
        for _ in range(int(hopfield_iter)):
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

        # final scoring
        sims = np.real(K @ np.conj(r)) / self.d
        # Pack 194: mean-subtract belief field -- removes global attractor pull.
        if belief_field and len(cand_list) > 1:
            sims = sims - bf_alpha * float(sims.mean())
        # Pack 198: unigram-prior belief field. Subtract log P(target) so
        # globally frequent targets (the/of/and) lose advantage. Brings
        # rare-but-relevant targets into argmax range.
        if belief_field and self._unigram_total > 0 and len(cand_list) > 1:
            import math as _math
            N = max(self._unigram_total, 2)
            logN = _math.log(N)
            priors = np.array([
                _math.log(max(self._unigram_count.get(c, 0), 1) + 1.0) / logN
                for c in cand_list], dtype=np.float32)
            sims = sims - 0.5 * priors
        idx = int(np.argmax(sims))
        return cand_list[idx], float(sims[idx])

    # ── Pack 224 -- Attention-Resonator Factorization ─────────────────────
    def resonator_recall(self, target_hv, candidate_words=None, role=None,
                          n_iters=10, beta=8.0, momentum=0.5,
                          belief_field=True, bf_alpha=1.0, top_k=5):
        """Pack 224 -- Resonator Network decoding (Frady & Kent 2020).

        Bypasses the 1/sqrt(K) cosine ceiling of static VSA arithmetic by
        iteratively cleaning up `target_hv` via continuous Hopfield
        energy minimization over a fixed codebook. Each iter:

            sims_t = beta * Re(K @ conj(r_t)) / d
            w_t    = softmax(sims_t)
            r_new  = sum(w_t[:, None] * K, axis=0)         # weighted recall
            r_t+1  = momentum * r_t + (1 - momentum) * r_new
            renorm

        After n_iters the target has snapped to the dominant attractor mode.
        Then rank candidates by cosine vs the converged target.

        target_hv: complex64[d] -- query vector (e.g. concept arithmetic result).
        candidate_words: iterable of words to score against; None -> use all
                         word keys ever computed via ck.key().
        role: optional role string. If given, candidate_words defaults to
              self.targets(role).
        n_iters: Hopfield iter count. 0 = pure single-pass cosine.
        beta: softmax sharpness. Higher = more selective.
        momentum: blending factor for new estimate. 0.5 = balanced.
        belief_field: if True, mean-subtract sims to kill global attractor pull.
        bf_alpha: belief-field subtraction strength.
        top_k: how many ranked candidates to return.

        Returns list of (word, score) sorted desc by score.
        """
        if candidate_words is None and role is not None:
            candidate_words = self.targets(role)
        if not candidate_words:
            return []
        # Pack 251-S: cache K candidate matrix across repeated calls (eval/absorb
        # loops reuse same candidate set; rebuilding (N, d) complex64 every call
        # is the dominant cost). id-keyed so callers reusing the same list reap
        # the win; new lists each call simply rebuild without churn.
        if not hasattr(self, '_K_cache'):
            self._K_cache = {}
        _key = (id(self.ck), id(candidate_words), len(candidate_words))
        ent = self._K_cache.get(_key)
        if ent is not None:
            K, cand_list = ent
        else:
            cand_list = list(candidate_words)
            K = np.stack([self.ck.key(c) for c in cand_list])
            if len(self._K_cache) >= 8:
                self._K_cache.pop(next(iter(self._K_cache)))
            self._K_cache[_key] = (K, cand_list)
        r = np.asarray(target_hv, dtype=np.complex64).copy()
        # Renorm input target
        mag = float(np.abs(r).mean())
        if mag > 1e-9:
            r = r / mag

        # Resonator iters
        for _ in range(int(n_iters)):
            sims = np.real(K @ np.conj(r)) / self.d            # (N,)
            logits = beta * sims
            logits -= logits.max()
            w = np.exp(logits).astype(np.float32)
            w /= (w.sum() + 1e-12)
            r_new = (w[:, None] * K).sum(axis=0).astype(np.complex64)
            r = (momentum * r + (1.0 - momentum) * r_new).astype(np.complex64)
            mag = float(np.abs(r).mean())
            if mag > 1e-9:
                r = r / mag

        # Final scoring
        sims = np.real(K @ np.conj(r)) / self.d
        if belief_field and len(cand_list) > 1:
            sims = sims - bf_alpha * float(sims.mean())
        if belief_field and self._unigram_total > 0 and len(cand_list) > 1:
            import math as _math
            N = max(self._unigram_total, 2)
            logN = _math.log(N)
            priors = np.array([
                _math.log(max(self._unigram_count.get(c, 0), 1) + 1.0) / logN
                for c in cand_list], dtype=np.float32)
            sims = sims - 0.5 * priors
        # Rank top-k
        order = np.argsort(-sims)[:top_k]
        return [(cand_list[int(i)], float(sims[int(i)])) for i in order]

    # ── Pack 195 (substrate-as-protocol, kill stack #10): .ikg format ─────
    def reset_substrate(self):
        """Zero substrate matrices + clear all per-word state. Roles + role-vectors
        + tile-bank geometry preserved. Use to wipe poisoned data without losing
        the .ikg file or org wiring. Pack 192 v1.2.
        Pack 241: also zeros all multi-bank banks.
        """
        # Pack 241: enumerate ALL banks (multi-bank or legacy 2-bank).
        all_banks = (list(self.banks.values())
                       if self._bank_assignment is not None
                       else [self.sdm, self.sdm_rel])
        for b in all_banks:
            b.C[...] = 0
        self._seen.clear()
        self._cooccur_seen.clear()
        self._verb_seen.clear()
        for k in list(self._role_targets.keys()):
            self._role_targets[k] = set()
        # Pack 197 surprise gate state
        self._unigram_count.clear()
        self._unigram_total = 0
        # clear loc-caches on banks if present
        for bank in all_banks:
            if hasattr(bank, '_loc_cache'):
                try:
                    bank._loc_cache.clear()
                except Exception:
                    pass

    def _frame_state_for_save(self):
        """Pack 197 -- pull frame state from attached FrameField (set by
        IkigaiOrganism before save). Returns empty dict if no frames attached.
        """
        ff = getattr(self, '_frame_field_ref', None)
        if ff is None:
            return {}
        return ff.to_dict()

    def save_ikg(self, path):
        """Save substrate to .ikg file (raw matrices, no pickle wrapper).

        Kill stack #10 -- Substrate-As-Protocol format. Contains:
            C_dense, C_rel (the actual synaptic counter banks)
            seed (Hconj regenerable from this)
            metadata (d, M, k, seen, role_targets)

        ONE file. Substrate IS the file. No teacher, no LLM, no sidecar.
        """
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        rt_keys = list(self._role_targets.keys())
        rt_vals = [list(self._role_targets[k]) for k in rt_keys]
        # Pack 197 frame field + unigram count export
        frame_state = self._frame_state_for_save()
        ug_items = list(self._unigram_count.items())
        ug_keys = np.array([k for k, _ in ug_items], dtype=object)
        ug_vals = np.array([v for _, v in ug_items], dtype=np.int64)
        # Pack 246b (Day 70): bigram_count (Pack 199 surprise gate). Store as
        # (prev, curr) string pairs + counts. Skipped if empty.
        bg_items = list(self._bigram_count.items()) if self._bigram_count else []
        bg_prev = np.array([p for (p, _), _ in bg_items], dtype=object)
        bg_curr = np.array([c for (_, c), _ in bg_items], dtype=object)
        bg_vals = np.array([v for _, v in bg_items], dtype=np.int64)
        # Pack 220 wired-module state (pickle bytes per module)
        p220_state = getattr(self, '_pack220_wired_state', None) or {}
        p220_keys = np.array(list(p220_state.keys()), dtype=object)
        p220_vals = np.array(list(p220_state.values()), dtype=object)
        # Pack 247z-CM (Day 72) per-role v_common HV cache. Skip if empty.
        vc_dict = getattr(self, '_v_common', None) or {}
        vc_keys = np.array(list(vc_dict.keys()), dtype=object)
        if vc_dict:
            vc_vals = np.stack(list(vc_dict.values()))
        else:
            vc_vals = np.empty((0, self.d), dtype=np.complex64)
        # Pack 242 multi-bank persistence (Day 68).
        # If multi-bank mode: store each bank's C + its config under
        # bank_<id>_C / bank_<id>_M / bank_<id>_seed; bank assignment as JSON.
        # The sdm/sdm_rel aliases still get saved for back-compat load.
        save_kwargs = {}
        if self._bank_assignment is not None:
            import json as _json
            ba_dict = {bid: {
                'M': int(self.banks[bid].M),
                'seed': int(self.banks[bid].seed),
                'roles': self._bank_assignment[bid].get('roles', []),
            } for bid in self.banks}
            save_kwargs['multi_bank_mode'] = np.int8(1)
            save_kwargs['bank_assignment_json'] = np.array(
                [_json.dumps(ba_dict)], dtype=object)
            for bid in self.banks:
                save_kwargs[f'bank_{bid}_C'] = self.banks[bid].C
        # pass file handle so np doesn't auto-append .npz to .ikg path
        with open(path, 'wb') as f:
            np.savez_compressed(f,
                C_dense=self.sdm.C,
                C_rel=self.sdm_rel.C,
                d=np.int32(self.d),
                M_dense=np.int32(self.sdm.M),
                M_rel=np.int32(self.sdm_rel.M),
                k_dense=np.int32(self.sdm.k),
                k_rel=np.int32(self.sdm_rel.k),
                seed=np.int32(self._init_seed),  # Pack 242 fix
                sdm_seed=np.int32(self.sdm.seed),
                sdm_rel_seed=np.int32(self.sdm_rel.seed),
                seen=np.array(list(self._seen), dtype=object),
                cooccur_seen=np.array(list(self._cooccur_seen), dtype=object),
                verb_seen=np.array(list(self._verb_seen), dtype=object),
                rt_keys=np.array(rt_keys, dtype=object),
                rt_vals=np.array(rt_vals, dtype=object),
                roles_keys=np.array(list(self.roles.keys()), dtype=object),
                roles_vals=np.stack(list(self.roles.values())),
                unigram_keys=ug_keys,
                unigram_vals=ug_vals,
                unigram_total=np.int64(self._unigram_total),
                bigram_prev=bg_prev,
                bigram_curr=bg_curr,
                bigram_vals=bg_vals,
                bigram_total=np.int64(self._bigram_total),
                pack220_keys=p220_keys,
                pack220_vals=p220_vals,
                vcommon_keys=vc_keys,
                vcommon_vals=vc_vals,
                **frame_state,
                **save_kwargs,
            )
        size = os.path.getsize(path)
        return {'path': path, 'size_mb': size / 1_048_576}

    @classmethod
    def load_ikg(cls, path):
        """Load substrate from .ikg file (raw matrices, Hconj regenerated).
        Pack 242: detects multi-bank mode via `multi_bank_mode` field.
        Legacy 2-bank files load via the existing path; multi-bank files
        reconstruct via the saved bank_assignment_json.
        """
        z = np.load(path, allow_pickle=True)
        d = int(z['d'])
        k_d = int(z['k_dense'])
        seed = int(z['seed'])
        multi_bank = ('multi_bank_mode' in z.files and
                       int(z['multi_bank_mode']) == 1)
        if multi_bank:
            import json as _json
            ba_json = str(z['bank_assignment_json'][0])
            bank_assignment = _json.loads(ba_json)
            mr = cls(d=d, k=k_d, seed=seed,
                       bank_assignment=bank_assignment)
            for bid in mr.banks:
                key = f'bank_{bid}_C'
                if key in z.files:
                    mr.banks[bid].C = z[key].astype(np.complex64)
        else:
            M_d = int(z['M_dense'])
            M_r = int(z['M_rel'])
            mr = cls(d=d, M=M_d, k=k_d, seed=seed, M_rel=M_r)
            mr.sdm.C = z['C_dense'].astype(np.complex64)
            mr.sdm_rel.C = z['C_rel'].astype(np.complex64)
        mr._seen = set(z['seen'].tolist())
        mr._cooccur_seen = set(z['cooccur_seen'].tolist())
        if 'verb_seen' in z.files:
            mr._verb_seen = set(z['verb_seen'].tolist())
        rt_keys = z['rt_keys'].tolist()
        rt_vals = z['rt_vals'].tolist()
        mr._role_targets = {k: set(v) for k, v in zip(rt_keys, rt_vals)}
        # Restore role HVs from save. Pack 247w: saved values are authoritative
        # (overwrites __init__-generated HVs). Critical for minted roles whose
        # HV was sampled from a separate RNG (RoleMinter seed != mr seed+999).
        # Without overwrite, minted role HV diverges across save/load round-trip.
        if 'roles_keys' in z.files:
            rk = z['roles_keys'].tolist()
            rv = z['roles_vals']
            for i, name in enumerate(rk):
                mr.roles[str(name)] = rv[i].astype(np.complex64)
        # Pack 197 unigram counts
        if 'unigram_keys' in z.files:
            uk = z['unigram_keys'].tolist()
            uv = z['unigram_vals'].tolist()
            mr._unigram_count = {k: int(v) for k, v in zip(uk, uv)}
            mr._unigram_total = int(z['unigram_total']) if 'unigram_total' in z.files else sum(uv)
        # Pack 246b (Day 70) bigram counts (Pack 199 surprise gate)
        if 'bigram_prev' in z.files and 'bigram_curr' in z.files:
            bp = z['bigram_prev'].tolist()
            bc = z['bigram_curr'].tolist()
            bv = z['bigram_vals'].tolist()
            mr._bigram_count = {(str(p), str(c)): int(v)
                                  for p, c, v in zip(bp, bc, bv)}
            mr._bigram_total = (int(z['bigram_total'])
                                  if 'bigram_total' in z.files else sum(bv))
        # Pack 197 frame field (lazy: attach if present, IkigaiOrganism reads it)
        if 'frame_K' in z.files:
            mr._pending_frame_state = {k: z[k] for k in z.files if k.startswith('frame_')}
        else:
            mr._pending_frame_state = None
        # Pack 220 wired-module state (lazy: IkigaiOrganism applies it on load_ikg)
        if 'pack220_keys' in z.files and 'pack220_vals' in z.files:
            keys = z['pack220_keys'].tolist()
            vals = z['pack220_vals'].tolist()
            mr._pending_pack220_state = {k: v for k, v in zip(keys, vals)}
        else:
            mr._pending_pack220_state = None
        # Pack 247z-CM (Day 72) restore v_common cache
        if 'vcommon_keys' in z.files and 'vcommon_vals' in z.files:
            vk = z['vcommon_keys'].tolist()
            vv = z['vcommon_vals']
            mr._v_common = {str(k): vv[i].astype(np.complex64)
                              for i, k in enumerate(vk)}
        else:
            mr._v_common = {}
        mr._dirty = True
        return mr

    def chain(self, word, role_a, cands_a, role_b, cands_b):
        """Two-hop cross-channel: role_b( role_a(word) ). Returns (mid, end)."""
        mid, _ = self.query(word, role_a, cands_a)
        end, _ = self.query(mid, role_b, cands_b)
        return mid, end

    def reason_chain(self, word, hops):
        """
        N-hop multi-role reasoning. hops = [(role_1, cands_1), (role_2, cands_2), ...]
        Returns list of waypoints: [word, hop1_target, hop2_target, ..., final].
        Pack 133: extends chain() to arbitrary depth.
        """
        path = [word]
        cur = word
        for role, cands in hops:
            if cur is None: break
            nxt, _ = self.query(cur, role, cands)
            path.append(nxt)
            cur = nxt
        return path

    # ── co-occurrence similarity (mean-removed, Channel 1 in shared bank) ───
    def _refresh_dirs(self):
        if not self._dirty and self._dirs is not None:
            return
        if self.remove_r <= 0 or not self._cooccur_seen:
            self._dirs = np.zeros((0, self.d), dtype=np.complex64)
            self._dirty = False
            return
        words = list(self._cooccur_seen)
        if len(words) > self.svd_sample:
            rng = np.random.default_rng(0)
            words = [words[i] for i in rng.choice(len(words), self.svd_sample,
                                                  replace=False)]
        Mtx = np.stack([self.recall(w, 'cooccur') for w in words])
        _, _, Vh = np.linalg.svd(Mtx, full_matrices=False)
        self._dirs = Vh[:self.remove_r].astype(np.complex64)
        self._dirty = False

    def cooccur_recall(self, word):
        self._refresh_dirs()
        m = self.recall(word, 'cooccur')
        for v in self._dirs:
            m = m - np.vdot(v, m) * v
        return _renorm(m)

    def similarity(self, w1, w2):
        if w1 not in self._cooccur_seen or w2 not in self._cooccur_seen:
            return None
        return _cos(self.cooccur_recall(w1), self.cooccur_recall(w2), self.d)

    # ── introspection ───────────────────────────────────────────────────────
    def substrate_bytes(self):
        # Pack 241 multi-bank: sum across all banks.
        if self._bank_assignment is not None:
            return sum(b.substrate_bytes() for b in self.banks.values())
        return self.sdm.substrate_bytes() + self.sdm_rel.substrate_bytes()

    def role_orthogonality(self, word, role_a, role_b):
        a = self._addr(word, role_a)
        b = self._addr(word, role_b)
        return _cos(a, b, self.d)

    def status(self):
        base = {
            'roles':          list(self.roles.keys()),
            'vocab':          len(self._seen),
            'cooccur_vocab':  len(self._cooccur_seen),
            'substrate_mb':   round(self.substrate_bytes() / 1_048_576, 1),
            'flat':           True,
        }
        if self._bank_assignment is not None:
            base['mode']        = 'multi_bank'
            base['n_banks']     = len(self.banks)
            base['banks_mb']    = {b_id: round(b.substrate_bytes()/1_048_576, 1)
                                     for b_id, b in self.banks.items()}
            base['role_to_bank'] = dict(self._role_to_bank)
        else:
            base['mode']         = 'legacy_2_bank'
            base['cooccur_mb']   = round(self.sdm.substrate_bytes()/1_048_576, 1)
            base['relational_mb']= round(self.sdm_rel.substrate_bytes()/1_048_576, 1)
        return base
