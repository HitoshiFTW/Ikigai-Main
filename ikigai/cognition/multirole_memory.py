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
                     'isa', 'sensory', 'property', 'verb', 'class')

    def __init__(self, d=512, M=16384, k=64, seed=114, window=3, remove_r=1,
                 svd_sample=2000, roles=DEFAULT_ROLES, M_rel=8192,
                 consolidate_every=0, q_omega=0.05, q_seed=12345):
        self.d = int(d); self.window = int(window)
        self.remove_r = int(remove_r); self.svd_sample = int(svd_sample)
        self.ck = ComputedKey(d=d, seed=seed)
        # TWO traffic classes in separate fixed banks (Pack 117 finding):
        #   cooccur = high traffic (millions of writes) -> own bank
        #   relational (isa/sensory/property/verb) = sparse, equal low traffic;
        #   role-binding separates them WITHIN one shared relational bank.
        # A shared bank across traffic classes lets co-occurrence mass (~2000)
        # swamp a single relational write (~22). Separating by traffic fixes it.
        # Online consolidation on cooccur bank only -- relational bank must
        # preserve write-count reinforcement (assert_isa n=50 = 50x magnitude).
        self.sdm = VSASDM(d=d, M=M, k=k, seed=seed,
                          consolidate_every=consolidate_every)            # cooccur
        self.sdm_rel = VSASDM(d=d, M=M_rel, k=k, seed=seed + 1)            # relational
        rng = np.random.default_rng(seed + 999)
        self.roles = {}
        for name in roles:
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            self.roles[name] = np.exp(1j * ph).astype(np.complex64)
        self._seen = set()        # words with any relation
        self._cooccur_seen = set()
        self._role_targets = {}   # role -> set of target words seen (cleanup candidates)
        self._verb_seen = set()
        self._dirs = None
        self._dirty = True
        # Quantity-axis for verb-rotor encoding (Channel 2). Phase encodes a scalar.
        self.q_omega = float(q_omega)
        rng_axis = np.random.default_rng(q_seed)
        self.q_axis = (rng_axis.integers(0, 2, size=self.d) * 2 - 1).astype(np.float32)

    # Roles partition by TRAFFIC CLASS:
    #   dense (high traffic, every token):     cooccur, next
    #   sparse (low traffic, explicit facts):  isa, sensory, property, verb
    # Sparse signal would drown in the dense bank, so they get their own.
    DENSE_ROLES = {'cooccur', 'next', 'next2', 'next3'}

    def _bank(self, role):
        return self.sdm if role in self.DENSE_ROLES else self.sdm_rel

    #  binding
    def _bind(self, a, r):
        return (a * r).astype(np.complex64)

    def _unbind(self, c, r):
        return (c * np.conj(r)).astype(np.complex64)

    def _addr(self, word, role):
        return self._bind(self.ck.key(word), self.roles[role])

    def _slot(self, word, role):
        return f'{word}\x00{role}'        # loc-cache key

    #  writing
    def write_relation(self, word, role, value_hv):
        self._bank(role).write(self._addr(word, role),
                               np.asarray(value_hv, dtype=np.complex64),
                               word=self._slot(word, role))
        self._seen.add(word)

    def relate(self, word, role, target_word):
        """word --role--> target_word  (stores the target's computed key)."""
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
        rolev = self.roles['cooccur']
        ukeys = np.stack([self._bind(self.ck.key(t), rolev) for t in order])
        slots = [self._slot(t, 'cooccur') for t in order]
        locs = self.sdm.locs_batch(ukeys, slots)
        for t, idx in zip(order, locs):
            self.sdm.C[idx] += agg[t].astype(np.complex64)
        self._dirty = True
        return n

    #  Channel 2 (verb rotor) on flat memory
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

    def _expose_ngram_role(self, tokens, n_ctx, role):
        """Write each (ctx_n_ctx -> curr) under `role`. Aggregates per ctx."""
        rolev = self.roles[role]
        bank = self._bank(role)
        agg_data = {}
        slot_to_ctx_hv = {}
        order = []
        for i in range(n_ctx, len(tokens)):
            ctx = tokens[i - n_ctx:i]
            curr = tokens[i]
            slot = '|'.join(ctx) + f'\x00{role}'
            curr_key = self.ck.key(curr)
            if slot in agg_data:
                agg_data[slot] = agg_data[slot] + curr_key
            else:
                agg_data[slot] = curr_key.astype(np.complex64).copy()
                slot_to_ctx_hv[slot] = self._ngram_ctx_hv(ctx)
                order.append(slot)
                # track first-token of context as a "seen prev"
                self._role_targets.setdefault(role, set()).add(ctx[0])
                self._seen.add(ctx[0])
        if not order: return
        ukeys = np.stack([self._bind(slot_to_ctx_hv[s], rolev) for s in order])
        locs = bank.locs_batch(ukeys, order)
        for s, idx in zip(order, locs):
            bank.C[idx] += agg_data[s]

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

    #  reading
    def recall(self, word, role):
        return self._bank(role).read(self._addr(word, role),
                                     word=self._slot(word, role))

    def query(self, word, role, candidates=None):
        """
        Recall the value, clean up against candidate words. Returns (best, score).
        candidates=None -> use all targets ever stored under this role.
        """
        if candidates is None:
            candidates = self.targets(role)
        r = self.recall(word, role)
        best, bscore = None, -9.0
        for c in candidates:
            s = _cos(r, self.ck.key(c), self.d)
            if s > bscore:
                bscore, best = s, c
        return best, bscore

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

    #  co-occurrence similarity (mean-removed, Channel 1 in shared bank)
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

    #  introspection
    def substrate_bytes(self):
        return self.sdm.substrate_bytes() + self.sdm_rel.substrate_bytes()

    def role_orthogonality(self, word, role_a, role_b):
        a = self._addr(word, role_a)
        b = self._addr(word, role_b)
        return _cos(a, b, self.d)

    def status(self):
        return {
            'roles':          list(self.roles.keys()),
            'vocab':          len(self._seen),
            'cooccur_vocab':  len(self._cooccur_seen),
            'cooccur_mb':     round(self.sdm.substrate_bytes() / 1_048_576, 1),
            'relational_mb':  round(self.sdm_rel.substrate_bytes() / 1_048_576, 1),
            'substrate_mb':   round(self.substrate_bytes() / 1_048_576, 1),
            'flat':           True,
        }
