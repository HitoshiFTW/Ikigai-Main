"""
ikigai.cognition.cat3_absorb -- Pack 253 cat-3 reasoning state-graph absorb.

Day 73. Streams LLM-generated reasoning chains (DeepSeek-R1-Distill
<think>...</think> blocks) into the organism via:

    1. Pack 251 gated next-token writes        ('next' role, b_lang)
    2. Pack 252 FPE magnitude bindings         ('magnitude' role, b_world)
    3. windowed co-occurrence                  ('cooccur' role, b_lang)
       -- crucially this is where 'five' <-> '5' grounding emerges:
       reasoning chains routinely contain BOTH forms near each other,
       and cooccur binding lets the substrate transitively recall
       magnitude for the word form.

NO hardcoded number-word -> int table. Word-form number magnitude is
EMERGENT via cross-form co-occurrence. parse_number() in Pack 252 only
recognizes digit strings.

Pack 253 is composition over existing primitives -- no new substrate
math. Hand-tuned routing for 'magnitude' role to b_world (numeric
state) per research return.
"""

import re
import numpy as np

from ikigai.cognition.numeric_encoder import parse_number

# Same tokenizer convention as wiki + WordNet + LLMTeacher.
# Pack 253 extended to recognize digit strings; Pack 291.6 (Day 77)
# adds punctuation arithmetic operators `+ - * / =` as standalone
# tokens so the arithmetic verifier in `_absorb_operators` can ground
# them through a*b==c / a+b==c / etc patterns the same way as
# alphabetic operator words.  `-?\d+` still wins on `-12` (greedy
# regex order), so signed integers parse unchanged.
_TOK_RE = re.compile(r"-?\d+|[a-z]+|[+\-*/=]")


def tokenize_chain(text, min_len=1, max_len=20):
    """Lowercase + digit-aware tokenizer for cat-3 reasoning chains."""
    out = []
    for t in _TOK_RE.findall(text.lower()):
        if min_len <= len(t) <= max_len:
            out.append(t)
    return out


class Cat3Absorb:
    """Pack 253 absorb engine. Pairs with RemoteLLMTeacher (strip_think
    OFF) to ingest DeepSeek-R1-Distill reasoning chains."""

    def __init__(self, mr, opv, num_enc,
                  cooccur_window=5,
                  magnitude_role='magnitude',
                  next_role='next',
                  cooccur_role='cooccur',
                  operator_role='operator',
                  bind_digit_word=True,
                  bind_operators=True,
                  mag_encoder='rhc',
                  rhc_moduli=(7, 11, 13, 17)):
        """
        Args:
            mr               -- MultiRoleMemory
            opv              -- OnPolicyEvaluator (Pack 251)
            num_enc          -- NumericEncoder (Pack 252)
            cooccur_window   -- +/- window for cooccur writes
            magnitude_role   -- relation role for digit -> FPE HV
            operator_role    -- relation role for operator tokens ->
                                 canonical op HV (Day 74: retires the
                                 Pack 254 hardcoded _ADD_TOKENS lexicon)
            bind_digit_word  -- also bind word_form -> magnitude via
                                  cooccur (emergent path)
            bind_operators   -- write operator-context tokens to the
                                 operator role so Pack 254 can recall
                                 op from substrate instead of lexicon
        """
        self.mr = mr
        self.opv = opv
        self.num_enc = num_enc
        self.cooccur_window = int(cooccur_window)
        self.magnitude_role = str(magnitude_role)
        self.next_role = str(next_role)
        self.cooccur_role = str(cooccur_role)
        self.operator_role = str(operator_role)
        self.bind_digit_word = bool(bind_digit_word)
        self.bind_operators = bool(bind_operators)
        # Day 75: RHC magnitude encoder fixes FPE off-by-one decode noise
        # at small N. mag_encoder='rhc' (default) -> Pack 257 ResidueHDC.
        # 'fpe' falls back to legacy Pack 252 NumericEncoder.
        self.mag_encoder = str(mag_encoder).lower()
        self._rhc = None
        if self.mag_encoder == 'rhc':
            from ikigai.cognition.residue_hdc import ResidueHDC
            self._rhc = ResidueHDC(d=mr.d, moduli=tuple(rhc_moduli), seed=257)
        self._ensure_role(self.magnitude_role)
        if self.bind_operators:
            self._ensure_role(self.operator_role)
            self._init_operator_codebook()
        self.stats = {
            'chains': 0, 'tokens': 0,
            'next_writes': 0, 'magnitude_writes': 0,
            'cooccur_chains': 0, 'digit_tokens': 0,
            'operator_writes': 0,
        }

    def _init_operator_codebook(self):
        """Generate two deterministic canonical HVs -- op_add, op_sub.
        NO hardcoded token list. Operator semantics emerge from
        ARITHMETIC CONTEXT in absorbed chains:

            scan chain for pattern: [num_a] [TOKEN] [num_b] [equals|=] [num_c]
            if a + b == c:  write TOKEN -> op_add
            if a - b == c:  write TOKEN -> op_sub

        Both 'plus' (a+b=c) and 'gave' (a-b=c, 'Alice had 10 apples,
        gave 4 to Bob, has 6') accrue to their respective op via
        emergent grounding. Token 'equals' will trigger when seen as
        a num separator -- and it doesn't matter what it is called.

        Removes Pack 254 _ADD_TOKENS / _SUB_TOKENS dependency.
        """
        rng = np.random.default_rng(74_001)
        ph_a = rng.uniform(-np.pi, np.pi, self.mr.d).astype(np.float32)
        ph_s = rng.uniform(-np.pi, np.pi, self.mr.d).astype(np.float32)
        ph_m = rng.uniform(-np.pi, np.pi, self.mr.d).astype(np.float32)
        ph_d = rng.uniform(-np.pi, np.pi, self.mr.d).astype(np.float32)
        self.op_add_hv = np.exp(1j * ph_a).astype(np.complex64)
        self.op_sub_hv = np.exp(1j * ph_s).astype(np.complex64)
        # Pack 291.5: third canonical for multiplication.  Grounding
        # via arithmetic verifier (a*b==c) inside `_absorb_operators`.
        self.op_mul_hv = np.exp(1j * ph_m).astype(np.complex64)
        # Pack 291.7: fourth canonical for division.  Grounding via
        # arithmetic verifier (a/b==c using integer division when b!=0)
        # inside `_absorb_operators`.
        self.op_div_hv = np.exp(1j * ph_d).astype(np.complex64)
        # NO hardcoded operator OR equality marker tokens.
        # Detection is purely positional + arithmetic verification.

    def _ensure_role(self, role):
        """Add `role` to mr.roles with a deterministic phasor if absent.
        Routes to b_world per research recommendation for state/numeric
        roles."""
        mr = self.mr
        if role in mr.roles:
            return
        rng = np.random.default_rng(252_000 + sum(ord(c) for c in role))
        ph = rng.uniform(-np.pi, np.pi, mr.d).astype(np.float32)
        mr.roles[role] = np.exp(1j * ph).astype(np.complex64)
        # Bank routing -- 'magnitude' goes to b_world (numeric state).
        if getattr(mr, '_bank_assignment', None) is not None:
            bank_ids = list(mr._bank_assignment.keys())
            if 'b_world' in bank_ids:
                target_bank = 'b_world'
            else:
                target_bank = bank_ids[0]
            mr._role_to_bank[role] = target_bank
            roles_list = mr._bank_assignment[target_bank].setdefault(
                'roles', [])
            if role not in roles_list:
                roles_list.append(role)
        mr._role_targets.setdefault(role, set())

    # ---- per-chain absorb --------------------------------------------

    def absorb_chain(self, text, candidates=None):
        """Absorb one reasoning chain. text is decoded LLM output."""
        toks = tokenize_chain(text)
        if len(toks) < 2:
            return
        self.stats['chains'] += 1
        self.stats['tokens'] += len(toks)

        # 1. Pack 251 gated next-token (cat-1 floor on the chain)
        for i in range(len(toks) - 1):
            prev = toks[i - 1] if i > 0 else None
            cur = toks[i]; nxt = toks[i + 1]
            self.opv.gated_observe(prev, cur, nxt,
                                     role=self.next_role,
                                     candidates=candidates)
            self.stats['next_writes'] += 1

        # 2. Magnitude bindings for digit strings.
        # Day 75: encoder selectable -- 'rhc' (default, Pack 257) bypasses
        # the FPE off-by-one decode noise at small N; 'fpe' is the
        # legacy Pack 252 path. Word-form numbers acquire magnitude via
        # cooccur (step 3) or parallel-frame alignment (step 5).
        for tok in toks:
            n = parse_number(tok)
            if n is None:
                continue
            self.stats['digit_tokens'] += 1
            mag_hv = self._encode_magnitude(int(n))
            self.mr.write_relation(tok, self.magnitude_role, mag_hv)
            self.mr._role_targets[self.magnitude_role].add(tok)
            self.stats['magnitude_writes'] += 1

        # 3. windowed co-occurrence (cross-form binding source)
        try:
            self.mr.expose_cooccur(' '.join(toks))
            self.stats['cooccur_chains'] += 1
        except Exception as e:
            pass

        # 4. Day 74 -- emergent operator grounding (no lexicon).
        # Scan for arithmetic patterns: num op num eq num. Verify the
        # arithmetic; bind the op-slot token to op_add or op_sub HV
        # based on which computation matches. The eq-slot token is
        # ignored -- could be 'equals', 'is', '=', anything.
        if self.bind_operators:
            self._absorb_operators(toks)

        # 5. Day 75 -- emergent word->int magnitude grounding (no lexicon).
        # Scan chain for PARALLEL digit-arithmetic + word-arithmetic frames.
        # If both share the same op-slot and eq-slot tokens, bind each
        # word to its positionally aligned digit's FPE magnitude HV.
        # Example chain: "5 plus 3 equals 8. five plus three equals eight."
        #   digit frame: [5, plus, 3, equals, 8] verifies 5+3=8
        #   word frame:  [five, plus, three, equals, eight] aligns by op+eq
        #   binds: five -> FPE(5), three -> FPE(3), eight -> FPE(8)
        # NO word->int lookup table. NO predefined number words.
        if self.bind_digit_word:
            self._absorb_word_digit_alignment(toks)

    # ---- emergent operator binding -----------------------------------

    def _absorb_operators(self, toks):
        """Day 74 emergent operator grounding.

        Slide a 5-token window. If the window has shape
        [num_a, T_op, num_b, T_eq, num_c] (T_op and T_eq any non-digit
        tokens), verify which arithmetic holds:
            a + b == c  -> T_op binds to op_add
            a - b == c  -> T_op binds to op_sub
        Otherwise skip.

        T_eq is NOT bound -- its identity (could be 'equals', '=',
        'is', 'gives', 'makes', whatever) does not matter for
        operator dispatch.

        No predefined operator vocabulary. Substrate accumulates op
        semantics over many chains via the arithmetic verifier.
        """
        # parse_number per token, cached
        ns = [parse_number(t) for t in toks]
        for i in range(len(toks) - 4):
            a, op, b, eq, c = ns[i], toks[i+1], ns[i+2], toks[i+3], ns[i+4]
            if a is None or b is None or c is None:
                continue
            if parse_number(op) is not None:
                # Op slot must be non-numeric
                continue
            if parse_number(eq) is not None:
                # Eq slot must be non-numeric
                continue
            target_hv = None
            if a + b == c:
                target_hv = self.op_add_hv
            elif a - b == c:
                target_hv = self.op_sub_hv
            elif a * b == c:
                # Pack 291.5: arithmetic verifier grounds multiplicative
                # operator tokens (times, *, x, multiplied, product, ...)
                # to op_mul_hv via the same emergent path as add/sub.
                # Skip degenerate matches: a+b==a-b==a*b only when
                # b is 0 (or a=0/b=0/b=1), which already binds add OR sub
                # via earlier branches.
                target_hv = self.op_mul_hv
            elif b != 0 and a // b == c and a % b == 0:
                # Pack 291.7: integer division grounding.  Only fires
                # when b divides a evenly to keep the verifier strict
                # (no accidental floor-div matches on a=7, b=3, c=2
                # that would collide with a-b=4 or similar).
                target_hv = self.op_div_hv
            else:
                continue
            self.mr.write_relation(toks[i+1], self.operator_role,
                                     target_hv)
            self.mr._role_targets[self.operator_role].add(toks[i+1])
            self.stats['operator_writes'] += 1

    def _absorb_word_digit_alignment(self, toks):
        """Day 75 emergent word->int magnitude grounding via parallel
        arithmetic frames.

        Scan chain for both:
          digit frame : [num_a, T_op, num_b, T_eq, num_c]  arithmetic verifies
          word  frame : [W_a,   T_op, W_b,   T_eq, W_c  ]  all positions non-digit

        When SAME op-slot token + SAME eq-slot token aligns the two frames,
        bind word -> FPE(positionally-aligned digit value) into magnitude
        role. No predefined word->int table; mapping emerges purely from
        the chain's own structure.
        """
        if len(toks) < 5:
            return
        ns = [parse_number(t) for t in toks]
        digit_frames = []  # (i, op, eq, a, b, c)
        word_frames = []   # (i, op, eq, wa, wb, wc)
        for i in range(len(toks) - 4):
            a, op, b, eq, c = toks[i:i+5]
            if parse_number(op) is not None:  continue   # op must be non-numeric
            if parse_number(eq) is not None:  continue   # eq must be non-numeric
            a_n, b_n, c_n = ns[i], ns[i+2], ns[i+4]
            if a_n is not None and b_n is not None and c_n is not None:
                if a_n + b_n == c_n or a_n - b_n == c_n:
                    digit_frames.append((i, op, eq, a_n, b_n, c_n))
            elif a_n is None and b_n is None and c_n is None:
                word_frames.append((i, op, eq, a, b, c))
        if not digit_frames or not word_frames:
            return
        for (di, dop, deq, da, db, dc) in digit_frames:
            for (wi, wop, weq, wa, wb, wc) in word_frames:
                if wop == dop and weq == deq:
                    # Aligned frame -- bind word -> magnitude(digit value)
                    for word, val in [(wa, da), (wb, db), (wc, dc)]:
                        mag_hv = self._encode_magnitude(int(val))
                        self.mr.write_relation(word, self.magnitude_role, mag_hv)
                        self.mr._role_targets[self.magnitude_role].add(word)
                        self.stats['magnitude_writes'] += 1

    def detect_operator(self, tok, threshold=0.05):
        """Substrate-recall an operator's identity for `tok`.
        Returns 'add', 'sub', or None.

        Gate 1: token must be in operator role_targets (was bound by
                cat-3 _absorb_operators in at least one chain) -- this
                excludes SDM crosstalk from other roles.
        Gate 2: recall HV must have meaningful cosine with one canonical
                op HV above `threshold` and clearly above the other.

        Pack 254 calls this as its ONLY operator detection path
        (Pack 263 deleted the hardcoded lexicon fallback). Returns
        None when substrate has no signal -- caller treats that as
        a hard miss.
        """
        targets = self.mr._role_targets.get(self.operator_role, set())
        if tok not in targets:
            return None
        hv = self.mr.recall(tok, self.operator_role)
        if hv is None:
            return None
        hv = np.asarray(hv, dtype=np.complex64)
        norm = float(np.abs(hv).mean()) + 1e-12
        hv = hv / norm
        sim_add = float(np.real(np.vdot(hv, self.op_add_hv)) / self.mr.d)
        sim_sub = float(np.real(np.vdot(hv, self.op_sub_hv)) / self.mr.d)
        sim_mul = float(np.real(np.vdot(hv, self.op_mul_hv)) / self.mr.d)
        sim_div = float(np.real(np.vdot(hv, self.op_div_hv)) / self.mr.d)
        if max(sim_add, sim_sub, sim_mul, sim_div) < threshold:
            return None
        # Pack 291.5 + Pack 291.7: argmax over four canonical ops
        best = 'add'
        best_sim = sim_add
        if sim_sub > best_sim:
            best, best_sim = 'sub', sim_sub
        if sim_mul > best_sim:
            best, best_sim = 'mul', sim_mul
        if sim_div > best_sim:
            best, best_sim = 'div', sim_div
        return best

    # ---- decode helper -----------------------------------------------

    def _encode_magnitude(self, n):
        """Encode int n via the selected magnitude encoder."""
        if self._rhc is not None:
            return self._rhc.encode(int(n))
        return self.num_enc.encode(float(n))

    def _decode_magnitude(self, hv):
        """Decode magnitude HV via the selected encoder. Returns
        (int_pred, score)."""
        if self._rhc is not None:
            pred, score = self._rhc.decode(hv, mode='brute',
                                             x_min=0, x_max=self._rhc.range)
            return int(pred), float(score)
        pred, score = self.num_enc.decode(np.asarray(hv,
                                                       dtype=np.complex64),
                                            x_min=-50, x_max=1100, step=1)
        return int(pred), float(score)

    def recall_magnitude(self, tok):
        """Recall the magnitude HV bound to tok and decode to int.
        Day 75 default: Pack 257 RHC for noise-robust decode across
        the full moduli range. Falls back to Pack 252 FPE if RHC
        not enabled.

        For digit tokens this is direct round-trip. For word tokens,
        magnitude was bound via parallel-frame alignment (step 5 of
        absorb) or cooccur chain. Quality depends on absorb saturation.
        """
        hv = self.mr.recall(tok, self.magnitude_role)
        if hv is None:
            return None, 0.0
        return self._decode_magnitude(np.asarray(hv, dtype=np.complex64))

    def summary(self):
        return dict(self.stats)
