"""
ikigai.cognition.math_eval -- Pack 254 substrate-native math eval harness.

Day 73 origin. Pack 263 (Day 76) removed the hardcoded
_ADD_TOKENS/_SUB_TOKENS lexicon scaffold. Operator detection now
goes through Pack 253 Cat3Absorb.detect_operator ONLY -- substrate
op_targets role recall + canonical-op cosine gate. No string match.

Two eval modes (NEITHER uses _safe_eval, NEITHER has hardcoded ops):

  1. chain_completion
        Prompt: "five plus three equals"
        Substrate: org.vs_fsm.step("equals", prev_token="three", ...)
                   takes top-1 prediction.
        Pass if prediction matches answer word ('eight') or digit ('8').
        Tests CAT-1 next-token + CAT-3 chain memorization. No FPE math.

  2. substrate_arith
        Parse problem: extract two number tokens (words or digits) and
        ask cat3.detect_operator(tok) on each. Recall magnitude HV
        for each number via mr.recall(token, 'magnitude'). Apply
        Pack 257 RHC compose (auto when digits present) or Pack 252
        FPE compose. Decode to integer.

Operator dispatch (Pack 263 onward): substrate ONLY. cat3 must be
present and must have absorbed enough chains to ground 'plus' /
'minus' / etc. into the operator role. If detect_operator returns
None for every token in the problem, substrate_arith returns
('no op found in substrate') -- no fallback, no lexicon.
"""

import re
import numpy as np

from ikigai.cognition.numeric_encoder import parse_number


# Pack 291.6: include `*`, `/`, `=` so substrate_arith can detect
# punctuation-form multiplicative + division operators that cat-3
# now grounds via the arithmetic verifier.  Matches cat3._TOK_RE.
_TOK_RE = re.compile(r"-?\d+|[a-zA-Z]+|[+\-*/=]")


def tokenize_problem(text):
    return [t.lower() for t in _TOK_RE.findall(text)]


class MathEval:
    """Pack 254 substrate-only math eval.

    No call to ikigai.cognition.conversation._safe_eval. No
    IntentRouter routing. Inference is purely (a) next-token via
    fsm.step or (b) magnitude recall + FPE op or (c) Pack 257 RHC
    direct compositional integer math.

    engine='fpe' : Pack 252 NumericEncoder path (range ~100).
    engine='rhc' : Pack 257 ResidueHDC path (range 17K, no recall
                    needed for digit-form; parses digits directly).
    engine='auto': try RHC if digits present, fall back to FPE for
                    word-form (needs cat-3 cooccur grounding).
    """

    def __init__(self, org, cat3=None, engine='auto', rhc=None):
        self.org = org
        self.mr = org.unified
        self.fsm = org.vs_fsm
        self.num_enc = org.num_enc
        self.cat3 = cat3 or org.cat3
        self.engine = str(engine).lower()
        if self.engine not in ('fpe', 'rhc', 'auto'):
            raise ValueError(f'engine must be fpe/rhc/auto, got {engine!r}')
        # Lazy Pack 257 RHC instance (default moduli give range 17K)
        if rhc is None and self.engine in ('rhc', 'auto'):
            from ikigai.cognition.residue_hdc import ResidueHDC
            rhc = ResidueHDC(d=self.mr.d, moduli=(7, 11, 13, 17),
                              seed=257)
        self.rhc = rhc

    # ---- mode 1: chain completion -----------------------------------

    def chain_completion(self, prompt, max_steps=1, candidates=None):
        """Run fsm.step on the last 1-2 tokens of prompt to predict
        the next token. Returns the top-1 prediction string."""
        toks = tokenize_problem(prompt)
        if len(toks) < 1:
            return None
        cur = toks[-1]
        prev = toks[-2] if len(toks) >= 2 else None
        cands = candidates or list(self.mr._role_targets.get('next', set()))
        try:
            res = self.fsm.step(cur, prev_token=prev, candidates=cands,
                                  n_iters=3, beta=8.0, top_k=5)
        except Exception:
            return None
        if not res:
            return None
        return res[0][0]

    # ---- mode 2: substrate arithmetic --------------------------------

    def _word_magnitude(self, tok):
        """Try to recall a magnitude HV for a token. For digit tokens
        we may have a direct binding; for word tokens we go through
        cooccur (chain to digit form) then to magnitude.

        Returns (numeric HV, predicted_int) or (None, None).
        """
        mr = self.mr
        # 1. Direct magnitude binding (digit tokens have this).
        hv = mr.recall(tok, 'magnitude') if 'magnitude' in mr.roles else None
        if hv is not None and np.linalg.norm(np.asarray(hv)) > 1e-3:
            pred, score = self.num_enc.decode(
                np.asarray(hv, dtype=np.complex64),
                x_min=-50, x_max=1100, step=1)
            return np.asarray(hv, dtype=np.complex64), int(pred)
        # 2. Word -> digit via cooccur recall + cleanup over digit
        #    targets in 'magnitude' role.
        co = mr.recall(tok, 'cooccur')
        if co is None:
            return None, None
        digit_tokens = [t for t in mr._role_targets.get('magnitude', set())
                         if parse_number(t) is not None]
        if not digit_tokens:
            return None, None
        best_tok = None; best_cos = -1e9
        co = np.asarray(co, dtype=np.complex64)
        for d in digit_tokens:
            d_co = mr.recall(d, 'cooccur')
            if d_co is None: continue
            d_co = np.asarray(d_co, dtype=np.complex64)
            num = (np.conj(co) * d_co).real.mean()
            denom = ((np.abs(co)**2).mean()**0.5
                      * (np.abs(d_co)**2).mean()**0.5) + 1e-12
            c = float(num / denom)
            if c > best_cos:
                best_cos = c; best_tok = d
        if best_tok is None:
            return None, None
        # Bind to the digit's magnitude HV
        hv = mr.recall(best_tok, 'magnitude')
        if hv is None:
            return None, None
        return np.asarray(hv, dtype=np.complex64), parse_number(best_tok)

    def substrate_arith(self, problem):
        """Parse problem into (left, op, right). Apply substrate math.
        Returns (predicted_int, op_label, debug).

        Op detection: substrate-only via Pack 253
        Cat3Absorb.detect_operator. No lexicon, no string match.
        If cat3 is missing or has no operator coverage for any token
        in the problem, returns (None, None, {'err': ...}).
        """
        toks = tokenize_problem(problem)
        op_idxs = []
        cat3 = getattr(self.org, '_cat3', None) or self.cat3
        if cat3 is None or not hasattr(cat3, 'detect_operator'):
            return None, None, {'err': 'no cat3 detect_operator'}
        for i, t in enumerate(toks):
            op = cat3.detect_operator(t)
            if op is not None:
                op_idxs.append((i, op))
        if not op_idxs:
            return None, None, {'err': 'no op found in substrate'}
        op_i, op_kind = op_idxs[0]

        # Choose engine. Auto: RHC if at least 2 digit-form tokens.
        use_rhc = self.engine == 'rhc'
        if self.engine == 'auto':
            digit_count = sum(1 for t in toks if parse_number(t) is not None)
            use_rhc = digit_count >= 2 and self.rhc is not None

        if use_rhc and self.rhc is not None:
            # Pack 257 direct path: parse digits, RHC compose, RHC decode.
            left_int = right_int = None
            for j in range(op_i - 1, -1, -1):
                n = parse_number(toks[j])
                if n is not None:
                    left_int = int(n)
                    break
            for j in range(op_i + 1, len(toks)):
                n = parse_number(toks[j])
                if n is not None:
                    right_int = int(n)
                    break
            if left_int is None or right_int is None:
                return None, op_kind, {
                    'engine': 'rhc',
                    'left_int': left_int, 'right_int': right_int,
                    'err': 'digit parse failed'}
            # Pack 305: stay in FACTORED form and decode per-modulus via
            # CRT (O(K*d) ~us) instead of the O(range*d) brute decode
            # (~300 ms).  Each branch produces (pred, score); div returns
            # an int directly from its own CRT recovery.
            substrate_native = True
            if op_kind == 'div':
                # Pack 291.8: substrate-native ⋆-inverse division.  EXACT
                # (b | a) -> div_int (phasor algebra + CRT over invertible
                # moduli).  INEXACT -> Pack 291.7 floor fallback.
                if right_int == 0:
                    return None, op_kind, {
                        'engine': 'rhc', 'err': 'div by zero',
                        'left_int': left_int, 'right_int': right_int}
                if left_int % right_int == 0:
                    pred = self.rhc.div_int(left_int, right_int)
                else:
                    pred = left_int // right_int
                    substrate_native = False
                score = 1.0
            else:
                fa = self.rhc.encode_factored(left_int)
                fb = self.rhc.encode_factored(right_int)
                if op_kind == 'mul':
                    out_factors = self.rhc.mul_factored(fa, fb)
                    pred, score = self.rhc.decode_factored(out_factors)
                elif op_kind == 'add':
                    out_factors = self.rhc.add_factored(fa, fb)
                    pred, score = self.rhc.decode_factored(out_factors)
                else:   # sub -- signed decode so a-b<0 recovers correctly
                    out_factors = self.rhc.sub_factored(fa, fb)
                    pred, score = self.rhc.decode_factored(out_factors,
                                                            signed=True)
            return int(pred), op_kind, {
                'engine': 'rhc',
                'left_int': left_int, 'right_int': right_int,
                'substrate_native': substrate_native,
                'decode_score': float(score)}

        # Pack 252 FPE fallback (word-form needs cat-3 cooccur grounding)
        left_hv = left_int = right_hv = right_int = None
        for j in range(op_i - 1, -1, -1):
            hv, n = self._word_magnitude(toks[j])
            if hv is not None:
                left_hv, left_int = hv, n
                break
        for j in range(op_i + 1, len(toks)):
            hv, n = self._word_magnitude(toks[j])
            if hv is not None:
                right_hv, right_int = hv, n
                break
        if left_hv is None or right_hv is None:
            return None, op_kind, {
                'engine': 'fpe',
                'left_int': left_int, 'right_int': right_int,
                'err': 'magnitude recall failed'}
        if op_kind == 'add':
            result_hv = self.num_enc.add(left_hv, right_hv)
        else:
            result_hv = self.num_enc.sub(left_hv, right_hv)
        pred, score = self.num_enc.decode(result_hv,
                                            x_min=-100, x_max=1100, step=1)
        return int(pred), op_kind, {
            'engine': 'fpe',
            'left_int': left_int, 'right_int': right_int,
            'decode_score': float(score)}

    # ---- bulk eval ---------------------------------------------------

    def evaluate(self, problems):
        """Run both modes on a list of dicts:
            {'problem': str, 'answer': int, 'answer_words': list[str]}
        Returns per-mode pass counts + per-item detail.
        """
        results = []
        chain_pass = arith_pass = 0
        for p in problems:
            prompt = p['problem']
            answer = p['answer']
            answer_words = set(p.get('answer_words', []))
            answer_words.add(str(answer))
            # Mode 1
            c_pred = self.chain_completion(prompt)
            c_ok = (c_pred is not None
                     and (c_pred in answer_words
                          or c_pred == str(answer)))
            if c_ok: chain_pass += 1
            # Mode 2
            a_pred, op_kind, dbg = self.substrate_arith(prompt)
            a_ok = (a_pred is not None
                     and abs(a_pred - answer) < 0.5)
            if a_ok: arith_pass += 1
            results.append({
                'problem': prompt, 'answer': answer,
                'chain_pred': c_pred, 'chain_pass': c_ok,
                'arith_pred': a_pred, 'arith_op': op_kind,
                'arith_dbg': dbg, 'arith_pass': a_ok,
            })
        return {
            'n': len(problems),
            'chain_pass': chain_pass, 'chain_acc': chain_pass/max(len(problems),1),
            'arith_pass': arith_pass, 'arith_acc': arith_pass/max(len(problems),1),
            'detail': results,
        }


# ---- canonical problem sets -------------------------------------------

# Small hand-set. Tests addition + subtraction on small integers (0..30).
# Word forms paired with digit forms so chain_completion can match either.
DEFAULT_ADD_PROBLEMS = [
    {'problem': '5 plus 3 equals',         'answer':  8, 'answer_words': ['eight']},
    {'problem': '7 plus 2 equals',         'answer':  9, 'answer_words': ['nine']},
    {'problem': '10 plus 5 equals',        'answer': 15, 'answer_words': ['fifteen']},
    {'problem': '4 plus 4 equals',         'answer':  8, 'answer_words': ['eight']},
    {'problem': 'five plus three equals',  'answer':  8, 'answer_words': ['eight']},
    {'problem': 'two plus six equals',     'answer':  8, 'answer_words': ['eight']},
    {'problem': 'ten plus ten equals',     'answer': 20, 'answer_words': ['twenty']},
    {'problem': '12 plus 7 equals',        'answer': 19, 'answer_words': ['nineteen']},
    {'problem': '8 plus 6 equals',         'answer': 14, 'answer_words': ['fourteen']},
    {'problem': '20 plus 30 equals',       'answer': 50, 'answer_words': ['fifty']},
]

DEFAULT_SUB_PROBLEMS = [
    {'problem': '10 minus 3 equals',       'answer':  7, 'answer_words': ['seven']},
    {'problem': '15 minus 5 equals',       'answer': 10, 'answer_words': ['ten']},
    {'problem': '20 minus 7 equals',       'answer': 13, 'answer_words': ['thirteen']},
    {'problem': '8 minus 8 equals',        'answer':  0, 'answer_words': ['zero']},
    {'problem': '9 minus 1 equals',        'answer':  8, 'answer_words': ['eight']},
    {'problem': 'ten minus four equals',   'answer':  6, 'answer_words': ['six']},
    {'problem': '50 minus 20 equals',      'answer': 30, 'answer_words': ['thirty']},
    {'problem': '12 minus 4 equals',       'answer':  8, 'answer_words': ['eight']},
]
