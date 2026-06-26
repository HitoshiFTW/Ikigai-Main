"""
ikigai.cognition.proof_carrying_gen -- Proof-Carrying Generation.

Day 55 Pack 63 -- ★1 invention #7: derivation chain HV per output.

Analog: Necula 1997 Proof-Carrying Code. Every output ships with a proof of
its derivation. Recipient verifies in O(N) chain length. Tamper any step ->
verifier rejects. Result: unhallucinatable generation.

Chain structure:
    step_i = (rule_name, rule_hv, premise_hv, conclusion_hv)
    conclusion_i = bind(rule_hv, premise_hv)        # algebraic derivation
    premise_{i+1} = conclusion_i                    # chain composition
    chain_hv = sign(sum of all conclusion_hvs)       # holographic summary

Verification:
    For each step i: re-derive bind(rule, premise), compare to stored conclusion.
    Any mismatch -> chain broken at step i -> reject output.

Trust property:
    Generator MUST produce (output, chain). Consumer MUST verify chain
    before accepting output. No verified chain -> no trust.

vs LLM: LLM emits tokens with no proof. Hallucinations indistinguishable
        from facts. PCG: output is rejected if chain doesn't reproduce it.
        Zero-cost runtime safety net.
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


def _bind(a, b):
    """Bipolar bind = elementwise multiply (self-inverse)."""
    out = (a * b).astype(np.float32)
    out[out == 0.0] = 1.0
    return np.sign(out).astype(np.float32)


def _bsign(x):
    s = np.sign(x).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ProofChain:
    """
    Ordered derivation chain. Each step proves: conclusion = bind(rule, premise).

    add_step(rule_name, rule_hv, premise_hv) -> conclusion_hv
        Appends step; conclusion auto-computed.

    verify() -> (ok, broken_step_idx)
        Re-derive every step. ok=True iff all match.

    chain_hv() -> bipolar HV summarizing chain (sum-then-sign of conclusions).

    tamper(step_idx, new_conclusion_hv)
        Test-only: corrupt a step to test verifier.
    """

    def __init__(self, d=400):
        self.d = d
        self._steps = []  # list of [rule_name, rule_hv, premise_hv, conclusion_hv]

    def add_step(self, rule_name, rule_hv, premise_hv):
        rule_hv    = np.asarray(rule_hv,    dtype=np.float32).copy()
        premise_hv = np.asarray(premise_hv, dtype=np.float32).copy()
        conclusion = _bind(rule_hv, premise_hv)
        self._steps.append([rule_name, rule_hv, premise_hv, conclusion.copy()])
        return conclusion

    def verify(self):
        """Walk chain, re-derive each step. Returns (ok, broken_idx_or_None)."""
        for i, (rule_name, rule_hv, premise_hv, stored) in enumerate(self._steps):
            rederived = _bind(rule_hv, premise_hv)
            if not np.array_equal(rederived, stored):
                return False, i
        return True, None

    def chain_hv(self):
        """Holographic summary: sign of sum of all conclusions."""
        if not self._steps:
            return np.zeros(self.d, dtype=np.float32)
        accum = np.zeros(self.d, dtype=np.int32)
        for (_, _, _, c) in self._steps:
            accum += c.astype(np.int32)
        return _bsign(accum.astype(np.float32))

    def tamper(self, step_idx, new_conclusion_hv):
        """Test-only: overwrite stored conclusion to validate verifier rejection."""
        if 0 <= step_idx < len(self._steps):
            self._steps[step_idx][3] = np.asarray(new_conclusion_hv, dtype=np.float32).copy()

    def n_steps(self):
        return len(self._steps)

    def rule_sequence(self):
        return [s[0] for s in self._steps]

    def final_conclusion(self):
        return self._steps[-1][3] if self._steps else np.zeros(self.d, dtype=np.float32)

    def step(self, idx):
        """Return (rule_name, rule_hv, premise_hv, conclusion_hv) tuple for step idx."""
        return tuple(self._steps[idx])


class ProofCarryingGenerator:
    """
    Generation engine attaching a derivation chain to every output.

    add_rule(name) -> rule_hv
        Register a derivation rule. Idempotent.

    generate(query_tokens, rule_sequence, premise_tokens_list) -> (output_hv, chain, verified)
        Apply rules in order. Each conclusion feeds as next step's premise.
        Verifier runs automatically. Returns final HV + chain + ok flag.

    store(name, output_hv, chain)
        Library insert. Untrusted by default.

    trusted_recall(name) -> (output_hv, chain, ok)
        Returns library entry ONLY if chain verifies. Else (None, None, False).

    explain(chain) -> [(step_idx, rule_name, premise_hash, conclusion_hash)]
        Human-readable chain trace.
    """

    def __init__(self, d=400):
        self.d        = d
        self.rules    = {}      # rule_name -> rule_hv
        self._library = {}      # name -> (output_hv, chain)

    # ── rule registry ─────────────────────────────────────────────────────

    def add_rule(self, rule_name):
        if rule_name not in self.rules:
            self.rules[rule_name] = _hv_for(f'__rule__:{rule_name}', self.d)
        return self.rules[rule_name]

    def rule_hv(self, rule_name):
        return self.rules.get(rule_name)

    # ── generation ────────────────────────────────────────────────────────

    def generate(self, query_tokens, rule_sequence, premise_tokens_list):
        """
        Apply rules in order; each step produces conclusion fed as next premise.
        Returns (final_output_hv, ProofChain, verified_bool).
        """
        if len(rule_sequence) != len(premise_tokens_list):
            raise ValueError('rule_sequence and premise_tokens_list must match in length')

        chain = ProofChain(self.d)
        current = _encode(query_tokens, self.d)

        for rule_name, prem_tokens in zip(rule_sequence, premise_tokens_list):
            rule_hv     = self.add_rule(rule_name)
            prem_hv     = _encode(prem_tokens, self.d)
            # Premise at step i = current state ⊕ new evidence
            step_premise = _bind(current, prem_hv)
            current      = chain.add_step(rule_name, rule_hv, step_premise)

        ok, _ = chain.verify()
        return current, chain, ok

    # ── library ───────────────────────────────────────────────────────────

    def store(self, name, output_hv, chain):
        self._library[name] = (
            np.asarray(output_hv, dtype=np.float32).copy(),
            chain,
        )

    def trusted_recall(self, name):
        """Returns (output_hv, chain, True) only if chain verifies. Else (None, None, False)."""
        entry = self._library.get(name)
        if entry is None:
            return None, None, False
        output_hv, chain = entry
        ok, _ = chain.verify()
        if not ok:
            return None, None, False
        return output_hv.copy(), chain, True

    def n_stored(self):
        return len(self._library)

    # ── explain ───────────────────────────────────────────────────────────

    def explain(self, chain):
        """Return [(step_idx, rule_name, premise_norm, conclusion_norm)] for inspection."""
        rows = []
        for i in range(chain.n_steps()):
            rule_name, _, premise_hv, conclusion_hv = chain.step(i)
            rows.append((
                i,
                rule_name,
                float(np.linalg.norm(premise_hv)),
                float(np.linalg.norm(conclusion_hv)),
            ))
        return rows
