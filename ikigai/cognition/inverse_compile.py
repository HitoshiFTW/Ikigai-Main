"""
ikigai.cognition.inverse_compile -- Inverse Compilation Engine.

Day 55 Pack 65 -- invention #9: behavior trace -> program HV.

Forward compilation:
    program_tokens -> program_hv (encoded VSA)
    program_hv ⊕ input_hv -> output_hv  (forward execution)

Inverse compilation:
    Given many (input_i, output_i) examples,
    each yields:  candidate_program_i_hv = bind(input_i, output_i)
                  (since bind is self-inverse: bind(input_i, candidate) = output_i)
    Aggregate: program_hv = sign(sum of all candidate_i hvs)
    -> nearest program in library = the program that produced these I/O traces.

Use cases:
    1. Imitation: watch agent do task, infer program
    2. Black-box reverse-engineering: query LLM with I/O, infer its function
    3. Program induction: given few examples, infer general program HV

Bio analog: mirror neurons. Observe action -> retrieve own motor program for it.

vs LLM: program induction = expensive search + symbolic synthesis.
        InvCompile: O(N) cosine lookup in program library. Zero search.
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
    return np.sign(a * b).astype(np.float32)


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


class InverseCompiler:
    """
    Program library + inverse compilation from behavior traces.

    register(program_name, program_tokens, io_examples=None)
        Add program to library. If io_examples given, also store them
        as forward bindings program_hv = mean(bind(input, output)).

    register_hv(program_name, program_hv)
        Add program by direct HV (for synthetic programs).

    forward(program_name, input_tokens) -> predicted_output_hv
        Apply program forward: bind(program_hv, input_hv).

    induce(io_examples) -> aggregated_program_hv
        From list of (input_tokens, output_tokens), produce inferred program HV.

    identify(io_examples, top_k) -> [(program_name, sim), ...]
        Inverse compile: which library program best matches these I/O examples?

    verify(program_name, io_examples) -> (n_correct, n_total, accuracy)
        For each (i, o): forward(program, i) ~= o? Counts agreements.
    """

    def __init__(self, d=400):
        self.d        = d
        self._programs = {}     # name -> program_hv

    # ── registration ──────────────────────────────────────────────────────

    def register(self, name, program_tokens, io_examples=None):
        """
        Register program by its tokens. If io_examples supplied,
        BLEND token-based HV with averaged I/O-binding HV for robustness.
        """
        token_hv = _encode(program_tokens, self.d)
        if io_examples:
            accum = np.zeros(self.d, dtype=np.int32)
            for (inp, out) in io_examples:
                accum += _bind(_encode(inp, self.d), _encode(out, self.d)).astype(np.int32)
            io_hv = _bsign(accum.astype(np.float32))
            # Equal blend
            program_hv = _bsign((token_hv + io_hv).astype(np.float32))
        else:
            program_hv = token_hv
        self._programs[name] = program_hv
        return program_hv

    def register_hv(self, name, program_hv):
        self._programs[name] = np.asarray(program_hv, dtype=np.float32).copy()
        return self._programs[name]

    def program_hv(self, name):
        return self._programs.get(name)

    # ── forward execution ─────────────────────────────────────────────────

    def forward(self, name, input_tokens):
        """Apply program forward: bind(program_hv, input_hv) -> predicted output HV."""
        p = self._programs.get(name)
        if p is None:
            return None
        return _bind(p, _encode(input_tokens, self.d))

    # ── inverse compilation ──────────────────────────────────────────────

    def induce(self, io_examples):
        """
        From (input, output) pairs, induce aggregate program HV.
        program_hv ≈ sign(sum over i of bind(input_i, output_i)).
        """
        if not io_examples:
            return np.zeros(self.d, dtype=np.float32)
        accum = np.zeros(self.d, dtype=np.int32)
        for (inp, out) in io_examples:
            i_hv = _encode(inp, self.d)
            o_hv = _encode(out, self.d)
            accum += _bind(i_hv, o_hv).astype(np.int32)
        return _bsign(accum.astype(np.float32))

    def identify(self, io_examples, top_k=3):
        """
        Inverse compile: which library program best matches these I/O examples?
        Returns [(program_name, sim), ...] descending.
        """
        induced = self.induce(io_examples)
        results = [(n, _cosine(induced, hv)) for n, hv in self._programs.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ── verification ──────────────────────────────────────────────────────

    def verify(self, program_name, io_examples, threshold=0.3):
        """
        For each (input, expected_output): forward(name, input) ~= expected?
        Counts predictions whose cosine with expected exceeds threshold.
        Returns (n_correct, n_total, accuracy).
        """
        if program_name not in self._programs:
            return 0, len(io_examples), 0.0
        n_correct = 0
        for (inp, out) in io_examples:
            pred  = self.forward(program_name, inp)
            exp_h = _encode(out, self.d)
            if _cosine(pred, exp_h) >= threshold:
                n_correct += 1
        n_total = len(io_examples)
        return n_correct, n_total, (n_correct / n_total if n_total else 0.0)

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def n_programs(self):
        return len(self._programs)

    def program_names(self):
        return list(self._programs.keys())
