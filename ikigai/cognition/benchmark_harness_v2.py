"""
ikigai.cognition.benchmark_harness_v2 -- Phase C Benchmark Harness.

Day 55 Pack 76 -- pluggable harness for GSM8K, HumanEval, MMLU, BBH, ARC, etc.

Architecture:
    Benchmark = list of Problem(id, prompt, expected, metadata)
    Solver    = callable: prompt -> SolverResult(answer, trace, score_self)
    Scorer    = callable: (problem, result) -> bool (correct?)
    Harness   = run(benchmark, solver, scorer, n_max=None) -> Report

Report tracks:
    - accuracy
    - per-problem trace
    - latency / memory
    - failure analysis (categorized errors)

Compare two solvers (Ikigai vs baseline) -> head-to-head report.

Plug in: GSM8K loader, HumanEval loader, MMLU loader, custom JSONL.

vs Day 54 BenchmarkRunner: that was toy-scoped. This is real-data ready.
"""

import time
import json
import os


class Problem:
    """Single benchmark problem."""

    __slots__ = ('id', 'prompt', 'expected', 'metadata')

    def __init__(self, problem_id, prompt, expected, metadata=None):
        self.id       = problem_id
        self.prompt   = prompt
        self.expected = expected
        self.metadata = metadata or {}

    def __repr__(self):
        return f'Problem(id={self.id!r}, prompt={str(self.prompt)[:40]!r}...)'


class SolverResult:
    """Solver output for one problem."""

    __slots__ = ('answer', 'trace', 'score_self', 'metadata', 'elapsed_ms')

    def __init__(self, answer, trace=None, score_self=0.0, metadata=None, elapsed_ms=0.0):
        self.answer     = answer
        self.trace      = trace
        self.score_self = float(score_self)
        self.metadata   = metadata or {}
        self.elapsed_ms = float(elapsed_ms)


class BenchmarkReport:
    """Run results + per-problem breakdown + summary stats."""

    def __init__(self, benchmark_name, solver_name):
        self.benchmark_name = benchmark_name
        self.solver_name    = solver_name
        self._records       = []   # list of dict per problem

    def add(self, problem_id, expected, answer, correct, elapsed_ms,
            trace=None, metadata=None):
        self._records.append({
            'problem_id':  problem_id,
            'expected':    expected,
            'answer':      answer,
            'correct':     bool(correct),
            'elapsed_ms':  float(elapsed_ms),
            'trace':       trace,
            'metadata':    metadata or {},
        })

    @property
    def n_total(self):
        return len(self._records)

    @property
    def n_correct(self):
        return sum(1 for r in self._records if r['correct'])

    @property
    def accuracy(self):
        return self.n_correct / self.n_total if self.n_total else 0.0

    @property
    def mean_latency_ms(self):
        if not self._records:
            return 0.0
        return sum(r['elapsed_ms'] for r in self._records) / len(self._records)

    def records(self):
        return list(self._records)

    def failures(self, top_k=10):
        """Wrong answers for inspection."""
        wrong = [r for r in self._records if not r['correct']]
        return wrong[:top_k]

    def summary(self):
        return {
            'benchmark':       self.benchmark_name,
            'solver':          self.solver_name,
            'n_total':         self.n_total,
            'n_correct':       self.n_correct,
            'accuracy':        round(self.accuracy, 4),
            'mean_latency_ms': round(self.mean_latency_ms, 2),
        }

    def save_jsonl(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.summary()) + '\n')
            for r in self._records:
                sanitized = {k: v for k, v in r.items() if k != 'trace'}
                f.write(json.dumps(sanitized) + '\n')


class BenchmarkHarness:
    """
    Plug a benchmark + solver + scorer. Run. Get Report.

    Usage:
        harness = BenchmarkHarness('gsm8k_hard', solver_fn, scorer_fn)
        harness.add_problems(problems)
        report = harness.run(n_max=100)
        print(report.summary())
    """

    def __init__(self, benchmark_name, solver, scorer, solver_name='ikigai'):
        self.benchmark_name = benchmark_name
        self.solver         = solver
        self.scorer         = scorer
        self.solver_name    = solver_name
        self._problems      = []

    def add_problem(self, problem):
        self._problems.append(problem)

    def add_problems(self, problems):
        self._problems.extend(problems)

    def load_jsonl(self, path, prompt_key='prompt', expected_key='expected',
                   id_key='id'):
        """Load problems from JSONL. Each line: {id, prompt, expected}."""
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                self.add_problem(Problem(
                    problem_id = obj.get(id_key, f'p{i}'),
                    prompt     = obj[prompt_key],
                    expected   = obj[expected_key],
                    metadata   = {k: v for k, v in obj.items()
                                  if k not in (prompt_key, expected_key, id_key)},
                ))

    def run(self, n_max=None, verbose=False):
        """Run solver across problems. Returns BenchmarkReport."""
        report = BenchmarkReport(self.benchmark_name, self.solver_name)
        problems = self._problems if n_max is None else self._problems[:n_max]

        for i, prob in enumerate(problems):
            t0 = time.perf_counter()
            try:
                result = self.solver(prob.prompt)
                if not isinstance(result, SolverResult):
                    result = SolverResult(answer=result)
            except Exception as e:
                result = SolverResult(answer=None, metadata={'error': str(e)})
            elapsed = (time.perf_counter() - t0) * 1000.0

            try:
                correct = bool(self.scorer(prob, result))
            except Exception:
                correct = False

            report.add(
                problem_id = prob.id,
                expected   = prob.expected,
                answer     = result.answer,
                correct    = correct,
                elapsed_ms = elapsed,
                metadata   = result.metadata,
            )

            if verbose and (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(problems)}] acc so far: {report.accuracy:.3f}')

        return report

    @property
    def n_problems(self):
        return len(self._problems)


# ── Built-in scorers ───────────────────────────────────────────────────────

def numeric_scorer(problem, result):
    """Compare extracted numeric answer to expected number (with tolerance)."""
    if result.answer is None:
        return False
    try:
        ans      = float(str(result.answer).strip().replace(',', ''))
        expected = float(str(problem.expected).strip().replace(',', ''))
        return abs(ans - expected) < 1e-6
    except (ValueError, TypeError):
        return False


def exact_match_scorer(problem, result):
    """String exact match (after strip)."""
    if result.answer is None:
        return False
    return str(result.answer).strip() == str(problem.expected).strip()


def multiple_choice_scorer(problem, result):
    """Letter (A/B/C/D) match."""
    if result.answer is None:
        return False
    return str(result.answer).strip().upper()[:1] == str(problem.expected).strip().upper()[:1]


def code_exec_scorer(problem, result, test_cases=None):
    """
    Execute generated code on test cases. Returns True if all pass.
    test_cases: list of (input_tuple, expected_output).
    Code expected in result.answer as Python source string.
    """
    if result.answer is None:
        return False
    if test_cases is None:
        test_cases = problem.metadata.get('test_cases', [])
    code = str(result.answer)
    try:
        ns = {}
        exec(compile(code, '<benchmark>', 'exec'), ns)   # noqa: trust-bench
        fn_name = problem.metadata.get('fn_name')
        if fn_name and fn_name in ns:
            fn = ns[fn_name]
        else:
            # Find first callable
            fn = next(
                (v for v in ns.values() if callable(v) and not isinstance(v, type)),
                None,
            )
        if fn is None:
            return False
        for (inp, expected_out) in test_cases:
            if not isinstance(inp, tuple):
                inp = (inp,)
            if fn(*inp) != expected_out:
                return False
        return True
    except Exception:
        return False
