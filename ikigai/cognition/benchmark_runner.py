"""
ikigai.cognition.benchmark_runner -- Day 60 Benchmark Runner.

Day 55 Pack 53 -- formal no-forgetting vs frozen-baseline comparison.

FrozenBaseline: simulates LLM in-context only learning.
    - Stores last example verbatim per pattern.
    - Evicts oldest patterns when context_window exceeded.
    - Cannot generate for evicted patterns (forgotten).

BenchmarkRunner: trains both pipeline + baseline, tests recall.
    - pipeline_recall@N = fraction of N patterns pipeline can still generate.
    - baseline_recall@N = fraction baseline can still generate.
    - advantage() = pipeline - baseline.

Day 60 expected result:
    pipeline_recall@20 = 1.0  (no forgetting: HV bind + schema append-only)
    baseline_recall@20 < 1.0  (context window exceeded, early patterns lost)
    advantage > 0             (provable structural advantage)

Report:
    BenchmarkRunner.report() -> structured comparison table.
"""

import numpy as np
from ikigai.cognition.generation_pipeline import GenerationPipeline


class FrozenBaseline:
    """
    In-context LLM simulation: stores only last example per pattern.
    Evicts oldest patterns when context_window exceeded.

    train(name, examples)  -- store last example, evict if window full
    apply(name, *args)     -- return last example verbatim (no schema)
    recall(name)           -- True if name still in context
    """

    def __init__(self, context_window=5):
        self.context_window = context_window
        self._context = {}    # name -> last example tokens
        self._order   = []    # insertion order for LRU eviction

    def train(self, name, examples):
        """Store last example. Evict oldest if context full."""
        self._context[name] = list(examples[-1])
        if name not in self._order:
            self._order.append(name)
        # Evict patterns beyond context window
        while len(self._order) > self.context_window:
            evicted = self._order.pop(0)
            self._context.pop(evicted, None)

    def apply(self, name, *args):
        """Return last stored example verbatim. Empty if forgotten."""
        return list(self._context.get(name, []))

    def recall(self, name):
        return name in self._context

    @property
    def n_in_context(self):
        return len(self._context)

    @property
    def context_names(self):
        return list(self._context.keys())


class BenchmarkRunner:
    """
    Formal Day 60 benchmark: pipeline (no-forgetting) vs baseline (context-limited).

    train(patterns)       -- register all patterns in both systems
    test_recall(patterns, slot_args) -> results dict
    pipeline_recall()     -> float [0, 1]
    baseline_recall()     -> float [0, 1]
    advantage()           -> pipeline - baseline
    report()              -> structured summary string
    """

    def __init__(self, pipeline, baseline, B_U):
        self.pipeline = pipeline
        self.baseline = baseline
        self.B_U      = B_U
        self._results = {}    # name -> {pipeline_ok, baseline_ok, ...}
        self._trained = []    # ordered list of pattern names trained

    # ── training ──────────────────────────────────────────────────────────

    def train(self, patterns):
        """
        patterns: list of (name, intent_tokens, proc_tokens, schema_examples).
        Registers all in pipeline and baseline.
        """
        for name, intent, proc, examples in patterns:
            self.pipeline.register(name, intent, proc, examples)
            self.baseline.train(name, examples)
            if name not in self._trained:
                self._trained.append(name)

    # ── evaluation ────────────────────────────────────────────────────────

    def test_recall(self, patterns, slot_args=None):
        """
        Test which patterns each system can still generate.
        slot_args: {name: (arg1, arg2, ...)} for pipeline generate.
        Returns {name: {pipeline_ok, baseline_ok, pipeline_tokens, baseline_tokens}}.
        """
        if slot_args is None:
            slot_args = {}
        results = {}
        for name, intent, proc, examples in patterns:
            args = slot_args.get(name, ())
            # Pipeline
            pipe_r = self.pipeline.run(intent, self.B_U, *args)
            pipe_ok = len(pipe_r.tokens) > 0
            # Baseline
            base_tokens = self.baseline.apply(name, *args)
            base_ok = len(base_tokens) > 0
            results[name] = {
                'pipeline_ok':     pipe_ok,
                'baseline_ok':     base_ok,
                'pipeline_tokens': pipe_r.tokens,
                'baseline_tokens': base_tokens,
                'pipe_score':      pipe_r.score,
            }
        self._results = results
        return results

    # ── metrics ───────────────────────────────────────────────────────────

    def pipeline_recall(self):
        if not self._results:
            return 0.0
        return float(np.mean([r['pipeline_ok'] for r in self._results.values()]))

    def baseline_recall(self):
        if not self._results:
            return 0.0
        return float(np.mean([r['baseline_ok'] for r in self._results.values()]))

    def advantage(self):
        """Pipeline recall - baseline recall."""
        return self.pipeline_recall() - self.baseline_recall()

    def pipeline_only_recalled(self):
        """Patterns pipeline recalls that baseline forgot."""
        return [
            name for name, r in self._results.items()
            if r['pipeline_ok'] and not r['baseline_ok']
        ]

    def both_recalled(self):
        return [
            name for name, r in self._results.items()
            if r['pipeline_ok'] and r['baseline_ok']
        ]

    def both_forgotten(self):
        return [
            name for name, r in self._results.items()
            if not r['pipeline_ok'] and not r['baseline_ok']
        ]

    # ── report ────────────────────────────────────────────────────────────

    def report(self):
        n = len(self._results)
        pipe_r = self.pipeline_recall()
        base_r = self.baseline_recall()
        adv    = self.advantage()
        pipe_only = self.pipeline_only_recalled()
        lines = [
            f'{"="*60}',
            f'Day 60 Benchmark: No-Forgetting vs Frozen Baseline',
            f'{"="*60}',
            f'  Patterns trained:    {len(self._trained)}',
            f'  Patterns tested:     {n}',
            f'  Baseline context:    {self.baseline.context_window} patterns',
            f'',
            f'  Pipeline recall:     {pipe_r:.1%}  ({int(pipe_r*n)}/{n})',
            f'  Baseline recall:     {base_r:.1%}  ({int(base_r*n)}/{n})',
            f'  Advantage:           {adv:+.1%}',
            f'',
            f'  Pipeline-only:       {len(pipe_only)} patterns',
            f'  {"  ".join(pipe_only[:5])}{"..." if len(pipe_only)>5 else ""}',
            f'{"="*60}',
        ]
        return '\n'.join(lines)
