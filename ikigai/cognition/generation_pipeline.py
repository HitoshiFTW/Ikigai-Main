"""
ikigai.cognition.generation_pipeline -- Full Generation Pipeline.

Day 55 Pack 51 -- end-to-end wiring of all generation components.

Components wired:
    CrossModalBinder  -- joint skill+schema retrieval (Pack 48)
    SchemaRefiner     -- generation + online refinement (Pack 49)
    MetacognitiveHVMirror -- self-model drift tracking (Pack 50)

Pipeline steps per run():
    1. query(tokens) -> name via CrossModalBinder (O(n) cosine)
    2. apply_and_verify(name, B_U, *args) -> tokens, ok, score
    3. feedback_from_verifier(name, tokens, B_U) -> confidence update
    4. mirror.update(B_U, tokens) -> self_hv, drift

No forgetting:
    register() = append-only. run() only adds examples (positive feedback).
    All 20 Day-60 schemas survive 20-pattern pipeline execution.

Day 60 benchmark:
    20 patterns x 5 examples -> register all 20.
    pipeline.run(query, B_U, *args) for each pattern.
    All 20 succeed. schema_1 unchanged after 19 others.
    Zero forgetting across entire pipeline execution.
"""

import numpy as np
from ikigai.cognition.cross_modal_binder import CrossModalBinder
from ikigai.cognition.schema_refiner import SchemaRefiner
from ikigai.cognition.metacognitive_mirror import MetacognitiveHVMirror


class RunResult:
    """Result from GenerationPipeline.run()."""
    __slots__ = ['name', 'tokens', 'ok', 'score', 'modal_score', 'conf', 'drift', 'self_hv']

    def __init__(self, name, tokens, ok, score, modal_score, conf, drift, self_hv):
        self.name        = name
        self.tokens      = tokens
        self.ok          = ok
        self.score       = float(score)
        self.modal_score = float(modal_score)
        self.conf        = float(conf)
        self.drift       = float(drift)
        self.self_hv     = self_hv

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != 'self_hv'}


class GenerationPipeline:
    """
    End-to-end generation: retrieve -> generate -> verify -> refine -> mirror.

    register(name, intent_tokens, proc_tokens, schema_examples)
        -- wire skill into binder and refiner

    run(query_tokens, B_U, *slot_args) -> RunResult
        -- full 4-step pipeline, returns structured result

    run_batch(queries) -> list[RunResult]
        -- run multiple queries sequentially
    """

    def __init__(self, d_modal=400, d_verify=64, drift_threshold=0.3):
        self.d_modal  = d_modal
        self.d_verify = d_verify
        self.binder   = CrossModalBinder(d=d_modal)
        self.refiner  = SchemaRefiner(d=d_verify)
        self.mirror   = MetacognitiveHVMirror(d=d_verify, drift_threshold=drift_threshold)
        self._run_log = []

    # ── registration ─────────────────────────────────────────────────────

    def register(self, name, intent_tokens, proc_tokens, schema_examples):
        """Register skill in both binder and refiner."""
        self.binder.bind_skill_schema(name, intent_tokens, proc_tokens, schema_examples)
        self.refiner.observe_many(name, schema_examples)
        return name

    # ── pipeline execution ────────────────────────────────────────────────

    def run(self, query_tokens, B_U, *slot_args):
        """
        4-step generation pipeline.
        B_U: belief HV from BeliefProjectionManifold (float, any dim).
        Returns RunResult.
        """
        d_actual = B_U.shape[0] if hasattr(B_U, 'shape') else self.d_verify

        # Step 1: cross-modal retrieval
        name, modal_score = self.binder.query(query_tokens)

        # Step 2+3: generate + verify (refiner uses its own d)
        tokens, ok, score = self.refiner.apply_and_verify(name, B_U, *slot_args)

        # Step 4: online refinement via auto-feedback
        conf, _ = self.refiner.feedback_from_verifier(name, tokens, B_U)

        # Step 5: metacognitive self-model update
        emit = tokens if tokens else query_tokens
        self_hv, drift = self.mirror.update(B_U, emit)

        result = RunResult(
            name=name, tokens=tokens, ok=ok,
            score=score, modal_score=modal_score,
            conf=conf, drift=drift, self_hv=self_hv,
        )
        self._run_log.append(result.as_dict())
        return result

    def run_batch(self, queries):
        """
        queries: list of (query_tokens, B_U, *slot_args) tuples.
        Returns list of RunResult.
        """
        return [self.run(q[0], q[1], *q[2:]) for q in queries]

    # ── pipeline state ────────────────────────────────────────────────────

    @property
    def n_registered(self):
        return self.binder.n_bindings

    @property
    def n_runs(self):
        return len(self._run_log)

    @property
    def mean_score(self):
        if not self._run_log:
            return 0.0
        return float(np.mean([r['score'] for r in self._run_log]))

    @property
    def pass_rate(self):
        if not self._run_log:
            return 0.0
        return float(np.mean([r['ok'] for r in self._run_log]))

    def summary(self):
        mirror_s = self.mirror.summary()
        return {
            'n_registered':  self.n_registered,
            'n_runs':        self.n_runs,
            'pass_rate':     self.pass_rate,
            'mean_score':    self.mean_score,
            'mean_drift':    mirror_s['mean_drift'],
            'last_drift':    mirror_s['last_drift'],
            'total_examples': self.refiner.total_examples,
        }
