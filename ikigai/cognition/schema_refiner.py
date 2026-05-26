"""
ikigai.cognition.schema_refiner -- Online Schema Refinement via Feedback.

Day 55 Pack 49 -- closes the generation-verification loop.

Problem: SchemaInducer generates from static examples.
         No improvement mechanism after deployment.
         Generated output quality not measured against belief state.

Fix: SchemaRefiner wraps SchemaInducer + SelfVerifier.
     After apply(), SelfVerifier grades coherence with B_U belief HV.
     Positive: add generated tokens as new example -> re-induce (tighter schema).
     Negative: increment neg count, reduce confidence.

Loop:
     observe(name, examples)              -- seed schema
     tokens, ok, score = apply_and_verify(name, B_U, *args)
     confidence = feedback_from_verifier(name, tokens, B_U)
     -- positive feedbacks accumulate -> schema specificity increases

No forgetting:
     observe() append-only. Negative feedback tracked separately.
     Confidence = pos/(pos+neg), never removes examples.
     Schema always re-induced from full example set.

Day 60 benchmark:
     5 seed examples -> initial schema.
     10 positive feedbacks -> support=15, confidence=1.0.
     specificity >= initial_specificity (consistent outputs fix slots).
     Zero forgetting: all 20 schemas persist.
"""

import numpy as np
from ikigai.cognition.schema_inducer import SchemaInducer, apply_schema, SLOT, schema_specificity
from ikigai.cognition.self_verifier import SelfVerifier


class SchemaRefiner:
    """
    Online schema refinement via feedback loop.

    observe(name, tokens)                      -- add training example
    observe_many(name, examples)               -- batch observe
    apply(name, *args)                         -- generate from schema
    apply_and_verify(name, B_U, *args)         -- generate + coherence check
    feedback(name, tokens, positive)           -- manual feedback
    feedback_from_verifier(name, tokens, B_U)  -- auto-feedback via verifier
    confidence(name)                           -- pos/(pos+neg)
    schema_info(name)                          -- full info dict
    """

    def __init__(self, d=64):
        self.d = d
        self.inducer  = SchemaInducer()
        self.verifier = SelfVerifier(d=d)
        self._pos  = {}   # name -> int
        self._neg  = {}   # name -> int
        self._conf = {}   # name -> float

    # ── observation ───────────────────────────────────────────────────────

    def observe(self, name, tokens):
        """Append-only example storage. Returns support count."""
        return self.inducer.observe(name, tokens)

    def observe_many(self, name, examples):
        """Batch observe. Returns final support count."""
        return self.inducer.observe_many(name, examples)

    # ── application ───────────────────────────────────────────────────────

    def apply(self, name, *args):
        """Generate token list from schema + slot args."""
        return self.inducer.apply(name, *args)

    def apply_and_verify(self, name, B_U, *args):
        """
        Generate from schema, verify coherence with belief HV B_U.
        Returns (tokens, ok, coherence_score).
        """
        tokens = self.inducer.apply(name, *args)
        if not tokens:
            return tokens, False, 0.0
        ok, score = self.verifier.verify_coherence(tokens, B_U)
        return tokens, ok, float(score)

    # ── feedback ──────────────────────────────────────────────────────────

    def feedback(self, name, tokens, positive):
        """
        Manual feedback update.
        Positive: add tokens as new training example (re-induction on next apply).
        Returns updated confidence.
        """
        if positive and tokens:
            self.inducer.observe(name, tokens)
            self._pos[name] = self._pos.get(name, 0) + 1
        else:
            self._neg[name] = self._neg.get(name, 0) + 1
        return self._update_conf(name)

    def feedback_from_verifier(self, name, tokens, B_U):
        """
        Auto-feedback: SelfVerifier grades coherence, result drives feedback.
        Returns (confidence, coherence_score).
        """
        ok, score = self.verifier.verify_coherence(tokens, B_U)
        conf = self.feedback(name, tokens, positive=ok)
        return conf, float(score)

    def _update_conf(self, name):
        pos   = self._pos.get(name, 0)
        neg   = self._neg.get(name, 0)
        total = pos + neg
        conf  = pos / total if total > 0 else 0.5
        self._conf[name] = conf
        return conf

    # ── stats ─────────────────────────────────────────────────────────────

    def confidence(self, name):
        return self._conf.get(name, 0.5)

    def specificity(self, name):
        schema = self.inducer.induce(name)
        return schema_specificity(schema) if schema else 0.0

    def schema_info(self, name):
        info = self.inducer.schema_info(name) or {}
        info['confidence'] = self.confidence(name)
        info['pos_count']  = self._pos.get(name, 0)
        info['neg_count']  = self._neg.get(name, 0)
        return info

    def support(self, name):
        return self.inducer.support(name)

    @property
    def n_patterns(self):
        return self.inducer.n_patterns

    @property
    def n_schemas(self):
        return self.inducer.n_schemas

    @property
    def total_examples(self):
        return self.inducer.total_examples
