"""
ikigai.cognition.self_modifying_refiner -- Self-Modifying Schema Cognition.

Day 55 Pack 52 -- *5 Decisive stack: runtime schema rewriting.

Problem: SchemaRefiner accumulates positive examples but schema never changes.
         SLOTs stay SLOTs forever even if every use fills them identically.
         No mechanism to SPECIALIZE based on observed usage.

Fix: SelfModifyingRefiner tracks per-slot fill history.
     When SLOT at position i gets identical fill >= promote_threshold times
     -> promote SLOT to fixed token.
     self_modifying_apply() uses promoted values; free SLOTs still take args.

Algorithm:
     feedback(positive, tokens) -> extract {slot_idx: fill_token}
     promote_check(name) -> find slots with dominant fill >= threshold
     SelfModifyingSchema.apply(args) -> fill promoted slots auto, free slots from args

No forgetting:
     promote_check() only adds promotions, never removes.
     Original schema unchanged (promotions stored separately).
     _fill_counts monotone (Counter values only increase).

vs LLM: LLM cannot modify own weights at inference time.
        SelfModifyingRefiner: schema rewrites at inference time.
        Zero gradient required. O(n_fills) per promote_check().
"""

from collections import Counter
import numpy as np
from ikigai.cognition.schema_refiner import SchemaRefiner
from ikigai.cognition.schema_inducer import SLOT, schema_specificity


def _extract_fills(schema, tokens):
    """
    Align schema against generated tokens.
    Returns {slot_idx: fill_token} where fixed positions must agree.
    Returns {} if alignment fails (length mismatch or fixed-token disagreement).
    """
    if len(schema) != len(tokens):
        return {}
    fills = {}
    for i, (s_tok, t_tok) in enumerate(zip(schema, tokens)):
        if s_tok is SLOT:
            fills[i] = str(t_tok)
        elif s_tok != t_tok:
            return {}
    return fills


class SelfModifyingSchema:
    """
    Schema with runtime slot promotions.
    promotions: {slot_idx: promoted_token}
    apply(args): uses promoted tokens for promoted slots, args for free slots.
    """

    def __init__(self, schema):
        self.schema = list(schema)
        self.promotions = {}   # slot_idx -> promoted token

    def apply(self, args):
        """
        Fill schema: promoted SLOTs auto-filled, free SLOTs take from args in order.
        Returns token list.
        """
        result = []
        arg_idx = 0
        for i, tok in enumerate(self.schema):
            if tok is not SLOT:
                result.append(tok)
            elif i in self.promotions:
                result.append(self.promotions[i])
            else:
                if arg_idx < len(args):
                    result.append(str(args[arg_idx]))
                    arg_idx += 1
        return result

    @property
    def n_promotions(self):
        return len(self.promotions)

    @property
    def n_free_slots(self):
        n_slots = sum(1 for t in self.schema if t is SLOT)
        return n_slots - self.n_promotions

    def effective_specificity(self):
        n_fixed = sum(1 for t in self.schema if t is not SLOT)
        return (n_fixed + self.n_promotions) / len(self.schema) if self.schema else 0.0


class SelfModifyingRefiner(SchemaRefiner):
    """
    SchemaRefiner + runtime SLOT promotion.

    promote_threshold: min fills of same token to trigger promotion
    self_modifying_apply(name, *args): use promoted + free slots
    promote_check(name): detect and apply promotions
    n_promotions(name): count promoted slots
    promoted_tokens(name): {slot_idx: token} dict
    """

    def __init__(self, d=64, promote_threshold=3):
        super().__init__(d=d)
        self.promote_threshold = promote_threshold
        self._sms   = {}   # name -> SelfModifyingSchema (may be stale)
        self._fills = {}   # name -> {slot_idx -> Counter}

    #  SelfModifyingSchema cache

    def _get_sms(self, name):
        """Get (or rebuild) SelfModifyingSchema for name."""
        schema = self.inducer.induce(name)
        if schema is None:
            return None
        existing = self._sms.get(name)
        # Rebuild if schema changed (new examples may have shifted it)
        if existing is None or existing.schema != schema:
            sms = SelfModifyingSchema(schema)
            # Preserve promotions that are still valid for new schema
            if existing:
                for idx, tok in existing.promotions.items():
                    if idx < len(schema) and schema[idx] is SLOT:
                        sms.promotions[idx] = tok
            self._sms[name] = sms
        return self._sms[name]

    #  fill extraction

    def _record_fills(self, name, tokens):
        """Extract slot fills from generated tokens aligned to schema."""
        schema = self.inducer.induce(name)
        if schema is None:
            return
        fills = _extract_fills(schema, tokens)
        if not fills:
            return
        if name not in self._fills:
            self._fills[name] = {}
        for idx, tok in fills.items():
            if idx not in self._fills[name]:
                self._fills[name][idx] = Counter()
            self._fills[name][idx][tok] += 1

    #  override feedback to capture fills

    def feedback(self, name, tokens, positive):
        conf = super().feedback(name, tokens, positive)
        if positive and tokens:
            self._record_fills(name, tokens)
        return conf

    def feedback_from_verifier(self, name, tokens, B_U):
        conf, score = super().feedback_from_verifier(name, tokens, B_U)
        if tokens:
            self._record_fills(name, tokens)
        return conf, score

    #  promotion

    def promote_check(self, name):
        """
        Find SLOTs with a dominant fill >= promote_threshold.
        Apply promotions to SelfModifyingSchema.
        Returns current promotions dict {slot_idx: token}.
        """
        sms = self._get_sms(name)
        if sms is None:
            return {}
        fills = self._fills.get(name, {})
        for idx, counter in fills.items():
            if idx in sms.promotions:
                continue
            if not counter:
                continue
            dominant, count = counter.most_common(1)[0]
            if count >= self.promote_threshold:
                sms.promotions[idx] = dominant
        return dict(sms.promotions)

    def promote_all(self):
        """Check and apply promotions for all patterns."""
        return {name: self.promote_check(name) for name in self.inducer._examples}

    #  self-modifying application

    def self_modifying_apply(self, name, *args):
        """
        Apply schema with auto-promoted SLOTs.
        Promoted SLOTs auto-filled. Free SLOTs take from args in order.
        """
        self.promote_check(name)
        sms = self._get_sms(name)
        if sms is None:
            return self.inducer.apply(name, *args)
        return sms.apply(list(args))

    #  stats

    def n_promotions(self, name):
        sms = self._sms.get(name)
        return sms.n_promotions if sms else 0

    def n_free_slots(self, name):
        sms = self._get_sms(name)
        return sms.n_free_slots if sms else 0

    def promoted_tokens(self, name):
        sms = self._sms.get(name)
        return dict(sms.promotions) if sms else {}

    def effective_specificity(self, name):
        sms = self._get_sms(name)
        return sms.effective_specificity() if sms else 0.0

    def fill_counts(self, name):
        return {idx: dict(c) for idx, c in self._fills.get(name, {}).items()}
