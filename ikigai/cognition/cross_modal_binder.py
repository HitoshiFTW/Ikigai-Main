"""
ikigai.cognition.cross_modal_binder -- Cross-Modal HV Binding.

Day 55 Pack 48 -- joint skill+schema retrieval via triplet HV binding.

Problem: SkillCrystal retrieves procedures. SchemaInducer retrieves templates.
         Two separate lookups. No joint representation.

Fix: CrossModalBinder binds (intent x proc x schema) into one modal_hv.
     Single query retrieves BOTH procedure and output template.
     Unbinding: modal x intent = proc x schema -> schema via proc unbind.

Binding algebra (bipolar +-1, self-inverse x):
     skill_hv  = intent x proc
     modal_hv  = skill_hv x schema_hv = intent x proc x schema

     Given query q ~= intent:
       modal x q ~= proc x schema
       (proc x schema) x proc ~= schema

No forgetting: _bindings dict append-only. Learning skill_N never touches skill_1.

Day 60 benchmark:
     20 skills each with schema template.
     query(new_intent) -> correct skill+schema in one cosine lookup.
     generate(query, *args) -> filled token list.
"""

import numpy as np
from ikigai.cognition.skill_crystal import _encode, _bind, _sim
from ikigai.cognition.schema_inducer import SchemaInducer, apply_schema, SLOT


def _encode_schema(schema, d):
    """Encode schema tokens (SLOT -> sentinel '__SLOT__') as HV."""
    tokens = [t if t is not SLOT else '__SLOT__' for t in schema]
    return _encode(tokens, d)


class CrossModalBinder:
    """
    Joint skill+schema retrieval via triplet HV binding.

    bind_skill_schema(name, intent_tokens, proc_tokens, schema_examples)
        -- store modal_hv = intent x proc x schema_hv

    query(query_tokens) -> (name, score)
        -- cosine(unbind(modal, q), proc x schema)

    generate(query_tokens, *slot_args) -> (tokens, score)
        -- query + apply schema slots

    recover_proc(name, query_tokens) -> (proc_hv, sim)
        -- two-step unbind recovers proc_hv
    """

    def __init__(self, d=400):
        self.d = d
        self.inducer = SchemaInducer()
        self._bindings = {}   # name -> {intent_hv, proc_hv, schema_hv, modal_hv}

    # ── registration ─────────────────────────────────────────────────────

    def bind_skill_schema(self, name, intent_tokens, proc_tokens, schema_examples):
        """
        Register skill + schema as triplet binding.
        schema_examples: list of token-lists for SchemaInducer.
        Returns modal_hv dimension.
        """
        self.inducer.observe_many(name, schema_examples)

        intent_hv = _encode(intent_tokens, self.d)
        proc_hv   = _encode(proc_tokens,   self.d)
        schema    = self.inducer.induce(name, min_examples=2)
        schema_hv = _encode_schema(schema, self.d) if schema else proc_hv.copy()

        skill_hv  = _bind(intent_hv, proc_hv)
        modal_hv  = _bind(skill_hv, schema_hv)

        self._bindings[name] = {
            'intent_hv': intent_hv,
            'proc_hv':   proc_hv,
            'schema_hv': schema_hv,
            'skill_hv':  skill_hv,
            'modal_hv':  modal_hv,
        }
        return self.d

    # ── query ─────────────────────────────────────────────────────────────

    def query(self, query_tokens):
        """
        Find best match by score = cosine(modal x q, proc x schema).
        Returns (name, score).
        """
        if not self._bindings:
            return None, 0.0

        q_hv = _encode(query_tokens, self.d)
        best_name, best_score = None, -2.0

        for name, e in self._bindings.items():
            recovered = _bind(e['modal_hv'], q_hv)
            expected  = _bind(e['proc_hv'], e['schema_hv'])
            score = _sim(recovered, expected, self.d)
            if score > best_score:
                best_score = score
                best_name  = name

        return best_name, float(best_score)

    def recover_proc(self, name, query_tokens):
        """
        Two-step unbind: (modal x intent) x schema ~= proc.
        Returns (recovered_proc_hv, cosine_sim_vs_stored_proc).
        """
        e = self._bindings.get(name)
        if e is None:
            return None, 0.0
        q_hv        = _encode(query_tokens, self.d)
        proc_schema = _bind(e['modal_hv'], q_hv)
        recovered   = _bind(proc_schema, e['schema_hv'])
        sim         = _sim(recovered, e['proc_hv'], self.d)
        return recovered, float(sim)

    # ── generation ────────────────────────────────────────────────────────

    def generate(self, query_tokens, *slot_args):
        """
        Query cross-modal binding, fill schema slots with slot_args.
        Returns (token_list, score).
        """
        name, score = self.query(query_tokens)
        if name is None:
            return [], 0.0
        tokens = self.inducer.apply(name, *slot_args)
        return tokens, score

    def generate_from_name(self, name, *slot_args):
        """Direct generation bypassing query (known name)."""
        tokens = self.inducer.apply(name, *slot_args)
        schema = self.inducer.induce(name)
        return tokens, schema

    # ── stats ─────────────────────────────────────────────────────────────

    @property
    def n_bindings(self):
        return len(self._bindings)

    def binding_info(self, name):
        schema = self.inducer.induce(name)
        return {
            'dim':     self.d,
            'schema':  schema,
            'n_slots': schema.count(SLOT) if schema else 0,
            'support': self.inducer.support(name),
            'bound':   name in self._bindings,
        }
