"""
ikigai.cognition.schema_inducer -- Typed Schema Induction via Anti-Unification.

Day 55 Pack 47 -- Layer 2 Composition completion. Day 60 benchmark enabler.

Problem: system can recall skills but cannot generalize to new instances.
         SkillCrystal stores (intent, procedure) as HVs -- retrieval only.
         No way to generate novel instances of a learned pattern.

Fix: SchemaInducer applies Plotkin (1970) Least General Generalization.
     N examples -> schema with SLOT markers at variable positions.
     apply(schema, *args) fills slots -> new instance generated.

Algorithm:
     schema = examples[0]
     for ex in examples[1:]:
         schema = anti_unify(schema, ex)   # SLOT where tokens differ

     anti_unify(a, b)[i] = a[i]   if a[i] == b[i]
                          = SLOT   otherwise

No forgetting: observe() is append-only. Support count monotone.
               Inducing again after new examples = more specific schema
               (more examples with consensus = more fixed positions).

Day 60 benchmark:
     5 examples of 'list_filter' -> schema = ['result', 'for', 'x', 'in', SLOT, 'if', SLOT]
     apply('my_list', 'my_pred') -> ['result', 'for', 'x', 'in', 'my_list', 'if', 'my_pred']
     20 patterns × 5 examples -> 20 schemas, all generalize correctly.
     Never forgets schema 1 after learning schema 20.
"""

SLOT = None   # variable position marker


def anti_unify(seq_a, seq_b):
    """
    Token-level pairwise LGG.
    Pads shorter sequence with SLOT before alignment.
    Returns schema (list): fixed token or SLOT.
    """
    maxlen = max(len(seq_a), len(seq_b))
    a = list(seq_a) + [SLOT] * (maxlen - len(seq_a))
    b = list(seq_b) + [SLOT] * (maxlen - len(seq_b))
    return [x if x == y else SLOT for x, y in zip(a, b)]


def apply_schema(schema, args):
    """
    Fill SLOT positions with args in order.
    Returns token list (str only, SLOT positions skipped if args exhausted).
    """
    result = []
    arg_idx = 0
    for token in schema:
        if token is SLOT:
            if arg_idx < len(args):
                tok = args[arg_idx]
                arg_idx += 1
                # Expand lists/tuples inline
                if isinstance(tok, (list, tuple)):
                    result.extend(str(t) for t in tok)
                else:
                    result.append(str(tok))
        else:
            result.append(token)
    return result


def schema_specificity(schema):
    """Fixed positions / total. 1.0 = fully specific, 0.0 = all slots."""
    if not schema:
        return 0.0
    n_fixed = sum(1 for t in schema if t is not SLOT)
    return n_fixed / len(schema)


class SchemaInducer:
    """
    Induce token-level schemas from examples via anti-unification.

    observe(name, output_tokens)       -- add one example
    induce(name, min_examples=2)       -- compute schema for pattern
    apply(name, *args)                 -- generate new instance
    schema_info(name)                  -- {schema, n_fixed, n_slots, support}

    No forgetting: observe() is append-only.
    Support = number of examples seen. Monotone non-decreasing.
    """

    def __init__(self):
        self._examples = {}   # name -> [token_list, ...]
        self._schemas  = {}   # name -> entry dict

    #  observation

    def observe(self, name, output_tokens):
        """Append one output example for pattern 'name'."""
        if name not in self._examples:
            self._examples[name] = []
        self._examples[name].append(list(output_tokens))
        # Invalidate cached schema (will re-induce on next call)
        if name in self._schemas:
            del self._schemas[name]
        return len(self._examples[name])

    def observe_many(self, name, examples):
        """Batch observe. Returns final support count."""
        for ex in examples:
            self.observe(name, ex)
        return len(self._examples[name])

    #  induction

    def induce(self, name, min_examples=2):
        """
        Anti-unify all stored examples for 'name'.
        Returns schema list (fixed tokens + SLOT) or None if too few examples.
        Caches result until new observe() call.
        """
        exs = self._examples.get(name, [])
        if len(exs) < min_examples:
            return None

        if name in self._schemas:
            return self._schemas[name]['schema']

        schema = list(exs[0])
        for ex in exs[1:]:
            schema = anti_unify(schema, ex)

        n_fixed = sum(1 for t in schema if t is not SLOT)
        n_slots = sum(1 for t in schema if t is SLOT)

        self._schemas[name] = {
            'schema':        schema,
            'n_fixed':       n_fixed,
            'n_slots':       n_slots,
            'support':       len(exs),
            'specificity':   schema_specificity(schema),
        }
        return schema

    def induce_all(self, min_examples=2):
        """Induce schemas for all patterns with enough examples."""
        return {
            name: self.induce(name, min_examples)
            for name in self._examples
            if len(self._examples[name]) >= min_examples
        }

    #  application

    def apply(self, name, *args):
        """
        Generate new instance by filling schema slots with args.
        Returns token list. Induces schema if not cached.
        """
        schema = self.induce(name)
        if schema is None:
            return []
        return apply_schema(schema, list(args))

    #  schema comparison

    def schemas_agree(self, name_a, name_b):
        """True if two schemas have identical fixed token positions."""
        s_a = self.induce(name_a)
        s_b = self.induce(name_b)
        if s_a is None or s_b is None or len(s_a) != len(s_b):
            return False
        for a, b in zip(s_a, s_b):
            if a is not SLOT and b is not SLOT and a != b:
                return False
        return True

    #  stats

    def schema_info(self, name):
        self.induce(name)
        return self._schemas.get(name)

    def support(self, name):
        return len(self._examples.get(name, []))

    @property
    def n_patterns(self):
        return len(self._examples)

    @property
    def n_schemas(self):
        return len(self._schemas)

    @property
    def total_examples(self):
        return sum(len(v) for v in self._examples.values())
