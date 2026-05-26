"""
ikigai.cognition.world_model -- Symbolic World Model KB.

Day 55 Pack 46 -- Layer 3 Reasoning completion.

Problem: no factual recall. Every claim exists only in B_U belief HV.
         No explicit (entity, relation, entity) knowledge store.
         No inference over stored facts.

Fix: SymbolicWorldModel
     assert_fact(s, p, o)     -- store (s,p,o) triple, count monotone
     query(s, p, o)            -- pattern match, None = wildcard
     add_rule(head, body)      -- Datalog-style rule (uppercase = variable)
     infer(s, p, o, depth)     -- backward chaining

Backward chaining: try direct fact, then unify head of each rule with goal,
                   recursively prove body. Depth-limited to prevent loops.

No forgetting: fact counts monotone non-decreasing.
               assert_fact(s,p,o) twice -> count=2, never erased.

vs LLM: LLM factual knowledge = frozen at pretraining.
        WorldModel: grows from every conversation turn.
        query() = O(n_facts) exact lookup, not probabilistic.
"""

import numpy as np

_rule_counter = [0]


def _is_var(token):
    """Uppercase first char = Datalog variable (e.g. 'X', 'Y', 'Entity')."""
    return isinstance(token, str) and bool(token) and token[0].isupper()


def _rename_rule(head, body):
    """Fresh variable names per rule invocation -- prevents cross-invocation binding conflicts."""
    _rule_counter[0] += 1
    sfx = f'__{_rule_counter[0]}'
    def r(triple):
        return tuple(t + sfx if _is_var(t) else t for t in triple)
    return r(head), [r(b) for b in body]


def _match(pattern, ground):
    """
    Unify pattern triple with ground triple.
    pattern: (s, p, o) possibly with uppercase variables.
    ground:  (s, p, o) all ground terms.
    Returns binding dict or None on failure.
    """
    bindings = {}
    for p_tok, g_tok in zip(pattern, ground):
        if _is_var(p_tok):
            if p_tok in bindings:
                if bindings[p_tok] != g_tok:
                    return None
            else:
                bindings[p_tok] = g_tok
        elif p_tok != g_tok:
            return None
    return bindings


def _apply_bindings(triple, bindings):
    return tuple(bindings.get(t, t) for t in triple)


class SymbolicWorldModel:
    """
    Minimal Datalog KB: facts + backward-chaining rules.

    assert_fact(s, p, o)          -- add triple, count monotone
    query(s=None, p=None, o=None) -- wildcard match over fact store
    add_rule(head, body)          -- head :- body1, body2, ...
    infer(s, p, o, depth=5)       -- prove via facts + rules
    fact_count(s, p, o)           -- count matching facts
    """

    def __init__(self):
        self._facts = {}   # (s, p, o) -> int count (monotone)
        self._rules = []   # [(head_triple, [body_triples]), ...]

    #  facts

    def assert_fact(self, s, p, o):
        """Monotone assert. Returns new count."""
        key = (str(s), str(p), str(o))
        self._facts[key] = self._facts.get(key, 0) + 1
        return self._facts[key]

    def assert_facts(self, triples):
        """Bulk assert. Returns count of new unique triples."""
        new = 0
        for s, p, o in triples:
            if (str(s), str(p), str(o)) not in self._facts:
                new += 1
            self.assert_fact(s, p, o)
        return new

    def query(self, s=None, p=None, o=None):
        """
        Pattern match over fact store.
        None = wildcard. Returns list of matching (s, p, o) tuples.
        """
        results = []
        for (fs, fp, fo) in self._facts:
            if ((s is None or fs == s) and
                (p is None or fp == p) and
                (o is None or fo == o)):
                results.append((fs, fp, fo))
        return results

    def fact_exists(self, s, p, o):
        return (str(s), str(p), str(o)) in self._facts

    #  rules

    def add_rule(self, head, body):
        """
        head: (s, p, o) template -- uppercase tokens = variables.
        body: list of (s, p, o) templates sharing variables with head.

        Example (transitivity):
          add_rule(('X', 'is_a', 'Z'), [('X', 'is_a', 'Y'), ('Y', 'is_a', 'Z')])
        """
        self._rules.append((tuple(head), [tuple(b) for b in body]))

    #  inference

    def _prove(self, goals, bindings, depth, seen):
        """
        Prove a list of goals sequentially, threading bindings.
        goals: list of (s,p,o) triples possibly containing variables.
        Yields binding dicts on each successful proof.
        """
        if not goals:
            yield bindings
            return
        if depth <= 0:
            return

        raw, *rest = goals
        goal = _apply_bindings(raw, bindings)
        has_vars = any(_is_var(t) for t in goal)

        # Try matching against all known facts (handles variable goals)
        for fact in self._facts:
            m = _match(goal, fact)
            if m is not None:
                yield from self._prove(rest, {**bindings, **m}, depth, seen)

        # Try rules only for ground goals (avoids uncontrolled variable expansion)
        if not has_vars:
            if goal in seen:
                return
            new_seen = seen | {goal}
            for head, body in self._rules:
                r_head, r_body = _rename_rule(head, body)
                m = _match(r_head, goal)
                if m is None:
                    continue
                yield from self._prove(
                    list(r_body) + list(rest),
                    {**bindings, **m},
                    depth - 1,
                    new_seen,
                )

    def infer(self, s, p, o, depth=5, _seen=None):
        """
        Backward chaining: prove (s, p, o) via facts + rules.
        depth: recursion limit (prevents infinite loops).
        Returns True if provable, False otherwise.
        """
        goal = (str(s), str(p), str(o))
        seen = set() if _seen is None else _seen
        for _ in self._prove([goal], {}, depth, seen):
            return True
        return False

    def query_inferred(self, s=None, p=None, o=None, depth=3):
        """
        Return facts provable (directly or via rules) matching pattern.
        More expensive than query() -- uses infer() per candidate.
        """
        direct = self.query(s, p, o)
        return direct  # direct for now; infer() used for specific goals

    #  learning from conversation

    def learn_from_tokens(self, tokens, role='subject'):
        """
        Simple triple extraction from token window: [w_i, w_{i+1}, w_{i+2}]
        as (subject, relation, object). Heuristic, not semantic parsing.
        """
        added = 0
        for i in range(len(tokens) - 2):
            self.assert_fact(tokens[i], tokens[i+1], tokens[i+2])
            added += 1
        return added

    #  stats

    def fact_count(self, s=None, p=None, o=None):
        return len(self.query(s, p, o))

    @property
    def n_facts(self):
        return len(self._facts)

    @property
    def n_rules(self):
        return len(self._rules)

    @property
    def total_assertions(self):
        return sum(self._facts.values())

    def summary(self):
        return {
            'n_facts':          self.n_facts,
            'n_rules':          self.n_rules,
            'total_assertions': self.total_assertions,
        }
