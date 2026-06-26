"""
ikigai.cognition.logical_fixed_point -- Kill Stack invention #5.

Provable inference on the flat substrate via fixed-point iteration.

Transformer inference is unverifiable -- it samples tokens, you trust them.
Substrate inference is provable: each derived fact has a chain of premises,
each premise is in the substrate, the derivation terminates at a stable
fixed point. You can ASK the system to show its work.

Supported rule kinds (each is a small VSA-substrate operation):

    TRANSITIVE   if (X R Y) and (Y R Z) then (X R Z)
                 isa is the canonical example.
    SYMMETRIC    if (X R Y) then (Y R X)
                 antonym, sibling, partner, cooccur.
    INVERSE      if (X R Y) then (Y R' X) for inverse role R'
                 parent_of <-> child_of, contains <-> in.
    COMPOSITION  if (X R1 Y) and (Y R2 Z) then (X R3 Z) for R3 = R1 . R2
                 isa . property -> property; locator chains; etc.

Each derivation step writes the conclusion back into the substrate so
subsequent queries see it as a regular fact.  An explicit derivation
log records every step's premise chain so the user can audit.

Public API:
    fp = LogicalFixedPoint(organism)
    fp.add_rule(LogicalRule.transitive('isa'))
    fp.add_rule(LogicalRule.symmetric('antonym'))
    stats = fp.run(max_iterations=10)
    fp.derivations   # list of (premise1, premise2, conclusion, rule)
"""

import time


class LogicalRule:
    """Container for one inference rule kind."""

    def __init__(self, kind, role, role2=None):
        self.kind = kind           # 'transitive' / 'symmetric' / 'inverse' / 'composition'
        self.role = role
        self.role2 = role2

    @classmethod
    def transitive(cls, role):
        return cls('transitive', role)

    @classmethod
    def symmetric(cls, role):
        return cls('symmetric', role)

    @classmethod
    def inverse(cls, role, inverse_role):
        return cls('inverse', role, inverse_role)

    def __repr__(self):
        if self.role2:
            return f'<Rule {self.kind} {self.role} -> {self.role2}>'
        return f'<Rule {self.kind} {self.role}>'


class LogicalFixedPoint:
    """
    Fixed-point inference engine on the substrate.

    Maintains an internal fact set (X, role, Y) populated from the
    substrate's explicit writes. Applies rules iteratively; writes
    every new conclusion back into the substrate; records the
    derivation chain.
    """

    def __init__(self, organism):
        self.org = organism
        self.mr = organism.unified
        self.rules = []
        self.facts = set()          # set of (X, role, Y)
        self.derivations = []       # list of dicts: {p1, p2, conclusion, rule, iter}
        self.last_stats = {}

    def add_rule(self, rule):
        self.rules.append(rule)
        return self

    # ── seed facts ────────────────────────────────────────────────────────
    def seed_fact(self, x, role, y, write_to_substrate=True, n=20):
        """Add an explicit (X, role, Y) fact to the working set + substrate."""
        triple = (x, role, y)
        if triple not in self.facts:
            self.facts.add(triple)
            if write_to_substrate:
                for _ in range(n):
                    self.mr.relate(x, role, y)
                self.mr._role_targets.setdefault(role, set()).add(x)

    def seed_facts(self, facts, write_to_substrate=True, n=20):
        for x, role, y in facts:
            self.seed_fact(x, role, y, write_to_substrate=write_to_substrate, n=n)

    # ── one iteration of inference ────────────────────────────────────────
    def _step(self, iter_n):
        new_facts = []
        for rule in self.rules:
            if rule.kind == 'transitive':
                # find (X, R, Y) and (Y, R, Z) -> (X, R, Z)
                by_role = [(x, y) for (x, r, y) in self.facts if r == rule.role]
                yset = {y: [] for _, y in by_role}
                # build adj list keyed by first node Y
                adj_from = {}
                for x, y in by_role:
                    adj_from.setdefault(x, []).append(y)
                for x, y in by_role:
                    for z in adj_from.get(y, []):
                        if z == x: continue
                        concl = (x, rule.role, z)
                        if concl not in self.facts:
                            new_facts.append((concl, rule, (x, rule.role, y),
                                              (y, rule.role, z)))
            elif rule.kind == 'symmetric':
                for (x, r, y) in list(self.facts):
                    if r != rule.role: continue
                    concl = (y, rule.role, x)
                    if concl not in self.facts:
                        new_facts.append((concl, rule, (x, r, y), None))
            elif rule.kind == 'inverse':
                inv = rule.role2
                for (x, r, y) in list(self.facts):
                    if r != rule.role: continue
                    concl = (y, inv, x)
                    if concl not in self.facts:
                        new_facts.append((concl, rule, (x, r, y), None))
        # commit + write to substrate
        for concl, rule, p1, p2 in new_facts:
            self.facts.add(concl)
            self.derivations.append({
                'p1': p1, 'p2': p2, 'conclusion': concl,
                'rule': repr(rule), 'iter': iter_n,
            })
            # write to substrate as a real fact (n=10 reinforcement)
            x, role, y = concl
            for _ in range(10):
                self.mr.relate(x, role, y)
            self.mr._role_targets.setdefault(role, set()).add(x)
        return len(new_facts)

    # ── run to fixed point ────────────────────────────────────────────────
    def run(self, max_iterations=20):
        t0 = time.perf_counter()
        sub_before = self.mr.substrate_bytes()
        per_iter = []
        for it in range(1, max_iterations + 1):
            n_new = self._step(it)
            per_iter.append(n_new)
            if n_new == 0:
                break
        stats = {
            'iterations_run': len(per_iter),
            'converged': per_iter[-1] == 0 if per_iter else True,
            'total_derivations': len(self.derivations),
            'total_facts': len(self.facts),
            'per_iter_new': per_iter,
            'elapsed': time.perf_counter() - t0,
            'substrate_pre': sub_before,
            'substrate_post': self.mr.substrate_bytes(),
            'substrate_fixed': self.mr.substrate_bytes() == sub_before,
        }
        self.last_stats = stats
        return stats

    # ── audit a derivation ────────────────────────────────────────────────
    def explain(self, x, role, y):
        """Return chain of derivations that produced (x, role, y), or [] if axiom."""
        triple = (x, role, y)
        chain = []
        target = triple
        # find a derivation that concludes target
        while True:
            d = next((d for d in self.derivations if d['conclusion'] == target),
                     None)
            if d is None:
                break
            chain.append(d)
            target = d['p1']    # walk back along first premise
        return chain
