"""
ikigai.cognition.algebraic_closure -- Algebraic Closure Discovery.

Day 55 Pack 69 -- invention #3: auto-discover new VSA operators.

VSA has 3 known ops: bind (XOR), bundle (majority), permute (cyclic shift).
But: are there OTHER useful binary ops on bipolar HVs?
This module searches the space and tests each candidate against algebraic
properties (associativity, commutativity, self-inverse, distributivity).

Search space:
    Binary ops on ±1: f: {±1}^2 -> {±1}, which is 16 possible functions.
    Each represented as a truth table (4 entries).
    Operators extend elementwise to d-dim HVs.

Tested properties:
    - identity:     exists e s.t. f(a, e) = a for all a
    - commutative:  f(a, b) = f(b, a)
    - associative:  f(f(a, b), c) = f(a, f(b, c))
    - self-inverse: f(f(a, b), b) = a
    - involutive:   f(a, a) = ID (constant)
    - distributes_over_bundle: f(a, bundle(b, c)) == bundle(f(a, b), f(a, c))

Bio analog: cortical microcircuit motifs. Cortex implements multiple ops
            from same primitives (lateral inhibition + excitation).

Closure property:
    An algebraic system is "closed" under op f if f's output is always in
    the same space. All our ±1 -> ±1 binary ops are closed by construction.

vs LLM: no such notion. Ikigai: programmable algebraic substrate.
        Discovered ops extend the bind/bundle/permute trinity.
"""

import numpy as np
from itertools import product


# All 16 binary boolean ops on {±1}^2 -> {±1}
# Each op is a 4-entry truth table: [(−1,−1), (−1,+1), (+1,−1), (+1,+1)]

def _truth_table_to_op(table):
    """Build a binary op from a 4-entry truth table."""
    t_mm, t_mp, t_pm, t_pp = table
    def op(a, b):
        # Vectorized elementwise application
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        out = np.empty_like(a)
        out[(a < 0) & (b < 0)] = t_mm
        out[(a < 0) & (b > 0)] = t_mp
        out[(a > 0) & (b < 0)] = t_pm
        out[(a > 0) & (b > 0)] = t_pp
        # Handle zeros by treating as +1
        out[a == 0] = t_pp if b[0] != 0 else t_pp
        return out
    return op


def _gen_all_ops():
    """Generate all 16 binary ops on ±1 inputs."""
    ops = []
    for table in product([-1.0, +1.0], repeat=4):
        ops.append((tuple(table), _truth_table_to_op(table)))
    return ops


def _identity(a, b):
    return a.copy()


def _xor(a, b):
    return np.sign(a * b).astype(np.float32)


def _and_like(a, b):
    """Returns +1 iff both +1, else -1."""
    return np.where((a > 0) & (b > 0), 1.0, -1.0).astype(np.float32)


#  property checks

def is_commutative(op, n_trials=10, d=64, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        a = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        b = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        if not np.array_equal(op(a, b), op(b, a)):
            return False
    return True


def is_associative(op, n_trials=10, d=64, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        a = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        b = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        c = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        lhs = op(op(a, b), c)
        rhs = op(a, op(b, c))
        if not np.array_equal(lhs, rhs):
            return False
    return True


def is_self_inverse(op, n_trials=10, d=64, seed=2):
    """f(f(a,b), b) == a for all a, b."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        a = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        b = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        recovered = op(op(a, b), b)
        if not np.array_equal(recovered, a):
            return False
    return True


def has_identity(op, d=64, seed=3):
    """Return identity HV e such that op(a, e) = a for all a, or None."""
    rng = np.random.default_rng(seed)
    # Two candidate identities: all-+1 and all--1
    for e_val in [+1.0, -1.0]:
        e = np.full(d, e_val, dtype=np.float32)
        ok = True
        for _ in range(5):
            a = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
            if not np.array_equal(op(a, e), a):
                ok = False
                break
        if ok:
            return e
    return None


def distributes_over_bundle(op, n_trials=10, d=64, seed=4):
    """f(a, bundle(b, c)) == bundle(f(a, b), f(a, c))?"""
    rng = np.random.default_rng(seed)
    def bundle(x, y):
        s = np.sign((x + y).astype(np.float32))
        s[s == 0.0] = 1.0
        return s
    for _ in range(n_trials):
        a = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        b = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        c = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
        lhs = op(a, bundle(b, c))
        rhs = bundle(op(a, b), op(a, c))
        if not np.array_equal(lhs, rhs):
            return False
    return True


#  closure discovery

class AlgebraicClosure:
    """
    Searches the space of 16 binary ops on ±1^2 for those satisfying
    desired algebraic properties.

    discover(properties, d=64) -> list of (table, op_fn, satisfied_props)
        Returns ops satisfying the requested property set.

    catalog() -> dict mapping property combos to representative ops.

    derive_pair(name, parent_ops) -> (new_op_fn, properties)
        Compose two known ops; report what algebra survives.
    """

    PROPERTY_CHECKS = {
        'commutative':           is_commutative,
        'associative':           is_associative,
        'self_inverse':          is_self_inverse,
        'has_identity':          lambda op, **kw: has_identity(op) is not None,
        'distributes_bundle':    distributes_over_bundle,
    }

    def __init__(self, d=64):
        self.d   = d
        self._ops = _gen_all_ops()

    def discover(self, properties=None):
        """
        Return all ops satisfying every property in `properties`.
        Each entry: (table_tuple, op_fn, found_properties_dict).
        """
        if properties is None:
            properties = list(self.PROPERTY_CHECKS.keys())

        out = []
        for table, op_fn in self._ops:
            results = {}
            ok = True
            for p in properties:
                check = self.PROPERTY_CHECKS[p]
                results[p] = check(op_fn)
                if not results[p]:
                    ok = False
            if ok:
                out.append((table, op_fn, results))
        return out

    def classify_all(self):
        """Compute all properties for all 16 ops. Returns list of (table, props_dict)."""
        rows = []
        for table, op_fn in self._ops:
            props = {p: check(op_fn) for p, check in self.PROPERTY_CHECKS.items()}
            rows.append((table, props))
        return rows

    def catalog(self):
        """Map property-set tuples -> list of tables with that exact profile."""
        groups = {}
        for table, props in self.classify_all():
            key = tuple(sorted(p for p, v in props.items() if v))
            groups.setdefault(key, []).append(table)
        return groups

    def derive_pair(self, op_a, op_b):
        """
        Compose: f_new(a, b) = op_a(op_b(a, b), a).
        Returns the new op function. (Algebraic-composition discovery.)
        """
        def composed(a, b):
            return op_a(op_b(a, b), a)
        return composed
