"""
ikigai.cognition.reasoning — Pure VSA reasoning primitives.

Houses (Day 54 Packs 13-17):
    RelationAlgebra       — analogical reasoning + relation composition/inversion
                            (Packs 14, 15). Used for A:B::C:? and R1 (X) R2.
    SemanticRoleMemory    — agent/action/patient role-filler binding (Pack 16).
    EventSequenceMemory   — N events compressed into one positional-bound HV
                            (Pack 17).
    PatternComposer       — algebraic pattern detection + composition + code gen
                            (Pack 13). detect/compose/synthesize functions.

All classes operate on bipolar (+-1) HVs and use majority bundling.
No gradient descent. No transformer. Pure VSA + ndarray.
"""

import random
import re
import numpy as np


HV_DIM = 400


# ── Shared HV primitives ──────────────────────────────────────────────────────

def make_bipolar_hv(seed_str, dim=HV_DIM):
    rng = random.Random(hash(seed_str) & 0x7FFFFFFF)
    return np.array(
        [1 if rng.randint(0, 1) else -1 for _ in range(dim)],
        dtype=np.int8,
    )


def make_binary_hv(seed_str, dim=HV_DIM):
    rng = random.Random(hash(seed_str) & 0x7FFFFFFF)
    return np.array([rng.randint(0, 1) for _ in range(dim)], dtype=np.uint8)


def hamming_sim(a, b):
    return float(np.mean(a == b))


def cosine_sim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── RelationAlgebra (Packs 14, 15) ────────────────────────────────────────────

class RelationAlgebra:
    """
    Binary-HV analogical reasoning + relation composition/decomposition.
    For all pairs (A, B) in a relation R: vocab[B] = vocab[A] XOR R (exact).
    Analogy query: A:B::C:?  =  HV(C) XOR HV(A) XOR HV(B)  =  HV(C) XOR R
    """

    def __init__(self, dim=HV_DIM):
        self.dim = dim
        self.vocab = {}
        self.relation_hvs = {}

    def _make_hv(self, key):
        return make_binary_hv(key, self.dim)

    def add_item(self, key):
        if key not in self.vocab:
            self.vocab[key] = self._make_hv(key)

    def define_relation(self, name, pairs):
        """B = A XOR R for every (A, B) pair. R is a dedicated random HV."""
        R = self._make_hv(f'__rel__{name}')
        self.relation_hvs[name] = R
        for a, b in pairs:
            self.add_item(a)
            self.vocab[b] = np.bitwise_xor(self.vocab[a], R)
        return R

    def add_to_relation(self, name, a, b):
        R = self.relation_hvs[name]
        self.add_item(a)
        self.vocab[b] = np.bitwise_xor(self.vocab[a], R)

    def compose_relations(self, r1, r2, new_name):
        """R_new = R1 XOR R2.  C = B XOR R2 = (A XOR R1) XOR R2 = A XOR R_new."""
        R_new = np.bitwise_xor(self.relation_hvs[r1], self.relation_hvs[r2])
        self.relation_hvs[new_name] = R_new
        return R_new

    def decompose_relation(self, composed, known, new_name):
        """R_unknown = R_composed XOR R_known (XOR self-inverse)."""
        R_new = np.bitwise_xor(
            self.relation_hvs[composed], self.relation_hvs[known]
        )
        self.relation_hvs[new_name] = R_new
        return R_new

    def apply_relation(self, name, item_key):
        return np.bitwise_xor(self.vocab[item_key], self.relation_hvs[name])

    def solve_analogy(self, a, b, c, exclude=None):
        """A:B::C:?  query = HV(C) XOR HV(A) XOR HV(B)."""
        ha = self.vocab[a]; hb = self.vocab[b]; hc = self.vocab[c]
        query = np.bitwise_xor(np.bitwise_xor(ha, hb), hc)
        excl = {a, b, c} | (set(exclude or []))
        best, best_sim = None, -1.0
        for k, h in self.vocab.items():
            if k in excl:
                continue
            s = hamming_sim(query, h)
            if s > best_sim:
                best_sim = s; best = k
        return best, best_sim

    def identify_relation(self, a, b):
        R_obs = np.bitwise_xor(self.vocab[a], self.vocab[b])
        best, best_sim = None, -1.0
        for name, R in self.relation_hvs.items():
            s = hamming_sim(R_obs, R)
            if s > best_sim:
                best_sim = s; best = name
        return best, best_sim

    def relation_sim(self, r1, r2):
        return hamming_sim(self.relation_hvs[r1], self.relation_hvs[r2])


# ── SemanticRoleMemory (Pack 16) ──────────────────────────────────────────────

class SemanticRoleMemory:
    """
    Encode (agent, action, patient) triples as ONE HV via role-filler binding.
        event_hv = bundle(R_agent XOR HV(agent), R_action XOR HV(action),
                          R_patient XOR HV(patient))
    Query: event XOR R_role  ~  HV(filler)   [sim ~ 0.75, reliable retrieval]
    """

    def __init__(self, dim=HV_DIM):
        self.dim = dim
        self.vocab = {}
        self.roles = {}
        self.events = {}
        self._nid = 0

    def _make_hv(self, key):
        return make_binary_hv(key, self.dim)

    def add_item(self, w):
        if w not in self.vocab:
            self.vocab[w] = self._make_hv(w)

    def define_roles(self, names):
        for n in names:
            self.roles[n] = self._make_hv(f'__role__{n}')

    def _bind(self, a, b):
        return np.bitwise_xor(a, b)

    def _bundle(self, hvs):
        s = np.zeros(self.dim, dtype=np.int32)
        for h in hvs:
            s += h.astype(np.int32)
        return (s > len(hvs) // 2).astype(np.uint8)

    def encode_event(self, agent, action, patient):
        for w in (agent, action, patient):
            self.add_item(w)
        return self._bundle([
            self._bind(self.roles['agent'],   self.vocab[agent]),
            self._bind(self.roles['action'],  self.vocab[action]),
            self._bind(self.roles['patient'], self.vocab[patient]),
        ])

    def store_event(self, agent, action, patient):
        ev = self.encode_event(agent, action, patient)
        eid = self._nid; self._nid += 1
        self.events[eid] = (agent, action, patient, ev)
        return eid, ev

    def query_role(self, event_hv, role_name, exclude=None):
        query = np.bitwise_xor(event_hv, self.roles[role_name])
        excl = set(exclude or [])
        best, best_sim = None, -1.0
        for name, hv in self.vocab.items():
            if name in excl:
                continue
            s = hamming_sim(query, hv)
            if s > best_sim:
                best_sim = s; best = name
        return best, best_sim


# ── EventSequenceMemory (Pack 17) ─────────────────────────────────────────────

class EventSequenceMemory:
    """
    Compress N events into one HV via positional binding:
        seq = bundle(P[0] XOR ev[0], P[1] XOR ev[1], ..., P[N-1] XOR ev[N-1])
    Step query: ev_approx = seq XOR P[k], then NN over stored events for the
    discrete event_id, then exact role lookup.
    """

    def __init__(self, dim=HV_DIM):
        self.dim = dim
        self.vocab = {}
        self.roles = {}
        self.pos_hvs = []
        self.event_store = {}
        self.sequence_hv = None

    def _make_hv(self, key):
        return make_binary_hv(key, self.dim)

    def _pos(self, k):
        while len(self.pos_hvs) <= k:
            self.pos_hvs.append(self._make_hv(f'__pos__{len(self.pos_hvs)}'))
        return self.pos_hvs[k]

    def add_item(self, w):
        if w not in self.vocab:
            self.vocab[w] = self._make_hv(w)

    def define_roles(self, names):
        for n in names:
            self.roles[n] = self._make_hv(f'__role__{n}')

    def _bind(self, a, b):
        return np.bitwise_xor(a, b)

    def _bundle(self, hvs):
        s = np.zeros(self.dim, dtype=np.int32)
        for h in hvs:
            s += h.astype(np.int32)
        return (s > len(hvs) // 2).astype(np.uint8)

    def _encode_event(self, agent, action, patient):
        for w in (agent, action, patient):
            self.add_item(w)
        return self._bundle([
            self._bind(self.roles['agent'],   self.vocab[agent]),
            self._bind(self.roles['action'],  self.vocab[action]),
            self._bind(self.roles['patient'], self.vocab[patient]),
        ])

    def encode_sequence(self, events):
        bound = []
        for k, (ag, act, pat) in enumerate(events):
            ev_hv = self._encode_event(ag, act, pat)
            self.event_store[k] = (ag, act, pat, ev_hv)
            bound.append(self._bind(self._pos(k), ev_hv))
        self.sequence_hv = self._bundle(bound)
        return self.sequence_hv

    def query_step(self, k):
        ev_approx = self._bind(self.sequence_hv, self._pos(k))
        best_k, best_sim = None, -1.0
        for eid, (*_, ev_hv) in self.event_store.items():
            s = hamming_sim(ev_approx, ev_hv)
            if s > best_sim:
                best_sim = s; best_k = eid
        ag, act, pat, _ = self.event_store[best_k]
        return best_k, ag, act, pat, best_sim

    def query_step_role(self, k, role_name):
        ev_approx = self._bind(self.sequence_hv, self._pos(k))
        best_k, best_sim = None, -1.0
        for eid, (*_, ev_hv) in self.event_store.items():
            s = hamming_sim(ev_approx, ev_hv)
            if s > best_sim:
                best_sim = s; best_k = eid
        _, _, _, exact_ev = self.event_store[best_k]
        query = self._bind(exact_ev, self.roles[role_name])
        best, best_s = None, -1.0
        for name, hv in self.vocab.items():
            s = hamming_sim(query, hv)
            if s > best_s:
                best_s = s; best = name
        return best, best_s


# ── PatternComposer (Pack 13) ─────────────────────────────────────────────────

def detect_pattern(examples):
    """
    Infer algebraic pattern from (x, y) pairs. Returns (name, params, fn) or None.
    Supported: negate, halve, square, mul_k, add_k, sub_k, mod_k, linear.
    """
    if len(examples) < 2:
        return None
    xs = [e[0] for e in examples]
    ys = [e[1] for e in examples]

    if all(abs(y - (-x)) < 1e-6 for x, y in examples):
        return ('negate', {}, lambda x: -x)
    if all(isinstance(x, int) and y == x // 2 for x, y in examples):
        return ('halve', {}, lambda x: x // 2)
    if all(abs(y - x ** 2) < 1e-6 for x, y in examples):
        return ('square', {}, lambda x: x ** 2)
    if xs[0] != 0:
        k = ys[0] / xs[0]
        if k != 1.0 and k != 0.0 and all(abs(y - k * x) < 1e-6 for x, y in examples):
            ki = round(k)
            return ('mul_k', {'k': ki}, lambda x, _k=ki: x * _k)
    k = ys[0] - xs[0]
    if k != 0 and all(abs(y - (x + k)) < 1e-6 for x, y in examples):
        ki = round(k)
        return ('add_k', {'k': ki}, lambda x, _k=ki: x + _k)
    k = xs[0] - ys[0]
    if k > 0 and all(abs(y - (x - k)) < 1e-6 for x, y in examples):
        ki = round(k)
        return ('sub_k', {'k': ki}, lambda x, _k=ki: x - _k)
    for k in range(2, 21):
        if all(isinstance(x, int) and y == x % k for x, y in examples):
            return ('mod_k', {'k': k}, lambda x, _k=k: x % _k)
    if len(examples) >= 2 and xs[1] != xs[0]:
        a = (ys[1] - ys[0]) / (xs[1] - xs[0])
        b = ys[0] - a * xs[0]
        if (abs(a - round(a)) < 1e-6 and abs(b - round(b)) < 1e-6
                and all(abs(y - (a * x + b)) < 1e-6 for x, y in examples)):
            ai, bi = round(a), round(b)
            return ('linear', {'a': ai, 'b': bi}, lambda x, _a=ai, _b=bi: _a * x + _b)
    return None


def compose(fn_a, fn_b, order):
    """order='a_of_b' -> fn_a(fn_b(x)). 'b_of_a' -> fn_b(fn_a(x))."""
    if order == 'a_of_b':
        return lambda x: fn_a(fn_b(x))
    return lambda x: fn_b(fn_a(x))


def synthesize_composition(examples_a, examples_b, test_cases):
    """Detect A and B from examples, try both compositions, return matching one."""
    ra = detect_pattern(examples_a)
    rb = detect_pattern(examples_b)
    if ra is None or rb is None:
        return None, None, ra, rb
    for order in ('a_of_b', 'b_of_a'):
        fn = compose(ra[2], rb[2], order)
        try:
            if all(abs(fn(x) - y) < 1e-6 for x, y in test_cases):
                return fn, order, ra, rb
        except Exception:
            pass
    return None, None, ra, rb


def _slot_expr(name, params, var):
    if name == 'negate':  return f'-({var})'
    if name == 'halve':   return f'({var}) // 2'
    if name == 'square':  return f'({var}) ** 2'
    if name == 'mul_k':   return f'({var}) * {params["k"]}'
    if name == 'add_k':   return f'({var}) + {params["k"]}'
    if name == 'sub_k':   return f'({var}) - {params["k"]}'
    if name == 'mod_k':   return f'({var}) % {params["k"]}'
    if name == 'linear':  return f'{params["a"]} * ({var}) + {params["b"]}'
    return var


def generate_composition_code(ra, rb, order, func_name='composed'):
    name_a, params_a, _ = ra
    name_b, params_b, _ = rb
    if order == 'a_of_b':
        inner = _slot_expr(name_b, params_b, 'x')
        body  = _slot_expr(name_a, params_a, f'({inner})')
    else:
        inner = _slot_expr(name_a, params_a, 'x')
        body  = _slot_expr(name_b, params_b, f'({inner})')
    return f'def {func_name}(x):\n    return {body}'
