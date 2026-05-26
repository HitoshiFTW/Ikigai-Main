"""
ikigai.cognition.dssc_coupling -- Dual-Stream Semantic/Syntactic Coupling.

Day 55 Pack 33 -- conversational substrate primitive #2.
Replaces free-text autoregressive drift with grammar-locked emit.

Two parallel streams compose every utterance:
    S in R^{d_sem}            fluid semantic intent (continuous HV)
    G in CFG nodes            rigid syntactic AST node (one-hot)

Third-order tensor coupling:
    T_couple in R^{d_sem x |V| x d_emit}
    emit_t = einsum('i,j,ijk->k', S_t, G_t_onehot, T_couple)
           = T_couple[:, G_t, :].T @ S_t          (when G is one-hot)

Grammar pick: argmax over CFG successors of cosine(S, proto_g).
Learning: Hebbian outer product. No gradient. No backprop.

Invariant: every emit lies on grammar manifold. No ill-formed output possible.
"""

import math
import time
import numpy as np


# -----------------------------------------------------------------------
# Tiny CFG with 16 dialogue-act nodes
# -----------------------------------------------------------------------

class GrammarCFG:
    """Minimal CFG: nodes (named), edges (allowed successors), one start."""

    def __init__(self):
        self.nodes = {}          # name -> id
        self.id_to_name = {}     # id -> name
        self.successors = {}     # id -> list of ids
        self.start = None

    def add_node(self, name):
        if name not in self.nodes:
            idx = len(self.nodes)
            self.nodes[name] = idx
            self.id_to_name[idx] = name
            self.successors[idx] = []
        return self.nodes[name]

    def add_edge(self, from_name, to_name):
        a = self.add_node(from_name)
        b = self.add_node(to_name)
        if b not in self.successors[a]:
            self.successors[a].append(b)

    def set_start(self, name):
        self.start = self.add_node(name)

    def valid_successors(self, current_id):
        return self.successors.get(current_id, [])

    def name(self, node_id):
        return self.id_to_name.get(node_id, f'<id_{node_id}>')

    def __len__(self):
        return len(self.nodes)


def build_default_cfg():
    """16-node conversational dialogue-act grammar."""
    g = GrammarCFG()
    nodes = [
        'GREET', 'IDENTIFY', 'QUERY', 'CLARIFY',
        'STATEMENT', 'AFFIRM', 'NEGATE', 'CONJUNCT',
        'CAUSE', 'EFFECT', 'EXAMPLE', 'COMPARE',
        'CONCLUDE', 'QUESTION', 'ACKNOWLEDGE', 'FAREWELL',
    ]
    for n in nodes:
        g.add_node(n)
    g.set_start('GREET')
    edges = [
        ('GREET', 'IDENTIFY'), ('GREET', 'QUERY'), ('GREET', 'STATEMENT'),
        ('IDENTIFY', 'QUERY'), ('IDENTIFY', 'STATEMENT'),
        ('QUERY', 'CLARIFY'), ('QUERY', 'STATEMENT'),
        ('QUERY', 'AFFIRM'), ('QUERY', 'NEGATE'),
        ('CLARIFY', 'STATEMENT'), ('CLARIFY', 'QUERY'),
        ('STATEMENT', 'CONJUNCT'), ('STATEMENT', 'CAUSE'),
        ('STATEMENT', 'EXAMPLE'), ('STATEMENT', 'COMPARE'),
        ('STATEMENT', 'CONCLUDE'), ('STATEMENT', 'QUESTION'),
        ('AFFIRM', 'STATEMENT'), ('AFFIRM', 'CONCLUDE'),
        ('NEGATE', 'CAUSE'), ('NEGATE', 'STATEMENT'),
        ('CONJUNCT', 'STATEMENT'),
        ('CAUSE', 'EFFECT'), ('EFFECT', 'STATEMENT'),
        ('EXAMPLE', 'STATEMENT'), ('EXAMPLE', 'CONCLUDE'),
        ('COMPARE', 'CONCLUDE'), ('COMPARE', 'STATEMENT'),
        ('CONCLUDE', 'QUESTION'), ('CONCLUDE', 'ACKNOWLEDGE'),
        ('CONCLUDE', 'FAREWELL'),
        ('QUESTION', 'STATEMENT'), ('QUESTION', 'CLARIFY'),
        ('ACKNOWLEDGE', 'STATEMENT'), ('ACKNOWLEDGE', 'FAREWELL'),
        ('FAREWELL', 'GREET'),
    ]
    for a, b in edges:
        g.add_edge(a, b)
    return g


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _token_hv(token, d_sem):
    """Deterministic bipolar HV per token. Lock-free, reproducible."""
    seed = hash(f'sem::{token}') & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=d_sem) * 2 - 1).astype(np.float32)


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------------------------------------------------
# Parallel Semantic/Syntactic Tensor Coupling
# -----------------------------------------------------------------------

class ParallelSemSynCoupling:
    """
    Third-order tensor coupling between semantic HV S and CFG node G.

    emit_vec = T_couple[:, G_id, :].T @ S        # (d_emit,)

    Each grammar node owns its own (d_sem x d_emit) slice of T_couple.
    Hebbian online learning: T[:, G, :] += eta * outer(S, target).

    Grammar pick: argmax over CFG.valid_successors(current_node) of
                  cosine(S, proto[g]).

    Invariant: emit always belongs to grammar manifold (one of |V| typed slices).
    """

    def __init__(self, cfg=None, d_sem=64, d_emit=64, eta=0.05, seed=42):
        self.cfg = cfg if cfg is not None else build_default_cfg()
        self.d_sem = d_sem
        self.d_emit = d_emit
        self.eta = eta
        self.n_grammar = len(self.cfg)

        rng = np.random.RandomState(seed)
        # Coupling tensor: small random init
        self.T_couple = rng.normal(
            0.0, 0.01, size=(d_sem, self.n_grammar, d_emit)
        ).astype(np.float32)
        # Grammar prototypes (one semantic HV per node, L2-normalized)
        proto = rng.standard_normal(size=(self.n_grammar, d_sem)).astype(np.float32)
        for i in range(self.n_grammar):
            proto[i] = _l2_normalize(proto[i])
        self.proto = proto

        self.current_node_id = self.cfg.start
        self.emit_history = []
        self.transition_log = []   # (prev_node, picked_node) tuples for verification

    #  encoding

    def encode_semantic(self, tokens):
        """Bundle bipolar token HVs into a single L2-normalized semantic vector."""
        if not tokens:
            return np.zeros(self.d_sem, dtype=np.float32)
        s = np.zeros(self.d_sem, dtype=np.float32)
        for t in tokens:
            s = s + _token_hv(t, self.d_sem)
        return _l2_normalize(s)

    #  grammar selection

    def pick_grammar(self, S, allow_global=False):
        """Argmax cosine(S, proto[g]) over valid CFG successors of current node."""
        if allow_global or self.current_node_id is None:
            valid = list(range(self.n_grammar))
        else:
            valid = self.cfg.valid_successors(self.current_node_id)
            if not valid:
                # Dead-end: snap back to start
                valid = [self.cfg.start] if self.cfg.start is not None else list(range(self.n_grammar))
        sims = np.array([_cosine(S, self.proto[j]) for j in valid], dtype=np.float32)
        best = int(np.argmax(sims))
        return valid[best]

    #  emit

    def emit(self, S, G_id):
        """emit[k] = sum_i S[i] * T[i, G_id, k]"""
        return self.T_couple[:, G_id, :].T @ S

    #  full step

    def step(self, tokens, allow_global=False, learn_target=None):
        """Encode -> pick grammar -> emit -> (optional) Hebbian learn."""
        S = self.encode_semantic(tokens)
        prev_node = self.current_node_id
        G_id = self.pick_grammar(S, allow_global=allow_global)
        emit_vec = self.emit(S, G_id)
        if learn_target is not None:
            self.hebbian_update(S, G_id, learn_target)
            # re-emit reflects the learning update on subsequent calls
        self.current_node_id = G_id
        self.transition_log.append((prev_node, G_id))
        self.emit_history.append((tokens, G_id, emit_vec))
        return emit_vec, G_id

    #  learning

    def hebbian_update(self, S, G_id, target):
        """T[i, G_id, k] += eta * S[i] * target[k]."""
        delta = (self.eta * np.outer(S, target)).astype(np.float32)
        self.T_couple[:, G_id, :] += delta

    #  utility

    def grammar_name(self, node_id):
        return self.cfg.name(node_id)

    def coupling_size_bytes(self):
        return int(self.T_couple.nbytes)

    def grammar_conformance(self):
        """Fraction of transitions in transition_log that are valid CFG edges."""
        if not self.transition_log:
            return 1.0
        ok = 0
        total = 0
        for prev, curr in self.transition_log:
            if prev is None:
                # initial step, accept any
                ok += 1
                total += 1
                continue
            if curr in self.cfg.valid_successors(prev):
                ok += 1
            total += 1
        return ok / total if total else 0.0

    def reset(self):
        self.current_node_id = self.cfg.start
        self.emit_history.clear()
        self.transition_log.clear()
