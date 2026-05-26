"""
ikigai.cognition.belief_field -- Free-Energy Belief Field.

Day 55 Pack 59 -- complete #2: contradictions as XOR gradient, beliefs self-heal.

Free energy = sum of contradictions in the belief field.
    E(field) = -sum cosine(A_i, A_j) for conflicting pairs (i,j)
Minimizing E heals contradictions: beliefs converge toward consistent consensus.

Algorithm:
    assert_belief(name, tokens)  -> store bipolar HV belief
    deny_belief(name, tokens)    -> store as negated (anti-asserted)
    conflict(a, b)               -> cosine < threshold -> contradiction detected
    xor_gradient(a, b)           -> positions where a,b disagree (conflict vector)
    heal(a, b)                   -> Hebbian update: move both toward consensus
    propagate(rounds)            -> sweep all pairs, heal conflicts until stable
    field_consistency()          -> mean pairwise cosine (higher = less conflict)

Hebbian heal step:
    a_new = sign(a + heal_rate * b)
    b_new = sign(b + heal_rate * a)
    Increases cosine(a, b) at each conflict position.

vs LLM: LLM beliefs are frozen in weights (hallucinations persist).
        BeliefField: contradictions detected + healed at inference time.
        Zero gradient computation. O(d) per heal step.
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    """Full-sequence hash to unique bipolar HV."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    seq = '|'.join(str(t) for t in tokens)
    return _hv_for(seq, d)


def _encode_semantic(tokens, d):
    """Position-sensitive bundle for partial matching."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    out = np.sign(accum).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _bind(a, b):
    out = np.sign(a * b).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BeliefField:
    """
    Associative belief network in VSA space.
    Contradictions are XOR gradients driving Hebbian self-healing.
    """

    def __init__(self, d=400, conflict_threshold=-0.05, heal_rate=0.5):
        self.d                  = d
        self.conflict_threshold = conflict_threshold
        self.heal_rate          = heal_rate
        self._beliefs    = {}   # name -> raw belief HV (before polarity)
        self._polarities = {}   # name -> +1 or -1
        self._heal_log   = []   # list of {step, pair, pre_sim, post_sim}
        self._step       = 0

    #  assertion

    def assert_belief(self, name, tokens):
        """Assert belief: tokens encode a proposition held as true."""
        self._beliefs[name]    = _encode_semantic(tokens, self.d)
        self._polarities[name] = +1

    def deny_belief(self, name, tokens):
        """Deny belief: tokens encode a proposition held as false."""
        self._beliefs[name]    = _encode_semantic(tokens, self.d)
        self._polarities[name] = -1

    def assert_hv(self, name, hv, polarity=+1):
        """Assert a pre-computed HV directly."""
        self._beliefs[name]    = np.asarray(hv, dtype=np.float32).copy()
        self._polarities[name] = polarity

    def effective_hv(self, name):
        """Effective belief HV = raw_hv * polarity."""
        return self._beliefs[name] * self._polarities[name]

    #  conflict detection

    def similarity(self, name_a, name_b):
        """Cosine between effective HVs."""
        return _cosine(self.effective_hv(name_a), self.effective_hv(name_b))

    def conflict(self, name_a, name_b):
        """
        Returns (is_conflict, sim, xor_gradient_mask).
        xor_gradient: 1.0 at positions where effective HVs disagree.
        """
        a   = self.effective_hv(name_a)
        b   = self.effective_hv(name_b)
        sim = _cosine(a, b)
        if sim < self.conflict_threshold:
            xor_grad = (a != b).astype(np.float32)
            return True, sim, xor_grad
        return False, sim, None

    def n_conflicts(self):
        """Count conflicting pairs."""
        names = list(self._beliefs.keys())
        count = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ok, _, _ = self.conflict(names[i], names[j])
                if ok:
                    count += 1
        return count

    def all_conflicts(self):
        """Return list of (name_a, name_b, sim) for all conflicting pairs."""
        names  = list(self._beliefs.keys())
        result = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ok, sim, _ = self.conflict(names[i], names[j])
                if ok:
                    result.append((names[i], names[j], sim))
        return result

    #  healing

    def heal(self, name_a, name_b):
        """
        Consensus healing: compute C = sign(a + b), move both toward C.
        Naive Hebbian (a->sign(a+lr*b)) only swaps conflict positions.
        Consensus breaks symmetry: both converge to same value at conflict positions.
        a_new = sign(a + heal_rate * C)
        b_new = sign(b + heal_rate * C)
        Returns (pre_sim, post_sim).
        """
        is_conflict, pre_sim, xor_grad = self.conflict(name_a, name_b)
        if not is_conflict:
            return pre_sim, pre_sim

        a = self.effective_hv(name_a)
        b = self.effective_hv(name_b)

        # Consensus with tie-breaking noise (avoids bias toward +1)
        noise     = np.random.default_rng(self._step).choice(
            [-1, 1], size=self.d).astype(np.float32) * 1e-3
        consensus = np.sign(a + b + noise).astype(np.float32)
        consensus[consensus == 0.0] = 1.0

        a_new = np.sign(a + self.heal_rate * consensus).astype(np.float32)
        b_new = np.sign(b + self.heal_rate * consensus).astype(np.float32)
        a_new[a_new == 0.0] = 1.0
        b_new[b_new == 0.0] = 1.0

        self._beliefs[name_a] = a_new * self._polarities[name_a]
        self._beliefs[name_b] = b_new * self._polarities[name_b]

        post_sim = _cosine(self.effective_hv(name_a), self.effective_hv(name_b))
        self._heal_log.append({
            'step': self._step, 'pair': (name_a, name_b),
            'pre_sim': pre_sim, 'post_sim': post_sim,
        })
        self._step += 1
        return pre_sim, post_sim

    def propagate(self, max_rounds=5):
        """
        Sweep all pairs, heal conflicts. Repeat until stable or max_rounds.
        Returns {healed_pairs, rounds_taken, final_conflicts}.
        """
        healed_total = 0
        for rnd in range(max_rounds):
            conflicts = self.all_conflicts()
            if not conflicts:
                return {'healed_pairs': healed_total, 'rounds': rnd,
                        'final_conflicts': 0}
            for name_a, name_b, _ in conflicts:
                self.heal(name_a, name_b)
                healed_total += 1
        return {'healed_pairs': healed_total, 'rounds': max_rounds,
                'final_conflicts': self.n_conflicts()}

    #  field metrics

    def field_consistency(self):
        """Mean pairwise cosine across all asserted beliefs."""
        names = [n for n, p in self._polarities.items() if p == +1]
        if len(names) < 2:
            return 1.0
        sims = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sims.append(self.similarity(names[i], names[j]))
        return float(np.mean(sims))

    def free_energy(self):
        """
        Negative mean pairwise cosine (lower = more consistent = less free energy).
        Minimized when all beliefs are mutually consistent.
        """
        return -self.field_consistency()

    def xor_gradient_field(self):
        """
        Sum of XOR gradients across all conflict pairs.
        High at positions that are disputed across many belief pairs.
        Returns d-dim vector of conflict intensities.
        """
        d      = self.d
        result = np.zeros(d, dtype=np.float32)
        pairs  = self.all_conflicts()
        for name_a, name_b, _ in pairs:
            a = self.effective_hv(name_a)
            b = self.effective_hv(name_b)
            result += (a != b).astype(np.float32)
        if pairs:
            result /= len(pairs)
        return result

    #  introspection

    @property
    def n_beliefs(self):
        return len(self._beliefs)

    def belief_names(self):
        return list(self._beliefs.keys())

    def heal_log(self):
        return list(self._heal_log)
