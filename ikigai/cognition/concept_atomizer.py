"""
ikigai.cognition.concept_atomizer -- Wake-Sleep Concept Atomizer.

Day 55 Pack 62 -- #11: cluster episodes into reusable concept atoms.

Architecture:
    Wake phase: record(name, tokens) -> encode episode as bipolar HV
    Sleep phase: sleep(n_atoms) -> K-means-like clustering in VSA space
                                -> centroid per cluster = concept atom
    Recall: recall(tokens) -> [(atom_name, sim), ...] nearest atoms

Clustering (VSA K-means):
    1. Seed n_atoms centers = spread via max-dissimilarity (avoid same cluster)
    2. Assign each episode to nearest center (cosine)
    3. Recompute centroid = sign(sum of cluster HVs)
    4. Iterate until stable or max_iter

Biological analogy:
    Hippocampal replay during NREM sleep (Stickgold 2005):
    - Wake:  hippocampus encodes episodes as sparse firing patterns
    - Sleep: replay -> cortex extracts shared structure -> concept neuron
    Each concept atom = a cortical neuron tuned to a recurring theme.
    replay() in CrossTimeResonance feeds raw episodes -> sleep() atomizes them.

vs LLM: LLM concepts = frozen in weights. No new concept discovery post-training.
        ConceptAtomizer: discovers new atoms from any N episodes. Zero gradient.
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    """Position-sensitive bundle: correlates within shared-prefix clusters."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    s = np.sign(accum).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _bsign(x):
    s = np.sign(x).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


class ConceptAtomizer:
    """
    Wake-Sleep concept discovery: cluster episodes -> atomic concept HVs.

    record(name, tokens, tick=None)
        Store episode during wake phase.

    sleep(n_atoms, max_iter) -> [atom_names]
        Cluster accumulated episodes. Returns list of new atom names.
        Atoms accumulate across sleep cycles (no forgetting).

    recall(tokens, top_k) -> [(atom_name, sim), ...]
        Encode query, find nearest concept atoms.

    recall_hv(hv, top_k) -> [(atom_name, sim), ...]
        Find nearest atoms by pre-encoded HV.

    atom_members(atom_name) -> [episode_names]
        Episodes assigned to this atom in last sleep.

    compression_ratio -> float
        n_total_episodes / n_atoms (how many episodes per concept).
    """

    def __init__(self, d=400):
        self.d          = d
        self._episodes  = {}       # name -> hv (all recorded episodes)
        self._ep_order  = []       # insertion order
        self._atoms     = {}       # atom_name -> centroid_hv
        self._atom_members = {}    # atom_name -> [episode_names]
        self._cluster_of   = {}    # episode_name -> atom_name
        self.n_sleep_cycles = 0

    #  wake phase

    def record(self, name, tokens, tick=None):
        """Record episode during wake. Returns episode HV."""
        hv = _encode(tokens, self.d)
        if name not in self._episodes:
            self._ep_order.append(name)
        self._episodes[name] = hv
        return hv

    #  sleep phase

    def sleep(self, n_atoms=5, max_iter=20):
        """
        Cluster episodes into n_atoms concept atoms.
        Seeds centers via max-dissimilarity spread.
        Iterates centroid update until stable.
        Returns list of newly created atom names.
        """
        names = self._ep_order[:]
        if not names:
            return []
        n_atoms = min(n_atoms, len(names))
        hvs = [self._episodes[n] for n in names]

        # Seed: first episode, then max-dissimilarity selection
        seed_idx = [0]
        for _ in range(n_atoms - 1):
            best_i, best_dist = 0, float('inf')
            for i, h in enumerate(hvs):
                if i in seed_idx:
                    continue
                # Distance = 1 - max_cosine_to_any_center
                max_cos = max(_cosine(h, hvs[s]) for s in seed_idx)
                dist = 1.0 - max_cos
                if dist < best_dist:
                    best_dist, best_i = dist, i
            seed_idx.append(best_i)
        centers = [hvs[i].copy() for i in seed_idx]

        # Iterative centroid update
        assignments = [-1] * len(names)
        for _ in range(max_iter):
            new_assignments = []
            for h in hvs:
                sims = [_cosine(h, c) for c in centers]
                new_assignments.append(int(np.argmax(sims)))
            if new_assignments == assignments:
                break
            assignments = new_assignments
            # Recompute centroids
            for k in range(n_atoms):
                cluster_hvs = [hvs[i] for i, a in enumerate(assignments) if a == k]
                if cluster_hvs:
                    accum = sum(h.astype(np.int32) for h in cluster_hvs)
                    centers[k] = _bsign(accum.astype(np.float32))

        # Store atoms
        new_atoms = []
        for k in range(n_atoms):
            members = [names[i] for i, a in enumerate(assignments) if a == k]
            if not members:
                continue
            atom_name = f'atom_{self.n_sleep_cycles}_{k}'
            self._atoms[atom_name] = centers[k].copy()
            self._atom_members[atom_name] = members
            for ep_name in members:
                self._cluster_of[ep_name] = atom_name
            new_atoms.append(atom_name)

        self.n_sleep_cycles += 1
        return new_atoms

    #  recall

    def recall_hv(self, hv, top_k=3):
        """Nearest atoms to given HV. Returns [(atom_name, sim), ...]."""
        q = np.asarray(hv, dtype=np.float32)
        results = [(n, _cosine(q, a)) for n, a in self._atoms.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def recall(self, tokens, top_k=3):
        """Encode tokens, find nearest concept atoms."""
        return self.recall_hv(_encode(tokens, self.d), top_k)

    #  introspection

    def atom_members(self, atom_name):
        """Episode names assigned to this atom in last sleep."""
        return list(self._atom_members.get(atom_name, []))

    def cluster_of(self, episode_name):
        """Which atom was this episode assigned to?"""
        return self._cluster_of.get(episode_name)

    @property
    def n_episodes(self):
        return len(self._episodes)

    @property
    def n_atoms(self):
        return len(self._atoms)

    @property
    def compression_ratio(self):
        """Episodes per concept atom (higher = more compression)."""
        return self.n_episodes / self.n_atoms if self.n_atoms > 0 else 0.0

    def atom_centroid(self, atom_name):
        return self._atoms.get(atom_name)
