"""
ikigai.cognition.moe — Mixture-of-Experts at the HV level (Day 54 Pack 22)

Multiple specialized codebooks indexed by domain centroid HVs.
Query -> centroid sim -> activate single best codebook -> retrieve.

Breaks the 30MB RAM ceiling: only ONE codebook loaded at a time.
Disk-backed codebooks (via numpy.memmap, when added) allow 100M+ templates
while working set stays at single-codebook size.

Public API:
    Codebook(name, items_dict)        — lazy-loaded HV-indexed item bag
    MoERouter()                       — registers experts, routes queries
        register_expert(name, centroid_keywords, codebook_items)
        route(text)              -> (expert_name, sim)
        query(text, top_k)       -> (expert_name, [(item, sim), ...])
        active_memory_bytes()    -> int
        n_loaded_codebooks()     -> int
"""

import re
import random
import numpy as np


HV_DIM = 400
_HV_CACHE = {}


def _hv(key):
    if key not in _HV_CACHE:
        rng = random.Random(hash(key) & 0x7FFFFFFF)
        _HV_CACHE[key] = np.array(
            [1 if rng.randint(0, 1) else -1 for _ in range(HV_DIM)],
            dtype=np.int8,
        )
    return _HV_CACHE[key]


def _bundle(hvs):
    if not hvs:
        return np.zeros(HV_DIM, dtype=np.int32)
    s = np.zeros(HV_DIM, dtype=np.int32)
    for h in hvs:
        s += h.astype(np.int32)
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kw_bundle(keywords):
    return _bundle([_hv(f'__id__{k}') for k in keywords])


_STOP = {'a','an','the','of','to','in','on','for','and','or','is','it','this',
         'that','given','from','with','as','at','by','be','are'}


def encode_query(text):
    """Bag-of-words bundle HV for an NL query."""
    words = re.findall(r'[a-z]+', text.lower())
    content = [w for w in words if w not in _STOP and len(w) >= 2]
    if not content:
        return np.zeros(HV_DIM, dtype=np.int32)
    atoms = [_hv(f'__id__{w}') for w in content]
    for w in content:
        if len(w) > 3 and w.endswith('s'):
            atoms.append(_hv(f'__id__{w[:-1]}'))
    return _bundle(atoms)


class Codebook:
    """A bag of HV-indexed items. Loaded on demand, unloadable to free RAM."""

    def __init__(self, name, items):
        """items: dict {item_key: keyword_list_for_encoding}"""
        self.name = name
        self.items = dict(items)
        self._hvs = None
        self._loaded = False

    def load(self):
        if not self._loaded:
            self._hvs = {k: _kw_bundle(kws) for k, kws in self.items.items()}
            self._loaded = True

    def unload(self):
        self._hvs = None
        self._loaded = False

    def memory_bytes(self):
        if not self._loaded:
            return 0
        return sum(h.nbytes for h in self._hvs.values())

    def retrieve(self, query_hv, top_k=3):
        self.load()
        sims = [(k, _cosine(query_hv, h)) for k, h in self._hvs.items()]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]


class MoERouter:
    """
    Routes queries to one of N expert codebooks via centroid HV similarity.
    Holds at most ONE active codebook in RAM at a time.
    """

    def __init__(self):
        self.experts = {}      # name -> {'centroid': hv, 'codebook': Codebook}
        self._active = None    # currently loaded expert name

    def register_expert(self, name, centroid_keywords, codebook_items):
        self.experts[name] = {
            'centroid': _kw_bundle(centroid_keywords),
            'codebook': Codebook(name, codebook_items),
        }

    def route(self, query_text):
        q = encode_query(query_text)
        best, best_sim = None, -2.0
        for name, ex in self.experts.items():
            s = _cosine(q, ex['centroid'])
            if s > best_sim:
                best_sim = s
                best = name
        return best, best_sim

    def query(self, query_text, top_k=3):
        """Route -> swap active codebook -> retrieve top-K items."""
        name, _ = self.route(query_text)
        if self._active and self._active != name:
            self.experts[self._active]['codebook'].unload()
        self._active = name
        q = encode_query(query_text)
        return name, self.experts[name]['codebook'].retrieve(q, top_k=top_k)

    def active_memory_bytes(self):
        if self._active is None:
            return 0
        return self.experts[self._active]['codebook'].memory_bytes()

    def n_loaded_codebooks(self):
        return sum(1 for ex in self.experts.values() if ex['codebook']._loaded)
