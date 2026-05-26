"""
ikigai.cognition.cross_modal_space -- Unified Cross-Modal VSA Space.

Day 55 Pack 61 -- #4: vision/audio/text in one d-dimensional bipolar HV space.

Architecture:
    Text:   token sequence -> position-sensitive bundle -> +/-1 HV
    Vision: pixel patch (H*W floats) -> random projection -> sign -> +/-1 HV
    Audio:  frequency spectrum (N bins) -> random projection -> sign -> +/-1 HV

Concept storage:
    concept_hv = bundle(text_hv, vision_hv, audio_hv)
    = sign(text_hv + vision_hv + audio_hv)

Cross-modal retrieval:
    sim(text_hv, concept_hv) ~= 0.5  (well above noise floor ~0)
    sim(vision_hv, concept_hv) ~= 0.5
    Query any modality -> same concept surfaces.

Biological analogy:
    Superior colliculus multisensory neurons: same neuron fires to BOTH
    the sight and sound of the same event. Bundle = multisensory convergence.
    Random projection = feedforward sensory projection (V1 -> IT, A1 -> AC).

vs LLM: LLM has no native modality. Cross-modal = separate encoder tower + fusion.
        CrossModalSpace: one algebraic space, zero learned parameters.
"""

import numpy as np


_HV_CACHE  = {}
_PROJ_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _rand_proj(input_dim, d, seed):
    key = (input_dim, d, seed)
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(seed)
        # Gaussian projection scaled by 1/sqrt(input_dim) -> unit-norm rows
        _PROJ_CACHE[key] = rng.normal(
            0, 1.0 / float(np.sqrt(input_dim)), (input_dim, d)
        ).astype(np.float32)
    return _PROJ_CACHE[key]


def _bsign(x):
    s = np.sign(x).astype(np.float32)
    s[s == 0.0] = 1.0
    return s


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


#  modality encoders

def encode_text(tokens, d):
    """Position-sensitive bundle HV for token sequence."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    return _bsign(accum.astype(np.float32))


def encode_vision(patch, d, seed=1001):
    """
    Pixel patch -> +/-1 HV via seeded random projection.
    patch: any-shape numpy array of floats, flattened internally.
    """
    flat = np.asarray(patch, dtype=np.float32).ravel()
    proj = _rand_proj(len(flat), d, seed)
    return _bsign(flat @ proj)


def encode_audio(spectrum, d, seed=2002):
    """
    Frequency spectrum (power per bin) -> +/-1 HV via random projection.
    spectrum: 1-D array of floats.
    """
    flat = np.asarray(spectrum, dtype=np.float32).ravel()
    proj = _rand_proj(len(flat), d, seed)
    return _bsign(flat @ proj)


#  CrossModalSpace

class CrossModalSpace:
    """
    Unified VSA store across text, vision, and audio modalities.

    store(name, *, text=None, vision=None, audio=None)
        Encode each provided modality, bundle into concept_hv.

    query(hv, top_k) -> [(name, sim), ...]
        Cosine lookup of query HV against all stored concept_hvs.

    query_modal(data, modality, top_k) -> [(name, sim), ...]
        Encode data then query.

    Properties:
        - concept_hv responds to any of its component modality HVs
        - Cross-modal sim ~= 1/n_modalities per queried modality
        - Different concepts near-orthogonal -> clear discrimination
    """

    MODAL_TEXT   = 'text'
    MODAL_VISION = 'vision'
    MODAL_AUDIO  = 'audio'

    def __init__(self, d=400, patch_seed=1001, audio_seed=2002):
        self.d          = d
        self.patch_seed = patch_seed
        self.audio_seed = audio_seed
        self._concepts  = {}  # name -> {'hv', 'modalities', 'modal_hvs'}

    #  encode

    def encode(self, data, modality):
        """Route data through correct encoder for given modality."""
        if modality == self.MODAL_TEXT:
            return encode_text(data, self.d)
        elif modality == self.MODAL_VISION:
            return encode_vision(data, self.d, self.patch_seed)
        elif modality == self.MODAL_AUDIO:
            return encode_audio(data, self.d, self.audio_seed)
        else:
            raise ValueError(f'Unknown modality: {modality!r}')

    #  store

    def store(self, name, *, text=None, vision=None, audio=None):
        """
        Store concept with one or more modality encodings.
        concept_hv = sign(sum of all modal HVs).
        Returns concept_hv.
        """
        modal_hvs = {}
        parts     = []

        if text is not None:
            hv = self.encode(text, self.MODAL_TEXT)
            modal_hvs['text'] = hv
            parts.append(hv)

        if vision is not None:
            hv = self.encode(vision, self.MODAL_VISION)
            modal_hvs['vision'] = hv
            parts.append(hv)

        if audio is not None:
            hv = self.encode(audio, self.MODAL_AUDIO)
            modal_hvs['audio'] = hv
            parts.append(hv)

        if not parts:
            raise ValueError('At least one modality required')

        accum      = sum(p.astype(np.int32) for p in parts)
        concept_hv = _bsign(accum.astype(np.float32))

        self._concepts[name] = {
            'hv':         concept_hv,
            'modalities': list(modal_hvs.keys()),
            'modal_hvs':  modal_hvs,
        }
        return concept_hv

    #  query

    def query(self, hv, top_k=3):
        """
        Cosine lookup of query HV against all concept_hvs.
        Returns [(name, sim), ...] descending.
        """
        q = np.asarray(hv, dtype=np.float32)
        results = [(name, _cosine(q, c['hv'])) for name, c in self._concepts.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def query_modal(self, data, modality, top_k=3):
        """Encode data in given modality, then query unified space."""
        hv = self.encode(data, modality)
        return self.query(hv, top_k)

    #  similarity

    def concept_sim(self, name_a, name_b):
        """Cosine between two stored concept_hvs."""
        ca = self._concepts.get(name_a)
        cb = self._concepts.get(name_b)
        if ca is None or cb is None:
            return 0.0
        return _cosine(ca['hv'], cb['hv'])

    def modal_sim(self, name_a, name_b, modality):
        """Cosine between specific-modality HVs of two concepts."""
        ca = self._concepts.get(name_a)
        cb = self._concepts.get(name_b)
        if ca is None or cb is None:
            return 0.0
        hva = ca['modal_hvs'].get(modality)
        hvb = cb['modal_hvs'].get(modality)
        if hva is None or hvb is None:
            return 0.0
        return _cosine(hva, hvb)

    def cross_modal_sim(self, name, query_data, query_modality):
        """
        Cosine between query (one modality) and stored concept_hv.
        Expected: ~1/n_modalities for correct concept.
        """
        c = self._concepts.get(name)
        if c is None:
            return 0.0
        q_hv = self.encode(query_data, query_modality)
        return _cosine(q_hv, c['hv'])

    #  introspection

    @property
    def n_concepts(self):
        return len(self._concepts)

    def concept_info(self, name):
        c = self._concepts.get(name)
        if c is None:
            return None
        return {'name': name, 'modalities': c['modalities'], 'd': self.d,
                'n_modal': len(c['modalities'])}

    def all_names(self):
        return list(self._concepts.keys())
