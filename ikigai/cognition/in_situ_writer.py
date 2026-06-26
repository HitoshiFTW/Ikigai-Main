"""ikigai.cognition.in_situ_writer -- Pack 191 must-invent primitive #1.

In-situ non-interfering substrate writes.

Today's substrate write is a destructive sum: bank.C[loc] += hv. When two
models share the substrate, model A's writes corrupt model B's reads.
This module provides a write protocol that lets N models share the same
substrate cells with bounded cross-model interference.

Mechanism:
  Each model gets an ORTHOGONAL namespace phasor n_M = exp(i * theta_M)
  where theta_M is a deterministic per-cell phase. Writes for model M go in
  as substrate[loc] += n_M * hv. Reads for model M unbind by conjugate.

Cross-model crosstalk = | <n_M, n_M'> | which is small for different M.
Same-model recall = | <n_M, n_M> | = 1.

The orthogonal namespace phasors live in a small lookup table (model_id
-> theta vector of length d). One model = 4 KB at d=1024 -- negligible.

Public API:
    isw = InSituWriter(substrate)
    isw.register_model('qwen-1.5b')
    isw.register_model('llama-3-8b')
    isw.write_namespace('qwen-1.5b', loc_indices, hv)
    out = isw.read_namespace('qwen-1.5b', address_hv)
    crosstalk = isw.measure_crosstalk('qwen-1.5b', 'llama-3-8b')

Reversibility: each model namespace has a deterministic phase tag derived
from hash(model_id) -- writes can be subtracted by recomputing the tag.
Models can be evicted without scrubbing the substrate (just stop reading
with that tag).
"""

import numpy as np


class InSituWriter:
    """Multi-model substrate writer with orthogonal namespace phasors.

    Wraps a VSASDM-like substrate. Tracks per-model phasor tags. Writes
    superpose without cross-model interference (bounded by 1/sqrt(d)).
    """

    def __init__(self, substrate):
        self.substrate = substrate
        self.d = int(getattr(substrate, 'd', 1024))
        self.namespaces = {}     # model_id -> phasor vector (complex64, d)
        self.write_log = {}      # model_id -> list of (loc, hv) for reversibility

    def register_model(self, model_id):
        """Allocate orthogonal phasor for new model. Deterministic from id."""
        if model_id in self.namespaces:
            return self.namespaces[model_id]
        seed = abs(hash(f'isw_ns_{model_id}')) % (2 ** 31)
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        phasor = np.exp(1j * theta).astype(np.complex64)
        self.namespaces[model_id] = phasor
        self.write_log[model_id] = []
        return phasor

    def write_namespace(self, model_id, loc_indices, hv):
        """Superpose phasor-tagged HV into substrate cells.
        bank.C[loc] += namespace_phasor * hv (complex elementwise).
        """
        if model_id not in self.namespaces:
            self.register_model(model_id)
        tag = self.namespaces[model_id]
        hv = np.asarray(hv, dtype=np.complex64).reshape(-1)
        if hv.shape[0] != self.d:
            raise ValueError(f'hv length {hv.shape[0]} != substrate d {self.d}')
        tagged = (tag * hv).astype(np.complex64)
        for loc in np.asarray(loc_indices, dtype=np.int64).reshape(-1):
            self.substrate.C[int(loc)] += tagged
        self.write_log[model_id].append((np.asarray(loc_indices).copy(), hv.copy()))

    def read_namespace(self, model_id, address_hv):
        """Unbind namespace phasor on read. Returns hv with bounded crosstalk."""
        if model_id not in self.namespaces:
            raise KeyError(f'model {model_id} not registered')
        tag = self.namespaces[model_id]
        tag_conj = np.conj(tag).astype(np.complex64)
        raw = self.substrate.read(np.asarray(address_hv, dtype=np.complex64))
        return (tag_conj * raw).astype(np.complex64)

    def evict_model(self, model_id):
        """Subtract all writes for model_id (reversible eviction)."""
        if model_id not in self.write_log:
            return 0
        tag = self.namespaces[model_id]
        n_writes = 0
        for loc_indices, hv in self.write_log[model_id]:
            tagged = (tag * hv).astype(np.complex64)
            for loc in np.asarray(loc_indices, dtype=np.int64).reshape(-1):
                self.substrate.C[int(loc)] -= tagged
                n_writes += 1
        del self.write_log[model_id]
        del self.namespaces[model_id]
        return n_writes

    def measure_crosstalk(self, model_a, model_b):
        """Inner product of two namespace tags (bounded ~1/sqrt(d) random)."""
        if model_a not in self.namespaces or model_b not in self.namespaces:
            raise KeyError('both models must be registered')
        ta = self.namespaces[model_a]
        tb = self.namespaces[model_b]
        return float(np.abs(np.vdot(ta, tb)) / self.d)

    def n_models(self):
        return len(self.namespaces)

    def memory_overhead_bytes(self):
        """Phasor tag storage = 8 bytes/cell * d * n_models."""
        return 8 * self.d * len(self.namespaces)
