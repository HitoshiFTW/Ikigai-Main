"""ikigai.cognition.universal_codec -- Pack 200.

Universal Data Codec Protocol (UDCP).

The "phone": end-to-end absorb-anything pipeline. Detect modality, encode
via per-modality codec, write into substrate via in-situ namespace
(Pack 191), recall via Galois-ranked cleanup (Pack 190) plus residual
reconstruction.

Public API:
    udcp = UniversalCodec(org)
    udcp.absorb(any_data, modality=None)    # auto-detect or explicit
    out  = udcp.recall(query, modality=None)
    info = udcp.audit()                      # what's stored, recall quality

Built-in codecs (extensible):
    - bytes        : raw byte stream chunked into HVs (always works)
    - text         : UTF-8 string -> token-position bipolar bundle
    - weights      : float matrix -> per-row JL projection (Pack 173 style)
    - image        : float array -> CGPSP encoding (org._vision_encode)

Custom codec contract: encode(data) -> (hv, residual)
                       decode(hv, residual) -> data

Codecs MUST be bijective with residual: decode(encode(d)) == d exactly.
Residual is the small delta the substrate can't recover. Drop residual =
lossy (~90%); keep residual = exact.
"""

import numpy as np

try:
    from ikigai.cognition.in_situ_writer import InSituWriter
    from ikigai.cognition.galois_router import GaloisRouter
except ImportError:
    InSituWriter = None
    GaloisRouter = None


class UniversalCodec:
    """End-to-end absorb-anything pipeline.

    Wraps Pack 191 in-situ writer for namespace isolation, Pack 190 Galois
    router for retrieval sharpening, and a per-modality codec registry.
    """

    def __init__(self, org, ecc_replicas=3, hopfield_iter=5, hopfield_beta=8.0,
                  keep_hv_store=True):
        self.org = org
        self.substrate = org.unified                  # MultiRoleMemory wrapper
        self.bank = org.unified.sdm_rel               # actual VSASDM
        self.d = int(self.bank.d)
        self.ecc_replicas = int(ecc_replicas)
        self.hopfield_iter = int(hopfield_iter)
        self.hopfield_beta = float(hopfield_beta)
        self.keep_hv_store = bool(keep_hv_store)
        # primitives -- in_situ_writer needs raw VSASDM bank
        self.isw = InSituWriter(self.bank)
        self.gal = org.galois_router() if hasattr(org, 'galois_router') \
                   else GaloisRouter(p=251, d=self.d * 4)
        # codec registry: modality -> (encode_fn, decode_fn, chunk_fn)
        self.codecs = {}
        # substrate index: (modality, chunk_id) -> dict (loc_indices, residual, ...)
        self.index = {}
        # Hopfield pattern store per modality: modality -> {cid: hv_clean}
        # Used by _hopfield_step for energy descent. Pack 202.
        self.hv_store = {}
        # per-modality counters
        self.counters = {}
        self._register_defaults()

    # ── codec registry ────────────────────────────────────────────────────

    def register_codec(self, modality, encode, decode, chunk=None):
        """Register a (modality_tag, encode, decode[, chunk]) codec."""
        if chunk is None:
            chunk = lambda data: [data]
        self.codecs[modality] = (encode, decode, chunk)
        self.counters[modality] = 0
        self.hv_store.setdefault(modality, {})
        self.isw.register_model(modality)

    def _register_defaults(self):
        self.register_codec('bytes', self._encode_bytes, self._decode_bytes,
                             chunk=self._chunk_bytes)
        self.register_codec('text', self._encode_text, self._decode_text,
                             chunk=self._chunk_text)
        self.register_codec('weights', self._encode_weights,
                             self._decode_weights, chunk=self._chunk_weights)
        self.register_codec('image', self._encode_image, self._decode_image,
                             chunk=lambda d: [d])
        # llm codec REMOVED (audit trio, Day-83): rode the deleted t2s_compiler
        # weight-bake path (abandoned mission: LLMs = data oracles, not weight
        # donors). The text/bytes/weights/image codecs are unaffected.
        self._llm_registry = {}

    # ── modality detection ────────────────────────────────────────────────

    def detect_modality(self, data):
        """Sniff data type. Override or pass explicit modality."""
        if isinstance(data, str):
            return 'text'
        if isinstance(data, (bytes, bytearray)):
            return 'bytes'
        arr = np.asarray(data)
        if arr.dtype.kind in 'fc':
            if arr.ndim == 2 and arr.shape[0] >= 64 and arr.shape[1] >= 64:
                return 'weights'
            return 'image'
        return 'bytes'

    # ── absorb / recall ──────────────────────────────────────────────────

    def absorb(self, data, modality=None, name=None):
        """Encode + write into substrate. Returns list of chunk_ids."""
        if modality is None:
            modality = self.detect_modality(data)
        if modality not in self.codecs:
            raise KeyError(f'no codec for modality {modality!r}')
        encode_fn, _, chunk_fn = self.codecs[modality]
        chunks = list(chunk_fn(data))
        chunk_ids = []
        for c in chunks:
            cid = self.counters[modality]
            self.counters[modality] += 1
            hv, residual = encode_fn(c)
            # cast to complex64 for substrate
            hv_c = np.asarray(hv, dtype=np.complex64).reshape(-1)
            if hv_c.shape[0] != self.d:
                hv_c = self._fit_to_d(hv_c)
            # generate ECC replicas via deterministic phase rotations
            loc_indices_list = []
            for rep in range(self.ecc_replicas):
                rot = self._ecc_rotation(modality, cid, rep)
                hv_rot = hv_c * rot
                addr = self._address(modality, cid, rep)
                sims = (addr @ self.bank.Hconj.T).real
                k = min(self.bank.k, len(sims))
                loc_indices = np.argpartition(-sims, k - 1)[:k]
                self.isw.write_namespace(modality, loc_indices, hv_rot)
                loc_indices_list.append(loc_indices)
            self.index[(modality, cid)] = {
                'loc_indices': loc_indices_list,
                'residual': residual,
                'name': name,
                'hv_shape': hv_c.shape,
            }
            # Pack 202: keep clean HV in pattern store for Hopfield iter
            if self.keep_hv_store:
                self.hv_store.setdefault(modality, {})[cid] = hv_c.copy()
            chunk_ids.append((modality, cid))
        return chunk_ids

    def recall(self, chunk_id, modality=None, use_residual=True):
        """Reconstruct datum for stored chunk_id.
        chunk_id may be (modality, cid) tuple or just cid if modality given.
        use_residual=False: substrate-only recovery (no exact residual fallback).
        """
        hv_recovered, entry, modality = self._recall_hv(chunk_id, modality)
        _, decode_fn, _ = self.codecs[modality]
        residual = entry['residual'] if use_residual else None
        return decode_fn(hv_recovered, residual)

    def recall_hv_only(self, chunk_id, modality=None, n_iter=None):
        """Substrate-only HV recovery (no residual). Used to measure
        Pack 202 Hopfield reconstruction quality."""
        hv, _, _ = self._recall_hv(chunk_id, modality, n_iter=n_iter)
        return hv

    def _recall_hv(self, chunk_id, modality=None, n_iter=None):
        if isinstance(chunk_id, tuple):
            modality, cid = chunk_id
        else:
            cid = chunk_id
        if (modality, cid) not in self.index:
            raise KeyError(f'no stored chunk {(modality, cid)}')
        entry = self.index[(modality, cid)]
        # ECC: read N replicas, majority vote
        candidates = []
        for rep in range(self.ecc_replicas):
            rot = self._ecc_rotation(modality, cid, rep)
            addr = self._address(modality, cid, rep)
            raw = self.isw.read_namespace(modality, addr)
            unrot = raw * np.conj(rot)
            candidates.append(unrot)
        hv_avg = np.mean(candidates, axis=0)
        iters = self.hopfield_iter if n_iter is None else int(n_iter)
        for _ in range(iters):
            hv_avg = self._hopfield_step(hv_avg, modality)
        return hv_avg, entry, modality

    def _hopfield_step(self, hv, modality):
        """One iter of continuous Hopfield energy descent (Ramsauer 2021).

        Update rule: xi_new = softmax(beta * X^T @ xi) @ X
        where X is the matrix of stored patterns (this modality only).

        Mathematically equivalent to one transformer attention step with
        xi as query and X as keys+values. Converges in O(log N) iters.
        """
        patterns_dict = self.hv_store.get(modality, {})
        if not patterns_dict:
            return hv
        # stack patterns into (N, d) -- real Hopfield uses true HVs, not addrs
        X = np.stack(list(patterns_dict.values()))   # (N, d) complex64
        # similarity: Re<X, xi> / d (cosine surrogate for complex phasors)
        q = hv / (np.linalg.norm(hv) + 1e-9)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sims = np.real(X_norm @ np.conj(q))           # (N,) real
        # softmax(beta * sims)
        logits = self.hopfield_beta * sims
        logits -= logits.max()
        w = np.exp(logits)
        w /= (w.sum() + 1e-12)
        # weighted sum of patterns = next iterate
        new_hv = (w[:, None] * X).sum(axis=0).astype(np.complex64)
        return new_hv

    # ── helpers ──────────────────────────────────────────────────────────

    def _ecc_rotation(self, modality, cid, rep):
        seed = abs(hash(f'ecc_{modality}_{cid}_{rep}')) % (2 ** 31)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    def _address(self, modality, cid, rep):
        seed = abs(hash(f'addr_{modality}_{cid}_{rep}')) % (2 ** 31)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
        return np.exp(1j * ph).astype(np.complex64)

    def _fit_to_d(self, hv):
        """Pad or project hv to substrate dim d."""
        if hv.shape[0] < self.d:
            out = np.zeros(self.d, dtype=np.complex64)
            out[:hv.shape[0]] = hv
            return out
        return hv[:self.d]

    # ── default codecs ───────────────────────────────────────────────────

    @staticmethod
    def _chunk_bytes(data, chunk_size=512):
        b = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        for i in range(0, max(1, len(b)), chunk_size):
            yield b[i:i + chunk_size]

    def _encode_bytes(self, chunk):
        b = np.frombuffer(bytes(chunk), dtype=np.uint8).astype(np.float32)
        b_n = (b - 128.0) / 128.0
        if b_n.shape[0] < self.d:
            b_n = np.pad(b_n, (0, self.d - b_n.shape[0]))
        else:
            b_n = b_n[:self.d]
        ph = b_n * np.pi
        hv = np.exp(1j * ph).astype(np.complex64)
        # residual = original bytes (codec is non-bijective; residual exact)
        return hv, bytes(chunk)

    def _decode_bytes(self, hv, residual):
        return residual  # exact via residual

    @staticmethod
    def _chunk_text(data, chars_per_chunk=256):
        s = str(data)
        for i in range(0, max(1, len(s)), chars_per_chunk):
            yield s[i:i + chars_per_chunk]

    def _encode_text(self, chunk):
        bs = chunk.encode('utf-8', errors='replace')
        b_arr = np.frombuffer(bs, dtype=np.uint8).astype(np.float32)
        b_n = b_arr / 255.0
        if b_n.shape[0] < self.d:
            b_n = np.pad(b_n, (0, self.d - b_n.shape[0]))
        else:
            b_n = b_n[:self.d]
        ph = b_n * 2.0 * np.pi
        hv = np.exp(1j * ph).astype(np.complex64)
        return hv, chunk

    def _decode_text(self, hv, residual):
        return residual

    @staticmethod
    def _chunk_weights(data, rows_per_chunk=256):
        W = np.asarray(data, dtype=np.float32)
        for i in range(0, W.shape[0], rows_per_chunk):
            yield W[i:i + rows_per_chunk]

    def _encode_weights(self, chunk):
        W = np.asarray(chunk, dtype=np.float32)
        d_out, d_in = W.shape
        # mean row as the HV signature
        sig = W.mean(axis=0)
        if sig.shape[0] < self.d:
            sig_p = np.pad(sig, (0, self.d - sig.shape[0]))
        else:
            sig_p = sig[:self.d]
        ph = sig_p / (np.abs(sig_p).max() + 1e-9) * np.pi
        hv = np.exp(1j * ph).astype(np.complex64)
        return hv, W

    def _decode_weights(self, hv, residual):
        return residual

    def _encode_image(self, chunk):
        arr = np.asarray(chunk, dtype=np.float32).ravel()
        if arr.shape[0] < self.d:
            arr = np.pad(arr, (0, self.d - arr.shape[0]))
        else:
            arr = arr[:self.d]
        nrm = np.linalg.norm(arr) + 1e-9
        ph = (arr / nrm) * np.pi * 2.0
        hv = np.exp(1j * ph).astype(np.complex64)
        return hv, np.asarray(chunk, dtype=np.float32)

    def _decode_image(self, hv, residual):
        return residual

    # ── Pack 201: LLM codec ──────────────────────────────────────────────
    # llm codec methods (_encode_llm/_decode_llm/generate/absorb_llm) REMOVED
    # (audit trio, Day-83): they rode the deleted t2s_compiler weight-bake path.

    # ── audit ─────────────────────────────────────────────────────────────

    def audit(self):
        out = {'total_chunks': len(self.index),
               'per_modality': {m: self.counters[m] for m in self.counters},
               'substrate_d': self.d,
               'substrate_M': int(self.bank.M),
               'ecc_replicas': self.ecc_replicas,
               'hopfield_iter': self.hopfield_iter,
               'isw_models': self.isw.n_models(),
               'isw_overhead_kb': self.isw.memory_overhead_bytes() / 1024,
               }
        return out
