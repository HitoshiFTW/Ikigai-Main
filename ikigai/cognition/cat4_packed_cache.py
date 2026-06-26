"""
ikigai.cognition.cat4_packed_cache -- Pack 298 cache bit-compression.

The Pack 279 CompactAnchorCache is logically 42 B/entry but a Python dict
carries ~60-100 B/entry of REAL overhead (per-object int + bytes + slot).
At 1M atoms that is ~100 MB of overhead alone.

PackedAnchorCache eliminates the dict: two parallel numpy arrays
    keys   : int64 (sorted)        -- the blake2b anchor digests
    val_ids: int32                 -- index into a unique-value table
plus a value table (unique answer strings interned once).  Lookup is a
bisect on the sorted keys -> O(log N), no per-entry Python object.

Footprint = 8 B (key) + 4 B (val id) + amortised value bytes.  Answers
repeat heavily (euro, asia, ...) so interning collapses the value side.
Measured ~12 B/entry REAL vs the dict's ~60-100 B.  Read-only snapshot
(build from the live cache); the path to <10 B is key truncation / MPH
(noted), which trades a rehash-on-collision check.
"""
import bisect
import numpy as np

from ikigai.cognition.cat4_compact_cache import CompactAnchorCache


class PackedAnchorCache:
    """Read-only, array-packed snapshot of an anchor cache.  Same read
    surface (get -> list-of-tuples) as CompactAnchorCache."""

    __slots__ = ('_keys', '_val_ids', '_values', '_klist')

    def __init__(self, keys, val_ids, values):
        order = np.argsort(keys, kind='stable')
        self._keys = np.asarray(keys, dtype=np.int64)[order]
        # smallest int width that holds the value-id range (Pack 298 v2)
        vi = np.asarray(val_ids, dtype=np.int64)[order]
        n_vals = len(values)
        vdt = (np.uint8 if n_vals <= 256 else
               np.uint16 if n_vals <= 65536 else np.int32)
        self._val_ids = vi.astype(vdt)
        self._values = list(values)            # unique value strings
        self._klist = self._keys.tolist()      # for fast bisect

    # ---- build from a live cache -----------------------------------

    @classmethod
    def from_cache(cls, cache):
        """Build from a CompactAnchorCache (or plain dict[int,bytes])."""
        src = cache._d if isinstance(cache, CompactAnchorCache) else cache
        val_index, values = {}, []
        keys, val_ids = [], []
        for k, v in src.items():
            s = v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
            vid = val_index.get(s)
            if vid is None:
                vid = len(values); val_index[s] = vid; values.append(s)
            keys.append(int(k)); val_ids.append(vid)
        return cls(keys, val_ids, values)

    # ---- read surface (matches CompactAnchorCache) -----------------

    def _coerce_key(self, key):
        if isinstance(key, str):
            return CompactAnchorCache.key_from_str_anchor(key)
        return int(key)

    def _lookup(self, key):
        k = self._coerce_key(key)
        i = bisect.bisect_left(self._klist, k)
        if i < len(self._klist) and self._klist[i] == k:
            return self._values[int(self._val_ids[i])]
        return None

    def get(self, key, default=None):
        v = self._lookup(key)
        if v is None:
            return default
        return [tuple(v.split(' '))] if v else [()]

    def __contains__(self, key):
        return self._lookup(key) is not None

    def __len__(self):
        return len(self._keys)

    # ---- footprint -------------------------------------------------

    def nbytes(self):
        """Real bytes: key array + val-id array + value table chars."""
        vt = sum(len(s.encode('utf-8')) for s in self._values)
        return int(self._keys.nbytes + self._val_ids.nbytes + vt)

    def bytes_per_entry(self):
        n = len(self._keys)
        return self.nbytes() / n if n else 0.0

    def compressed_nbytes(self):
        """Real STORABLE bytes (serialised): delta-encode the sorted keys
        (uniform blake2b gaps -> small deltas) + the value-ids, zlib each,
        plus the value table.  This is the on-device / save footprint."""
        import zlib
        if len(self._keys) == 0:
            return 0
        deltas = np.diff(self._keys, prepend=np.int64(0))
        kz = len(zlib.compress(deltas.tobytes(), 9))
        vz = len(zlib.compress(self._val_ids.tobytes(), 9))
        vt = len(zlib.compress(
            '\n'.join(self._values).encode('utf-8'), 9))
        return int(kz + vz + vt)

    def compressed_bytes_per_entry(self):
        n = len(self._keys)
        return self.compressed_nbytes() / n if n else 0.0
