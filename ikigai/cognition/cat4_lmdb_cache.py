"""
ikigai.cognition.cat4_lmdb_cache -- Pack 282 LMDB-backed anchor cache.

When fact count crosses the in-memory ceiling (Pack 279 compact gives us
~40 MB per 1 M facts, so ~5 M facts fits in the 1 GB Prince allotted for
cache; beyond that the dict swaps and lookup latency explodes), the
cache is moved to LMDB.  LMDB is a B-tree over a memory-mapped file:
hot keys end up in OS page cache automatically, cold keys read from
SSD in ~50 us.  Same dict-like interface as `CompactAnchorCache`, so
the cat4 absorb + general_reasoner lookup paths require no change.

Sizing math (per-entry):
    in-memory dict (Pack 273+274)  : ~290 bytes
    in-memory compact (Pack 279)   : ~50 bytes
    LMDB on disk (Pack 282)        : ~40 bytes SSD + ~5 bytes RAM
        (RAM = OS page cache slice for the hot working set;
         we do NOT keep an in-process LRU on top -- LMDB +
         mmap is already faster than rolling our own.)

Persistence model:
    Cache lives at organism.ikg sidecar dir `organism.ikg.lmdb/`.
    save_ikg writes the substrate; LMDB env stays open and flushes
    on commit.  load_ikg opens the sidecar env (created on first
    write if missing).  No bytes copied into organism.ikg attrs.

Compatibility:
    Drop-in for `CompactAnchorCache` via the same back-compat shim
    (string anchor in, list-of-tuples out).
"""
import os
import struct

try:
    import lmdb as _lmdb
    HAVE_LMDB = True
except ImportError:
    _lmdb = None
    HAVE_LMDB = False

from ikigai.cognition.cat4_compact_cache import CompactAnchorCache


_KEY_STRUCT = struct.Struct('>Q')        # big-endian uint64 -- 8 bytes


def _pack_key(k):
    return _KEY_STRUCT.pack(int(k) & 0xFFFFFFFFFFFFFFFF)


def _unpack_key(b):
    return _KEY_STRUCT.unpack(b)[0]


class LMDBAnchorCache:
    """LMDB-backed dict[int64, bytes] with the CompactAnchorCache API.

    Open one env per organism.  Writes are committed in the same call
    so a process crash never loses anchors -- LMDB is ACID.
    """

    _STR_PREFIX = CompactAnchorCache._STR_PREFIX

    def __init__(self, path, map_size_gb=1.0, readonly=False):
        if not HAVE_LMDB:
            raise RuntimeError(
                'lmdb not installed.  pip install lmdb')
        self.path = str(path)
        self.readonly = bool(readonly)
        os.makedirs(self.path, exist_ok=True)
        # subdir=True so LMDB lives in a directory (data.mdb + lock.mdb)
        # max_dbs=1 keeps the env minimal; we only use the default db.
        self._env = _lmdb.open(
            self.path,
            map_size=int(map_size_gb * (1 << 30)),
            subdir=True,
            readonly=self.readonly,
            create=not self.readonly,
            sync=True,
            map_async=True,         # async msync; safer + faster
            metasync=True,
            lock=not self.readonly,
            max_dbs=1)
        self._closed = False

    # ---- key + value codecs (delegate to CompactAnchorCache) ---------

    key_from_str_anchor = staticmethod(
        CompactAnchorCache.key_from_str_anchor)
    str_anchor_from_key = staticmethod(
        CompactAnchorCache.str_anchor_from_key)
    key_from_toks = staticmethod(CompactAnchorCache.key_from_toks)
    value_from_token_tuple = staticmethod(
        CompactAnchorCache.value_from_token_tuple)
    token_tuple_from_value = staticmethod(
        CompactAnchorCache.token_tuple_from_value)

    def _coerce_key(self, key):
        if isinstance(key, str):
            return self.key_from_str_anchor(key)
        return int(key)

    def _coerce_value(self, val):
        if isinstance(val, (bytes, bytearray)):
            return bytes(val)
        if isinstance(val, list):
            if not val:
                return b''
            last = val[-1]
            if isinstance(last, tuple):
                return self.value_from_token_tuple(last)
            return self.value_from_token_tuple((str(last),))
        if isinstance(val, tuple):
            return self.value_from_token_tuple(val)
        if isinstance(val, str):
            return val.encode('utf-8')
        raise TypeError(
            f'unsupported cache value type {type(val).__name__}')

    # ---- dict-like API ----------------------------------------------

    def __setitem__(self, key, value):
        if self.readonly:
            raise RuntimeError('cache opened readonly')
        k = _pack_key(self._coerce_key(key))
        v = self._coerce_value(value)
        with self._env.begin(write=True) as txn:
            txn.put(k, v, overwrite=True)

    def __getitem__(self, key):
        k = _pack_key(self._coerce_key(key))
        with self._env.begin(write=False) as txn:
            v = txn.get(k)
        if v is None:
            raise KeyError(key)
        return [self.token_tuple_from_value(v)]

    def get(self, key, default=None):
        k = _pack_key(self._coerce_key(key))
        with self._env.begin(write=False) as txn:
            v = txn.get(k)
        if v is None:
            return default
        return [self.token_tuple_from_value(v)]

    def __contains__(self, key):
        k = _pack_key(self._coerce_key(key))
        with self._env.begin(write=False) as txn:
            return txn.get(k) is not None

    def __len__(self):
        with self._env.begin(write=False) as txn:
            return int(txn.stat()['entries'])

    def __iter__(self):
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for k in cursor.iternext(keys=True, values=False):
                yield self.str_anchor_from_key(_unpack_key(bytes(k)))

    def items(self):
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                yield (self.str_anchor_from_key(_unpack_key(bytes(k))),
                        [self.token_tuple_from_value(bytes(v))])

    def keys(self):
        return iter(self)

    def values(self):
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for v in cursor.iternext(keys=False, values=True):
                yield [self.token_tuple_from_value(bytes(v))]

    # ---- bulk ops for migration / Pack 282 throughput ---------------

    def update_from_compact(self, compact_cache):
        """Bulk-import a CompactAnchorCache into LMDB.  Single big
        write txn so OS sync overhead amortizes."""
        if self.readonly:
            raise RuntimeError('cache opened readonly')
        with self._env.begin(write=True) as txn:
            for k, v in compact_cache._d.items():
                txn.put(_pack_key(k), v, overwrite=True)

    def update_from_dict(self, d):
        """Bulk-import a Pack 273-style dict[str, list[tuple[str]]]."""
        if self.readonly:
            raise RuntimeError('cache opened readonly')
        with self._env.begin(write=True) as txn:
            for k, v in d.items():
                ik = self._coerce_key(k)
                iv = self._coerce_value(v)
                txn.put(_pack_key(ik), iv, overwrite=True)

    # ---- persistence + sizing --------------------------------------

    def disk_bytes(self):
        """ACTUAL data bytes used in LMDB (sum of branch + leaf +
        overflow pages × page size).  Distinct from data.mdb file
        size, which is pre-allocated to map_size on Windows/macOS
        so file-size reporting massively overstates real usage."""
        with self._env.begin(write=False) as txn:
            s = txn.stat()
        psize = int(s.get('psize', 4096))
        used_pages = (int(s.get('branch_pages', 0))
                        + int(s.get('leaf_pages', 0))
                        + int(s.get('overflow_pages', 0)))
        return psize * used_pages

    def file_bytes(self):
        """Allocated data.mdb file size (= map_size on Windows).  For
        comparison vs `disk_bytes()` which is the real data weight."""
        data = os.path.join(self.path, 'data.mdb')
        if os.path.exists(data):
            return os.path.getsize(data)
        return 0

    def close(self):
        if not self._closed:
            self._env.close()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
