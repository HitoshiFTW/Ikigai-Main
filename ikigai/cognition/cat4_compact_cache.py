"""
ikigai.cognition.cat4_compact_cache -- Pack 279 compact anchor cache.

Pack 273+274 used dict[str_anchor, list[tuple[str]]] with anchors like
'__state_701357096285' (70 bytes per key string + 32 bytes dict slot)
and values like [('berlin',)] (60 bytes per tuple-in-list + 32 bytes slot
+ 32 bytes intern overhead).  Per-entry footprint: ~270 bytes.

At 1M facts that footprint is 270 MB -- exceeds the 192 MB substrate cap.
At 1B (brain-scale) it is 270 GB -- unreachable on a single device.

Pack 279 swaps the representation to dict[int64, bytes] where the int64
key is the raw blake2b digest used to produce the original string anchor
and the bytes value is the utf-8 encoding of the canonical action tokens
joined by a single space.  Per-entry footprint drops to ~50 bytes (8 key
+ 32 slot + ~10 utf-8 payload).  1M facts in 50 MB; 1B in 50 GB (still
needs LMDB Pack 282 for that, but the per-entry overhead is fixed).

Semantics change: last-wins on duplicate anchor writes.  Pack 273+274
stored the FULL list of unique observed token tuples and the runtime
read used the LAST entry anyway.  Dropping the list is faithful to the
runtime behavior.

Migration: `migrate_dict_cache(old_dict)` walks the existing string-keyed
dict and produces a CompactAnchorCache.  The string anchor format is
'__state_' + str(int.from_bytes(blake2b8(...), 'big') % 10**12), so the
integer suffix parses back without recomputing the hash.
"""
import hashlib
import sys


class CompactAnchorCache:
    """dict[int64, bytes] with the back-compat surface Pack 273 readers
    expect (string anchors return list-of-tuples).

    Reads return list-of-tuples (the Pack 273 readers expect this shape:
    `chosen = cache_entry[-1]` followed by `' '.join(chosen)`).  Writes
    accept either bytes or Pack 273-style list-of-tuples (the last tuple
    in the list wins).
    """

    __slots__ = ('_d',)

    _STR_PREFIX = '__state_'
    # Pack 330: multi-value payload separator.  A key's bytes value holds one
    # OR MORE distinct action values, each a space-joined token string, joined
    # by this unit-separator (never appears in tokenized word text).  A single
    # value (no separator) round-trips identically to the Pack 279 layout, so
    # old persisted caches load unchanged.  Readers using entry[-1] still get
    # the most-recently-appended value (last-wins preserved); atoms() gets all.
    _VSEP = b'\x1f'

    def __init__(self):
        self._d = {}

    # ---- key + value codecs ----------------------------------------

    @staticmethod
    def key_from_str_anchor(s):
        """'__state_701357096285' -> int."""
        if s.startswith(CompactAnchorCache._STR_PREFIX):
            return int(s[len(CompactAnchorCache._STR_PREFIX):])
        return int(s)

    @staticmethod
    def str_anchor_from_key(k):
        return f'{CompactAnchorCache._STR_PREFIX}{int(k)}'

    @staticmethod
    def key_from_toks(state_toks):
        """blake2b 8-byte digest -> int (matches cat4_absorb _stable_anchor)."""
        h = hashlib.blake2b(
            '|'.join(state_toks).encode('utf-8'),
            digest_size=8).digest()
        return int.from_bytes(h, 'big') % 10**12

    @staticmethod
    def value_from_token_tuple(tok_tuple):
        """('paris',) -> b'paris'; ('mexico', 'city') -> b'mexico city'."""
        if not tok_tuple:
            return b''
        return ' '.join(str(t) for t in tok_tuple).encode('utf-8')

    @staticmethod
    def token_tuple_from_value(b):
        if not b:
            return ()
        # back-compat: the LAST value if multi (matches entry[-1] readers)
        if CompactAnchorCache._VSEP in b:
            b = b.rsplit(CompactAnchorCache._VSEP, 1)[-1]
        return tuple(b.decode('utf-8').split(' '))

    @staticmethod
    def token_tuples_from_value(b):
        """Pack 330 -- ALL values stored at a key, as a list of token-tuples.
        Splits the multi-value payload on _VSEP; a single-value (Pack 279)
        payload yields a 1-element list."""
        if not b:
            return [()]
        return [tuple(part.decode('utf-8').split(' '))
                for part in b.split(CompactAnchorCache._VSEP)]

    # ---- dict-like API with back-compat -----------------------------

    def _coerce_key(self, key):
        if isinstance(key, str):
            return self.key_from_str_anchor(key)
        return int(key)

    def _coerce_value(self, val):
        """Pack 273-style list-of-tuples OR bytes -> bytes.  Pack 330: a list
        keeps ALL distinct values (multi-value payload), not just the last."""
        if isinstance(val, (bytes, bytearray)):
            return bytes(val)
        if isinstance(val, list):
            if not val:
                return b''
            parts = []
            seen = set()
            for item in val:
                if isinstance(item, tuple):
                    enc = self.value_from_token_tuple(item)
                else:
                    enc = self.value_from_token_tuple((str(item),))
                if enc and enc not in seen:
                    seen.add(enc)
                    parts.append(enc)
            return self._VSEP.join(parts)
        if isinstance(val, tuple):
            return self.value_from_token_tuple(val)
        if isinstance(val, str):
            return val.encode('utf-8')
        raise TypeError(f'unsupported cache value type {type(val).__name__}')

    def add_value(self, key, tok_tuple):
        """Pack 330 -- append a distinct value to a key's multi-value payload,
        preserving insertion order (so entry[-1] stays the most recent).  This
        is the write path that actually persists multiple values (the old
        `cache.get(k).append(...)` idiom mutated a throwaway list under the
        compact representation).  Returns True if the value was newly stored
        (new key, or a value not already present), False if a duplicate."""
        enc = self.value_from_token_tuple(
            tok_tuple if isinstance(tok_tuple, tuple) else (str(tok_tuple),))
        if not enc:
            return False
        k = self._coerce_key(key)
        cur = self._d.get(k)
        if cur is None:
            self._d[k] = enc
            return True
        if enc in cur.split(self._VSEP):
            return False
        self._d[k] = cur + self._VSEP + enc
        return True

    def __setitem__(self, key, value):
        self._d[self._coerce_key(key)] = self._coerce_value(value)

    def __getitem__(self, key):
        v = self._d[self._coerce_key(key)]
        return self.token_tuples_from_value(v)

    def get(self, key, default=None):
        v = self._d.get(self._coerce_key(key))
        if v is None:
            return default
        return self.token_tuples_from_value(v)

    def __contains__(self, key):
        return self._coerce_key(key) in self._d

    def __delitem__(self, key):
        del self._d[self._coerce_key(key)]

    def pop(self, key, *default):
        return self._d.pop(self._coerce_key(key), *default)

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        for k in self._d:
            yield self.str_anchor_from_key(k)

    def items(self):
        for k, v in self._d.items():
            yield (self.str_anchor_from_key(k),
                    self.token_tuples_from_value(v))

    def keys(self):
        return iter(self)

    def values(self):
        for v in self._d.values():
            yield self.token_tuples_from_value(v)

    # ---- persistence helpers ---------------------------------------

    def to_persist_state(self):
        """Return picklable state for integrate._PERSIST_ATTRS.

        A plain dict[int, bytes] persists smaller than the wrapper.
        """
        return dict(self._d)

    @classmethod
    def from_persist_state(cls, state):
        c = cls()
        if isinstance(state, dict):
            for k, v in state.items():
                c._d[int(k)] = (
                    v if isinstance(v, (bytes, bytearray)) else
                    cls.value_from_token_tuple(
                        v if isinstance(v, tuple) else (str(v),)))
        return c

    # ---- size accounting -------------------------------------------

    def memory_bytes(self):
        """Coarse estimate of in-RAM footprint.

        Per-entry: 8 bytes int key + ~32 bytes dict slot + len(value).
        Plus base dict overhead (~64 bytes).  Tracks pretty closely
        with the actual sys.getsizeof on small dicts; underestimates
        slightly on large dicts due to load-factor padding.
        """
        per = sum(8 + 32 + len(v) for v in self._d.values())
        return 64 + per


def migrate_dict_cache(old):
    """Pack 273/274 dict -> CompactAnchorCache.

    `old` must be a dict[str_anchor, list[tuple[str]]] following the
    Pack 273 layout.  Last entry in each value list wins (matches
    runtime read semantics).  Returns a new CompactAnchorCache with
    len() == len(old).
    """
    out = CompactAnchorCache()
    for k, v in old.items():
        if isinstance(v, list) and v:
            last = v[-1]
        else:
            last = v
        if isinstance(last, tuple):
            payload = CompactAnchorCache.value_from_token_tuple(last)
        elif isinstance(last, str):
            payload = last.encode('utf-8')
        elif isinstance(last, (bytes, bytearray)):
            payload = bytes(last)
        else:
            payload = b''
        out._d[CompactAnchorCache.key_from_str_anchor(k)] = payload
    return out


def measure_dict_cache_bytes(d):
    """Walk a Pack 273-style dict[str, list[tuple[str]]] and estimate
    its in-RAM footprint via sys.getsizeof on each component."""
    total = sys.getsizeof(d)
    for k, v in d.items():
        total += sys.getsizeof(k)
        total += sys.getsizeof(v)
        if isinstance(v, list):
            for t in v:
                total += sys.getsizeof(t)
                if isinstance(t, tuple):
                    for s in t:
                        total += sys.getsizeof(s)
    return total
