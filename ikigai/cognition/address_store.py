"""
ikigai.cognition.address_store -- FACTS AS PACKED ADDRESS-LISTS.

Day 96.  The production form of the founder's Day-90 idea ("facts as address-lists"):

    "A new fact brings no new words -- all its words are already addresses in the fixed lexicon;
     a fact is a LIST of addresses, and its consequences are derived."   -- Day_090, section 4

What is FREE (and stays free):
  * WORDS.  An entity's substrate address is COMPUTED from its characters (flat_memory.ComputedKey);
    zero bytes per word, unbounded vocabulary.  We never store word vectors.
  * CONSEQUENCES.  Transitive closure / inheritance / composition are DERIVED on demand from the
    stored atoms.  Stored zero.
  * THE SUBSTRATE.  The SDM banks are fixed size; writing more into them adds no bytes.

What COSTS bytes (the only thing, and it is irreducible):
  * THE ATOMIC EDGES -- *which* combinations are true.  "paris capitalOf france" is one triple out
    of |E|^2 * |R| possible ones; which ones hold is information about the WORLD, not derivable from
    the lexicon or from geometry.  Shannon floor ~= log2(possible / true) bits per edge (~4 B at
    world scale).  Superposition does NOT dodge this: bundle capacity is ~O(d) before crosstalk
    (the Frady-Sommer wall this project measured as the "20k knee"), so holding N facts readably in
    superposition needs d ~ N -- the vector IS the bytes.

The waste this module removes:  the running derive engine (`compositional.py`) stores each fact as
PYTHON STRINGS in a dict -- measured 662 B/edge, i.e. ~150x above the Shannon floor (4.16M Wikidata
p279 edges -> 3.5 GB RSS).  Here a fact is three int32 indices over a shared lexicon, held in a
packed CSR array: 4 B/edge, exact, with an exact BFS closure.  Same answers, ~150x less RAM.

Reasoning never needs the surface strings -- only the integer addresses.  The str<->id lexicon is
required ONLY to EMIT text ("surface strings only at the edges", Day-90).  Build with
`keep_lexicon=False` for a reasoning-only store that carries no vocabulary table at all.
"""

import os
import zlib
from array import array

import numpy as np

_BLOCK = 64          # surface strings per compressed block (measured: 3.3x, ~4 us to inflate)
_BLKCACHE = 16       # inflated blocks held (an emitted answer touches only a handful)


def _arr_i(np_int32):
    """numpy int32 array -> array.array('i') with no intermediate python list (frombytes = memcpy).
    Same 4 B/element footprint; python-int element access (the hot-loop win)."""
    a = array('i')
    a.frombytes(np.ascontiguousarray(np_int32, dtype=np.int32).tobytes())
    return a


def _nbytes(a):
    """Bytes held by any of the forms an array takes in this module: array.array (built),
    memoryview (mmap-loaded), bytes (nibbles/escapes), ndarray, or PackedParents."""
    if a is None:
        return 0
    if isinstance(a, array):
        return a.buffer_info()[1] * a.itemsize
    if isinstance(a, memoryview):
        return a.nbytes
    nb = getattr(a, 'nbytes', None)
    if callable(nb):                           # PackedParents.nbytes() is a method, not an attr
        return int(nb())
    return int(nb) if nb is not None else len(a)


class PackedParents:
    """The parent array of a near-tree relation, DICTIONARY-CODED.  O(1) random access.

    Day 97, and it corrects a claim this module itself made.  The docstring above asserted a
    "Shannon floor ~4 B/edge", reasoning that a parent id is one of |E| entities so it must cost
    log2(3.58M) = 21.8 bits.  That argument assumes the parent is UNIFORM.  It is not, and nobody
    had measured it.  On Wikidata p279 (MEASURED, day97):

        distinct parents        262,332   (for 3.52M children)
        top-10 parents          68.6% of ALL 4.16M edges
        q11173  chemical compound   802,242 direct children
        q20747295 protein-coding gene 773,880
        H(parent)               7.07 bits = 0.88 B      <- not 21.8 bits, not 4 B

    The parent id was never 4 bytes of information.  So: rank the parents by frequency, and spend
    bits in proportion to how often a parent is actually used.

        code[child] = 4 bits
            0..13  -> the 14 most common parents, DIRECT (covers 69.3% of children)
            14     -> NO PARENT (a root)
            15     -> escape: the full rank lives in a side array
        esc[]  = EW-bit packed ranks for the 30.7% that escape
        ptab[] = rank -> entity id
        EW     = ceil(log2(#parents)) -- SIZED FROM THE DATA, not hardcoded.  Day-98 fix: this was
                 a hardcoded 18 bits (max 262,143 parents).  Wikidata p279 has 262,332 distinct
                 parents -- already 189 OVER the limit, so the rarest parents silently decoded to
                 the WRONG entity (the Day-97 gate's 2000-sample missed them).  A latent
                 correctness bug.  EW now holds the true max rank; verified exact to 1e8 nodes.

    Random access stays O(1) and BRANCH-CHEAP.  To find an escape's slot we need "how many escapes
    precede child i".  A scan of the child's 64-nibble block gives that -- and MEASURED, that scan
    cost 2,128 ns/access (12x the raw int32's 175 ns): a python loop over 32 nibbles, on 31% of
    lookups, on every hop of every walk.  So there is no scan: each block carries a 64-bit MASK of
    which of its children escape, and the slot is `blk[b] + popcount(mask & ((1<<j)-1))` -- one
    int.bit_count(), no loop.  The mask costs 1 bit/child and buys back the whole latency."""

    FAST = 14                     # codes 0..13 are direct ranks
    NOPAR = 14                    # code 14 = no parent (root)
    ESC = 15                      # code 15 = escape
    BLK = 64                      # children per block (one 64-bit escape mask each)

    __slots__ = ('nib', 'esc', 'blk', 'mask', 'ptab', 'n', 'ew')

    def __init__(self, nib, esc, blk, mask, ptab, n, ew):
        self.nib, self.esc, self.blk = nib, esc, blk
        self.mask, self.ptab, self.n, self.ew = mask, ptab, int(n), int(ew)

    @classmethod
    def build(cls, first):
        """first: int32[n], parent entity id per child, -1 for a root."""
        n = len(first)
        fa = np.asarray(first, dtype=np.int64)
        has = fa >= 0
        # frequency-rank the parents (the whole trick: spend bits where the mass is)
        vals, counts = np.unique(fa[has], return_counts=True)
        order = np.argsort(-counts, kind='stable')
        ptab = vals[order].astype(np.int32)                    # rank -> entity id
        rank_of = np.full(int(fa.max()) + 2 if len(fa) else 1, -1, dtype=np.int32)
        rank_of[ptab] = np.arange(len(ptab), dtype=np.int32)
        rank = np.where(has, rank_of[np.clip(fa, 0, None)], -1)

        code = np.full(n, cls.ESC, dtype=np.uint8)
        code[~has] = cls.NOPAR
        direct = has & (rank < cls.FAST)
        code[direct] = rank[direct].astype(np.uint8)
        esc_mask = has & (rank >= cls.FAST)
        esc_ranks = rank[esc_mask].astype(np.int64)

        # nibbles, 2 children per byte (little end first)
        pad = (-n) % 2
        c = np.concatenate([code, np.zeros(pad, np.uint8)])
        nib = (c[0::2] | (c[1::2] << 4)).astype(np.uint8)

        # escapes: EW-bit packed, LSB-first.  EW sized to hold the largest rank (= #parents-1),
        # NOT a hardcoded 18 -- else >262k parents (Wikidata has 262,332) corrupts the rare tail.
        ew = max(1, (len(ptab) - 1).bit_length()) if len(ptab) else 1
        esc = cls._pack(esc_ranks, ew)

        # per block: escapes strictly BEFORE it (int32) + a 64-bit mask of WHICH children escape.
        # slot(i) = blk[b] + popcount(mask[b] & ((1 << j) - 1))  -- O(1), no scan.
        nb = (n + cls.BLK - 1) // cls.BLK
        em = esc_mask.astype(np.int64)
        blk = np.zeros(nb + 1, dtype=np.int64)
        np.cumsum(np.add.reduceat(em, np.arange(0, n, cls.BLK)) if n else np.zeros(0),
                  out=blk[1:])
        pad = (-n) % cls.BLK
        bits = np.concatenate([esc_mask, np.zeros(pad, bool)]).reshape(nb, cls.BLK)
        w = (bits * (1 << np.arange(cls.BLK, dtype=object))).sum(axis=1)   # exact 64-bit words
        mask = array('Q'); mask.frombytes(np.asarray(w, dtype=np.uint64).tobytes())
        return cls(nib.tobytes(), esc, _arr_i(blk.astype(np.int32)), mask, _arr_i(ptab), n, ew)

    @staticmethod
    def _pack(ranks, ew):
        """LSB-first EW-bit packing.  +8 bytes of tail padding so an 8-byte read at the last
        element never runs off the end (sh<=7, ew<=57 -> fits one 64-bit little-endian read)."""
        out = bytearray((len(ranks) * ew + 7) // 8 + 8)
        for i, r in enumerate(ranks):
            bit = i * ew
            byi, sh = bit >> 3, bit & 7
            v = int(r) << sh
            for k in range(8):                                 # spread across up to 8 bytes
                if not v:
                    break
                out[byi + k] |= v & 0xFF
                v >>= 8
        return bytes(out)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        """parent entity id of child i, or -1 if a root.  O(1) -- one nibble, one popcount."""
        c = (self.nib[i >> 1] >> ((i & 1) << 2)) & 0xF
        if c == self.NOPAR:
            return -1
        if c != self.ESC:
            return self.ptab[c]
        b, j = i >> 6, i & 63                                   # BLK = 64
        k = self.blk[b] + (self.mask[b] & ((1 << j) - 1)).bit_count()
        bit = k * self.ew
        byi, sh = bit >> 3, bit & 7
        e = self.esc
        w = int.from_bytes(bytes(e[byi:byi + 8]), 'little')    # one 64-bit little-endian window
        return self.ptab[(w >> sh) & ((1 << self.ew) - 1)]

    def nbytes(self):
        return sum(_nbytes(x) for x in (self.nib, self.esc, self.blk, self.mask, self.ptab))


class OverflowParents:
    """The EXTRA parents of multi-parent children (a near-tree node with >1 parent).  Day 97.

    Day 96 stored these as three int32 CSR arrays -- ov_subj (the child ids), ov_indptr (a CSR
    pointer per child), ov_tgt (the extra parent ids).  MEASURED on Wikidata p279 that was 7.25 MB
    for 642k edges = 11.3 B/edge, i.e. 54% of the whole store and 7x fatter per edge than the main
    parent array.  Two of the three arrays were pure waste:

      * ov_subj (2.34 MB) is REDUNDANT with the `multi` bitmask that already marks which children
        overflow.  A child's rank among overflow-children = popcount of the multi bits before it --
        the same O(1) trick PackedParents uses for escapes.  So ov_subj is dropped entirely and a
        tiny per-64-block cumulative-popcount array (`bpop`, 4 B per 64 children) stands in.
      * ov_tgt (2.57 MB) has the SAME parent skew as the main array (H = 6.78 bits), so it is
        dictionary-coded by the same PackedParents.

    What survives is ov_indptr (a real CSR: a child can have 3+ parents) and the small bpop.
    Result MEASURED: 7.25 MB -> ~3.3 MB, exact, O(1) access."""

    __slots__ = ('bpop', 'indptr', 'tgt', 'n')

    def __init__(self, bpop, indptr, tgt, n):
        self.bpop, self.indptr, self.tgt, self.n = bpop, indptr, tgt, int(n)

    @classmethod
    def build(cls, flags, ex_subs, ex_tgts, n):
        """flags: bool[n], True where the child has extra parents (== the multi bitmask).
        ex_subs/ex_tgts: the overflow edges, sorted by subject (so unique subjects are contiguous)."""
        nb = (n + 63) // 64
        pad = (-n) % 64
        fb = np.concatenate([flags, np.zeros(pad, bool)]).reshape(nb, 64).sum(1)
        bpop = np.zeros(nb + 1, dtype=np.int64)
        np.cumsum(fb, out=bpop[1:])                         # overflow-children before each 64-block
        # CSR over the unique overflow subjects, in the SAME order popcount-rank produces (sorted)
        uniq, first_idx, counts = np.unique(ex_subs, return_index=True, return_counts=True)
        indptr = np.zeros(len(uniq) + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])
        order = np.argsort(ex_subs, kind='stable')
        tgt = PackedParents.build(np.asarray(ex_tgts, dtype=np.int32)[order])   # dict-code the ids
        return cls(_arr_i(bpop.astype(np.int32)), _arr_i(indptr), tgt, n)

    def targets(self, si, multi):
        """The extra parent ids of child si.  O(1): one popcount for the rank, then a CSR slice.
        `multi` is the func relation's bitmask (bytes/memoryview), passed in (not duplicated)."""
        b = si >> 6
        word = int.from_bytes(bytes(multi[b * 8:b * 8 + 8]), 'little')
        j = self.bpop[b] + (word & ((1 << (si & 63)) - 1)).bit_count()
        return [self.tgt[k] for k in range(self.indptr[j], self.indptr[j + 1])]

    def nbytes(self):
        return _nbytes(self.bpop) + _nbytes(self.indptr) + self.tgt.nbytes()


def _mv(buf, fmt):
    """Zero-copy typed view over an np.memmap (or any buffer).

    Day 97, MEASURED (200k random probes over a 3.58M int32 array, same page cache):
        np.memmap[i]                      473 ns   -> boxes a numpy scalar per access
        memoryview(...).cast('i')[i]      219 ns   -> a PYTHON int, no boxing
        array.array('i')  (in-RAM copy)   192 ns
    So the mmap'd store can walk at ~in-RAM speed while RAM stays flat: the memoryview is a view,
    not a copy.  (`cast` cannot go format->format directly; hop through bytes.)"""
    return memoryview(buf).cast('B').cast(fmt)


class AddressFactStore:
    """Facts as packed address-lists: (s_id, rel, o_id) int indices in per-relation CSR arrays.

    build(triples)              intern strings -> ids, pack each relation into CSR
    objects_of(s, rel)          direct objects  (exact)
    transitive_reach(rel, x)    full ancestor closure by BFS over CSR  (exact, derived-not-stored)
    edge_bytes() / lexicon_bytes()   MEASURED footprint, not asserted
    """

    def __init__(self, keep_lexicon=True):
        self.keep_lexicon = bool(keep_lexicon)
        self._id = {}            # str -> int32 id   (ingest-time intern; dropped by compact())
        self._str = []           # int32 id -> str   (EMISSION ONLY; dropped by compact())
        self._csr = {}           # rel -> (indptr int32[n+1], indices int32[m])   general form
        self._func = {}          # rel -> (first[n], multi-bitset, ov_subj, ov_indptr, ov_tgt)
        #                          near-tree form: child = array position (0 bits), parent = int32
        self.n_entities = 0
        self.n_edges = 0
        # compact lexicon (built by compact_lexicon(); the python dict/list are then released)
        self._hash = None        # int64[n] sorted 64-bit string hashes  -> id lookup by bisect
        self._hid = None         # int32[n] ids parallel to _hash
        # SURFACE TABLE (emission only; the reasoning path never touches it).  Block-compressed:
        # K strings per block, each block one zlib payload carrying its own in-block int32 offsets.
        # That subsumes the per-string offset array (was int64[n] = 8 B/entity) into the block.
        self._blk = None         # bytes/memoryview: the concatenated compressed blocks
        self._blkoff = None      # int32[nb+1] byte offsets of each block within _blk
        self._blkk = _BLOCK      # strings per block
        self._blkcache = {}      # small LRU of inflated blocks (emit locality: an answer's ancestors
        self._blkorder = []      #   are usually a handful of ids -> a handful of blocks)
        self._stamp = None       # int32[n] visited-generation stamps (BFS accelerator, O(1) reset)
        self._gen = 0

    # ---- lexicon (addresses).  Reasoning uses ids only; strings are for the edges of the system.
    @staticmethod
    def _h64(w):
        import hashlib
        return int.from_bytes(hashlib.blake2b(w.encode('utf-8'), digest_size=8).digest(),
                              'little', signed=False) - (1 << 63)      # -> int64 range

    def intern(self, w):
        w = str(w).strip().lower()
        i = self._id.get(w)
        if i is None:
            i = len(self._id)
            self._id[w] = i
            self._str.append(w)
        return i

    def compact_lexicon(self, keep_strings=True, block=_BLOCK):
        """Replace the python str->id dict (~109 B/entity) with packed arrays.

          id lookup (REASONING edge) : sorted int64 hash + int32 id   -> 12 B/entity, binary search
          surface table (EMIT edge)  : zlib blocks of `block` strings -> ~2.7 B/entity, O(1) access

        Day-96 shipped the surface table as a raw blob + an int64 offset PER STRING: 9.06 + 8.00 =
        17.06 B/entity (61 MB of the 104 MB lexicon), and the int64 offsets indexed a 32 MB blob --
        66x more range than they could ever use.  Here each block carries its own int32 in-block
        offsets INSIDE the compressed payload, so the per-string offset array disappears entirely and
        only a block-offset array survives (4 B per `block` strings = 0.06 B/entity).  Compression is
        a general algorithm over whatever bytes the data happens to contain -- no authored vocabulary,
        nothing specific to Wikidata.

        keep_strings=False -> a REASONING-ONLY store: no surface table at all, ids only.  The Day-90
        'surface strings only at the edges' taken to its limit."""
        n = len(self._str)
        h = np.fromiter((self._h64(w) for w in self._str), dtype=np.int64, count=n)
        order = np.argsort(h, kind='stable')
        # array.array, not numpy: id_of() is on the hot path, and np.searchsorted costs ~15 us of
        # call overhead for a SINGLE lookup, while bisect over an array.array is ~1-2 us.
        ha = array('q'); ha.frombytes(np.ascontiguousarray(h[order], dtype=np.int64).tobytes())
        self._hash = ha
        self._hid = _arr_i(order.astype(np.int32))
        if keep_strings:
            self._blk, self._blkoff = self._pack_surface(self._str, block)
            self._blkk = block
            self.keep_lexicon = True
        else:
            self._blk = None; self._blkoff = None
            self.keep_lexicon = False
        self._blkcache = {}; self._blkorder = []
        self._id = {}            # release the python dict + list (the 390 MB)
        self._str = []
        return self.lexicon_bytes()

    @staticmethod
    def _pack_surface(strs, block):
        """Compress the surface strings into self-describing zlib blocks of `block` entries.

        Block payload = (varint len, utf-8 bytes) * block.  Length-prefixed, NOT an offset table:
        an int32 offset array inside the block costs 4 B/string and barely compresses (measured:
        it inflated the packed surface table from 5.3 MB to 19.1 MB -- the offsets were bigger than
        the strings they indexed).  A varint length is 1 B for anything under 128 bytes, and length
        bytes are highly repetitive, so zlib eats them.  Decoding is a linear scan of `block` items,
        which is ~64 iterations of nothing."""
        n = len(strs)
        nb = (n + block - 1) // block
        blocks, boff = [], np.zeros(nb + 1, dtype=np.int64)
        for bi in range(nb):
            buf = bytearray()
            for s in strs[bi * block:(bi + 1) * block]:
                e = s.encode('utf-8')
                m = len(e)
                while m >= 0x80:                      # LEB128 varint: no length cap, 1 B in practice
                    buf.append((m & 0x7F) | 0x80)
                    m >>= 7
                buf.append(m)
                buf += e
            blocks.append(zlib.compress(bytes(buf), 6))
            boff[bi + 1] = boff[bi] + len(blocks[-1])
        if boff[-1] >= (1 << 31):                     # keep block offsets int32-addressable
            raise ValueError('surface table exceeds 2 GB compressed; raise the block size')
        return b''.join(blocks), _arr_i(boff.astype(np.int32))

    def id_of(self, w):
        w = str(w).strip().lower()
        if self._id:
            return self._id.get(w)
        if self._hash is None:
            return None
        import bisect
        k = self._h64(w)
        j = bisect.bisect_left(self._hash, k)          # C bisect over array.array (~1-2 us)
        if j < len(self._hash) and self._hash[j] == k:
            return self._hid[j]
        return None

    def str_of(self, i):
        """id -> surface string.  THE EMISSION EDGE, and the only place strings exist at all.
        Inflates one block (~600 B) and caches it; a reasoning-only store returns None."""
        if self._str:
            return self._str[i] if 0 <= i < len(self._str) else None
        if self._blk is None or not (0 <= i < self.n_entities):
            return None                      # reasoning-only store: no surface table
        b = i // self._blkk
        blk = self._blkcache.get(b)
        if blk is None:
            if b + 1 >= len(self._blkoff):
                return None
            raw = zlib.decompress(bytes(self._blk[self._blkoff[b]:self._blkoff[b + 1]]))
            blk, p, m = [], 0, len(raw)                  # unpack the varint-length-prefixed block
            while p < m:
                ln, sh = 0, 0
                while True:
                    c = raw[p]; p += 1
                    ln |= (c & 0x7F) << sh
                    if not c & 0x80:
                        break
                    sh += 7
                blk.append(raw[p:p + ln]); p += ln
            self._blkcache[b] = blk
            self._blkorder.append(b)
            if len(self._blkorder) > _BLKCACHE:           # tiny LRU: an answer touches a few blocks
                self._blkcache.pop(self._blkorder.pop(0), None)
        j = i - b * self._blkk
        return blk[j].decode('utf-8') if j < len(blk) else None

    # ---- build: pack each relation into the encoding its DEGREE DISTRIBUTION calls for
    def build(self, triples, functional_max_degree=1.35):
        """triples: iterable of (subj, rel, obj) strings.

        Encoding is chosen from the DATA, not hardcoded. Two forms:

        FUNCTIONAL (near-tree: mean out-degree of edge-bearing subjects < functional_max_degree).
          Taxonomies are like this -- Wikidata p279 has 1.16 parents/class. CSR then wastes an
          entire int32 offset PER ENTITY to index a list that is usually one element long: measured
          3.44 of the 7.44 B/edge was `indptr`, i.e. ~46% pure overhead.
          Instead: `first[child] = parent` -- the array POSITION is the child, so the child costs
          ZERO bits, and one int32 holds the edge. Nodes with >1 parent set a bit and put their
          extra parents in a small overflow CSR. Walking up is then a pointer chase, no slicing.

        CSR (everything else): the general sorted-adjacency form.
        """
        by_rel = {}
        for s, r, o in triples:
            si, oi = self.intern(s), self.intern(o)
            by_rel.setdefault(str(r).strip().lower(), []).append((si, oi))
        self.n_entities = len(self._id)
        self.n_edges = 0
        for rel, pairs in list(by_rel.items()):
            arr0 = np.asarray(pairs, dtype=np.int32)
            counts0 = np.bincount(arr0[:, 0], minlength=self.n_entities)
            with_edges = int((counts0 > 0).sum())
            mean_deg = len(pairs) / max(1, with_edges)
            if mean_deg < functional_max_degree:
                self._build_functional(rel, arr0, counts0)
                self.n_edges += len(pairs)
                del by_rel[rel]
        for rel, pairs in by_rel.items():
            arr = np.asarray(pairs, dtype=np.int32)              # (m, 2)
            order = np.argsort(arr[:, 0], kind='stable')         # sort by subject -> CSR
            arr = arr[order]
            counts = np.bincount(arr[:, 0], minlength=self.n_entities).astype(np.int32)
            indptr = np.zeros(self.n_entities + 1, dtype=np.int32)
            np.cumsum(counts, out=indptr[1:])
            indices = np.ascontiguousarray(arr[:, 1], dtype=np.int32)
            # Store as array.array('i'), NOT numpy: identical 4 B/element, but element access
            # returns a PYTHON int with no numpy-scalar boxing. The closure walk is a tight loop
            # over tiny frontiers (mean depth ~6, 1-5 nodes wide), where numpy's ~1-10 us per-call
            # overhead dominates -- measured: numpy per-node 267 us, numpy vectorized 978 us,
            # array.array ~30 us (parity with the python dict walk it replaces).
            self._csr[rel] = (_arr_i(indptr), _arr_i(indices))
            self.n_edges += len(indices)
        return {'entities': self.n_entities, 'edges': self.n_edges,
                'relations': len(self._csr) + len(self._func),
                'functional': sorted(self._func), 'csr': sorted(self._csr)}

    def _build_functional(self, rel, arr, counts):
        """Near-tree relation -> parent-array + overflow. The child is the ARRAY POSITION, so it
        costs zero bits; one int32 carries the edge. Kills the per-entity `indptr` (46% of a CSR
        on this graph). Multi-parent nodes flag a bit and spill into a small overflow CSR."""
        n = self.n_entities
        order = np.argsort(arr[:, 0], kind='stable')
        subs = arr[order, 0]
        tgts = arr[order, 1]
        is_first = np.ones(len(subs), dtype=bool)
        is_first[1:] = subs[1:] != subs[:-1]
        first = np.full(n, -1, dtype=np.int32)
        first[subs[is_first]] = tgts[is_first]                     # the single parent, in place
        ex_subs = subs[~is_first]
        ex_tgts = tgts[~is_first]
        if len(ex_subs):
            flags = np.zeros(n, dtype=bool)
            flags[np.unique(ex_subs)] = True
            multi = np.packbits(flags, bitorder='little').tobytes()  # 1 bit/entity
            # Day-97: overflow was 54% of the store (7.25 MB).  Drop ov_subj (redundant with the
            # multi bitmask -> popcount rank) and dict-code the targets.  See OverflowParents.
            overflow = OverflowParents.build(flags, ex_subs, ex_tgts, n)
        else:
            multi = b'\x00' * ((n + 7) // 8)
            overflow = None
        # Day-97: the parent array is DICTIONARY-CODED (see PackedParents).  The parent id is not
        # 4 B of information -- MEASURED H(parent) = 7.07 bits on Wikidata p279, because 10 classes
        # carry 68.6% of all edges.  14.32 MB -> 5.70 MB, exact, O(1) access.
        self._func[rel] = (PackedParents.build(first), multi, overflow)

    # ---- reads (exact, on addresses)
    def edges(self):
        """Day-99 -- stream every (subj, rel, obj) edge back out as strings: the inverse of build().

        Exists so a store can be MERGED with new facts (build() is all-or-nothing, and there was no
        way to get the edges back out -- so `ingest_addressed` could only REPLACE, silently
        destroying the previous corpus).

        A GENERATOR on purpose. The whole point of the packed form is that the string triples do NOT
        exist in RAM (662 B/edge -> 4.16M edges = 3.5 GB). Materialising them all at once undoes
        that, so callers must stream and stay bounded. Requires surface strings (a store loaded with
        surface=False has no str_of and cannot be merged -- rebuild from source instead)."""
        for rel in sorted(self._func):
            first = self._func[rel][0]
            for si in range(len(first)):
                s = self.str_of(si)
                if not s:
                    continue
                for oi in self.objects_of(si, rel):
                    o = self.str_of(int(oi))
                    if o:
                        yield (s, rel, o)
        for rel in sorted(self._csr):
            indptr, indices = self._csr[rel]
            for si in range(len(indptr) - 1):
                if indptr[si] == indptr[si + 1]:
                    continue
                s = self.str_of(si)
                if not s:
                    continue
                for k in range(indptr[si], indptr[si + 1]):
                    o = self.str_of(int(indices[k]))
                    if o:
                        yield (s, rel, o)

    def has_relation(self, rel):
        """True if this store holds the relation (and therefore ALL of its edges -- the store is
        built from the complete edge set, so it is AUTHORITATIVE for the relations it carries)."""
        rel = str(rel).strip().lower()
        return rel in self._csr or rel in self._func

    def objects_of(self, s, rel):
        """Direct objects of s under rel. Accepts a string or an int id. Returns int ids."""
        rel = str(rel).strip().lower()
        si = s if isinstance(s, (int, np.integer)) else self.id_of(s)
        if si is None:
            return []
        fn = self._func.get(rel)
        if fn is not None:
            first, multi, overflow = fn
            if si >= len(first):
                return []
            out = []
            p = first[si]
            if p >= 0:
                out.append(p)
            if overflow is not None and multi[si >> 3] >> (si & 7) & 1:
                out.extend(overflow.targets(si, multi))
            return out
        csr = self._csr.get(rel)
        if csr is None or si + 1 >= len(csr[0]):
            return []
        indptr, indices = csr
        return indices[indptr[si]:indptr[si + 1]].tolist()

    def transitive_reach(self, rel, x, max_hops=64):
        """Full transitive closure from x under rel -- DERIVED by address arithmetic, stored zero.
        Returns a set of int ids (the ancestors of x, excluding x). Exact: same answers as a
        string-dict graph walk, at 4 B/edge instead of 662."""
        rel = str(rel).strip().lower()
        xi = x if isinstance(x, (int, np.integer)) else self.id_of(x)
        if xi is None:
            return set()
        fn = self._func.get(rel)
        if fn is not None:
            return self._reach_functional(fn, int(xi), max_hops)
        csr = self._csr.get(rel)
        if csr is None:
            return set()
        indptr, indices = csr
        n = len(indptr) - 1
        if xi >= n:
            return set()
        # Tight PURE-PYTHON walk over array.array. The frontiers here are tiny (mean depth ~6,
        # 1-5 nodes wide), so numpy is a net LOSS: measured 267 us (numpy per-node) and 978 us
        # (numpy vectorized) vs ~30 us for this loop -- numpy's per-call overhead dwarfs the work.
        # array.array indexing yields a python int directly (no scalar boxing), at the same 4 B.
        seen = set()
        seen_add = seen.add                                # bind once (hot loop)
        frontier = [xi]
        for _ in range(max_hops):
            nxt = []
            nxt_append = nxt.append
            for u in frontier:
                a = indptr[u]
                b = indptr[u + 1]
                if a == b:
                    continue
                for v in indices[a:b]:                     # slice-iterate: one C memcpy, then a
                    if v != xi and v not in seen:          # tight iterator (no per-item getitem
                        seen_add(v); nxt_append(v)         # bounds-check / index boxing)
            if not nxt:
                break
            frontier = nxt
        return seen

    def _reach_functional(self, fn, xi, max_hops):
        """Closure over the near-tree form: follow ONE parent per node (a pointer chase, no slicing
        and no inner loop for the ~86% single-parent case); only bit-flagged nodes touch overflow."""
        first, multi, overflow = fn
        n = len(first)
        if xi >= n:
            return set()
        seen = set()
        seen_add = seen.add
        frontier = [xi]
        for _ in range(max_hops):
            nxt = []
            nxt_append = nxt.append
            for u in frontier:
                p = first[u]
                if p < 0:
                    continue                                   # no parent: a root
                if p != xi and p not in seen:
                    seen_add(p); nxt_append(p)
                if overflow is not None and multi[u >> 3] >> (u & 7) & 1:   # rare: >1 parent
                    for v in overflow.targets(u, multi):
                        if v != xi and v not in seen:
                            seen_add(v); nxt_append(v)
            if not nxt:
                break
            frontier = nxt
        return seen

    def ancestors_str(self, rel, x):
        """Closure as surface strings (emission edge only)."""
        return {self.str_of(i) for i in self.transitive_reach(rel, x)} if self.keep_lexicon else set()

    # ---- MEASURED footprint (honest: no asserted constants)
    def edge_bytes(self):
        """Exact bytes of the packed edge arrays (the only thing that scales with knowledge)."""
        _b = self._nb
        n = sum(_b(p) + _b(i) for (p, i) in self._csr.values())
        for (first, multi, overflow) in self._func.values():
            n += _b(first) + _b(multi) + (overflow.nbytes() if overflow is not None else 0)
        return int(n)

    def lexicon_bytes(self):
        """Bytes of the str<->id table. Needed ONLY to emit text -- a reasoning-only store
        (compact_lexicon(keep_strings=False)) carries none of the surface table at all.
        Reports whichever form is live: the python dict, or the compacted arrays."""
        import sys
        if self._id:                                  # uncompacted python form
            b = sys.getsizeof(self._id) + sys.getsizeof(self._str)
            for w in self._id:
                b += sys.getsizeof(w)
            b += 8 * len(self._str)
            return int(b)
        return self.lookup_bytes() + self.surface_bytes()

    _nb = staticmethod(_nbytes)

    def lookup_bytes(self):
        """Bytes needed to REASON (resolve a name -> id). No surface strings."""
        if self._hash is None and self._id:
            return self.lexicon_bytes()
        return int(self._nb(self._hash) + self._nb(self._hid))

    def surface_bytes(self):
        """Bytes of the SURFACE TABLE -- needed only to EMIT text. A reasoning-only store has none."""
        return int(self._nb(self._blk) + self._nb(self._blkoff))

    def bytes_per_edge(self):
        return self.edge_bytes() / max(1, self.n_edges)

    # ---- FLAT ACCESS: the store lives on DISK; RAM holds only the pages a query touches.
    # This is the north-star property (flat-memory-sdm): the SUBSTRATE is fixed, and the fact
    # store grows on DISK, never in RAM. np.memmap gives us that for free -- the OS page cache
    # pulls in only the ~1-2 pages a closure walk actually reads.
    def save(self, path):
        """Write the packed store to one binary file (arrays back-to-back + a json header)."""
        import json
        blobs, meta = [], {'entities': self.n_entities, 'edges': self.n_edges,
                           'csr': {}, 'func': {}, 'lex': {}}
        off = [0]

        def put(kind, name, key, a, dtype):
            b = a.tobytes() if hasattr(a, 'tobytes') else bytes(a)
            blobs.append(b)
            off.append(off[-1] + len(b))
            meta[kind].setdefault(name, {})[key] = [off[-2], len(b), dtype]

        def put_pp(kind, name, pp, pfx):                      # serialize a PackedParents
            put(kind, name, pfx + 'nib', pp.nib, 'u1'); put(kind, name, pfx + 'esc', pp.esc, 'u1')
            put(kind, name, pfx + 'blk', pp.blk, 'i4'); put(kind, name, pfx + 'mask', pp.mask, 'u8')
            put(kind, name, pfx + 'ptab', pp.ptab, 'i4')
            meta[kind][name][pfx + 'n'] = pp.n
            meta[kind][name][pfx + 'ew'] = pp.ew

        for rel, (ip, ix) in self._csr.items():
            put('csr', rel, 'indptr', ip, 'i4'); put('csr', rel, 'indices', ix, 'i4')
        for rel, (first, multi, overflow) in self._func.items():
            put_pp('func', rel, first, '')
            put('func', rel, 'multi', multi, 'u1')
            if overflow is not None:
                put('func', rel, 'ov_bpop', overflow.bpop, 'i4')
                put('func', rel, 'ov_indptr', overflow.indptr, 'i4')
                put_pp('func', rel, overflow.tgt, 'ovt_')
                meta['func'][rel]['ov_n'] = overflow.n
        if self._hash is not None:
            put('lex', 'lex', 'hash', self._hash, 'i8'); put('lex', 'lex', 'hid', self._hid, 'i4')
        if self._blk is not None:
            put('lex', 'lex', 'blk', self._blk, 'u1')
            put('lex', 'lex', 'blkoff', self._blkoff, 'i4')
            meta['lex']['lex']['k'] = self._blkk
        head = json.dumps(meta).encode('utf-8')
        with open(path, 'wb') as f:
            f.write(len(head).to_bytes(8, 'little'))
            f.write(head)
            for b in blobs:
                f.write(b)
        return os.path.getsize(path)

    @classmethod
    def load(cls, path, mmap=True, surface=True):
        """Load a packed store.

        mmap=True  -> every array is a ZERO-COPY memoryview over an np.memmap.  DISK grows, RAM does
                      not; only the pages a query actually touches are ever resident.  This is the
                      flat-access property, and it now holds for the WHOLE store -- Day-96's load()
                      called .tobytes() on the surface blob, which copied 32.4 MB straight back into
                      RAM and quietly defeated the mmap for the single largest array.
                      The views are memoryviews, not raw np.memmaps: element access then yields a
                      python int at 219 ns instead of a boxed numpy scalar at 473 ns (measured).
        surface=False -> load a REASONING-ONLY store: skip the surface table entirely.  Ids in, ids
                      out, zero bytes of vocabulary.  (Day-90: 'surface strings only at the edges'.)
        """
        import json
        st = cls(keep_lexicon=True)
        with open(path, 'rb') as f:
            hlen = int.from_bytes(f.read(8), 'little')
            meta = json.loads(f.read(hlen).decode('utf-8'))
        base = 8 + hlen
        _FMT = {'i4': 'i', 'i8': 'q', 'u1': 'B', 'u8': 'Q'}

        def get(spec):
            o, ln, dt = spec
            n = ln if dt == 'u1' else ln // np.dtype(dt).itemsize
            npdt = np.uint8 if dt == 'u1' else dt
            arr = np.memmap(path, dtype=npdt, mode='r', offset=base + o, shape=(n,))
            return _mv(arr, _FMT[dt]) if mmap else _mv(np.array(arr), _FMT[dt])

        def get_pp(d, pfx):                                # reconstruct a PackedParents
            return PackedParents(get(d[pfx + 'nib']), get(d[pfx + 'esc']), get(d[pfx + 'blk']),
                                 get(d[pfx + 'mask']), get(d[pfx + 'ptab']), d[pfx + 'n'],
                                 d.get(pfx + 'ew', 18))     # ew defaults to 18 for legacy stores

        st.n_entities = meta['entities']; st.n_edges = meta['edges']
        for rel, d in meta['csr'].items():
            st._csr[rel] = (get(d['indptr']), get(d['indices']))
        for rel, d in meta['func'].items():
            if 'nib' in d:                                # Day-97 dictionary-coded parent array
                pp = get_pp(d, '')
            else:                                         # legacy (Day-96 raw int32 parent array)
                pp = PackedParents.build(np.asarray(get(d['first']), dtype=np.int32))
            if 'ov_bpop' in d:                            # Day-97 popcount-ranked overflow
                overflow = OverflowParents(get(d['ov_bpop']), get(d['ov_indptr']),
                                           get_pp(d, 'ovt_'), d['ov_n'])
            elif 'ov_subj' in d:                          # legacy (Day-96 CSR overflow)
                ov_subj, ov_indptr, ov_tgt = get(d['ov_subj']), get(d['ov_indptr']), get(d['ov_tgt'])
                flags = np.zeros(st.n_entities, bool)
                if len(ov_subj):
                    flags[np.asarray(ov_subj, dtype=np.int64)] = True
                overflow = (OverflowParents.build(flags, np.repeat(
                    np.asarray(ov_subj, np.int64),
                    np.diff(np.asarray(ov_indptr, np.int64))),
                    np.asarray(ov_tgt, np.int32), st.n_entities) if len(ov_subj) else None)
            else:
                overflow = None
            st._func[rel] = (pp, get(d['multi']), overflow)
        lex = meta['lex'].get('lex', {})
        if 'hash' in lex:
            st._hash = get(lex['hash']); st._hid = get(lex['hid'])
        if surface and 'blk' in lex:
            st._blk = get(lex['blk']); st._blkoff = get(lex['blkoff'])
            st._blkk = int(lex.get('k', _BLOCK))
        elif surface and 'blob' in lex:                     # legacy (Day-96 raw blob + int64 off)
            blob, off = get(lex['blob']), get(lex['off'])
            st._blk, st._blkoff = cls._pack_surface(
                [bytes(blob[off[i]:off[i + 1]]).decode('utf-8') for i in range(st.n_entities)],
                _BLOCK)
        st.keep_lexicon = st._blk is not None
        st._id = {}; st._str = []
        return st
