"""ikigai.cognition.holo_generate -- Day 87: holographic SEQUENCE MEMORY.

The generation counterpart to holo_read.  Measured on the real substrate: a single
holographic transition operator is short-range (~8-32 transitions at d=400/512 before
crosstalk wins -- the research's ~40-48 claim was ~5x optimistic; see
Research/fluent_generation/FINDINGS_day87.md).  Two mechanisms live here:

  CODEC (lossless, repeats OK, used by encode/decode) -- BOUNDED POSITIONAL CHUNKS.
  Split the sequence into chunks of <= `chunk` tokens; store each chunk as ONE phasor
  vector  bundle_t key(tok_t) (x) pos(t)  where pos(t) is a distinct positional key.
  Regenerate by unbinding each position and cleaning up over the sequence's codebook.
  Position-indexing (not predecessor-indexing) means REPEATED tokens are separable, so
  natural text with repeats round-trips exactly; bounded chunk size keeps crosstalk low,
  so length is uncapped.

  GENERATIVE ROLL (short-range, _operator/_roll) -- predecessor->successor transition
  operator for deriving a next symbol.  Retained for the generation path; a single
  operator branches poorly, so it is NOT the codec.

HONEST SCOPE: this is a LOSSLESS sequence codec / substrate representation -- it
encodes a token stream into bounded phasor chunks and regenerates it exactly, making a
sequence a first-class substrate object.  It is the LENGTH primitive future generation
builds on.  It is NOT novel generation: inventing an unseen sequence (branching at a
decision point) is a separate, open axis.  A body-part: the organism's holographic
sequence memory.
"""
import numpy as np

from ikigai.cognition.flat_memory import ComputedKey
from ikigai.cognition.phasor_state import bind, unbind, superpose, normalize_phase


class HolographicSequenceMemory:
    """Encode a token sequence into a recursive hierarchy of bounded holographic
    transition operators; regenerate it losslessly.  Own key space (d=512)."""

    def __init__(self, d=512, chunk=8, seed=114, ck=None):
        # `ck` = the organism's SHARED key space (org.unified.ck).  When given, the
        # codec uses the SAME token identities as reasoning, so an encoded sequence is
        # a first-class substrate object (composable/recallable), not a private island.
        # Positional binding keeps the codec robust to the shared keys' morphological
        # correlation.  Falls back to a private space if none.
        self.ck = ck if ck is not None else ComputedKey(d=d, seed=seed)
        self.d = int(self.ck.key('__probe__').shape[0])
        self.chunk = int(chunk)
        self._keys = {}

    def key(self, tok):
        k = self._keys.get(tok)
        if k is None:
            k = self.ck.key(tok)
            self._keys[tok] = k
        return k

    # ---- core VSA ops ------------------------------------------------
    def _pos(self, t):
        """A distinct phasor key per position -- makes repeated tokens separable
        (branching is not a problem for a codec: we index by position, not
        predecessor)."""
        return self.key(f'__POS_{t}')

    def _chunk_bundle(self, chunk):
        """A chunk stored as one phasor vector: bundle_t  key(tok_t) (x) pos(t)."""
        return normalize_phase(superpose([bind(self.key(chunk[t]), self._pos(t))
                                          for t in range(len(chunk))]))

    def _cleanup(self, vec, codebook):
        M = np.stack([self.key(c) for c in codebook])
        return codebook[int(np.argmax(np.abs(M.conj() @ vec)))]

    def _operator(self, seq):
        """GENERATIVE transition operator (predecessor->successor), for the
        derive-a-next-symbol path.  Short-range; the codec above does not use it."""
        if len(seq) < 2:
            return None
        return normalize_phase(superpose([unbind(self.key(seq[t + 1]), self.key(seq[t]))
                                          for t in range(len(seq) - 1)]))

    def _roll(self, T, start, steps, codebook):
        out = [start]
        cur = start
        for _ in range(steps):
            cur = self._cleanup(bind(T, self.key(cur)), codebook)
            out.append(cur)
        return out

    # ---- public codec (positional chunks -- lossless, repeats OK) ----
    def encode(self, tokens):
        """Encode a token sequence into a handle: an ordered list of bounded
        positional chunk-bundles (each <= `chunk` tokens, so crosstalk never
        dominates) + the sequence's token codebook.  Arbitrary length, repeats OK.
        Pure phasor build, read-only on the world."""
        tokens = [str(t) for t in tokens]
        chunks = [tokens[i:i + self.chunk] for i in range(0, len(tokens), self.chunk)]
        return {'codebook': sorted(set(tokens)),
                'chunks': [(self._chunk_bundle(c), len(c)) for c in chunks]}

    def decode(self, handle):
        """Regenerate the exact token sequence: unbind each position out of each
        chunk-bundle and clean up over the codebook."""
        cb = handle['codebook']
        out = []
        for bundle, ln in handle['chunks']:
            for t in range(ln):
                out.append(self._cleanup(unbind(bundle, self._pos(t)), cb))
        return out

    def roundtrip(self, tokens):
        """encode then decode -- returns the regenerated sequence."""
        return self.decode(self.encode(list(tokens)))

    def fidelity(self, tokens):
        """Fraction of positions regenerated correctly (1.0 = lossless)."""
        toks = [str(t) for t in tokens]
        got = self.roundtrip(toks)
        n = len(toks)
        return sum(1 for a, b in zip(got, toks) if a == b) / max(1, n)


class SurfaceRealizer:
    """Day-87 -- the thin, content-BLIND surface smoother (File 5's only sanctioned
    learned component).  It turns a grounded fact `(s, r, o)` from the substrate's own
    surface ('wug isa florp') into fluent text ('a wug is a florp') -- WITHOUT any
    hardcoded per-relation template.  The frame is INDUCED from example (fact, sentence)
    pairs by anti-unification: whatever token equals the subject becomes the {S} slot,
    whatever equals the object becomes {O}, and the tokens CONSTANT across examples are
    the relation's surface frame (function words -- 'a', 'is', 'has').  Realisation
    fills the induced frame with the fact's own s/o.

    THE BOUNDARY (the cannot-lie law survives): the smoother may only add/reorder FRAME
    (function) tokens; the CONTENT tokens are always the fact's s and o, never invented.
    So a realised sentence is still grounded on content -- it cannot state a fact the
    substrate did not derive.  No backprop: pure alignment + majority vote."""

    def __init__(self):
        self.templates = {}      # relation -> frame list with '{S}'/'{O}' slots

    def learn(self, pairs):
        """pairs: list of ((s, r, o), sentence).  Induce a frame per relation by
        aligning the sentence tokens to the fact and keeping the constant remainder."""
        from collections import Counter, defaultdict
        frames = defaultdict(list)
        for (s, r, o), sent in pairs:
            s, r, o = str(s).lower(), str(r).lower(), str(o).lower()
            toks = str(sent).lower().split()
            frame = tuple('{S}' if t == s else ('{O}' if t == o else t) for t in toks)
            # only accept a frame that actually contains both slots (a real alignment)
            if '{S}' in frame and '{O}' in frame:
                frames[r].append(frame)
        for r, fs in frames.items():
            self.templates[r] = list(Counter(fs).most_common(1)[0][0])   # majority frame
        return len(self.templates)

    def realize(self, s, r, o):
        """Render a fact as fluent text using the induced frame; fall back to the
        substrate's own surface ('s r o') when the relation has no learned frame."""
        s, r, o = str(s).lower(), str(r).lower(), str(o).lower()
        frame = self.templates.get(r)
        if not frame:
            return f"{s} {r} {o}"
        return ' '.join(s if t == '{S}' else (o if t == '{O}' else t) for t in frame)

    @staticmethod
    def content_tokens(s, r, o):
        """The tokens that MUST be grounded (the frame is content-blind)."""
        return {str(s).lower(), str(o).lower()}


class HolographicBranchGenerator:
    """Day-87 -- NOVELTY / BRANCHING: the axis a single transition operator fails.
    Generate an UNSEEN sequence by (1) recovering the SET of valid next tokens at a
    branch and (2) SELECTING one -- so the whole path is novel but every step is a
    real, observed transition (grounded, not hallucinated).

    Each context (the last `order` tokens) owns a bundle = superposition of the
    successor keys ever seen after it.  At a branch the bundle holds several
    successors; MATCHING PURSUIT (cleanup nearest, subtract, repeat) recovers the
    discrete set from the superposition -- a set, by resonance, not a python list.
    Selection (goal / grammar / sampling) then produces a novel, valid continuation.

    Own RANDOM quasi-orthogonal key space -- abstract tokens must not share the
    morphological correlations ck.key gives similar spellings (that correlation
    destroys the membership/extraction; measured on Day 87)."""

    def __init__(self, d=512, order=1, seed=114, ck=None):
        # `ck` = the organism's SHARED key space.  When given, the generator draws on
        # the SAME token identities as reasoning (it can harvest the organism's own
        # vocabulary and generate over it).  Per-context bundles are small (1-3
        # successors), so the shared keys' morphological correlation does not degrade
        # branch-set extraction (measured: 100%/100% either way).  Falls back to a
        # private quasi-orthogonal space if none.
        self.ck = ck
        self.d = int(ck.key('__probe__').shape[0]) if ck is not None else d
        self.order = int(order)
        self._keys = {}
        self.vocab = []
        self._bundles = {}          # context-token-tuple -> superposed successor bundle
        # Day 90 -- address-space speedups (output-preserving): a codebook matrix built
        # ONCE (not rebuilt per call) and a per-context successor cache (Kanerva bypass),
        # so a warm context resolves in O(1) and cost stops growing with the lexicon.
        self._VM = None             # cached codebook matrix (len(vocab) x d)
        self._VM_n = -1             # vocab length _VM was built at (rebuild when it grows)
        self._succ_cache = {}       # context-tuple -> tuple(successor tokens)

    def key(self, tok):
        k = self._keys.get(tok)
        if k is None:
            if self.ck is not None:
                k = self.ck.key(tok)
            else:
                r = np.random.RandomState((abs(hash(('bg', tok))) % (2 ** 31)))
                k = np.exp(1j * r.uniform(-np.pi, np.pi, self.d)).astype(np.complex64)
            self._keys[tok] = k
            self.vocab.append(tok)
        return k

    def _ctx(self, tokens_prefix):
        return tuple(tokens_prefix[-self.order:])

    def observe(self, sequences):
        """Learn transitions from example sequences (no backprop): add each
        successor's key into its context's bundle (Hebbian superposition)."""
        for seq in sequences:
            seq = [str(t) for t in seq]
            for t in range(len(seq) - 1):
                self.key(seq[t + 1])                    # register vocab
                c = self._ctx(seq[:t + 1])
                add = self.key(seq[t + 1]).astype(np.complex128)
                self._bundles[c] = self._bundles.get(
                    c, np.zeros(self.d, dtype=np.complex128)) + add
                self._succ_cache.pop(c, None)           # a changed bundle invalidates its cache
        return len(self._bundles)

    def _codebook(self):
        """Codebook matrix, built once and reused; rebuilt only when the vocab grows.
        Day-90 speedup -- the old path re-stacked every vocab key on EVERY call."""
        if self._VM is None or self._VM_n != len(self.vocab):
            self._VM = np.stack([self.key(v) for v in self.vocab]) if self.vocab \
                else np.zeros((0, self.d), dtype=np.complex64)
            self._VM_n = len(self.vocab)
        return self._VM

    def successors(self, context_tokens, thresh=0.30, max_k=6):
        """Recover the valid-next set for a context by matching pursuit over its
        bundle.  Returns [] for an unseen context (honest: no fabricated options).
        Day-90: identical result, but a warm context returns from cache in O(1) and the
        codebook is reused, so cost no longer grows with the lexicon (address-space win)."""
        c = self._ctx(list(context_tokens))
        hit = self._succ_cache.get(c)
        if hit is not None:
            return list(hit)                            # cached (byte-identical to a recompute)
        b = self._bundles.get(c)
        if b is None:
            return []
        found = self._cleanup_bundle(b, thresh, max_k)
        self._succ_cache[c] = tuple(found)
        return found

    def _cleanup_bundle(self, b, thresh=0.30, max_k=6):
        """Matching pursuit: recover the discrete token set superposed in a bundle vector.
        Shared by the transient per-context store (successors) and, after consolidation,
        by a bundle read back from the durable SDM -- both decode the same way."""
        if b is None:
            return []
        resid = b.copy()
        VM = self._codebook()
        found = []
        for _ in range(max_k):
            sims = np.abs(VM.conj() @ resid) / self.d
            i = int(np.argmax(sims))
            if sims[i] < thresh:
                break
            tok = self.vocab[i]
            if tok in found:
                break
            found.append(tok)
            k = self.key(tok)
            resid = resid - ((k.conj() @ resid) / (k.conj() @ k)) * k
        return found

    def generate(self, start, steps, select=None, seed=0):
        """Generate a sequence by rolling: at each step extract the valid successor
        set and SELECT one.  `select(options, context)` defaults to random choice
        (diversity); pass a goal/grammar selector for controlled generation.
        Stops when a context has no known successor (honest halt)."""
        import random as _r
        rng = _r.Random(seed)
        select = select or (lambda opts, ctx: rng.choice(opts))
        path = [str(x) for x in (start if isinstance(start, (list, tuple)) else [start])]
        for _ in range(steps):
            opts = self.successors(path)
            if not opts:
                break
            path.append(select(opts, tuple(path)))
        return path


class AddressGenerator:
    """Day 90 -- the FLUENT-GENERATION FRAMEWORK, in ADDRESS space.

    The idea (Prince): a natural language has a finite lexicon.  Store every word as
    an HV, and store the RELATIONS between words as HV too -- but reading and generating
    do NOT carry the words.  They carry the words' ADDRESSES.  Surface strings are
    materialised only at the two edges: surface->address on the way in (read), and
    address->surface on the way out (emit).  Everything between is address arithmetic.

    This is the no-backprop, near-zero-compute analog of a transformer's next-token head.
    A transformer predicts the next token by an attention pass over learned weights; here
    the next ADDRESS is recovered by RESONANCE over a learned transition memory -- one HV
    per context holding the superposed successor addresses -- and the surface word is
    decoded only to emit it.  It is a FRAMEWORK: it is proved data-free on a made-up
    lexicon and grammar; the fluency it produces scales with the transition data fed to
    observe().  Given a real corpus, the same code generates from real word statistics.

    Design (all substrate ops, capacity-honest):
      * CODEBOOK   -- word <-> code (a small integer address) <-> key (its HV).  The
                      lexicon is a fixed d x V matrix of addresses; V is unbounded and
                      costs ~nothing (keys are computed, not stored).
      * TRANSITIONS-- relations stored IN HV: per context (indexed by its ADDRESS, a code
                      tuple) a bundle = superposition of the successor addresses seen after
                      it.  Per-context bundles hold 1-3 successors, so they stay well under
                      the sqrt(d) crosstalk wall -- a single global transition HV would not.
      * NEXT       -- address -> addresses: matching pursuit over the context's bundle
                      recovers the discrete successor code set by resonance (a set, not a
                      python list of words).  A Kanerva-style cache returns a seen context's
                      successor codes in O(1), so per-step cost is INDEPENDENT of lexicon
                      size once contexts are warm -- the property that makes a 500k-word
                      lexicon as cheap per step as a 2k-word one.
      * WALK       -- generation rolls entirely in code space; decode() runs only at emit.
                      Constraint critics (coherence / theme / grammar) score CANDIDATE
                      ADDRESSES by resonance, so the Day-88/89 composed critics apply here
                      unchanged, still in address space.
    """

    def __init__(self, ck, order=1):
        self.ck = ck
        self.d = int(ck.key('__probe__').shape[0])
        self.order = int(order)
        self._code = {}          # surface -> code (address)
        self._word = []          # code -> surface (decode side of the codebook)
        self._keys = []          # code -> key(HV) (the address book, list -> matrix)
        self._K = None           # cached dense codebook matrix (V x d)
        self._trans = {}         # context code-tuple -> successor bundle (relations in HV)
        self._succ_cache = {}    # context code-tuple -> tuple(successor codes) (Kanerva)

    # ---- the two EDGES: surface <-> address --------------------------------------
    def encode(self, word):
        """READ edge: surface word -> its address (code).  Registers new vocabulary."""
        w = str(word)
        c = self._code.get(w)
        if c is None:
            c = len(self._word)
            self._code[w] = c
            self._word.append(w)
            self._keys.append(self.ck.key(w).astype(np.complex64))
            self._K = None
        return c

    def decode(self, code):
        """EMIT edge: address (code) -> surface word.  The ONLY surface materialisation."""
        return self._word[int(code)]

    def render(self, codes):
        return [self._word[int(c)] for c in codes]

    @property
    def vocab_size(self):
        return len(self._word)

    def _codebook(self):
        if self._K is None:
            self._K = np.stack(self._keys) if self._keys else np.zeros((0, self.d), np.complex64)
        return self._K

    # ---- learn transitions (relations in HV), addressed by context code ----------
    def observe(self, sequences):
        """Learn word->word relations with no backprop: for each observed transition add
        the successor's ADDRESS into its context's bundle (Hebbian superposition).  The
        context is keyed by its code tuple -- an address, not a surface string."""
        n = 0
        for seq in sequences:
            codes = [self.encode(t) for t in seq]
            for t in range(len(codes) - 1):
                c = tuple(codes[max(0, t - self.order + 1):t + 1])
                add = self._keys[codes[t + 1]].astype(np.complex128)
                self._trans[c] = self._trans.get(
                    c, np.zeros(self.d, dtype=np.complex128)) + add
                self._succ_cache.pop(c, None)
                n += 1
        return n

    # ---- address -> addresses: the next-token head, by resonance -----------------
    def next_codes(self, ctx_codes, thresh=0.30, max_k=6):
        """Recover the valid next ADDRESSES for a context.  O(1) for a warm context
        (Kanerva cache); otherwise matching pursuit over the codebook.  Returns () for
        an unseen context (honest halt, no fabrication)."""
        c = tuple(int(x) for x in ctx_codes[-self.order:])
        hit = self._succ_cache.get(c)
        if hit is not None:
            return hit
        b = self._trans.get(c)
        if b is None:
            return ()
        VM = self._codebook()
        resid = b.copy()
        found = []
        for _ in range(max_k):
            sims = np.abs(VM.conj() @ resid) / self.d
            i = int(np.argmax(sims))
            if sims[i] < thresh or i in found:
                break
            found.append(i)
            k = VM[i]
            resid = resid - ((k.conj() @ resid) / (k.conj() @ k)) * k
        found = tuple(found)
        self._succ_cache[c] = found
        return found

    # ---- constraint critic in ADDRESS space (composes Day-88/89 critics) ---------
    def theme_bundle(self, words):
        """A coherence constraint as an ADDRESS bundle: superpose the theme words'
        keys; score a candidate address by cosine resonance to it (axis-independent)."""
        b = np.zeros(self.d, dtype=np.complex64)
        for w in words:
            b = b + self._keys[self.encode(w)]
        nb = float(np.linalg.norm(b))
        def _crit(code, _b=b, _nb=nb):
            k = self._keys[int(code)]
            n = _nb * float(np.linalg.norm(k))
            return float(np.real(np.vdot(k, _b)) / n) if n else 0.0
        return _crit

    # ---- WALK: generation entirely in code space ---------------------------------
    def generate(self, start, steps=40, critic=None, seed=0):
        """Roll a sequence in ADDRESS space and return the CODES.  At each step the
        successor addresses are recovered by resonance and one is SELECTED -- by a
        constraint critic (max resonance) if given, else at random for diversity.
        Decoding to surface happens only when the caller calls render()/decode()."""
        import random as _r
        rng = _r.Random(seed)
        if isinstance(start, str):
            path = [self.encode(start)]
        elif isinstance(start, (list, tuple)):
            path = [self.encode(x) if isinstance(x, str) else int(x) for x in start]
        else:
            path = [int(start)]
        for _ in range(int(steps)):
            opts = self.next_codes(path)
            if not opts:
                break                              # honest halt
            if critic is None:
                nxt = rng.choice(opts)
            else:
                nxt = max(opts, key=lambda code: critic(code))
            path.append(int(nxt))
        return path
