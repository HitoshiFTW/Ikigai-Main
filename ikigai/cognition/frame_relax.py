"""
ikigai.cognition.frame_relax -- Pack 313 (Day 80).

Mechanism #4 from the FHRR-generation research, WIRED to the real substrate:
HOLOGRAPHIC GRAMMAR FRAME + BIDIRECTIONAL PARALLEL RELAXATION.

The "beat next-token prediction" generator. Unlike vs_fsm.generate (greedy
left-to-right walk with a no_repeat_window band-aid), this does NOT predict the
next token. It:

  1. fixes a MESSAGE (content words to convey),
  2. lays a syntactic FRAME = a category sequence sampled from a 2nd-order
     category FSM (1st-order MIXES frames -> invalid; 2nd-order keeps each
     frame's path distinct), then
  3. FILLS every slot at once via PARALLEL RELAXATION -- each slot re-scored
     from BOTH neighbors' current estimates (forward + backward bigram
     evidence) under its category mask, iterated to a fixpoint.

Function words occupy STRUCTURAL slots (from the frame), not frequency-
predicted positions, so they cannot form an attractor. This is Levelt
message-first / frame-then-fill in substrate form.

Energy-based reading (see Proto, Merchant/Hie 2026): the per-slot score is a
sum of constraint energies (category mask + forward + backward n-gram), the
relaxation is coordinate-descent on that composed energy. Same primitive as
the Pack 311 product-of-experts.

SUBSTRATE-NATIVE backward evidence: the production `next` bank is directional
(addr=context -> value=next), so there is no backward bank. We approximate
P(right | candidate) by reverse-querying the SAME forward bank: for each
candidate c, score the right neighbor under next_word_candidates(c). No new
bank, no state mutation.

Categories in production come from the substrate itself -- a FrameField's
learned frame clusters (frame_of_word / frame_bigram / frame_vocab) or isa
parents. `from_frame_field` builds the generator from a populated FrameField.
The category structures are inputs; this module authors no lexicon.
"""

import numpy as np


class FrameRelaxGenerator:
    """Frame-then-fill generator with bidirectional parallel relaxation over a
    MultiRoleMemory.

    category_of : dict token -> category label
    fsm2        : dict (cat_a, cat_b) -> {cat_c: count}   2nd-order category FSM
                  (use 'START' padding at the front, 'END' as terminal target)
    cat_vocab   : dict category -> iterable of candidate tokens
    """

    def __init__(self, mr, category_of, fsm2, cat_vocab, pmi=True,
                 templates=None):
        self.mr = mr
        self.cat_of = dict(category_of)
        self.fsm2 = {k: dict(v) for k, v in fsm2.items()}
        self.cat_vocab = {c: list(v) for c, v in cat_vocab.items()}
        self.pmi = bool(pmi)
        self._eps = 1.0 / np.sqrt(2.0 * mr.d)
        # Pack 316: whole-frame TEMPLATE bank. list of (frame_tuple, weight).
        # When present, sample_frame retrieves a COMPLETE observed category
        # sequence atomically -> bypasses the finite-order Markov frame-mixing
        # wall (a Markov walk can't tell a start-state from an end-state; a
        # whole template can never stitch start to end). This is the Levelt
        # "lay the whole frame, then fill" move.
        self.templates = None
        if templates:
            self.templates = [(tuple(f), float(w)) for f, w in templates]

    # ── frame sampling (the grammar) ──────────────────────────────────────
    def sample_frame(self, rng, max_len=12):
        """Return a syntactic frame (category sequence).

        Pack 316: if a TEMPLATE bank is present, retrieve a WHOLE observed
        frame atomically (no Markov stitching -> no frame-mixing). Otherwise
        fall back to walking the 2nd-order category FSM.
        """
        if self.templates:
            frames, wts = zip(*self.templates)
            wts = np.array(wts, dtype=np.float64)
            wts = wts / wts.sum()
            return list(frames[int(rng.choice(len(frames), p=wts))])
        prev, cur, out = 'START', 'START', []
        for _ in range(max_len):
            nxts = self.fsm2.get((prev, cur))
            if not nxts:
                break
            cats = list(nxts.keys())
            wts = np.array([nxts[c] for c in cats], dtype=np.float64)
            wts = wts / wts.sum()
            nxt = cats[int(rng.choice(len(cats), p=wts))]
            if nxt == 'END':
                break
            out.append(nxt)
            prev, cur = cur, nxt
        return out

    # ── substrate evidence ────────────────────────────────────────────────
    def _prior(self, cand_list):
        ucount = getattr(self.mr, '_unigram_count', None) or {}
        N = max(getattr(self.mr, '_unigram_total', 0) or 0, 1)
        return np.array([ucount.get(c, 0) / N for c in cand_list],
                        dtype=np.float64)

    def _fwd(self, prev_tok, cand_list, prior=None):
        """P(cand | prev_tok) over cand_list via the real forward next bank."""
        ranked = self.mr.next_word_candidates(prev_tok, candidates=cand_list,
                                              top_k=len(cand_list))
        sim = {w: max(s, 0.0) for w, s in ranked}
        p = np.array([sim.get(c, 0.0) for c in cand_list], dtype=np.float64)
        if self.pmi and prior is not None:
            p = p / (prior + self._eps)
        s = p.sum()
        return p / s if s > 1e-12 else p

    def _bwd(self, cand_list, right_tok):
        """P(right_tok | cand) for each cand -- reverse lookup through the SAME
        forward bank (production has no backward bank)."""
        out = np.zeros(len(cand_list), dtype=np.float64)
        for i, c in enumerate(cand_list):
            rr = self.mr.next_word_candidates(c, candidates=[right_tok], top_k=1)
            if rr:
                out[i] = max(rr[0][1], 0.0)
        s = out.sum()
        return out / s if s > 1e-12 else out

    # ── generation ────────────────────────────────────────────────────────
    def generate(self, message=None, frame=None, n_iters=6, seed=0,
                 no_repeat=True):
        """Returns the realized token list. message: dict category -> word
        (pinned content). frame: explicit category list, else sampled.

        no_repeat: forbid a slot taking the same token as an adjacent slot
        (universal anti-loop, not a lexicon). Kills the `the the`/`number
        number` within-frame repetition that arises when adjacent slots share
        a category and both pick the category's argmax word.
        """
        rng = np.random.default_rng(seed)
        frame = frame if frame is not None else self.sample_frame(rng)
        if not frame:
            return []
        message = dict(message or {})
        L = len(frame)
        # init: message word per slot if given, else category's first candidate
        slots = []
        for c in frame:
            if c in message and message[c] in self.cat_of:
                slots.append(message[c])
            else:
                v = self.cat_vocab.get(c) or [None]
                slots.append(v[0])
        # parallel relaxation: re-fill every non-pinned slot from BOTH neighbors
        for _ in range(int(n_iters)):
            new = list(slots)
            for i, c in enumerate(frame):
                if c in message and message[c] in self.cat_of:
                    continue
                cands = self.cat_vocab.get(c)
                if not cands:
                    continue
                prior = self._prior(cands)
                score = np.zeros(len(cands), dtype=np.float64)
                if i > 0 and slots[i-1] is not None:
                    score += self._fwd(slots[i-1], cands, prior)
                if i < L-1 and slots[i+1] is not None:
                    score += self._bwd(cands, slots[i+1])
                if no_repeat:
                    banned = {slots[i-1] if i > 0 else None,
                              slots[i+1] if i < L-1 else None}
                    order = np.argsort(-score)
                    pick = cands[int(order[0])]
                    for j in order:
                        if cands[int(j)] not in banned:
                            pick = cands[int(j)]
                            break
                    new[i] = pick
                else:
                    new[i] = cands[int(np.argmax(score))]
            if new == slots:
                break
            slots = new
        return [t for t in slots if t is not None]

    # ── build from a populated FrameField (substrate-native categories) ───
    @classmethod
    def from_frame_field(cls, mr, frame_field, pool=None, pmi=True):
        """Construct categories from a FrameField's learned clusters.
        category = frame_of_word; cat_vocab = frame_vocab per cluster (limited
        to `pool` if given); fsm2 = 2nd-order over categories derived from the
        frame_bigram-ordered vocab is not available, so we fall back to a
        permissive FSM that allows any learned category transition. Returns
        None if the field has no usable structure."""
        ff = frame_field
        if ff is None or not getattr(ff, 'word_to_frame', None):
            return None
        if pool is not None:
            pool = set(pool)
        category_of = {w: int(k) for w, k in ff.word_to_frame.items()
                       if pool is None or w in pool}
        if not category_of:
            return None
        cat_vocab = {}
        for w, k in category_of.items():
            cat_vocab.setdefault(k, []).append(w)
        # 2nd-order FSM from frame_bigram (1st-order K x K) lifted to a
        # permissive 2nd-order: (any, a) -> b allowed with frame_bigram weight.
        fsm2 = {}
        K = ff.K
        fb = ff.frame_bigram
        starts = {}
        for a in range(K):
            row = fb[a].astype(np.float64)
            tot = row.sum()
            if tot <= 0:
                continue
            for b in range(K):
                if row[b] <= 0:
                    continue
                # allow (START,a)->b and (x,a)->b uniformly via the bigram
                fsm2.setdefault(('START', a), {})[b] = float(row[b])
                for x in range(K):
                    fsm2.setdefault((x, a), {})[b] = float(row[b])
            starts[a] = tot
        # START,START -> a by overall frequency
        for a, tot in starts.items():
            fsm2.setdefault(('START', 'START'), {})[a] = float(tot)
        if not fsm2:
            return None
        return cls(mr, category_of, fsm2, cat_vocab, pmi=pmi)
