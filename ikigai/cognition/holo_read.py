"""ikigai.cognition.holo_read -- Pack 333: TEMPLATE-FREE holographic reading.

Prince's vision: give the organism ANY sentence, it learns it; ask it back in
plain English, the answer falls out -- no templates, no relation lists, no
grammar rules, no stopword/wh lists, no stemmer.  Everything is the substrate.

Mechanism (pure FHRR + Kanerva SDM, nothing else):
  read(sentence):
      for EACH position in the sentence, build an ORDER-FREE context = the
      product (bind) of every OTHER token's key, and write  context (x) key(token)
      into a Sparse Distributed Memory.  Every token is made recoverable from the
      others; the "relation" is never named, it is only the other words.
  ask(question):  (plain-English question, no hole to mark)
      a question carries CUES -- the words it shares with the fact.  Each query
      token is mapped to a vocab token: exact, else nearest by char-trigram
      cosine (morphology, 'report'->'reports' for free), else dropped (query-only
      words like 'who'/'does' map to nothing and remove themselves -- no wh-list).
      The cues' context is read from the SDM and the answer is the stored token
      it resonates to that ISN'T already a cue.
  answer(sentence_with_a_hole):  explicit fill-in-the-blank variant.

Two key spaces: binding uses clean whole-word keys (entities stay separate);
the morphology MATCH uses a word_weight=0 trigram-only space so inflections
collapse.  The SDM distributes facts across hard locations, so crosstalk grows
sub-linearly -> thousands of facts per bank (a flat bundle holds ~25).
Honest-unknown: a context that resonates to nothing returns None (the LLM
differentiator -- it abstains instead of hallucinating).  A body-part: a
dedicated reading memory, holographic and native.
"""
import collections
import math
import re

import numpy as np

from ikigai.cognition.flat_memory import ComputedKey
from ikigai.cognition.phasor_state import bind, unbind

_TOK = re.compile(r"[a-z0-9'][a-z0-9']*")
_SENT_SPLIT = re.compile(r"[.!?\n]+")


def _tokens(text):
    return _TOK.findall(str(text).lower())


class HolographicReader:
    """Template-free holographic sentence memory: read anything, ask anything."""

    def __init__(self, d=512, M=8000, k=32, seed=114, k_sigma=5.0,
                 morph_thresh=0.45, ck=None):
        self.d, self.M, self.k = int(d), int(M), int(k)
        self.k_sigma = float(k_sigma)
        self.morph_thresh = float(morph_thresh)
        self.ck = ck if ck is not None else ComputedKey(d=self.d, seed=seed)
        # trigram-only space for morphology matching (word_weight=0 keeps the
        # char-trigram signal that the clean binding keys suppress).
        self.morph = ComputedKey(d=self.d, seed=seed, word_weight=0.0)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(-np.pi, np.pi, (self.M, self.d)).astype(np.float32)
        self.addr = np.exp(1j * ph).astype(np.complex64)          # hard locations
        self.acc = np.zeros((self.M, self.d), dtype=np.complex64)  # accumulators
        self.vocab = []                                            # ordered
        self._vset = set()
        self._vkeys = None
        self._vmorph = None
        self.n_sentences = 0
        self.n_writes = 0
        # for emergent atom extraction: document-frequency per token + the
        # raw token streams read (the episodic log). Relations are LEARNED by
        # recurrence -- a token that recurs across facts is relational; a novel
        # token is an argument. No relation list, no grammar.
        self.df = collections.Counter()
        self.read_log = []
        # word-order structure: a token that EVER leads a fact is a subject, one
        # that EVER ends a fact is an object -> it is an ARGUMENT, not a relation,
        # no matter how often it recurs. Relations live in the interior.
        self.first_tok = set()
        self.last_tok = set()
        # RL: reward-driven per-token relation bias (dopamine). Starts empty ->
        # pure structural prior; reinforced when a parse reaches the gold answer.
        self.rel_bias = collections.defaultdict(float)

    # ---- internals ------------------------------------------------------

    def _ctx(self, tokens):
        """Order-free context = product of token keys (binding is commutative,
        so word order does not matter -- a question may reorder the words)."""
        c = np.ones(self.d, dtype=np.complex64)
        for w in tokens:
            c = bind(c, self.ck.key(w))
        return c

    def _activate(self, ctx):
        sims = np.abs(self.addr @ np.conj(ctx)) / self.d
        kk = min(self.k, self.M)
        return np.argpartition(-sims, kk - 1)[:kk]

    def _add_vocab(self, w):
        if w not in self._vset:
            self._vset.add(w); self.vocab.append(w)
            self._vkeys = None; self._vmorph = None

    def _vkey_matrix(self):
        if self._vkeys is None:
            self._vkeys = np.stack([self.ck.key(w) for w in self.vocab])
        return self._vkeys

    def _vmorph_matrix(self):
        if self._vmorph is None:
            self._vmorph = np.stack([self.morph.key(w) for w in self.vocab])
        return self._vmorph

    def _match(self, token):
        """Query token -> vocab token: exact, else nearest by trigram cosine,
        else None (query-only words drop themselves -- no wh/stopword list)."""
        if token in self._vset:
            return token
        if not self.vocab:
            return None
        sims = np.abs(self._vmorph_matrix() @ np.conj(self.morph.key(token))) / self.d
        i = int(np.argmax(sims))
        return self.vocab[i] if sims[i] >= self.morph_thresh else None

    @property
    def boundary(self):
        return self.k_sigma / math.sqrt(2 * self.d)

    def _resolve(self, cues, exclude, top_k):
        """Read the SDM at the cues' context, recover + clean up the answer over
        the vocabulary (excluding the cues themselves). Honest-unknown below the
        noise floor."""
        if not cues:
            return (None, 0.0) if top_k == 1 else []
        ctx = self._ctx(cues)
        cand = unbind(self.acc[self._activate(ctx)].sum(axis=0), ctx)
        K = self._vkey_matrix()
        cn = np.linalg.norm(cand) + 1e-12
        kn = np.linalg.norm(K, axis=1) + 1e-12
        sims = np.abs(K @ np.conj(cand)) / (kn * cn)        # true cosine [0,1]
        order = np.argsort(-sims)
        out = [(self.vocab[i], float(sims[i])) for i in order
               if self.vocab[i] not in exclude and sims[i] >= self.boundary][:top_k]
        if top_k == 1:
            return out[0] if out else (None, 0.0)
        return out

    # ---- read (learn) ---------------------------------------------------

    def read_sentence(self, sentence):
        ws = _tokens(sentence)
        if len(ws) < 2:
            return 0
        for w in ws:
            self._add_vocab(w)
        for w in set(ws):
            self.df[w] += 1
        self.first_tok.add(ws[0]); self.last_tok.add(ws[-1])
        self.read_log.append(ws)
        for p in range(len(ws)):
            ctx = self._ctx(ws[:p] + ws[p + 1:])
            self.acc[self._activate(ctx)] += bind(ctx, self.ck.key(ws[p]))
        self.n_sentences += 1
        self.n_writes += len(ws)
        return len(ws)

    def read(self, text):
        n = 0
        for sent in _SENT_SPLIT.split(str(text or "")):
            if sent.strip():
                n += self.read_sentence(sent)
        return n

    # ---- ask (plain-English question) -----------------------------------

    def ask(self, question, top_k=1):
        """Answer a free-form question. Cues = matched question tokens; the
        answer = what those cues resonate to that isn't itself a cue."""
        cues, seen = [], set()
        for t in _tokens(question):
            m = self._match(t)
            if m and m not in seen:
                seen.add(m); cues.append(m)
        return self._resolve(cues, seen, top_k)

    # ---- answer (explicit fill-in-the-blank) ----------------------------

    def answer(self, sentence_with_hole, hole_token="_", top_k=1):
        if hole_token not in sentence_with_hole:
            return (None, 0.0) if top_k == 1 else []
        left, _, right = sentence_with_hole.partition(hole_token)
        cues = _tokens(left) + _tokens(right)
        return self._resolve(cues, set(cues), top_k)

    # ---- emergent atom extraction (feed the derive-not-store engine) ----

    def rel_cut(self, min_rel_df=2, rel_frac=0.7):
        """Emergent relation threshold. A RELATION is the connective glue that
        recurs across ~ALL facts, so its df is near the MAXIMUM df; a recurring
        ENTITY (e.g. a middle node of a chain) recurs only a few times, well
        below that.  Cut = a fraction of the max df (scale-robust -- works for 3
        facts or 3 million), floored at min_rel_df -- so 'reports' (in every
        fact) is relational while 'mendaro vale' (in two) is not.  No list."""
        max_df = max(self.df.values()) if self.df else 1
        return max(int(min_rel_df), math.ceil(rel_frac * max_df))

    def _arg_spans(self, ws, cut):
        """Group CONSECUTIVE argument tokens (df < cut) into multi-word entity
        spans -- 'buenos aires' is one entity, not two.  Emergent: a run of
        novel tokens with no recurring connective between them is a single
        argument.  Returns the list of space-joined spans in order."""
        spans, cur = [], []
        for w in ws:
            if not self.is_relational(w, cut):
                cur.append(w)
            elif cur:
                spans.append(" ".join(cur)); cur = []
        if cur:
            spans.append(" ".join(cur))
        return spans

    def is_relational(self, token, cut=None):
        """A token is a RELATION iff it recurs (df >= rel_cut) AND lives in the
        interior -- never a subject (sentence-initial) and never an object
        (sentence-final).  Word-order structure separates a frequent ENTITY
        (qualan, a hub) or TYPE-word (team) from a true connective. Emergent,
        no list.  An RL `rel_bias` (set by reward) can override at the margin."""
        if cut is None:
            cut = self.rel_cut()
        bias = self.rel_bias.get(token, 0.0)
        if token in self.first_tok or token in self.last_tok:
            return bias > 0.5                        # edges are args unless RL insists
        return (self.df[token] + bias) >= cut

    def extract_atoms(self, min_rel_df=2, rel_frac=0.7):
        """Turn what was READ into clean (subject, relation, object) atoms,
        EMERGENTLY -- no templates, no relation list, no grammar.  A token is
        RELATIONAL if its df clears the emergent rel_cut (recurs across most
        facts); the rest are arguments/entities.  CONSECUTIVE argument tokens
        merge into one multi-word entity span ('buenos aires').  The relation is
        the recurring glue; span order (first/last) gives subject/object from the
        text's own linearity.

        Relations that haven't recurred enough stay ambiguous until more text
        disambiguates them -- honest distributional learning, not a hand-coded
        parser.  Returns a list of (subject, relation, object) triples.
        """
        cut = self.rel_cut(min_rel_df, rel_frac)
        atoms = []
        for ws in self.read_log:
            spans = self._arg_spans(ws, cut)
            rels = [w for w in ws if self.is_relational(w, cut)]
            if len(spans) >= 2 and rels:
                atoms.append((spans[0], " ".join(rels), spans[-1]))
        return atoms

    def parse_question(self, question, min_rel_df=2, rel_frac=0.7):
        """Emergently split a plain-English question into (entity, relation) for
        the derive engine -- no wh-list, no relation list.  Relation tokens are
        the recurring connectives (df >= rel_cut), morphology-mapped to known
        vocab ('report'->'reports'); the entity is the novel argument the
        question is about (lowest-df matched token); query-only words ('who',
        'does') map to nothing and drop themselves.  Returns (entity, relation)."""
        cut = self.rel_cut(min_rel_df, rel_frac)
        rels, spans, cur = [], [], []
        for t in _tokens(question):
            m = self._match(t)
            if m is None:                       # query-only word -> drops itself
                if cur:
                    spans.append(cur); cur = []
                continue
            if self.is_relational(m, cut):
                rels.append(m)
                if cur:
                    spans.append(cur); cur = []
            else:
                cur.append(m)                   # consecutive args = one entity
        if cur:
            spans.append(cur)
        # the asked entity = the most specific argument span (lowest summed df)
        ent = None
        if spans:
            best = min(spans, key=lambda s: sum(self.df[w] for w in s))
            ent = " ".join(best)
        return ent, " ".join(rels)

    def parse_chain(self, question, min_rel_df=2, rel_frac=0.7):
        """Parse a question into (entity, [relation_mentions]) for EMERGENT-depth
        multi-hop -- no depth count passed, no '\\'s' rule.  A relation-mention is
        a maximal run of relational tokens; mentions are separated by argument
        tokens, so the NUMBER of mentions = the hop count, read from the question
        itself.  The entity is the most specific (lowest-df) argument span.
        Mentions are returned in text order (outer-first); apply innermost-out.

        Honest scope: this resolves depth when the hops are separated by argument
        material (entity-separated chains) and single-hop.  Nested function-word
        chains ('the R1 of the R2 of X', articles) need 3-tier glue induction --
        the next rung -- so for those an explicit depth is still accepted."""
        cut = self.rel_cut(min_rel_df, rel_frac)
        mentions, cur, spans, ecur = [], [], [], []
        for t in _tokens(question):
            m = self._match(t)
            if m is None:
                continue
            if self.is_relational(m, cut):
                cur.append(m)
                if ecur:
                    spans.append(ecur); ecur = []
            else:
                ecur.append(m)
                if cur:
                    mentions.append(cur); cur = []
        if cur:
            mentions.append(cur)
        if ecur:
            spans.append(ecur)
        if not spans:
            return None, []
        entity = " ".join(min(spans, key=lambda s: sum(self.df[w] for w in s)))
        return entity, [" ".join(m) for m in mentions]

    def reinforce(self, subject, gold, reward=2.0):
        """Native dopamine-RL for relation discovery. A quiz gives (subject ->
        gold) but the parse missed -- the connecting tokens were mis-classified
        as arguments (e.g. a RARE relation below the df threshold).  Find the
        read fact 'subject ... gold' and reward its INTERIOR tokens (the true
        relation) so their rel_bias clears the threshold next time.  Reward-
        driven, not hand-set -- the organism learns which tokens are relations
        from whether the derived answer was right.  Returns True if it learned."""
        e, g = _tokens(subject), _tokens(gold)
        for ws in self.read_log:
            if len(ws) > len(e) + len(g) and ws[:len(e)] == e and ws[-len(g):] == g:
                for w in ws[len(e):len(ws) - len(g)]:        # the relation tokens
                    self.rel_bias[w] += float(reward)
                return True
        return False

    def comprehend(self, text, organism=None, min_rel_df=2, rel_frac=0.7):
        """Read messy text, then hand the EMERGENT atoms to the derive-not-store
        engine (organism.ingest_triples) so composites/inheritance/multi-hop are
        DERIVED, never stored.  The reader is the episodic front door; the derive
        engine is the semantic store.  Returns the extracted triples."""
        self.read(text)
        atoms = self.extract_atoms(min_rel_df=min_rel_df, rel_frac=rel_frac)
        if organism is not None and atoms:
            organism.ingest_triples(atoms)
        return atoms
