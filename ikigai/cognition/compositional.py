"""
ikigai.cognition.compositional -- Pack 304 Compositional substrate v0.

DERIVE-NOT-STORE.  Store only irreducible ATOMS (arbitrary facts that
cannot be computed -- capital(France)=Paris, continent(France)=Europe);
DERIVE relations and compositions at query time by composing atoms via
structural rules, instead of caching every composite.

The win is combinatorial: from n atomic country attributes the engine
answers O(n^2) composite facts (same-continent comparisons, attribute
inheritance, multi-hop chains) with ZERO cache entries for the derived
answers.  This is the math ring (291.8 -- derive any sum, store none)
generalised from arithmetic to relations.

Honesty (Rule: NO hardcoding on the production path):
    The rules here are RELATION ALGEBRA (grammar), not facts.  "A capital
    is in the same continent as its country" is a structural rewrite, the
    same class of object as the 293 multi-hop templates and the math-ring
    operators -- it composes stored atoms, it does not encode any specific
    country.  No country/continent is named in this module; every concrete
    value comes from a cache atom lookup.

Atoms live in the existing Pack 273 anchor_actions cache, phrased as
questions ("what continent is <country> in" -> <continent>).  This module
only READS them -- it never writes the cache, so derived facts cost no
storage.
"""
import hashlib
import re

import numpy as np


# Canonical question phrasings per relation.  atom(rel, entity) tries each
# until one resolves against the cache.  Templates only -- no facts.
_REL_TEMPLATES = {
    'continent': ('what continent is {e} in', 'continent of {e}'),
    'capital':   ('what is the capital of {e}', 'capital of {e}'),
    'currency':  ('what is the currency of {e}', 'currency of {e}'),
    'language':  ('what language is spoken in {e}', 'language of {e}'),
    'region':    ('what region is {e} in', 'region of {e}'),
    'population':('what is the population of {e}', 'population of {e}'),
    'country':   ('what country is {e} the capital of', 'country of {e}'),
}


class CompositionEngine:
    """Derive composite facts from atomic cache relations.

    Constructed with a GeneralReasoner (for tokenize + cat4 access).
    All lookups are read-only; the engine never mutates the cache.
    """

    # ---- composite-query templates (relation algebra, not facts) -----
    # "what continent is the capital of France in" -> continent(France)
    # (a capital inherits its country's continent: continent o capital = continent)
    _CAP_CONT_WHICH = re.compile(
        r'^\s*(?:what|which)\s+(continent|region)\s+is\s+(?:the\s+)?'
        r'capital\s+(?:city\s+)?of\s+(.+?)\s+in\s*\??\s*$', re.IGNORECASE)
    # "is the capital of France in Europe" -> continent(France) == Europe
    _CAP_CONT_YESNO = re.compile(
        r'^\s*is\s+(?:the\s+)?capital\s+(?:city\s+)?of\s+(.+?)\s+in\s+'
        r'(.+?)\s*\??\s*$', re.IGNORECASE)
    # Generalised same-attribute comparison over ANY comparable relation:
    #   "are France and Japan in/on the same continent"
    #   "do France and Germany use/have/share the same currency"
    #   "do France and Spain speak the same language"
    _SAME = re.compile(
        r'^\s*(?:are|is|do|does)\s+(.+?)\s+and\s+(.+?)\s+'
        r'(?:in|on|have|share|use|speak)\s+the\s+same\s+'
        r'(\w+)\s*\??\s*$', re.IGNORECASE)
    # "is France in the same continent as Japan"
    _SAME_AS = re.compile(
        r'^\s*is\s+(.+?)\s+in\s+the\s+same\s+(\w+)\s+as\s+'
        r'(.+?)\s*\??\s*$', re.IGNORECASE)
    # Numeric comparison: "is France bigger than Japan" (default population),
    # "does China have a larger population than India"
    _BIGGER = re.compile(
        r'^\s*is\s+(.+?)\s+(bigger|larger|smaller|higher|lower|greater)\s+'
        r'than\s+(.+?)\s*\??\s*$', re.IGNORECASE)
    _BIGGER_ATTR = re.compile(
        r'^\s*does\s+(.+?)\s+have\s+(?:a\s+)?'
        r'(bigger|larger|smaller|higher|lower|greater)\s+(\w+)\s+than\s+'
        r'(.+?)\s*\??\s*$', re.IGNORECASE)
    # which-of-two: "which is bigger France or Japan"
    _WHICH_BIGGER = re.compile(
        r'^\s*which\s+(?:country\s+)?(?:is\s+|has\s+(?:a\s+)?)?'
        r'(bigger|larger|smaller|higher|lower|greater)(?:\s+\w+)?\s*,?\s+'
        r'(.+?)\s+or\s+(.+?)\s*\??\s*$', re.IGNORECASE)
    # generic 2-hop: "what is the currency of the capital of France"
    # = attr1(attr2_value); capital inherits country attrs, so the capital's
    # currency is its country's currency -> attr1(country) directly.
    _CHAIN = re.compile(
        r'^\s*what\s+is\s+the\s+(\w+)\s+of\s+(?:the\s+)?'
        r'capital\s+(?:city\s+)?of\s+(.+?)\s*\??\s*$', re.IGNORECASE)
    # Pack 304.2 generic multi-relation chain: "what is the <rel1> of the
    # <rel2> of <X>" = rel1(rel2(X)).  The inner link rel2(X) may resolve
    # directly OR via a LEARNED inverse rule (reverse-lookup, 305.1).
    _CHAIN2 = re.compile(
        r'^\s*what\s+is\s+the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)\s+of\s+'
        r'(.+?)\s*\??\s*$', re.IGNORECASE)
    # Pack 317 arbitrary N-hop: "what is the <rel1> of the <rel2> of ... <X>"
    # = rel1(rel2(...(X))).  Parsed by stripping leading "the <rel> of"
    # prefixes; resolved innermost-out via _chain_inner (each hop direct atom
    # or learned inverse).  Generalises _CHAIN2 (2-hop) to any depth.
    _CHAIN_HEAD = re.compile(
        r'^\s*what\s+is\s+the\s+(\w+)\s+of\s+(.+?)\s*\??\s*$', re.IGNORECASE)
    _CHAIN_INNER = re.compile(
        r'^the\s+(\w+)\s+of\s+(.+)$', re.IGNORECASE)

    def __init__(self, reasoner):
        self.gr = reasoner
        self._stats = {'derive_calls': 0, 'derived': 0, 'atom_lookups': 0,
                       'atom_hits': 0}
        # Pack 305 -- enumerable atom index + learned-rule store.  The
        # hash-keyed anchor cache cannot be enumerated by (subj, rel), so
        # the rule miner needs this structured view.  It self-populates
        # whenever atom() resolves a hit (no curated lists).  Persisted on
        # the organism so it survives reloads.
        self.triples = {}        # (subj, rel) -> val   MINING INDEX (bounded), not the fact store
        self.entities = set()    # discovered subjects
        self.relations = set()   # discovered relations
        self.learned_rules = []  # promoted derivation rules (dicts)
        # Day-96 -- the mining index is a SAMPLE, not a store. Unbounded it silently became the
        # fact store (4.16M Wikidata edges as python string keys = 662 B/edge -> 3.5 GB). The rule
        # miner needs a universe to mine, not every fact; bulk facts live in `addr` at 5.3 B/edge.
        self._triples_cap = 250_000
        self.addr = None         # AddressFactStore: packed bulk graph facts (attach_addresses)
        # When True (test/proof mode), inheritance derivations require a
        # LEARNED rule -- proves discovery is load-bearing, not the
        # authored regex.  Default False so 304/304.1 behaviour holds.
        self.require_learned = False
        self._load_state()

    # ---- persisted state (rides on the organism) --------------------

    def _load_state(self):
        st = getattr(self.gr.org, '_comp_state', None)
        if isinstance(st, dict):
            raw = st.get('triples')
            self.triples = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    parts = k.split('\t') if isinstance(k, str) else list(k)
                    if len(parts) == 2:
                        self.triples[(parts[0], parts[1])] = v
            self.entities = set(st.get('entities', []))
            self.relations = set(st.get('relations', []))
            self.learned_rules = list(st.get('learned_rules', []))
            self.rebuild_rule_bank()

    def save_state(self):
        """Snapshot the atom index + learned rules onto the organism so a
        subsequent save_ikg persists them."""
        self.gr.org._comp_state = {
            'triples': {f'{s}\t{r}': v for (s, r), v in self.triples.items()},
            'entities': sorted(self.entities),
            'relations': sorted(self.relations),
            'learned_rules': self.learned_rules,
        }

    def _record(self, subj, rel, val):
        subj = str(subj).strip().lower()
        rel = str(rel).strip().lower()
        if not subj or not val:
            return
        # Day-96 -- `triples` is the MINING INDEX, not the fact store. Its job is to give the rule
        # miner an enumerable universe; the facts themselves live in the anchor cache (exact) and,
        # for bulk graph relations, in the packed address store (5.3 B/edge). Left unbounded it
        # silently became the fact store: 4.16M Wikidata edges of python STRING keys = 662 B/edge
        # -> 3.5 GB RSS. A miner does not need the universe, it needs a SAMPLE -- so bound it.
        if (subj, rel) not in self.triples and len(self.triples) >= self._triples_cap:
            self.relations.add(rel)      # relations are few; entities are NOT (3.5M strings = ~300 MB)
            return
        self.triples[(subj, rel)] = val
        self.entities.add(subj)          # `entities` == "the cache/mining universe", stays bounded
        self.relations.add(rel)

    def attach_addresses(self, store):
        """Day-96 -- back the atom layer with the PACKED ADDRESS STORE (AddressFactStore) for bulk
        graph relations. Facts become three indices over a shared lexicon (5.3 B/edge, measured)
        instead of python strings in a dict (662 B/edge). atoms()/atom() fall through to it, so
        transitive_reach derives the COMPLETE closure over ALL edges without the cache holding
        millions of entries."""
        self.addr = store
        return store

    # ---- atom layer (read-only over the cache) ----------------------

    def _templates_for(self, rel):
        """Pack 326 -- question templates for a relation.  Known relations use
        their curated phrasings; ANY other relation (e.g. an arbitrary KG
        predicate like 'isa' / 'partof' / 'usedfor') falls back to a generic
        template so the read/write round-trip is consistent.  This is what lets
        a real knowledge graph -- with relations we never hand-listed -- ingest
        and read back without curated lists."""
        t = _REL_TEMPLATES.get(rel)
        if t:
            return t
        r = str(rel).strip().lower()
        return (f'what is the {r} of {{e}}',)

    def atom(self, rel, entity):
        """Look up an atomic relation value from the cache.  Read-only on
        the cache; auto-records the resolved (subj, rel, val) into the
        enumerable atom index so the rule miner has a universe to mine
        (no curated lists).

        Returns the lowercased value string, or None if not stored.
        """
        self._stats['atom_lookups'] += 1
        templates = self._templates_for(rel)
        if not templates:
            return None
        from ikigai.cognition.cat4_absorb import _stable_anchor
        cat4 = getattr(self.gr.org, 'cat4', None)
        ent = str(entity).strip().lower()
        # Day-87 -- counterfactual intervention: a temporarily installed override
        # replaces one atom so the SAME derive-chaining flows over the intervened
        # world (do-operator on the substrate).  Read-only; removed after measuring.
        ov = getattr(self, '_cf_override', None)
        if ov:
            hit = ov.get((ent, str(rel).strip().lower()))
            if hit is not None:
                return hit
        # Day-99 -- the packed store is consulted BEFORE the anchor-cache guard. It used to sit
        # below `if not cat4.anchor_actions: return None`, so on an organism whose knowledge came
        # ONLY from a bulk ingest the cache is empty and atom() returned None without ever asking
        # the store -- measured: reach_addressed found {korumimek, rumikodor} while atom() said None
        # and compose() said "unknown". The store needs no cache to answer; it is its own index.
        _si = self._addr_id(rel, ent)                # Day-99: store authoritative for THIS entity
        if _si is not None:
            vals = self._addr_objects(rel, _si)      # int id -> objects_of skips a second id_of
            if vals:
                self._stats['atom_hits'] += 1
                return vals[-1]                      # no _record: the store IS the index here
            return None
        if cat4 is None or not getattr(cat4, 'anchor_actions', None):
            return None
        for tmpl in templates:
            q = tmpl.format(e=ent)
            toks = self.gr.tokenize(q)
            entry = cat4.anchor_actions.get(_stable_anchor(toks))
            if entry:
                val = ' '.join(entry[-1]).strip().lower()
                if val:
                    self._stats['atom_hits'] += 1
                    self._record(ent, rel, val)
                    return val
        return None

    def atoms(self, rel, entity):
        """Pack 329 -- ALL stored values for (rel, entity), not just the last.
        The anchor cache already keeps every distinct answer per question
        (ingest appends them); atom() returns one, this returns the full list
        -- the multi-value meaning web (cat isa feline AND pet AND carnivore).
        Read-only. Returns [] if none."""
        from ikigai.cognition.cat4_absorb import _stable_anchor
        cat4 = getattr(self.gr.org, 'cat4', None)
        ent = str(entity).strip().lower()
        out = []
        # Day-96 -- consult the anchor cache ONLY for entities it actually holds. `entities` is the
        # cache/mining universe (bounded); a bulk entity that lives only in the packed store is not
        # in it. Skipping the miss path for those is not just speed (the miss costs a template
        # format + tokenize + anchor hash PER NODE -- measured 541 us/query on Wikidata closures);
        # it is also what keeps the walk O(packed lookup) at scale.
        # Day-99 -- asked BEFORE the cat4 guard: the store is its own index and needs no cache, and
        # an organism taught only by bulk ingest has an EMPTY cache (see atom()).
        _si = self._addr_id(rel, ent)
        if _si is not None:
            return self._addr_objects(rel, _si)      # store authoritative for THIS entity (int id)
        if cat4 is None or not getattr(cat4, 'anchor_actions', None):
            return []
        for tmpl in self._templates_for(rel):
            entry = cat4.anchor_actions.get(
                _stable_anchor(self.gr.tokenize(tmpl.format(e=ent))))
            if entry:
                for toks in entry:
                    v = ' '.join(toks).strip().lower()
                    if v and v not in out:
                        out.append(v)
        return out

    def _rel_packed(self, rel, ent=None):
        """Day-96/99 -- is the packed address store authoritative for this lookup?

        Day-96 scoped authority to the RELATION: the store was built from the COMPLETE edge set of
        that relation, so it holds every fact of it. The anchor cache must then be skipped, for two
        independent reasons:

        1. CORRECTNESS. The cache stores a value as a TOKEN-JOIN, and the tokenizer drops digits
           (`[^\\W\\d_]+`). A Wikidata id 'Q12345' round-trips through it as 'q'. Merging those
           mangled values into a closure injects junk nodes -- measured 44/500 closures wrong.
        2. SPEED. A cache lookup for a bulk entity is a guaranteed MISS that still pays a template
           format + tokenize + anchor hash PER NODE -- measured 541 us/query on Wikidata closures.

        Gated on the RELATION, never on `self.entities`: that set is populated BY successful atom()
        hits, so gating on membership in it blocks every not-yet-seen entity and silently kills rule
        mining (measured: is_transitive('p279') flipped True -> False).

        Day-99 -- RELATION-WIDE authority was a real bug, and the docstring above carries the wrong
        assumption that caused it: "built from the COMPLETE edge set" is true for a ONE-SHOT bulk
        load and FALSE the moment any other door teaches that relation. Measured: with a Wikidata-
        style store attached for `isa`, a fact taught by ingest_triples/study/mem.teach about an
        entity the store has never heard of became INVISIBLE -- `compose` went from "The barukozin
        is a rukobamek, and ultimately a kobarudor" to "The barukozin is unknown". The organism did
        not fabricate (calibration held), it FORGOT. That silently breaks every "feed data and keep
        learning" workflow.

        So authority is now scoped to the (relation, ENTITY): the store speaks only for entities it
        actually holds.
          * entity IN the store  -> store authoritative, cache skipped. Both Day-96 reasons still
            apply in full: no digit-mangled cache junk, no per-node miss penalty. Wikidata keeps its
            correctness and its 56 us/query.
          * entity NOT in the store -> the store has nothing to say; consult the cache. Additivity
            restored: a newly taught entity is visible again.
        Cost is one id_of() hash lookup, which objects_of() already pays internally.

        Honest residual gap (named, not hidden): a NEW fact taught via the cache about an entity the
        store ALREADY holds stays invisible, because the store still speaks for that entity. Closing
        that needs a write path into the store, not a read-order change."""
        if ent is None:                       # relation-level question (has this store got `rel`?)
            st = self.addr
            if st is None:
                return False
            try:
                return st.has_relation(rel)
            except Exception:
                return False
        return self._addr_id(rel, ent) is not None

    def _addr_id(self, rel, ent):
        """Day-99 -- resolve `ent` to the packed store's int id IFF the store is authoritative for
        (rel, ent); else None.  Callers pass the returned INT straight to _addr_objects.

        Resolving ONCE matters.  objects_of() already does its own id_of(), so the naive form --
        ask _rel_packed (id_of), then call _addr_objects with the STRING (id_of again) -- pays the
        hash twice: measured +1.95 us/atom, +20.3% on real Wikidata p279 edges.  Handing the int
        down skips the second lookup and makes entity-scoped authority cost nothing over the old
        relation-scoped path."""
        st = self.addr
        if st is None:
            return None
        try:
            if not st.has_relation(rel):
                return None
            return st.id_of(str(ent).strip().lower())
        except Exception:
            return None

    def _addr_objects(self, rel, ent):
        """Day-96 -- objects of (ent, rel) from the PACKED ADDRESS STORE (5.3 B/edge) when the
        anchor cache does not hold them. This is what lets bulk graph knowledge (Wikidata-scale)
        stay OUT of the 290-B/entry cache while remaining fully derivable. Returns [] if no store
        is attached or the relation is not in it."""
        st = self.addr
        if st is None:
            return []
        try:
            ids = st.objects_of(ent, str(rel).strip().lower())
        except Exception:
            return []
        out = []
        for i in ids:
            s = st.str_of(i)
            if s and s not in out:
                out.append(s)
        return out

    def reverse_atom(self, rel, value):
        """Inverse lookup: find the subject whose `rel` equals `value`
        (e.g. capital='paris' -> 'france'), by scanning the enumerable
        atom index.  The inverse direction is the round-trip a learned
        inverse rule (Pack 305.1) sanctions.  Returns subject or None."""
        v = str(value).strip().lower()
        for (s, r), val in self.triples.items():
            if r == rel and val == v:
                return s
        return None

    def _parse_chain(self, text):
        """Pack 317 -- parse 'what is the r1 of the r2 of ... of X' into
        (rels=[r1,r2,...], entity=X).  Returns None if not a chain query.
        rels are normalised; entity is the bare innermost subject."""
        m = self._CHAIN_HEAD.match(text or '')
        if not m:
            return None
        rels = [self._norm_rel(m.group(1))]
        rest = m.group(2).strip()
        while True:
            mm = self._CHAIN_INNER.match(rest)
            if not mm:
                break
            rels.append(self._norm_rel(mm.group(1)))
            rest = mm.group(2).strip()
        entity = rest.rstrip('?').strip()
        if not entity:
            return None
        return rels, entity

    def _chain_resolve(self, rels, entity):
        """Pack 317 -- resolve rel1(rel2(...(entity))) innermost-out.  Each
        hop via _chain_inner (direct atom OR learned inverse).  Read-only;
        never writes the cache.  Returns the final value or None."""
        cur = entity
        for rel in reversed(rels):
            cur = self._chain_inner(rel, cur)
            if not cur:
                return None
        return cur

    def _chain_inner(self, rel, x):
        """Resolve rel(x) for the inner hop of a chain: try the atom
        directly; else, if a LEARNED inverse rule covers `rel`, reverse-
        lookup via the inverse relation (e.g. country(berlin) by reverse
        of capital).  Returns the inner entity/value, or None."""
        direct = self.atom(rel, x)
        if direct:
            return direct
        inv = self.sanctioned_inverse(rel)
        if inv:
            return self.reverse_atom(inv, x)
        return None

    def _chain_inner_all(self, rel, x):
        """Day-96 -- ALL parents of x under rel, not just one.

        `_chain_inner` resolves ONE value via atom(), which returns entry[-1] -- the LAST value the
        anchor cache holds.  But the cache is MULTI-VALUED (atoms(), Pack 329): 'zebu isa kolu' AND
        'zebu isa mira' are both stored.  Following only the last one silently DROPS every other
        parent, so a transitive closure over a multi-parent taxonomy came back INCOMPLETE -- the
        organism would answer 'no, zebu is not a kolu' when it demonstrably is.  For a
        correct-or-abstain system an omission that reads as a confident 'no' is the worst failure
        mode there is, and on Wikidata p279 (4.16M edges, 3.52M subjects) ~15% of edges are
        multi-parent.  This returns the full parent set so the closure can BFS over all of them.
        """
        vals = self.atoms(rel, x)
        if vals:
            # keep the miner's enumerable universe fed exactly as atom() did (it recorded the
            # value it returned, i.e. the last one) -- mining behaviour is unchanged.
            self._record(str(x).strip().lower(), rel, vals[-1])
            return vals
        inv = self.sanctioned_inverse(rel)
        if inv:
            v = self.reverse_atom(inv, x)
            return [v] if v else []
        return []

    def is_transitive(self, rel):
        """Pack 317.2 -- True if a LEARNED transitive rule covers `rel`."""
        rel = self._norm_rel(rel)
        return any(r.get('type') == 'transitive'
                   and self._norm_rel(r.get('rel')) == rel
                   for r in self.learned_rules)

    def transitive_reach(self, rel, x, max_depth=None):
        """Pack 317.2 + 318 -- follow a LEARNED-transitive `rel` from x along
        the chain to its root: [x, rel(x), rel(rel(x)), ...].  Computes the
        transitive closure on demand (derive-not-store) instead of storing
        every ancestor pair.  Read-only.  None if rel isn't sanctioned
        transitive; single-element list if x has no outgoing rel.

        Pack 318 (CONVERGENCE-BOUNDED, not fixed-cap): the hop count is
        bounded by the DATA, not an arbitrary number -- it follows the chain
        until it hits the root or a cycle.  max_depth defaults to the entity
        count (the longest possible acyclic chain), so 'hop as many as needed'
        is honoured; pass an int only as an explicit safety override.
        """
        if not self.is_transitive(rel):
            return None
        if max_depth is None:
            # longest acyclic chain. With a packed store attached, the entity universe is ITS
            # lexicon (millions), not the bounded mining index -- otherwise the walk would stop
            # short of the root on bulk graphs.
            n_ent = len(self.entities)
            if self.addr is not None:
                n_ent = max(n_ent, getattr(self.addr, 'n_entities', 0))
            max_depth = n_ent + 2
        # Day-96 -- BFS over ALL parents, not a single-parent chain. The old walk took one value
        # per hop (atom() = the cache's LAST value), so a node with two parents lost one of them
        # and every ancestor above it: transitive_reach('isa','zebu') returned ['zebu','mira',
        # 'rulo'] while the truth is {kolu, mira, pova, rulo}. Return shape is UNCHANGED --
        # [x, *ancestors] -- so every caller's chain[1:] still means "the ancestors of x", it is
        # now simply COMPLETE. Still derive-not-store and convergence-bounded (stops at roots).
        x0 = str(x).strip().lower()
        # Day-96 -- when the relation lives in the packed store, let the STORE walk it. It closes in
        # ID space; the engine's generic walk closes in STRING space, paying a blake2b + bisect
        # (str->id) and a blob decode (id->str) on EVERY hop -- measured 189 us/query vs 14 us for
        # the same closure in id space. Convert to surface strings ONCE, at the end.
        if self._rel_packed(rel):
            st = self.addr
            return [x0] + [s for s in (st.str_of(i) for i in st.transitive_reach(rel, x0)) if s]
        chain = [x0]
        seen = {x0}
        frontier = [x0]
        for _ in range(int(max_depth)):
            nxt = []
            for cur in frontier:
                for nb in self._chain_inner_all(rel, cur):
                    if nb and nb not in seen:
                        seen.add(nb)
                        chain.append(nb)
                        nxt.append(nb)
            if not nxt:                      # converged: every branch hit a root or a cycle
                break
            frontier = nxt
        return chain

    def transitive_related(self, rel, x, target):
        """Pack 317.2 -- is `target` reachable from x via transitive `rel`?
        (ancestor / closure membership), derived not stored.  None if rel
        not sanctioned transitive."""
        chain = self.transitive_reach(rel, x)
        if chain is None:
            return None
        t = str(target).strip().lower()
        return t in chain[1:]

    def inherited_atom(self, attr, entity):
        """Pack 350 -- attribute inheritance DOWN a taxonomy, derive-not-store.
        Resolve attr(entity) when it is not stored on the entity itself by
        climbing a TRANSITIVE inheritance link (e.g. subclassof) to the nearest
        ancestor that does carry the attribute -- so an attribute stored ONCE on
        a class is answered for every one of its (transitive) descendants without
        storing a single descendant copy.  This is the deep multiplier the
        comparison count masks: one `warm_blooded(mammal)` serves every mammal.

        Read-only.  Falls back to the direct atom; returns None when neither the
        entity nor any sanctioned ancestor carries the attribute.  Exceptions are
        honoured automatically -- a descendant with its OWN stored value shadows
        the inherited one (the direct lookup wins)."""
        attr = self._norm_rel(attr)
        direct = self.atom(attr, entity)
        if direct:
            return direct
        # every transitive link over which attr inherits (isa, subclassof, ...)
        links = []
        for r in self.learned_rules:
            if r.get('type') != 'inheritance':
                continue
            if not (r.get('attr') == '*' or self._norm_rel(r.get('attr')) == attr):
                continue
            link = r.get('link')
            if link not in links:
                links.append(link)     # an inheritance rule sanctions climbing its link
        if not links:
            return None
        # climb the taxonomy up ALL of them together -- a real hierarchy mixes
        # instance-of and subclass-of, so a breadth-first climb over every
        # inheritance link reaches the nearest ancestor that defines attr, even
        # across mixed hops (vanadium -isa-> metal -subclassof-> material).
        entity = str(entity).strip().lower()
        seen, frontier = {entity}, [entity]
        while frontier:
            nxt = []
            for x in frontier:
                for link in links:
                    anc = self.atom(link, x)
                    if anc and anc not in seen:
                        v = self.atom(attr, anc)
                        if v:
                            return v
                        seen.add(anc); nxt.append(anc)
            frontier = nxt
        return None

    def entity_signature(self, entity):
        """Day-85 -- the SUBSTRATE signature of an entity: a single holographic
        vector bundling bind(key(relation), key(value)) over every fact stored
        about it.  This is what the entity IS, in the phasor algebra -- the
        object concept-invention reasons over.  Returns (sig_hv, props, n)."""
        from ikigai.cognition.phasor_state import bind
        ck = self.gr.org.unified.ck
        e = str(entity).strip().lower()
        sig, props = None, []
        for (s, r), v in self.triples.items():
            if s == e and v:
                b = bind(ck.key(r), ck.key(v))
                sig = b if sig is None else sig + b
                props.append((r, v))
        return sig, props, len(props)

    def reverse_reach(self, target, rels=None):
        """Day-86 -- REVERSE derivation: every entity that reaches `target`
        through the given relations, derived NOT stored -- the inverse of a
        forward query.  'what things ARE a metal' = every isa / subclass
        descendant of metal, found by a reverse breadth-first walk up the edges,
        the closure computed on demand and never materialised.  Defaults to the
        taxonomic links (learned-transitive relations plus 'isa').  Returns the
        descendant entities in discovery order."""
        target = str(target).strip().lower()
        if rels is None:
            # taxonomic LINK relations (values are themselves entities) -- isa,
            # subclassof and the like -- detected structurally, so the reverse
            # closure follows the class hierarchy even where the transitive RULE
            # was not mined (too few chains), while attribute relations (leaf
            # values, e.g. 'capital') are excluded.
            from ikigai.cognition.rule_discovery import RuleMiner
            link, _attr = RuleMiner.classify_relations(
                self.triples, self.relations, self.entities)
            rels = set(link)
        rels = set(rels)
        children = {}                              # value -> [subjects] over these rels
        for (s, r), v in self.triples.items():
            if r in rels and v:
                children.setdefault(v, []).append(s)
        seen, out, frontier = set(), [], [target]
        while frontier:
            nxt = []
            for t in frontier:
                for s in children.get(t, []):
                    if s not in seen:
                        seen.add(s); out.append(s); nxt.append(s)
            frontier = nxt
        return out

    def _cleanup(self, vec, candidates):
        """Resonance read-out: return the codebook token whose key best matches
        `vec` by cosine (the VSA cleanup step), with its score.  This is the
        geometric decision a superposition read REQUIRES -- not a python test
        over stored values."""
        from ikigai.cognition.phasor_state import cosine
        ck = self.gr.org.unified.ck
        best, bs = None, -1.0
        for c in candidates:
            s = cosine(vec, ck.key(c))
            if s > bs:
                best, bs = c, s
        return best, bs

    def analogy(self, a, b, c):
        """Day-86 GOLD -- A:B :: C:? by PURE SUBSTRATE ALGEBRA.  Recover the
        relation linking A->B by UNBINDING it out of A's holographic signature
        and cleaning up over the relation vocabulary (the relation is read by
        resonance, never scanned), then APPLY that relation to C the same way --
        unbind it out of C's signature and clean up over the value vocabulary.
        No relation list, no dict search: bind/unbind/cleanup only.  Faithful --
        the (relation, answer) is verified against the store when present.
        Returns (answer, relation, score, verified)."""
        from ikigai.cognition.phasor_state import unbind
        ck = self.gr.org.unified.ck
        sa, _pa, _na = self.entity_signature(a)
        sc, _pc, _nc = self.entity_signature(c)
        if sa is None or sc is None:
            return None, None, 0.0, False
        rel, _rs = self._cleanup(unbind(sa, ck.key(str(b).strip().lower())),
                                 list(self.relations))
        if rel is None:
            return None, None, 0.0, False
        objs = {v for (_s, _r), v in self.triples.items() if v}
        ans, ascore = self._cleanup(unbind(sc, ck.key(rel)), objs)
        verified = (self.atom(rel, c) == ans) if ans else False
        return ans, rel, round(float(ascore), 3), bool(verified)

    def dream_discover(self, max_deductive=8, max_analogical=6,
                       resonance_thresh=0.45, seed=None, max_scan=500):
        """Day-87 GOLD -- CREATIVE SLEEP.  Unprompted, the organism recombines
        what it knows and WAKES KNOWING things nobody told it -- two honestly
        separated ways, both pure substrate work:

        (1) DEDUCTIVE dreaming (proven true).  It spontaneously chases its own
        learned closures -- transitive links and attribute inheritance -- and
        surfaces facts that are ENTAILED by its rules but were never directly
        stored or queried (platinum -> metal -> material; a mammal's warm blood
        inherited onto a whale it was never told about).  Each is re-derived and
        VERIFIED before it is reported, so a discovery is a proof, not a guess --
        derive-chaining, the sanctioned forward reach, run of the organism's own
        accord instead of on demand.  These are TRUE; they are not stored (the
        closure is free to re-derive -- that IS the multiplier).

        (2) ANALOGICAL dreaming (a conjecture).  It picks an entity, finds the
        peer whose holographic SIGNATURE most RESONATES with its own (the same
        entity_signature bundle + cosine that concept invention and analogy use --
        a geometric decision, not a dict scan), and transfers a property the peer
        has and it lacks: most things this shape have property P, so maybe this
        does too.  Unlike a deduction this cannot be verified from the store, so
        it is returned as a low-confidence CONJECTURE -- a testable hypothesis the
        life loop can later CONFIRM or be SURPRISED by, closing active inference.

        Read-only on the store.  Returns {'discoveries': [(s,r,v,provenance)],
        'conjectures': [(s,r,v,score,provenance)]}.  `seed` makes the wander
        reproducible for the gate."""
        import numpy as _np
        from ikigai.cognition.phasor_state import cosine
        rng = _np.random.default_rng(seed)
        ents = sorted(self.entities)
        discoveries, seen_d = [], set()

        # ---- (1) DEDUCTIVE: chase learned closures the organism was never told
        trans_links = [r for r in sorted(self.relations) if self.is_transitive(r)]
        inh = [(rr.get('link'), rr.get('attr')) for rr in self.learned_rules
               if rr.get('type') == 'inheritance']
        sample = list(ents)
        rng.shuffle(sample)
        for e in sample:
            if len(discoveries) >= max_deductive:
                break
            # transitive: every ancestor BEYOND the direct parent is derived-not-stored
            for link in trans_links:
                chain = self.transitive_reach(link, e) or []
                for anc in chain[2:]:                       # chain[1] is the stored parent
                    key = (e, link, anc)
                    if key in seen_d:
                        continue
                    if self.transitive_related(link, e, anc):   # VERIFY the entailment
                        seen_d.add(key)
                        discoveries.append((e, link, anc, f'derived via {link}-closure'))
            # inheritance: an attribute the entity carries only through an ancestor
            for link, attr in inh:
                if not attr or attr == '*':
                    continue
                if self.atom(attr, e):                      # already its own -- not a discovery
                    continue
                v = self.inherited_atom(attr, e)
                if v:
                    key = (e, attr, v)
                    if key not in seen_d:
                        seen_d.add(key)
                        discoveries.append((e, attr, v, f'inherited {attr} up {link}'))
        discoveries = discoveries[:max_deductive]

        # ---- (2) ANALOGICAL: resonance-driven conjecture (a testable guess)
        conjectures, seen_c = [], set()
        # A dream SAMPLES -- it does not exhaustively scan the whole mind.  Bound
        # the signature work to a random subset so a large store (tens of
        # thousands of entities) does not stall the heartbeat; the subset is
        # reshuffled each night by `seed`, so over many sleeps the whole store is
        # visited.  Cost is O(max_scan) signatures, independent of store size.
        scan = list(ents)
        rng.shuffle(scan)
        scan = scan[:max(max_analogical * 4, int(max_scan))]
        sigs = {}
        for e in scan:
            s, props, n = self.entity_signature(e)
            if s is not None and n:
                sigs[e] = (s, dict(props))
        keys = list(sigs)
        rng.shuffle(keys)
        for e in keys:
            if len(conjectures) >= max_analogical:
                break
            se, pe = sigs[e]
            # the peer whose SIGNATURE resonates most -- geometric, not a scan of shared keys
            best_p, best_s = None, -1.0
            for p in sigs:
                if p == e:
                    continue
                cs = cosine(se, sigs[p][0])
                if cs > best_s:
                    best_p, best_s = p, cs
            if best_p is None or best_s < resonance_thresh:
                continue
            pp = sigs[best_p][1]
            for r, v in pp.items():                         # a property the peer has
                if r in pe:                                 # entity already has this relation
                    continue
                if self.inherited_atom(r, e):               # or inherits it -- not a gap
                    continue
                ck = (e, r)
                if ck in seen_c:
                    continue
                seen_c.add(ck)
                conjectures.append((e, r, v, round(float(best_s), 3),
                                    f'analogy with {best_p}'))
                break                                       # one conjecture per entity per dream
        conjectures = conjectures[:max_analogical]
        return {'discoveries': discoveries, 'conjectures': conjectures}

    def odd_one_out(self, entities):
        """Day-87 -- fluid-reasoning IQ primitive: which one does NOT belong?
        Each entity is its holographic SIGNATURE (bundle of bind(rel,val) over its
        facts).  The outlier is the one that RESONATES LEAST with the bundle of the
        others -- a pure geometric decision (cosine to the rest), no feature list,
        no rule authored.  Returns (outlier, scores) where a lower score = more
        different from the group."""
        from ikigai.cognition.phasor_state import cosine
        sigs = {}
        for e in entities:
            s, _p, n = self.entity_signature(e)
            if s is not None and n:
                sigs[e] = s
        if len(sigs) < 3:
            return None, {}
        total = None
        for s in sigs.values():
            total = s if total is None else total + s
        scores = {}
        for e, s in sigs.items():
            rest = total - s                      # the group WITHOUT e
            scores[e] = round(float(cosine(s, rest)), 3)
        outlier = min(scores, key=scores.get)
        return outlier, scores

    def counterfactual(self, entity, relation, new_value, probe_attrs=None):
        """Day-87 GOLD -- NATIVE CAUSAL INTERVENTION.  Answer 'if ENTITY's
        RELATION were NEW_VALUE, what would follow?' by INTERVENING on the
        substrate -- installing a do-operator override on that one atom -- and
        RE-DERIVING the downstream closure (its taxonomic chain + every attribute
        it inherits), then diffing against the factual baseline.  There is no
        learned causal model: causation here is STRUCTURAL DEPENDENCY made visible
        by derive-chaining over the intervened world -- change an upstream atom and
        the derived consequences move with it.  Read-only: the override is
        installed, measured, and removed.  Returns {baseline, intervention,
        counterfactual, changes}."""
        entity = str(entity).strip().lower()
        relation = self._norm_rel(relation)
        new_value = str(new_value).strip().lower()
        if probe_attrs is None:
            from ikigai.cognition.rule_discovery import RuleMiner
            _link, attr = RuleMiner.classify_relations(
                self.triples, self.relations, self.entities)
            probe_attrs = sorted(attr)

        def snapshot():
            s = {a: (self.atom(a, entity) or self.inherited_atom(a, entity))
                 for a in probe_attrs}
            if self.is_transitive(relation):
                s['__chain__'] = self.transitive_reach(relation, entity)
            return s

        base = snapshot()
        self._cf_override = {(entity, relation): new_value}
        try:
            cf = snapshot()
        finally:
            self._cf_override = None
        changes = [{'what': k, 'was': base[k], 'now': cf[k]}
                   for k in base if base[k] != cf[k]]
        return {'baseline': base,
                'intervention': f'{entity}.{relation} = {new_value}',
                'counterfactual': cf, 'changes': changes}

    def composed_atom(self, r1, r2, x):
        """Apply a COMPOSED relation R1 o R2 to x: R1(R2(x)) -- resolve the inner
        hop, then the outer, by derive-chaining over stored atoms (the sanctioned
        substrate op, same as transitive_reach).  Read-only.  None if either hop
        is undefined."""
        y = self.atom(r2, x) or self.inherited_atom(r2, x)
        if not y:
            return None
        return self.atom(r1, y) or self.inherited_atom(r1, y)

    def composed_key(self, r1, r2):
        """The invented relation's IDENTITY in the phasor algebra:
        key(R1 o R2) = bind(key(R1), key(R2)) -- so a composed relation is a real
        object in the substrate, not just a python label."""
        from ikigai.cognition.phasor_state import bind
        ck = self.gr.org.unified.ck
        return bind(ck.key(str(r1)), ck.key(str(r2)))

    def invent_relations(self, min_support=2, max_new=8):
        """Day-87 GOLD -- RELATION INVENTION.  The organism grows its own
        conceptual MACHINERY, not just facts: it COMPOSES two relations it
        already has into a NEW named relation R3 = R1 o R2 (R3(x) = R1(R2(x))),
        discovers which compositions are USEFUL, and can then derive facts that
        no single stored relation could answer (nationality = country-of-birthplace,
        never stored, always derivable).

        DISCOVERY is store-structure mining -- the same honest meta-cognition the
        RuleMiner does -- ranking ordered relation pairs by how many entities the
        composition resolves for, and KEEPING only those whose (x -> z) reach is
        NOT already provided by any single stored relation (genuinely new machinery,
        not a rename).  The invented relation is given a real identity in the phasor
        algebra (key = bind(key R1, key R2)); its APPLICATION is derive-chaining,
        the sanctioned substrate op.  Recorded as learned 'composition' rules on
        the organism.  Returns [{name, r1, r2, support, examples}], best first."""
        rels = sorted(self.relations)
        # a stored (subject -> value) pair under ANY single relation -- the reach
        # the organism ALREADY has; a composition is only useful if it adds to it.
        stored_pairs = {(s, v) for (s, _r), v in self.triples.items() if v}
        # per-relation subject->value index (values may themselves be entities)
        idx = {}
        for r in rels:
            idx[r] = {s: v for (s, rr), v in self.triples.items() if rr == r and v}
        found = []
        for r2 in rels:                        # inner hop (applied first)
            for r1 in rels:                    # outer hop
                if r1 == r2:
                    continue                   # same-relation composition is transitivity (already mined)
                support, examples = 0, []
                for x, y in idx[r2].items():
                    z = idx[r1].get(y)
                    if z and (x, z) not in stored_pairs:   # genuinely NEW reach
                        support += 1
                        if len(examples) < 3:
                            examples.append((x, z))
                if support >= min_support:
                    found.append({'name': f'{r1}-of-{r2}', 'r1': r1, 'r2': r2,
                                  'support': support, 'examples': examples})
        found.sort(key=lambda d: -d['support'])
        found = found[:max_new]
        # record as learned composition rules (so they persist + are enumerable)
        existing = {(r.get('r1'), r.get('r2')) for r in self.learned_rules
                    if r.get('type') == 'composition'}
        for f in found:
            if (f['r1'], f['r2']) not in existing:
                self.learned_rules.append({'type': 'composition', 'name': f['name'],
                                           'r1': f['r1'], 'r2': f['r2'],
                                           'support': f['support']})
        return found

    def invented_relations(self):
        """List the composition rules the organism has invented."""
        return [r for r in self.learned_rules if r.get('type') == 'composition']

    def derive_invented(self, name, x):
        """Derive a named invented relation for x by chaining its two hops."""
        for r in self.learned_rules:
            if r.get('type') == 'composition' and r.get('name') == name:
                return self.composed_atom(r['r1'], r['r2'], x)
        return None

    def induce_concept(self, examples, thresh=0.5):
        """Day-85 GOLD -- INVENT a concept by anti-unification IN THE SUBSTRATE.
        A property belongs to the concept iff its bind(key(rel),key(val))
        RESONATES with EVERY example's holographic signature -- the shared
        structure surfaces by binding + resonance, decided geometrically, NOT by
        a python set-intersection.  (The atom store only lists each example's
        candidate facts; the reasoning -- 'is this common?' -- is the cosine.)
        Count-robust: the resonance is rescaled by sqrt(n_props) so a property's
        presence does not wash out in a large bundle.  Returns
        (schema {rel: val}, concept_hv).  No hardcoding, no curated features."""
        from ikigai.cognition.phasor_state import bind, cosine
        import math
        ck = self.gr.org.unified.ck
        sigs, cand = [], {}
        for e in examples:
            sig, props, n = self.entity_signature(e)
            if sig is None:
                return {}, None
            sigs.append((sig, max(1, n)))
            for (r, v) in props:
                cand[(r, v)] = bind(ck.key(r), ck.key(v))
        schema, concept_hv = {}, None
        for (r, v), b in cand.items():
            # present in EVERY example by resonance (norm-rescaled, count-robust)
            if all(cosine(b, sig) * math.sqrt(n) >= thresh for sig, n in sigs):
                schema[r] = v
                concept_hv = b if concept_hv is None else concept_hv + b
        return schema, concept_hv

    def concept_member(self, entity, schema, thresh=0.5):
        """Day-85 -- CLASSIFY an entity into an invented concept by SUBSTRATE
        RESONANCE: it belongs iff every concept-property binding resonates with
        the entity's own holographic signature (norm-rescaled).  Geometric
        membership, not a dict lookup -- the substrate decides.  Returns
        (is_member, min_score)."""
        from ikigai.cognition.phasor_state import bind, cosine
        import math
        ck = self.gr.org.unified.ck
        sig, _props, n = self.entity_signature(entity)
        if sig is None or not schema:
            return False, 0.0
        rn = math.sqrt(max(1, n))
        scores = [cosine(bind(ck.key(r), ck.key(v)), sig) * rn for r, v in schema.items()]
        return (all(s >= thresh for s in scores), min(scores) if scores else 0.0)

    def has_inheritance_rule(self, attr, link):
        """Pack 326 -- True if a LEARNED inheritance rule (per-attr or wildcard)
        covers attr across link, regardless of require_learned. Used to apply
        the inheritance shortcut for ANY link, not just the authored 'capital'
        path -- so a self-compressed KG (where attr(link(x)) was deleted) still
        answers via attr(x)."""
        attr = self._norm_rel(attr)
        for r in self.learned_rules:
            if r.get('type') == 'inheritance' and r.get('link') == link:
                if r.get('attr') == '*' or self._norm_rel(r.get('attr')) == attr:
                    return True
        return False

    def sanctioned_inverse(self, rel):
        """Return the relation `rel` is the inverse OF, if a learned
        inverse rule covers it (305.1).  e.g. learned {rel:'country',
        inv:'capital'} -> sanctioned_inverse('country') == 'capital'
        (country(x) reachable by reverse-lookup of capital).  None when
        require_learned and no rule; the rel name itself when off."""
        for r in self.learned_rules:
            if r.get('type') == 'inverse':
                if r.get('rel') == rel:
                    return r.get('inv')
                if r.get('inv') == rel:
                    return r.get('rel')
        return None

    def atom_del(self, rel, entity):
        """Drop a now-redundant atom (the organism learned a rule that
        derives it) from BOTH the cache and the enumerable index.  Returns
        the number of cache entries removed.  Used for self-compression
        after a rule is promoted."""
        from ikigai.cognition.cat4_absorb import _stable_anchor
        cat4 = getattr(self.gr.org, 'cat4', None)
        if cat4 is None or not getattr(cat4, 'anchor_actions', None):
            return 0
        ent = str(entity).strip().lower()
        removed = 0
        for tmpl in self._templates_for(rel):
            a = _stable_anchor(self.gr.tokenize(tmpl.format(e=ent)))
            if a in cat4.anchor_actions:
                del cat4.anchor_actions[a]
                removed += 1
        self.triples.pop((ent, rel), None)
        if removed:
            # invalidate cat4 recall caches that snapshot anchors
            for attr in ('_pack280_recall_states', '_pack280_recall_bounds',
                         '_pack280_recall_anchors', '_pack272_cb_vocab',
                         '_pack272_cb_K'):
                if hasattr(cat4, attr):
                    setattr(cat4, attr, None)
        return removed

    # ---- Pack 305.1 substrate-native HV rule store ------------------
    # Each learned rule lives in the body as a bound phasor HV
    # (role(type) (x) role(field0) (x) role(field1)), not just a Python
    # dict.  Application matches a query's (type, fields) HV against the
    # rule bank by cosine cleanup; the dict is the fast index + audit
    # trail.  hv_rules=True routes sanctioning through the substrate match.

    hv_rules = True

    def _rule_dim(self):
        return int(getattr(getattr(self.gr, 'mr', None), 'd', 0) or 400)

    def _role_hv(self, token):
        """Deterministic d-dim unit-phasor role HV for a token
        (blake2b-seeded, reproducible across processes)."""
        d = self._rule_dim()
        seed = int.from_bytes(
            hashlib.blake2b(str(token).encode('utf-8'),
                            digest_size=8).digest(), 'big')
        rng = np.random.default_rng(seed)
        return np.exp(1j * rng.uniform(-np.pi, np.pi, d)).astype(np.complex64)

    @staticmethod
    def _rule_fields(rule):
        """Canonical (type, field0, field1) component tokens for a rule."""
        t = rule.get('type')
        if t == 'inheritance':
            return ('inheritance', f'attr:{rule.get("attr")}',
                    f'link:{rule.get("link")}')
        if t == 'inverse':
            return ('inverse', f'rel:{rule.get("rel")}',
                    f'inv:{rule.get("inv")}')
        if t == 'synonymy':
            return ('synonymy', f'a:{rule.get("a")}', f'b:{rule.get("b")}')
        return (str(t), '', '')

    def _rule_hv(self, fields):
        """Bind component role HVs into one rule HV (FHRR Hadamard)."""
        hv = np.ones(self._rule_dim(), dtype=np.complex64)
        for f in fields:
            if f:
                hv = hv * self._role_hv(f)
        return hv.astype(np.complex64)

    def rebuild_rule_bank(self):
        """Encode every learned rule to its substrate HV.  Stores
        (fields, hv) so the bank can be matched and round-tripped."""
        self._rule_bank = []
        for r in self.learned_rules:
            fields = self._rule_fields(r)
            self._rule_bank.append((fields, r, self._rule_hv(fields)))
        if self._rule_bank:
            self._rule_bank_mat = np.stack([h for _, _, h in self._rule_bank])
        else:
            self._rule_bank_mat = np.zeros((0, self._rule_dim()),
                                           dtype=np.complex64)
        return len(self._rule_bank)

    def rule_match(self, fields, thresh=0.99):
        """Match a (type, field0, field1) query against the substrate rule
        bank by cosine cleanup.  Returns (rule, sim) or (None, sim)."""
        bank = getattr(self, '_rule_bank', None)
        if bank is None:
            self.rebuild_rule_bank()
            bank = self._rule_bank
        if not bank:
            return None, 0.0
        q = self._rule_hv(fields)
        sims = np.real(self._rule_bank_mat.conj() @ q) / q.shape[0]
        i = int(np.argmax(sims))
        return (bank[i][1] if sims[i] >= thresh else None), float(sims[i])

    # ---- learned-rule application -----------------------------------

    def sanctioned_inheritance(self, attr, link='capital'):
        """True if a LEARNED rule sanctions attr(link(x)) == attr(x).
        When require_learned is False, authored inheritance is always
        allowed (304 behaviour).  When hv_rules is on, the check goes
        through the substrate HV rule bank (Pack 305.1)."""
        if not self.require_learned:
            return True
        attr = self._norm_rel(attr)
        # Pack 317 wildcard: a single learned schema attr='*' for this link
        # sanctions inheritance of EVERY attribute (the generalisation "a
        # capital inherits ALL its country's attributes"), so the organism
        # need not learn one rule per attr. Checked first; falls back to the
        # per-attr rule otherwise.
        for r in self.learned_rules:
            if (r.get('type') == 'inheritance' and r.get('attr') == '*'
                    and r.get('link') == link):
                return True
        if self.hv_rules:
            rule, _ = self.rule_match(
                ('inheritance', f'attr:{attr}', f'link:{link}'))
            return rule is not None
        for r in self.learned_rules:
            if (r.get('type') == 'inheritance'
                    and self._norm_rel(r.get('attr')) == attr
                    and r.get('link') == link):
                return True
        return False

    def promote_wildcard_inheritance(self, min_attrs=2):
        """Pack 317 -- anti-unify per-attr inheritance rules into ONE wildcard
        schema. When >= min_attrs distinct attributes are each learned to
        inherit across the SAME link, promote {type:'inheritance', attr:'*',
        link} -- the organism generalises 'this attr inherits' to 'ALL attrs
        inherit'. Idempotent; rebuilds the HV bank + snapshots state. Returns
        the newly added wildcard rules."""
        by_link = {}
        for r in self.learned_rules:
            if r.get('type') == 'inheritance' and r.get('attr') != '*':
                by_link.setdefault(r.get('link'), set()).add(
                    self._norm_rel(r.get('attr')))
        existing = {r.get('link') for r in self.learned_rules
                    if r.get('type') == 'inheritance' and r.get('attr') == '*'}
        added = []
        for link, attrs in by_link.items():
            if len(attrs) >= int(min_attrs) and link not in existing:
                rule = {'type': 'inheritance', 'attr': '*', 'link': link}
                self.learned_rules.append(rule)
                added.append(rule)
        if added:
            self.rebuild_rule_bank()
            self.save_state()
        return added

    # ---- native autonomous discovery (Pack 305.1) -------------------

    def discover(self, min_support=6, min_conf=0.7, self_compress=False,
                 verbose=False):
        """Mine composition rules from the organism's OWN atom index and
        promote them -- no external entity/relation lists (link vs attr
        auto-classified from store structure).  Merges into learned_rules
        (dedup), rebuilds the HV bank, snapshots state.  Returns the list
        of newly added rules.  This is the native, autonomous form of the
        Pack 305 gate -- the organism discovers its own rules."""
        from ikigai.cognition.rule_discovery import RuleMiner
        miner = RuleMiner(self)
        ents = sorted(self.entities)
        link_rels, attr_rels = RuleMiner.classify_relations(
            self.triples, self.relations, self.entities)
        # Scale guard: the pair-miners (synonymy/inverse/inheritance) are
        # O(rels^2 * entities).  A relation that appears on fewer than
        # min_support subjects can never reach min_support in any rule, so it
        # only inflates the quadratic.  Drop the low-support tail up front --
        # on raw Wikidata truthy this collapses thousands of junk predicates
        # (one-off external IDs, sitelinks) to the dense structural backbone,
        # turning an intractable mine into an O(real_rels^2 * E) one.
        rel_support = {}
        for (s, r) in self.triples:
            rel_support[r] = rel_support.get(r, 0) + 1
        keep = {r for r, c in rel_support.items() if c >= min_support}
        link_rels = [r for r in link_rels if r in keep]
        attr_rels = [r for r in attr_rels if r in keep]
        found = miner.mine_all(ents, link_rels, attr_rels,
                               min_support=min_support, min_conf=min_conf,
                               verbose=verbose)
        existing = {self._rule_fields(r) for r in self.learned_rules}
        added = [r for r in found if self._rule_fields(r) not in existing]
        self.learned_rules.extend(added)
        self.rebuild_rule_bank()
        self.save_state()
        if self_compress and added:
            self._self_compress(added)
        return added

    def _self_compress(self, rules):
        """Drop atoms a newly-learned rule now derives -- LOSSLESSLY.

        Pack 323 safety: a rule with conf < 1.0 (a real-world pattern with
        EXCEPTIONS) is safe to promote, because self-compression deletes ONLY
        the atoms the rule reproduces EXACTLY.  An exception -- a linked entity
        whose attribute differs from the inherited value -- is KEPT, so a
        direct query of it still returns the true stored fact and the rule
        never overwrites ground truth.  This is what makes rule discovery safe
        on noisy data: a spurious or approximate rule cannot delete a fact it
        gets wrong.
        """
        removed = 0
        for r in rules:
            if r.get('type') != 'inheritance':
                continue
            attr, link = r.get('attr'), r.get('link')
            if attr == '*':          # wildcard schema is not a per-attr compressor
                continue
            if self.is_transitive(link):
                # TAXONOMIC inheritance (attr inherits DOWN a transitive link such
                # as subclassof): the DESCENDANT copy is redundant, derivable by
                # climbing to the ancestor that defines it -- so drop attr(x) when
                # the nearest ancestor carrying attr holds the SAME value, keeping
                # the ancestor source and any exception (a descendant whose value
                # differs).  Decisions use a snapshot so the climb is not perturbed
                # by deletions; inherited_atom re-derives every dropped descendant.
                snap = {x: self.triples.get((x, attr)) for x in self.entities}
                for x in sorted(self.entities):
                    xv = snap.get(x)
                    if xv is None:
                        continue
                    chain = self.transitive_reach(link, x) or [x]
                    for anc in chain[1:]:
                        av = snap.get(anc)
                        if av is not None:
                            if av == xv:
                                removed += self.atom_del(attr, x)
                            break          # nearest ancestor with attr decides
            else:
                for x in sorted(self.entities):
                    mid = self.triples.get((x, link))        # link(x)
                    base = self.triples.get((x, attr))       # inherited value attr(x)
                    outer = self.triples.get((mid, attr)) if mid else None
                    # delete attr(link(x)) ONLY if the rule reproduces it exactly
                    if mid and outer is not None and base is not None \
                            and outer == base:
                        removed += self.atom_del(attr, mid)
        return removed

    # ---- derivation -------------------------------------------------

    def derive(self, text):
        """Answer a composite query by composing atoms -- NEVER writes the
        cache.  Returns (answer, 'derive') or None when the template does
        not match or a required atom is missing.
        """
        self._stats['derive_calls'] += 1
        t = text or ''

        # continent/region of the capital of X  ==  continent/region of X
        # (inheritance -- gated on a LEARNED rule when require_learned)
        m = self._CAP_CONT_WHICH.match(t)
        if m:
            rel, country = m.group(1).lower(), m.group(2).strip()
            if not self.sanctioned_inheritance(rel, 'capital'):
                return None
            val = self.atom(rel, country)
            return self._ok(val)

        # is the capital of X in <target>  ==  (continent(X) == target)
        m = self._CAP_CONT_YESNO.match(t)
        if m:
            country = m.group(1).strip()
            target = m.group(2).strip().lower().rstrip('?').strip()
            if not (self.sanctioned_inheritance('continent', 'capital')
                    or self.sanctioned_inheritance('region', 'capital')):
                return None
            cont = self.atom('continent', country) or self.atom('region', country)
            if cont is None:
                return None
            yes = (target in cont) or (cont in target)
            return self._ok('yes' if yes else 'no')

        # same-<attr> comparison over ANY comparable relation (continent,
        # region, currency, language, ...)  ==  attr(X) == attr(Y)
        m = self._SAME.match(t)
        if m:
            a, b, rel = m.group(1).strip(), m.group(2).strip(), m.group(3)
            return self._same(a, b, rel)
        m = self._SAME_AS.match(t)
        if m:
            a, rel, b = m.group(1).strip(), m.group(2), m.group(3).strip()
            return self._same(a, b, rel)

        # numeric comparison  ==  num(attr(X)) vs num(attr(Y))
        m = self._BIGGER.match(t)
        if m:
            a, cmp_, b = m.group(1).strip(), m.group(2).lower(), m.group(3).strip()
            return self._numcmp(a, b, 'population', cmp_)
        m = self._BIGGER_ATTR.match(t)
        if m:
            a, cmp_, rel, b = (m.group(1).strip(), m.group(2).lower(),
                               m.group(3), m.group(4).strip())
            return self._numcmp(a, b, rel, cmp_)
        m = self._WHICH_BIGGER.match(t)
        if m:
            cmp_, a, b = m.group(1).lower(), m.group(2).strip(), m.group(3).strip()
            return self._numcmp(a, b, 'population', cmp_, answer='name')

        # Pack 317 arbitrary N-hop (>=3): "the r1 of the r2 of the r3 of X".
        # Resolved innermost-out via _chain_inner. Handled before _CHAIN2 so
        # deep chains compose; 2-hop still falls to _CHAIN2 below (preserving
        # its capital-inheritance carve-out).
        parsed = self._parse_chain(t)
        if parsed and len(parsed[0]) >= 3:
            # 3+ hops: the 2-hop inheritance regexes don't apply, so resolve
            # the full chain innermost-out (each hop atom or learned inverse).
            rels, x = parsed
            val = self._chain_resolve(rels, x)
            if val:
                return self._ok(val)

        # Pack 304.2 generic chain: "what is the <rel1> of the <rel2> of X"
        # = rel1(rel2(X)), inner via direct atom or a LEARNED inverse rule.
        # Checked before _CHAIN so non-capital rel2 (country/...) compose;
        # the capital-inheritance _CHAIN below still handles rel2='capital'.
        m = self._CHAIN2.match(t)
        if m:
            rel1, rel2, x = (self._norm_rel(m.group(1)),
                             self._norm_rel(m.group(2)), m.group(3).strip())
            # Pack 326: generalized inheritance across ANY learned link. If a
            # learned rule says rel1 inherits across rel2, the answer is the
            # base atom rel1(x) -- even when rel1(rel2(x)) was self-compressed
            # away. Removes the 'capital'-specific hardcoding for the learned
            # case; the authored capital path (_CHAIN below) still serves when
            # no rule is learned.
            if self.has_inheritance_rule(rel1, rel2):
                val = self.atom(rel1, x)
                if val:
                    return self._ok(val)
            if rel2 != 'capital':
                inner = self._chain_inner(rel2, x)
                if inner:
                    val = self.atom(rel1, inner)
                    if val:
                        return self._ok(val)

        # what is the <attr> of the capital of X  ==  <attr>(X)
        # (inheritance -- gated on a LEARNED rule when require_learned)
        m = self._CHAIN.match(t)
        if m and self.sanctioned_inheritance(self._norm_rel(m.group(1)), 'capital'):
            rel, country = self._norm_rel(m.group(1)), m.group(2).strip()
            if rel in _REL_TEMPLATES:
                val = self.atom(rel, country)
                return self._ok(val)
        return None

    # ---- attribute normalisation (relation algebra, not facts) -------
    _REL_ALIASES = {
        'continent': 'continent', 'continents': 'continent',
        'region': 'region', 'regions': 'region',
        'currency': 'currency', 'currencies': 'currency', 'money': 'currency',
        'language': 'language', 'languages': 'language', 'tongue': 'language',
        'population': 'population', 'people': 'population', 'size': 'population',
        'capital': 'capital', 'capitals': 'capital',
    }

    def _norm_rel(self, word):
        return self._REL_ALIASES.get(str(word).strip().lower(), str(word).strip().lower())

    def _claim(self):
        """A comparison template matched but a required atom is missing.
        CLAIM the query as derive ('unknown') so it never falls through to
        active learning -- a compositional comparison is NOT an atomic fact
        to teach, and letting the teacher absorb it would pollute the cache
        (breaking derive-not-store).  Not counted as a real derivation."""
        return ('unknown', 'derive')

    def _same(self, a, b, rel):
        rel = self._norm_rel(rel)
        va = self.atom(rel, a)
        vb = self.atom(rel, b)
        if va is None or vb is None:
            return self._claim()
        return self._ok('yes' if va == vb else 'no')

    @staticmethod
    def _to_num(val):
        """Parse a numeric atom value ('67', '1.4 billion') to a float."""
        if val is None:
            return None
        s = str(val).lower().replace(',', '')
        mult = 1.0
        if 'billion' in s:
            mult = 1e9
        elif 'million' in s:
            mult = 1e6
        m = re.search(r'-?\d+(?:\.\d+)?', s)
        return float(m.group(0)) * mult if m else None

    def _numcmp(self, a, b, rel, cmp_, answer='yesno'):
        rel = self._norm_rel(rel)
        na = self._to_num(self.atom(rel, a))
        nb = self._to_num(self.atom(rel, b))
        if na is None or nb is None:
            return self._claim()
        bigger = cmp_ in ('bigger', 'larger', 'higher', 'greater')
        if answer == 'name':
            return self._ok(a.lower() if (na > nb) == bigger else b.lower())
        cond = (na > nb) if bigger else (na < nb)
        return self._ok('yes' if cond else 'no')

    def _ok(self, val):
        if val is None:
            return None
        self._stats['derived'] += 1
        return (val, 'derive')
