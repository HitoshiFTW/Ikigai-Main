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
        self.triples = {}        # (subj, rel) -> val   (enumerable atoms)
        self.entities = set()    # discovered subjects
        self.relations = set()   # discovered relations
        self.learned_rules = []  # promoted derivation rules (dicts)
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
        self.triples[(subj, rel)] = val
        self.entities.add(subj)
        self.relations.add(rel)

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
        if cat4 is None or not getattr(cat4, 'anchor_actions', None):
            return None
        ent = str(entity).strip().lower()
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
        if cat4 is None or not getattr(cat4, 'anchor_actions', None):
            return []
        ent = str(entity).strip().lower()
        out = []
        for tmpl in self._templates_for(rel):
            entry = cat4.anchor_actions.get(
                _stable_anchor(self.gr.tokenize(tmpl.format(e=ent))))
            if entry:
                for toks in entry:
                    v = ' '.join(toks).strip().lower()
                    if v and v not in out:
                        out.append(v)
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
            max_depth = len(self.entities) + 2     # longest acyclic chain
        chain = [str(x).strip().lower()]
        seen = {chain[0]}
        cur = chain[0]
        for _ in range(int(max_depth)):
            nxt = self._chain_inner(rel, cur)
            if not nxt or nxt in seen:      # stop at root or cycle (converged)
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
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
            for x in sorted(self.entities):
                mid = self.triples.get((x, link))            # link(x)
                base = self.triples.get((x, attr))           # inherited value attr(x)
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
