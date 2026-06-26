"""
ikigai.cognition.rule_discovery -- Pack 305 Rule Discovery v0.

Stop AUTHORING composition rules.  Mine relational regularities from the
organism's own atom store, validate by support+confidence, PROMOTE the
confirmed ones to learned derivation rules.  The 304/304.1 rules that were
hand-written become DISCOVERED; new rules appear with zero code; and the
organism can DROP atoms it can now derive (self-compression).

This is inductive logic programming (ILP) over the atom store: we supply
the rule SCHEMAS (the shapes to search); the organism discovers WHICH
concrete rules hold within them.  Inventing new schemas is out of scope
(AGI-hard) -- learning which rules hold in a schema is genuine rule
learning, and we state that boundary honestly.

A learned rule is a plain dict (so it serialises with organism.ikg):

  inheritance:
    {'type':'inheritance', 'attr':<r>, 'link':<r_link>,
     'support':int, 'conf':float}
    meaning: r(link(x)) == r(x)  -- the linked entity inherits x's attr,
    so r(link(x)) is DERIVABLE from r(x) (drop the r(link(x)) atoms).

  synonymy:
    {'type':'synonymy', 'a':<r_a>, 'b':<r_b>, 'support':int, 'conf':float}
    meaning: r_a(x) == r_b(x) for all x  -- the two relations coincide.
"""


class RuleMiner:
    """Mines learned derivation rules from a CompositionEngine's atoms.

    Read-only over the atom store (atom() lookups only); never writes the
    cache.  Returns a list of promoted-rule dicts.
    """

    def __init__(self, engine):
        self.eng = engine
        self.stats = {'candidates': 0, 'promoted': 0}

    # ---- inheritance: r(link(x)) == r(x) ----------------------------

    def mine_inheritance(self, entities, link_rels, attr_rels,
                          min_support=10, min_conf=0.9, verbose=False):
        """For each (link_rel, attr_rel), test whether the value of
        attr_rel on link_rel(x) equals attr_rel on x across `entities`.

        e.g. link=capital, attr=continent:
            for country x, cap = capital(x); does continent(cap) ==
            continent(x)?  If it holds for >= min_support entities with
            >= min_conf confidence, promote the rule -- continent(cap) is
            then derivable from continent(x), so the continent(cap) atoms
            are redundant.
        """
        rules = []
        for link in link_rels:
            for attr in attr_rels:
                if attr == link:
                    continue
                self.stats['candidates'] += 1
                support = match = 0
                for x in entities:
                    link_val = self.eng.atom(link, x)
                    if not link_val:
                        continue
                    outer = self.eng.atom(attr, link_val)   # attr of linked entity
                    base = self.eng.atom(attr, x)           # attr of x
                    if outer is None or base is None:
                        continue
                    support += 1
                    match += (outer == base)
                conf = match / support if support else 0.0
                if verbose:
                    print(f'    [mine] {attr} o {link}: support={support} '
                          f'match={match} conf={conf:.2f}')
                if support < min_support:
                    continue
                if conf >= min_conf:
                    rules.append({'type': 'inheritance', 'attr': attr,
                                  'link': link, 'support': support,
                                  'conf': round(conf, 4)})
                    self.stats['promoted'] += 1
        return rules

    # ---- synonymy: r_a(x) == r_b(x) ---------------------------------

    def mine_synonymy(self, entities, rels, min_support=10, min_conf=0.95):
        """Test whether two relations carry the same value on every
        entity (e.g. 'continent' and 'region' taught identically)."""
        rules = []
        rels = list(rels)
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                ra, rb = rels[i], rels[j]
                self.stats['candidates'] += 1
                support = match = 0
                for x in entities:
                    va = self.eng.atom(ra, x)
                    vb = self.eng.atom(rb, x)
                    if va is None or vb is None:
                        continue
                    support += 1
                    match += (va == vb)
                if support < min_support:
                    continue
                conf = match / support
                if conf >= min_conf:
                    rules.append({'type': 'synonymy', 'a': ra, 'b': rb,
                                  'support': support, 'conf': round(conf, 4)})
                    self.stats['promoted'] += 1
        return rules

    # ---- inverse: r_inv(r(x)) == x  (round-trip) --------------------

    def mine_inverse(self, entities, rels, min_support=6, min_conf=0.85,
                     verbose=False):
        """Discover round-trip pairs: r_inv(r(x)) == x.  e.g. capital maps
        country->city and country_of maps city->country, so for every
        country x: country_of(capital(x)) == x.  Enables deriving one
        direction from the other (drop the redundant inverse atoms)."""
        rules = []
        rels = list(rels)
        for r in rels:
            for r_inv in rels:
                if r == r_inv:
                    continue
                self.stats['candidates'] += 1
                support = match = 0
                for x in entities:
                    mid = self.eng.atom(r, x)
                    if not mid:
                        continue
                    back = self.eng.atom(r_inv, mid)
                    if back is None:
                        continue
                    support += 1
                    match += (back == x)
                conf = match / support if support else 0.0
                if verbose and support:
                    print(f'    [mine] inverse {r_inv} o {r}: support={support} '
                          f'match={match} conf={conf:.2f}')
                if support >= min_support and conf >= min_conf:
                    rules.append({'type': 'inverse', 'rel': r, 'inv': r_inv,
                                  'support': support, 'conf': round(conf, 4)})
                    self.stats['promoted'] += 1
        return rules

    # ---- transitive: R(a)=b & R(b)=c  =>  R chains (a..c) -----------

    def mine_transitive(self, entities, link_rels, min_support=3,
                        min_conf=0.9, verbose=False):
        """Pack 317.2 -- discover TRANSITIVE link relations: R where R(a)=b
        and R(b)=c both hold (b is itself a subject of R), acyclically, for
        >= min_support entities.  Promotes {'type':'transitive','rel':R}.
        Sanctions N-hop reach over R (ancestor / closure) without storing the
        closure -- derive-not-store at the relation level.

        Validation is positive-only (single-valued atoms can't store the
        closure), so conf = fraction of observed 2-chains that are acyclic
        (c != a); a relation that immediately cycles is not transitive."""
        rules = []
        for R in link_rels:
            self.stats['candidates'] += 1
            chains = acyclic = 0
            for a in entities:
                b = self.eng.atom(R, a)
                if not b:
                    continue
                c = self.eng.atom(R, b)
                if not c:
                    continue
                chains += 1
                acyclic += (c != a)
            conf = acyclic / chains if chains else 0.0
            if verbose and chains:
                print(f'    [mine] transitive {R}: chains={chains} '
                      f'acyclic={acyclic} conf={conf:.2f}')
            if chains >= min_support and conf >= min_conf:
                rules.append({'type': 'transitive', 'rel': R,
                              'support': chains, 'conf': round(conf, 4)})
                self.stats['promoted'] += 1
        return rules

    # ---- auto link/attr classification (no hardcoded relation names) -

    @staticmethod
    def classify_relations(triples, relations, entities):
        """A relation is a LINK if its values are themselves entities
        (subjects in the store) -- e.g. capital(country)=city, and city is
        a subject.  Otherwise it is an ATTR (value is a leaf, e.g.
        continent).  Derived from store structure, not a curated list."""
        ent = set(entities)
        link, attr = set(), set()
        for key, v in triples.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            s, r = key
            if v in ent:
                link.add(r)
            else:
                attr.add(r)
        # a relation seen as both -> treat as link (it chains)
        attr -= link
        return sorted(link & set(relations)), sorted(attr & set(relations))

    # ---- full pass --------------------------------------------------

    def mine_all(self, entities, link_rels, attr_rels,
                 min_support=10, min_conf=0.9, verbose=False):
        rules = []
        rules += self.mine_inheritance(entities, link_rels, attr_rels,
                                       min_support, min_conf, verbose=verbose)
        rules += self.mine_synonymy(entities, attr_rels,
                                    min_support, max(min_conf, 0.95))
        rules += self.mine_inverse(entities, link_rels + attr_rels,
                                   min_support, max(min_conf, 0.85),
                                   verbose=verbose)
        rules += self.mine_transitive(entities, link_rels,
                                      max(3, min_support // 3),
                                      max(min_conf, 0.9), verbose=verbose)
        return rules
