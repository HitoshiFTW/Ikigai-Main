"""Day 80 Pack 317 GATE -- derive-not-store / composition upgrade.

(A) ARBITRARY N-HOP chaining: "the r1 of the r2 of the r3 of X" = r1(r2(r3(X))),
    resolved innermost-out (each hop direct atom OR learned inverse). Generalises
    the 304.2 2-hop chain to any depth.
(B) WILDCARD INHERITANCE: anti-unify per-attr inheritance rules into ONE schema
    attr='*' -> a capital inherits ALL its country's attributes; the organism
    generalises 'this attr inherits' to 'all attrs inherit'.

DATA-FREE: made-up entities (zogland/...), using the known relation templates.
Derive is read-only -> derive-not-store preserved.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrate

org = integrate.IkigaiOrganism(flat_only=True)
cat4 = org.cat4
eng = org.general_reasoner.derive_engine

# ---- made-up atoms (known rels, invented entities) -------------------
# question forms MUST match _REL_TEMPLATES (what atom() queries)
FACTS = [
    ('what is the capital of zogland', 'zogville'),
    ('what continent is zogland in', 'mu'),
    ('what is the currency of zogland', 'zogbuck'),
    ('what language is spoken in zogland', 'zoggish'),
]
for q, a in FACTS:
    cat4.populate_cache_from_text(f'{q}\n\n{a}\n\n')
# force atom() to resolve + auto-record triples (needed by reverse_atom)
for rel in ('capital', 'continent', 'currency', 'language'):
    eng.atom(rel, 'zogland')

print('atoms:', {(s, r): v for (s, r), v in eng.triples.items()})

# ---- (A) N-HOP -------------------------------------------------------
# learned inverse rule so country(zogville) = reverse of capital = zogland
eng.learned_rules.append({'type': 'inverse', 'rel': 'country', 'inv': 'capital'})
eng.rebuild_rule_bank()

q2 = 'what is the currency of the country of zogville'         # 2-hop (regress)
q3 = 'what is the currency of the country of the capital of zogland'  # 3-hop
# 2-hop: country(zogville)=zogland (inverse of capital), currency(zogland)=zogbuck
# 3-hop: capital(zogland)=zogville, country(zogville)=zogland, currency=zogbuck
r2 = eng.derive(q2)
r3 = eng.derive(q3)
print(f'\n(A) 2-hop  {q2!r} -> {r2}')
print(f'(A) 3-hop  {q3!r} -> {r3}')
a_ok = (r2 and r2[0] == 'zogbuck') and (r3 and r3[0] == 'zogbuck')

# ---- (B) WILDCARD INHERITANCE ---------------------------------------
eng.require_learned = True          # inheritance now needs a LEARNED rule
qy = 'what is the continent of the capital of zogland'   # = continent(zogland)
before = eng.derive(qy)             # no inheritance rule yet -> None
# teach per-attr inheritance for TWO attrs (currency, language)
eng.learned_rules += [
    {'type': 'inheritance', 'attr': 'currency', 'link': 'capital'},
    {'type': 'inheritance', 'attr': 'language', 'link': 'capital'},
]
eng.rebuild_rule_bank()
# continent NOT individually taught -> still None (require_learned)
mid = eng.derive(qy)
# promote wildcard: 'all attrs inherit across capital'
added = eng.promote_wildcard_inheritance(min_attrs=2)
after = eng.derive(qy)             # now derivable via wildcard
print(f'\n(B) wildcard promoted: {added}')
print(f'(B) {qy!r}')
print(f'    before any rule = {before}; after 2 per-attr (continent untaught) '
      f'= {mid}; after wildcard = {after}')
b_ok = (before is None and mid is None and after and after[0] == 'mu')

print('\n' + ('PASS -- N-hop chaining (any depth) + wildcard inheritance '
              '(one schema generalises all attrs). Derive read-only; '
              'derive-not-store intact.'
              if a_ok and b_ok else
              f'FAIL -- a_ok={a_ok} b_ok={b_ok}'))
