"""Day 81 Pack 326 -- KG INGESTION ADAPTER + device profile.

Proves a raw (subject, relation, object) dump -- with ARBITRARY predicates we
never hand-listed -- ingests through one call, round-trips, and feeds the
325 discover->self-compress loop. Plus a measured post-compression footprint.

DATA-FREE: a made-up animal KG with invented relations (habitat/diet/cls) so
nothing is curated; alpha-only names (digit tokenizer splits 'a1').
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrate

_AL = 'abcdefghijklmnopqrstuvwxyz'
def nm(i): return _AL[i // 26] + _AL[i % 26]

HAB = ['forest', 'ocean', 'desert', 'tundra']
DIET = ['herb', 'carn', 'omni']

def make_kg(C):
    """Species (primary) link to a genus via 'genus'; the dump ALSO stores the
    genus's habitat/diet redundantly (== the species', the inheritance the
    rule will discover). Arbitrary relation names, none in _REL_TEMPLATES.
    Returns triples, primary-gt, and redundant-gt mapping (r, genus)->(val, species)."""
    triples = []
    primary = {}; redundant = {}
    for i in range(C):
        sp, gen = f'sp{nm(i)}', f'gen{nm(i)}'
        hab, diet = HAB[i % 4], DIET[i % 3]
        triples += [
            (sp, 'genus', gen),
            (sp, 'habitat', hab),       # species = primary holder
            (sp, 'diet', diet),
            (gen, 'habitat', hab),      # redundant copy on the linked genus
            (gen, 'diet', diet),
        ]
        primary[('genus', sp)] = gen
        primary[('habitat', sp)] = hab; primary[('diet', sp)] = diet
        redundant[('habitat', gen)] = (hab, sp); redundant[('diet', gen)] = (diet, sp)
    return triples, primary, redundant

C = 40
triples, primary, redundant = make_kg(C)
gt = dict(primary); gt.update({k: v[0] for k, v in redundant.items()})
org = integrate.IkigaiOrganism(flat_only=True)

# --- (1) ingest arbitrary-predicate triples through ONE call ----------
res = org.ingest_triples(triples)
print(f'(1) ingested {res["ingested"]} triples ({len(triples)} given), '
      f'{res["atoms_before"]} atoms')

# --- (2) generic-relation round-trip (predicates never hand-listed) ---
eng = org.general_reasoner.derive_engine
ok = sum(eng.atom(r, e) == v for (r, e), v in gt.items())
print(f'(2) round-trip recall on arbitrary relations: {ok}/{len(gt)}')

# --- (3) discover + LOSSLESS self-compress the dump -------------------
eng.require_learned = True
res2 = org.ingest_triples([], discover=True, self_compress=True,
                          min_support=5, min_conf=0.75)
print(f'(3) discovered {res2["rules"]} rules, compressed {res2["compressed"]} '
      f'atoms ({res2["atoms_before"]} -> {res2["atoms_after"]})')

# coverage after compression: deleted genus-attrs derived via the chain
cov = 0
for (r, e), v in primary.items():
    cov += (eng.atom(r, e) == v)                 # primary atoms kept
for (r, gen), (v, sp) in redundant.items():
    got = eng.atom(r, gen)                        # may be deleted
    if got is None:
        gd = eng.derive(f'what is the {r} of the genus of {sp}')
        got = gd[0] if gd else None
    cov += (got == v)
print(f'    coverage after compress: {cov}/{len(gt)} answerable')

# --- (4) device footprint (measured) ---------------------------------
cat4 = org.cat4
n_cache = len(cat4.anchor_actions) if getattr(cat4, 'anchor_actions', None) else 0
B_PER = 7.0           # packed cache bytes/entry (Pack 298 measured, stored)
print(f'(4) post-compress cache: {n_cache} atoms x {B_PER:.0f} B = '
      f'{n_cache*B_PER/1024:.1f} KB  (+ 191 MB constant body)')

verdict = (res['ingested'] == len(triples) and ok == len(gt)
           and res2['compressed'] > 0 and cov == len(gt))
print('\n' + ('PASS -- arbitrary-predicate KG ingests through one call, '
              'round-trips with no curated templates, feeds the discover-> '
              'self-compress loop, stays fully answerable after compression. '
              'Drop in Wikidata/ConceptNet when it lands.'
              if verdict else
              f'FAIL ingest={res["ingested"]}/{len(triples)} rt={ok}/{len(gt)} '
              f'comp={res2["compressed"]} cov={cov}/{len(gt)}'))
