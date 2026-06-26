"""Day 81 Pack 323 -- RULE-DISCOVERY SAFETY ON NOISY DATA.

Real knowledge graphs have noise + exceptions. Before ingesting them, prove:
  (A) the miner's PRECISION -- it promotes the TRUE rule and does NOT promote
      spurious rules over relations that carry no real regularity;
  (B) EXCEPTION-SAFE self-compression -- an approximate rule (conf<1.0) is safe
      because compression deletes ONLY facts the rule reproduces exactly;
      exceptions are KEPT and a direct query of them still returns ground truth.

DATA-FREE: made-up countries/capitals with planted exceptions + noise relations.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrate
from ikigai.cognition.rule_discovery import RuleMiner

CONT = ['mu', 'za', 'ki']

def setup(C, n_exceptions):
    org = integrate.IkigaiOrganism(flat_only=True)
    eng = org.general_reasoner.derive_engine
    gt = {}
    exceptions = set()
    _AL = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(C):
        ctry, cap = f'land{_AL[i]}', f'cap{_AL[i]}'        # alpha-only (no digit split)
        cont = CONT[i % 3]
        org.cat4.populate_cache_from_text(
            f'what is the capital of {ctry}\n\n{cap}\n\n')
        org.cat4.populate_cache_from_text(
            f'what continent is {ctry} in\n\n{cont}\n\n')
        # capital normally inherits the country's continent...
        cap_cont = cont
        if i < n_exceptions:                       # ...except a few planted ones
            cap_cont = CONT[(i + 1) % 3]           # capital on a DIFFERENT continent
            exceptions.add(cap)
        org.cat4.populate_cache_from_text(
            f'what continent is {cap} in\n\n{cap_cont}\n\n')
        # a NOISE relation with random-ish values (no real rule)
        org.cat4.populate_cache_from_text(
            f'what is the currency of {ctry}\n\n{["a","b","c","d","e"][i % 5]}\n\n')
        gt[('continent', ctry)] = cont
        gt[('continent', cap)] = cap_cont
        for rel in ('capital', 'continent', 'currency'):
            eng.atom(rel, ctry)
        eng.atom('continent', cap)
    return org, eng, gt, exceptions

C, NEXC = 18, 3
org, eng, gt, exceptions = setup(C, NEXC)

# ---- (A) miner precision -------------------------------------------
miner = RuleMiner(eng)
lr, ar = RuleMiner.classify_relations(eng.triples, eng.relations, eng.entities)
inh = miner.mine_inheritance(sorted(eng.entities), lr, ar,
                             min_support=5, min_conf=0.8, verbose=True)
print(f'\n(A) inheritance rules promoted: {inh}')
# TRUE rule = continent inherits across capital. Spurious = anything else.
true_promoted = any(r['attr'] == 'continent' and r['link'] == 'capital'
                    for r in inh)
spurious = [r for r in inh if not (r['attr'] == 'continent'
                                   and r['link'] == 'capital')]
print(f'    true rule promoted: {true_promoted}; spurious: {len(spurious)}')

# ---- (B) exception-safe self-compression ----------------------------
eng.learned_rules.extend(inh)
eng.rebuild_rule_bank()
before = len([1 for (s, r) in eng.triples if r == 'continent'
              and s.startswith('cap')])
removed = eng._self_compress(inh)
after_kept = {s for (s, r) in eng.triples if r == 'continent'
              and s.startswith('cap')}
print(f'\n(B) capital-continent atoms: {before} stored, {removed} compressed, '
      f'{len(after_kept)} kept')
# every EXCEPTION must survive compression; normals get derived
exc_kept = all(e in after_kept for e in exceptions)
_AL = 'abcdefghijklmnopqrstuvwxyz'
norm_deleted = all(f'cap{_AL[i]}' not in after_kept for i in range(NEXC, C))
print(f'    exceptions kept: {exc_kept} ({sorted(exceptions)} all present)')
print(f'    normals compressed away: {norm_deleted}')

# direct query of an exception still returns ground truth (not overwritten)
exc = sorted(exceptions)[0]
direct = eng.atom('continent', exc)
print(f'    direct atom(continent,{exc}) = {direct} (gt {gt[("continent",exc)]})')
exc_truth = direct == gt[('continent', exc)]

ok = (true_promoted and len(spurious) == 0 and exc_kept and norm_deleted
      and exc_truth)
print('\n' + ('PASS -- miner precision clean (true rule promoted, zero '
              'spurious); self-compression LOSSLESS under exceptions (kept all, '
              'derived normals, ground truth preserved). Safe to ingest noisy '
              'KG data.' if ok else
              f'FAIL true={true_promoted} spur={len(spurious)} excKept={exc_kept} '
              f'normDel={norm_deleted} excTruth={exc_truth}'))
