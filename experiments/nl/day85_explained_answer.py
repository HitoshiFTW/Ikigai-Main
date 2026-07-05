"""Day 85 -- the stacked differentiator: a grounded answer WITH its verifiable
derivation.  answer(q, explain=True) routes through the proof-carrying path, so
the answer is emitted only if its chain re-derives + verifies, and attaches a
grounded 'because' (each hop = premise relation conclusion, every token from the
chain).  Faithful + transparent + verifiable -- what a trained LM cannot offer.
Alpha-only names; no hardcoding, no real data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import integrate

org = integrate.IkigaiOrganism()
edges = [('dog', 'isa', 'mammal'), ('mammal', 'isa', 'animal'), ('animal', 'isa', 'organism'),
         ('animal', 'breathes', 'air')]
edges += [('dog', 'breathes', 'air'), ('mammal', 'breathes', 'air')]   # redundancy -> inheritance
org.ingest_triples(edges, discover=True, min_support=2, min_conf=0.8, self_compress=True)

print('=== explained multi-hop chain ===')
r = org.answer('what is the isa of the isa of dog', explain=True)
print(f'  text     : {r["text"]}')
print(f'  because  : {r["because"]}')
print(f'  verified : {r["verified"]}   grounded : {r["grounded"]}')
multi_ok = (r['text'] == 'dog isa animal' and r['verified'] and r['grounded']
            and 'dog isa mammal' in r['because'] and 'mammal isa animal' in r['because'])

print('\n=== explained single (inherited) ===')
r2 = org.answer('what does dog breathes', explain=True)
print(f'  text {r2["text"]!r} because {r2["because"]!r} verified={r2["verified"]} grounded={r2["grounded"]}')
single_ok = (r2['fact'] is not None and r2['verified'] and r2['grounded'])

print('\n=== unknown -> abstain (no hallucination, even with proof asked) ===')
r3 = org.answer('what is zorblax', explain=True)
abstain_ok = (r3['text'] == "i don't know" and r3['fact'] is None and not r3['verified'])
print(f'  {r3["text"]!r}  abstained={abstain_ok}')

print(f'\nmulti {multi_ok} · single {single_ok} · abstain {abstain_ok}')
assert multi_ok and single_ok and abstain_ok, 'explained answer regressed'
print('GATE GREEN -- faithful + transparent + verifiable answer.')
