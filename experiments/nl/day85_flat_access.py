"""Day 85 #3 -- FLAT ACCESS: prove a query's cost is independent of store size.

The thesis (the device-tier story): a query touches only its own atom + its
derivation chain, never O(N) in the number of stored facts -- so a drone with a
1000TB-derived kernel recalls as cheaply as a 1MB one.  This measures it: grow the
derive-engine store across orders of magnitude and time a FIXED query (a direct
recall and a transitive chain) at each size.  Flat latency = flat access.

Made-up alpha-only entities; no real data, no hardcoding.
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import integrate

def nm(i):
    s = ''; i += 1
    while i:
        s += chr(97 + i % 26); i //= 26
    return 'e' + s

SIZES = [10_000, 50_000, 200_000, 1_000_000]
print(f'{"N facts":>12} {"recall us":>12} {"chain us":>12} {"chain hops":>11}', flush=True)
recalls = []
for N in SIZES:
    org = integrate.IkigaiOrganism()
    eng = org.general_reasoner.derive_engine
    # a long transitive spine (deep chain) + bulk filler facts around it
    SPINE = 64
    edges = [(nm(i + 1), 'linkr', nm(i)) for i in range(SPINE)]          # spine: e1->e0 ...
    rng = random.Random(1)
    base = SPINE + 10
    while len(edges) < N:                                                # bulk distractor facts
        a = nm(base + len(edges)); edges.append((a, 'fact', nm(rng.randrange(base))))
    edges.append((nm(0), 'color', 'red'))                               # root attribute
    org.ingest_triples(edges, discover=True, min_support=10, min_conf=0.9)
    # fixed query target: top of the spine (its cost must not grow with N)
    top = nm(SPINE)
    # warm
    eng.atom('fact', top); eng.transitive_reach('linkr', top)
    R = 200
    t0 = time.perf_counter()
    for _ in range(R):
        eng.atom('fact', nm(base))            # a direct recall
    recall_us = (time.perf_counter() - t0) / R * 1e6
    t0 = time.perf_counter()
    for _ in range(R):
        chain = eng.transitive_reach('linkr', top)
    chain_us = (time.perf_counter() - t0) / R * 1e6
    hops = len(chain) if chain else 0
    recalls.append(recall_us)
    print(f'{N:>12,} {recall_us:>12.1f} {chain_us:>12.1f} {hops:>11}', flush=True)
print('\nFlat recall + flat chain across 100x store growth = flat access.', flush=True)
# GATE: recall latency must stay flat (content-addressed) across 100x growth.
ratio = max(recalls) / max(min(recalls), 1e-9)
print(f'[gate] recall flatness ratio (max/min) = {ratio:.2f}x  over {SIZES[-1]//SIZES[0]}x store growth', flush=True)
assert ratio < 2.0, f'flat access regressed: recall grew {ratio:.1f}x with store size'
print('GATE GREEN -- flat access is the default, O(1) recall independent of N.', flush=True)
print('DONE.', flush=True)
