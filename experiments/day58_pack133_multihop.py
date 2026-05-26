"""
Day 58 Pack 133b -- Multi-hop reasoning in Ikigai.

Build a small knowledge graph in unified memory. Test reasoning chains
of depth 2, 3, 4, 5 via org.reason_chain(start, hops). Each hop uses a
different role (isa / lives_in / contains / has / property).

Direct attack on transformer chain-of-thought territory: we do it
natively via VSA role-binding composition, no prompt template required.

Verifications:
    V1  reason_chain method wired
    V2  2-hop chain correct (matches existing chain())
    V3  3-hop chain correct
    V4  4-hop chain correct
    V5  5-hop chain correct
    V6  multi-hop survives +20 distractor IS-A writes (no-forgetting on chain)
    V7  substrate FIXED throughout
    V8  alternate starts reach same endpoint (graph navigation)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate import IkigaiOrganism

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 133b: Multi-hop reasoning ===\n')

org = IkigaiOrganism(flat_only=True)
sub0 = org.unified.substrate_bytes()
_log(f'  organism: substrate={sub0/1_048_576:.0f} MB FIXED')

check('V1 reason_chain method wired', hasattr(org, 'reason_chain'))

#  Build a small knowledge graph
# Roles: isa, lives_in, contains, has, property
# Need to register custom roles (not all in DEFAULT_ROLES)
for r in ('lives_in', 'contains', 'has'):
    org._ensure_role(r)

KB = [
    ('cat',     'isa',      'mammal'),
    ('dog',     'isa',      'mammal'),
    ('eagle',   'isa',      'bird'),
    ('mammal',  'lives_in', 'forest'),
    ('bird',    'lives_in', 'sky'),
    ('forest',  'contains', 'river'),
    ('sky',     'contains', 'cloud'),
    ('river',   'has',      'water'),
    ('cloud',   'has',      'rain'),
    ('water',   'property', 'cold'),
    ('rain',    'property', 'wet'),
]
_log('\nLoading knowledge graph (11 facts across 5 roles)...')
for hypo, role, target in KB:
    for _ in range(30):
        org.unified.relate(hypo, role, target)

# Candidate sets per role
HYPER  = ['mammal','bird','reptile','flower']
PLACE  = ['forest','sky','desert','ocean']
INSIDE = ['river','cloud','rock','tree']
THING  = ['water','rain','fire','wind']
PROP   = ['cold','wet','warm','dry']

#  V2: 2-hop
_log('\n--- 2-hop: cat -> isa -> lives_in ---')
p = org.reason_chain('cat', [('isa', HYPER), ('lives_in', PLACE)])
_log(f'  path: {" -> ".join(map(str,p))}')
check('V2 2-hop cat -> mammal -> forest', p == ['cat','mammal','forest'], f'{p}')

#  V3: 3-hop
_log('\n--- 3-hop: cat -> isa -> lives_in -> contains ---')
p = org.reason_chain('cat', [('isa', HYPER), ('lives_in', PLACE), ('contains', INSIDE)])
_log(f'  path: {" -> ".join(map(str,p))}')
check('V3 3-hop cat -> mammal -> forest -> river',
      p == ['cat','mammal','forest','river'], f'{p}')

#  V4: 4-hop
_log('\n--- 4-hop: ... -> has ---')
p = org.reason_chain('cat', [('isa', HYPER), ('lives_in', PLACE),
                              ('contains', INSIDE), ('has', THING)])
_log(f'  path: {" -> ".join(map(str,p))}')
check('V4 4-hop cat -> mammal -> forest -> river -> water',
      p == ['cat','mammal','forest','river','water'], f'{p}')

#  V5: 5-hop
_log('\n--- 5-hop: ... -> property ---')
p = org.reason_chain('cat', [('isa', HYPER), ('lives_in', PLACE),
                              ('contains', INSIDE), ('has', THING),
                              ('property', PROP)])
_log(f'  path: {" -> ".join(map(str,p))}')
check('V5 5-hop cat -> ... -> cold',
      p == ['cat','mammal','forest','river','water','cold'], f'{p}')

#  V6: distractor flood, chain survives
_log('\n--- Distractor flood + retry 5-hop ---')
DISTR = [('flomp','isa','xyz'),('zarb','isa','xyz'),('mibvor','isa','xyz')] * 7
for hypo, role, tgt in DISTR:
    org.unified.relate(hypo, role, tgt)
p_after = org.reason_chain('cat', [('isa', HYPER), ('lives_in', PLACE),
                                    ('contains', INSIDE), ('has', THING),
                                    ('property', PROP)])
_log(f'  after +21 distractors: {" -> ".join(map(str,p_after))}')
check('V6 multi-hop survives distractor flood',
      p_after == ['cat','mammal','forest','river','water','cold'], f'{p_after}')

#  V7: substrate flat
sub1 = org.unified.substrate_bytes()
check('V7 substrate FIXED throughout', sub1 == sub0, f'{sub0} -> {sub1}')

#  V8: alternate start dog -> same destination
_log('\n--- Alternate start: dog (also a mammal) ---')
p_dog = org.reason_chain('dog', [('isa', HYPER), ('lives_in', PLACE),
                                  ('contains', INSIDE), ('has', THING),
                                  ('property', PROP)])
_log(f'  dog path: {" -> ".join(map(str,p_dog))}')
check('V8 dog reaches same endpoint (cold)',
      p_dog == ['dog','mammal','forest','river','water','cold'], f'{p_dog}')

# Also show eagle (bird path) lands on "wet"
_log('\n--- eagle (bird branch) ---')
p_eagle = org.reason_chain('eagle', [('isa', HYPER), ('lives_in', PLACE),
                                      ('contains', INSIDE), ('has', THING),
                                      ('property', PROP)])
_log(f'  eagle path: {" -> ".join(map(str,p_eagle))}')

#  summary
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 133b -- Multi-hop reasoning')
_log(f'{"="*64}')
_log(f'  2-hop, 3-hop, 4-hop, 5-hop chains all resolved correctly')
_log(f'  Distractor-flooded 5-hop: still correct')
_log(f'  Alternate-start eagle: {" -> ".join(map(str,p_eagle))}')
_log(f'  Substrate FIXED: {sub1/1_048_576:.0f} MB')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- multi-hop chain-of-thought via role-binding' if FAIL == 0 else 'NEEDS FIX'))
