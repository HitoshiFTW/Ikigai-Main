import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 335 -- plain-English question -> emergent parse -> DERIVE engine answer.

Unifies "ask anything" with derive-not-store: the question is parsed
emergently (entity by novelty, relation by recurrence + trigram morphology, no
wh/relation lists), and the answer comes from the SEMANTIC derive engine
(atoms + on-demand derivation), not the episodic holo store.  Multi-hop chains
are DERIVED, never stored.
"""
import integrate


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    prose = ("zorvex reports to qualan. qualan reports to mendaro. "
             "mendaro reports to thessaly. thessaly reports to volgrim. "
             "brundle reports to kelmar.")
    org.comprehend(prose)        # text -> emergent atoms -> derive engine

    print("== A. plain-English question -> derive-engine answer ==", flush=True)
    a, e, r = org.ask_derive("who does zorvex report to")
    check("'who does zorvex report to' -> qualan (via DERIVE)", a == "qualan",
          f"(ent={e}, rel={r!r}, ans={a})")
    a, e, r = org.ask_derive("who does brundle report to")
    check("'who does brundle report to' -> kelmar", a == "kelmar", f"(ans={a})")
    # morphology: 'reporting' should still map to the learned relation
    a, e, r = org.ask_derive("who is mendaro reporting to")
    check("morphology 'reporting' -> reports -> thessaly", a == "thessaly",
          f"(rel={r!r}, ans={a})")

    print("\n== B. multi-hop plain-English via chained derive (not stored) ==", flush=True)
    eng = org.general_reasoner.derive_engine
    nb = len(eng.triples)
    a, e, r = org.ask_derive("who does zorvex report to", depth=4)
    check("4-hop 'who does zorvex report to' (depth=4) -> volgrim", a == "volgrim",
          f"(ans={a})")
    check("derive-not-store: no composites stored", len(eng.triples) == nb,
          f"({nb}->{len(eng.triples)})")

    print("\n== C. honest-unknown on an unknown entity ==", flush=True)
    a, e, r = org.ask_derive("who does flarn report to")
    check("unknown entity -> None", a is None, f"(ent={e}, ans={a})")

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
