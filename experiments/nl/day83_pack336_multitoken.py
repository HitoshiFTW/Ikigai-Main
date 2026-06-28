import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 336 -- MULTI-TOKEN entities (the real-data blocker).

Real facts have multi-word entities ('buenos aires', 'carbon dioxide'). The
holographic reader now groups CONSECUTIVE argument tokens (low df) into one
entity span, EMERGENTLY -- a run of novel tokens with no recurring connective
between them is a single argument. No list, no NER, no grammar. Multi-word
subjects AND objects round-trip through the derive-not-store engine.

Made-up multi-word entities (alpha-only -> dodges the digit-tokenizer bug).
"""
import integrate


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    # 'reports to' recurs -> relation. Entities are multi-word made-up names.
    prose = ("zorvex quay reports to mendaro vale. "
             "mendaro vale reports to thessaly crag. "
             "brundle fenn reports to kelmar dune.")

    print("== A. emergent multi-word atom extraction ==", flush=True)
    atoms = org.comprehend(prose)
    for a in atoms:
        print("     ", a, flush=True)
    check("subject is multi-word 'zorvex quay'",
          ("zorvex quay", "reports to", "mendaro vale") in atoms)
    check("chain mid-node multi-word preserved",
          ("mendaro vale", "reports to", "thessaly crag") in atoms)
    check("3 atoms", len(atoms) == 3)

    print("\n== B. derive over multi-word entities ==", flush=True)
    eng = org.general_reasoner.derive_engine
    a1 = eng.atom("reports to", "zorvex quay")
    check("atom('reports to','zorvex quay') -> 'mendaro vale'", a1 == "mendaro vale",
          f"(got {a1!r})")
    nb = len(eng.triples)
    cur = "zorvex quay"
    for _ in range(2):
        cur = eng.atom("reports to", cur)
    check("2-hop multi-word DERIVED -> 'thessaly crag'", cur == "thessaly crag",
          f"(got {cur!r})")
    check("derive-not-store intact", len(eng.triples) == nb, f"({nb}->{len(eng.triples)})")

    print("\n== C. plain-English question with multi-word entity ==", flush=True)
    a, e, r = org.ask_derive("who does zorvex quay report to")
    check("'who does zorvex quay report to' -> 'mendaro vale'", a == "mendaro vale",
          f"(ent={e!r}, rel={r!r}, ans={a!r})")
    a, e, r = org.ask_derive("who does brundle fenn report to")
    check("'who does brundle fenn report to' -> 'kelmar dune'", a == "kelmar dune",
          f"(ent={e!r}, ans={a!r})")

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
