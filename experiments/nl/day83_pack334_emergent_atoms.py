import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 334 -- text -> EMERGENT atoms -> derive-not-store reasoning.

The honest fix to the derive-not-store concern: the holographic reader is the
EPISODIC front door (it stores what it reads, verbatim, bounded), but it does
NOT become the knowledge store.  It EXTRACTS clean (subj, rel, obj) atoms and
hands them to the derive engine, which holds only the atoms and DERIVES every
composite (multi-hop chains, etc.) on demand -- never stored.

Everything emergent, ZERO hardcoding:
  - which token is a RELATION is learned by RECURRENCE (df), not a list;
  - subject/object from the text's own word order;
  - the multi-hop answer is DERIVED by the engine, not stored.
"""
import integrate


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    # messy prose, made-up entities. 'reports to' RECURS -> learned as relation;
    # every name is novel -> argument. Nothing is told which is which.
    prose = ("zorvex reports to qualan. "
             "qualan reports to mendaro. "
             "mendaro reports to thessaly. "
             "thessaly reports to volgrim. "
             "brundle reports to kelmar.")

    print("== A. emergent atom extraction (relation learned by recurrence) ==", flush=True)
    atoms = org.comprehend(prose)            # read -> extract -> ingest
    print("   extracted atoms:", flush=True)
    for a in atoms:
        print("     ", a, flush=True)
    rels = {r for _, r, _ in atoms}
    check("relation emerged as 'reports to' (not hardcoded)", rels == {"reports to"},
          f"(rels={rels})")
    check("5 atoms extracted", len(atoms) == 5)
    check("subject/object correct from word order",
          ("zorvex", "reports to", "qualan") in atoms and
          ("mendaro", "reports to", "thessaly") in atoms)

    print("\n== B. derive-not-store: multi-hop DERIVED from the atoms ==", flush=True)
    eng = org.general_reasoner.derive_engine
    n_atoms_before = len(eng.triples)
    # 4-hop chain: zorvex -> qualan -> mendaro -> thessaly -> volgrim
    cur = "zorvex"
    for _ in range(4):
        cur = eng.atom("reports to", cur)
    check("4-hop chain DERIVED zorvex ->...-> volgrim", cur == "volgrim",
          f"(got {cur})")
    n_atoms_after = len(eng.triples)
    check("derive-not-store intact: no composites stored during derivation",
          n_atoms_after == n_atoms_before,
          f"({n_atoms_before} -> {n_atoms_after} atoms)")

    print("\n== C. full pipe: messy text -> emergent atom -> single-hop recall ==", flush=True)
    a1 = eng.atom("reports to", "brundle")
    check("'brundle reports to' -> kelmar (from prose, via emergent atom)",
          a1 == "kelmar", f"(got {a1})")

    print(f"\n{passed}/{total} checks passed", flush=True)
    print(f"\nstored atoms: {n_atoms_before}   |   a 4-hop fact was DERIVED, not stored", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
