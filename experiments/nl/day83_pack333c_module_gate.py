import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 333 module gate -- HolographicReader as a native cognition module,
wired through the organism API. Proves: read ANY sentence (no answer slot
pre-chosen), ask back ANY position, honest-unknown, capacity, all template-free.
"""
import random
import integrate


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    hr = org.holo_reader
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    print("== A. read passage, ask back ANY position (no pre-chosen hole) ==", flush=True)
    org.read_holo("zorvex reports to qualan. "
                  "the glorp of vextil is mwumbo. "
                  "brundle drifts beneath kelmar.")
    # object slot
    a, s = org.answer_holo("zorvex reports to _")
    check("object: 'zorvex reports to _' -> qualan", a == "qualan", f"({a}, {s:.3f})")
    # SUBJECT slot (only possible because every position was stored)
    a, s = org.answer_holo("_ reports to qualan")
    check("subject: '_ reports to qualan' -> zorvex", a == "zorvex", f"({a}, {s:.3f})")
    # middle content slot
    a, s = org.answer_holo("the glorp of vextil is _")
    check("'the glorp of vextil is _' -> mwumbo", a == "mwumbo", f"({a}, {s:.3f})")
    a, s = org.answer_holo("brundle drifts beneath _")
    check("'brundle drifts beneath _' -> kelmar", a == "kelmar", f"({a}, {s:.3f})")

    print("\n== B. honest-unknown ==", flush=True)
    a, s = org.answer_holo("flarn wibbles past _")
    check("never-read context -> abstain", a is None, f"(sim {s:.3f})")

    print("\n== C. top-k ranking ==", flush=True)
    ranked = org.answer_holo("zorvex reports to _", top_k=3)
    check("top-1 of ranked is qualan", ranked and ranked[0][0] == "qualan",
          f"({ranked})")

    print("\n== C2. plain-English ASK (no hole, no wh-list, morphology) ==", flush=True)
    org.read_holo("zorvex likes mellow.")     # same subject, 2nd relation
    a, s = org.ask_holo("who does zorvex report to")
    check("'who does zorvex report to' -> qualan", a == "qualan", f"({a}, {s:.3f})")
    a, s = org.ask_holo("what does zorvex like")
    check("'what does zorvex like' -> mellow (disambiguated)", a == "mellow", f"({a}, {s:.3f})")
    a, s = org.ask_holo("who does flarn admire")
    check("unknown question -> abstain", a is None, f"(sim {s:.3f})")

    print("\n== D. capacity through the module (SDM) ==", flush=True)
    org2 = integrate.IkigaiOrganism(flat_only=True)
    rng = random.Random(7)
    def rname(): return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
    stored = []
    for _ in range(500):
        a, b = rname(), rname()
        org2.read_holo(f"{a} reports to {b}.")
        stored.append((a, b))
    ok = sum(org2.answer_holo(f"{a} reports to _")[0] == b for a, b in stored)
    check("N=500 object recall == 100%", ok == 500, f"({ok}/500)")

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
