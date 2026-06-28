import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 339 -- EMERGENT depth (no count passed for single-hop; honest map of
multi-hop coverage).

Depth is read from the question: hop count = number of relation-MENTIONS (a
maximal run of relational tokens), mentions separated by argument tokens. This
closes single-hop fully (no depth param) and any mention-separated chain. Nested
function-word chains ('the R1 of the R2 of X') still collapse to one mention --
documented limit, needs 3-tier glue induction (next rung); explicit depth still
works for forced same-relation chains.
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
             "mendaro reports to thessa. thessa reports to volgrim.")
    org.comprehend(prose)

    print("== A. EMERGENT single-hop depth (no count passed) ==", flush=True)
    a, e, m = org.ask_derive("who does zorvex report to")   # depth=None default
    check("'who does zorvex report to' -> qualan, 1 mention", a == "qualan" and len(m) == 1,
          f"(ent={e!r}, mentions={m}, ans={a!r})")
    a, e, m = org.ask_derive("who does mendaro report to")
    check("'who does mendaro report to' -> thessa", a == "thessa", f"(ans={a!r})")

    print("\n== B. explicit-depth multi-hop preserved ==", flush=True)
    a, e, r = org.ask_derive("who does zorvex report to", depth=4)
    check("forced depth=4 -> volgrim", a == "volgrim", f"(ans={a!r})")

    print("\n== C. honest limit: nested function-word chain ==", flush=True)
    ent, mentions = org.holo_reader.parse_chain("who does zorvex report to")
    print(f"   parse_chain('who does zorvex report to') = ({ent!r}, {mentions})", flush=True)
    print("   (nested 'R1 of R2 of X' would collapse to 1 mention -> 3-tier glue "
          "induction is the next rung)", flush=True)

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
