import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 337 -- STRESS the NL pipe on varied prose (find the next wall).

Not a pass/fail gate -- a measurement. Mixed relations, function words, and
multi-word entities (all made-up, alpha-only). Read it all, then ask a
plain-English question per fact and measure end-to-end accuracy + WHERE it
breaks. The known suspects: (a) function words ('the','is') are top-df ->
polluting the relation; (b) a rare relation mis-split as an entity; (c) a
question whose connective only partially matches the statement's.
"""
import integrate

# made-up entities (alpha-only). multi-word places/teams included.
FACTS = [
    # (subject, relation_key, object, statement, question)
    ("zorvex", "reportsto", "qualan",        "zorvex reports to qualan",                  "who does zorvex report to"),
    ("qualan", "reportsto", "mendaro",       "qualan reports to mendaro",                 "who does qualan report to"),
    ("brundle", "reportsto", "kelmar",       "brundle reports to kelmar",                 "who does brundle report to"),
    ("thessa", "reportsto", "volgrim",       "thessa reports to volgrim",                 "who does thessa report to"),
    ("zorvex", "manages", "krellan team",    "zorvex manages krellan team",               "what does zorvex manage"),
    ("mendaro", "manages", "velmar team",    "mendaro manages velmar team",               "what does mendaro manage"),
    ("kelmar", "manages", "draxen team",     "kelmar manages draxen team",                "what does kelmar manage"),
    ("qualan", "leads", "orrin project",     "qualan leads orrin project",                "what does qualan lead"),
    ("volgrim", "leads", "siltan project",   "volgrim leads siltan project",              "what does volgrim lead"),
    ("brundle", "leads", "phenwick project", "brundle leads phenwick project",            "what does brundle lead"),
]


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    prose = ". ".join(f[3] for f in FACTS) + "."
    org.comprehend(prose)

    hr = org.holo_reader
    print(f"corpus: {len(FACTS)} facts, {hr.n_sentences} sentences", flush=True)
    print(f"emergent rel_cut = {hr.rel_cut()}  (max_df={max(hr.df.values())})", flush=True)
    relish = sorted([w for w in hr.df if hr.df[w] >= hr.rel_cut()],
                    key=lambda w: -hr.df[w])
    print(f"tokens classified RELATIONAL: {relish}", flush=True)

    print("\n--- extracted atoms ---", flush=True)
    for a in org.holo_reader.extract_atoms():
        print("   ", a, flush=True)

    print("\n--- end-to-end QA (plain English) ---", flush=True)
    ok = 0
    per_rel = {}
    for s, rk, o, stmt, q in FACTS:
        ans, e, r = org.ask_derive(q)
        hit = (ans == o)
        ok += hit
        per_rel.setdefault(rk, [0, 0])
        per_rel[rk][0] += hit; per_rel[rk][1] += 1
        flag = "" if hit else "   <-- MISS"
        print(f"   [{'ok ' if hit else 'XX '}] {q!r}", flush=True)
        print(f"          ent={e!r} rel={r!r} -> {ans!r}  (want {o!r}){flag}", flush=True)

    print(f"\noverall: {ok}/{len(FACTS)} = {ok/len(FACTS):.0%}", flush=True)
    print("per relation:", flush=True)
    for rk, (c, n) in per_rel.items():
        print(f"   {rk:10s} {c}/{n}", flush=True)


if __name__ == "__main__":
    main()
