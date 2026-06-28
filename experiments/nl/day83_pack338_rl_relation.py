import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 338 -- native dopamine-RL for RELATION DISCOVERY (no hardcoding).

The structural prior (edges=args, interior=relations) handles frequent
relations, but a RARE relation -- one appearing too few times to clear the df
threshold -- is misclassified as an entity, and its facts break.  RL fixes it
from REWARD: quiz the organism; when a parse misses, reinforce the interior
tokens of the 'subject ... gold' fact (dopamine), so the rare relation's bias
clears the threshold.  The organism LEARNS which tokens are relations from
whether the derived answer was right -- emergent, not hand-set.
"""
import integrate


def quiz(org, qa):
    ok = 0
    for subj, gold, q in qa:
        ans, _, _ = org.ask_derive(q)
        ok += (ans == gold)
    return ok


def main():
    org = integrate.IkigaiOrganism(flat_only=True)
    hr = org.holo_reader

    # 5 frequent 'reports to' facts + 2 RARE 'advises' facts (df=2, below cut)
    prose = ("zorvex reports to qualan. qualan reports to mendaro. "
             "mendaro reports to thessa. thessa reports to volgrim. "
             "brundle reports to kelmar. "
             "qualan advises zorvex. kelmar advises brundle.")
    org.comprehend(prose)
    cut = hr.rel_cut()
    print(f"rel_cut={cut}  df(reports)={hr.df['reports']}  df(advises)={hr.df['advises']}", flush=True)
    print(f"'advises' relational before RL? {hr.is_relational('advises')}", flush=True)

    rare = [("qualan", "zorvex", "what does qualan advise"),
            ("kelmar", "brundle", "what does kelmar advise")]
    freq = [("zorvex", "qualan", "who does zorvex report to"),
            ("brundle", "kelmar", "who does brundle report to")]

    print("\n-- BEFORE RL --", flush=True)
    print(f"  rare 'advises' quiz : {quiz(org, rare)}/2", flush=True)
    print(f"  freq 'reports' quiz : {quiz(org, freq)}/2", flush=True)

    # RL: reward-driven reinforcement on the misses (dopamine)
    print("\n-- RL: reinforce misses --", flush=True)
    for subj, gold, q in rare:
        ans, _, _ = org.ask_derive(q)
        if ans != gold:
            learned = hr.reinforce(subj, gold)          # reward the true relation
            print(f"  miss '{q}' -> reinforce({subj!r},{gold!r}) learned={learned}", flush=True)
    # re-extract with updated bias + re-ingest (no re-reading)
    org.ingest_triples(hr.extract_atoms())
    print(f"  'advises' relational after RL? {hr.is_relational('advises')} "
          f"(bias={hr.rel_bias['advises']:.1f})", flush=True)

    print("\n-- AFTER RL --", flush=True)
    r_after = quiz(org, rare); f_after = quiz(org, freq)
    print(f"  rare 'advises' quiz : {r_after}/2", flush=True)
    print(f"  freq 'reports' quiz : {f_after}/2  (must stay -- no forgetting)", flush=True)

    ok = (r_after == 2 and f_after == 2)
    print(f"\n{'PASS' if ok else 'FAIL'} -- RL discovered the rare relation from reward; "
          f"frequent relation intact", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
