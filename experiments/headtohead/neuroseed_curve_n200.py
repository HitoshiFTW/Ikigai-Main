import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""NeuroSeed-only paper-grade curve, n=200 per depth. No LLM calls.

Same equal-knowledge multi-hop task as multihop_vs_llm.py: made-up org-chart
facts + distractor edges, follow 'reportsto' N steps. NeuroSeed derives the
chain by exact lookup (O(1)/hop). This run measures the NeuroSeed column at
paper sample size across depths 1..20, and writes a plot.
"""
import random, time
import integrate

random.seed(11)
_AL = "abcdefghijklmnopqrstuvwxyz"
def name(rng): return "".join(rng.choice(_AL) for _ in range(6))

def make_case(depth, n_distract, rng):
    people = [name(rng) for _ in range(depth + 1 + n_distract)]
    chain = people[:depth + 1]
    facts = [(chain[i], "reportsto", chain[i + 1]) for i in range(depth)]
    for p in people[depth + 1:]:
        tgt = rng.choice(people)
        if tgt != p: facts.append((p, "reportsto", tgt))
    rng.shuffle(facts)
    return facts, chain[0], chain[depth]

def neuroseed_answer(facts, start, depth):
    org = integrate.IkigaiOrganism(flat_only=True)
    org.ingest_triples(facts)
    eng = org.general_reasoner.derive_engine
    cur = start
    for _ in range(depth):
        cur = eng.atom("reportsto", cur)
        if not cur: return None
    return cur

def main():
    rng = random.Random(7)
    depths = [1, 2, 3, 5, 8, 12, 16, 20]
    n_per = 200; n_distract = 20
    rows = []
    print(f"NeuroSeed-only curve  n={n_per}  distractors={n_distract}", flush=True)
    print(f"{'depth':>6} {'acc':>8} {'sec':>8}", flush=True)
    t_all = time.time()
    for d in depths:
        cases = [make_case(d, n_distract, rng) for _ in range(n_per)]
        t0 = time.time()
        correct = sum(neuroseed_answer(f, s, g_d) == g for f, s, g in cases
                      for g_d in (d,))
        dt = time.time() - t0
        acc = correct / n_per
        rows.append((d, acc, dt))
        print(f"{d:>6} {acc:>7.1%} {dt:>7.1f}s", flush=True)
    print(f"total {time.time()-t_all:.1f}s", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = [r[0] for r in rows]; accs = [r[1]*100 for r in rows]
        plt.figure(figsize=(7,4.2))
        plt.plot(ds, accs, "o-", color="#1b9e77", lw=2.2, ms=7, label="NeuroSeed (exact derivation)")
        plt.ylim(-3, 105); plt.xlabel("reasoning depth (hops)"); plt.ylabel("accuracy (%)")
        plt.title(f"NeuroSeed multi-hop accuracy  (n={n_per}/depth, 20 distractors)")
        plt.grid(alpha=0.3); plt.legend(loc="lower left")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuroseed_curve_n200.png")
        plt.tight_layout(); plt.savefig(out, dpi=130)
        print("plot:", out, flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)

if __name__ == "__main__":
    main()
