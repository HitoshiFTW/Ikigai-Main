import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 333 rung 1 -- TEMPLATE-FREE holographic completion (Prince's vision).

Give it any sentence, it stores the STRUCTURE; ask it the sentence with a hole,
the answer falls out by resonance.  NO templates, NO relation lists, NO grammar
rules -- the only thing that exists is the substrate: bind tokens by position
into one hypervector, unbind the question's known tokens, clean up the answer
over the vocabulary.  The "relation" is never named; it is just the other words.

Mechanism (pure FHRR):
    store(sentence):
        for the content token at position p, build a CONTEXT key from every
        OTHER token bound to its absolute position (rotate(key(w_j), j)), and
        accumulate  M += bind(context, key(answer)).
    answer(sentence_with_hole):
        rebuild the context from the known positioned tokens, unbind it from M,
        and clean up the result against the vocabulary (argmax cosine).  If the
        best cosine is below the substrate noise floor (k/sqrt(2d)), say UNKNOWN
        -- honest calibration, the LLM differentiator, same geometry as line #11.

This rung proves the COMPLETION mechanism on fill-in-the-blank questions (the
purest test, positions aligned).  Mapping a wh-question -> the blanked slot is
rung 2; multi-fact paragraphs + the cleanup wall are the rungs after.
"""
import math, random
import numpy as np
from ikigai.cognition.flat_memory import ComputedKey
from ikigai.cognition.phasor_state import bind, unbind, rotate, cosine

OMEGA = 0.30  # position phase step


def toks(s):
    return [t for t in s.lower().replace("?", " ").replace(".", " ").split() if t]


class HoloMem:
    """One holographic bundle. Template-free: store/answer over raw tokens."""

    def __init__(self, d=512, seed=114):
        self.d = d
        self.ck = ComputedKey(d=d, seed=seed)
        self.M = np.zeros(d, dtype=np.complex64)
        self.vocab = set()
        self.n = 0
        # noise floor for honest-unknown: cleanup over a bundle of N terms has
        # off-target cosine ~ 1/sqrt(2d); accept boundary = k * that.
        self.k_sigma = 5.0

    def _ctx(self, positioned):
        """Bind every (token, position) into one context key. Order preserved
        by absolute-position rotation -> 'a likes b' != 'b likes a'."""
        c = np.ones(self.d, dtype=np.complex64)
        for w, p in positioned:
            c = bind(c, rotate(self.ck.key(w), p, OMEGA))
        return c

    def store(self, sentence, answer_index):
        """Learn one sentence. answer_index marks which slot is the content
        answer; every other token is context. No relation named."""
        ws = toks(sentence)
        ans = ws[answer_index]
        ctx = self._ctx([(w, p) for p, w in enumerate(ws) if p != answer_index])
        self.M += bind(ctx, self.ck.key(ans))
        self.vocab.update(ws)
        self.n += 1

    def answer(self, sentence_with_hole, hole_token="_"):
        """Fill the blank. Pure resonance + vocab cleanup. Honest-unknown when
        the best match is below the substrate noise floor."""
        ws = toks(sentence_with_hole)
        p_hole = ws.index(hole_token)
        ctx = self._ctx([(w, p) for p, w in enumerate(ws) if p != p_hole])
        cand = unbind(self.M, ctx)                      # ~ key(answer) + noise
        boundary = self.k_sigma / math.sqrt(2 * self.d)
        best, best_sim = None, -1.0
        for w in self.vocab:
            s = cosine(cand, self.ck.key(w))
            if s > best_sim:
                best, best_sim = w, s
        if best_sim < boundary:
            return None, best_sim                       # honest unknown
        return best, best_sim


def section(t): print("\n== " + t + " ==", flush=True)


def main():
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    # ---- A. template-free completion on ARBITRARY made-up sentences ------
    section("A. any sentence in, answer out -- no templates")
    hm = HoloMem(d=512)
    facts = [
        ("zorvex reports to qualan", 3),
        ("the glorp of vextil is mwumbo", 5),     # nonsense relation 'glorp'
        ("brundle quayle drifts beneath kelmar", 4),
        ("plenix owns a wexxel", 3),
        ("the mibbon under threngar equals zaxic", 5),
    ]
    for s, ai in facts:
        hm.store(s, ai)
    # ask each back as a fill-in-the-blank (same surface, hole at the answer)
    qa = [
        ("zorvex reports to _", "qualan"),
        ("the glorp of vextil is _", "mwumbo"),
        ("brundle quayle drifts beneath _", "kelmar"),
        ("plenix owns a _", "wexxel"),
        ("the mibbon under threngar equals _", "zaxic"),
    ]
    for q, gold in qa:
        got, sim = hm.answer(q)
        check(f"'{q}' -> {gold}", got == gold, f"(got {got}, sim {sim:.3f})")

    # ---- B. it answers from CONTEXT, not position alone ------------------
    section("B. order matters -- 'a verb b' != 'b verb a'")
    hm2 = HoloMem(d=512)
    hm2.store("krenn likes ploom", 2)
    hm2.store("ploom likes krenn", 2)
    g1, s1 = hm2.answer("krenn likes _")
    g2, s2 = hm2.answer("ploom likes _")
    check("'krenn likes _' -> ploom", g1 == "ploom", f"(got {g1})")
    check("'ploom likes _' -> krenn", g2 == "krenn", f"(got {g2})")

    # ---- C. honest UNKNOWN on a sentence it never stored -----------------
    section("C. honest-unknown (calibration, not hallucination)")
    got, sim = hm.answer("flarn wibbles past _")     # never stored context
    check("unknown context -> None (abstain)", got is None, f"(sim {sim:.3f})")

    # ---- D. capacity wall -- WHERE does it break? (the honest measurement)
    section("D. crosstalk wall: recall vs #facts in one bundle")
    rng = random.Random(7)
    def rname():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
    for N in (10, 25, 50, 100, 200):
        hm3 = HoloMem(d=512)
        stored = []
        for _ in range(N):
            a, b = rname(), rname()
            hm3.store(f"{a} reports to {b}", 3)
            stored.append((a, b))
        ok = 0
        for a, b in stored:
            got, _ = hm3.answer(f"{a} reports to _")
            ok += (got == b)
        print(f"   N={N:>4}  recall={ok/N:>5.0%}", flush=True)

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
