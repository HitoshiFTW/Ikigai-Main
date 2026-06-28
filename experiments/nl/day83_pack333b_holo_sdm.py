import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 333 rung 2a -- SDM-backed template-free completion (break the wall).

Rung 1 (day83_pack333_holo_qa.py) proved the mechanism but a single holographic
bundle holds only ~25 facts before crosstalk eats it.  The fix is the substrate's
actual memory: a Sparse Distributed Memory (Kanerva).  Each fact's context
addresses a SET of hard locations; the answer-binding is added only there.  Reads
activate the same locations and sum them, so a fact only competes with the few
others that share its address neighborhood -- crosstalk grows sub-linearly, not
linearly.  Still pure substrate: bind by position, write to SDM, read + cleanup.
No templates, no relation list, no grammar.
"""
import math, random
import numpy as np
from ikigai.cognition.flat_memory import ComputedKey
from ikigai.cognition.phasor_state import bind, unbind, rotate, cosine

OMEGA = 0.30


def toks(s):
    return [t for t in s.lower().replace("?", " ").replace(".", " ").split() if t]


class HoloSDM:
    """Template-free holographic completion, distributed over SDM locations."""

    def __init__(self, d=512, M=4000, k=32, seed=114):
        self.d, self.M, self.k = d, M, k
        self.ck = ComputedKey(d=d, seed=seed)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(-np.pi, np.pi, (M, d)).astype(np.float32)
        self.addr = np.exp(1j * ph).astype(np.complex64)        # (M,d) locations
        self.store_acc = np.zeros((M, d), dtype=np.complex64)    # accumulators
        self.vocab = set()
        self.n = 0
        self.k_sigma = 5.0

    def _ctx(self, positioned):
        c = np.ones(self.d, dtype=np.complex64)
        for w, p in positioned:
            c = bind(c, rotate(self.ck.key(w), p, OMEGA))
        return c

    def _activate(self, ctx):
        """Top-k hard locations whose address best matches the context."""
        sims = np.abs(self.addr @ np.conj(ctx)) / self.d        # (M,)
        return np.argpartition(-sims, self.k)[: self.k]

    def store(self, sentence, answer_index):
        ws = toks(sentence)
        ans = ws[answer_index]
        ctx = self._ctx([(w, p) for p, w in enumerate(ws) if p != answer_index])
        payload = bind(ctx, self.ck.key(ans))
        self.store_acc[self._activate(ctx)] += payload
        self.vocab.update(ws)
        self.n += 1

    def answer(self, sentence_with_hole, hole_token="_"):
        ws = toks(sentence_with_hole)
        p_hole = ws.index(hole_token)
        ctx = self._ctx([(w, p) for p, w in enumerate(ws) if p != p_hole])
        read = self.store_acc[self._activate(ctx)].sum(axis=0)
        cand = unbind(read, ctx)
        boundary = self.k_sigma / math.sqrt(2 * self.d)
        best, best_sim = None, -1.0
        for w in self.vocab:
            s = cosine(cand, self.ck.key(w))
            if s > best_sim:
                best, best_sim = w, s
        if best_sim < boundary:
            return None, best_sim
        return best, best_sim


def main():
    passed = total = 0
    def check(name, cond, extra=""):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)

    print("== A. correctness + honest-unknown (SDM-backed) ==", flush=True)
    hm = HoloSDM(d=512, M=4000, k=32)
    for s, ai in [("zorvex reports to qualan", 3),
                  ("the glorp of vextil is mwumbo", 5),
                  ("krenn likes ploom", 2), ("ploom likes krenn", 2)]:
        hm.store(s, ai)
    for q, gold in [("zorvex reports to _", "qualan"),
                    ("the glorp of vextil is _", "mwumbo"),
                    ("krenn likes _", "ploom"), ("ploom likes _", "krenn")]:
        got, sim = hm.answer(q)
        check(f"'{q}' -> {gold}", got == gold, f"(got {got}, sim {sim:.3f})")
    got, sim = hm.answer("flarn wibbles past _")
    check("unknown -> abstain", got is None, f"(sim {sim:.3f})")

    print("\n== B. capacity wall, SDM vs rung-1 bundle (~25) ==", flush=True)
    rng = random.Random(7)
    def rname():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
    for N in (25, 50, 100, 200, 500, 1000, 2000):
        hm3 = HoloSDM(d=512, M=8000, k=32)
        stored = []
        for _ in range(N):
            a, b = rname(), rname()
            hm3.store(f"{a} reports to {b}", 3)
            stored.append((a, b))
        ok = sum(hm3.answer(f"{a} reports to _")[0] == b for a, b in stored)
        print(f"   N={N:>5}  recall={ok/N:>5.0%}", flush=True)

    print(f"\n{passed}/{total} checks passed", flush=True)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
