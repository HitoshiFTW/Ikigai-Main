import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""Pack 333 rung 2b -- ask a REAL question (no hole marker), get the answer.
Native, zero hardcoding: no wh-list, no stopword list, no df threshold, no
templates.

Insight from the 333d probe: dropping the relation as 'fluff' is what broke
disambiguation. Keep EVERY content token. Bridge question<->statement phrasing
with three substrate facts:
  (1) ORDER-FREE bag context  -> word order stops mattering.
  (2) TRIGRAM morphology       -> 'report' resonates to stored 'reports'
       (ComputedKey is built from char-trigrams; a query token not in vocab is
       mapped to its nearest vocab key by cosine -- a stemmer for free).
  (3) "the answer is the missing cue" -> a question carries its cues (the words
       it shares with the fact); the answer is simply the stored token those
       cues resonate to that ISN'T already a cue.  No hole to mark, no wh-word
       to recognise -- query-only words ('who','does') map to nothing and drop
       themselves.
"""
import math, random
import numpy as np
from ikigai.cognition.flat_memory import ComputedKey
from ikigai.cognition.phasor_state import bind, unbind, cosine

_TOK = __import__("re").compile(r"[a-z0-9'][a-z0-9']*")
def toks(s): return _TOK.findall(str(s).lower())


class QReader:
    def __init__(self, d=512, M=8000, k=32, seed=114, morph_thresh=0.45):
        self.d, self.M, self.k = d, M, k
        self.ck = ComputedKey(d=d, seed=seed)               # binding (clean ids)
        self.morph = ComputedKey(d=d, seed=seed, word_weight=0.0)  # trigram-only
        rng = np.random.default_rng(seed)
        self.addr = np.exp(1j * rng.uniform(-np.pi, np.pi, (M, d))).astype(np.complex64)
        self.acc = np.zeros((M, d), dtype=np.complex64)
        self.vocab = []                 # ordered, for stable matrix cleanup
        self._vset = set()
        self._vkeys = None
        self._vmorph = None
        self.morph_thresh = morph_thresh

    def _ctx(self, tokens):
        c = np.ones(self.d, dtype=np.complex64)
        for w in tokens:                # order-free: product is commutative
            c = bind(c, self.ck.key(w))
        return c

    def _activate(self, ctx):
        sims = np.abs(self.addr @ np.conj(ctx)) / self.d
        return np.argpartition(-sims, self.k - 1)[: self.k]

    def _add_vocab(self, w):
        if w not in self._vset:
            self._vset.add(w); self.vocab.append(w)
            self._vkeys = None; self._vmorph = None

    def read_sentence(self, sentence):
        ws = toks(sentence)
        if len(ws) < 2:
            return
        for w in ws:
            self._add_vocab(w)
        for p in range(len(ws)):
            others = ws[:p] + ws[p + 1:]
            ctx = self._ctx(others)
            self.acc[self._activate(ctx)] += bind(ctx, self.ck.key(ws[p]))

    def read(self, text):
        for s in __import__("re").split(r"[.!?\n]+", str(text or "")):
            if s.strip():
                self.read_sentence(s)

    def _vkey_matrix(self):
        if self._vkeys is None:
            self._vkeys = np.stack([self.ck.key(w) for w in self.vocab])
        return self._vkeys

    def _vmorph_matrix(self):
        if self._vmorph is None:
            self._vmorph = np.stack([self.morph.key(w) for w in self.vocab])
        return self._vmorph

    def _match(self, token):
        """Map a query token to a vocab token: exact, else nearest by trigram
        cosine in the word_weight=0 space (morphology), else None (query-only
        word like 'who'/'does' -> drops itself)."""
        if token in self._vset:
            return token
        if not self.vocab:
            return None
        K = self._vmorph_matrix()
        q = self.morph.key(token)
        sims = np.abs(K @ np.conj(q)) / self.d
        i = int(np.argmax(sims))
        return self.vocab[i] if sims[i] >= self.morph_thresh else None

    def ask(self, question, top_k=1):
        """Answer a free-form question. Cues = matched question tokens; the
        answer = what those cues resonate to that isn't itself a cue."""
        cues, seen = [], set()
        for t in toks(question):
            m = self._match(t)
            if m and m not in seen:
                seen.add(m); cues.append(m)
        if not cues:
            return (None, 0.0, cues) if top_k == 1 else ([], cues)
        ctx = self._ctx(cues)
        cand = unbind(self.acc[self._activate(ctx)].sum(axis=0), ctx)
        K = self._vkey_matrix()
        sims = np.abs(K @ np.conj(cand)) / self.d
        boundary = 5.0 / math.sqrt(2 * self.d)
        order = np.argsort(-sims)
        out = [(self.vocab[i], float(sims[i])) for i in order
               if self.vocab[i] not in seen and sims[i] >= boundary][:top_k]
        if top_k == 1:
            return (out[0][0], out[0][1], cues) if out else (None, 0.0, cues)
        return out, cues


def main():
    print("== morphology check (ComputedKey trigram) ==", flush=True)
    qr0 = QReader(d=512)
    qr0.read_sentence("zorvex reports to qualan")
    print("  match('report') ->", qr0._match("report"), flush=True)
    print("  match('reporting') ->", qr0._match("reporting"), flush=True)
    print("  match('who') ->", qr0._match("who"), flush=True)

    print("\n== A. reworded wh-questions, 40 facts ==", flush=True)
    rng = random.Random(3)
    def nm(): return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
    pairs = [(nm(), nm()) for _ in range(40)]
    qr = QReader(d=512)
    for a, b in pairs:
        qr.read_sentence(f"{a} reports to {b}")
    ok = sum(qr.ask(f"who does {a} report to")[0] == b for a, b in pairs)
    print(f"  reworded wh recall = {ok}/40 = {ok/40:.0%}", flush=True)

    print("\n== B. collision: same subject, two relations (the 333d break) ==", flush=True)
    qr2 = QReader(d=512)
    for s in ["zorvex reports to qualan", "zorvex likes mellow",
              "brundle reports to kelmar"]:
        qr2.read_sentence(s)
    a1, s1, c1 = qr2.ask("who does zorvex report to")
    a2, s2, c2 = qr2.ask("what does zorvex like")
    print(f"  'who does zorvex report to' -> {a1} (sim {s1:.3f}, cues {c1})", flush=True)
    print(f"  'what does zorvex like'      -> {a2} (sim {s2:.3f}, cues {c2})", flush=True)
    print(f"  DISAMBIGUATED: {a1=='qualan' and a2=='mellow'}", flush=True)

    print("\n== C. honest-unknown ==", flush=True)
    a3, s3, _ = qr2.ask("who does flarn admire")
    print(f"  'who does flarn admire' -> {a3} (sim {s3:.3f})  abstain={a3 is None}", flush=True)

    print("\n== D. capacity via ask() ==", flush=True)
    for N in (100, 500, 1000):
        q = QReader(d=512, M=8000)
        st = []
        for _ in range(N):
            a, b = nm(), nm(); q.read_sentence(f"{a} reports to {b}"); st.append((a, b))
        ok = sum(q.ask(f"who does {a} report to")[0] == b for a, b in st)
        print(f"  N={N:>4}  reworded recall = {ok/N:.0%}", flush=True)


if __name__ == "__main__":
    main()
