"""CommonsenseQA head-to-head for the Ikigai organism -- with the native 'ask'
role (Pack 331): the organism LEARNS question -> relation from data, then answers
by reading the typed ConceptNet edge that relation points to.

    python bench_csqa.py --conceptnet path/to/conceptnet-assertions-5.7.0.csv.gz

CommonsenseQA (Talmor 2019) is 5-way multiple choice built FROM ConceptNet, so
the knowledge is already in the organism's substrate -- the hard part is knowing
WHICH relation each question asks for.  Two stages, both native, no hardcoding:

  LEARN  (train split): for each (question, concept, correct answer), the
         relation is whichever typed edge connects concept -> answer in the
         ingested graph; bind the question's cues to that relation via the
         organism's 'ask' role (substrate superposition).
  ANSWER (dev split): recall the asked relation(s) from the stem, then score
         each choice by the typed edge concept --[relation]--> choice.

Reference (dev, 1221 Q): random 20% | ConceptNet baselines ~40-55% |
LMs 60-75% | frontier LLM ~80% | human ~89%.
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request

import numpy as np

import integrate

DEV_URL = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
DEV_PATH = "csqa_dev.jsonl"
TRAIN_PATH = "csqa_train.jsonl"


def fetch(url, path):
    if not os.path.exists(path):
        print(f"downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)


def load_csqa(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = r["question"]
        rows.append({
            "concept": q["question_concept"].replace("_", " ").strip().lower(),
            "stem": q["stem"],
            "choices": [(c["label"], c["text"].replace("_", " ").strip().lower())
                        for c in q["choices"]],
            "answer": r.get("answerKey", ""),
        })
    return rows


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def toks_of(s):
    return [t for t in norm(s).split() if t]


def stem_concepts(stem, nodeset):
    """ConceptNet nodes named in the stem (1-3 grams).  No stopword list --
    any n-gram that is an actual graph node counts; non-concepts simply are not
    nodes, so they drop out on their own."""
    toks = toks_of(stem)
    found = set()
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i:i + n])
            if g in nodeset:
                found.add(g)
    return found


class Distributional:
    """The organism's distributional channel (LLM-style): how well an answer's
    words co-occur, in the trained substrate, with the question's words.  Pure
    learned co-occurrence -- nothing hardcoded."""

    def __init__(self, organism):
        self.mr = organism.unified
        self.d = self.mr.d
        self.vocab = getattr(self.mr, "_cooccur_seen", set())
        self._cache = {}

    def _recall(self, w):
        r = self._cache.get(w)
        if r is None:
            r = self.mr.recall(w, "cooccur")
            self._cache[w] = r
        return r

    def context_hv(self, words):
        """Aggregate co-occurrence neighbourhood of a bag of words."""
        hv = np.zeros(self.d, dtype=np.complex128)
        n = 0
        for w in words:
            if w in self.vocab:
                hv += self._recall(w)
                n += 1
        if n == 0:
            return None
        m = np.abs(hv).mean()
        return (hv / m) if m > 1e-9 else None

    def score(self, choice_words, ctx_hv):
        """How much the choice's words lie in the stem's co-occurrence
        neighbourhood (mean cosine of each choice word's key vs ctx)."""
        if ctx_hv is None:
            return 0.0
        vals = []
        for c in choice_words:
            if c in self.vocab:
                vals.append(float(np.real(self.mr.ck.key(c) @ np.conj(ctx_hv))
                                  / self.d))
        return sum(vals) / len(vals) if vals else 0.0


def relatedness(adj, a, b):
    na, nb = adj.get(a), adj.get(b)
    if not na or not nb:
        return 0.0
    s = 3.0 if b in na else 0.0
    ov = len(na & nb)
    if ov:
        s += ov / math.sqrt(len(na) * len(nb))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conceptnet", required=True)
    ap.add_argument("--limit", type=int, default=4_000_000)
    ap.add_argument("--min-weight", type=float, default=1.0)
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--train-q", type=int, default=0, help="cap train Q (0=all)")
    ap.add_argument("--ikg", metavar="PATH", default="",
                    help="load a trained organism body (for the distributional "
                         "cooccur channel); empty = no distributional signal")
    ap.add_argument("--dump", metavar="PATH", default="")
    args = ap.parse_args()

    from ikigai.cognition.kg_ingest import parse_conceptnet

    print("booting organism (CPU)...")
    org = integrate.IkigaiOrganism(flat_only=True)
    if args.ikg:
        print(f"loading trained body {args.ikg} (distributional channel) ...")
        org.load_ikg(args.ikg)
    dist = Distributional(org)
    print(f"  distributional cooccur vocab = {len(dist.vocab):,}")
    eng = org.general_reasoner.derive_engine

    print(f"ingesting ConceptNet (all relations, weight>={args.min_weight}, "
          f"limit={args.limit:,}) ...")
    t0 = time.time()
    res = org.ingest_triples(parse_conceptnet(
        args.conceptnet, relations=None, min_weight=args.min_weight,
        limit=args.limit))
    print(f"  {res['ingested']:,} edges in {time.time()-t0:.0f}s "
          f"-> {len(eng.triples):,} atoms, {len(eng.entities):,} entities")

    # typed + untyped adjacency over the organism's stored graph
    adj = {}
    trel = {}
    relvocab = set()
    for (s, r), o in eng.triples.items():
        adj.setdefault(s, set()).add(o)
        adj.setdefault(o, set()).add(s)
        trel.setdefault((s, o), set()).add(r)
        relvocab.add(r)
    nodeset = set(adj)
    relvocab = sorted(relvocab)
    print(f"  graph: {len(nodeset):,} nodes, {len(relvocab)} relation types")

    def rels_between(a, b):
        return trel.get((a, b), set()) | trel.get((b, a), set())

    # ---- LEARN the ask-role from the train split ------------------------
    fetch(TRAIN_URL, TRAIN_PATH)
    train = load_csqa(TRAIN_PATH)
    if args.train_q:
        train = train[:args.train_q]
    learned = 0
    for r in train:
        gold = next((t for l, t in r["choices"] if l == r["answer"]), None)
        if not gold:
            continue
        rs = rels_between(r["concept"], gold)
        if not rs:
            continue
        for rel in rs:
            org.learn_ask(r["stem"], rel)
        learned += 1
    print(f"  ask-role learned from {learned}/{len(train)} train questions "
          f"(direct concept->answer edges)")

    # ---- ANSWER the dev split -------------------------------------------
    fetch(DEV_URL, DEV_PATH)
    rows = load_csqa(DEV_PATH)
    if args.max_q:
        rows = rows[:args.max_q]
    print(f"scoring {len(rows)} dev questions ...\n")

    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None
    correct = correct_ans = answered = 0
    nct = 0
    for qi, r in enumerate(rows):
        concept = r["concept"]
        stem_tok = toks_of(r["stem"])
        stems = stem_concepts(r["stem"], nodeset)
        pred = org.ask_relation(r["stem"], candidates=relvocab, top_k=3)
        # distributional context = the stem's words + the question concept
        ctx_hv = dist.context_hv(stem_tok + concept.split())
        scored = []
        for lbl, txt in r["choices"]:
            nct += 1
            # (1) typed KG via the learned ask-role relation
            typed = 0.0
            for rel, w in pred:
                if rel in rels_between(concept, txt):
                    typed += 3.0 * w
                for sc in stems:
                    if sc != txt and rel in rels_between(sc, txt):
                        typed += 1.0 * w
            # (2) untyped ConceptNet relatedness
            untyped = relatedness(adj, txt, concept)
            for sc in stems:
                if sc != txt and sc != concept:
                    untyped += 0.5 * relatedness(adj, txt, sc)
            # (3) distributional fit (LLM-style) from the trained cooccur channel
            distr = dist.score(txt.split(), ctx_hv)
            scored.append((lbl, txt,
                           6.0 * typed + untyped + 40.0 * distr))
        best = max(scored, key=lambda x: x[2])
        hit = best[2] > 0.0 and best[0] == r["answer"]
        if best[2] > 0.0:
            answered += 1
            correct_ans += hit
        correct += hit
        if dump is not None:
            gold = next((t for l, t in r["choices"] if l == r["answer"]), "?")
            ranked = sorted(scored, key=lambda x: -x[2])
            dump.write(f"[{'OK ' if hit else 'X  '}] Q{qi+1}: {r['stem']}\n")
            dump.write(f"      concept={concept!r}  asked={pred}\n")
            dump.write("      " + ", ".join(f"{t}={s:.2f}" for _l, t, s in ranked) + "\n")
            dump.write(f"      pick={best[1]!r}  gold={gold!r}\n\n")
    if dump is not None:
        dump.close()
        print(f"per-question trace -> {args.dump}\n")

    n = len(rows)
    rand = n / nct
    print("=" * 60)
    print("  CommonsenseQA (dev, 1221 Q) -- Ikigai organism + ask-role")
    print("=" * 60)
    print(f"  accuracy (strict, abstain=wrong) : {correct}/{n} = {correct/n:.1%}")
    if answered:
        print(f"  accuracy among answered          : "
              f"{correct_ans}/{answered} = {correct_ans/answered:.1%}")
    print(f"  answered (had signal)            : {answered}/{n} = {answered/n:.1%}")
    print(f"  random baseline                  : {rand:.1%}")
    print("-" * 60)
    print("  reference: ConceptNet ~40-55% | LMs 60-75% | LLM ~80% | human ~89%")
    print("=" * 60)


if __name__ == "__main__":
    main()
