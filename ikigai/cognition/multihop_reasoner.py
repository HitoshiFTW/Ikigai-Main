"""
ikigai.cognition.multihop_reasoner -- the reasoning engine.

The thesis in one object: given a knowledge graph and a question, REASON to the
answer the way a transformer cannot -- by EXACT multi-hop derivation over typed
relations, with coincidence-detection convergence and calibrated honest-unknown.
No hallucinated chains.  No frozen knowledge.  The reasoning is separable from
the knowledge: feed it a small KB or a frontier-scale one -- the engine is the
same, and the engine is the edge.

Pipeline (each stage a named, testable step):
    comprehend(question)  -> (seed concepts, intent relations)
        seeds   : concepts the question names that exist in the KB
        intent  : the relation(s) the question asks for (the learned ask-role,
                  or supplied)
    derive(seed)          -> spreading activation over the typed graph
    score(question, cand) -> typed relation-path features (1-hop + 2-hop pairs)
                             fused with convergence (coincidence across seeds),
                             combined by a LEARNED weight vector
    answer(...)           -> best candidate + confidence; abstain below the
                             calibrated boundary (honest-unknown)

The KB is a typed adjacency: node -> {neighbour: {relation: weight}}.  Built
from any (subject, relation, object) stream (ConceptNet / Wikidata / live
facts).  Reasoning is read-only over it -- derive-not-store: the chains are
computed, never stored.
"""
import math
import re
from collections import defaultdict

import numpy as np

_WORD = re.compile(r"[^a-z0-9 ]")


def _toks(s):
    return [t for t in _WORD.sub(" ", str(s).lower()).split() if t]


class MultiHopReasoner:
    # default feature weights (a reasonable prior; call fit() to learn them)
    _DEFAULT_W = {
        "conv": 1.0, "add": 0.2, "frac": 0.5, "deg": -0.15,
        "rel1": 0.8, "rel2": 0.6, "intent": 1.2,
    }

    def __init__(self, organism=None, hops=2, decay=0.5, max_deg=64,
                 n_top_rel=25):
        self.org = organism
        self.hops = int(hops)
        self.decay = float(decay)
        self.max_deg = int(max_deg)
        self.n_top_rel = int(n_top_rel)
        self.adj = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.flat = defaultdict(dict)
        self.nodes = set()
        self.rels = []
        self._ridx = {}
        self.w = dict(self._DEFAULT_W)
        self._scorer = None              # optional plug-in (e.g. fitted GBM)
        self._spread_cache = {}
        self.abstain_margin = 0.0        # calibrated honest-unknown boundary

    # ---- knowledge ------------------------------------------------------
    def load_triples(self, triples):
        """Build the typed adjacency (the KB) from a (subj, rel, obj) stream."""
        rc = defaultdict(int)
        for s, r, o in triples:
            s = str(s).strip().lower(); r = str(r).strip().lower(); o = str(o).strip().lower()
            if not (s and r and o) or s == o:
                continue
            self.adj[s][o][r] += 1.0
            self.adj[o][s][r] += 1.0
            self.flat[s][o] = self.flat[s].get(o, 0.0) + 1.0
            self.flat[o][s] = self.flat[o].get(s, 0.0) + 1.0
            rc[r] += 1
        for k in list(self.flat):
            if len(self.flat[k]) > self.max_deg:
                keep = dict(sorted(self.flat[k].items(), key=lambda x: -x[1])[:self.max_deg])
                self.flat[k] = keep
                self.adj[k] = {x: self.adj[k][x] for x in keep if x in self.adj[k]}
        self.nodes = set(self.flat)
        self.rels = [r for r, _ in sorted(rc.items(), key=lambda x: -x[1])[:self.n_top_rel]]
        self._ridx = {r: i for i, r in enumerate(self.rels)}
        self._spread_cache.clear()
        return self

    def set_adjacency(self, adj, flat, nodes, rels):
        """Attach a prebuilt typed adjacency (e.g. from a fast cache loader)."""
        self.adj, self.flat, self.nodes = adj, flat, set(nodes)
        self.rels = list(rels)[:self.n_top_rel]
        self._ridx = {r: i for i, r in enumerate(self.rels)}
        self._spread_cache.clear()
        return self

    # ---- comprehend -----------------------------------------------------
    def comprehend(self, question, concept=None):
        """Question -> (seed concepts in the KB, intent relations)."""
        tk = _toks(question)
        seeds = []
        for n in (3, 2, 1):
            for i in range(len(tk) - n + 1):
                g = " ".join(tk[i:i + n])
                if g in self.nodes:
                    seeds.append(g)
        if concept:
            for w in [str(concept).lower()] + str(concept).lower().split():
                if w in self.nodes:
                    seeds.append(w)
        seeds = list(dict.fromkeys(seeds))
        intent = set()
        if self.org is not None and hasattr(self.org, "ask_relation"):
            try:
                intent = {r for r, _ in self.org.ask_relation(
                    question, candidates=self.rels, top_k=3)}
            except Exception:
                intent = set()
        return seeds, intent

    # ---- derive ---------------------------------------------------------
    def derive(self, seed):
        """Spreading activation from a seed over the typed graph (cached)."""
        c = self._spread_cache.get(seed)
        if c is not None:
            return c
        act = {seed: 1.0}; frontier = {seed: 1.0}
        for _ in range(self.hops):
            nxt = defaultdict(float)
            for node, a in frontier.items():
                nb = self.flat.get(node)
                if not nb:
                    continue
                tot = sum(nb.values()) or 1.0
                for m, wgt in nb.items():
                    nxt[m] += a * self.decay * (wgt / tot)
            frontier = nxt
            for m, a in nxt.items():
                act[m] = act.get(m, 0.0) + a
        self._spread_cache[seed] = act
        return act

    def _rels_between(self, a, b):
        out = set()
        if a in self.adj and b in self.adj[a]:
            out |= set(self.adj[a][b])
        if b in self.adj and a in self.adj[b]:
            out |= set(self.adj[b][a])
        return out

    # ---- score ----------------------------------------------------------
    def features(self, concept, seeds, per, intent, cand):
        cwords = cand.split() + ([cand] if cand in self.nodes else [])
        acts = [max((a.get(w, 0.0) for w in cwords), default=0.0) for a in per]
        conv = sum(math.log(x + 1e-6) for x in acts)
        add = sum(acts)
        frac = len([1 for x in acts if x > 0]) / max(len(acts), 1)
        deg = math.log(max((len(self.flat.get(w, {})) for w in cwords), default=0) + 1)
        srcs = ([concept] if concept else []) + seeds
        NR = len(self.rels)
        r1 = np.zeros(NR)                 # 1-hop typed: src --r--> cand
        r2 = np.zeros(NR * NR)            # 2-hop typed PAIR: src --ra--> mid --rb--> cand
        intent_hit = 0.0
        for s in srcs:
            for cw in cwords:
                for r in self._rels_between(s, cw):
                    if r in self._ridx:
                        r1[self._ridx[r]] += 1.0
                        if r in intent:
                            intent_hit += 1.0
            nb_s = self.adj.get(s, {})
            for mid, rels_sm in nb_s.items():
                nb_m = self.adj.get(mid)
                if not nb_m:
                    continue
                for cw in cwords:
                    if cw not in nb_m:
                        continue
                    for ra in rels_sm:
                        ia = self._ridx.get(ra)
                        if ia is None:
                            continue
                        for rb in nb_m[cw]:
                            ib = self._ridx.get(rb)
                            if ib is not None:
                                r2[ia * NR + ib] += 1.0
        return {"conv": conv, "add": add, "frac": frac, "deg": deg,
                "rel1": float(r1.sum()), "rel2": float(r2.sum()),
                "intent": intent_hit, "_r1": r1, "_r2": np.log1p(r2)}

    def _combine(self, f):
        if self._scorer is not None:
            return float(self._scorer(f))
        return sum(self.w.get(k, 0.0) * f[k] for k in self.w)

    # ---- answer ---------------------------------------------------------
    def answer_mc(self, question, choices, concept=None):
        """Multiple-choice: return (label, confidence, abstain).  choices is a
        list of (label, text).  Reasoning over the KB; honest-unknown below the
        calibrated margin."""
        seeds, intent = self.comprehend(question, concept)
        if not seeds:
            return None, 0.0, True
        per = [self.derive(s) for s in seeds]
        scored = []
        for lbl, txt in choices:
            f = self.features(concept, seeds, per, intent, txt)
            scored.append((lbl, self._combine(f)))
        scored.sort(key=lambda x: -x[1])
        best, top = scored[0]
        margin = top - (scored[1][1] if len(scored) > 1 else top)
        abstain = margin < self.abstain_margin
        return best, float(margin), abstain

    # ---- learn ----------------------------------------------------------
    def fit(self, examples, l2=1e-3, lr=0.3, epochs=200):
        """Learn the feature weights from labelled examples
        [(question, concept, choices, gold_label)] by simple logistic GD over
        the named scalar features (no external deps).  For stronger nonlinear
        scoring, plug a fitted model via set_scorer()."""
        keys = list(self.w.keys())
        rows = []
        for question, concept, choices, gold in examples:
            seeds, intent = self.comprehend(question, concept)
            if not seeds:
                continue
            per = [self.derive(s) for s in seeds]
            grp = []
            for lbl, txt in choices:
                f = self.features(concept, seeds, per, intent, txt)
                grp.append(([f[k] for k in keys], 1.0 if lbl == gold else 0.0))
            if any(y for _, y in grp):
                rows.append(grp)
        X = np.array([fx for grp in rows for fx, _ in grp], dtype=np.float64)
        if len(X) == 0:
            return self
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = np.zeros(len(keys)); b = 0.0
        for _ in range(epochs):
            gw = np.zeros(len(keys)); gb = 0.0
            for grp in rows:
                xs = np.array([(np.array(fx) - mu) / sd for fx, _ in grp])
                ys = np.array([y for _, y in grp])
                logits = xs @ w + b
                p = 1.0 / (1.0 + np.exp(-(logits - logits.max())))
                gw += xs.T @ (p - ys); gb += float((p - ys).sum())
            w -= lr * (gw / len(rows) + l2 * w); b -= lr * gb / len(rows)
        self.w = {k: float(w[i] / sd[i]) for i, k in enumerate(keys)}
        return self

    def set_scorer(self, fn):
        """Plug a stronger scorer: fn(features_dict) -> float."""
        self._scorer = fn
        return self

    def _vec(self, f):
        """Full feature vector (scalars + typed-path relation histograms) for a
        nonlinear scorer."""
        base = np.array([f["conv"], f["add"], f["frac"], f["deg"],
                         f["intent"]], dtype=np.float64)
        return np.concatenate([base, f["_r1"], f["_r2"]])

    def fit_gbm(self, examples, max_iter=300, max_depth=4, lr=0.1):
        """Stronger nonlinear reasoner: gradient-boosting over the full typed-
        path feature vector, plugged in via set_scorer.  Needs scikit-learn.
        Same engine, stronger decision surface."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = []; Y = []
        for question, concept, choices, gold in examples:
            seeds, intent = self.comprehend(question, concept)
            if not seeds:
                continue
            per = [self.derive(s) for s in seeds]
            for lbl, txt in choices:
                f = self.features(concept, seeds, per, intent, txt)
                X.append(self._vec(f)); Y.append(1.0 if lbl == gold else 0.0)
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter, max_depth=max_depth, learning_rate=lr
        ).fit(np.array(X), np.array(Y))
        self.set_scorer(lambda f: float(clf.predict_proba(self._vec(f)[None, :])[0, 1]))
        self._gbm = clf
        return self
