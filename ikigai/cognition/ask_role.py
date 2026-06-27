"""ikigai.cognition.ask_role -- Pack 331: the INTERROGATIVE ('ask') role.

The organism already maps every relation to a question template
(relation -> question, e.g. atlocation -> "what is the atlocation of {e}").
What it lacks is the INVERSE: question -> relation -- recognising, from a
natural-language question, WHICH typed relation is being asked for.  That is
the missing native primitive behind multiple-choice commonsense QA: the
knowledge (typed ConceptNet edges) is already in the substrate; the gap is
knowing which edge the question wants.

This module adds that as a learned channel, not a hardcoded table.  A question's
cue tokens are bound -- through the organism's own MultiRoleMemory, under the
'ask' role -- to the relation that question is asking for.  Supervision comes
from data: given (question, source concept, correct answer), the relation is
whichever typed edge actually connects concept -> answer in the graph.  At test
the relation is recalled from the stem's cues by substrate superposition.

No cue->relation list is written by hand.  Even function words ('where', 'why',
'used') are kept as cues -- they ARE the signal -- and the substrate learns
their relation weight from how often each co-occurs with each relation; the
belief-field mean-subtraction in recall downweights uninformative cues
automatically.
"""
import re

import numpy as np


def _tokens(text):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split() if t]


class AskRole:
    ROLE = "ask"

    def __init__(self, organism):
        self.org = organism
        self.mr = organism.unified
        self.mr.ensure_role(self.ROLE)
        self._relations = set()

    def cues(self, stem):
        """Cue features of a question: every token plus adjacent bigrams (so
        multiword cues like 'used for' / 'serves as' are learnable).  Generic;
        nothing relation-specific."""
        toks = _tokens(stem)
        cues = list(dict.fromkeys(toks))
        cues += [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
        return cues

    # ---- learning (from data) -------------------------------------------

    def learn(self, stem, relation):
        """Bind this question's cues to `relation` in the substrate.  Repeated
        calls accumulate (superposition), so a cue that recurs with a relation
        recalls it more strongly -- frequency weighting for free."""
        relation = str(relation).strip().lower()
        if not relation:
            return
        self._relations.add(relation)
        for c in self.cues(stem):
            self.mr.relate(c, self.ROLE, relation)

    # ---- recall (question -> relation) ----------------------------------

    def predict(self, stem, candidates=None, top_k=3):
        """Recall the ranked relations a question is asking for.  Sums each
        cue's substrate recall against the candidate relation keys (belief-field
        subtracted per cue), so informative cues dominate and frequent
        uninformative ones wash out.  Returns [(relation, weight), ...]."""
        cands = list(candidates) if candidates is not None else list(self._relations)
        if not cands:
            return []
        K = np.stack([self.mr.ck.key(c) for c in cands])      # (n, d)
        votes = np.zeros(len(cands), dtype=np.float64)
        for c in self.cues(stem):
            if c not in self.mr._seen:
                continue
            r = self.mr.recall(c, self.ROLE)
            sims = np.real(K @ np.conj(r)) / self.mr.d
            if len(cands) > 1:
                sims = sims - float(sims.mean())              # belief field
            votes += np.maximum(sims, 0.0)
        order = np.argsort(-votes)
        return [(cands[i], float(votes[i])) for i in order[:top_k] if votes[i] > 0]
