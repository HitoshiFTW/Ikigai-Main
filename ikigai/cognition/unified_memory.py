"""
ikigai.cognition.unified_memory -- Day 91.  THE ONE MEMORY.

The organism had ~7 stores scattered across integrate.py (SDM co-occurrence, VSA role-binding, RHC
residue addressing, factored meaning codes, an exact anchor cache, grounding modules) plus a wall of
methods.  This is the single object that unifies them over the ONE shared identity space (ck): every
memory mechanism, one entry, one persistence, one continuous self.

It does NOT re-implement storage -- it OWNS the proven substrate and exposes it coherently:

  SEMANTIC   .similar / .neighbors / .observe / .emit   -> factored meaning (SDM/LSH, past the knee)
  RELATIONAL .relate / .query                            -> unified roles (VSA bind/unbind)
  EXACT      .fact / .knows / .derive                    -> RHC fact store (1e9 address-tuples)
  EPISODIC   .kv_put / .kv_get                            -> in-substrate RHC cache
  SELF       .remember / .recall / .mood / .people / .goals -> autobiographical (people, goals,
                                                             emotions, events)

One `state_dict` persists the self (episodes + lexicon + mood); the semantic / exact / relational
banks persist through their own .ikg hooks.  Reload restores the whole self.
"""

import time as _time


class UnifiedMemory:
    """One memory over the shared substrate: SDM + VSA + RHC + factored + cache + the self."""

    KINDS = ('fact', 'person', 'event', 'goal', 'feeling')

    def __init__(self, org):
        self.org = org
        self._lex = {}            # string -> id   (free RHC-addressable lexicon, shared by facts/self)
        self._rev = {}            # id -> string
        self.episodes = []        # autobiographical records (durable, tiny)
        self._by_who = {}; self._by_goal = {}; self._by_kind = {}
        self._mood_v = 0.0; self._mood_a = 0.0; self._mood_decay = 0.85

    # ── shared identity: one free lexicon over the RHC address space ──────────
    def id_of(self, name):
        if name is None:
            return None
        s = str(name).strip().lower()
        i = self._lex.get(s)
        if i is None:
            i = len(self._lex); self._lex[s] = i; self._rev[i] = s
        return i

    def name_of(self, i):
        return self._rev.get(i)

    # ── SEMANTIC (factored meaning) ──────────────────────────────────────────
    def observe(self, docs):
        """Feed co-occurrence into the semantic (factored) bank; then .consolidate()."""
        return self.org.observe_meaning(docs)

    def consolidate(self):
        return self.org.consolidate_meaning(drop_acc=False)

    def similar(self, a, b):
        """Distributional similarity (factored -> unified cooccur -> flat)."""
        return self.org.semantic_sim(a, b)

    def neighbors(self, word, candidates, k=5):
        return self.org.factored.neighbors(word, candidates, k=k)

    def emit(self, tokens, k=3):
        """Emit a grounded word by composing meaning (decodable codebook)."""
        return self.org.emit_meaning(tokens, k=k)

    # ── RELATIONAL (VSA role-binding over the unified substrate) ─────────────
    def relate(self, subj, role, obj):
        return self.org.unified.relate(str(subj).lower(), role, str(obj).lower())

    def query(self, subj, role, candidates=None):
        best, _ = self.org.unified.query(str(subj).lower(), role, candidates)
        return best

    # ── EXACT (RHC fact store, 1e9 address-tuples) ───────────────────────────
    def fact(self, subj, rel, obj):
        """Store an exact fact by NAME (auto-lexicon'd to RHC ids).  Consequences derived."""
        return self.org.store_fact(self.id_of(subj), self.id_of(rel), self.id_of(obj))

    def knows(self, subj, rel, obj):
        return self.org.fact_store.has(self.id_of(subj), self.id_of(rel), self.id_of(obj))

    def recall_fact(self, subj, rel):
        """Exact one-hop recall: the object name(s) for (subject, relation) from the RHC fact
        store.  This is how the ONE generator asks the ONE memory for a fact to realize."""
        oids = self.org.fact_store.objects_of(self.id_of(subj), self.id_of(rel))
        return [self.name_of(o) for o in oids]

    def derive(self, subj, rel, max_hops=8):
        """Transitive closure of a relation from a subject (derived, stored zero)."""
        chain = self.org.derive_facts(self.id_of(subj), self.id_of(rel), max_hops=max_hops)
        return [self._rev.get(o) for _, o in chain]

    # ── DERIVE (deep) -- the compositional engine, unified behind the one memory ──
    @property
    def derive_engine(self):
        """The rich derive-not-store engine (inheritance, invented relations, transitive,
        analogy) -- fronted here so ONE memory exposes exact recall AND deep derivation."""
        return self.org.general_reasoner.derive_engine

    def teach(self, triples, discover=True, min_support=4, min_conf=0.8, self_compress=False):
        """Teach (s,r,o) triples into BOTH banks at once: the exact RHC fact store (one-shot
        exact recall) AND the compositional derive engine (inheritance / invented relations /
        transitive closure).  One teach, both powers -- the single door for knowledge."""
        triples = [tuple(str(x).strip().lower() for x in t) for t in triples]
        for s, r, o in triples:
            self.fact(s, r, o)                                  # exact RHC
        return self.org.ingest_triples(triples, discover=discover, min_support=min_support,
                                       min_conf=min_conf, self_compress=self_compress)

    def inherited(self, attr, entity):
        """An attribute derived by climbing the taxonomy (never stored on the entity)."""
        return self.derive_engine.inherited_atom(attr, entity)

    def invent(self, min_support=2):
        """Grow invented relations (R3 = R1 o R2) from what is stored."""
        return self.org.invent_relations(min_support=min_support)

    def describe(self, entity):
        """EVERYTHING the memory can DERIVE about an entity -- transitive class chain +
        inherited attributes + invented-relation values -- as (relation, value) pairs, derived
        not stored.  This is the MESSAGE source for grounded open-ended generation."""
        eng = self.derive_engine
        e = str(entity).strip().lower()
        out, seen = [], set()
        for link in ('isa', 'subclassof'):
            chain = eng.transitive_reach(link, e) or []
            for anc in chain[1:]:
                if (link, anc) not in seen:
                    seen.add((link, anc)); out.append((link, anc))
        for r in eng.learned_rules:
            if r.get('type') == 'inheritance':
                a = r.get('attr')
                if a and a != '*':
                    v = eng.inherited_atom(a, e)
                    if v and (a, v) not in seen:
                        seen.add((a, v)); out.append((a, v))
        for r in eng.invented_relations():
            v = self.org.derive_composed(r['name'], e)
            if v and (r['name'], v) not in seen:
                seen.add((r['name'], v)); out.append((r['name'], v))
        return out

    # ── EPISODIC key->value (in-substrate RHC cache) ─────────────────────────
    def kv_put(self, key, value):
        return self.org.remember_kv(key, value)

    def kv_get(self, key, candidates):
        return self.org.recall_kv(key, candidates)

    # ── SELF (autobiographical: people, goals, emotions, events) ─────────────
    def remember(self, text, kind='event', who=None, about=None, relation=None,
                 emotion=None, goal=None, t=None):
        """One memory across all banks: content -> SEMANTIC; (who,relation,about) -> EXACT fact;
        who/goal -> lexicon ids; emotion=(valence,arousal); tagged episode -> SELF."""
        toks = [w for w in str(text).lower().replace('.', ' ').split() if w]
        self.org.observe_meaning([toks])                              # semantic
        who_id = self.id_of(who) if who else None
        goal_id = self.id_of(goal) if goal else None
        if who and about and relation:
            self.fact(who, relation, about)                          # exact knowledge
        v, a = (emotion if emotion else (0.0, 0.0))
        rec = {'i': len(self.episodes), 'kind': kind, 'text': str(text), 'tokens': toks,
               'who': who_id, 'goal': goal_id, 'v': float(v), 'a': float(a),
               't': float(t if t is not None else _time.time())}
        self.episodes.append(rec)
        if who_id is not None: self._by_who.setdefault(who_id, []).append(rec['i'])
        if goal_id is not None: self._by_goal.setdefault(goal_id, []).append(rec['i'])
        self._by_kind.setdefault(kind, []).append(rec['i'])
        self._mood_v = self._mood_decay * self._mood_v + (1 - self._mood_decay) * float(v)
        self._mood_a = self._mood_decay * self._mood_a + (1 - self._mood_decay) * float(a)
        return rec['i']

    def recall(self, query=None, by='meaning', who=None, goal=None, kind=None, k=5):
        """Recall episodes by meaning / person / goal / kind / emotion."""
        if by == 'person':
            return [self.episodes[i] for i in self._by_who.get(self.id_of(who), [])][:k]
        if by == 'goal':
            return [self.episodes[i] for i in self._by_goal.get(self.id_of(goal), [])][:k]
        if by == 'kind':
            return [self.episodes[i] for i in self._by_kind.get(kind, [])][:k]
        if by == 'emotion':
            tv, ta = (query if isinstance(query, tuple) else (self._mood_v, self._mood_a))
            return sorted(self.episodes, key=lambda r: abs(r['v'] - tv) + abs(r['a'] - ta))[:k]
        if self.org.factored._acc:
            self.org.consolidate_meaning(drop_acc=False)
        qtok = [w for w in str(query).lower().split() if w] if query else []
        fm = self.org.factored
        def score(rec):
            sims = [fm.sim(q, w) for q in qtok for w in rec['tokens']]
            sims = [s for s in sims if s is not None]
            return sum(sims) / len(sims) if sims else -9
        return sorted(self.episodes, key=lambda r: -score(r))[:k]

    def mood(self):
        return {'valence': round(self._mood_v, 3), 'arousal': round(self._mood_a, 3)}

    def people(self):
        return [self._rev[i] for i in self._by_who]

    def goals(self):
        return [self._rev[i] for i in self._by_goal]

    # ── introspection + persistence ──────────────────────────────────────────
    @property
    def n_memories(self):
        return len(self.episodes)

    def status(self):
        return {'memories': len(self.episodes), 'people': len(self._by_who),
                'goals': len(self._by_goal), 'lexicon': len(self._lex),
                'facts': self.org.fact_store.n_facts if getattr(self.org, '_fact_store', None) else 0,
                'semantic_words': len(self.org.factored.codes) if getattr(self.org, '_factored', None) else 0,
                'mood': self.mood()}

    def state_dict(self):
        return {'lex': self._lex, 'episodes': self.episodes,
                'mood_v': self._mood_v, 'mood_a': self._mood_a}

    def load_state(self, st):
        self._lex = dict(st.get('lex', {}))
        self._rev = {i: s for s, i in self._lex.items()}
        self.episodes = list(st.get('episodes', []))
        self._by_who = {}; self._by_goal = {}; self._by_kind = {}
        for rec in self.episodes:
            if rec.get('who') is not None: self._by_who.setdefault(rec['who'], []).append(rec['i'])
            if rec.get('goal') is not None: self._by_goal.setdefault(rec['goal'], []).append(rec['i'])
            self._by_kind.setdefault(rec['kind'], []).append(rec['i'])
        self._mood_v = float(st.get('mood_v', 0.0)); self._mood_a = float(st.get('mood_a', 0.0))
        return len(self.episodes)
