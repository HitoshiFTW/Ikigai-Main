"""
ikigai.cognition.vs_fsm -- Pack 225 Vector Symbolic Finite State Machine.

Per Day 67 external research (Cotteret et al. 2024, arXiv:2212.01196):
FSM-style generation on a flat VSA-SDM substrate requires explicit
state-dependent attractor switching. Standard bigram next-role does
state -> token; VS-FSM adds:

  1. Sleep-time ISA-ABSTRACTION: for every observed (a, next, b) transition,
     also write (isa_parent(a), next, b) and (isa_parent(a), next, isa_parent(b)).
     Concrete + abstract levels co-exist in the substrate.

  2. RESONATOR DECODING at generation: use multirole_memory.resonator_recall
     to clean up the noisy next-role recall against candidate tokens.
     Pulls signal past the 1/sqrt(K) cosine ceiling.

  3. STATE-DEPENDENT step: optionally condition on (prev_token, current_token)
     pair via Hadamard bind for trigram FSM behavior.

Architectural commitment (Day 67):
  - AWAKE = just record transitions via observe_chain(tokens).
  - SLEEP = abstract via isa parents + reinforce.
  - GENERATE = walk attractors via Resonator-cleaned next-role queries.

NO online grammar induction. NO one-shot learning. The substrate gets sharper
across sleep cycles like a human.
"""

import numpy as np


class VSFiniteStateMachine:
    """FSM-style generation with isa-abstracted transitions + Resonator decode.

    Awake API: observe_chain(tokens) -- bigram + optional trigram transitions.
    Sleep API: lift_to_abstract() -- iterate stored transitions, write isa-
              parent copies for every concrete transition. Strengthens
              grammatical state attractors without backprop.
    Generate API: step() -- single resonator-cleaned next-token prediction.
                  generate(seed, max_tokens) -- autoregressive walk.
    """

    NEXT_ROLE = 'next'        # use existing role to avoid substrate growth
    PREV_ROLE = 'next2'       # use existing trigram role

    def __init__(self, organism):
        self.org = organism
        self.mr = organism.unified
        self.transition_count = 0
        self.abstracted_count = 0
        self.skipped = 0
        # Pack 226: schema induction state
        self.schemas = []           # list of anti-unified schemas
        self.schema_transitions = 0
        self._isa_inverse = None    # parent -> set(children) cache
        # Pack 226: own role for schema-level state transitions (separate from
        # concrete next role to avoid polluting bigram statistics).
        if 'schema_next' not in self.mr.roles:
            import numpy as _np
            rng = _np.random.default_rng(22601)
            ph = rng.uniform(-_np.pi, _np.pi, self.mr.d).astype(_np.float32)
            self.mr.roles['schema_next'] = _np.exp(1j * ph).astype(_np.complex64)

    # ── awake: record transitions only ────────────────────────────────────
    def observe_chain(self, tokens, n_reinforce=3, do_trigram=True,
                       surprise_gate=True):
        """Awake-time recording. Writes (a, next, b) for each adjacent pair.
        If do_trigram, also writes (a, next2, c) for adjacent triples.
        NO abstraction here -- that's a sleep job.

        Pack 238: surprise_gate=True scales n_reinforce by
        min(write_strength(a), write_strength(b)) so stopwords get ~0.05x
        writes and rare semantic words get ~1.0x. Protects the next/next2
        banks from being flooded by 'the'/'of'/'and' transitions.
        """
        if not tokens or len(tokens) < 2:
            return 0
        toks = [t for t in tokens if t]
        if surprise_gate:
            self.mr.observe_unigrams(toks)
        n = 0
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i+1]
            if not a or not b: continue
            if surprise_gate:
                ws = min(self.mr.write_strength(a), self.mr.write_strength(b))
                eff = max(1, int(round(n_reinforce * ws)))
            else:
                eff = n_reinforce
            for _ in range(eff):
                self.mr.relate(a, self.NEXT_ROLE, b)
            n += 1
        if do_trigram and len(toks) >= 3:
            for i in range(len(toks) - 2):
                a, c = toks[i], toks[i+2]
                if not a or not c: continue
                if surprise_gate:
                    ws = min(self.mr.write_strength(a),
                             self.mr.write_strength(c))
                    eff = max(1, int(round(n_reinforce * ws)))
                else:
                    eff = n_reinforce
                for _ in range(eff):
                    self.mr.relate(a, self.PREV_ROLE, c)
        self.transition_count += n
        return n

    # ── sleep: abstract via isa parents ───────────────────────────────────
    def _isa_parent(self, word):
        """Find substrate-recorded isa parent of word. Returns None if unknown
        or self-loop."""
        if not word: return None
        try:
            cands = self.mr.targets('isa')
            if not cands:
                return None
            best, score = self.mr.query(word, 'isa', cands)
        except Exception:
            return None
        if not best or best == word or score <= 0:
            return None
        return best

    def lift_to_abstract(self, n_reinforce=2, only_for_words=None, verbose=False):
        """SLEEP-TIME abstraction. For each (a, NEXT_ROLE, b) transition in the
        substrate, also write (isa_parent(a), NEXT_ROLE, b) and (a, NEXT_ROLE,
        isa_parent(b)) and (isa_parent(a), NEXT_ROLE, isa_parent(b)).

        only_for_words: limit to a vocab subset (faster for testing).
        """
        words = list(only_for_words) if only_for_words else \
                list(self.mr._role_targets.get(self.NEXT_ROLE, set()))
        n_lifted = 0
        for a in words:
            a_parent = self._isa_parent(a)
            # All seen next-targets of a
            try:
                cands = self.mr._role_targets.get(self.NEXT_ROLE, set())
                if not cands: continue
                # Score each candidate via recall for a; keep top-N.
                best, score = self.mr.query(a, self.NEXT_ROLE, cands)
            except Exception:
                continue
            if not best or score <= 0:
                continue
            b_parent = self._isa_parent(best)
            if a_parent and a_parent != a:
                for _ in range(n_reinforce):
                    self.mr.relate(a_parent, self.NEXT_ROLE, best)
                n_lifted += 1
            if b_parent and b_parent != best:
                for _ in range(n_reinforce):
                    self.mr.relate(a, self.NEXT_ROLE, b_parent)
                n_lifted += 1
            if a_parent and b_parent and a_parent != a and b_parent != best:
                for _ in range(n_reinforce):
                    self.mr.relate(a_parent, self.NEXT_ROLE, b_parent)
                n_lifted += 1
        self.abstracted_count += n_lifted
        if verbose:
            print(f'    VS-FSM lifted {n_lifted} abstract transitions')
        return n_lifted

    # ── Pack 226 -- offline abductive schema induction ────────────────────
    def _build_isa_inverse(self):
        """Build parent -> set(children) map from current substrate state."""
        inv = {}
        seen = set(self.mr._cooccur_seen) | set(self.mr._seen)
        cands = self.mr.targets('isa')
        if not cands:
            self._isa_inverse = inv
            return inv
        for w in seen:
            try:
                best, score = self.mr.query(w, 'isa', cands)
                if best and best != w and score > 0:
                    inv.setdefault(best, set()).add(w)
            except Exception:
                continue
        self._isa_inverse = inv
        return inv

    def induce_schemas(self, exposure_buffer=None, texts=None,
                        max_chains=1000, min_chain_len=3, max_chain_len=10,
                        n_reinforce=3, verbose=False):
        """Pack 226 -- abductive schema crystallization during sleep.

        Architectural commitment: this is the ONLY learning phase. No
        grammar discovered during awake reads. Anti-unification across
        many chains finds the structural motifs that bigram statistics
        cannot.

        Procedure:
          1. Collect chains from exposure_buffer.snapshot() OR from texts arg.
          2. Replace each token with its isa parent (-> abstract chain).
          3. Group by abstract chain length.
          4. Anti-unify within each group -> schema with SLOT markers.
          5. Crystallize: write (a, schema_next, b) for consecutive
             non-SLOT positions in each schema. These are GRAMMATICAL
             state transitions, not concrete bigrams.
        """
        from ikigai.cognition.schema_inducer import anti_unify, SLOT

        # Collect text sources
        chains = []
        if texts:
            for text in texts:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if exposure_buffer is not None and len(chains) < max_chains:
            snapshot = exposure_buffer.snapshot()
            for text, _, _ in snapshot:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break

        if not chains:
            if verbose: print('    no chains to induce from')
            return {'chains': 0, 'schemas': 0, 'transitions': 0}

        # Build isa inverse for downstream generation
        self._build_isa_inverse()

        # Abstract every chain via isa parents
        abstracted = []
        for chain in chains:
            ach = [self._isa_parent(t) or t for t in chain]
            abstracted.append(ach)

        # Group by length, anti-unify within each group
        by_len = {}
        for ach in abstracted:
            by_len.setdefault(len(ach), []).append(ach)

        new_schemas = []
        n_transitions = 0
        for length, group in by_len.items():
            if len(group) < 2: continue
            schema = list(group[0])
            for ach in group[1:]:
                schema = anti_unify(schema, ach)
            n_fixed = sum(1 for t in schema if t is not SLOT)
            if n_fixed < 2: continue        # too generic to be useful
            new_schemas.append(schema)
            if verbose:
                printable = [t if t is not SLOT else '_' for t in schema]
                print(f'    schema (len={length}, support={len(group)}): '
                      f'{printable}')

            # Crystallize: substrate-write transitions between consecutive
            # non-SLOT positions. Skipping SLOTs preserves abstract structure.
            last_fixed = None
            for tok in schema:
                if tok is SLOT: continue
                if last_fixed is not None:
                    for _ in range(n_reinforce):
                        self.mr.relate(last_fixed, 'schema_next', tok)
                    n_transitions += 1
                last_fixed = tok

        self.schemas.extend(new_schemas)
        self.schema_transitions += n_transitions
        return {
            'chains': len(chains),
            'schemas': len(new_schemas),
            'transitions': n_transitions,
        }

    # ── Pack 228 -- Trigram-conditioned delta-rule sleep refinement ──────
    def iterative_refine_trigram(self, chains, n_epochs=8, predict_iters=3,
                                   delta_strength=3, hebbian_strength=1,
                                   shuffle=True, verbose=False, tol=0.001):
        """Pack 228 -- trigram-conditioned delta rule.

        Like iterative_refine but step uses (prev_token, current_token)
        joint state. Disambiguates per-state entropy: 'cat' under (small, cat)
        learns different next than under (the, cat).

        Delta writes target BOTH next role (bigram) AND next2 role (trigram)
        on errors. Push-pull operates on both channels.

        Returns per-epoch stats.
        """
        import random as _random
        rng = _random.Random(2280)
        stats = []
        prev_acc = -1.0
        for epoch in range(n_epochs):
            order = list(range(len(chains)))
            if shuffle:
                rng.shuffle(order)
            n_correct = 0; n_total = 0; n_delta = 0
            for ci in order:
                chain = chains[ci]
                for i in range(len(chain) - 1):
                    a = chain[i]; b = chain[i+1]
                    prev = chain[i-1] if i >= 1 else None
                    if not a or not b: continue
                    candidates = list(self.mr._role_targets.get(
                        self.NEXT_ROLE, set()))
                    if not candidates:
                        for _ in range(hebbian_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                            if prev:
                                self.mr.relate(prev, self.PREV_ROLE, b)
                        continue
                    # Trigram-aware predict (uses both next + next2 recall)
                    pred, score = self.step(a, prev_token=prev,
                                              candidates=candidates,
                                              n_iters=predict_iters,
                                              beta=8.0, top_k=1)
                    n_total += 1
                    if pred == b:
                        n_correct += 1
                        for _ in range(hebbian_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                            if prev:
                                self.mr.relate(prev, self.PREV_ROLE, b)
                    else:
                        # Delta on BOTH bigram + trigram channels
                        if pred is not None:
                            for _ in range(delta_strength):
                                self.mr.unrelate(a, self.NEXT_ROLE, pred)
                                if prev:
                                    self.mr.unrelate(prev, self.PREV_ROLE,
                                                       pred)
                        for _ in range(delta_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                            if prev:
                                self.mr.relate(prev, self.PREV_ROLE, b)
                        n_delta += 1
            acc = (n_correct / n_total) if n_total else 0.0
            stats.append({'epoch': epoch, 'accuracy': acc,
                          'correct': n_correct, 'total': n_total,
                          'delta_writes': n_delta})
            if verbose:
                print(f'    epoch {epoch+1:>2d}: acc={acc:.3f} '
                      f'({n_correct}/{n_total})  delta_writes={n_delta}')
            if prev_acc >= 0 and abs(acc - prev_acc) < tol:
                if verbose:
                    print(f'    converged at epoch {epoch+1}')
                break
            prev_acc = acc
        return stats

    # ── Pack 227 -- Iterative delta-rule sleep refinement ────────────────
    def iterative_refine(self, chains, n_epochs=5, predict_iters=3,
                          delta_strength=2, hebbian_strength=1,
                          shuffle=True, verbose=False, tol=0.001):
        """Pack 227 -- generalized delta rule for SDM sleep refinement.

        For each epoch, walk every stored chain. At each adjacent (a, b) pair,
        predict via current FSM state. If wrong, subtract wrong target HV
        (unrelate) + add correct target HV (relate). If right, small Hebbian
        reinforce. Iterate until prediction error stable.

        Tracks accuracy per epoch. Returns list of (epoch, accuracy, n_correct,
        n_total, n_delta_writes).

        Mathematically: this is the SDM equivalent of the perceptron delta
        rule. Pure substrate writes/unwrites (Kill-Stack #4 Reversible Writes
        primitive shipped Day 60 Pack 162). No gradients. No parameter updates.
        Just push-pull on the existing fixed counter banks.

        chains: list of lists of tokens.
        n_epochs: max refinement passes.
        predict_iters: Resonator iter count for each step's prediction.
        delta_strength: how many subtract+add writes per wrong prediction.
        hebbian_strength: how many reinforce writes per correct prediction.
        tol: stop when |epoch_acc - prev_acc| < tol.
        """
        import random as _random
        rng = _random.Random(2270)
        stats = []
        prev_acc = -1.0
        for epoch in range(n_epochs):
            order = list(range(len(chains)))
            if shuffle:
                rng.shuffle(order)
            n_correct = 0
            n_total = 0
            n_delta = 0
            for ci in order:
                chain = chains[ci]
                for i in range(len(chain) - 1):
                    a = chain[i]; b = chain[i+1]
                    if not a or not b: continue
                    # Predict next via current FSM step (uses Resonator).
                    candidates = list(self.mr._role_targets.get(
                        self.NEXT_ROLE, set()))
                    if not candidates:
                        # First epoch -- no prior transitions; bootstrap.
                        for _ in range(hebbian_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                        continue
                    pred, score = self.step(a, candidates=candidates,
                                              n_iters=predict_iters, beta=8.0,
                                              top_k=1)
                    n_total += 1
                    if pred == b:
                        n_correct += 1
                        # Small Hebbian reinforce.
                        for _ in range(hebbian_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                    else:
                        # Delta rule: subtract wrong + add right, both x delta_strength.
                        if pred is not None:
                            for _ in range(delta_strength):
                                self.mr.unrelate(a, self.NEXT_ROLE, pred)
                        for _ in range(delta_strength):
                            self.mr.relate(a, self.NEXT_ROLE, b)
                        n_delta += 1
            acc = (n_correct / n_total) if n_total else 0.0
            stats.append({
                'epoch': epoch,
                'accuracy': acc,
                'correct': n_correct,
                'total': n_total,
                'delta_writes': n_delta,
            })
            if verbose:
                print(f'    epoch {epoch+1:>2d}: acc={acc:.3f} '
                      f'({n_correct}/{n_total})  delta_writes={n_delta}')
            if prev_acc >= 0 and abs(acc - prev_acc) < tol:
                if verbose:
                    print(f'    converged at epoch {epoch+1}')
                break
            prev_acc = acc
        return stats

    # ── Pack 231 -- substrate-native HV clustering of abstract chains ────
    def _chain_hv(self, abstract_chain):
        """Build a positional-bound HV for an abstract chain. Each token
        gets bound to its position via the existing pos_0..pos_7 roles
        (Pack 199 NEW2). Renormed. Tokens past position 7 use a default
        secondary role binding via Hadamard with `pos_7`.

        Returns complex64[d].
        """
        import numpy as _np
        d = self.mr.d
        accum = _np.zeros(d, dtype=_np.complex64)
        for i, tok in enumerate(abstract_chain):
            if not tok: continue
            pos_role = f'pos_{min(i, 7)}'
            if pos_role not in self.mr.roles:
                # Substrate didn't have pos roles; bootstrap one anchor.
                rng = _np.random.default_rng(2310 + i)
                ph = rng.uniform(-_np.pi, _np.pi, d).astype(_np.float32)
                self.mr.roles[pos_role] = _np.exp(1j*ph).astype(_np.complex64)
            v = (self.mr.ck.key(tok) * self.mr.roles[pos_role]).astype(
                _np.complex64)
            accum = accum + v
        mag = float(_np.abs(accum).mean()) + 1e-9
        return (accum / mag).astype(_np.complex64)

    def cluster_abstract_chains(self, chains, sim_threshold=0.30,
                                  max_clusters=500, verbose=False):
        """Pack 231 -- greedy nearest-centroid clustering.

        For each chain:
          - abstract via isa parents
          - build positional-bound HV
          - find nearest cluster centroid by cosine sim
          - if sim > threshold, join (drift centroid toward chain)
          - else if room, spawn new cluster
          - else assign to nearest

        Returns (clusters, centroids) where clusters[i] is list of
        abstracted chains in cluster i.
        """
        import numpy as _np
        if self._isa_inverse is None:
            self._build_isa_inverse()
        centroids = []          # list of complex64[d]
        clusters = []           # list of lists of abstracted chains
        for chain in chains:
            abstract_chain = [self._isa_parent(t) or t for t in chain]
            chv = self._chain_hv(abstract_chain)
            if not centroids:
                centroids.append(chv)
                clusters.append([abstract_chain])
                continue
            K = _np.stack(centroids)        # (n_clusters, d)
            sims = _np.real(K @ _np.conj(chv)) / self.mr.d
            best = int(_np.argmax(sims))
            best_sim = float(sims[best])
            if best_sim > sim_threshold:
                # Drift centroid by simple average toward chain.
                n = len(clusters[best])
                new_c = ((n * centroids[best] + chv) / (n + 1)).astype(
                    _np.complex64)
                mag = float(_np.abs(new_c).mean()) + 1e-9
                centroids[best] = (new_c / mag).astype(_np.complex64)
                clusters[best].append(abstract_chain)
            elif len(centroids) < max_clusters:
                centroids.append(chv)
                clusters.append([abstract_chain])
            else:
                # full -- force best
                clusters[best].append(abstract_chain)
        if verbose:
            print(f'    clustered {len(chains)} chains -> {len(clusters)} '
                  f'clusters')
        return clusters, centroids

    def induce_schemas_length_clustered(self, exposure_buffer=None,
                                            texts=None, max_chains=5000,
                                            min_chain_len=3, max_chain_len=12,
                                            n_reinforce=3,
                                            sim_threshold=0.40,
                                            max_clusters_per_length=50,
                                            min_cluster=2, verbose=False):
        """Pack 231 v2 -- length bucket FIRST, then HV cluster WITHIN length.

        Combines Pack 226 (length grouping eliminates spurious cross-length
        merges) with Pack 231 v1 (HV clustering separates structural patterns
        within same length). Each length bucket -> own cluster set -> own
        schema set.
        """
        from ikigai.cognition.schema_inducer import anti_unify, SLOT
        from collections import defaultdict
        chains = []
        if texts:
            for text in texts:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if exposure_buffer is not None and len(chains) < max_chains:
            snapshot = exposure_buffer.snapshot()
            for text, _, _ in snapshot:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if not chains:
            return {'chains': 0, 'schemas': 0, 'transitions': 0,
                    'clusters_total': 0}
        if self._isa_inverse is None:
            self._build_isa_inverse()

        # Abstract + group by length
        by_len = defaultdict(list)
        for chain in chains:
            ach = [self._isa_parent(t) or t for t in chain]
            by_len[len(ach)].append(ach)

        new_schemas = []; n_transitions = 0; total_clusters = 0; n_kept = 0
        for length, group in by_len.items():
            if len(group) < min_cluster:
                continue
            clusters, _ = self.cluster_abstract_chains(
                group, sim_threshold=sim_threshold,
                max_clusters=max_clusters_per_length, verbose=False)
            total_clusters += len(clusters)
            for cluster in clusters:
                if len(cluster) < min_cluster: continue
                schema = list(cluster[0])
                for ach in cluster[1:]:
                    schema = anti_unify(schema, ach)
                n_fixed = sum(1 for t in schema if t is not SLOT)
                if n_fixed < 2: continue
                new_schemas.append(schema)
                n_kept += 1
                if verbose:
                    printable = [t if t is not SLOT else '_' for t in schema]
                    print(f'    len={length} cluster(size={len(cluster)}): '
                          f'{printable}')
                # Crystallize
                last_fixed = None
                for tok in schema:
                    if tok is SLOT: continue
                    if last_fixed is not None:
                        for _ in range(n_reinforce):
                            self.mr.relate(last_fixed, 'schema_next', tok)
                        n_transitions += 1
                    last_fixed = tok
        self.schemas.extend(new_schemas)
        self.schema_transitions += n_transitions
        return {
            'chains': len(chains),
            'clusters_total': total_clusters,
            'schemas': n_kept,
            'transitions': n_transitions,
        }

    def induce_schemas_clustered(self, exposure_buffer=None, texts=None,
                                   max_chains=5000, min_chain_len=3,
                                   max_chain_len=12, n_reinforce=3,
                                   sim_threshold=0.30, max_clusters=500,
                                   min_cluster=2, verbose=False):
        """Pack 231 -- the AGI-path schema mining.

        Drops Pack 226's length-bucket grouping. Replaces it with substrate-
        native positional-bound HV clustering. Within each cluster, runs
        anti_unify. Each cluster yields one schema. K clusters = K schemas.

        Scales O(N * K) where K is small. Suitable for raw web text.

        Returns dict with stats + list of induced schemas.
        """
        from ikigai.cognition.schema_inducer import anti_unify, SLOT
        chains = []
        if texts:
            for text in texts:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if exposure_buffer is not None and len(chains) < max_chains:
            snapshot = exposure_buffer.snapshot()
            for text, _, _ in snapshot:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if not chains:
            return {'chains': 0, 'clusters': 0, 'schemas': 0, 'transitions': 0}

        # 1) Cluster
        clusters, centroids = self.cluster_abstract_chains(
            chains, sim_threshold=sim_threshold,
            max_clusters=max_clusters, verbose=verbose)

        # 2) Anti-unify within each cluster
        new_schemas = []; n_transitions = 0
        n_kept = 0; n_skipped = 0
        for ci, cluster in enumerate(clusters):
            if len(cluster) < min_cluster:
                n_skipped += 1
                continue
            schema = list(cluster[0])
            for ach in cluster[1:]:
                schema = anti_unify(schema, ach)
            n_fixed = sum(1 for t in schema if t is not SLOT)
            if n_fixed < 2:
                n_skipped += 1
                continue
            new_schemas.append(schema)
            n_kept += 1
            if verbose:
                printable = [t if t is not SLOT else '_' for t in schema]
                print(f'    cluster {ci} (size={len(cluster)}): {printable}')
            # Crystallize schema as substrate writes
            last_fixed = None
            for tok in schema:
                if tok is SLOT: continue
                if last_fixed is not None:
                    for _ in range(n_reinforce):
                        self.mr.relate(last_fixed, 'schema_next', tok)
                    n_transitions += 1
                last_fixed = tok
        self.schemas.extend(new_schemas)
        self.schema_transitions += n_transitions
        return {
            'chains': len(chains),
            'clusters': len(clusters),
            'schemas': n_kept,
            'transitions': n_transitions,
            'skipped_clusters': n_skipped,
        }

    # ── Pack 233 -- Unsupervised distributional POS via context cluster ──
    def induce_unsupervised_pos(self, min_freq=3, sim_threshold=0.50,
                                  max_clusters=200, verbose=False,
                                  feature_mode='next_pos_cooccur_pos'):
        """Pack 233 / 247a -- emergent POS without taxonomy.

        For every vocab token w with freq >= min_freq:
          - Build context HV per feature_mode.
        Cluster context HVs greedily by cosine sim. Each cluster = emergent
        POS category. Label = the cluster's most-central token (member with
        highest cosine to centroid).

        feature_mode options (Pack 247a Day 70):
          'next_pos_cooccur_pos'  -- Pack 233 default: hv_next*pos_0 + hv_co*pos_1
                                     (collapses to 1 mega-cluster at wiki scale
                                     due to cooccur Zipfian dragger)
          'next_only'             -- recall(w, 'next') only. Pack 245J finding:
                                     gives 10+ meaningful clusters at tau=0.80
                                     on wiki scale.
          'cooccur_only'          -- recall(w, 'cooccur') only.
          'next_plus_cooccur'     -- simple sum (no pos binding).

        Stores result in self._emergent_pos: dict[word, str(cluster_label)].
        Stores cluster sizes in self._emergent_pos_clusters.

        Returns dict with stats.
        """
        import numpy as _np
        d = self.mr.d
        vocab = sorted(self.mr._cooccur_seen)
        # Filter by frequency
        unigram = self.mr._unigram_count
        eligible = [w for w in vocab if unigram.get(w, 0) >= min_freq]
        if verbose:
            print(f'    {len(eligible)} tokens eligible (freq >= {min_freq}) '
                  f'/ {len(vocab)} total vocab')

        # Build per-token context HV per feature_mode (Pack 247a).
        contexts = []
        kept_words = []
        for w in eligible:
            try:
                if feature_mode == 'next_only':
                    hv = self.mr.recall(w, 'next').astype(_np.complex64)
                elif feature_mode == 'cooccur_only':
                    hv = self.mr.recall(w, 'cooccur').astype(_np.complex64)
                elif feature_mode == 'next_plus_cooccur':
                    a = self.mr.recall(w, 'next').astype(_np.complex64)
                    b = self.mr.recall(w, 'cooccur').astype(_np.complex64)
                    hv = (a + b).astype(_np.complex64)
                else:  # 'next_pos_cooccur_pos' -- Pack 233 default
                    hv_next = self.mr.recall(w, 'next')
                    hv_co = self.mr.recall(w, 'cooccur')
                    pos_r = self.mr.roles.get('pos_0',
                        self.mr.roles[next(iter(self.mr.roles))])
                    pos_l = self.mr.roles.get('pos_1',
                        self.mr.roles[next(iter(self.mr.roles))])
                    hv = (hv_next * pos_r + hv_co * pos_l).astype(_np.complex64)
                mag = float(_np.abs(hv).mean()) + 1e-9
                hv = (hv / mag).astype(_np.complex64)
                contexts.append(hv); kept_words.append(w)
            except Exception:
                continue
        if not contexts:
            return {'tokens': 0, 'clusters': 0}

        # Greedy nearest-centroid clustering
        centroids = []                  # complex64[d] each
        cluster_members = []            # list[list[(word, hv)]]
        for w, hv in zip(kept_words, contexts):
            if not centroids:
                centroids.append(hv)
                cluster_members.append([(w, hv)])
                continue
            K = _np.stack(centroids)
            sims = _np.real(K @ _np.conj(hv)) / d
            best = int(_np.argmax(sims))
            if float(sims[best]) > sim_threshold:
                # Join cluster; drift centroid
                n = len(cluster_members[best])
                new_c = ((n * centroids[best] + hv) / (n + 1)).astype(
                    _np.complex64)
                mag = float(_np.abs(new_c).mean()) + 1e-9
                centroids[best] = (new_c / mag).astype(_np.complex64)
                cluster_members[best].append((w, hv))
            elif len(centroids) < max_clusters:
                centroids.append(hv)
                cluster_members.append([(w, hv)])
            else:
                cluster_members[best].append((w, hv))

        # Assign POS labels: most-central token per cluster.
        emergent_pos = {}
        cluster_sizes = {}
        cluster_labels = {}
        for ci, members in enumerate(cluster_members):
            if len(members) < 2:
                # Singleton cluster -- use the word itself as label.
                w, _ = members[0]
                cluster_label = f'pos_{ci}__{w}'
            else:
                # Find most-central member.
                stacked = _np.stack([h for _, h in members])
                sims_to_centroid = _np.real(
                    stacked @ _np.conj(centroids[ci])) / d
                top = int(_np.argmax(sims_to_centroid))
                center_word = members[top][0]
                cluster_label = f'pos_{ci}__{center_word}'
            for w, _ in members:
                emergent_pos[w] = cluster_label
            cluster_sizes[cluster_label] = len(members)
            cluster_labels[cluster_label] = [w for w, _ in members[:8]]
            if verbose:
                preview = ', '.join(w for w, _ in members[:6])
                print(f'    cluster {ci:>3d} ({len(members):>3d}): '
                      f'{cluster_label} <- {preview}')

        self._emergent_pos = emergent_pos
        self._emergent_pos_clusters = cluster_labels
        return {
            'tokens': len(kept_words),
            'clusters': len(centroids),
            'avg_cluster_size': len(kept_words) / max(len(centroids), 1),
        }

    def _emergent_parent(self, word):
        """Pack 233 alternative to _isa_parent: returns emergent POS cluster
        label discovered via context clustering. Falls back to _isa_parent
        if Pack 233 hasn't been run yet."""
        emp = getattr(self, '_emergent_pos', None)
        if emp is not None and word in emp:
            return emp[word]
        return self._isa_parent(word)

    def induce_schemas_emergent(self, exposure_buffer=None, texts=None,
                                  max_chains=5000, min_chain_len=3,
                                  max_chain_len=12, n_reinforce=3,
                                  sim_threshold=0.40,
                                  max_clusters_per_length=50,
                                  min_cluster=2, verbose=False):
        """Pack 233 wire -- Pack 231 v2 but uses emergent POS instead of
        hand-asserted isa for abstraction. Run induce_unsupervised_pos FIRST.
        """
        from ikigai.cognition.schema_inducer import anti_unify, SLOT
        from collections import defaultdict
        chains = []
        if texts:
            for text in texts:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if exposure_buffer is not None and len(chains) < max_chains:
            snapshot = exposure_buffer.snapshot()
            for text, _, _ in snapshot:
                toks = [t for t in str(text).lower().split() if t]
                if min_chain_len <= len(toks) <= max_chain_len:
                    chains.append(toks)
                if len(chains) >= max_chains: break
        if not chains:
            return {'chains': 0, 'schemas': 0, 'transitions': 0}

        # Abstract via emergent POS (Pack 233)
        by_len = defaultdict(list)
        for chain in chains:
            ach = [self._emergent_parent(t) or t for t in chain]
            by_len[len(ach)].append(ach)

        new_schemas = []; n_transitions = 0
        total_clusters = 0; n_kept = 0
        for length, group in by_len.items():
            if len(group) < min_cluster: continue
            clusters, _ = self.cluster_abstract_chains(
                group, sim_threshold=sim_threshold,
                max_clusters=max_clusters_per_length, verbose=False)
            total_clusters += len(clusters)
            for cluster in clusters:
                if len(cluster) < min_cluster: continue
                schema = list(cluster[0])
                for ach in cluster[1:]:
                    schema = anti_unify(schema, ach)
                n_fixed = sum(1 for t in schema if t is not SLOT)
                if n_fixed < 2: continue
                new_schemas.append(schema)
                n_kept += 1
                if verbose:
                    printable = [t if t is not SLOT else '_' for t in schema]
                    print(f'    len={length} cluster(size={len(cluster)}): '
                          f'{printable}')
                last_fixed = None
                for tok in schema:
                    if tok is SLOT: continue
                    if last_fixed is not None:
                        for _ in range(n_reinforce):
                            self.mr.relate(last_fixed, 'schema_next', tok)
                        n_transitions += 1
                    last_fixed = tok
        self.schemas.extend(new_schemas)
        self.schema_transitions += n_transitions
        return {
            'chains': len(chains),
            'clusters_total': total_clusters,
            'schemas': n_kept,
            'transitions': n_transitions,
        }

    # ── Pack 226 schema-aware generation ──────────────────────────────────
    def step_via_schema(self, current_token, n_iters=5, beta=8.0,
                         no_repeat=None):
        """Hybrid step: query SCHEMA next at abstract level, then resolve
        to a concrete token via isa-inverse + concrete next bigram.
        Falls back to plain step() if schemas can't produce a candidate.
        """
        if not current_token: return (None, 0.0)
        if self._isa_inverse is None:
            self._build_isa_inverse()

        # 1) Look up abstract parent of current.
        current_abstract = self._isa_parent(current_token) or current_token

        # 2) Query schema_next at abstract level.
        schema_cands = list(self.mr._role_targets.get('schema_next', set()))
        next_abstract = None
        if schema_cands:
            try:
                r = self.mr.recall(current_abstract, 'schema_next')
                results = self.mr.resonator_recall(r,
                    candidate_words=schema_cands, n_iters=n_iters, beta=beta,
                    top_k=3)
                if results and results[0][0]:
                    next_abstract = results[0][0]
            except Exception:
                pass

        # 3) If we got abstract, resolve to a concrete child via isa-inverse.
        if next_abstract:
            # If the abstract IS a concrete word (no further parent), just use it.
            children = self._isa_inverse.get(next_abstract, set())
            candidate_pool = (list(children) if children
                              else [next_abstract])
            if no_repeat:
                candidate_pool = [c for c in candidate_pool if c not in no_repeat]
            if candidate_pool:
                # Use concrete next-role to pick which child fits the current
                # token's bigram statistics.
                concrete_next_cands = list(
                    self.mr._role_targets.get(self.NEXT_ROLE, set()))
                pool = [c for c in candidate_pool if c in concrete_next_cands]
                if not pool: pool = candidate_pool
                try:
                    r_concrete = self.mr.recall(current_token, self.NEXT_ROLE)
                    pick = self.mr.resonator_recall(r_concrete,
                        candidate_words=pool, n_iters=n_iters, beta=beta,
                        top_k=1)
                    if pick:
                        return pick[0]
                except Exception:
                    pass
                # Last resort: just first child.
                return (candidate_pool[0], 0.0)

        # Fallback: plain FSM step (concrete bigram path).
        return self.step(current_token, n_iters=n_iters, beta=beta,
                          exclude=no_repeat)

    def generate_via_schema(self, seed_tokens, max_tokens=20, n_iters=5,
                              beta=8.0, stop_tokens=None, no_repeat_window=3,
                              verbose=False):
        """Pack 226 schema-aware autoregression. Walks both schema_next (abstract)
        and concrete next attractors.
        """
        if isinstance(seed_tokens, str):
            seed_tokens = seed_tokens.split()
        out = list(seed_tokens)
        stop_tokens = set(stop_tokens or ())
        for step_i in range(max_tokens):
            current = out[-1]
            exclude = set(out[-no_repeat_window:]) if no_repeat_window > 0 else None
            nxt, score = self.step_via_schema(current, n_iters=n_iters,
                                                 beta=beta, no_repeat=exclude)
            if verbose:
                print(f'    [{step_i}] {current} -> {nxt} ({score:.3f})')
            if not nxt or nxt in stop_tokens:
                break
            out.append(nxt)
        return out

    # ── generate ──────────────────────────────────────────────────────────
    def step(self, current_token, prev_token=None, candidates=None,
              n_iters=5, beta=8.0, temperature=1.0, top_k=1, exclude=None):
        """Single next-token prediction. Uses RESONATOR cleanup over candidates.
        prev_token: optional previous token for trigram conditioning.
        candidates: optional restricted vocab (default = all next-role targets).
        Returns (token, score) for top_k=1, else list of (token, score).
        """
        if not current_token: return (None, 0.0) if top_k == 1 else []
        if candidates is None:
            candidates = list(self.mr._role_targets.get(self.NEXT_ROLE, set()))
        if exclude:
            candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return (None, 0.0) if top_k == 1 else []
        # Recall noisy next HV from current
        try:
            r_next = self.mr.recall(current_token, self.NEXT_ROLE)
        except Exception:
            return (None, 0.0) if top_k == 1 else []
        # Trigram condition: if prev_token, also recall (prev, next2, ?)
        if prev_token:
            try:
                r_skip = self.mr.recall(prev_token, self.PREV_ROLE)
                r_next = (r_next + r_skip).astype(np.complex64)
                mag = float(np.abs(r_next).mean()) + 1e-9
                r_next = r_next / mag
            except Exception:
                pass
        # Resonator cleanup against candidates
        results = self.mr.resonator_recall(r_next, candidate_words=candidates,
                                             n_iters=n_iters, beta=beta,
                                             top_k=max(top_k, 5))
        if top_k == 1:
            return results[0] if results else (None, 0.0)
        return results[:top_k]

    def generate(self, seed_tokens, max_tokens=20, n_iters=5, beta=8.0,
                  stop_tokens=None, no_repeat_window=3, verbose=False):
        """Autoregressive generation from seed. Returns list of tokens.

        seed_tokens: list of starting tokens (last is current state).
        max_tokens: how many to emit.
        stop_tokens: set of tokens that halt generation.
        no_repeat_window: forbid emitting tokens that appeared in last N positions.
        """
        if isinstance(seed_tokens, str):
            seed_tokens = seed_tokens.split()
        out = list(seed_tokens)
        stop_tokens = set(stop_tokens or ())
        for step in range(max_tokens):
            current = out[-1]
            prev = out[-2] if len(out) >= 2 else None
            exclude = set(out[-no_repeat_window:]) if no_repeat_window > 0 else None
            nxt, score = self.step(current, prev_token=prev, exclude=exclude,
                                     n_iters=n_iters, beta=beta, top_k=1)
            if verbose:
                print(f'    [{step}] {current} -> {nxt} ({score:.3f})')
            if not nxt or nxt in stop_tokens:
                break
            out.append(nxt)
        return out
