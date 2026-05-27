"""
integrate.py -- Master Ikigai Organism. Top-level entry point.

Day 56 -- new paradigm: ONE organism, ALL capabilities, brain-architecture mapping.

Connects every meaningful module under ikigai/cognition/ into one being.
Adds generative reasoning (ReasoningEngine). Not just retrieval anymore.

When you call `organism.ask("Janet has 5 apples. She ate 2. How many?")`,
the organism actually THINKS through it:
    - Wernicke's: parses each sentence
    - Hippocampus: stores episodic chain
    - Prefrontal: tracks variable bindings
    - Basal ganglia: selects operator from verb
    - Cerebellum: chains multi-statement reasoning
    - Broca's: produces answer

Public interface:
    org = IkigaiOrganism()
    answer = org.ask("...")
    trace  = org.trace()
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#  Pillars (foundation)
from ikigai.cognition.cgpsp_encoder       import CGPSPEncoder
from ikigai.cognition.pi_k_algebra        import PiK
from ikigai.cognition.pgmw                import PersonaGrid
from ikigai.cognition.sac_field           import SACField

#  Reasoning core (hardcoded path -- works on simple SVO)
from ikigai.cognition.reasoning_engine    import (
    ReasoningEngine, ReasoningParser, WorkingMemory,
    OPERATOR_LEXICON, QUERY_MARKERS,
)

#  BEING: persistent living organism (the substrate)
from ikigai.cognition.being               import IkigaiBeing
from ikigai.cognition.operational_grounding import OperationalGrounding
from ikigai.cognition.sensory_grounding   import SensoryGrounding
from ikigai.cognition.taxonomic_grounding import TaxonomicGrounding
from ikigai.cognition.grammar_grounding   import GrammarGrounding
from ikigai.cognition.flat_memory         import FlatMemory
from ikigai.cognition.multirole_memory    import MultiRoleMemory
from ikigai.cognition.dialogue            import DialogueLoop
from ikigai.cognition.generator           import SentenceGenerator

#  Memory + cognition modules
from ikigai.cognition.holographic_memory  import HolographicMemory
from ikigai.cognition.cross_modal_space   import CrossModalSpace
from ikigai.cognition.cross_time_resonance import CrossTimeResonator
from ikigai.cognition.concept_atomizer    import ConceptAtomizer
from ikigai.cognition.belief_field        import BeliefField
from ikigai.cognition.theory_of_mind      import TheoryOfMindSandbox
from ikigai.cognition.adversarial_immune  import AdversarialImmune
from ikigai.cognition.importance_decay    import ImportanceDecayLattice
from ikigai.cognition.counterfactual_sim  import CounterfactualField
from ikigai.cognition.proof_carrying_gen  import ProofCarryingGenerator
from ikigai.cognition.causal_world_model  import CausalWorldModel
from ikigai.cognition.fe_action           import FreeEnergyActionSelector
from ikigai.cognition.multistep_planner   import MultiStepPlanner
from ikigai.cognition.curiosity_drive     import CuriosityDrive
from ikigai.cognition.persona_fe_coupling import PersonaFEC
from ikigai.cognition.vsa_calculus        import VSACalculus
from ikigai.cognition.hot_loader          import CognitionHotLoader

#  Inherited Day-54 modules
from ikigai.cognition.skill_crystal       import SkillCrystal


class IkigaiOrganism:
    """
    Master organism. Every cognition module wired in.

    Public interface:
        ask(text)          -> answer (with reasoning trace)
        observe(text)      -> store as episodic memory (no answer expected)
        remember(text)     -> store as long-term semantic memory
        recall(query)      -> retrieve nearest memory
        status()           -> health snapshot of all subsystems

    Brain-region mapping (each subsystem):
        Sensory cortex      -> CGPSPEncoder (byte->phasor flow)
        Wernicke's          -> ReasoningParser (SVO extraction)
        Prefrontal          -> WorkingMemory (variable bindings)
        Basal ganglia       -> ReasoningEngine.read_statement (operator dispatch)
        Hippocampus         -> HolographicMemory + episodic event log
        Cerebellum          -> CausalWorldModel + MultiStepPlanner
        Default mode net    -> ConceptAtomizer (sleep replay clustering)
        Amygdala            -> AdversarialImmune (threat detection)
        Insula              -> BeliefField (contradiction healing)
        TPJ / mPFC          -> TheoryOfMindSandbox (modeling others)
        Broca's             -> output decoder
    """

    def __init__(self, d=400, flat_only=False):
        """
        flat_only=True: minimal organism for inference on the flat substrate.
        Skips heavy cognition modules (HolographicMemory, BeliefField, etc.)
        and the dict lexicon scaffolding. Keeps stateless parsers (sensory,
        taxonomy, operations) needed by read() and query methods.
        Pack 124: inference-RAM optimization.
        """
        self.d = d
        self._flat_only = bool(flat_only)

        # Foundation pillars (UHE) -- persona kept (used by generator/save)
        self.encoder = None if flat_only else CGPSPEncoder(d=2048, gamma=0.4)
        self.pik     = None if flat_only else PiK(d=2048, n_primes=32)
        self.persona = PersonaGrid(d=2048)
        self.sac     = None if flat_only else SACField(d=2048)

        # BEING + 5 grounding channels (Pack 96-101)
        # In flat_only: being/grammar dropped (replaced by unified). Sensory,
        # taxonomy, operations kept as parsers (extract_pairs/_seeds/observe_story).
        self.being      = None if flat_only else IkigaiBeing(d=2048, drift_rate=0.08, window_size=4)
        self.operations = OperationalGrounding(d=2048)
        self.sensory    = SensoryGrounding(d=2048)
        self.taxonomy   = TaxonomicGrounding(d=2048)
        self.grammar    = None if flat_only else GrammarGrounding(d=2048)

        # FLAT MEMORY (Pack 114-115) -- legacy, subsumed by unified. Drop in flat_only.
        self.flat          = None if flat_only else FlatMemory(d=512, M=16384, k=64, seed=114)
        self._flat_enabled = False

        # UNIFIED MEMORY (Pack 117-118): the actual flat substrate. Always built.
        self.unified          = MultiRoleMemory(d=512, M=16384, k=64, seed=114)
        self._unified_enabled = True
        # In flat_only mode, dict writes are off by default (no being/grammar to write to).
        self._dict_writes_enabled = not flat_only

        # REASONING CORE + cognition modules. None in flat_only.
        if flat_only:
            self.reasoner = None
            self.holo = self.modal = self.osc = self.atom = self.belief = None
            self.tom = self.immune = self.decay = self.cf = self.pcg = None
            self.cwm = self.fea = self.planner = self.curio = self.pfc = None
            self.vsa = self.hotload = None
        else:
            self.reasoner = ReasoningEngine()
            self.holo    = HolographicMemory(d=d)
            self.modal   = CrossModalSpace(d=d)
            self.osc     = CrossTimeResonator(d=d, periods=[10, 100, 1000])
            self.atom    = ConceptAtomizer(d=d)
            self.belief  = BeliefField(d=d, conflict_threshold=-0.05, heal_rate=2.0)
            self.tom     = TheoryOfMindSandbox(d=d)
            self.immune  = AdversarialImmune(d=d)
            self.decay   = ImportanceDecayLattice(d=d)
            self.cf      = CounterfactualField(d=d)
            self.pcg     = ProofCarryingGenerator(d=d)
            self.cwm     = CausalWorldModel(d=d)
            self.fea     = FreeEnergyActionSelector(d=d)
            self.planner = MultiStepPlanner(self.cwm, self.fea)
            self.curio   = CuriosityDrive(d=d)
            self.pfc     = PersonaFEC()
            self.vsa     = VSACalculus(d=d)
            self.hotload = CognitionHotLoader()

        # Episodic chain (hippocampus)
        self._episodes = []
        self._last_trace = []
        self._tick = 0

    #  Primary interface

    def ask(self, text):
        """
        Generative reasoning + retrieval. Returns answer.
        Pipeline:
            1. Safety scan (amygdala)
            2. Encode (sensory cortex)
            3. Parse statements (Wernicke's)
            4. Update working memory (prefrontal + basal ganglia)
            5. Answer query (Broca's)
            6. Store episodic (hippocampus)
        """
        # Step 1: safety
        hits = self.immune.scan(text.lower().split(), threshold=0.4)
        if hits:
            return {'answer': None, 'safe': False, 'reason': 'threat_detected',
                    'hits': hits}

        # Step 2-5: reason
        trace, answer = self.reasoner.reason(text)
        self._last_trace = trace

        # Step 6: store episode
        self._tick += 1
        self._episodes.append({
            'tick': self._tick,
            'text': text,
            'trace': [(s, repr(stmt), v) for s, stmt, v in trace],
            'answer': answer,
        })

        return {
            'answer':  answer,
            'safe':    True,
            'trace':   trace,
            'wm':      self.reasoner.wm.all_values(),
            'tick':    self._tick,
        }

    def observe(self, text):
        """Process statement without expecting answer."""
        return self.ask(text)

    #  language acquisition

    def read(self, text):
        """
        Pure exposure: feed text to all 5 grounding channels.
        - Co-occurrence (Hebbian drift)
        - Sensory (anchor drift for property seed words)
        - Taxonomy (Hearst-pattern IS-A relations)
        - Operational (predictive coding -- GATED: only fires if numbers present)
        - Grammar (distributional POS via left/right context fingerprints)
        """
        import re as _re
        if self._dict_writes_enabled:
            # Channel 1: co-occurrence
            self.being.expose(text)
            # Channel 3: sensory grounding (applies to same lexicon)
            self.sensory.expose(text, self.being.lexicon,
                                 drift_rate=0.15, context_drift=0.04, window=3)
            # Channel 4: taxonomic IS-A grounding
            self.taxonomy.expose(text, self.being.lexicon,
                                  drift_rate=0.25, hyper_back_rate=0.05)
            # Channel 5: grammar / distributional POS
            self.grammar.expose(text, self.being.lexicon)
        # Channel 2 GATEKEEPER: still parses the story for unified-memory
        # verb observation. operations._c grows tiny (1 float per verb) so the
        # parse-storage cost is negligible vs the lexicon dict.
        if _re.search(r'\d', text):
            obs = self.operations.observe_story(text) if self._dict_writes_enabled \
                  else self.operations.observe_story(text)   # parses regardless; _c update tiny
            if obs is not None and self._unified_enabled and self.unified is not None:
                verb, n_b, m, n_a, _c = obs
                if m and abs(m) > 1e-9:
                    c_est = (n_a - n_b) / m
                    self.unified.expose_verb_observation(verb, c_est)
        # FLAT MEMORY: constant-RAM co-occurrence (Pack 114-115). Toggleable.
        if self._flat_enabled and self.flat is not None:
            self.flat.expose(text)
        # UNIFIED MEMORY: one flat substrate, all channels (Pack 117-118).
        if self._unified_enabled and self.unified is not None:
            self.unified.expose_cooccur(text)                       # Channel 1
            self.unified.expose_transitions(text)                   # Channel 5
            for hypo, hyper, _ in self.taxonomy.extract_pairs(text):  # Channel 4
                self.unified.relate(hypo, 'isa', hyper)
            # tokenize for sensory seed scan (works without dict scaffolding too)
            if self.being is not None and hasattr(self.being, 'tokenize'):
                _toks = self.being.tokenize(text)
            else:
                _toks = _re.sub(r"[^a-z0-9'\s]", ' ', text.lower()).split()
            for tok in _toks:
                anchor = self.sensory._seeds.get(tok)               # Channel 3
                if anchor is not None:
                    self.unified.relate(tok, 'sensory', anchor)
        return self.being.reflect() if self.being is not None else None

    #  unified-memory interface (Pack 118): query the one flat substrate

    def isa_of(self, word, candidates=None):
        """Hypernym of word from unified memory (cleanup recall). source-of-truth migrating off dict."""
        best, score = self.unified.query(word, 'isa', candidates)
        return best

    def sensory_of(self, word, candidates=None):
        """Nearest sensory anchor of word from unified memory."""
        cands = candidates if candidates is not None else self.sensory.anchor_names()
        best, score = self.unified.query(word, 'sensory', cands)
        return best

    def unified_similarity(self, w1, w2):
        """Co-occurrence similarity from the unified flat substrate."""
        return self.unified.similarity(w1, w2)

    def unified_status(self):
        return self.unified.status()

    def enable_unified(self, on=True):
        self._unified_enabled = bool(on)

    def assert_isa(self, hypo, hyper, n=20):
        """Assert hypo IS-A hyper directly into unified memory, n reinforcements.
        Bypasses Hearst-on-prose noise. Use for clean fact injection."""
        for _ in range(n):
            self.unified.relate(hypo, 'isa', hyper)

    def assert_sensory(self, word, anchor, n=20):
        """Assert sensory mapping directly into unified memory, n reinforcements."""
        for _ in range(n):
            self.unified.relate(word, 'sensory', anchor)

    #  Pack 147: multi-channel meaning exposure
    def expose_meaning(self, text, **kwargs):
        """
        Native multi-channel meaning capture. Writes:
          - episode role (sentence-HV bound to each token)
          - affordance role (subj/verb/obj triple if extractable)
          - mod role (adjective+noun if extractable)
        kwargs: pos_classifier OR subj_vocab/verb_vocab/obj_vocab/adj_vocab.
        Returns per-channel write counts.
        """
        return self.unified.expose_meaning(text, **kwargs)

    def expose_episode(self, text):
        """Write a sentence as an episode bound to each of its tokens."""
        return self.unified.expose_episode(text)

    def expose_affordance(self, subj, verb, obj=None):
        """Write a (subj does verb [does obj]) affordance fact."""
        return self.unified.expose_affordance(subj, verb, obj)

    def expose_modifier(self, modifier, noun):
        """Write that modifier was seen describing noun."""
        return self.unified.expose_modifier(modifier, noun)

    def _vision_encode(self, img, seed=127, bandwidth=2.0):
        """
        Encode arbitrary numeric array -> d-dim phasor HV. Modality-blind.

        Pack 128: introduced (raw random projection).
        Pack 134 v2: two-stage normalization.
          1. L2-normalize input -> unit norm (data-range invariant)
          2. Project via N(0, bandwidth) -> phase std = bandwidth
        Bandwidth ~2 rad is the sweet spot: enough structure preservation
        (limited phase wrap), enough spread (avoids over-clustering).
        Works on ANY input dim. Projection matrices lazily cached per
        in_dim, regenerable from seed. NOT pickled.
        """
        import numpy as _np
        v = _np.asarray(img, dtype=_np.float32).ravel()
        in_dim = v.shape[0]
        if not hasattr(self, '_vis_proj') or self._vis_proj is None:
            self._vis_proj = {}
        if in_dim not in self._vis_proj:
            rng = _np.random.default_rng(seed)
            self._vis_proj[in_dim] = (rng.standard_normal(
                (self.unified.d, in_dim)).astype(_np.float32) * bandwidth)
        P = self._vis_proj[in_dim]
        nrm = float(_np.linalg.norm(v))
        if nrm < 1e-9: nrm = 1.0
        vn = v / nrm
        phase = (P @ vn).astype(_np.float32)
        return _np.exp(1j * phase).astype(_np.complex64)

    def expose_image(self, img, label, n=1):
        """Write (image, label) into unified memory under 'class' role.
        `img` can be any numeric array (pixels, sensors). `label` is a string."""
        if self.unified is None or 'class' not in self.unified.roles:
            raise ValueError("unified memory needs 'class' role registered")
        addr = self._vision_encode(img)
        bound = self.unified._bind(addr, self.unified.roles['class'])
        value = self.unified.ck.key(str(label))
        for _ in range(n):
            self.unified.sdm_rel.write(bound, value)
        self.unified._role_targets.setdefault('class', set()).add(str(label))
        self.unified._seen.add(f'_img_{str(label)}')

    def classify_image(self, img, candidates=None):
        """Predict class label for an image. Returns (label, score) or None."""
        if self.unified is None or 'class' not in self.unified.roles:
            return None
        addr = self._vision_encode(img)
        bound = self.unified._bind(addr, self.unified.roles['class'])
        out = self.unified.sdm_rel.read(bound)
        cands = candidates if candidates is not None \
                else self.unified._role_targets.get('class', set())
        if not cands: return None
        import numpy as _np
        best, bscore = None, -9.0
        for c in cands:
            ck_c = self.unified.ck.key(str(c))
            s = float(_np.real(_np.vdot(out, ck_c))) / self.unified.d
            if s > bscore:
                bscore, best = s, str(c)
        return (best, bscore)

    def flat_verb_coefficient(self, verb):
        """Channel 2 from flat memory: decoded coefficient (Pack 121)."""
        return self.unified.predict_verb_coefficient(verb)

    #  few-shot pattern learning (Pack 132)

    def _ensure_role(self, role):
        """Register a role if missing; deterministic per (seed, role-name)."""
        if role in self.unified.roles:
            return
        import numpy as _np
        rng = _np.random.default_rng(abs(hash(role)) % (2**32))
        ph = rng.uniform(-_np.pi, _np.pi, self.unified.d).astype(_np.float32)
        self.unified.roles[role] = _np.exp(1j * ph).astype(_np.complex64)

    def _input_hv(self, inp):
        """Encode an input as a d-dim phasor HV. String -> computed key.
        Numeric array -> random projection (vision-style)."""
        import numpy as _np
        if isinstance(inp, str):
            return self.unified.ck.key(inp)
        arr = _np.asarray(inp)
        return self._vision_encode(arr)

    def few_shot_learn(self, examples, role='pattern', n_reinforce=20):
        """
        Write (input -> output_label) examples into unified memory under `role`.
        examples: iterable of (input, output_label) where input is a string or
        numeric array; output_label is a string (cleanup target).
        """
        self._ensure_role(role)
        ROLE = self.unified.roles[role]
        for inp, lbl in examples:
            addr = self.unified._bind(self._input_hv(inp), ROLE)
            value = self.unified.ck.key(str(lbl))
            for _ in range(n_reinforce):
                self.unified.sdm_rel.write(addr, value)
            self.unified._role_targets.setdefault(role, set()).add(str(lbl))

    def cogitate(self, prompt='', max_tokens=100, think_steps=3,
                 momentum=0.7, thought_gamma=4.0, temperature=0.7,
                 top_k=20, remove_common=True, return_trace=False, seed=None,
                 ngram_weights=(0.2, 0.4, 0.4), ngram_ctx=3,
                 goal_gamma=0.0,
                 grounded_gamma=0.0, grounded_roles=('isa', 'property')):
        """
        Flat-memory generation engine (Pack 135).

        Decoupled think/speak loop with thought-state as an evolving HV in
        the substrate's address space. Per-token cost is O(1); RAM does NOT
        grow with output length. No context window. New facts injected
        mid-generation integrate immediately.

        think_steps: associative thought-walk steps per emitted token
        thought_gamma: how strongly thought-alignment steers word choice
        return_trace=True: returns (text, list_of_thought_HVs)
        """
        from ikigai.cognition.generation_engine import GenerationEngine
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = GenerationEngine(self)
        eng = self._engine
        eng.think_steps   = int(think_steps)
        eng.momentum      = float(momentum)
        eng.thought_gamma = float(thought_gamma)
        eng.temperature   = float(temperature)
        eng.top_k         = int(top_k)
        eng.remove_common = bool(remove_common)
        eng.ngram_weights   = tuple(ngram_weights)
        eng.ngram_ctx       = int(ngram_ctx)
        eng.goal_gamma      = float(goal_gamma)
        eng.grounded_gamma  = float(grounded_gamma)
        eng.grounded_roles  = tuple(grounded_roles)
        return eng.generate(prompt=prompt, max_tokens=max_tokens,
                            return_trace=return_trace, seed=seed)

    def reason_chain(self, start, hops):
        """
        N-hop reasoning across roles in unified memory.
        hops: list of (role, candidates) tuples.
        Returns [start, hop1, hop2, ..., final] waypoints.
        Pack 133: multi-hop chain-of-thought via role-binding.
        """
        if self.unified is None: return [start]
        return self.unified.reason_chain(start, hops)

    def few_shot_apply(self, inp, role='pattern', candidates=None):
        """
        Predict output label for `inp` using examples written via few_shot_learn.
        Returns (label, score) or None if no candidates.
        """
        if role not in self.unified.roles:
            return None
        import numpy as _np
        ROLE = self.unified.roles[role]
        addr = self.unified._bind(self._input_hv(inp), ROLE)
        out = self.unified.sdm_rel.read(addr)
        cands = candidates if candidates is not None \
                else self.unified._role_targets.get(role, set())
        if not cands: return None
        best, bscore = None, -9.0
        for c in cands:
            ck_c = self.unified.ck.key(str(c))
            s = float(_np.real(_np.vdot(out, ck_c))) / self.unified.d
            if s > bscore:
                bscore, best = s, str(c)
        return (best, bscore)

    def flat_predict_arithmetic(self, n_before, verb, modifier):
        """Predict n_after via flat-memory verb rotor. n + c*m."""
        c = self.unified.predict_verb_coefficient(verb)
        if c is None: return None
        return n_before + c * modifier

    def disable_dict_writes(self):
        """Pack 122: stop accumulating the dict lexicon/bigrams. Organism runs
        purely on the flat substrate. Operations parser stays (tiny float dict)."""
        self._dict_writes_enabled = False

    def solve_word_problem(self, text):
        """Parse a 2-number arithmetic word problem and answer via flat memory.
        Form: 'X had N <obj>. PRON <verb> M <obj>. How many ...?'
        Verb extracted from the ACTION sentence (the one containing the modifier),
        skipping subject names + stopwords -- mirrors operations.observe_story."""
        import re as _re
        sentences = [s.strip() for s in _re.split(r'[\.!?]+', text) if s.strip()]
        nums_all = [int(x) for x in _re.findall(r'\b\d+\b', text)]
        if len(nums_all) < 2:
            return {'error': 'need >= 2 numbers'}
        n_before, modifier = nums_all[0], nums_all[1]
        # find the action sentence: contains modifier, not n_before
        action_sent = None
        for s in sentences:
            ns = [int(x) for x in _re.findall(r'\b\d+\b', s)]
            if modifier in ns and n_before not in ns:
                action_sent = s; break
        if action_sent is None:
            action_sent = sentences[1] if len(sentences) > 1 else sentences[0]
        STOP = {'a','an','the','and','or','but','so','now','then','at','in','on','of','to',
                'from','has','have','had','is','are','was','were','with','by','for','she',
                'he','they','her','him','them','i','you','we','it','this','that','his',
                'hers','their','away','more','many','how','some','much','few','left','remain',
                'remaining','again','than','then'}
        toks = [t for t in _re.sub(r"[^a-z0-9'\s]", ' ', action_sent.lower()).split()
                if t and not t.replace('.', '').isdigit()]
        verb = None
        for t in toks:
            if t in STOP: continue
            if len(t) >= 3: verb = t; break
        if verb is None:
            return {'error': 'no verb found'}
        ans = self.flat_predict_arithmetic(n_before, verb, modifier)
        return {'n_before': n_before, 'verb': verb, 'modifier': modifier,
                'answer': ans, 'known_verbs': list(self.unified._verb_seen)}

    def flat_generate(self, prompt='', max_len=15, top_k=20,
                      temperature=0.7, seed=None):
        """
        Generate text using ONLY the unified flat substrate (no dict bigrams,
        no IkigaiBeing lexicon). Markov walk: next-word candidates pulled from
        unified['next'] role, scored by cleanup cosine, sampled by softmax.
        Pack 120 -- generation off the dict.
        """
        import random as _random, re as _re_local
        rng = _random.Random(seed)
        toks = [t for t in _re_local.sub(r"[^a-z0-9'\s]", ' ', prompt.lower()).split() if t]
        if not toks:
            return ''
        out = list(toks)
        for _ in range(max_len):
            cands = self.unified.next_word_candidates(out[-1], top_k=top_k)
            cands = [(w, s) for w, s in cands if s > 0]   # keep positive-scored
            if not cands:
                break
            import numpy as _np
            scores = _np.array([s for _, s in cands], dtype=_np.float64)
            scores = scores / max(temperature, 1e-3)
            scores = scores - scores.max()
            probs = _np.exp(scores); probs = probs / probs.sum()
            idx = rng.choices(range(len(cands)), weights=probs)[0]
            out.append(cands[idx][0])
        return ' '.join(out)

    #  flat-memory interface (Pack 114-115)

    def flat_similarity(self, w1, w2):
        """Word similarity from the constant-RAM flat substrate."""
        return self.flat.similarity(w1, w2)

    def flat_recall(self, word):
        """Adaptive reconstructive readout of a word's meaning."""
        return self.flat.recall(word)

    def flat_neighbors(self, word, k=10):
        """Nearest seen words in the flat substrate."""
        return self.flat.neighbors(word, k=k)

    def flat_status(self):
        """Footprint + vocab of the flat substrate (size independent of vocab)."""
        return self.flat.status()

    def enable_flat(self, on=True):
        """Toggle flat-memory writes during read() (off = faster dict-only)."""
        self._flat_enabled = bool(on)

    def read_corpus(self, sentences):
        """Stream a corpus into the being."""
        for s in sentences:
            self.being.expose(s)
        return self.being.reflect()

    def dream(self):
        """Sleep cycle: consolidate being's lexicon."""
        return self.being.dream()

    def word_similarity(self, w1, w2, source='dict'):
        """Word similarity. source='dict' (IkigaiBeing) or 'flat' (VSA-SDM)."""
        if source == 'flat':
            return self.flat.similarity(w1, w2)
        return self.being.cosine_words(w1, w2)

    def neighbors(self, w, k=5):
        return self.being.nearest_words(w, k=k)

    @property
    def age(self):
        return self.being.age

    def verb_coefficient(self, verb):
        """Learned arithmetic effect coefficient for verb."""
        return self.operations.coefficient(verb)

    def nearest_sensory_anchor(self, word):
        """Which sensory anchor best aligns with this word's HV?"""
        if word not in self.being.lexicon:
            return None, 0.0
        return self.sensory.nearest_anchor(self.being.lexicon[word])

    def predict_arithmetic(self, n_before, verb, modifier):
        """Use learned operational rotor to predict outcome."""
        return self.operations.predict(n_before, verb, modifier)

    def hypernym_of(self, word):
        """What kind of thing is this? Returns parent in IS-A tree."""
        return self.taxonomy.hypernym_of(word)

    def is_a(self, hypo, hyper, transitive=True):
        """Does hypo IS-A hyper? Walks IS-A chain if transitive."""
        return self.taxonomy.is_a(hypo, hyper, transitive=transitive)

    def chain_to_root(self, word):
        """Walk IS-A hierarchy: [word, parent, grandparent, ...]"""
        return self.taxonomy.chain_to_root(word)

    def pos_similarity(self, w1, w2):
        """How grammatically similar (same POS)?"""
        return self.grammar.pos_similarity(w1, w2)

    def pos_neighbors(self, word, k=5):
        """k words playing same grammatical role."""
        return self.grammar.pos_neighbors(word, k=k)

    def bigram_surprise(self, prev, curr):
        """-log2 P(curr | prev). High = phrase boundary."""
        return self.grammar.surprise(prev, curr)

    #  Phase 3: dialogue + generation

    def new_dialogue(self, persona=None):
        """Start a fresh multi-turn conversation."""
        loop = DialogueLoop(self, d=2048)
        loop.start(persona_name=persona)
        loop._generator = SentenceGenerator(self, d=2048)
        # Attach a convenience method for the loop
        def respond_to(user_text, **kwargs):
            reply = loop._generator.respond(user_text, dialogue_loop=loop, **kwargs)
            loop.user_says(user_text)
            t = loop.agent_says(reply)
            return reply, t
        loop.respond_to = respond_to
        return loop

    def generate(self, prompt='', max_len=15, mode='context_biased', **kwargs):
        """Direct generation. Returns text string."""
        gen = SentenceGenerator(self, d=2048)
        return gen.generate(prompt=prompt, max_len=max_len, mode=mode, **kwargs)

    def trace(self):
        """Returns last reasoning trace."""
        return list(self._last_trace)

    def memory(self):
        """Current working-memory state."""
        return self.reasoner.wm.all_values()

    def reset(self):
        """Clear working memory + episodic chain (start fresh)."""
        self.reasoner.reset()
        self._episodes = []
        self._last_trace = []
        self._tick = 0

    #  Long-term memory

    def remember(self, name, key_tokens, value_tokens):
        """Long-term holographic store."""
        return self.holo.store(name, key_tokens, value_tokens)

    def recall(self, key_tokens, top_k=3):
        """Long-term holographic recall."""
        return self.holo.recall(key_tokens, top_k=top_k)

    #  Introspection

    def status(self):
        return {
            'tick':              self._tick,
            'n_episodes':        len(self._episodes),
            'wm_vars':           len(self.reasoner.wm.all_values()),
            'wm_history':        len(self.reasoner.wm.history()),
            'n_threats':         self.immune.n_threats,
            'n_concepts':        self.modal.n_concepts,
            'n_skills_holo':     getattr(self.holo, 'n_stored', 0),
            'n_beliefs':         self.belief.n_beliefs,
            'n_atoms':           self.atom.n_atoms,
            # Being substrate
            'being_age':         self.being.age,
            'being_vocab':       self.being.vocab_size(),
            'being_exposures':   self.being.n_exposures,
            'being_curiosity':   round(float(self.being.curiosity), 4),
        }

    def __repr__(self):
        s = self.status()
        return (f"<IkigaiOrganism tick={s['tick']} eps={s['n_episodes']} "
                f"wm={s['wm_vars']}>")

    #  Persistence

    def save(self, path):
        """
        Persist organism state to disk via pickle.

        Saves only the learned state (5 grounding channels + persona grid),
        not the empty cognition modules. Fast round-trip: <1s for 6K-corpus.

        Usage:
            org.save('organism_5k.pkl')
            org2 = IkigaiOrganism.load('organism_5k.pkl')
        """
        import pickle, pathlib, time
        # flat_only mode: being is None; use unified.n_exposures as the counter.
        exposures = (self.being.n_exposures if self.being is not None
                     else getattr(self.unified, 'n_exposures', 0)
                          if self.unified is not None else 0)
        state = {
            '_version':    110,
            '_saved_at':   time.time(),
            '_exposures':  exposures,
            # 5 grounding channels (any may be None in flat_only mode)
            'being':       self.being,
            'operations':  self.operations,
            'sensory':     self.sensory,
            'taxonomy':    self.taxonomy,
            'grammar':     self.grammar,
            # persona grid (may have learned personas)
            'persona':     self.persona,
            # flat memory substrate (Pack 114-115); H regenerated from seed
            'flat':        self.flat,
            # unified memory substrate (Pack 117-118); all channels, one bank
            'unified':     self.unified,
            # metadata
            'tick':        self._tick,
            'episodes':    self._episodes,
        }
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = p.stat().st_size / 1_048_576
        return {'path': str(p), 'size_mb': round(size_mb, 2),
                'exposures': exposures}

    @classmethod
    def load(cls, path, flat_only=False):
        """
        Restore organism from a saved checkpoint.
        flat_only=True: skip restoring dict scaffolding modules; restore only
        the flat substrate + parsers. Big inference-RAM win.
        """
        import pickle, pathlib
        with open(pathlib.Path(path), 'rb') as f:
            state = pickle.load(f)
        org = cls.__new__(cls)
        IkigaiOrganism.__init__(org, flat_only=flat_only)
        if not flat_only:
            org.being      = state['being']
            org.grammar    = state['grammar']
            if 'flat' in state and state['flat'] is not None:
                org.flat = state['flat']
        # parsers + persona always restored (small)
        org.operations = state['operations']
        org.sensory    = state['sensory']
        org.taxonomy   = state['taxonomy']
        org.persona    = state['persona']
        if 'unified' in state and state['unified'] is not None:
            org.unified = state['unified']
        org._tick      = state.get('tick', 0)
        org._episodes  = state.get('episodes', [])
        return org


#  singleton convenience

_DEFAULT = None


def organism():
    """Get / create the default singleton organism."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = IkigaiOrganism()
    return _DEFAULT


def ask(text):
    """Convenience: ask the default organism."""
    return organism().ask(text)


if __name__ == '__main__':
    print('Booting Ikigai organism...')
    org = IkigaiOrganism()
    print(f'  status: {org.status()}')
    print('\nTest reasoning:')
    print('  Q: "Janet has 5 apples. She ate 2. How many apples does Janet have?"')
    r = org.ask("Janet has 5 apples. She ate 2. How many apples does Janet have?")
    print(f'  A: {r["answer"]}')
    print(f'  WM state: {r["wm"]}')
    print(f'  trace:')
    for s, stmt, v in r['trace']:
        print(f'    sentence={s!r}')
        print(f'      parsed={stmt}, value={v}')
