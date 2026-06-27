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

import sys
sys.path.insert(0, 'c:/neuroseed')

import numpy as np

# ── Pillars (foundation) ─────────────────────────────────────────────────────
from ikigai.cognition.cgpsp_encoder       import CGPSPEncoder
from ikigai.cognition.pi_k_algebra        import PiK
from ikigai.cognition.pgmw                import PersonaGrid
from ikigai.cognition.sac_field           import SACField

# ── Reasoning core (hardcoded path -- works on simple SVO) ───────────────────
from ikigai.cognition.reasoning_engine    import (
    ReasoningEngine, ReasoningParser, WorkingMemory,
    OPERATOR_LEXICON, QUERY_MARKERS,
)

# ── BEING: persistent living organism (the substrate) ────────────────────────
from ikigai.cognition.being               import IkigaiBeing
from ikigai.cognition.operational_grounding import OperationalGrounding
from ikigai.cognition.sensory_grounding   import SensoryGrounding
from ikigai.cognition.taxonomic_grounding import TaxonomicGrounding
from ikigai.cognition.grammar_grounding   import GrammarGrounding
from ikigai.cognition.flat_memory         import FlatMemory
from ikigai.cognition.multirole_memory    import MultiRoleMemory
from ikigai.cognition.dialogue            import DialogueLoop
from ikigai.cognition.generator           import SentenceGenerator

# ── Memory + cognition modules ───────────────────────────────────────────────
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

# ── Inherited Day-54 modules ─────────────────────────────────────────────────
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
        self.flat          = None if flat_only else FlatMemory(d=d, M=16384, k=64, seed=114)
        self._flat_enabled = False

        # UNIFIED MEMORY (Pack 117-118): the actual flat substrate. Always built.
        self.unified          = MultiRoleMemory(d=d, M=16384, k=64, seed=114)
        self._unified_enabled = True
        # Pack 197 Resonance Frame Field. Frame-conditioned binding so the
        # organism can hold N languages / N domains without cross-talk.
        from ikigai.cognition.frame_field import FrameField
        self.frames = FrameField(d=d, K=8, top_n=64, seed=42, alpha=0.5)
        self.unified._frame_field_ref = self.frames
        # Pack 210 -- BIG WIRE-UP. Activate 5 dead cognition modules so read()
        # runs reasoning during ingestion, not just statistics.
        from ikigai.cognition.free_energy_drive import (
            ConversationalVariationalFreeEnergyField)
        from ikigai.cognition.curiosity_drive import CuriosityDrive
        from ikigai.cognition.theory_of_mind import TheoryOfMindSandbox
        from ikigai.cognition.reasoning_engine import ReasoningEngine
        from ikigai.cognition.vsa_calculus import VSACalculus
        # Pack 211 -- Generation wire
        from ikigai.cognition.belief_field import BeliefField
        from ikigai.cognition.self_verifier import SelfVerifier
        from ikigai.cognition.proof_carrying_gen import ProofCarryingGenerator
        self.fe = ConversationalVariationalFreeEnergyField(d=64, window=8)
        self.curiosity = CuriosityDrive(d=d)
        self.tom = TheoryOfMindSandbox(d=d)
        self.tom.add_agent('default')
        self.re = ReasoningEngine()
        self.vsa = VSACalculus(d=d)
        # Pack 211 instances
        self.belief = BeliefField(d=d)
        self.verifier = SelfVerifier(d=d, threshold=0.5)
        self.proof_gen = ProofCarryingGenerator(d=d)
        # Pack 212 -- Sleep wire instances
        from ikigai.cognition.schema_inducer import SchemaInducer
        from ikigai.cognition.crystallizer import AtomicCrystallineStore
        self.schema = SchemaInducer()
        self.crystal = AtomicCrystallineStore()
        # Pack 213 -- Self wire instances
        from ikigai.cognition.persona_manifold import BeliefProjectionManifold
        from ikigai.cognition.metacognitive_mirror import MetacognitiveHVMirror
        from ikigai.cognition.importance_decay import ImportanceDecayLattice
        self.persona_proj = BeliefProjectionManifold(d=128)
        self.meta_mirror = MetacognitiveHVMirror(d=128)
        self.imp_lattice = ImportanceDecayLattice(d=128)
        self._self_tick = 0
        # Pack 214 -- Counterfactual wire instances
        from ikigai.cognition.counterfactual_sim import CounterfactualField
        from ikigai.cognition.causal_world_model import CausalWorldModel
        self.cf = CounterfactualField(d=128)
        self.cwm = CausalWorldModel(d=128, perm_shift=1)
        self._last_state_name = None
        self._cwm_state_counter = 0
        # Pack 216 -- ikigai.py BRIDGE (lazy)
        self._bridge = None
        # Pack 218 -- wire the 8 dead modules
        from ikigai.cognition.schema_refiner import SchemaRefiner
        from ikigai.cognition.self_modifying_refiner import SelfModifyingRefiner
        from ikigai.cognition.goal_decomposer import GoalDecomposer
        from ikigai.cognition.world_model import SymbolicWorldModel
        from ikigai.cognition.moe import MoERouter
        from ikigai.cognition.dssc_coupling import (
            ParallelSemSynCoupling, build_default_cfg)
        self.schema_refiner = SchemaRefiner(d=128)
        self.self_mod_refiner = SelfModifyingRefiner(d=128, promote_threshold=2)
        self.goals = GoalDecomposer(d=128)
        self.world = SymbolicWorldModel()
        self.moe = MoERouter()
        try:
            self.dssc = ParallelSemSynCoupling(
                cfg=build_default_cfg(), d_sem=128, d_emit=64,
                eta=0.05, seed=42)
        except Exception:
            self.dssc = None
        # belief_expander needs a vocab HV fn -- lazy build
        self._belief_exp = None
        # verifier.py is module-level functions, not a class -- import lazy
        self._verify_mod = None
        self._fe_log = []
        self._last_passage_hv_curiosity = None
        self._read_organism_count = 0
        self._verifier_scores = []   # Pack 211 -- post-gen verifier log
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

    # ── Primary interface ────────────────────────────────────────────────

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

    # ── language acquisition ─────────────────────────────────────────────

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

    # ── Pack 210 -- BIG WIRE-UP: read with reasoning ─────────────────────────
    def read_organism(self, text, speaker='default'):
        """Pack 210 + 219 -- absorb text through the full cognitive stack.

        Phase 0 (Pack 219): frame routing -- so non-LLM reads also activate
            Pack 197 frames, populate frame_vocab, build frame_bigram FSM.
        Phase 1: base statistical absorb (Pack 197/198/199 path).
        Phases 2-16: free_energy + curiosity + tom + reasoning + belief +
            schema + crystal + persona + meta + importance + cwm + cf +
            wm_sys + concept_graph + event_comp + cell_assembly +
            schema_refiner + world + dssc.
        """
        # Tokenize once for downstream modules
        from ikigai.cognition.flat_memory import tokenize as _utok
        tokens = _utok(text)
        if not tokens:
            return

        # Phase 0 (Pack 219): frame routing + unigram observation FIRST so
        # the substrate writes inside Phase 1's read() are frame-conditioned.
        try:
            self.unified.observe_unigrams(tokens)
            self.route_frame(tokens, observe=True, learn=True)
        except Exception:
            pass

        # Phase 1: existing statistical absorb (now frame-conditioned)
        self.read(text)
        self._read_organism_count += 1

        # Phase 2: free energy ingest (predictive coding)
        try:
            if self.frames is not None:
                passage_hv = self.frames.passage_hv(tokens, self.unified.ck)
                if passage_hv is not None:
                    # ConversationalVFEF expects a real bipolar HV at its own d.
                    # Project complex passage HV to bipolar via phase sign.
                    real_part = passage_hv.real[:self.fe.d]
                    bipolar = np.sign(real_part).astype(np.float32)
                    bipolar = np.where(bipolar == 0, 1.0, bipolar)
                    F = self.fe.ingest(bipolar)
                    self._fe_log.append(float(F))
        except Exception:
            pass

        # Phase 3: curiosity drive prediction-error tracking
        try:
            from ikigai.cognition.curiosity_drive import _encode as _cur_encode
            observed_hv = _cur_encode(tokens, self.curiosity.d)
            predicted_hv = self._last_passage_hv_curiosity \
                if self._last_passage_hv_curiosity is not None else observed_hv.copy()
            # state = first few tokens; action = 'read'
            state_tokens = tokens[:4] if tokens else ['_']
            self.curiosity.record_pe(state_tokens, ['read'],
                                       predicted_hv, observed_hv)
            self._last_passage_hv_curiosity = observed_hv
        except Exception:
            pass

        # Phase 4: theory of mind -- bind sentence as belief of speaker
        try:
            # agent_names() is a method on TheoryOfMindSandbox (NOT a property)
            if speaker not in self.tom.agent_names():
                self.tom.add_agent(speaker)
            key = [tokens[0]]
            val = tokens[1:6] if len(tokens) > 1 else [tokens[0]]
            self.tom.set_belief(speaker, key, val)
        except Exception:
            pass

        # Phase 5 REMOVED Pack 278 v1 (Day 77).  Legacy ReasoningEngine
        # arithmetic/SET/ADD parser was anti-pattern -- OPERATOR_LEXICON
        # purged in reasoning_engine.py.  Math + state-update semantics
        # now live in Pack 254 RHC + Pack 255 GeneralReasoner + Pack 291
        # multiplicative ⋆.  read_statement no longer routes through
        # the legacy parser; downstream Phase 6+ paths read from the
        # tokens already extracted above.

        # Phase 6 (Pack 211) -- belief_field assertion per content token
        try:
            for t in set(tokens):
                if len(t) >= 3:
                    # Use complex key.real as a bipolar projection for belief HV
                    k = self.unified.ck.key(t)
                    bip = np.sign(k.real).astype(np.float32)
                    bip = np.where(bip == 0, 1.0, bip)
                    self.belief.assert_hv(t, bip[:self.belief.d])
        except Exception:
            pass

        # Phase 7 (Pack 212) -- schema_inducer observes (name, tokens). Use
        # the first content token as the "pattern name" so similar-shaped
        # sentences cluster under the same pattern.
        try:
            if tokens:
                pat_name = tokens[0]
                self.schema.observe(pat_name, tokens)
        except Exception:
            pass

        # Phase 8 (Pack 212) -- crystallizer observes SVO triple if RE parsed
        try:
            stmt = self.re.parser.parse_statement(text)
            if stmt is not None and stmt.entity and stmt.obj:
                self.crystal.observe(stmt.entity, stmt.op, stmt.obj)
        except Exception:
            pass

        # Phase 9 (Pack 213) -- persona_manifold update + metacognitive mirror
        try:
            self.persona_proj.update(tokens[:32])
            # encode_utterance also returns the cumulative belief HV
            B_U = self.persona_proj.encode_utterance(tokens[:32])
            self.meta_mirror.update(B_U, tokens[:32])
        except Exception:
            pass

        # Phase 10 (Pack 213) -- importance_decay records sentence
        try:
            self._self_tick += 1
            # Surprise proxy: free-energy recent F value
            surp = self._fe_log[-1] if self._fe_log else 0.0
            name = tokens[0] if tokens else f'sent{self._self_tick}'
            self.imp_lattice.record(name, tokens[:8], surprise=float(surp),
                                       now=self._self_tick)
        except Exception:
            pass

        # Phase 11 (Pack 214) -- causal_world_model transition
        try:
            self._cwm_state_counter += 1
            state_name = f's{self._cwm_state_counter}'
            self.cwm.add_state(state_name, tokens[:8])
            if self._last_state_name is not None:
                # action = first verb-ish token (or 'read')
                self.cwm.add_action('read', ['read'])
                self.cwm.add_transition(self._last_state_name, 'read',
                                          state_name)
            self._last_state_name = state_name
        except Exception:
            pass

        # Phase 12 (Pack 214) -- counterfactual scenario per sentence
        try:
            scenario_name = f'sc{self._cwm_state_counter}'
            # action_tokens = first half, outcome_tokens = second half
            mid = max(1, len(tokens) // 2)
            self.cf.add_scenario(scenario_name, tokens[:mid], tokens[mid:])
        except Exception:
            pass

        # Phase 13 (Pack 217) -- WorkingMemorySystem holds last labels
        try:
            label = ' '.join(tokens[:4]) if tokens else 'empty'
            self.wm_sys.add(label, ado_level=0.1,
                              dlpfc_spikes=1, dlpfc_total=5)
            self.wm_sys.tick()
        except Exception:
            pass

        # Phase 14 (Pack 217) -- ConceptGraph ingests event = bag of content tokens
        try:
            event = {
                'tokens': tokens[:16],
                'valence': 0.0,
                'salience': 1.0,
                'tick': self._self_tick,
            }
            self.concept_graph.ingest_event(event)
        except Exception:
            pass

        # Phase 15 (Pack 217) -- EventCompressor ingests episode transition
        try:
            ep = {
                'tokens': tokens[:16],
                'tick': self._self_tick,
                'theme': tokens[0] if tokens else 'none',
            }
            self.event_comp.ingest_transition(ep)
        except Exception:
            pass

        # Phase 16a (Pack 218) -- schema_refiner observes pattern
        try:
            if tokens:
                self.schema_refiner.observe(tokens[0], tokens)
        except Exception:
            pass

        # Phase 16b (Pack 218) -- world_model learns symbolic facts from tokens
        try:
            self.world.learn_from_tokens(tokens)
        except Exception:
            pass

        # Phase 16c (Pack 218) -- DSSC dual-stream coupling (silent if no CFG)
        try:
            if self.dssc is not None and tokens:
                self.dssc.step(tokens[:8])
        except Exception:
            pass

        # Last assigned frame stat (for absorb_llm_deep + diagnostics)
        try:
            la = self.frames.last_assigned
            if la is not None and la >= 0:
                pass  # already counted via frames.assigns_per_frame
        except Exception:
            pass

        # Phase 16 (Pack 217) -- CellAssemblySystem (requires neuromod state).
        # Skip silently if neuromod not built (lazy attr).
        try:
            nm = getattr(self, '_neuro', None)
            if nm is not None:
                lvl = nm.level if hasattr(nm, 'level') else {}
                self.cell_assembly.update(
                    cort=lvl.get('cort', 0.0),
                    ne=lvl.get('ne', 0.3),
                    soma_m=0.0,
                    da=lvl.get('da', 0.5),
                    oxt=lvl.get('oxt', 0.3),
                    ach=lvl.get('ach', 0.4),
                    nov=0.5,
                    dmn_act=0.5,
                    res=0.0,
                    ht=lvl.get('ht', 0.6),
                    tick=self._self_tick,
                )
        except Exception:
            pass

        # Pack 219: clear frame so subsequent unrelated reads aren't conditioned
        try:
            self.clear_frame()
        except Exception:
            pass

    @property
    def bridge(self):
        """Pack 216 -- lazy-loaded IkigaiBridge. First access triggers exec()
        of patched ikigai.py (~0.5s). Returns IkigaiBridge with .classes,
        .cls(name), .has(name), .get(name)."""
        if self._bridge is None:
            from ikigai_bridge import IkigaiBridge
            self._bridge = IkigaiBridge.load(verbose=False)
        return self._bridge

    # Pack 217 -- four high-value ikigai.py classes auto-instantiated on first
    # use. WorkingMemorySystem (capacity-bounded working set), ConceptGraph
    # (role-bound knowledge graph), EventCompressor (temporal compression),
    # CellAssemblySystem (substrate cell pattern -- needs neuromod state).
    @property
    def wm_sys(self):
        if not hasattr(self, '_wm_sys') or self._wm_sys is None:
            self._wm_sys = self.bridge.cls('WorkingMemorySystem')(slots=8, decay=10)
        return self._wm_sys

    @property
    def concept_graph(self):
        if not hasattr(self, '_cg') or self._cg is None:
            self._cg = self.bridge.cls('ConceptGraph')(max_nodes=256,
                                                          similarity_threshold=0.85)
        return self._cg

    @property
    def event_comp(self):
        if not hasattr(self, '_ec') or self._ec is None:
            self._ec = self.bridge.cls('EventCompressor')(maxlen=500,
                                                              min_event_len=3)
        return self._ec

    @property
    def cell_assembly(self):
        if not hasattr(self, '_cas') or self._cas is None:
            self._cas = self.bridge.cls('CellAssemblySystem')()
        return self._cas

    # Pack 218 -- lazy properties
    @property
    def belief_exp(self):
        if self._belief_exp is None:
            from ikigai.cognition.belief_expander import BeliefConditionedExpander
            d = 128
            # Bipolar projection of ComputedKey for any token
            def _vhv(w):
                k = self.unified.ck.key(w)
                bip = np.sign(k.real).astype(np.float32)
                bip = np.where(bip == 0, 1.0, bip)
                return bip[:d]
            self._belief_exp = BeliefConditionedExpander(
                vocab_hv_fn=_vhv, d=d, n=2, max_expand=8, top_candidates=20)
        return self._belief_exp

    @property
    def verify_mod(self):
        if self._verify_mod is None:
            from ikigai.cognition import verifier as _v
            self._verify_mod = _v
        return self._verify_mod

    def organism_status(self):
        """Pack 210 -- quick snapshot of the cognitive stack."""
        out = {'reads': self._read_organism_count}
        try:
            out['fe_mean_F'] = float(np.mean(self._fe_log)) if self._fe_log else 0.0
            out['fe_recent'] = self._fe_log[-1] if self._fe_log else 0.0
        except Exception:
            pass
        try:
            # n_logged / n_visited_states are @property, NOT methods
            out['curiosity_n_logged'] = self.curiosity.n_logged
            out['curiosity_visited_states'] = self.curiosity.n_visited_states
            out['curiosity_top'] = self.curiosity.top_curious(top_k=5)
        except Exception as e:
            out['curiosity_err'] = str(e)[:80]
        try:
            out['tom_agents'] = self.tom.agent_names()
            out['tom_agent_beliefs'] = {a: self.tom.agent(a).n_beliefs()
                                            for a in self.tom.agent_names()}
        except Exception:
            pass
        try:
            out['wm_keys'] = list(self.re.wm.history())[-8:] if hasattr(self.re.wm,'history') else []
        except Exception:
            pass
        return out

    # ── unified-memory interface (Pack 118): query the one flat substrate ────

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

    # Pack 221 -- Self-Teaching Property Extractor
    # Mines property/mod/similar/antonym/affordance triples from raw text
    # using regex patterns. Closes the gap between raw-text absorb and the
    # property-role structure that Pack 158's analogy benchmark proved
    # delivers 100% top-3.
    _MINER_PATTERNS = None  # lazy-compiled

    @classmethod
    def _get_miner_patterns(cls):
        if cls._MINER_PATTERNS is None:
            import re as _re
            STOP = {'the','a','an','of','to','in','on','at','by','for','from',
                     'with','as','it','its','this','that','these','those'}
            cls._MINER_PATTERNS = {
                # property: "X is/are/was/were Y" where Y is adjective-like (not stop)
                'property_is': _re.compile(
                    r'\b(\w+)\s+(?:is|are|was|were)\s+(\w+)\b', _re.IGNORECASE),
                # property: "X has/have Y"
                'property_has': _re.compile(
                    r'\b(\w+)\s+(?:has|have)\s+(\w+)\b', _re.IGNORECASE),
                # mod: adjective-noun adjacency "ADJ NOUN" -- we cheat with bigram
                # mining since we have no POS tagger; the read substrate filters
                # which actually fire as adj.
                # Skipping mod here (would need POS). Pack 224 candidate.
                # similar: "X and Y" within bound length
                'similar_and': _re.compile(
                    r'\b(\w+)\s+and\s+(\w+)\b', _re.IGNORECASE),
                # antonym: "X but Y" "X not Y" "X unlike Y"
                'antonym_but': _re.compile(
                    r'\b(\w+)\s+(?:but|unlike|opposite)\s+(\w+)\b', _re.IGNORECASE),
                # affordance: "SUBJ VERB OBJ" -- too generic without POS, skip
            }
            cls._MINER_STOP = STOP
        return cls._MINER_PATTERNS

    # Pack 221 v2 -- SVO triple miner using crystallizer + targeted writes.
    # Extracts (subject, verb, object) via simple verb-anchored regex, feeds
    # to crystal.observe for pattern discovery, and writes to property /
    # affordance roles for concept arithmetic.
    _SVO_PATTERNS = None

    @classmethod
    def _get_svo_patterns(cls):
        if cls._SVO_PATTERNS is None:
            import re as _re
            # Verbs that imply property: "X is/are Y" => property(X, Y)
            PROP_VERBS = r'(?:is|are|was|were|am|be|been)'
            # Verbs that imply possession: "X has/have Y" => property(X, Y) too
            HAVE_VERBS = r'(?:has|have|had|having)'
            # Action verbs: any other verb between two nouns => affordance
            cls._SVO_PATTERNS = {
                # X is/are (a|an|the)? Y                       -> property
                'prop_is': _re.compile(
                    r'\b(\w+)\s+' + PROP_VERBS
                    + r'(?:\s+(?:a|an|the))?\s+(\w+)\b'),
                # X has/have (a|an|the)? Y                     -> property
                'prop_has': _re.compile(
                    r'\b(\w+)\s+' + HAVE_VERBS
                    + r'(?:\s+(?:a|an|the))?\s+(\w+)\b'),
                # X became/becomes Y                            -> property
                'prop_become': _re.compile(
                    r'\b(\w+)\s+(?:became|becomes|become)\s+(?:a|an|the)?\s*(\w+)\b'),
                # X (verb) Y where verb is common action       -> affordance
                'aff_action': _re.compile(
                    r'\b(\w+)\s+(eats|eat|drinks|drink|runs|run|jumps|jump|'
                    r'flies|fly|swims|swim|sleeps|sleep|sings|sing|barks|bark|'
                    r'meows|meow|moos|moo|hunts|hunt|chases|chase|catches|catch|'
                    r'lives|live|grows|grow|carries|carry|gives|give|sees|see|'
                    r'hears|hear|feels|feel|likes|like|loves|love)\s+(\w+)\b'),
            }
            cls._SVO_STOP = {
                'the','a','an','of','to','in','on','at','by','for','from',
                'with','as','it','its','this','that','these','those','he',
                'she','him','her','his','they','them','their','there','here',
                'who','whom','which','what','when','where','why','how','some',
                'any','one','two','three','also','very','more','most','only',
                'just','then','than','so','if','because','about','into',
                'i','you','we','us','our','my','your','out','up','down','over',
                'under','again','still','always','never',
            }
        return cls._SVO_PATTERNS

    def mine_svo_triples(self, sentences=None, n_reinforce=12,
                            crystal_observe=True, verbose=True):
        """Pack 221 v2 -- mine SVO triples and write to property + affordance
        roles. Also feeds crystallizer for pattern discovery."""
        patterns = self._get_svo_patterns()
        stop = self._SVO_STOP
        if sentences is None:
            buf = getattr(self, '_exposure_buf', None)
            if buf is None:
                return {'err': 'no exposure buf + no sentences'}
            sentences = list(buf)
        stats = {'property': 0, 'affordance': 0, 'crystal': 0, 'skipped': 0}
        seen_prop = set()
        seen_aff = set()
        mr = self.unified
        for text in sentences:
            tl = text.lower() if isinstance(text, str) else str(text).lower()
            # property patterns: X is/are/has/have Y
            for key in ('prop_is', 'prop_has', 'prop_become'):
                regex = patterns[key]
                for m in regex.finditer(tl):
                    s, o = m.group(1), m.group(2)
                    if s in stop or o in stop or len(s) < 2 or len(o) < 2 or s == o:
                        stats['skipped'] += 1; continue
                    if any(c.isdigit() for c in s + o):
                        stats['skipped'] += 1; continue
                    key_t = ('property', s, o)
                    if key_t in seen_prop: continue
                    seen_prop.add(key_t)
                    for _ in range(n_reinforce):
                        mr.relate(s, 'property', o)
                    mr._role_targets.setdefault('property', set()).add(o)
                    stats['property'] += 1
                    if crystal_observe and hasattr(self, 'crystal'):
                        try:
                            self.crystal.observe(s, 'is', o)
                            stats['crystal'] += 1
                        except Exception:
                            pass
            # affordance: X VERB Y for action verbs
            for m in patterns['aff_action'].finditer(tl):
                s, v, o = m.group(1), m.group(2), m.group(3)
                if s in stop or o in stop or len(s) < 2 or len(o) < 2 or s == o:
                    stats['skipped'] += 1; continue
                key_t = ('affordance', s, v, o)
                if key_t in seen_aff: continue
                seen_aff.add(key_t)
                # write to affordance role: subj -> verb, verb -> obj
                for _ in range(n_reinforce):
                    mr.relate(s, 'affordance', v)
                    mr.relate(v, 'affordance', o)
                mr._role_targets.setdefault('affordance', set()).add(v)
                mr._role_targets.setdefault('affordance', set()).add(o)
                stats['affordance'] += 1
                if crystal_observe and hasattr(self, 'crystal'):
                    try:
                        self.crystal.observe(s, v, o)
                        stats['crystal'] += 1
                    except Exception:
                        pass
        if verbose:
            print(f'  [Pack 221v2] mined SVO: {stats}', flush=True)
        return stats

    def propagate_multihop(self, max_iter=3, n_reinforce=6, verbose=True):
        """Pack 222 -- Multi-Hop Property Propagation. NOVEL primitive.

        Reads triples from crystallizer (populated by mine_svo_triples), runs
        inheritance + transitive closure across them, writes derived triples
        back to substrate's property + isa roles.

        Inference rules:
            R1 (transitive is): (X, is, Y) + (Y, is, Z) -> (X, is, Z)
            R2 (inheritance):   (X, is, Y) + (Y, property, Z) -> (X, property, Z)
            R3 (similar):       (X, similar, Y) -> (Y, similar, X)  [symmetric]

        This is the novel mechanism that lets `(kitten, is, young_cat)` +
        `(young, property, baby)` derive `(kitten, property, baby)` -- closing
        the gap between observed Hearst patterns and the property-axis
        structure analogies need.
        """
        if not hasattr(self, 'crystal'):
            return {'err': 'crystal not built; run mine_svo_triples first'}
        mr = self.unified
        # Build adj-list from crystal triples filtered to is/property relations
        is_edges = {}       # x -> set of y where (x, is, y)
        prop_edges = {}     # x -> set of z where (x, property, z) (mirrored from is)
        triples = list(self.crystal._counts.keys()) if hasattr(self.crystal, '_counts') else []
        for triple in triples:
            if len(triple) != 3: continue
            s, p, o = triple
            if p == 'is':
                is_edges.setdefault(s, set()).add(o)
                # crystal 'is' triples mirror property writes from mine_svo_triples
                prop_edges.setdefault(s, set()).add(o)

        stats = {'rule_R1_transitive': 0, 'rule_R2_inheritance': 0,
                 'iter': 0, 'pre_property': len(mr._role_targets.get('property', set())),
                 'pre_isa': len(mr._role_targets.get('isa', set()))}
        seen_R1 = set()
        seen_R2 = set()
        for it in range(int(max_iter)):
            new_count = 0
            # R1: transitive is
            for x, ys in list(is_edges.items()):
                for y in list(ys):
                    targets = is_edges.get(y, set())
                    for z in targets:
                        if z == x or z in ys: continue
                        key = ('R1', x, z)
                        if key in seen_R1: continue
                        seen_R1.add(key)
                        is_edges.setdefault(x, set()).add(z)
                        prop_edges.setdefault(x, set()).add(z)
                        # write to substrate
                        for _ in range(n_reinforce):
                            mr.relate(x, 'isa', z)
                            mr.relate(x, 'property', z)
                        mr._role_targets.setdefault('isa', set()).add(z)
                        mr._role_targets.setdefault('property', set()).add(z)
                        stats['rule_R1_transitive'] += 1
                        new_count += 1
            # R2: inheritance via property
            for x, ys in list(is_edges.items()):
                for y in list(ys):
                    props_of_y = prop_edges.get(y, set())
                    for z in props_of_y:
                        if z == x: continue
                        key = ('R2', x, z)
                        if key in seen_R2: continue
                        seen_R2.add(key)
                        prop_edges.setdefault(x, set()).add(z)
                        for _ in range(n_reinforce):
                            mr.relate(x, 'property', z)
                        mr._role_targets.setdefault('property', set()).add(z)
                        stats['rule_R2_inheritance'] += 1
                        new_count += 1
            stats['iter'] = it + 1
            if new_count == 0:
                break
        stats['post_property'] = len(mr._role_targets.get('property', set()))
        stats['post_isa'] = len(mr._role_targets.get('isa', set()))
        if verbose:
            print(f'  [Pack 222] propagated: {stats}', flush=True)
        return stats

    def mine_properties(self, sentences=None, n_reinforce=8, max_per_role=50000,
                          verbose=True):
        """Pack 221 -- mine property/similar/antonym triples from raw text.

        Reads given sentences (or self._exposure_buf entries) with regex
        patterns, writes triples into the substrate using relate() with
        n_reinforce repetitions. Returns stats dict.

        This is the bridge between raw text absorb and Pack 158's
        property-structured substrate.
        """
        patterns = self._get_miner_patterns()
        stop = self._MINER_STOP
        if sentences is None:
            buf = getattr(self, '_exposure_buf', None)
            if buf is None:
                return {'err': 'no exposure log + no sentences provided'}
            sentences = list(buf)
        stats = {role: 0 for role in ('property', 'similar', 'antonym')}
        seen_triples = set()
        mr = self.unified
        for text in sentences:
            tl = text.lower() if isinstance(text, str) else str(text).lower()
            # property_is + property_has
            for role_key, regex in patterns.items():
                role = role_key.split('_')[0]
                if stats[role] >= max_per_role:
                    continue
                for m in regex.finditer(tl):
                    a, b = m.group(1), m.group(2)
                    if a in stop or b in stop:
                        continue
                    if len(a) < 2 or len(b) < 2:
                        continue
                    if a == b:
                        continue
                    key = (role, a, b)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    # n reinforcements
                    for _ in range(n_reinforce):
                        mr.relate(a, role, b)
                    mr._role_targets.setdefault(role, set()).add(b)
                    stats[role] += 1
                    if role == 'similar' or role == 'antonym':
                        # symmetric
                        for _ in range(n_reinforce):
                            mr.relate(b, role, a)
                        mr._role_targets[role].add(a)
        if verbose:
            print(f'  [Pack 221] mined: {stats}', flush=True)
        return stats

    # ── Pack 162: Reversible Writes (Kill Stack #4) ───────────────────────
    def unlearn(self, word, role, target, n=20):
        """
        Reverse n writes of (word, role) -> target. Use the same n that
        was used when the fact was originally asserted.

        Kill Stack #4: clean knowledge editing on demand. Pop a specific
        fact off the substrate without retraining anything.
        """
        for _ in range(int(n)):
            self.unified.unrelate(word, role, target)

    def unlearn_isa(self, hypo, hyper, n=20):
        return self.unlearn(hypo, 'isa', hyper, n=n)

    def unlearn_sensory(self, word, anchor, n=20):
        return self.unlearn(word, 'sensory', anchor, n=n)

    # ── Pack 147: multi-channel meaning exposure ───────────────────────────
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

    # ── Pack 148/157: SelfDefiningConcepts + gamma presets ─────────────────
    def build_concepts(self, words=None, iterations=8, verbose=False,
                       weights=None, preset=None, write_to_substrate=True):
        """
        Build per-word concept HVs by iteratively condensing every role
        channel's facts about each word into a single fixed-point HV.

        preset (Pack 157): named gamma weighting.
            'general'        -- balanced (default)
            'analogy'        -- property-dominant, best for king-man+woman
                                arithmetic (Pack 157: 78% top-3)
            'categorical'    -- isa-dominant, best for "what category is X"
            'distributional' -- cooccur-dominant, word2vec-flavour neighbours
            'broad'          -- all channels equal-weighted

        weights overrides preset if both given.
        Returns the ConceptSynthesizer with `.concepts` populated and
        (optionally) written back to the substrate under role 'concept'.
        """
        from ikigai.cognition.concept_synthesizer import ConceptSynthesizer
        cs = ConceptSynthesizer(self.unified, weights=weights, preset=preset)
        self._concept_deltas = cs.build(words=words, iterations=iterations,
                                        verbose=verbose)
        if write_to_substrate:
            cs.write_to_substrate()
        self._concepts = cs
        return cs

    def concept(self, word):
        """Get the condensed concept HV for a word, if built."""
        cs = getattr(self, '_concepts', None)
        return None if cs is None else cs.concept_of(word)

    def what_means(self, word, top_k=10):
        """Return top-K semantically nearest concepts to `word`."""
        cs = getattr(self, '_concepts', None)
        return [] if cs is None else cs.neighbors(word, top_k=top_k)

    def concept_arithmetic(self, plus=None, minus=None, top_k=5):
        """
        VSA concept arithmetic on the substrate:
            sum(concept[w] for w in plus) - sum(concept[w] for w in minus)
        Returns top-K nearest concepts (excluding the input words).
        """
        cs = getattr(self, '_concepts', None)
        if cs is None: return []
        return cs.arithmetic(plus_words=plus, minus_words=minus, top_k=top_k)

    # ── Pack 225 -- Vector Symbolic Finite State Machine ─────────────────
    @property
    def vs_fsm(self):
        """Lazy-built VSFiniteStateMachine. Day 67 Pack 225 compositional
        generation primitive. Records transitions awake; abstracts via isa
        parents during sleep; generates via Resonator-cleaned next-role queries.
        """
        fsm = getattr(self, '_vs_fsm', None)
        if fsm is None:
            from ikigai.cognition.vs_fsm import VSFiniteStateMachine
            fsm = VSFiniteStateMachine(self)
            self._vs_fsm = fsm
        return fsm

    @property
    def opv(self):
        """Pack 251 On-Policy Evaluator -- teacher-gated three-factor
        plasticity. Lazy-built; stateless tunables, no persist needed.

        Use:
            org.opv.gated_observe(prev, cur, actual, role='next',
                                   candidates=org.unified._role_targets['next'])
        """
        opv = getattr(self, '_opv', None)
        if opv is None:
            from ikigai.cognition.on_policy_eval import OnPolicyEvaluator
            opv = OnPolicyEvaluator(self.unified, self.vs_fsm)
            self._opv = opv
        return opv

    @property
    def num_enc(self):
        """Pack 252 NumericEncoder -- Fractional Power Encoding for
        magnitude-aware numeric HVs. Persisted (phases vector is
        load-bearing -- not just seed)."""
        ne = getattr(self, '_num_enc', None)
        if ne is None:
            from ikigai.cognition.numeric_encoder import NumericEncoder
            ne = NumericEncoder(d=self.d, scale=10.0, seed=2520)
            self._num_enc = ne
        return ne

    @property
    def cat3(self):
        """Pack 253 cat-3 reasoning state-graph absorb engine.
        Composes opv (Pack 251) + num_enc (Pack 252) + cooccur."""
        c = getattr(self, '_cat3', None)
        if c is None:
            from ikigai.cognition.cat3_absorb import Cat3Absorb
            c = Cat3Absorb(self.unified, self.opv, self.num_enc)
            self._cat3 = c
        return c

    @property
    def general_reasoner(self):
        """Pack 255 GeneralReasoner -- substrate-native general
        reasoning. NO task-specific paths. Composes PiK + CausalWorldModel
        + LogicalFixedPoint + MultiStepPlanner + Pack 252 FPE + Pack 253
        cat-3 + Pack 251 opv. Math/code/language same entry. Distinct
        from `self.reasoner` (Day 56 Pack 93 hardcoded ReasoningEngine,
        kept for call-site compatibility; anti-pattern per Day 73 pivot)."""
        r = getattr(self, '_general_reasoner', None)
        if r is None:
            from ikigai.cognition.general_reasoner import GeneralReasoner
            r = GeneralReasoner(self)
            self._general_reasoner = r
            # Pack 294: auto-enable active learning when a teacher URL is
            # configured (opt-in via env, mirrors NEUROSEED_LMDB_CACHE).
            # Inert otherwise -- benches never trigger teacher calls.
            import os as _os
            if _os.environ.get('NEUROSEED_TEACHER_URL'):
                try:
                    self.enable_active_learning()
                except Exception:
                    pass
        return r

    def enable_active_learning(self, url=None, lo=0.0, hi=0.999,
                               multiword=False, backend=None,
                               groq_model='llama-3.3-70b-versatile'):
        """Pack 294 -- wire a teacher oracle into general_reasoner so the
        live organism self-teaches on uncertain fact queries.  Returns
        True when an oracle was attached, False otherwise.

        Pack 307 (Day 80 #1) -- multiword=True stores the FULL answer
        phrase ('south america', 'buenos aires', 'may 16 2015') instead of
        the single last token.

        Pack 308 (Day 80 #2) -- `backend` selects the teacher source:
          * 'groq'  -> Groq cloud instruct model (clean answers, no R1
                       think-block; stronger on obscure facts).  Needs
                       GROQ_API_KEY.  `groq_model` picks the model.
          * 'vllm' / None -> the 3090 RemoteLLMTeacher (env
                       NEUROSEED_TEACHER_URL).
        Defaults to env NEUROSEED_TEACHER_BACKEND, else vllm."""
        import os
        backend = (backend or os.environ.get('NEUROSEED_TEACHER_BACKEND')
                   or 'vllm').lower()
        from ikigai.cognition.cat4_dopamine import TeacherOracle
        ntok = 32 if multiword else 12
        try:
            if backend == 'groq':
                from ikigai.cognition.groq_teacher import GroqTeacher
                teacher = GroqTeacher(
                    model=groq_model, temperature=0.0, top_p=1.0,
                    max_new_tokens=ntok)
                oracle = TeacherOracle(teacher, max_tokens=ntok,
                                       multiword=multiword)
                self.general_reasoner.enable_active_learning(
                    oracle, lo=lo, hi=hi)
                self._active_learning_url = f'groq:{groq_model}'
                return True
            # vLLM 3090 backend
            from ikigai.cognition.remote_llm_teacher import RemoteLLMTeacher
            url = url or os.environ.get('NEUROSEED_TEACHER_URL')
            if not url:
                return False
            teacher = RemoteLLMTeacher(
                base_url=url, temperature=0.0, top_p=1.0,
                repetition_penalty=1.0, max_new_tokens=ntok,
                strip_think=True)
            oracle = TeacherOracle(teacher, max_tokens=ntok,
                                   multiword=multiword)
            self.general_reasoner.enable_active_learning(oracle, lo=lo, hi=hi)
            self._active_learning_url = url
            return True
        except Exception:
            return False

    def discover_rules(self, min_support=6, min_conf=0.7,
                       self_compress=False, verbose=False):
        """Pack 305.1 -- the organism mines composition rules from its own
        atom index and promotes them, autonomously (no external lists).
        Returns the newly discovered rules.  Hookable into the sleep/idle
        tick so the organism learns rules while resting."""
        eng = self.general_reasoner.derive_engine
        return eng.discover(min_support=min_support, min_conf=min_conf,
                            self_compress=self_compress, verbose=verbose)

    _WHAT_OF_RE = __import__('re').compile(
        r'^\s*what\s+is\s+the\s+(\w+)\s+of\s+(.+?)\s*\??\s*$',
        __import__('re').IGNORECASE)

    @property
    def lang_teacher(self):
        """Pack 300.1 -- lazy LanguageTeacher; restores persisted learned
        templates from `_lang_templates`."""
        lt = getattr(self, '_lang_teacher', None)
        if lt is None:
            from ikigai.cognition.language_teach import LanguageTeacher
            import os as _os
            teacher = None
            url = (getattr(self, '_active_learning_url', None)
                   or _os.environ.get('NEUROSEED_TEACHER_URL'))
            if url:
                try:
                    from ikigai.cognition.remote_llm_teacher import RemoteLLMTeacher
                    teacher = RemoteLLMTeacher(
                        base_url=url, temperature=0.0, top_p=1.0,
                        repetition_penalty=1.0, max_new_tokens=40,
                        strip_think=True)
                except Exception:
                    teacher = None
            lt = LanguageTeacher(self.general_reasoner, teacher)
            lt.load_state(getattr(self, '_lang_templates', {}) or {})
            self._lang_teacher = lt
        return lt

    def teach_language(self, demos_by_type, min_examples=2,
                       min_specificity=0.4, verbose=False):
        """Pack 300.1 -- learn sentence templates per query-type from
        teacher demonstrations (anti-unification).  demos_by_type:
        {qtype: [(subj, val), ...]}.  Persists learned templates into
        `_lang_templates` (NOT b_self).  Returns #types learned."""
        lt = self.lang_teacher
        learned = 0
        for qtype, demos in demos_by_type.items():
            if lt.teach(qtype, demos, min_examples=min_examples,
                        min_specificity=min_specificity, verbose=verbose):
                learned += 1
        self._lang_templates = lt.to_state()
        return learned

    def say(self, query, do_active=False):
        """Pack 300 v0 + 300.1 -- answer a query as a grammatical SENTENCE.
        reason() supplies the substrate answer.  If a LEARNED template
        (Pack 300.1) exists for the query-type, use it; else fall back to
        the schema framer (Pack 300 v0).  Returns {answer, sentence,
        method, grammatical, framed_by}."""
        framer = getattr(self, '_sentence_framer', None)
        if framer is None:
            from ikigai.cognition.sentence_frame import SentenceFramer
            framer = SentenceFramer()
            self._sentence_framer = framer
        r = self.general_reasoner.reason(query, do_active=do_active)
        ans = r.get('answer')
        if ans is None:
            return {'answer': None, 'sentence': None, 'method': r.get('method'),
                    'grammatical': False, 'framed_by': None}
        sent, framed_by = None, None
        # Pack 300.1 -- learned template for "what is the <rel> of <subj>"
        m = self._WHAT_OF_RE.match(query)
        if m:
            lt = getattr(self, '_lang_teacher', None) or self.lang_teacher
            qtype = m.group(1).lower()
            if qtype in lt.templates:
                sent = lt.say(qtype, m.group(2).strip(), ans)
                framed_by = 'learned'
        if sent is None:
            sent = framer.frame(query, ans, r.get('method'))
            framed_by = 'schema'
        return {'answer': ans, 'sentence': sent, 'method': r.get('method'),
                'grammatical': framer.is_grammatical(sent, str(ans)),
                'framed_by': framed_by}

    def say_free(self, seed='the', max_len=8, pmi=True,
                 no_immediate_repeat=True, candidates=None,
                 self_terminate=True, k_sigma=None):
        """Pack 311 + Pack 318 (Day 80) -- FREE-FLUENCY generation that STOPS
        ITSELF.  A greedy walk over the transition banks picking each next
        token via the Product-of-Experts AND-gate fuser (unified.poe_candidates).

        Pack 318 SELF-TERMINATION: the organism decides when it is done, we do
        not tell it.  TWO self-stop signals, either fires END:
          (1) CALIBRATION boundary -- raw recall confidence of the best
              continuation below k/sqrt(2d) (line #11 geometry): nothing it
              confidently has to add. (Works on clean banks; on the SDM
              transition bank, crosstalk can fabricate false confidence on an
              unwritten successor, so signal (2) is needed too.)
          (2) CYCLE detection -- the next bigram (prev,cur) has already been
              emitted this walk: it is repeating itself => no NEW content =>
              done.  This is the organism noticing it has run out of things to
              say, not a length we imposed.
        max_len is only a safety cap.  Returns {tokens, text, stopped} with
        stopped in {'self','cycle','cap','dead'}.
        """
        from ikigai.cognition.calibration import abstain_boundary
        mr = self.unified
        boundary = abstain_boundary(mr.d) if k_sigma is None else \
            abstain_boundary(mr.d, k_sigma)
        hist = list(seed.split()) if isinstance(seed, str) else list(seed)
        if not hist:
            hist = ['the']
        cands = candidates if candidates is not None else mr._cooccur_seen
        stopped = 'cap'
        seen_bigrams = set()
        for _ in range(max(0, int(max_len) - len(hist))):
            # Pack 318 signal (1): raw recall confidence below the calibration
            # boundary => no confident continuation => self-stop.
            if self_terminate:
                raw = mr.next_word_candidates(hist[-1], candidates=cands,
                                              top_k=1)
                if not raw or raw[0][1] < boundary:
                    stopped = 'self'
                    break
            ranked = mr.poe_candidates(hist, candidates=cands, top_k=8,
                                       pmi=pmi)
            if not ranked:
                stopped = 'dead'
                break
            nxt = None
            for w, _sc in ranked:
                # no_immediate_repeat is a universal anti-loop, not a lexicon
                if no_immediate_repeat and hist and w == hist[-1]:
                    continue
                nxt = w
                break
            if nxt is None:
                nxt = ranked[0][0]
            # Pack 318 signal (2): cycle detection -- repeating a bigram means
            # no new content, so the organism stops itself.
            if self_terminate:
                bg = (hist[-1], nxt)
                if bg in seen_bigrams:
                    stopped = 'cycle'
                    break
                seen_bigrams.add(bg)
            hist.append(nxt)
        return {'tokens': hist, 'text': ' '.join(hist), 'stopped': stopped}

    def say_frame(self, message=None, frame=None, category_of=None,
                  fsm2=None, cat_vocab=None, n_iters=6, seed=0, pmi=True):
        """Pack 313 (Day 80) -- FRAME-THEN-FILL generation (research mechanism
        #4, the "beat next-token prediction" engine).  Not autoregressive: lay
        a grammatical FRAME from a category FSM, then FILL all slots at once via
        bidirectional parallel relaxation over the real next bank.  Function
        words come from the frame's structural slots, so they cannot form an
        attractor (unlike say_free's greedy walk, which still phrase-loops).

        Categories come from the substrate.  If category_of/fsm2/cat_vocab are
        not supplied, build them from this organism's FrameField (learned
        clusters) when one is attached.  Returns {tokens, text} or None if no
        category structure is available.
        """
        from ikigai.cognition.frame_relax import FrameRelaxGenerator
        mr = self.unified
        if category_of is not None and fsm2 is not None and cat_vocab is not None:
            gen = FrameRelaxGenerator(mr, category_of, fsm2, cat_vocab, pmi=pmi)
        elif getattr(self, '_free_gen', None) is not None:
            # Pack 316-wire: cached generator fit via fit_free_fluency
            # (induced POS + whole-template frames).
            gen = self._free_gen
        else:
            ff = getattr(mr, '_frame_field_ref', None) or getattr(self, 'frame_field', None)
            pool = (mr._role_targets.get('next', set())
                    | mr._role_targets.get('next2', set())) & set(mr._cooccur_seen)
            gen = FrameRelaxGenerator.from_frame_field(mr, ff, pool=pool or None,
                                                       pmi=pmi)
            if gen is None:
                return None
        toks = gen.generate(message=message, frame=frame, n_iters=n_iters,
                            seed=seed)
        return {'tokens': toks, 'text': ' '.join(str(t) for t in toks)}

    def fit_free_fluency(self, texts, K=None, pool_size=700, n_anchors=12,
                         pmi=True):
        """Pack 316-wire -- fit the free-fluency generator from training text.

        Induces SYNTACTIC categories (Pack 314 distributional clustering over
        the next bank: words sharing left+right neighbor profiles = same POS,
        NO labels) and collects WHOLE category-sequence TEMPLATES (Pack 316:
        atomic frames that bypass the Markov frame-mixing wall) from `texts`.
        Caches a FrameRelaxGenerator on self._free_gen so say_frame() can run
        end-to-end on a trained organism. Returns a summary dict.

        texts: iterable of sentence strings (the prose the organism learned).
        Assumes the same text was already absorbed via expose_transitions /
        observe_unigrams so the next bank + unigram prior are populated.
        """
        import numpy as np
        from collections import Counter
        from sklearn.cluster import KMeans
        from ikigai.cognition.frame_relax import FrameRelaxGenerator
        from ikigai.cognition.flat_memory import tokenize
        mr = self.unified
        sents = [tokenize(t) for t in texts]
        sents = [s for s in sents if len(s) >= 2]
        if not sents:
            return None
        # pool = transition vocab, subsampled to most frequent (dodge junk)
        pool_all = (mr._role_targets.get('next', set())
                    | mr._role_targets.get('next2', set()))
        if mr._cooccur_seen:        # intersect only if cooccur populated
            pool_all = pool_all & set(mr._cooccur_seen)
        uc = mr._unigram_count or {}
        pool = [w for w, _ in sorted(((w, uc.get(w, 0)) for w in pool_all),
                                     key=lambda kv: -kv[1])[:int(pool_size)]]
        if len(pool) < 4:
            return None
        PI = {w: i for i, w in enumerate(pool)}
        anchors = [w for w, _ in sorted(uc.items(),
                                        key=lambda kv: -kv[1])[:int(n_anchors)]]
        A = len(anchors)
        # distributional features via real recall (fwd + bwd)
        fwd = np.zeros((len(pool), A)); bwd = np.zeros((len(pool), A))
        for w in pool:
            for a, s in mr.next_word_candidates(w, candidates=anchors, top_k=A):
                fwd[PI[w], anchors.index(a)] = max(s, 0.0)
        for a in anchors:
            for w, s in mr.next_word_candidates(a, candidates=pool, top_k=len(pool)):
                if w in PI:
                    bwd[PI[w], anchors.index(a)] = max(s, 0.0)
        def _l1(M):
            r = M.sum(1, keepdims=True); r[r == 0] = 1; return M / r
        feat = np.hstack([_l1(fwd), _l1(bwd)])
        if K is None:
            K = max(4, min(24, len(pool) // 6))
        K = min(int(K), len(pool))
        lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit(feat).labels_
        category_of = {w: int(lab[PI[w]]) for w in pool}
        cat_vocab = {}
        for w in pool:
            cat_vocab.setdefault(category_of[w], []).append(w)
        # whole-template frame bank from the training sentences
        tmpl = Counter(tuple(category_of[w] for w in s if w in category_of)
                       for s in sents)
        templates = [(list(f), c) for f, c in tmpl.items() if len(f) >= 2]
        if not templates:
            return None
        self._free_gen = FrameRelaxGenerator(
            mr, category_of, fsm2={}, cat_vocab=cat_vocab, pmi=pmi,
            templates=templates)
        return {'pool': len(pool), 'K': K, 'templates': len(templates),
                'categories': {k: len(v) for k, v in cat_vocab.items()}}

    def ingest_triples(self, triples, discover=False, self_compress=False,
                       min_support=6, min_conf=0.7, fast=True):
        """Pack 326 + 328 -- ingest a stream of (subject, relation, object)
        triples from a knowledge graph (Wikidata / ConceptNet / a TSV dump) as
        atoms, via the cache, using the generic relation template so ANY
        predicate -- not just the hand-listed ones -- round-trips.  This is the
        bridge from a raw KG dump to the derive-not-store kernel: optionally run
        autonomous rule discovery (+ LOSSLESS self-compression) right after, so
        the dump's redundant facts collapse into the irreducible kernel.

        Pack 328 fast=True: direct anchor-cache write (tokenize + hash + set),
        bypassing populate_cache_from_text's format-then-reparse roundtrip --
        the path for million-edge dumps. The anchor matches exactly what atom()
        reads (gr.tokenize of the same question), so the round-trip is identical.

        triples: iterable of (subject, relation, object) string triples.
        Returns {ingested, atoms_before, atoms_after, rules, compressed}.
        """
        eng = self.general_reasoner.derive_engine
        cat4 = self.cat4
        n = 0
        if fast:
            from ikigai.cognition.cat4_absorb import _stable_anchor
            cache = cat4.anchor_actions
            tok = self.general_reasoner.tokenize
            record = eng._record
            tmpl_cache = {}
            for tri in triples:
                if not tri or len(tri) < 3:
                    continue
                s = str(tri[0]).strip().lower()
                r = str(tri[1]).strip().lower()
                o = str(tri[2]).strip().lower()
                if not (s and r and o):
                    continue
                t = tmpl_cache.get(r)
                if t is None:
                    t = eng._templates_for(r)[0]
                    tmpl_cache[r] = t
                anchor = _stable_anchor(tok(t.format(e=s)))
                atoks = tuple(tok(o))
                if not atoks:
                    continue
                _av = getattr(cache, 'add_value', None)
                if _av is not None:                   # Pack 330 multi-value
                    _av(anchor, atoks)
                else:
                    ex = cache.get(anchor)
                    if ex is None:
                        cache[anchor] = [atoks]
                    elif atoks not in ex:
                        ex.append(atoks)
                record(s, r, o)
                n += 1
        else:
            for tri in triples:
                if not tri or len(tri) < 3:
                    continue
                s, r, o = (str(tri[0]).strip().lower(),
                           str(tri[1]).strip().lower(),
                           str(tri[2]).strip().lower())
                if not (s and r and o):
                    continue
                q = eng._templates_for(r)[0].format(e=s)
                cat4.populate_cache_from_text(f'{q}\n\n{o}\n\n')
                eng._record(s, r, o)
                n += 1
        before = len(eng.triples)
        out = {'ingested': n, 'atoms_before': before, 'atoms_after': before,
               'rules': 0, 'compressed': 0}
        if discover:
            added = eng.discover(min_support=min_support, min_conf=min_conf,
                                 self_compress=self_compress)
            out['rules'] = len(added)
            out['atoms_after'] = len(eng.triples)
            out['compressed'] = before - out['atoms_after']
        return out

    def knows(self, entity, rels=None):
        """Pack 329 -- the full MULTI-VALUE meaning web of an entity: every
        relation -> ALL its stored values (richer than describe, which shows
        one value each). Returns {relation: [values]}."""
        eng = self.general_reasoner.derive_engine
        ent = str(entity).strip().lower()
        if rels is None:
            rels = sorted({r for (s, r) in eng.triples if s == ent})
        web = {}
        for r in rels:
            vals = eng.atoms(r, ent)
            if vals:
                web[r] = vals
        return web

    @property
    def ask_role(self):
        """Pack 331 -- the interrogative 'ask' channel (question -> relation),
        lazily attached. Learned, not hardcoded."""
        ar = getattr(self, '_ask_role', None)
        if ar is None:
            from ikigai.cognition.ask_role import AskRole
            ar = AskRole(self)
            self._ask_role = ar
        return ar

    def learn_ask(self, stem, relation):
        """Bind a question's cues to the relation it asks for (from data)."""
        self.ask_role.learn(stem, relation)

    def ask_relation(self, stem, candidates=None, top_k=3):
        """Recall the relation(s) a natural-language question is asking for."""
        return self.ask_role.predict(stem, candidates=candidates, top_k=top_k)

    @property
    def kg_reasoner(self):
        """The multi-hop reasoning engine (comprehend -> derive -> calibrate)
        over a knowledge graph. Lazily attached; load a KB via
        kg_reasoner.load_triples(...) or kg_reasoner.set_adjacency(...)."""
        r = getattr(self, '_kg_reasoner', None)
        if r is None:
            from ikigai.cognition.multihop_reasoner import MultiHopReasoner
            r = MultiHopReasoner(self)
            self._kg_reasoner = r
        return r

    def reason_mc(self, question, choices, concept=None):
        """Answer a multiple-choice question by reliable multi-hop reasoning
        over the loaded knowledge graph. Returns (label, confidence, abstain)."""
        return self.kg_reasoner.answer_mc(question, choices, concept=concept)

    def read_passage(self, text):
        """Pack 302 v0 -- multi-token reading.  Parse a multi-sentence
        passage into atoms (sentence->fact, the inverse of the 300.1
        templates) and absorb them, so comprehension questions that COMBINE
        sentences can be answered via derive (304).  Returns {facts,
        absorbed}.  Relations are explicit in the text ('the <rel> of <X>
        is <Y>') -- no relation-type inference hardcoded."""
        import re as _re
        from ikigai.cognition.compositional import _REL_TEMPLATES
        facts = []
        for sent in _re.split(r'[.!?\n]+', text or ''):
            s = sent.strip()
            if not s:
                continue
            m = _re.match(r'(?i)^the\s+(\w+)\s+of\s+(.+?)\s+is\s+(.+)$', s)
            if m:
                facts.append((m.group(1).lower(), m.group(2).strip().lower(),
                              m.group(3).strip().lower()))
                continue
            m = _re.match(r'(?i)^(.+?)\s+is\s+the\s+capital\s+of\s+(.+)$', s)
            if m:
                facts.append(('capital', m.group(2).strip().lower(),
                              m.group(1).strip().lower()))
        cat4 = self.cat4
        eng = self.general_reasoner.derive_engine
        added = 0
        for rel, subj, val in facts:
            tmpl = _REL_TEMPLATES.get(rel)
            q = tmpl[0].format(e=subj) if tmpl else f'what is the {rel} of {subj}'
            added += cat4.populate_cache_from_text(f'{q}\n\n{val}\n\n')
            eng._record(subj, rel, val)
        return {'facts': facts, 'absorbed': added}

    def describe(self, entity, max_facts=6):
        """Pack 301 v0 -- coherent multi-fact speech.  Gather every fact the
        organism knows about `entity` from its OWN atom index, speak each as
        a sentence (learned template / schema), assemble a topic-anchored
        multi-sentence answer.  Coherence = every sentence is about the
        entity.  Returns {entity, sentences, text, facts, on_topic}."""
        eng = self.general_reasoner.derive_engine
        from ikigai.cognition.compositional import _REL_TEMPLATES
        ent = str(entity).strip().lower()
        # probe every relation the organism has GRAMMAR for (its known
        # relation vocabulary) -- atom() reads the fact cache.  Index-first
        # ordering, then any other known relations.
        idx_rels = [r for (s, r) in eng.triples if s == ent]
        ordered, seen = [], set()
        for r in list(idx_rels) + sorted(eng.relations) + list(_REL_TEMPLATES):
            if r not in seen:
                seen.add(r); ordered.append(r)
        from ikigai.cognition.sentence_frame import SentenceFramer
        framer = getattr(self, '_sentence_framer', None) or SentenceFramer()
        self._sentence_framer = framer
        lt = self.lang_teacher
        sentences, facts = [], []
        for rel in ordered[:max_facts]:
            val = eng.atom(rel, ent)              # value from the atom index
            if not val or val == 'unknown':
                continue
            if rel in lt.templates:
                s, by = lt.say(rel, ent, val), 'learned'
            else:
                s, by = framer.frame(f'what is the {rel} of {ent}', val), 'schema'
            if s:
                sentences.append(s)
                facts.append((rel, val, by))
        text = ' '.join(sentences)
        on_topic = sum(1 for s in sentences if ent in s.lower())
        return {'entity': ent, 'sentences': sentences, 'text': text,
                'facts': facts, 'on_topic': on_topic, 'n': len(sentences)}

    @property
    def cat4(self):
        """Pack 262 cat-4 ICL pair absorb -- Kanerva 2026 focus vector
        for b_self bootstrap. Lazy-built, uses Pack 252 num_enc + Pack
        85 PiK + b_self bank under role 'icl_pair'.

        Pack 273 (Day 76): if `_cat4_anchor_actions_cache` exists
        on the organism (set by a previous bootstrap + save_ikg),
        restore it into the freshly-built Cat4Absorb instance."""
        c = getattr(self, '_cat4', None)
        if c is None:
            from ikigai.cognition.cat4_absorb import Cat4Absorb
            from ikigai.cognition.pi_k_algebra import PiK
            pik = getattr(self, '_pik', None)
            if pik is None:
                pik = PiK(d=self.unified.d, n_primes=16)
                self._pik = pik
            c = Cat4Absorb(self.unified, self.num_enc, pik)
            # Pack 273 + Pack 279 cache restore.  Accepts both legacy
            # dict[str, list[tuple[str]]] format (Pack 273+274 organisms)
            # and the compact dict[int, bytes] format (Pack 279+).  The
            # CompactAnchorCache absorbs either via its `_coerce_value`
            # shim.
            cached_actions = getattr(
                self, '_cat4_anchor_actions_cache', None)
            if cached_actions:
                from ikigai.cognition.cat4_compact_cache import (
                    CompactAnchorCache, migrate_dict_cache)
                from ikigai.cognition.cat4_lmdb_cache import (
                    LMDBAnchorCache, HAVE_LMDB)
                live = c.anchor_actions
                if HAVE_LMDB and isinstance(live, LMDBAnchorCache):
                    # Pack 282.5 LMDB live -- bulk-load any persisted
                    # in-memory state into the LMDB env so we honor
                    # both surfaces during the transition phase.
                    if isinstance(cached_actions, dict):
                        sample = next(
                            iter(cached_actions.values()), None)
                        if isinstance(sample, (bytes, bytearray)):
                            live.update_from_compact(
                                CompactAnchorCache.from_persist_state(
                                    cached_actions))
                        else:
                            live.update_from_dict(cached_actions)
                elif isinstance(cached_actions, dict):
                    sample = next(iter(cached_actions.values()), None)
                    if isinstance(sample, (bytes, bytearray)):
                        # Native Pack 279 compact state
                        c.anchor_actions = (
                            CompactAnchorCache.from_persist_state(
                                cached_actions))
                    else:
                        # Legacy Pack 273/274 dict -- migrate on load
                        c.anchor_actions = migrate_dict_cache(
                            cached_actions)
                else:
                    # Already a cache instance (shouldn't happen
                    # via pickle, but defensive)
                    c.anchor_actions = cached_actions
            self._cat4 = c
        return c

    def _sync_cat4_cache_for_persist(self):
        """Pack 273 + Pack 279 -- before save_ikg, snapshot the live
        cat4 cache into the persistable attr.  The compact cache
        serializes as dict[int, bytes] (~50 B/entry) instead of the
        legacy dict[str, list[tuple[str]]] (~290 B/entry)."""
        c = getattr(self, '_cat4', None)
        cache = getattr(c, 'anchor_actions', None) if c is not None else None
        if not cache:
            return
        from ikigai.cognition.cat4_compact_cache import CompactAnchorCache
        from ikigai.cognition.cat4_lmdb_cache import (
            LMDBAnchorCache, HAVE_LMDB)
        if HAVE_LMDB and isinstance(cache, LMDBAnchorCache):
            # Pack 282.5 LMDB self-persists at its sidecar dir.  Do
            # not duplicate into organism.ikg -- skip the snapshot
            # entirely so save_ikg keeps a tiny attr (the LMDB env
            # is its own atomic store).
            self._cat4_anchor_actions_cache = {}
        elif isinstance(cache, CompactAnchorCache):
            self._cat4_anchor_actions_cache = cache.to_persist_state()
        else:
            self._cat4_anchor_actions_cache = dict(cache)

    def make_resonator(self, codebooks, max_iters=30, beta=8.0,
                         momentum=0.5):
        """Pack 256 ResonatorNetwork factory -- multi-factor decompose
        for FHRR phasor binds. codebooks = list of dicts {name: HV} per
        factor. Returns ResonatorNetwork bound to those codebooks.

        Use to decompose bound HV like bind(role, token1, token2) back
        into (role, token1, token2). Single-pass cleanup
        (mr.resonator_recall, Pack 224) handles single-factor only;
        Pack 256 handles N-factor with mean-field iteration. Capacity
        bound at d=400: ~16 items/factor for 3 factors before degrade
        (Frady+Sommer 2020 empirical match)."""
        from ikigai.cognition.resonator_network import ResonatorNetwork
        return ResonatorNetwork(d=self.unified.d, codebooks=codebooks,
                                  max_iters=max_iters, beta=beta,
                                  momentum=momentum)

    @property
    def active_planner(self):
        """Pack 258 ActiveInferencePlanner -- Expected Free Energy
        action selection. Friston-style: minimizes EFE = -(epistemic +
        pragmatic). Drives organism behavior when goal absent (curiosity
        / info-gain) and when goal present (goal-seeking). Composes
        Pack 72 CausalWorldModel + substrate cleanup confidence. No new
        substrate math. Cat-4 b_self bootstrap depends on this."""
        ap = getattr(self, '_active_planner', None)
        if ap is None:
            from ikigai.cognition.active_inference_planner import (
                ActiveInferencePlanner)
            from ikigai.cognition.causal_world_model import CausalWorldModel
            cwm = getattr(self, '_active_cwm', None)
            if cwm is None:
                cwm = CausalWorldModel(d=self.unified.d)
                self._active_cwm = cwm
            ap = ActiveInferencePlanner(cwm, self.unified)
            self._active_planner = ap
        return ap

    def fsm_observe(self, text, n_reinforce=3, do_trigram=True,
                      surprise_gate=True):
        """Pack 225 awake: record token transitions from text into substrate.
        Pack 238: surprise_gate=True scales writes by Pack 197 write_strength
        so stopwords don't flood the next/next2 banks.
        """
        toks = [t for t in str(text).lower().split() if t]
        return self.vs_fsm.observe_chain(toks, n_reinforce=n_reinforce,
                                            do_trigram=do_trigram,
                                            surprise_gate=surprise_gate)

    def fsm_generate(self, seed, max_tokens=20, n_iters=5, beta=8.0,
                       stop_tokens=None, verbose=False):
        """Pack 225 -- Resonator-decoded FSM generation."""
        return self.vs_fsm.generate(seed, max_tokens=max_tokens,
                                       n_iters=n_iters, beta=beta,
                                       stop_tokens=stop_tokens, verbose=verbose)

    def organism_step(self, prev, current, candidates=None, n_iters=3,
                        beta=8.0, top_k=5,
                        channels=('bigram', 'trigram', 'emergent_pos',
                                  'schema_next', 'crystal', 'frame',
                                  'belief', 'importance', 'concept_graph',
                                  'cwm', 'tom', 'meta', 'vsa', 'wm',
                                  'ngram_cooccur'),
                        weights=None, debug=False):
        """Pack 239b -- ORGANISM inference. Queries multiple cognition channels
        instead of just substrate bigram.

        Channels:
          bigram        : recall(current, 'next')  -- substrate baseline
          trigram       : recall(prev, 'next2')    -- substrate trigram
          emergent_pos  : pos(current) -> recall(pos, 'next') -> isa_inverse
          schema_next   : pos(current) -> recall(pos, 'schema_next') -> inverse
          crystal       : crystallizer SVO triples filtered by current

        Returns top_k [(token, score)] combined across channels.
        """
        import numpy as np
        fsm = self.vs_fsm
        mr = fsm.mr
        if candidates is None:
            candidates = list(mr._role_targets.get('next', set()))
        if not candidates:
            return [(None, 0.0)]
        if weights is None:
            weights = {'bigram': 1.0, 'trigram': 0.5,
                         'emergent_pos': 0.7, 'schema_next': 0.7,
                         'crystal': 0.4, 'frame': 0.3, 'belief': 0.3,
                         'importance': 0.2, 'concept_graph': 0.3,
                         'cwm': 0.3, 'tom': 0.2, 'meta': 0.2,
                         'vsa': 0.4, 'wm': 0.3, 'ngram_cooccur': 0.4}

        score_map = {}
        chan_hits = {c: 0 for c in channels}

        # --- Channel 1: bigram (substrate recall on current) ---
        if 'bigram' in channels:
            try:
                r = mr.recall(current, fsm.NEXT_ROLE)
                results = mr.resonator_recall(r, candidate_words=candidates,
                                                n_iters=n_iters, beta=beta,
                                                top_k=top_k*2)
                for tok, sc in results:
                    score_map[tok] = score_map.get(tok, 0.0) + \
                        weights['bigram'] * float(sc)
                    chan_hits['bigram'] += 1
            except Exception:
                pass

        # --- Channel 2: trigram (substrate recall on prev's next2) ---
        if 'trigram' in channels and prev:
            try:
                r = mr.recall(prev, fsm.PREV_ROLE)
                results = mr.resonator_recall(r, candidate_words=candidates,
                                                n_iters=n_iters, beta=beta,
                                                top_k=top_k*2)
                for tok, sc in results:
                    score_map[tok] = score_map.get(tok, 0.0) + \
                        weights['trigram'] * float(sc)
                    chan_hits['trigram'] += 1
            except Exception:
                pass

        # --- Channel 3: emergent_pos abstract ---
        if 'emergent_pos' in channels:
            emp = getattr(fsm, '_emergent_pos', None) or {}
            inv = getattr(fsm, '_isa_inverse', None) or {}
            pos_label = emp.get(current)
            if pos_label:
                try:
                    r = mr.recall(pos_label, fsm.NEXT_ROLE)
                    results = mr.resonator_recall(r,
                                                    candidate_words=candidates,
                                                    n_iters=n_iters, beta=beta,
                                                    top_k=top_k*4)
                    for tok, sc in results:
                        if tok in inv:
                            for child in list(inv[tok])[:top_k]:
                                if child in candidates:
                                    score_map[child] = score_map.get(child, 0.0) + \
                                        weights['emergent_pos'] * float(sc) * 0.5
                                    chan_hits['emergent_pos'] += 1
                        else:
                            score_map[tok] = score_map.get(tok, 0.0) + \
                                weights['emergent_pos'] * float(sc)
                            chan_hits['emergent_pos'] += 1
                except Exception:
                    pass

        # --- Channel 4: schema_next abstract ---
        if 'schema_next' in channels and 'schema_next' in mr.roles:
            emp = getattr(fsm, '_emergent_pos', None) or {}
            inv = getattr(fsm, '_isa_inverse', None) or {}
            pos_label = emp.get(current)
            if pos_label:
                try:
                    r = mr.recall(pos_label, 'schema_next')
                    results = mr.resonator_recall(r,
                                                    candidate_words=candidates,
                                                    n_iters=n_iters, beta=beta,
                                                    top_k=top_k*4)
                    for tok, sc in results:
                        if tok in inv:
                            for child in list(inv[tok])[:top_k]:
                                if child in candidates:
                                    score_map[child] = score_map.get(child, 0.0) + \
                                        weights['schema_next'] * float(sc) * 0.5
                                    chan_hits['schema_next'] += 1
                        else:
                            score_map[tok] = score_map.get(tok, 0.0) + \
                                weights['schema_next'] * float(sc)
                            chan_hits['schema_next'] += 1
                except Exception:
                    pass

        # --- Channel 5: crystal SVO triples filtered by current ---
        if 'crystal' in channels:
            cryst = getattr(self, '_crystallizer', None) or \
                    getattr(self, 'crystallizer', None)
            if cryst is not None:
                try:
                    triples = getattr(cryst, 'triples', None) or \
                              getattr(cryst, '_triples', None) or []
                    if hasattr(triples, '__iter__'):
                        for tr in triples:
                            if isinstance(tr, (list, tuple)) and len(tr) >= 3:
                                s, v, o = tr[0], tr[1], tr[2]
                                if s == current and v in candidates:
                                    score_map[v] = score_map.get(v, 0.0) + \
                                        weights['crystal'] * 0.5
                                    chan_hits['crystal'] += 1
                                if v == current and o in candidates:
                                    score_map[o] = score_map.get(o, 0.0) + \
                                        weights['crystal'] * 0.5
                                    chan_hits['crystal'] += 1
                except Exception:
                    pass

        # --- Channel 6: frame attractor ---
        if 'frame' in channels:
            try:
                ff = getattr(self, 'frames', None)
                if ff is not None:
                    cur_frame = None
                    for m in ('frame_of_word', 'route_word', 'current_frame'):
                        if hasattr(ff, m):
                            try:
                                v = getattr(ff, m)(current) if callable(
                                    getattr(ff, m)) else getattr(ff, m)
                                if v is not None:
                                    cur_frame = v
                                    break
                            except Exception:
                                continue
                    if cur_frame is not None and hasattr(ff,
                                                          'next_frame_probs'):
                        try:
                            probs = ff.next_frame_probs(cur_frame)
                            if isinstance(probs, dict):
                                wts = getattr(ff, 'word_to_frame', {}) or {}
                                for tok in candidates:
                                    f = wts.get(tok)
                                    if f is not None and f in probs:
                                        score_map[tok] = score_map.get(tok, 0.0) + \
                                            weights['frame'] * float(probs[f])
                                        chan_hits['frame'] += 1
                        except Exception:
                            pass
            except Exception:
                pass

        # --- Channel 7: belief_field topical gating ---
        if 'belief' in channels:
            try:
                bf = getattr(self, 'belief', None)
                if bf is not None and hasattr(bf, 'score'):
                    for tok in list(score_map.keys())[:top_k*4]:
                        try:
                            s = bf.score(tok)
                            if s and float(s) > 0:
                                score_map[tok] = score_map.get(tok, 0.0) + \
                                    weights['belief'] * float(s) * 0.3
                                chan_hits['belief'] += 1
                        except Exception:
                            continue
            except Exception:
                pass

        # --- Channel 8: importance_decay weighting ---
        if 'importance' in channels:
            try:
                imp = getattr(self, 'imp_lattice', None) or \
                      getattr(self, 'decay', None)
                if imp is not None and hasattr(imp, 'importance'):
                    for tok in list(score_map.keys())[:top_k*4]:
                        try:
                            w = imp.importance(tok)
                            if w is not None:
                                score_map[tok] = score_map.get(tok, 0.0) * \
                                    (1.0 + weights['importance'] * float(w))
                                chan_hits['importance'] += 1
                        except Exception:
                            continue
            except Exception:
                pass

        # --- Channel 9: concept_graph neighbors ---
        if 'concept_graph' in channels:
            cg = getattr(self, '_cg', None) or getattr(self, 'concept_graph_obj',
                                                         None)
            if cg is not None:
                try:
                    nbrs = None
                    for m in ('neighbors', 'get_neighbors', 'related'):
                        if hasattr(cg, m):
                            try:
                                nbrs = getattr(cg, m)(current)
                                if nbrs: break
                            except Exception:
                                continue
                    if nbrs:
                        for n in (nbrs if isinstance(nbrs, (list, tuple, set))
                                  else [nbrs]):
                            if n in candidates:
                                score_map[n] = score_map.get(n, 0.0) + \
                                    weights['concept_graph'] * 0.5
                                chan_hits['concept_graph'] += 1
                except Exception:
                    pass

        # --- Channel 10: causal_world_model next-state ---
        if 'cwm' in channels:
            try:
                cwm = getattr(self, 'cwm', None)
                if cwm is not None:
                    for m in ('predict_next', 'next_state', 'transition_from'):
                        if hasattr(cwm, m):
                            try:
                                pred = getattr(cwm, m)(current)
                                if pred and isinstance(pred, str) and \
                                        pred in candidates:
                                    score_map[pred] = score_map.get(pred, 0.0) + \
                                        weights['cwm'] * 0.6
                                    chan_hits['cwm'] += 1
                                elif isinstance(pred, dict):
                                    for tok, sc in pred.items():
                                        if tok in candidates:
                                            score_map[tok] = score_map.get(tok, 0.0) + \
                                                weights['cwm'] * float(sc)
                                            chan_hits['cwm'] += 1
                                break
                            except Exception:
                                continue
            except Exception:
                pass

        # --- Channel 11: theory_of_mind speaker bias ---
        if 'tom' in channels:
            try:
                tom = getattr(self, 'tom', None)
                if tom is not None:
                    for m in ('predict_continuation', 'speaker_bias',
                              'belief_of'):
                        if hasattr(tom, m):
                            try:
                                pred = getattr(tom, m)('default', current)
                                if pred and isinstance(pred, dict):
                                    for tok, sc in pred.items():
                                        if tok in candidates:
                                            score_map[tok] = score_map.get(tok, 0.0) + \
                                                weights['tom'] * float(sc)
                                            chan_hits['tom'] += 1
                                break
                            except Exception:
                                continue
            except Exception:
                pass

        # --- Channel 12: meta_mirror confidence filter ---
        if 'meta' in channels:
            try:
                meta = getattr(self, 'meta_mirror', None)
                if meta is not None and hasattr(meta, 'confidence'):
                    for tok in list(score_map.keys())[:top_k*4]:
                        try:
                            c = meta.confidence(tok)
                            if c:
                                score_map[tok] = score_map.get(tok, 0.0) * \
                                    (1.0 + weights['meta'] * float(c))
                                chan_hits['meta'] += 1
                        except Exception:
                            continue
            except Exception:
                pass

        # --- Channel 13: VSA calculus algebraic blend ---
        if 'vsa' in channels and prev:
            try:
                vsa = getattr(self, 'vsa', None)
                if vsa is not None:
                    import numpy as _np
                    try:
                        prev_hv = mr.ck.key(prev)
                        cur_hv = mr.ck.key(current)
                        blend = (prev_hv + cur_hv).astype(_np.complex64)
                        mag = float(_np.abs(blend).mean()) + 1e-9
                        blend = blend / mag
                        results = mr.resonator_recall(
                            blend, candidate_words=candidates,
                            n_iters=n_iters, beta=beta, top_k=top_k*2)
                        for tok, sc in results:
                            score_map[tok] = score_map.get(tok, 0.0) + \
                                weights['vsa'] * float(sc) * 0.3
                            chan_hits['vsa'] += 1
                    except Exception:
                        pass
            except Exception:
                pass

        # --- Channel 14: working memory rolling context ---
        if 'wm' in channels:
            wm = getattr(self, '_wm_sys', None)
            if wm is not None:
                try:
                    ctx_str = getattr(wm, 'context_string', None)
                    if callable(ctx_str): ctx_str = ctx_str()
                    if ctx_str:
                        ctx_toks = [t for t in str(ctx_str).lower().split()
                                       if t in candidates]
                        for t in ctx_toks[-8:]:
                            score_map[t] = score_map.get(t, 0.0) + \
                                weights['wm'] * 0.2
                            chan_hits['wm'] += 1
                except Exception:
                    pass

        # --- Channel 15: ngram cooccur recall ---
        if 'ngram_cooccur' in channels:
            try:
                r = mr.recall(current, 'cooccur')
                results = mr.resonator_recall(r, candidate_words=candidates,
                                                n_iters=n_iters, beta=beta,
                                                top_k=top_k*2)
                for tok, sc in results:
                    score_map[tok] = score_map.get(tok, 0.0) + \
                        weights['ngram_cooccur'] * float(sc) * 0.5
                    chan_hits['ngram_cooccur'] += 1
            except Exception:
                pass

        if debug:
            print(f'    organism_step({prev!r}, {current!r}) chan_hits={chan_hits}')

        if not score_map:
            return [(None, 0.0)]
        ranked = sorted(score_map.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def fsm_lift_abstract(self, only_for_words=None, n_reinforce=2, verbose=False):
        """Pack 225 sleep: abstract concrete transitions via isa parents."""
        return self.vs_fsm.lift_to_abstract(n_reinforce=n_reinforce,
                                               only_for_words=only_for_words,
                                               verbose=verbose)

    def fsm_induce_schemas(self, texts=None, max_chains=1000,
                              n_reinforce=3, verbose=False):
        """Pack 226 sleep: abductive schema induction via anti-unification.
        Replaces concrete tokens with isa parents, anti-unifies across chains,
        crystallizes schemas as substrate-level state transitions in role
        'schema_next'. This IS the grammar-learning phase.
        """
        return self.vs_fsm.induce_schemas(
            exposure_buffer=getattr(self, '_exposure_buf', None),
            texts=texts, max_chains=max_chains,
            n_reinforce=n_reinforce, verbose=verbose)

    def fsm_induce_schemas_clustered(self, texts=None, max_chains=5000,
                                         n_reinforce=3, sim_threshold=0.30,
                                         max_clusters=500, min_cluster=2,
                                         verbose=False):
        """Pack 231 v1 -- pure HV clustering (no length grouping).
        Collapses chains of different lengths if they share early-position
        tokens. Use only if all chains expected to be same length.
        """
        return self.vs_fsm.induce_schemas_clustered(
            exposure_buffer=getattr(self, '_exposure_buf', None),
            texts=texts, max_chains=max_chains,
            n_reinforce=n_reinforce, sim_threshold=sim_threshold,
            max_clusters=max_clusters, min_cluster=min_cluster,
            verbose=verbose)

    def assert_isa_balanced(self, taxonomy, total_per_category=200,
                              min_per_word=2, verbose=False):
        """Pack 232 -- mass-balanced isa assertion.

        Each category gets `total_per_category` total isa-role writes
        distributed evenly across its children. Prevents the popular-class
        bias that breaks _isa_parent recall when one category has 10-50x
        the children of another. (Wikipedia 'action' has 47 verbs, 'determiner'
        has 2, raw assert_isa(..., n=15) writes 705 vs 30 = 23x bias.)

        taxonomy: dict[parent_word, list[child_word]]
        total_per_category: target total mass per parent category
        min_per_word: minimum reps per child even if rounding gives less.
        Returns total writes performed.
        """
        n_total = 0
        for parent, children in taxonomy.items():
            chs = [w for w in set(children) if w in self.unified._cooccur_seen
                   or w in self.unified._seen]
            if not chs: continue
            n_per_word = max(min_per_word, total_per_category // len(chs))
            for w in chs:
                self.assert_isa(w, parent, n=n_per_word)
                n_total += n_per_word
            if verbose:
                print(f'    {parent}: {len(chs)} children x {n_per_word} reps '
                      f'= {len(chs)*n_per_word} total writes')
        return n_total

    def fsm_induce_unsupervised_pos(self, min_freq=3, sim_threshold=0.50,
                                          max_clusters=200, verbose=False):
        """Pack 233 -- emergent POS via co-occurrence context clustering.
        No hand-asserted isa needed. Substrate clusters its own vocabulary
        by distributional context similarity.
        """
        return self.vs_fsm.induce_unsupervised_pos(
            min_freq=min_freq, sim_threshold=sim_threshold,
            max_clusters=max_clusters, verbose=verbose)

    def fsm_induce_schemas_emergent(self, texts=None, max_chains=5000,
                                         n_reinforce=3, sim_threshold=0.40,
                                         max_clusters_per_length=50,
                                         min_cluster=2, verbose=False):
        """Pack 233 -- Pack 231 v2 abstraction via emergent POS instead of
        hand-asserted isa. Run fsm_induce_unsupervised_pos FIRST.
        """
        return self.vs_fsm.induce_schemas_emergent(
            exposure_buffer=getattr(self, '_exposure_buf', None),
            texts=texts, max_chains=max_chains,
            n_reinforce=n_reinforce, sim_threshold=sim_threshold,
            max_clusters_per_length=max_clusters_per_length,
            min_cluster=min_cluster, verbose=verbose)

    def fsm_induce_schemas_length_clustered(self, texts=None, max_chains=5000,
                                                 n_reinforce=3,
                                                 sim_threshold=0.40,
                                                 max_clusters_per_length=50,
                                                 min_cluster=2, verbose=False):
        """Pack 231 v2 -- length bucket FIRST, then HV cluster WITHIN length.
        Combines Pack 226 length structure with Pack 231 v1 structural cluster.
        Correct approach for heterogeneous corpora.
        """
        return self.vs_fsm.induce_schemas_length_clustered(
            exposure_buffer=getattr(self, '_exposure_buf', None),
            texts=texts, max_chains=max_chains,
            n_reinforce=n_reinforce, sim_threshold=sim_threshold,
            max_clusters_per_length=max_clusters_per_length,
            min_cluster=min_cluster, verbose=verbose)

    def fsm_generate_schema(self, seed, max_tokens=20, n_iters=5, beta=8.0,
                              stop_tokens=None, verbose=False):
        """Pack 226 schema-aware generation. Walks both abstract (schema_next)
        and concrete (next) attractors via Resonator decoding."""
        return self.vs_fsm.generate_via_schema(seed, max_tokens=max_tokens,
                                                  n_iters=n_iters, beta=beta,
                                                  stop_tokens=stop_tokens,
                                                  verbose=verbose)

    def fsm_iterative_refine_trigram(self, texts, n_epochs=8, predict_iters=3,
                                         delta_strength=3, hebbian_strength=1,
                                         verbose=False):
        """Pack 228 -- trigram-conditioned delta-rule refinement. Step uses
        (prev, current) joint state. Disambiguates per-state entropy."""
        chains = []
        for text in texts:
            toks = [t for t in str(text).lower().split() if t]
            if len(toks) >= 2: chains.append(toks)
        return self.vs_fsm.iterative_refine_trigram(chains,
                                                       n_epochs=n_epochs,
                                                       predict_iters=predict_iters,
                                                       delta_strength=delta_strength,
                                                       hebbian_strength=hebbian_strength,
                                                       verbose=verbose)

    def fsm_iterative_refine(self, texts, n_epochs=5, predict_iters=3,
                                delta_strength=2, hebbian_strength=1,
                                verbose=False):
        """Pack 227 -- iterative delta-rule sleep refinement.
        Replays text chains, predicts each next token via current FSM,
        applies unrelate(wrong) + relate(right) x delta_strength on misses
        and small hebbian reinforce on hits. Iterates n_epochs.
        Returns per-epoch stats with accuracy curve.
        """
        chains = []
        for text in texts:
            toks = [t for t in str(text).lower().split() if t]
            if len(toks) >= 2: chains.append(toks)
        return self.vs_fsm.iterative_refine(chains, n_epochs=n_epochs,
                                               predict_iters=predict_iters,
                                               delta_strength=delta_strength,
                                               hebbian_strength=hebbian_strength,
                                               verbose=verbose)

    def concept_arithmetic_resonator(self, plus=None, minus=None, top_k=5,
                                       n_iters=10, beta=8.0, momentum=0.5,
                                       belief_field=True):
        """Pack 224 -- Resonator-based concept arithmetic. Iteratively cleans
        up the arithmetic target via continuous Hopfield softmax-attention
        over the concept codebook. Bypasses the 1/sqrt(K) cosine ceiling
        that crippled cs.arithmetic() on rich-property concepts.
        """
        cs = getattr(self, '_concepts', None)
        if cs is None: return []
        return cs.arithmetic_resonator(plus_words=plus, minus_words=minus,
                                         top_k=top_k, n_iters=n_iters,
                                         beta=beta, momentum=momentum,
                                         belief_field=belief_field)

    # ── Pack 159: Sleep-Replay Consolidation (Kill Stack #6) ──────────────
    def enable_sleep_log(self, maxlen=10_000):
        """Start logging exposures into the sleep buffer. Idempotent."""
        from ikigai.cognition.sleep_replay import ExposureBuffer
        if not hasattr(self, '_exposure_buf') or self._exposure_buf is None:
            self._exposure_buf = ExposureBuffer(maxlen=maxlen)
        return self._exposure_buf

    def log_exposure(self, text, **meta):
        """Append an exposure to the sleep buffer (if enabled)."""
        buf = getattr(self, '_exposure_buf', None)
        if buf is not None and text:
            buf.log(text, meta=meta if meta else None)

    # ── Pack 160: Holographic Context (Kill Stack #1) ─────────────────────
    def open_context(self):
        """Open a new HolographicContext attached to this organism."""
        from ikigai.cognition.holographic_context import HolographicContext
        self._context = HolographicContext(self.unified)
        return self._context

    def context(self):
        """Current HolographicContext (if open)."""
        return getattr(self, '_context', None)

    # ── Pack 166: Bayesian HV Magnitudes (Kill Stack #10) ────────────────
    def bayesian(self):
        from ikigai.cognition.bayesian_hv import BayesianHV
        return BayesianHV(self)

    # ── Pack 167: Inverse Generation (Kill Stack #9) ─────────────────────
    def inverse(self):
        from ikigai.cognition.inverse_gen import InverseGenerator
        return InverseGenerator(self)

    # ── Pack 165: VSA-Attention (Kill Stack #2) ──────────────────────────
    def attention(self, roles=None):
        """
        Open a multi-head VSAAttention. Each role is one head. Query a
        thought HV through configured roles; substrate lookup replaces
        Q.K^T attention.
        """
        from ikigai.cognition.vsa_attention import VSAAttention
        return VSAAttention(self, roles=roles)

    # ── Pack 164: Time-As-A-Role (Kill Stack #8) ──────────────────────────
    def time_index(self):
        """
        Open a TimeRole indexer attached to this organism. Use it to write
        timed facts and query "what was X like at time T?".
        """
        from ikigai.cognition.time_role import TimeRole
        if not hasattr(self, '_time') or self._time is None:
            self._time = TimeRole(self.unified)
        return self._time

    # ── Pack 163: Federated Substrate Merging (Kill Stack #7) ────────────
    @staticmethod
    def merge(*organisms, alpha=None):
        """
        Merge N organisms into a new one via substrate superposition.
        All inputs must share (d, M, k, seed). Returns a fresh organism
        whose substrate counter banks are the (optionally weighted) sum
        of the inputs. O(M*d) work; substrate stays FIXED.
        """
        from ikigai.cognition.federated_merge import federated_merge
        return federated_merge(*organisms, alpha=alpha)

    # ── Pack 161: Logical Fixed-Point Reasoning (Kill Stack #5) ──────────
    def reasoning(self):
        """
        Open a LogicalFixedPoint engine attached to this organism.
        Add rules via lfp.add_rule(LogicalRule.transitive('isa')), seed
        facts via lfp.seed_facts([...]), then call lfp.run().
        """
        from ikigai.cognition.logical_fixed_point import (
            LogicalFixedPoint, LogicalRule
        )
        self._reasoning = LogicalFixedPoint(self)
        return self._reasoning

    def reasoning_engine(self):
        """Currently attached reasoner (if any)."""
        return getattr(self, '_reasoning', None)

    def sleep_consolidate(self, replay_factor=3, decay=None,
                          build_concepts=True, concept_words=None,
                          concept_preset=None, shuffle=True, verbose=False,
                          neuro_decay_steps=20):
        """
        Run a sleep-replay consolidation cycle. Replays the exposure buffer
        with optional amplification + cooccur-bank decay, then rebuilds
        concept HVs. Returns a stats dict.

        Pack 170: also decays neuromodulator state toward baseline if a
        NeuroModulators is attached (org.neuro). `neuro_decay_steps` is
        how many decay ticks the sleep represents.
        """
        from ikigai.cognition.sleep_replay import SleepConsolidator
        buf = getattr(self, '_exposure_buf', None)
        if buf is None:
            raise RuntimeError(
                "exposure log not enabled. Call org.enable_sleep_log() first.")
        sc = SleepConsolidator(self, buf)
        stats = sc.consolidate(replay_factor=replay_factor,
                               decay=decay,
                               build_concepts=build_concepts,
                               concept_words=concept_words,
                               concept_preset=concept_preset,
                               shuffle=shuffle,
                               verbose=verbose)
        # Pack 170+: sleep_step clears adenosine + allostatic drift
        # (full biological sleep, not just exponential decay).
        nm = getattr(self, '_neuro', None)
        if nm is not None and neuro_decay_steps:
            pre = dict(nm.level)
            nm.sleep_step(dt=int(neuro_decay_steps))
            stats['neuro_pre']  = pre
            stats['neuro_post'] = dict(nm.level)
            stats['allostatic_drift'] = nm.allostatic_drift()
        # Pack 212 -- Sleep Wire: schema induction + crystal mining + concept synth
        try:
            schemas = self.schema.induce_all()
            stats['schemas_induced'] = len(schemas) if schemas else 0
        except Exception as e:
            stats['schemas_err'] = str(e)[:80]
        try:
            mined = self.crystal.mine_schemas() if hasattr(self.crystal, 'mine_schemas') else []
            stats['crystal_schemas'] = len(mined) if mined else 0
            # unique_triples is a method (not @property) -- call it
            stats['crystal_unique_triples'] = self.crystal.unique_triples() \
                if hasattr(self.crystal, 'unique_triples') else 0
        except Exception as e:
            stats['crystal_err'] = str(e)[:80]
        # Re-run concept synth as part of sleep (if not already done)
        try:
            if not getattr(self, '_concepts', None):
                self.build_concepts(write_to_substrate=True)
                stats['concepts_built'] = True
        except Exception as e:
            stats['concepts_err'] = str(e)[:80]
        # Pack 218 -- self_modifying_refiner promote check at sleep
        try:
            n_prom = self.self_mod_refiner.promote_check()
            stats['self_mod_promotions'] = int(n_prom) if n_prom is not None else 0
        except Exception as e:
            stats['self_mod_err'] = str(e)[:80]
        # Pack 317.2 -- autonomous rule discovery during sleep: mine
        # inheritance / synonymy / inverse / TRANSITIVE rules from the
        # organism's own atoms and promote them. The organism learns
        # composition rules while resting; derive-not-store then computes
        # closures (N-hop, transitive reach) on demand instead of storing
        # them. No external lists -- pure self-discovery.
        try:
            new_rules = self.discover_rules()
            stats['rules_discovered'] = len(new_rules) if new_rules else 0
        except Exception as e:
            stats['rules_err'] = str(e)[:80]
        # Pack 229 -- unified compositional sleep cycle
        # (off by default to preserve old sleep_consolidate behavior;
        # opt in via sleep_compositional=True or call sleep_pack229 directly)
        if getattr(self, '_sleep_compositional_default', False):
            try:
                p229 = self.sleep_pack229()
                stats['pack229'] = p229
            except Exception as e:
                stats['pack229_err'] = str(e)[:80]
        return stats

    # ── Pack 234 -- FULL unified sleep with FrameField + Crystallizer +   ─
    # ── ImportanceDecay wired ────────────────────────────────────────────
    def sleep_pack234(self, texts=None, n_epochs=8, predict_iters=3,
                       delta_strength=3, hebbian_strength=1,
                       pos_min_freq=3, pos_tau=0.70,
                       schema_tau=0.60, min_cluster=2,
                       crystal_observe=True, frame_route=True,
                       importance_track=True, verbose=False):
        """Pack 234 -- full compositional sleep with shipped-module wiring.

        Phases:
          0. Per-text: frame_field route_passage + crystallizer SVO observe
             + importance_decay record. Three dormant modules fire in parallel
             with the awake recording.
          1. Pack 233: emergent POS via unsupervised context clustering.
          2. Pack 231 v2: schema induction via length+HV cluster using
             emergent POS (NO hand-asserted isa).
          3. Pack 228: trigram delta-rule refine, with hebbian/delta strengths
             optionally weighted by importance_decay.score per chain.

        Returns combined stats dict including frame counts, crystal triples,
        importance top-K, schema set, refine accuracy curve.
        """
        if texts is None:
            buf = getattr(self, '_exposure_buf', None)
            if buf is not None:
                texts = [t for t, _, _ in buf.snapshot()]
        if not texts:
            return {'error': 'no texts'}

        out = {}

        # Phase 0: dormant-module observations
        if frame_route:
            try:
                ff = self.frames        # lazy property
                n_routed = 0
                for text in texts:
                    toks = [t for t in str(text).lower().split() if t]
                    if toks:
                        ff.route_passage(toks, self.unified.ck,
                                          observe=True, learn=True)
                        n_routed += 1
                out['frames_routed'] = n_routed
                out['frame_assigns'] = list(ff.assigns) \
                    if hasattr(ff, 'assigns') else None
            except Exception as e:
                out['frame_err'] = str(e)[:80]

        if crystal_observe:
            try:
                cr = self.crystal       # lazy property
                n_obs = 0
                # Mine simple (subj, verb, obj) triples from each text.
                for text in texts:
                    toks = [t for t in str(text).lower().split() if t]
                    # Heuristic SVO: positions 1,2,3 of [det,subj,verb,obj]
                    if len(toks) >= 3:
                        # Take 3 content tokens skipping det.
                        content = [t for t in toks
                                    if t not in ('the','a','an','this','that')]
                        if len(content) >= 3:
                            cr.observe(content[0], content[1], content[2])
                            n_obs += 1
                out['crystal_observed'] = n_obs
                out['crystal_unique'] = cr.unique_triples() \
                    if hasattr(cr, 'unique_triples') else 0
            except Exception as e:
                out['crystal_err'] = str(e)[:80]

        if importance_track:
            try:
                idec = self.importance_decay        # lazy property
                for ti, text in enumerate(texts):
                    toks = [t for t in str(text).lower().split() if t]
                    if toks:
                        idec.record(name=f'chain_{ti}', tokens=toks,
                                      surprise=0.0)
                out['importance_records'] = len(texts)
            except Exception as e:
                out['importance_err'] = str(e)[:80]

        # Phase 1 -- emergent POS (Pack 233)
        try:
            pos_stats = self.fsm_induce_unsupervised_pos(
                min_freq=pos_min_freq, sim_threshold=pos_tau,
                max_clusters=200, verbose=False)
            out['emergent_pos'] = pos_stats
        except Exception as e:
            out['pos_err'] = str(e)[:80]

        # Phase 2 -- schemas via emergent POS + length+HV cluster
        try:
            ind = self.fsm_induce_schemas_emergent(
                texts=texts, n_reinforce=3,
                sim_threshold=schema_tau, min_cluster=min_cluster,
                verbose=False)
            out['schemas'] = ind
        except Exception as e:
            out['schemas_err'] = str(e)[:80]

        # Phase 3 -- trigram delta refine
        try:
            stats = self.fsm_iterative_refine_trigram(
                texts, n_epochs=n_epochs,
                predict_iters=predict_iters,
                delta_strength=delta_strength,
                hebbian_strength=hebbian_strength,
                verbose=verbose)
            out['refine_epochs'] = stats
            out['refine_peak_acc'] = max((s['accuracy'] for s in stats),
                                         default=0.0)
            out['refine_final_acc'] = stats[-1]['accuracy'] if stats else 0.0
        except Exception as e:
            out['refine_err'] = str(e)[:80]

        return out

    # ── Pack 229 -- Unified Compositional Sleep Cycle ────────────────────
    def sleep_pack229(self, texts=None, n_epochs=8, predict_iters=3,
                       delta_strength=3, hebbian_strength=1,
                       lift_n_reinforce=2, schema_n_reinforce=3,
                       verbose=False):
        """Pack 229 -- one sleep call runs the full compositional refinement:
          1. fsm_lift_abstract (Pack 225) -- isa-parent transitions written.
          2. fsm_induce_schemas (Pack 226) -- anti-unification of chains,
             schema_next role populated.
          3. fsm_iterative_refine_trigram (Pack 228) -- delta-rule loop
             over chains until prediction error converges.

        texts: optional explicit corpus. If None, pulls from ExposureBuffer.
        Returns combined stats dict.
        """
        out = {}
        # Source chains
        if texts is None:
            buf = getattr(self, '_exposure_buf', None)
            if buf is not None:
                texts = [t for t, _, _ in buf.snapshot()]
        if not texts:
            return {'error': 'no chains -- enable_sleep_log + log_exposure first'}
        # Step 1 -- lift to abstract via isa parents
        try:
            n_lifted = self.fsm_lift_abstract(n_reinforce=lift_n_reinforce,
                                                 verbose=verbose)
            out['lifted_transitions'] = int(n_lifted) if n_lifted else 0
        except Exception as e:
            out['lift_err'] = str(e)[:80]
        # Step 2 -- abductive schema induction
        try:
            ind = self.fsm_induce_schemas(texts=texts,
                                              n_reinforce=schema_n_reinforce,
                                              verbose=verbose)
            out['schemas'] = ind
        except Exception as e:
            out['schemas_err'] = str(e)[:80]
        # Step 3 -- delta-rule trigram refinement loop
        try:
            stats = self.fsm_iterative_refine_trigram(
                texts, n_epochs=n_epochs,
                predict_iters=predict_iters,
                delta_strength=delta_strength,
                hebbian_strength=hebbian_strength,
                verbose=verbose)
            out['refine_epochs'] = stats
            out['refine_peak_acc'] = max((s['accuracy'] for s in stats),
                                         default=0.0)
            out['refine_final_acc'] = stats[-1]['accuracy'] if stats else 0.0
        except Exception as e:
            out['refine_err'] = str(e)[:80]
        return out

    # ── Pack 170: Neuromodulator Generative Binding ──────────────────────
    @property
    def neuro(self):
        """Lazy-built NeuroModulators tensor (DA/EPI/CORT/5HT/ACh)."""
        nm = getattr(self, '_neuro', None)
        if nm is None:
            from ikigai.cognition.neuromod import NeuroModulators
            nm = NeuroModulators()
            self._neuro = nm
        return nm

    def neuro_spike(self, chem, delta, reason='manual'):
        """Manually spike a chemical level."""
        self.neuro.spike(chem, delta, reason=reason)
        return self.neuro.state()

    def neuro_expose(self, text):
        """Scan text for lexicon-flagged emotional tokens; spike chemicals."""
        return self.neuro.expose_tokens(text)

    def neuro_state(self):
        """Snapshot of current chemical levels + derived signals."""
        return self.neuro.state()

    # ── Pack 178+: persistence + chat API ────────────────────────────────
    def absorb(self, hf_model, path, d=2048, K=2, mode='fast', verbose=True):
        """Compile a HF transformer model into substrate and save to disk.
        mode='fast': raw weights stored (BLAS forward + linear attention).
        mode='vsa' : JL ensemble HVs stored (substrate-native, slower).
        Future runs: org.load_substrate(path) -- no HF/torch needed."""
        import time as _t
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        inner = hf_model.transformer if isinstance(hf_model, GPT2LMHeadModel) else hf_model
        t2s = self.t2s(d=d, K=K)
        t0 = _t.perf_counter()
        if mode == 'fast':
            t2s.compile_gpt2_model_fast(inner, verbose=verbose)
        else:
            t2s.compile_gpt2_model(inner, verbose=verbose)
        if verbose:
            print(f'  compile {_t.perf_counter()-t0:.1f}s', flush=True)
        t0 = _t.perf_counter()
        t2s.save(path)
        if verbose:
            print(f'  saved to {path} in {_t.perf_counter()-t0:.1f}s', flush=True)
        self._absorbed_path = path
        return t2s

    def load_substrate(self, path, d=2048, K=2):
        """Load a previously absorbed model. No HF/torch required."""
        t2s = self.t2s(d=d, K=K)
        t2s.load(path)
        self._absorbed_path = path
        return t2s

    def prompt(self, text, max_new=10, tokenizer=None, greedy=True,
                 linear=True, verbose=False):
        """Generate continuation from absorbed substrate.
        linear=True (default): O(1) per-token linear attention forward.
        linear=False: O(N²) softmax forward (exact GPT-2 reproduction)."""
        if tokenizer is None:
            from transformers import GPT2Tokenizer
            if not hasattr(self, '_tok') or self._tok is None:
                self._tok = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer = self._tok
        t2s = self.t2s()
        ids = tokenizer.encode(text)
        if linear:
            ids = t2s.gpt2_generate_linear(ids, max_new=int(max_new),
                                            greedy=greedy, verbose=verbose)
        else:
            for _ in range(int(max_new)):
                logits = t2s.gpt2_forward(ids)
                nxt = int(logits[-1].argmax())
                ids.append(nxt)
                if verbose:
                    print(f'  +{tokenizer.decode([nxt])!r}', flush=True)
        return tokenizer.decode(ids)

    # ── Pack 195p: .ikg auto-load / auto-save (kill stack #10) ──────────
    DEFAULT_IKG_PATH = 'c:/neuroseed/organism.ikg'

    def load_ikg(self, path=None):
        """Replace self.unified with substrate loaded from .ikg file.
        Path defaults to env IKIGAI_IKG or c:/neuroseed/organism.ikg.
        """
        import os
        from ikigai.cognition.multirole_memory import MultiRoleMemory
        from ikigai.cognition.frame_field import FrameField
        if path is None:
            path = os.environ.get('IKIGAI_IKG', self.DEFAULT_IKG_PATH)
        if not os.path.exists(path):
            return None
        self.unified = MultiRoleMemory.load_ikg(path)
        # Pack 197: rehydrate frame field if present in .ikg
        pfs = getattr(self.unified, '_pending_frame_state', None)
        if pfs:
            try:
                self.frames = FrameField.from_dict(pfs)
            except Exception:
                # legacy / malformed -- fall back to fresh frames
                self.frames = FrameField(d=self.unified.d, K=8, top_n=64,
                                           seed=42, alpha=0.5)
        else:
            self.frames = FrameField(d=self.unified.d, K=8, top_n=64,
                                       seed=42, alpha=0.5)
        self.unified._frame_field_ref = self.frames
        # Pack 220: restore wired-module state
        p220 = getattr(self.unified, '_pending_pack220_state', None)
        n_restored = 0
        if p220:
            try:
                self._apply_wired_state(p220)
                n_restored = len(p220)
            except Exception:
                pass
            # Sync org-side mirrors of module internal logs
            try:
                if hasattr(self.fe, 'F_log'):
                    self._fe_log = list(self.fe.F_log)
            except Exception:
                pass
        # Pack 246b (Day 70): auto-instantiate bridge instances (Pack 217
        # classes) so the organism's WM / ConceptGraph / EventCompressor /
        # CellAssembly are available after load_ikg without needing extra
        # wire-up calls. Eager construction.
        try:
            br = self.bridge
            if br is not None:
                if self.__dict__.get('_wm_sys') is None:
                    try:
                        self._wm_sys = br.cls(
                            'WorkingMemorySystem')(slots=8, decay=10)
                    except Exception: pass
                if self.__dict__.get('_cg') is None:
                    try:
                        self._cg = br.cls(
                            'ConceptGraph')(max_nodes=256,
                                               similarity_threshold=0.85)
                    except Exception: pass
                if self.__dict__.get('_ec') is None:
                    try:
                        self._ec = br.cls(
                            'EventCompressor')(maxlen=500, min_event_len=3)
                    except Exception: pass
                if self.__dict__.get('_cas') is None:
                    try:
                        self._cas = br.cls('CellAssemblySystem')()
                    except Exception: pass
        except Exception:
            pass
        self._ikg_path = path
        return {'path': path, 'd': self.unified.d,
                 'sdm_M': self.unified.sdm.M,
                 'sdm_rel_M': self.unified.sdm_rel.M,
                 'cooccur_vocab': len(self.unified._cooccur_seen),
                 'seen': len(self.unified._seen),
                 'frame_locked': bool(self.frames.locked),
                 'frame_assigns': self.frames.assigns_per_frame.tolist(),
                 'pack220_modules_restored': n_restored}

    # Pack 220 -- list of wired modules whose state persists in .ikg
    _PERSIST_ATTRS = (
        'fe', 'curiosity', 'tom', 're', 'vsa',
        'belief', 'verifier', 'proof_gen',
        'schema', 'crystal',
        'persona_proj', 'meta_mirror', 'imp_lattice',
        'cf', 'cwm',
        'schema_refiner', 'self_mod_refiner',
        'goals', 'world', 'moe', 'dssc',
        # Pack 247d -- hand-curated taxonomy survives roundtrip
        '_taxonomy_seed', '_taxonomy_word_to_class',
        # Pack 247g Phase 4-B -- read common-mode + HSP pooler
        '_predict_v_common', 'hsp',
        # Pack 249 (Day 72) -- UDSP spectral projection cache
        'udsp',
        # Pack 252 (Day 73) -- FPE numeric encoder phases vector
        '_num_enc',
        # Pack 273 (Day 76) -- cat-4 anchor-action cache (dict).
        # Persisted as a plain attr on the organism; cat4 lazy
        # property reads from it on first access.
        '_cat4_anchor_actions_cache',
        # Pack 305 (Day 79) -- compositional atom index + learned
        # derivation rules (CompositionEngine.save_state writes this).
        '_comp_state',
        # Pack 300.1 (Day 79) -- learned sentence templates (language
        # teach).  Lives off b_self.
        '_lang_templates',
    )

    # Pack 246b (Day 70) -- modules with back-references to organism. Cannot
    # be pickled whole. Custom state-dict extraction.
    _PERSIST_VS_FSM_KEYS = (
        'schemas', '_emergent_pos', '_emergent_pos_clusters',
        '_isa_inverse', 'transition_count', 'schema_transitions',
        'abstracted_count', 'skipped',
    )

    def _gather_wired_state(self):
        """Pack 220 -- serialize state of every wired cognition module via
        pickle. Returns dict of {attr_name: pickled_bytes}. The cognition
        modules don't have native to_dict; pickling __dict__ captures
        everything reliably. Bridge classes (wm_sys, concept_graph etc.)
        are NOT persisted -- bridge reloads ikigai.py from source fresh.

        Pack 246b (Day 70): handle vs_fsm specially (back-ref to organism
        prevents direct pickle -- extract key state dict instead)."""
        import pickle as _pkl
        out = {}
        for attr in self._PERSIST_ATTRS:
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                out[attr] = _pkl.dumps(obj, protocol=_pkl.HIGHEST_PROTOCOL)
            except Exception as e:
                # Some modules may hold un-pickleable handles (lambda, etc).
                # Skip silently; on reload that module returns to fresh state.
                out[attr] = _pkl.dumps({'_pack220_err': str(e)[:120]})
        # Pack 246b: vs_fsm needs custom state dict (back-references to org
        # would pickle the whole organism recursively).
        vs_fsm = getattr(self, '_vs_fsm', None)
        if vs_fsm is not None:
            vs_state = {}
            for k in self._PERSIST_VS_FSM_KEYS:
                try:
                    vs_state[k] = getattr(vs_fsm, k, None)
                except Exception:
                    pass
            try:
                out['_vs_fsm_state'] = _pkl.dumps(vs_state,
                                                    protocol=_pkl.HIGHEST_PROTOCOL)
            except Exception:
                pass
        return out

    def _apply_wired_state(self, state_dict):
        """Pack 220 -- restore state from .ikg-loaded dict.

        Pack 246b: handle vs_fsm specially (reconstruct via property + attach
        saved state dict)."""
        import pickle as _pkl
        for attr, blob in state_dict.items():
            if not blob:
                continue
            if attr == '_vs_fsm_state':
                # Restore vs_fsm state dict onto the lazy-initialized vs_fsm.
                try:
                    vs_state = _pkl.loads(blob)
                except Exception:
                    continue
                # Force lazy init so vs_fsm exists, then assign state attrs.
                try:
                    fsm = self.vs_fsm
                    for k, v in vs_state.items():
                        if v is None: continue
                        try:
                            setattr(fsm, k, v)
                        except Exception:
                            pass
                except Exception:
                    pass
                continue
            try:
                obj = _pkl.loads(blob)
                if isinstance(obj, dict) and '_pack220_err' in obj:
                    continue
                setattr(self, attr, obj)
            except Exception:
                pass
        # Pack 249: re-bind UDSP org back-ref after load
        try:
            self._rebind_udsp()
        except Exception:
            pass

    def save_ikg(self, path=None):
        """Save current substrate + frames + Pack 220 wired-module state to
        .ikg (kill stack #10 format)."""
        if path is None:
            path = getattr(self, '_ikg_path', None) or self.DEFAULT_IKG_PATH
        # Pack 197: ensure frame ref is set so save_ikg pulls frame state
        self.unified._frame_field_ref = self.frames
        # Pack 273: sync cat4 anchor-action cache up to persisted attr
        try:
            self._sync_cat4_cache_for_persist()
        except Exception:
            pass
        # Pack 220: pickle every wired module's state into a side blob
        try:
            self.unified._pack220_wired_state = self._gather_wired_state()
        except Exception:
            self.unified._pack220_wired_state = {}
        info = self.unified.save_ikg(path)
        self._ikg_path = path
        return info

    def reset_organism(self):
        """Pack 197 -- wipe substrate AND frame field. .ikg file kept.
        Use to fully start over with the universal organism.ikg."""
        from ikigai.cognition.frame_field import FrameField
        self.unified.reset_substrate()
        self.frames = FrameField(d=self.unified.d, K=8, top_n=64,
                                  seed=42, alpha=0.5)
        self.unified._frame_field_ref = self.frames

    def route_frame(self, tokens, observe=True, learn=True):
        """Pack 197 -- assign current frame from tokens. Sets the active frame
        on the substrate. Returns (frame_idx, score).
        """
        idx, fhv, score = self.frames.route_passage(tokens, self.unified.ck,
                                                       observe=observe,
                                                       learn=learn)
        if fhv is not None:
            self.unified.set_frame(fhv, frame_tag=f'f{idx}')
        return idx, score

    def clear_frame(self):
        self.unified.clear_frame()

    # ── Pack 247g Phase 4-B (Day 71) -- read-time common-mode cache ────────
    def build_predict_common_mode(self, vocab=None, role=None, force=False):
        """Compute v_common = mean(mr.recall(w, role) for w in vocab).
        Cached on self._predict_v_common. Cheap to compute (<1s for 3K vocab).
        Use force=True to rebuild after substrate updates."""
        import numpy as _np
        if not force and getattr(self, '_predict_v_common', None) is not None:
            return self._predict_v_common
        mr = self.unified
        if role is None:
            role = self.vs_fsm.NEXT_ROLE
        if vocab is None:
            vocab = list(mr._cooccur_seen)
        v = _np.zeros(mr.d, dtype=_np.complex64)
        n = 0
        for w in vocab:
            try:
                v = v + mr.recall(w, role)
                n += 1
            except Exception:
                continue
        self._predict_v_common = (v / max(n, 1)).astype(_np.complex64)
        return self._predict_v_common

    # ── Pack 249 (Day 72) -- UDSP attach + auto-rebind on load ───────────
    def attach_udsp(self, udsp_compiler=None, d=None, seed=24001,
                      scrambler=True, signed_im=True, device='cpu'):
        """Permanently attach a UDSPCompiler. State persists via
        _PERSIST_ATTRS. After load_ikg, the .org back-ref needs re-binding
        because __getstate__ drops it.

        device defaults to 'cpu' (Pack 249 SAFE default). Set to 'cuda'
        only when GPU pipeline is stable for big-matrix SVD."""
        from ikigai.cognition.udsp_compiler import UDSPCompiler
        if udsp_compiler is None:
            udsp_compiler = UDSPCompiler(self, d=d, seed=seed,
                                            scrambler=scrambler,
                                            signed_im=signed_im,
                                            device=device)
        udsp_compiler.org = self
        self.udsp = udsp_compiler
        return udsp_compiler

    def _rebind_udsp(self):
        """Post-load: re-bind org back-ref on UDSP (was None'd at pickle)."""
        if getattr(self, 'udsp', None) is not None:
            self.udsp.org = self

    # ── Pack 247g Phase 1c-4B -- HSP permanent attach ────────────────────
    def attach_hsp(self, hsp_pooler):
        """Permanently attach an HSPColumnPooler to this organism.
        State persists via _PERSIST_ATTRS in save_ikg."""
        self.hsp = hsp_pooler

    def build_hsp(self, M=512, k_active=10, lr=0.08, seed=42,
                    boost_strength=2.5, common_mode=True,
                    n_epochs=5, role=None):
        """Construct + fit_vocab + return a HSP. Uses behavioral
        NEXT-only input (Phase 3 doctrine)."""
        import numpy as _np
        from ikigai.cognition.hsp import HSPColumnPooler
        if role is None:
            role = self.vs_fsm.NEXT_ROLE
        mr = self.unified
        vocab = sorted(set(mr._cooccur_seen))

        class _NextOnlyKey:
            def __init__(self, _mr, _role):
                self.mr = _mr; self.role = _role; self._c = {}
            def key(self, word):
                v = self._c.get(word)
                if v is not None: return v
                try: nxt = self.mr.recall(word, self.role)
                except Exception: nxt = self.mr.ck.key(word)
                m = float(_np.abs(nxt).mean()) + 1e-9
                v = (nxt / m).astype(_np.complex64)
                self._c[word] = v
                return v

        bk = _NextOnlyKey(mr, role)
        for w in vocab: _ = bk.key(w)
        hsp = HSPColumnPooler(d=mr.d, M=M, k_active=k_active, lr=lr,
                                seed=seed, boost_strength=boost_strength,
                                common_mode=common_mode)
        hsp.fit_vocab(vocab, bk, n_epochs=n_epochs)
        self.hsp = hsp
        return hsp

    def fit_hsp_transitions(self, token_pairs, normalize='ppmi', alpha=0.75):
        """PPMI-normalized transition matrix on attached HSP."""
        if not hasattr(self, 'hsp') or self.hsp is None:
            raise RuntimeError('No HSP attached. Call build_hsp() first.')
        return self.hsp.fit_transitions(token_pairs, normalize=normalize,
                                            alpha=alpha)

    # ── Pack 247d/e/f -- ORGAN WIRES (taxonomy + frame, permanent) ─────────
    def set_taxonomy_seed(self, seed_dict, n_reinforce=3, write_substrate=True):
        """Pack 247d -- install hand-curated taxonomy.

        seed_dict: {CLASS_NAME: [member_word, ...]} mapping.
        Builds clean _isa_inverse on vs_fsm DIRECTLY from seed, bypassing
        substrate query (which is polluted by emergent_pos clusters). Also
        writes ISA + isa_inverse substrate relations for downstream consumers
        unless write_substrate=False.

        Returns dict with stats.
        """
        import numpy as _np
        mr = self.unified
        fsm = self.vs_fsm
        vocab_set = set(mr._cooccur_seen)

        # Register isa_inverse role if missing
        if 'isa_inverse' not in mr.roles:
            rng = _np.random.default_rng(20200 + abs(hash('isa_inverse')) % 1000)
            ph = rng.uniform(-_np.pi, _np.pi, mr.d).astype(_np.float32)
            mr.roles['isa_inverse'] = _np.exp(1j * ph).astype(_np.complex64)

        isa_inv_clean = {}
        word_to_class = {}
        n_writes = 0
        for cls_name, members in seed_dict.items():
            in_vocab = [m for m in members if m in vocab_set]
            if not in_vocab:
                continue
            isa_inv_clean[cls_name] = set(in_vocab)
            for w in in_vocab:
                word_to_class.setdefault(w, []).append(cls_name)
                if write_substrate:
                    for _ in range(n_reinforce):
                        mr.relate(w, 'isa', cls_name)
                        mr.relate(cls_name, 'isa_inverse', w)
                    n_writes += n_reinforce * 2

        # Install clean cache (bypasses _build_isa_inverse substrate query)
        fsm._isa_inverse = isa_inv_clean
        self._taxonomy_seed = dict(seed_dict)
        self._taxonomy_word_to_class = word_to_class
        return {
            'classes': len(isa_inv_clean),
            'coverage': len(word_to_class),
            'vocab': len(vocab_set),
            'substrate_writes': n_writes,
        }

    def predict_with_wires(self, prev, current, candidates=None, top_k=5,
                              do_taxonomy=True, do_frame_filter=True,
                              do_crystal=False, do_holo=False, do_hsp=False,
                              tax_boost_mult=1.5, tax_boost_add=0.05,
                              frame_penalty_mult=0.5, frame_in_boost=0.0,
                              frame_score_min=0.15, protect_taxonomy=True,
                              crystal_boost_per_count=0.02, crystal_max_boost=0.20,
                              holo_blend=0.20, holo_ctx=None,
                              hsp_pooler=None, hsp_jaccard_threshold=0.10,
                              hsp_boost_mult=1.3, hsp_boost_add=0.03,
                              hsp_mode='neighbor',
                              read_common_mode=None,
                              sharpen_beta=1.0, sharpen_softmax_lambda=0.0,
                              wire_confidence_gate=0.0,
                              frame_idx=None, frame_tokens=None):
        """Pack 247d/e/f -- substrate cosine + taxonomy + frame filter.

        - do_taxonomy: boost candidates that share an isa-class with current.
          Uses clean _isa_inverse cache (Pack 247d).
        - do_frame_filter: penalty for candidates not in current frame's
          vocab. Frame from frame_idx or routed from frame_tokens.
        - frame_score_min (Pack 247f Fix B): skip frame penalty when route
          confidence below this. Default 0.15 (mean route score ~0.16).
        - protect_taxonomy (Pack 247f Fix A): skip frame penalty for
          candidates that received taxonomy boost. Prevents frame from
          erasing taxonomy signal.

        Returns list of (token, score) of length up to top_k.
        """
        import numpy as _np
        mr = self.unified
        fsm = self.vs_fsm

        if candidates is None:
            candidates = list(mr._role_targets.get(fsm.NEXT_ROLE, set()))
        if not candidates:
            return []

        # Cache cand HVs per-call (small price, simpler than invalidation)
        cand_hvs = _np.stack([mr.ck.key(c) for c in candidates]).astype(_np.complex64)
        cand_norms = _np.abs(cand_hvs).mean(axis=1) + 1e-9
        cand_hvs_norm = cand_hvs / cand_norms[:, None]

        try:
            r_next = mr.recall(current, fsm.NEXT_ROLE)
            if prev:
                try:
                    r_skip = mr.recall(prev, fsm.PREV_ROLE)
                    r_next = (r_next + r_skip).astype(_np.complex64)
                except Exception:
                    pass
            # WIRE: Read-time common-mode subtraction (Pack 247g Phase 4-B).
            # Subtract mean(recall(w, NEXT) for w in vocab) BEFORE normalizing.
            # Linearity of binding preserves substrate algebra. Sharpens UNSEEN
            # by projecting query orthogonal to static background.
            # Auto-use cached v_common if not explicitly passed.
            cm = read_common_mode
            if cm is None:
                cm = getattr(self, '_predict_v_common', None)
            if cm is not None:
                r_next = (r_next - cm).astype(_np.complex64)
            mag = float(_np.abs(r_next).mean()) + 1e-9
            r_next = r_next / mag
        except Exception:
            return []

        cos_scores = _np.real(cand_hvs_norm.conj() @ r_next).astype(_np.float32) / mr.d
        # Optional polynomial sharpening on similarity vector (Phase 4-E).
        if sharpen_beta != 1.0:
            cos_scores = _np.sign(cos_scores) * _np.power(
                _np.abs(cos_scores), sharpen_beta).astype(_np.float32)
        # Optional softmax sharpening (Phase 4-E research recommendation).
        if sharpen_softmax_lambda > 0:
            sc = cos_scores - cos_scores.max()
            ex = _np.exp(sharpen_softmax_lambda * sc).astype(_np.float32)
            cos_scores = (ex / (ex.sum() + 1e-9)).astype(_np.float32)
        # Pack 247g Phase 4-D: substrate-confidence gate. At scale, substrate
        # alone often outperforms wired versions because wires reorder a
        # correct top-1 toward a wire-preferred wrong candidate. Skip ALL
        # wires when substrate is confident (top1 - top2 >= gate).
        cand_to_idx = {c: i for i, c in enumerate(candidates)}
        skip_wires = False
        if wire_confidence_gate > 0 and len(cos_scores) >= 2:
            top2_vals = _np.partition(cos_scores, -2)[-2:]
            margin = float(top2_vals.max() - top2_vals.min())
            if margin >= wire_confidence_gate:
                skip_wires = True

        # WIRE: taxonomy boost (Pack 247d clean cache lookup)
        boosted_indices = set()
        if do_taxonomy and not skip_wires:
            isa_inv = getattr(fsm, '_isa_inverse', None) or {}
            w2c = getattr(self, '_taxonomy_word_to_class', None) or {}
            if isa_inv and w2c:
                parents = w2c.get(current, ())
                for p in parents:
                    for c in isa_inv.get(p, ()):
                        if c in cand_to_idx and c not in boosted_indices:
                            i = cand_to_idx[c]
                            cos_scores[i] = cos_scores[i] * tax_boost_mult + tax_boost_add
                            boosted_indices.add(i)

        # WIRE: crystal triple prior (Pack 247h). Boost candidates that crystal
        # has observed in (?, current, c) shape. Walks _counts filtered by
        # predicate == current. Sparse but exact evidence.
        if do_crystal and not skip_wires:
            crystal = getattr(self, 'crystal', None)
            if crystal is not None and crystal._counts:
                pred_str = str(current)
                for (s, p, o), cnt in crystal._counts.items():
                    if p != pred_str:
                        continue
                    if o in cand_to_idx:
                        i = cand_to_idx[o]
                        boost = min(crystal_boost_per_count * cnt, crystal_max_boost)
                        cos_scores[i] = cos_scores[i] + boost

        # WIRE: HSP topology boost.
        # mode='neighbor' (Phase 2): boost candidates with SDR overlap to current.
        # mode='successor' (Phase 4-A/B): boost candidates with SDR overlap to
        #   EXPECTED next SDR (PPMI-normalized transition matrix P).
        # Auto-use attached self.hsp if pooler not explicitly passed.
        if do_hsp and hsp_pooler is None:
            hsp_pooler = getattr(self, 'hsp', None)
        if do_hsp and hsp_pooler is not None and not skip_wires:
            if hsp_mode == 'successor':
                target_sdr, _ = hsp_pooler.expected_next_sdr(current)
            else:
                target_sdr = hsp_pooler.word_sdr.get(current, frozenset())
            if target_sdr:
                cand_overlap = {}
                for col in target_sdr:
                    if col >= len(hsp_pooler.col_words): continue
                    for w in hsp_pooler.col_words[col]:
                        if w == current and hsp_mode != 'successor': continue
                        if w in cand_to_idx:
                            cand_overlap[w] = cand_overlap.get(w, 0) + 1
                for w, n_shared in cand_overlap.items():
                    sdr_w = hsp_pooler.word_sdr.get(w)
                    if not sdr_w: continue
                    union = len(target_sdr) + len(sdr_w) - n_shared
                    jacc = n_shared / union if union else 0.0
                    if jacc < hsp_jaccard_threshold: continue
                    i = cand_to_idx[w]
                    if protect_taxonomy and i in boosted_indices:
                        continue
                    cos_scores[i] = (cos_scores[i] * hsp_boost_mult
                                       + hsp_boost_add * jacc)

        # WIRE: holographic context (Pack 247h). If caller passes a live
        # HolographicContext, blend its predicted-next-slot cosine with
        # substrate cosine. holo_ctx already has prev/current appended by
        # caller. Query position+1 directly via inverse permutation.
        if do_holo and holo_ctx is not None and holo_ctx.position > 0 and not skip_wires:
            import numpy as _np2
            # Slot for the NEXT position (one beyond current append head)
            next_pos = holo_ctx.position
            slot = _np2.roll(holo_ctx.ctx, -next_pos).astype(_np2.complex64)
            slot_mag = float(_np2.abs(slot).mean()) + 1e-9
            slot = slot / slot_mag
            holo_scores = _np2.real(cand_hvs_norm.conj() @ slot).astype(
                _np2.float32) / mr.d
            cos_scores = ((1.0 - holo_blend) * cos_scores
                            + holo_blend * holo_scores).astype(_np2.float32)

        # WIRE: frame filter (Pack 247e + 247f confidence gate + protect taxonomy)
        if do_frame_filter and hasattr(self, 'frames') and not skip_wires:
            ff = self.frames
            fi = frame_idx
            route_score = None
            if fi is None and frame_tokens:
                try:
                    fi, _fhv, route_score = ff.route_prompt(list(frame_tokens), mr.ck)
                except Exception:
                    fi = None
            # Confidence gate: skip frame if route too uncertain
            apply_frame = (fi is not None and 0 <= fi < ff.K
                             and (route_score is None or route_score >= frame_score_min))
            if apply_frame:
                in_frame = ff.frame_vocab[fi]
                if in_frame:
                    for i, c in enumerate(candidates):
                        if protect_taxonomy and i in boosted_indices:
                            continue
                        if c in in_frame:
                            cos_scores[i] = cos_scores[i] + frame_in_boost
                        else:
                            cos_scores[i] = cos_scores[i] * frame_penalty_mult

        order = _np.argsort(-cos_scores)[:top_k]
        return [(candidates[int(i)], float(cos_scores[int(i)])) for i in order]

    # ── Pack 192 v1 (Day 64): DEEP absorb -- child learns from conversation ──
    def _diverse_seed_prompts(self, hf_model, tokenizer, n, device='cpu'):
        """Generate diverse seed prompts FROM THE LLM ITSELF (no hardcode).
        Strategy: pick high-entropy single tokens by embed-row variance as
        topic seeds. Each seed is a single token that the LLM will complete
        into a coherent paragraph during Phase B.
        """
        import torch
        # Universal accessor: get_input_embeddings() works across HF families
        # (Qwen2 .model.embed_tokens, Qwen3.5 hybrid, GPT-2, Llama, etc).
        emb_layer = hf_model.get_input_embeddings()
        # .float() forces fp32 in torch before numpy (bf16/fp16 unsupported by numpy)
        embed = emb_layer.weight.detach().float().to('cpu').numpy().astype(np.float32)
        # variance per row = information density of that token's embedding
        diversity = embed.std(axis=1)
        k = int(min(n, embed.shape[0]))
        top = np.argpartition(-diversity, k - 1)[:k]
        # filter to printable strings
        seeds = []
        for tid in top:
            try:
                s = tokenizer.decode([int(tid)]).strip()
            except Exception:
                continue
            if s and len(s) < 32:
                seeds.append(s)
            if len(seeds) >= n:
                break
        return seeds

    # ── Pack 192 v1.1: POS classifier + junk-text filter ────────────────────
    @staticmethod
    def _is_junk_token(tok):
        """Reject code-y / non-linguistic tokens before substrate write.
        Heuristics (not English-hardcoded): contains digits, too long, low
        alpha ratio, camel-case markers."""
        if not tok or len(tok) > 14:
            return True
        if any(ch.isdigit() for ch in tok):
            return True
        n_alpha = sum(1 for ch in tok if ch.isalpha())
        if n_alpha == 0 or n_alpha / len(tok) < 0.7:
            return True
        # camelCase / PascalCase mid-word uppercase (e.g. 'sendStatus', 'lastFilesWidget')
        if any(c.isupper() for c in tok[1:]):
            return True
        return False

    @classmethod
    def _filter_junk_text(cls, text):
        """Drop junk tokens from text. Preserves order, lowercases output."""
        import re as _re
        toks = _re.findall(r"[A-Za-z']+", text)
        kept = [t.lower() for t in toks if not cls._is_junk_token(t)]
        return ' '.join(kept)

    def _build_emergent_clusterer(self, hf_model, tokenizer, n_clusters=64,
                                    sample_tokens=20000):
        """Pack 192 v1.2 -- EMERGENT category clusterer. NO hardcoded labels.

        Clusters LLM embeddings via mini-batch k-means. Each token gets a
        cluster id (c0..c{K-1}). The categories EMERGE from distributional
        geometry; we never tell the model what a 'noun' or 'verb' is.

        FIX (v1.2.1): sample alpha-only ASCII tokens (no code, no foreign
        scripts, no numerics), mean-center before clustering (kills the
        frequent-word collapse axis), K=64 for finer granularity.

        Returns a callable: cluster_of(tokens) -> {token: 'c{id}' or '?'}
        """
        import numpy as _np
        from sklearn.cluster import MiniBatchKMeans

        emb_layer = hf_model.get_input_embeddings()
        embed = emb_layer.weight.detach().float().to('cpu').numpy()
        V, D = embed.shape

        # Build alpha-only token mask via tokenizer decode
        clean_ids = []
        for tid in range(V):
            try:
                s = tokenizer.decode([tid]).strip()
            except Exception:
                continue
            if not s or len(s) > 14:
                continue
            if not all(ch.isalpha() and ord(ch) < 128 for ch in s):
                continue  # require pure ASCII alpha
            if any(c.isupper() for c in s[1:]):
                continue  # drop camelCase fragments
            clean_ids.append(tid)
        clean_ids = _np.array(clean_ids, dtype=_np.int64)

        # Random sample without diversity bias
        rng = _np.random.default_rng(0)
        if len(clean_ids) > sample_tokens:
            pick = rng.choice(len(clean_ids), size=sample_tokens, replace=False)
            sample_ids = clean_ids[pick]
        else:
            sample_ids = clean_ids

        sample_vecs = embed[sample_ids]
        # Mean-center to remove dominant frequent-word axis
        center = sample_vecs.mean(axis=0)
        sample_centered = sample_vecs - center
        norms = _np.linalg.norm(sample_centered, axis=1, keepdims=True)
        norms = _np.where(norms > 1e-8, norms, 1.0)
        sample_unit = (sample_centered / norms).astype(_np.float32)

        km = MiniBatchKMeans(n_clusters=int(n_clusters), random_state=0,
                              batch_size=512, n_init=5, max_iter=400)
        km.fit(sample_unit)

        # Pre-compute centered+normalized embeds for ALL tokens (one-time cost)
        all_centered = embed - center
        all_norms = _np.linalg.norm(all_centered, axis=1, keepdims=True)
        all_norms = _np.where(all_norms > 1e-8, all_norms, 1.0)
        all_unit = (all_centered / all_norms).astype(_np.float32)

        token_id_cluster = {}

        def cluster_of(tokens):
            out = {}
            for tok in tokens:
                if tok in token_id_cluster:
                    out[tok] = token_id_cluster[tok]
                    continue
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if not ids:
                    token_id_cluster[tok] = '?'; out[tok] = '?'; continue
                v = all_unit[ids].mean(axis=0)
                n = _np.linalg.norm(v)
                if n == 0:
                    token_id_cluster[tok] = '?'; out[tok] = '?'; continue
                v = (v / n).reshape(1, -1)
                cid = int(km.predict(v)[0])
                tag = f'c{cid}'
                token_id_cluster[tok] = tag
                out[tok] = tag
            return out

        cluster_of.kmeans = km
        cluster_of.unit_embed = all_unit
        cluster_of.center = center
        cluster_of.n_clusters = int(n_clusters)
        cluster_of.n_clean_vocab = len(clean_ids)
        return cluster_of

    def absorb_llm_deep(self, hf_model, tokenizer, n_prompts=10000,
                         gen_len=80, batch_size=32, use_gpu=True,
                         seed_prompts=None, prompt_temperature=0.8,
                         sleep_after=True, auto_save_ikg=True, ikg_path=None,
                         verbose=True):
        """Pack 192 v1 -- DEEP compositional absorb. Child learns from
        diverse conversations, not from a dictionary.

        Generates n_prompts diverse LLM completions (batched on GPU),
        feeds each through org.read() so the full parser stack fires,
        plus explicit expose_meaning() per sentence for SVO/episode/mod.

        Phase A: build diverse prompt list (n_prompts strings)
        Phase B: batched LLM generation (gen_len tokens each)
        Phase C: per generation -> org.read + expose_meaning
        Phase D: sleep_consolidate
        Phase E: save .ikg

        On RTX 3050 + Qwen2.5-0.5B: ~15-30 min for n_prompts=10000.
        """
        import time as _t
        import torch
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        # Respect device_map dispatch -- do not force .to() on accelerate-managed models.
        _has_hf_device_map = hasattr(hf_model, 'hf_device_map') and bool(getattr(hf_model, 'hf_device_map'))
        if use_gpu and device == 'cuda' and not _has_hf_device_map:
            hf_model = hf_model.to(device)
        hf_model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        stats = {'n_prompts': 0, 'n_reads': 0, 'n_meaning': 0,
                 'gen_tokens': 0}

        # ── PHASE A: diverse prompts from the LLM itself ─────────────────
        if seed_prompts is None:
            seeds = self._diverse_seed_prompts(hf_model, tokenizer,
                                                  n=n_prompts, device=device)
        else:
            seeds = list(seed_prompts)[:n_prompts]
        stats['n_prompts'] = len(seeds)
        if verbose:
            print(f'  [P192v1/A] {len(seeds)} seed-token prompts ready', flush=True)

        # Pack 199: emergent clusterer DROPPED. Was collapsing all English to
        # one cluster; offered no signal. Frames + frame_vocab carry the
        # category structure now, emerging from distributional context (NO
        # hardcoded POS labels).
        cluster_of = None
        stats['n_junk_dropped'] = 0
        stats['n_pos_writes'] = 0
        stats['n_cluster_writes'] = 0

        # ── PHASE B+C: batched generation -> read -> meaning ─────────────
        if verbose:
            print(f'  [P192v1/BC] generating + reading '
                  f'(batch={batch_size}, gen_len={gen_len}, gpu={device}) ...',
                  flush=True)
        t0 = _t.perf_counter()
        import gc as _gc
        with torch.no_grad():
            for batch_start in range(0, len(seeds), batch_size):
                batch_end = min(batch_start + batch_size, len(seeds))
                batch_prompts = seeds[batch_start:batch_end]
                enc = tokenizer(batch_prompts, return_tensors='pt',
                                  padding=True, truncation=True,
                                  max_length=32).to(device)
                gen_ids = hf_model.generate(
                    **enc,
                    max_new_tokens=gen_len,
                    do_sample=True,
                    temperature=float(prompt_temperature),
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # decode + free GPU tensors before any CPU work
                texts = []
                for i in range(gen_ids.shape[0]):
                    texts.append(tokenizer.decode(gen_ids[i],
                                                   skip_special_tokens=True))
                gen_shape1 = int(gen_ids.shape[1])
                del enc, gen_ids
                if device == 'cuda':
                    torch.cuda.empty_cache()
                for text in texts:
                    if not text or len(text) < 5:
                        continue
                    # Junk filter: drop code-y tokens BEFORE substrate write.
                    raw_len = len(text)
                    text = self._filter_junk_text(text)
                    if len(text) < 5:
                        stats['n_junk_dropped'] += 1
                        continue
                    # Pack 219: frame routing now lives inside read_organism().
                    # Stats accumulated by reading org.frames.last_assigned after.
                    try:
                        # Pack 210 -- read through full cognitive stack
                        self.read_organism(text, speaker=getattr(hf_model,
                                                                      'name_or_path', 'llm'))
                        stats['n_reads'] += 1
                    except Exception:
                        pass
                    try:
                        m = self.unified.expose_meaning(text)
                        stats['n_meaning'] += sum(m.values())
                    except Exception:
                        pass
                    # Pack 199 NEW2: positional syntax-tree binding per sentence
                    try:
                        syn_n = self.unified.expose_syntax_tree(text, max_pos=8)
                        stats['n_syntax_writes'] = stats.get('n_syntax_writes', 0) + syn_n
                    except Exception:
                        pass
                    # Pack 199: emergent clusterer dropped. Frames + frame_vocab
                    # carry category structure now (no POS hardcode).
                    # Clear frame after writes so subsequent ops are unconditioned
                    self.clear_frame()
                    stats['gen_tokens'] += gen_shape1
                del texts
                # periodic full sweep every 20 batches
                if (batch_start // batch_size) % 20 == 19:
                    _gc.collect()
                if verbose and (batch_start // batch_size) % 10 == 0:
                    elapsed = _t.perf_counter() - t0
                    pct = 100.0 * batch_end / len(seeds)
                    # Pack 199 D: mid-absorb frame diagnostic
                    assigns = self.frames.assigns_per_frame.tolist() \
                        if hasattr(self, 'frames') else []
                    locked = self.frames.locked if hasattr(self, 'frames') else False
                    print(f'    progress {batch_end}/{len(seeds)} ({pct:.0f}%) '
                          f'reads={stats["n_reads"]} cooccur={len(self.unified._cooccur_seen)} '
                          f'frames_locked={locked} assigns={assigns} '
                          f'in {elapsed:.0f}s', flush=True)
        if verbose:
            print(f'  [P192v1/BC] {stats["n_reads"]} reads, '
                  f'{stats["n_meaning"]} meaning writes in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)

        # ── PHASE D: sleep consolidate ──────────────────────────────────
        if sleep_after:
            if verbose:
                print(f'  [P192v1/D] sleep_consolidate ...', flush=True)
            t0 = _t.perf_counter()
            if getattr(self, '_exposure_buf', None) is None:
                self.enable_sleep_log()
            try:
                self.sleep_consolidate(replay_factor=3, build_concepts=True,
                                          verbose=False)
                stats['phase_d_sleep'] = 1
                if verbose:
                    print(f'  [P192v1/D] consolidated in '
                          f'{_t.perf_counter()-t0:.1f}s', flush=True)
            except Exception as e:
                stats['phase_d_sleep'] = 0
                if verbose:
                    print(f'  [P192v1/D] sleep skipped: {e}', flush=True)

        # ── PHASE E: save .ikg ──────────────────────────────────────────
        stats['substrate_mb'] = self.unified.substrate_bytes() / 1_048_576
        stats['cooccur_vocab'] = len(self.unified._cooccur_seen)
        stats['seen_vocab'] = len(self.unified._seen)
        if auto_save_ikg:
            save_info = self.save_ikg(ikg_path)
            stats['ikg_path'] = save_info['path']
            stats['ikg_size_mb'] = save_info['size_mb']
            if verbose:
                print(f'  [P192v1/E] saved .ikg -> {save_info["path"]} '
                      f'({save_info["size_mb"]:.1f} MB)', flush=True)
        return stats

    # ── Pack 192 v0 (kept for archeology): vocab-sweep absorb ────────────
    def absorb_llm(self, hf_model, tokenizer, depth=2, common_k=5000,
                     neighbors=8, sleep_after=True, use_gpu=False,
                     auto_save_ikg=True, ikg_path=None,
                     verbose=True):
        """Decode LLM into the EXISTING org.unified MultiRoleMemory.

        Pack 192 (Day 64): LLM = book. Vocab sweep + multi-token forward
        feed the existing parsers (read), which write concepts into the
        fixed substrate via relate(). No new substrate file. No sidecar.
        LLM deleted at end.

        Phases:
          1. vocab embed similarity     -> relate(w, 'similar', neighbor)
          2. single-token forward       -> relate(w, 'next', top_logits)
          3. depth-N forward on common  -> self.read(text) all parsers fire
          4. sleep_consolidate          -> cluster + concept refresh

        Args:
          hf_model    : HF causal LM (Qwen/Llama/Mistral/Phi, etc)
          tokenizer   : matching HF tokenizer
          depth       : multi-token forward depth in Phase 3 (default 2)
          common_k    : how many most-distinctive vocab tokens to use in
                        Phase 3 (default 5000)
          neighbors   : top-k neighbors written per token in Phases 1+2
          sleep_after : run sleep_consolidate after absorb
          use_gpu     : move model + tensors to CUDA (default False)
          verbose     : print phase progress

        Returns: dict with counts per phase and final substrate stats.
        Caller responsibility: `del hf_model` after this returns.
        """
        import time as _t
        import torch
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        if use_gpu and device == 'cuda':
            hf_model = hf_model.to(device)
        hf_model.eval()

        # ensure roles exist (Pack 195 will add to DEFAULT_ROLES; v0 registers
        # 'similar' dynamically via _ensure_role)
        self._ensure_role('similar')

        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        embed_t = model_inner.embed_tokens.weight.detach()
        if device == 'cuda':
            embed_cpu = embed_t.cpu()
        else:
            embed_cpu = embed_t
        embed = embed_cpu.numpy().astype(np.float32)
        vocab = embed.shape[0]
        stats = {'phase1': 0, 'phase2': 0, 'phase3': 0, 'phase4': 0,
                 'vocab': int(vocab)}

        # Decode batch once (massive speedup over per-token decode)
        if verbose:
            print(f'  [P192/0] decoding vocab strings ...', flush=True)
        t0 = _t.perf_counter()
        words = []
        for tid in range(vocab):
            try:
                w = tokenizer.decode([int(tid)]).strip()
            except Exception:
                w = ''
            words.append(w if w and len(w) <= 64 else '')
        if verbose:
            n_good = sum(1 for w in words if w)
            print(f'  [P192/0] decoded {n_good}/{vocab} usable in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)

        # ── PHASE 1: vocab embed similarity (BATCHED) ────────────────────
        if verbose:
            print(f'  [P192/1] vocab embed similarity (batched) ...',
                  flush=True)
        t0 = _t.perf_counter()
        embed_n = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-9)
        BATCH = 512
        for chunk0 in range(0, vocab, BATCH):
            chunk1 = min(chunk0 + BATCH, vocab)
            # (chunk, vocab) similarity matrix
            sims = embed_n[chunk0:chunk1] @ embed_n.T
            # mask self
            for i, tid in enumerate(range(chunk0, chunk1)):
                sims[i, tid] = -1.0
            # top-k per row
            top_idx = np.argpartition(-sims, neighbors, axis=1)[:, :neighbors]
            for i, tid in enumerate(range(chunk0, chunk1)):
                w = words[tid]
                if not w:
                    continue
                for n in top_idx[i]:
                    nw = words[int(n)]
                    if not nw:
                        continue
                    self.unified.relate(w, 'similar', nw)
                    stats['phase1'] += 1
            if verbose and chunk0 // BATCH % 50 == 0:
                print(f'    P1 progress {chunk1}/{vocab} '
                      f'({stats["phase1"]} relations so far)', flush=True)
        if verbose:
            print(f'  [P192/1] {stats["phase1"]} similar-relations in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)

        # ── PHASE 2: single-token forward -> next transitions (BATCHED) ──
        if verbose:
            print(f'  [P192/2] single-token forward -> next (batched) ...',
                  flush=True)
        t0 = _t.perf_counter()
        FB = 256 if device == 'cuda' else 32
        with torch.no_grad():
            for chunk0 in range(0, vocab, FB):
                chunk1 = min(chunk0 + FB, vocab)
                ids_t = torch.arange(chunk0, chunk1, device=device).unsqueeze(1)
                logits = hf_model(ids_t).logits[:, -1].cpu().numpy()
                top_idx = np.argpartition(-logits, neighbors, axis=1)[:, :neighbors]
                for i, tid in enumerate(range(chunk0, chunk1)):
                    w = words[tid]
                    if not w:
                        continue
                    for n in top_idx[i]:
                        nw = words[int(n)]
                        if not nw:
                            continue
                        self.unified.relate(w, 'next', nw)
                        stats['phase2'] += 1
                if verbose and chunk0 // FB % 50 == 0:
                    print(f'    P2 progress {chunk1}/{vocab} '
                          f'({stats["phase2"]} relations so far)', flush=True)
        if verbose:
            print(f'  [P192/2] {stats["phase2"]} next-relations in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)

        # ── PHASE 3: depth-N forward on common tokens -> org.read ────────
        if verbose:
            print(f'  [P192/3] depth-{depth} forward on top {common_k} '
                  f'distinctive tokens ...', flush=True)
        t0 = _t.perf_counter()
        # pick the most distinctive tokens by embed variance
        diversity = embed_n.std(axis=1)
        k_eff = int(min(common_k, vocab))
        common = np.argpartition(-diversity, k_eff - 1)[:k_eff]
        with torch.no_grad():
            for tid in common:
                ids = [int(tid)]
                for _ in range(int(depth)):
                    ids_t = torch.tensor([ids], device=device)
                    out = hf_model(ids_t).logits[0, -1]
                    ids.append(int(out.argmax().item()))
                text = tokenizer.decode(ids)
                if text and len(text) <= 512:
                    self.read(text)
                    stats['phase3'] += 1
                if verbose and stats['phase3'] % 1000 == 0 and stats['phase3'] > 0:
                    print(f'    P3 progress {stats["phase3"]}/{k_eff}',
                          flush=True)
        if verbose:
            print(f'  [P192/3] {stats["phase3"]} compositional reads in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)

        # ── PHASE 4: sleep consolidate ───────────────────────────────────
        if sleep_after:
            if verbose:
                print(f'  [P192/4] sleep_consolidate ...', flush=True)
            t0 = _t.perf_counter()
            if getattr(self, '_exposure_buf', None) is None:
                self.enable_sleep_log()
            try:
                self.sleep_consolidate(replay_factor=3, build_concepts=True,
                                          verbose=False)
                stats['phase4'] = 1
                if verbose:
                    print(f'  [P192/4] consolidated in '
                          f'{_t.perf_counter()-t0:.1f}s', flush=True)
            except Exception as e:
                if verbose:
                    print(f'  [P192/4] sleep_consolidate skipped: {e}',
                          flush=True)

        stats['substrate_mb'] = self.unified.substrate_bytes() / 1_048_576
        stats['cooccur_vocab'] = len(getattr(self.unified, '_cooccur_seen', set()))
        stats['seen_vocab'] = len(getattr(self.unified, '_seen', set()))
        for k, v in (self.unified._role_targets or {}).items():
            stats[f'role_{k}_targets'] = len(v)
        if verbose:
            print(f'  [P192] DONE. substrate {stats["substrate_mb"]:.1f} MB, '
                  f'vocab seen {stats["seen_vocab"]}, '
                  f'phase counts: '
                  f'P1={stats["phase1"]} P2={stats["phase2"]} '
                  f'P3={stats["phase3"]} P4={stats["phase4"]}', flush=True)
        if auto_save_ikg:
            save_info = self.save_ikg(ikg_path)
            if verbose:
                print(f'  [P192] auto-saved .ikg -> {save_info["path"]} '
                      f'({save_info["size_mb"]:.1f} MB compressed)', flush=True)
            stats['ikg_path'] = save_info['path']
            stats['ikg_size_mb'] = save_info['size_mb']
        return stats

    # ── Pack 173: T2S Compiler v0 (NVIDIA Killer #2) ─────────────────────
    def t2s(self, role=None, seed=24001, d=None, K=1):
        """Open a Transformer-to-Substrate compiler.

        Returns the active T2S if one exists (preserves an absorbed model
        across calls with default args). Builds a fresh one only when
        explicit d/K differ from the active one, or when none exists.
        """
        from ikigai.cognition.t2s_compiler import T2SCompiler
        existing = getattr(self, '_t2s', None)
        # If caller passed defaults (d=None, K=1) and we already have one,
        # return the existing one regardless of its d/K -- supports the
        # "absorb once, prompt later" pattern.
        if existing is not None and d is None and K == 1:
            return existing
        cache_key = (d, K)
        if (existing is None
                or getattr(self, '_t2s_dk', None) != cache_key):
            self._t2s = T2SCompiler(self, role=role, seed=seed, d=d, K=K)
            self._t2s_dk = cache_key
        return self._t2s

    # ── Pack 190: Galois-field rank router (must-invent #2) ───────────────
    def galois_router(self, p=251, d=None, seed=4096):
        """Open the Galois-field rank router (lazy-instantiated).
        Crosstalk bounded by 1/p (vs 1/sqrt(d) for cosine). Sharpens dict
        atom lookup at scale. Pack 190 -- external research must-invent #2.
        """
        from ikigai.cognition.galois_router import GaloisRouter
        existing = getattr(self, '_galois', None)
        d_eff = int(d) if d is not None else int(self.unified.d) * 4
        if existing is None or getattr(self, '_galois_pd', None) != (p, d_eff):
            self._galois = GaloisRouter(p=int(p), d=d_eff, seed=int(seed))
            self._galois_pd = (int(p), d_eff)
        return self._galois

    # ── Pack 191: in-situ non-interfering multi-model writer (must-invent #1) ──
    def in_situ_writer(self, substrate=None):
        """Open the in-situ namespace-phasor writer (lazy-instantiated).
        Holds N models in the SAME substrate with bounded cross-model
        crosstalk via orthogonal namespace phasors. Pack 191 must-invent #1.
        """
        from ikigai.cognition.in_situ_writer import InSituWriter
        existing = getattr(self, '_isw', None)
        target = substrate if substrate is not None else self.unified.sdm_rel
        if existing is None or getattr(self, '_isw_target', None) is not target:
            self._isw = InSituWriter(target)
            self._isw_target = target
        return self._isw

    # ── Pack 211: substrate-native absorb (model gone, substrate runs) ────
    def _ensure_big_substrate(self, d=2048, M=131072, k=128, seed=4096):
        """Lazy-alloc the big VSA-SDM substrate (~4 GB at default).
        Reused across absorb/generate calls."""
        t2s = self.t2s(d=self.unified.d, K=1)
        if not hasattr(t2s, 'big_substrate'):
            t2s.create_big_substrate(d=d, M=M, k=k, seed=seed)
        return t2s

    def absorb_native(self, hf_model, tokenizer=None, model_id=None,
                       name_prefix='subst', K_writes=2,
                       d=2048, M=131072, verbose=True):
        """Substrate-native absorption: write LLM weights INTO substrate,
        precompute Pack 213 idx cache for fast forward, drop model reference.
        Caller must `del hf_model` in their scope to fully release."""
        import gc as _gc
        t2s = self._ensure_big_substrate(d=d, M=M)
        cfg = hf_model.config
        self._absorbed_config = {
            'num_hidden_layers': int(cfg.num_hidden_layers),
            'num_attention_heads': int(cfg.num_attention_heads),
            'num_key_value_heads': int(getattr(cfg, 'num_key_value_heads',
                                                  cfg.num_attention_heads)),
            'hidden_size': int(cfg.hidden_size),
            'intermediate_size': int(cfg.intermediate_size),
            'head_dim': int(getattr(cfg, 'head_dim',
                                      cfg.hidden_size // cfg.num_attention_heads)),
            'rope_theta': float(getattr(cfg, 'rope_theta', 10000.0)),
            'rms_norm_eps': float(getattr(cfg, 'rms_norm_eps', 1e-6)),
            'vocab_size': int(cfg.vocab_size),
            'model_id': model_id,
        }
        self._absorbed_tokenizer = tokenizer
        self._absorbed_name_prefix = name_prefix
        n_layers, n_writes = t2s.compile_llama_into_big_substrate(
            hf_model, K_writes=K_writes, name_prefix=name_prefix,
            verbose=verbose)
        t2s.precompute_big_idx_cache(name_prefix=name_prefix, verbose=verbose)
        _gc.collect()
        return {'n_layers': n_layers, 'n_writes': n_writes,
                'substrate_gb': t2s.big_substrate.substrate_bytes() / 1e9}

    def generate_native(self, prompt, max_new=10, tokenizer=None, fast=True,
                          verbose=False):
        """Substrate-only generation. Forward reads weights from substrate."""
        t2s = self.t2s(d=self.unified.d, K=1)
        if not hasattr(t2s, 'llm_n_layers'):
            raise RuntimeError('no model absorbed; call org.absorb_native() first')
        tok = tokenizer or self._absorbed_tokenizer
        if tok is None:
            raise ValueError('tokenizer required')
        prefix = self._absorbed_name_prefix
        fwd = (t2s.llama_forward_big_substrate_fast if fast
               else t2s.llama_forward_big_substrate)
        ids = list(tok.encode(prompt))
        for _ in range(int(max_new)):
            logits = fwd(ids, name_prefix=prefix)
            nxt = int(logits[-1].argmax())
            ids.append(nxt)
            if verbose:
                print(f'  +{tok.decode([nxt])!r}', flush=True)
        return tok.decode(ids)

    def forward_logits_native(self, prompt, tokenizer=None, fast=True):
        """One substrate-only forward pass. Returns logits (seq_len, vocab)."""
        t2s = self.t2s(d=self.unified.d, K=1)
        tok = tokenizer or self._absorbed_tokenizer
        prefix = self._absorbed_name_prefix
        fwd = (t2s.llama_forward_big_substrate_fast if fast
               else t2s.llama_forward_big_substrate)
        return fwd(tok.encode(prompt), name_prefix=prefix)

    def audit_native(self):
        """Honest accounting: what's IN substrate vs still sidecar."""
        t2s = self.t2s(d=self.unified.d, K=1)
        if not hasattr(t2s, 'big_substrate'):
            return {'absorbed': False}
        sub_bytes = int(t2s.big_substrate.substrate_bytes())
        sidecar = {}
        if hasattr(t2s, 'llm_embed'):
            sidecar['llm_embed'] = t2s.llm_embed.nbytes
        if hasattr(t2s, 'llm_lm_head'):
            shared = (hasattr(t2s, 'llm_embed')
                       and t2s.llm_lm_head is t2s.llm_embed)
            sidecar['llm_lm_head'] = ('shared_with_embed' if shared
                                       else t2s.llm_lm_head.nbytes)
        bc = getattr(t2s, 'bias_cache', {}) or {}
        sidecar['bias_cache'] = sum(v.nbytes for v in bc.values()
                                     if hasattr(v, 'nbytes'))
        lnc = getattr(t2s, 'ln_cache', {}) or {}
        sidecar['ln_cache'] = sum((v[0].nbytes if hasattr(v[0], 'nbytes') else 0)
                                   + (v[1].nbytes if hasattr(v[1], 'nbytes') else 0)
                                   for v in lnc.values())
        sidecar_bytes = sum(v for v in sidecar.values() if isinstance(v, int))
        return {
            'absorbed': True,
            'config': getattr(self, '_absorbed_config', None),
            'substrate_bytes': sub_bytes,
            'substrate_gb': sub_bytes / 1e9,
            'sidecar_breakdown': sidecar,
            'sidecar_bytes': sidecar_bytes,
            'sidecar_mb': sidecar_bytes / 1e6,
            'total_gb': (sub_bytes + sidecar_bytes) / 1e9,
            'in_substrate': ['Q/K/V/O', 'gate/up/down (per-layer weights)'],
            'pack_212_targets': ['llm_embed', 'llm_lm_head',
                                  'bias_cache', 'ln_cache'],
        }

    # ── Pack 200: Universal Data Codec Protocol -- the phone ─────────────
    def absorb_anything(self, ecc_replicas=3, hopfield_iter=5,
                          hopfield_beta=8.0, keep_hv_store=True):
        """Open the Universal Codec Protocol pipeline (lazy-instantiated).
        Absorb any data (text, weights, image, bytes, custom) into substrate
        with bijective recall via per-modality codecs + Pack 191 namespace
        isolation + Pack 190 Galois sharpening + Pack 202 Hopfield iterative
        refine. Pack 200 -- the END-TO-END phone.
        """
        from ikigai.cognition.universal_codec import UniversalCodec
        existing = getattr(self, '_udcp', None)
        if existing is None:
            self._udcp = UniversalCodec(self, ecc_replicas=ecc_replicas,
                                          hopfield_iter=hopfield_iter,
                                          hopfield_beta=hopfield_beta,
                                          keep_hv_store=keep_hv_store)
        return self._udcp

    def assert_isa_modulated(self, hypo, hyper, base_n=20):
        """Pack 170+: write reinforcement count scales with chemical state.
        Focused/rewarding/threatening moments encode MORE strongly than
        bored ones, mirroring biological memory."""
        n = self.neuro.write_strength(base_n)
        for _ in range(n):
            self.unified.relate(hypo, 'isa', hyper)
        return n

    def assert_relation_modulated(self, word, role, target, base_n=20):
        """Modulated reinforcement for any role."""
        n = self.neuro.write_strength(base_n)
        for _ in range(n):
            self.unified.relate(word, role, target)
        return n

    def attend_modulated(self, query, candidates, roles=None,
                         base_temperature=1.0):
        """
        Mood-aware multi-head substrate cleanup.
        Per-role weights and softmax temperature come from current
        neuromod state. Returns the same shape as VSAAttention.cleanup:
        [(candidate, prob), ...] sorted descending.
        """
        att = self.attention(roles=roles or ('cooccur', 'isa', 'property'))
        nm  = self.neuro
        weights = nm.attention_weights(att.roles)
        temp    = float(base_temperature) * nm.temperature_scale()
        return att.cleanup(query, candidates,
                           weights=weights, temperature=temp)

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

    # ── few-shot pattern learning (Pack 132) ────────────────────────────────

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
        out = eng.generate(prompt=prompt, max_tokens=max_tokens,
                             return_trace=return_trace, seed=seed)
        # Pack 211 -- post-gen verifier pass: check coherence vs prompt belief
        try:
            text_out = out if isinstance(out, str) else out[0]
            tokens_out = text_out.split()
            if tokens_out:
                # Reference belief HV = first prompt token's belief or accumulated
                ref_tok = tokens_out[0] if tokens_out else 'the'
                k = self.unified.ck.key(ref_tok)
                bip = np.sign(k.real).astype(np.float32)
                bip = np.where(bip == 0, 1.0, bip)
                B_U = bip[:self.verifier.d]
                ok, score = self.verifier.verify_coherence(tokens_out[:32], B_U)
                self._verifier_scores.append((bool(ok), float(score)))
        except Exception as e:
            self._verifier_scores.append(('err', str(e)[:60]))
        return out

    def cogitate_modulated(self, prompt='', max_tokens=100,
                           base_temperature=0.7, base_thought_gamma=4.0,
                           base_momentum=0.7, return_trace=False, seed=None,
                           grounded_gamma=0.0, top_k=20,
                           respect_forced_rest=True):
        """
        Pack 171: Neuromodulator-driven generation.

        Same engine as `cogitate()` but every knob is set from current
        neuromod state:
            temperature   <- base * neuro.temperature_scale()
            thought_gamma <- base * (1 + cortisol_excess)  (rigid focus)
            momentum      <- base + 0.1 * dopamine_excess  (loose under DA)
            grounded_gamma<- base + 1.5 * cortisol_excess  (force isa snap)

        If `respect_forced_rest` and `neuro.forced_rest()` returns True,
        skip generation (return placeholder) -- the organism is in a
        homeostatic crisis and must sleep first.
        """
        from ikigai.cognition.generation_engine import GenerationEngine
        nm = self.neuro
        if respect_forced_rest and nm.forced_rest():
            return '[forced_rest -- cortisol load too high; sleep first]'
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = GenerationEngine(self)
        eng = self._engine
        # base settings
        eng.think_steps   = 3
        eng.momentum      = float(np.clip(base_momentum +
                                          0.1 * (nm.level['dopamine']
                                                  - nm.baseline['dopamine']),
                                          0.0, 0.95))
        eng.thought_gamma = float(base_thought_gamma *
                                  (1.0 + max(0.0,
                                              nm.level['cortisol']
                                               - nm.baseline['cortisol'])))
        eng.temperature   = float(np.clip(base_temperature *
                                          nm.temperature_scale(),
                                          0.05, 5.0))
        eng.top_k         = int(top_k)
        eng.remove_common = True
        eng.ngram_weights = (0.2, 0.4, 0.4)
        eng.ngram_ctx     = 3
        eng.goal_gamma    = 0.0
        eng.grounded_gamma = float(grounded_gamma +
                                   1.5 * max(0.0,
                                              nm.level['cortisol']
                                               - nm.baseline['cortisol']))
        eng.grounded_roles = ('isa', 'property')
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

    # ── flat-memory interface (Pack 114-115) ─────────────────────────────────

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

    # ── Phase 3: dialogue + generation ───────────────────────────────────

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

    # ── Long-term memory ─────────────────────────────────────────────────

    def remember(self, name, key_tokens, value_tokens):
        """Long-term holographic store."""
        return self.holo.store(name, key_tokens, value_tokens)

    def recall(self, key_tokens, top_k=3):
        """Long-term holographic recall."""
        return self.holo.recall(key_tokens, top_k=top_k)

    # ── Introspection ────────────────────────────────────────────────────

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

    # ── Persistence ──────────────────────────────────────────────────────

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


# ── singleton convenience ────────────────────────────────────────────────────

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
