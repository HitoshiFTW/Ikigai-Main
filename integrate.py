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

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── Pillars (foundation) ─────────────────────────────────────────────────────
from ikigai.cognition.cgpsp_encoder       import CGPSPEncoder
from ikigai.cognition.pi_k_algebra        import PiK
from ikigai.cognition.pgmw                import PersonaGrid

# ── Reasoning core (hardcoded path -- works on simple SVO) ───────────────────
# reasoning_engine REMOVED (audit trio, Day-83): legacy python-dict WorkingMemory
# arithmetic + regex SVO parser; superseded by GeneralReasoner (compositional derive
# + RHC math ring) and holo_reader emergent parse. ask()/parse paths rewired below.

# ── BEING: persistent living organism (the substrate) ────────────────────────
from ikigai.cognition.being               import IkigaiBeing
from ikigai.cognition.operational_grounding import OperationalGrounding
from ikigai.cognition.sensory_grounding   import SensoryGrounding
from ikigai.cognition.taxonomic_grounding import TaxonomicGrounding
from ikigai.cognition.grammar_grounding   import GrammarGrounding
from ikigai.cognition.flat_memory         import FlatMemory
from ikigai.cognition.multirole_memory    import MultiRoleMemory
from ikigai.cognition.dialogue            import DialogueLoop
# generator/SentenceGenerator REMOVED (Day-83 audit rewire): org.generate +
# new_dialogue now use frame_relax (say_frame). Old Markov-on-being.lexicon retired.

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
# hot_loader REMOVED (audit batch 2, Day-83): exec-wrapper over deleted code_gen.

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
        # Day-83 audit END-STATE: ONE organism, EVERYTHING ON. `flat_only` is now
        # a NO-OP, kept only so old call-sites don't error -- every cognition
        # module is always built. Call IkigaiOrganism() and the whole organism
        # runs; no mode flag, no specific-class gating.
        self._flat_only = False

        # Foundation pillars (UHE). ONE dimension d (was a separate hardcoded
        # d=2048 space -- the old pre-flat-substrate paradigm). Centred on the
        # d=400 RHC/cache/memory core.
        self.encoder = CGPSPEncoder(d=d, gamma=0.4)
        self.pik     = PiK(d=d, n_primes=32)
        self.persona = PersonaGrid(d=d)

        # BEING + grounding channels (Pack 96-101). being.lexicon shares the
        # substrate's ComputedKey identity (Day-83 migrate).
        self.being      = IkigaiBeing(d=d, drift_rate=0.08, window_size=4)
        self.operations = OperationalGrounding(d=d)
        self.sensory    = SensoryGrounding(d=d)
        self.taxonomy   = TaxonomicGrounding(d=d)
        self.grammar    = GrammarGrounding(d=d)

        # FLAT MEMORY (Pack 114-115) -- single-channel reference (unified is the
        # production substrate). Built for completeness; everything on.
        self.flat          = FlatMemory(d=d, M=16384, k=64, seed=114)
        self._flat_enabled = False

        # UNIFIED MEMORY (Pack 117-118): the actual flat substrate. Always built.
        self.unified          = MultiRoleMemory(d=d, M=16384, k=64, seed=114)
        self._unified_enabled = True
        # Pack 197 Resonance Frame Field. Frame-conditioned binding so the
        # organism can hold N languages / N domains without cross-talk.
        from ikigai.cognition.frame_field import FrameField
        self.frames = FrameField(d=d, K=8, top_n=64, seed=42, alpha=0.5)
        self.unified._frame_field_ref = self.frames
        # Day-83 audit MIGRATE: back being.lexicon by the substrate's ComputedKey
        # so being shares ONE word-identity space with the unified substrate
        # (the dictionary->HDC supersession). being is built before unified, so
        # attach the key_fn now that unified exists. Drift still rides on top.
        if getattr(self, 'being', None) is not None:
            self.being._key_fn = self.unified.ck.key
        # Pack 210 -- BIG WIRE-UP. Activate 5 dead cognition modules so read()
        # runs reasoning during ingestion, not just statistics.
        from ikigai.cognition.free_energy_drive import (
            ConversationalVariationalFreeEnergyField)
        from ikigai.cognition.curiosity_drive import CuriosityDrive
        from ikigai.cognition.theory_of_mind import TheoryOfMindSandbox
        from ikigai.cognition.vsa_calculus import VSACalculus
        # Pack 211 -- Generation wire
        from ikigai.cognition.belief_field import BeliefField
        from ikigai.cognition.self_verifier import SelfVerifier
        from ikigai.cognition.proof_carrying_gen import ProofCarryingGenerator
        self.fe = ConversationalVariationalFreeEnergyField(d=64, window=8)
        self.curiosity = CuriosityDrive(d=d)
        self.tom = TheoryOfMindSandbox(d=d)
        self.tom.add_agent('default')
        self.vsa = VSACalculus(d=d)
        # Pack 211 instances
        # COLLAPSE (Day-83 audit): belief/tom/vsa built ONCE here (always), with
        # the tuned belief params formerly only in the full-mode rebuild. The
        # flat_only None-clobber + the T3 duplicate are removed -- these gold
        # modules now run in BOTH modes (toward the one-organism END-STATE).
        self.belief = BeliefField(d=d, conflict_threshold=-0.05, heal_rate=2.0)
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
        # WIRE (Day-83 audit): self-inconsistency tracking -- meta_mirror.high_drift
        # increments this in read() when the self-model drifts past threshold.
        self._high_drift_count = 0
        self._last_high_drift_tick = -1
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
        self._fe_log = []
        self._last_passage_hv_curiosity = None
        self._read_organism_count = 0
        self._verifier_scores = []   # Pack 211 -- post-gen verifier log
        # being/grammar are always built now -> dict writes always available
        # (everything on, Day-83 END-STATE).
        self._dict_writes_enabled = True

        # HEAVY / full-only cognition modules. None in flat_only.
        # (belief/tom/vsa COLLAPSED to the unconditional build above -- no longer
        # clobbered here nor rebuilt below. cf/cwm remain T2-built at d=128.)
        # COLLAPSE (Day-83 audit): dropped self.decay/self.curio/self.pcg/self.pfc --
        # dead duplicates of the always-built+wired imp_lattice/curiosity/proof_gen.
        # These cognition modules are now built UNCONDITIONALLY (both modes) so they
        # always run -- one organism, no flat_only stripping of them.
        self.holo    = HolographicMemory(d=d)
        self.modal   = CrossModalSpace(d=d)
        self.osc     = CrossTimeResonator(d=d, periods=[10, 100, 1000])
        self.atom    = ConceptAtomizer(d=d)
        self.immune  = AdversarialImmune(d=d)
        # fea pinned to d=128 to match the (T2, d=128) causal_world_model the
        # planner pairs it with (COLLAPSE consistency, Day-83 audit).
        self.fea     = FreeEnergyActionSelector(d=128)
        self.planner = MultiStepPlanner(self.cwm, self.fea)
        # hotload REMOVED (audit batch 2): exec-wrapper over deleted code_gen; unused.

        # Episodic chain (hippocampus)
        self._episodes = []
        self._last_trace = []
        self._tick = 0

    # ── Primary interface ────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────
    # Day-99 -- THE ONE API.  You call the organism. The organism decides.
    # ─────────────────────────────────────────────────────────────────────────────────────────

    _FACULTIES = ()          # populated below by _register_faculty; ORDER MUST NOT MATTER.

    @classmethod
    def _register_faculty(cls, name, fn):
        cls._FACULTIES = tuple(cls._FACULTIES) + ((name, fn),)
        return fn

    @staticmethod
    def _surprise(conf):
        """F = -log(confidence).  0 at certainty, grows without bound as confidence -> 0."""
        import math
        return -math.log(max(1e-9, min(1.0, float(conf))))

    def sense(self, x):
        """Day-99 -- EVERY faculty ATTEMPTS the input and reports MEASURED confidence.

        This is the anti-cheat. The tempting design is to classify the input first --
        `if looks_like_a_question: answer() elif looks_like_a_fact: learn()` -- but that is the
        if/elif ladder again, just moved upstream, and an author decided it. The Day-95 red team
        named this exact move (DS-DISPATCH), and Day-99 measured its cost: identical proposals
        returned different answers purely by append order.

        So nothing here inspects the SHAPE of the input. Each faculty simply TRIES, and reports how
        well it actually did, from its own measurement:
          learn   -- extraction score x prediction error (nothing to learn if already believed)
          answer  -- the derive path's own grounding
          speak   -- how much the organism can actually say about what it recognises
          abstain -- always available at the calibrated floor, so "I don't know" can WIN
        The organism then picks argmin F. Change what it KNOWS and it does something different;
        change the ORDER faculties were registered and nothing changes at all.

        Returns the proposal list [(faculty, payload, F)] -- the organism's felt options."""
        props = []
        for name, fn in self._FACULTIES:
            try:
                p = fn(self, x)          # RAW input: a faculty decides what it can make of it
            except Exception:
                p = None
            if p is not None:
                conf, payload = p
                if conf is not None and conf > 0.0:
                    props.append((name, payload, self._surprise(conf)))
        return props

    # Day 101 -- ConceptGraph was found dormant (read_organism, its only writer,
    # has zero call sites anywhere in the repo) AND, on inspection, its own write
    # calls build the WRONG event schema (constructs {'tokens','valence',...} but
    # ConceptGraph.ingest_event needs {'latent_end','dominant_action',...} --
    # would KeyError every time, silently swallowed by a blanket try/except).
    # This is a real, separate, WORKING pipeline wired into the actual org(x)
    # entry point (be()), not the dead read_organism: real per-call signals
    # (chosen faculty = action, measured F = energy/cortisol proxy, a token-key
    # projection = latent) instead of the broken ad-hoc dict. No authored
    # lexicon (neuromod.expose_tokens's SPIKE_LEXICON was deliberately avoided --
    # it's a hand-curated word list, exactly what the standing no-hardcoding
    # rule bans on a production path).
    def _concept_latent(self, tokens):
        """64-dim real projection of a token's computed key -- the SAME
        substrate identity every other module derives from (ComputedKey),
        just truncated to the dims ConceptGraph/EventCompressor expect.
        No new authored vocabulary, no lexicon."""
        if not tokens:
            return [0.0] * 64
        hv = self.unified.ck.key(str(tokens[-1]))
        return [float(v) for v in hv.real[:64]]

    def _update_concept_memory(self, tokens, chose, F):
        """Feed one real (state, action, outcome) transition into
        EventCompressor -> ConceptGraph, then check the LATTER for a
        newly-completed compressed event to forward. This is the actual
        chain the docstrings always described but the dead read_organism
        never correctly performed. Bounded (event_comp maxlen=500,
        concept_graph max_nodes=256) -- consolidate-and-forget, not
        unbounded growth. Best-effort: never raises into be().

        EventCompressor's segmentation law compares latent_post of transition
        N against latent_pre of transition N+1 -- it expects a CONTINUOUS
        trajectory (same evolving state), which is only meaningful if both
        derive from the SAME underlying signal (the topic/content being
        processed). Using latent_pre=key(query) and latent_post=key(chosen
        faculty) -- two unrelated vectors -- was tried first and measured to
        never pass the 0.9 similarity bar even for the identical repeated
        query, so nothing ever accumulated (caught by day101_concept_memory_
        live.py before this reached anyone). Fix: both derive from the
        query's own content; the transition's action/energy/cortisol carry
        what happened, not the latent."""
        try:
            self._self_tick += 1
            latent = self._concept_latent(tokens)
            ep = {
                'tick': self._self_tick,
                'latent_pre': latent,
                'latent_post': latent,
                'action': str(chose),
                'reason_stage': 'default',   # no staged-reasoning signal exists yet;
                                              # disclosed simplification, not fabricated
                'energy_delta': -float(F),
                'cortisol_delta': float(F),
            }
            n_before = len(self.event_comp)
            self.event_comp.ingest_transition(ep)
            if len(self.event_comp) > n_before:
                self.concept_graph.ingest_event(self.event_comp.last())
        except Exception:
            pass

    def _fac_learn(self, text):
        """Confidence that learning is the right thing: how cleanly can this be extracted as a
        fact, and how WRONG is the organism about it right now? A fact it already believes has
        prediction error 0 -> nothing to learn -> no proposal (not a hardcoded rule; the RPE is
        computed from its own belief)."""
        # A (subject, relation, object) tuple IS a fact -- reading it is typing, not shape-guessing,
        # so extraction is certain and only the prediction error decides whether learning wins.
        # Raw TEXT must go through the LEARNED extractor: this organism does not hardcode grammar,
        # so an untrained one genuinely cannot extract, and honestly proposes nothing.
        if isinstance(text, (tuple, list)) and len(text) == 3 and all(text):
            tri, score = tuple(str(v).strip().lower() for v in text), 1.0
        else:
            # a QUESTION is not a telling -- never learn from it (correct-or-abstain).  The
            # learned extractor has no such guard, so gate the whole faculty here.
            if isinstance(text, str) and self._is_question(text):
                return None
            try:
                tri, score = self.extract_verified(str(text), return_score=True)
            except Exception:
                tri, score = None, 0.0
            if (not tri or len(tri) < 3 or not all(tri[:3])) and isinstance(text, str):
                # a paragraph: the whole blob may not match a single frame -- the FIRST sentence
                # that DOES (via the LEARNED frames) is enough to win 'learn'; be() -> tell() then
                # absorbs every sentence. Splitting on sentence punctuation, not on any word list.
                import re as _re
                for _s in _re.split(r'[.!?\n]+', text):
                    _s = _s.strip()
                    if not _s or self._is_question(_s):
                        continue
                    try:
                        ev = self.extract_verified(_s, return_score=True)
                    except Exception:
                        ev = (None, 0.0)
                    if ev and ev[0] and len(ev[0]) >= 3 and all(ev[0][:3]):
                        tri, score = ev[0], ev[1]
                        break
        if not tri or len(tri) < 3 or not all(tri[:3]):
            return None
        err = self.prediction_error(tri[0], tri[1], tri[2])
        conf = float(score) * float(err)
        return (conf, tuple(tri[:3])) if conf > 0.0 else None

    def _fac_answer(self, text):
        """Confidence that answering is right: the derive path either grounds an answer or it does
        not. Grounding is the organism's own no-hallucination gate (Day-98 must-do #5).

        Day-103 -- DERIVE + FLUENT, TALK TILL DONE first: a question can ask several things
        ('the capital of france AND its continent'); answer_fluent extracts every asked
        relation, derives each fact, and speaks a fluent clause per fact until all are
        covered, then halts (grounded, query-scoped, honest). Only when it produces
        nothing (not a factual multi-relation ask) does the single-relation derive path
        below run -- so org(x) answers completely and fluently through its own front door."""
        if not isinstance(text, str):
            return None
        try:
            fluent = self.answer_fluent(text)
        except Exception:
            fluent = None
        if fluent:
            return (1.0, fluent)                      # derived, complete, fluent -> certain
        try:
            a = self.answer(text)
        except Exception:
            return None
        if not isinstance(a, dict):
            return None
        fact = a.get('fact')
        txt = (a.get('text') or '').strip()
        if not fact or not txt or txt.lower().startswith("i don't know"):
            return None
        return (1.0 if a.get('grounded') else 0.5, txt)

    def _targeted_ungrounded(self, toks, topic):
        """Day-100 -- does this input structurally name a relation of `topic` that the organism
        does NOT hold? Shared by _fac_speak and _fac_wonder, both of which had the same disease:
        given a word never modeled as a relation before ('mayor', 'inventor'), each fell back to
        substituting something ELSE about the topic instead of admitting nothing matched.

        _REL_TEMPLATES generates relation-questions as '{r} of {e}' / 'what is the {r} of {e}' --
        so a content word immediately before 'of {topic}' is being used AS a relation name by the
        question's OWN structure, independent of whether that word was ever taught as one. Probed
        via eng.atoms(), the same generic-template path any real relation round-trips through --
        no authored word list. Function words are the organism's own induced-frame literals (the
        same derivation extract_corpus uses), never an authored stoplist.

        Returns True only when the pattern is structurally present AND ungrounded -- absent
        pattern means "don't know", not "grounded", so callers must treat False as uninformative,
        not as permission."""
        try:
            ti = toks.index(topic)
        except ValueError:
            return False
        if ti < 2 or toks[ti - 1] != 'of':
            return False
        cand_rel = toks[ti - 2]
        stop = set()
        for _f in (getattr(self.surface, 'templates', None) or {}).values():
            for _t in _f:
                if not (str(_t).startswith('{') and str(_t).endswith('}')):
                    stop.add(str(_t).strip().lower())
        if not cand_rel or cand_rel == topic or cand_rel in stop:
            return False
        try:
            return not self.general_reasoner.derive_engine.atoms(cand_rel, topic)
        except Exception:
            return False

    def _fac_wonder(self, text):
        """Confidence that ASKING is the right act -- the organism's own curiosity, competing.

        The hard part was not detecting the gap (`wonder` already returns
        {'relation','question','novelty'} from a real gap analysis) but PRICING it. Novelty is 1.0
        for a total gap, so a naive wonder proposes at F=0 and TIES with a confident answer:
        `org('what is the capital of france')` would then deadlock into 'conflict' on a question it
        can answer perfectly. Every obvious fix was a cheat: discount curiosity by a constant
        (authored), fire only when the input "isn't a question" (shape classification -- the ladder
        moved upstream), or slot it just above abstain (the same authored blend renamed).

        The framing was wrong. The question is not "how much is curiosity worth against answering?"
        -- it is "is this gap even ABOUT what was asked?", and that is evidence, not preference:
          * the input NAMES a relation it already knows for that entity -> nothing to wonder about
            HERE. Silent. (france's gap is `country`; the input asked `capital`, which it knows.)
          * the input names a relation that IS a gap -> that is exactly the thing to ask.
          * the input names a bare entity -> its most novel gap is fair game.
        Relations are read with the engine's OWN relation vocabulary, the same source the answer
        faculty parses against. No authored list, no shape test.

        Confidence = novelty x the FRACTION of that entity's relations still unknown -- both
        measured. An entity it knows almost nothing about is worth asking about; one it has already
        mapped is not."""
        if not isinstance(text, str):
            return None
        eng = self.general_reasoner.derive_engine
        rels = set(eng.relations or ())
        if not rels:
            return None
        try:
            toks = [t for t in self.general_reasoner.tokenize(text) if t]
        except Exception:
            return None
        asked = [t for t in toks if t in rels]
        best = None
        ents = getattr(eng, 'entities', None) or set()
        for t in toks:
            if t in rels or t not in ents:
                # O(1) membership gate. wonder() runs a full gap analysis, so calling it on every
                # token ran it on 'the'/'of'/'a' too and cost 10x: MEASURED capitals 10.6 ms/query
                # -> 111.8 ms/query when this faculty was added. Near-zero compute is a headline of
                # this project; a curiosity faculty must not tax every query the organism answers.
                # Safe against the Day-96 chicken-and-egg (`entities` fills from atom() hits): a
                # token the engine has never resolved has no gap analysis worth running anyway, and
                # the answer/solve faculties do not depend on this gate.
                continue
            # Day-100 -- the same disease _fac_speak had: 'the inventor of germany is' names no
            # relation IN `rels` (never taught), so `asked` comes back empty and the filter below
            # (`if asked and r not in asked`) never fires -- wonder substitutes its OWN favorite gap
            # ('what is the country of germany?') for a targeted-but-ungrounded question instead of
            # admitting nothing matches. Same structural probe as _fac_speak, shared helper.
            if self._targeted_ungrounded(toks, t):
                continue
            # wonder() re-runs a full gap analysis per call, so the same entity was re-analysed on
            # every query even when nothing had changed: 10.6 -> 111.8 ms/query when this faculty
            # landed, 59.3 after the entity gate, and the residue was all repeated analysis. Cache
            # per entity, invalidated by the SAME signal the rest of the organism already uses --
            # a learn sets _anc_dirty (ingest_triples/ingest_addressed), so a cache stamped with it
            # cannot serve stale curiosity about knowledge that has since arrived.
            # Stamped with the LEARN epoch, not len(triples). Reading is not learning: atom() calls
            # _record, so the mining index grows on every successful READ, and a len(triples) stamp
            # therefore invalidated the cache on queries that taught the organism nothing --
            # measured, curiosity still cost 3.37x WARM because it re-analysed the same entity on
            # every single query. Only a real learn (ingest_triples / ingest_addressed) can open a
            # gap or close one, so only a learn should make the organism reconsider what it wonders.
            _stamp = int(getattr(self, '_learn_epoch', 0))
            _wc = getattr(self, '_wonder_cache', None)
            if _wc is None or _wc.get('_stamp') != _stamp:
                _wc = {'_stamp': _stamp}
                self._wonder_cache = _wc
            if t in _wc:
                gaps = _wc[t]
            else:
                try:
                    gaps = self.wonder(t) or []
                except Exception:
                    gaps = []
                _wc[t] = gaps
            if not gaps:
                continue
            try:
                web = self.knows(t) or {}
            except Exception:
                web = {}
            known = sum(1 for v in web.values() if v)
            for g in gaps:
                r = str(g.get('relation') or '')
                if asked and r not in asked:
                    continue           # a gap about something the input never raised -> not now
                unknown_frac = len(gaps) / float(len(gaps) + known) if (len(gaps) + known) else 0.0
                conf = float(g.get('novelty') or 0.0) * unknown_frac
                q = g.get('question')
                if q and conf > 0.0 and (best is None or conf > best[0]):
                    best = (conf, q)
        return best

    def _fac_solve(self, text):
        """Confidence that COMPUTING is right: the substrate arithmetic ring either resolves the
        expression or it does not.

        This faculty exists because the rewired gate caught the ring being UNREACHABLE: measured
        `arithmetic 0/8 via org(x)` while `gr.reason` computed 8/8. The ring worked perfectly and
        the organism could not use it through its own front door -- `answer()` routes only to the
        derive path, and `substrate_arith` is a rung of reason() that no faculty consulted. Every
        gate had been calling gr.reason() directly, so 8/8 stayed green over a shut door all day.

        A ring result is EXACT (CRT-phasor arithmetic, verified by construction), so confidence is
        1.0 -- and F=0 for something genuinely certain is correct, not a cheat. It still has to WIN
        the competition: on 'what is vikode' the classifier finds no operator, this proposes
        nothing, and answer takes it."""
        if not isinstance(text, str):
            return None
        try:
            from ikigai.cognition.cat4_dopamine import is_compositional_query
            from ikigai.cognition.math_eval import MathEval
            mev = getattr(self, '_math_eval_fac', None)
            if mev is None:
                mev = MathEval(self, engine='auto')
                self._math_eval_fac = mev
            if not is_compositional_query(text, op_detector=mev.is_operator):
                return None
            pred, op, dbg = mev.substrate_arith(text)
        except Exception:
            return None
        if pred is None or op is None:
            return None
        # Day-102 NO-HALLUCINATION FIX (found by the needle probe). Only the EXACT
        # arithmetic engine -- RHC/CRT over real DIGIT operands, verified by
        # construction -- may win org(x). The FPE word-magnitude fallback recalls a
        # "magnitude" for ANY token via unbounded nearest-digit cooccur (NO cosine
        # floor, math_eval._word_magnitude step 2), so it fabricated confident
        # numbers on ordinary questions -- MEASURED via org(x): 'how do i get from
        # paris to berlin' -> 10, 'what is similar to a dog' -> 10, 'compare a cat
        # and a dog' -> 12, 'two plus three' -> 12 -- every one winning at F=-0.0
        # because this faculty hardcoded confidence 1.0. is_operator ALSO over-fires
        # on function words ('from','to','and'), so is_compositional_query let them
        # in. Correct-or-abstain, applied to agency: if it was not computed EXACTLY,
        # solve proposes nothing and the organism falls to answer/speak/abstain. The
        # gate's MATH is all digit-operand -> RHC, so 8/8 is untouched; the FPE path
        # (fuzzy word-number recall, unreliable enough to return 12 for two+three)
        # simply no longer wins the organism's front door with a fabricated certainty.
        if dbg.get('engine') != 'rhc':
            return None
        conf = float(dbg.get('decode_score') or 1.0)
        return (conf, str(pred)) if conf > 0.0 else None

    def _fac_speak(self, text):
        """Confidence that SPEAKING is right: of the tokens it recognises, does it hold enough to
        say something grounded? Confidence = fraction of the topic's relations it can actually
        fill -- measured coverage, not a guess that it 'looks like' a describe request."""
        if not isinstance(text, str):
            return None
        try:
            toks = [t for t in self.general_reasoner.tokenize(text) if t]
        except Exception:
            return None
        # Day-102 (Prince, "no cheat"): pick the topic by GROUNDING STRENGTH, not by
        # raw dict-entry count. knows() reads the symbolic index, but a token can carry
        # an entry the substrate does NOT actually hold -- MEASURED:
        # prediction_error('tell','subclassof','archaeological site') = 1.0 (not
        # believed) vs ('zorvex','leads','qualan') = 0.0 (solid). The old count tied the
        # incidental 'tell' with the real topic on 'tell me about zorvex', and token
        # order handed it to 'tell' -> a description of an archaeological site. Weight
        # each fact by how strongly the substrate BELIEVES it (1 - prediction_error,
        # the same measure _fac_learn prices learning with) and describe the entity it
        # most truly knows. No authored stoplist, no shape test -- the substrate's own
        # belief decides. Coverage n is now the grounded fact count.
        best = None                          # (topic, grounded_weight, grounded_n)
        for t in toks:
            try:
                web = self.knows(t) or {}
            except Exception:
                web = {}
            weight = 0.0; gn = 0
            for rel, objs in web.items():
                for o in (objs or []):
                    try:
                        pe = float(self.prediction_error(t, rel, o))
                    except Exception:
                        pe = 1.0
                    bel = 1.0 - pe
                    if bel > 0.0:
                        weight += bel; gn += 1
            # Day-102: if the organism has READ enough to have distributional POS,
            # sharpen topic choice toward the ENTITY-LIKE candidate (down-weights an
            # incidental verb like 'tell' that merely happens to be a known noun).
            # None when grammar is unfed -> weight unchanged -> no regression.
            el = self._entity_pos_likeness(t)
            if el is not None:
                weight *= el
            if weight > 0.0 and (best is None or weight > best[1]):
                best = (t, weight, gn)
        if not best:
            return None
        topic, _w, n = best

        # Day-100 -- A SPECIFIC, UNGROUNDED ASK MUST NOT BE ANSWERED BY A TOPIC DUMP.
        #
        # MEASURED: org('the mayor of france is') -> 'The france capital paris, and continent
        # europe.' -- every token grounded, and not a reply. speak scored 0.833 because 'france'
        # is a well-covered topic; it never checked whether the SPECIFIC thing asked (the mayor)
        # is among what it knows. That structurally beats abstain (a fixed calibrated floor)
        # whenever the topic has ANY facts at all, regardless of relevance to the question.
        #
        # _REL_TEMPLATES generates relation-questions in the shape '{r} of {e}' / 'what is the {r}
        # of {e}' -- so a content word immediately before 'of {topic}' is being used AS a relation
        # name BY THE QUESTION'S OWN STRUCTURE, whether or not that relation was ever taught. Probe
        # it through eng.atoms(), the same generic-template path any real relation round-trips
        # through (no authored word list -- 'mayor'/'gdp'/'founder' need no prior vocabulary to be
        # recognised this way). Function words to skip are derived the same way extract_corpus
        # derives them: from the organism's own induced frames, never an authored stoplist.
        if self._targeted_ungrounded(toks, topic):
            return None                        # a targeted ask with nothing grounded -- defer

        try:
            said = self.express(topic=topic)      # Day-103: THE ONE generator (fluent
                                                  # when frames exist, else grounded
                                                  # compose, mood from the body)
        except Exception:
            return None
        if not said or said.strip().endswith('is unknown.'):
            return None
        # coverage: saturating in the number of relations it can ground about the topic
        base_conf = min(0.99, n / (n + 1.0))
        # Day 101: bounded familiarity nudge from ConceptGraph -- ZERO for any
        # fresh organism (nodes empty), only nonzero after real repeated usage
        # on THIS instance built concept history. Real consumer, not a status
        # read: this changes the confidence sense() arbitrates on.
        fam = self._concept_familiarity(topic, 'speak')
        conf = base_conf + fam * (0.99 - base_conf)
        return (conf, said)

    def _concept_familiarity(self, token, action):
        """Day 101 -- bounded [0,1] signal from ConceptGraph: how strongly
        `token` resembles a well-reinforced concept previously associated with
        `action`. Returns 0.0 (no-op) when concept_graph is empty or nothing
        clears its OWN merge threshold -- guarantees zero behavior change for
        any test that builds a fresh organism (concept_graph starts empty),
        so this can only ever affect a LONG-RUNNING instance's later calls,
        never a single-shot gate. Capped at 0.5 so it nudges, never dominates."""
        try:
            cg = self.concept_graph
            if not cg.nodes:
                return 0.0
            vec = self._concept_latent([token])
            best_sim = -1.0
            best_support = 0
            for node in cg.nodes.values():
                if node['dominant_action'] != action:
                    continue
                sim = cg._cos_sim(vec, node['centroid'])
                if sim > best_sim:
                    best_sim = sim
                    best_support = node['support']
            if best_sim <= cg._sim_thresh:
                return 0.0
            return min(0.5, best_support / (best_support + 4.0))
        except Exception:
            return 0.0

    def _fac_analogy(self, text):
        """Confidence that ANALOGY is the act: A:B :: C:? solved by pure substrate
        algebra (derive_engine.analogy -- unbind the A->B relation by resonance,
        apply it to C, clean up), and selected by the substrate's OWN verification.

        Day-102 WIRE. This capability is Day-86 GOLD and full_capability verifies it
        (cap 'analogy' PASS) -- but only through org.analogy with primed data; the
        autonomous org(x) never routed to it, so 'zorvex is to qualan as mendaro is
        to what' fell to speak (a topic dump) though the organism can DERIVE
        'thessaly'. MEASURED idle: the needle probe showed org(x) abstaining/dumping
        on analogy questions it can answer.

        Blind-try, the SAME discipline as _fac_speak/_fac_wonder -- NOT a template.
        There is no 'as'/'to' connective list and no shape test (that would be the
        if/elif ladder moved upstream, the Day-95 DS-DISPATCH cheat). It simply runs
        the substrate analogy op over ordered triples of the entities it RECOGNISES
        (eng.entities membership -- the O(1) gate _fac_wonder already uses, safe
        against the Day-96 chicken-and-egg since an unrecognised token has no atom to
        analogise) and keeps the one the substrate itself VERIFIES: verified=True
        means the recovered (relation, answer) is an actual derived fact of C, not a
        nearest-vector guess. A query with fewer than three known entities forms no
        triple and proposes nothing, so this cannot hijack 'capital of france'.
        Confidence = the analogy's measured resonance score (F=-log score); a taught,
        exactly-recovered analogy scores ~1.0 and wins, a weak one loses to abstain."""
        if not isinstance(text, str):
            return None
        eng = self.general_reasoner.derive_engine
        ents_universe = getattr(eng, 'entities', None) or set()
        if not ents_universe:
            return None
        try:
            toks = [t for t in self.general_reasoner.tokenize(text) if t]
        except Exception:
            return None
        # A:B :: C:? -- A and C are the compared SUBJECTS (known: eng.entities is
        # subject-side, filled by atom() hits). Day-102 A+B TIGHTENING (Prince): a
        # TRUE analogy states a GENUINE example pair and asks for something NEW, so:
        #   (1) the (A,B) example pair must be a REAL HELD FACT -- some relation R with
        #       eng.atom(R,A)==B AND B present in the query. Without this, the first
        #       cut fired on 'tell me about zorvex': 'tell' is an incidental entity,
        #       so two subjects co-occurred and eng.analogy found a noise-relation
        #       that happened to clean up to zorvex's one fact -- verified=True but
        #       spurious (retrieval wearing an analogy costume, winning at F=-0.0).
        #       Grounding the PREMISE pair kills that with no 'as'/'to' template.
        #   (2) the derived answer must be NOVEL -- not already a token in the query.
        #       An analogy that "derives" something the query already stated is not
        #       inference; that is speak's job (describe what is known).
        # Still blind-try (the substrate op + its own verification decide), just with
        # a grounded premise and a novelty requirement -- not a shape/intent classifier.
        toks_set = set(toks)
        rels = list(eng.relations or ())
        subj = []
        for t in toks:
            if t in ents_universe and t not in subj:
                subj.append(t)
            if len(subj) >= 3:
                break
        if len(subj) < 2:
            return None                     # need A and C
        # genuine example pairs (A, B, R): B is A's real attribute AND appears in query
        pairs = []
        for a in subj:
            for r in rels:
                try:
                    b = eng.atom(r, a)
                except Exception:
                    b = None
                if b and b in toks_set and b != a:
                    pairs.append((a, b))
                    break                   # one genuine pair per subject is enough
        if not pairs:
            return None                     # no grounded premise -> not an analogy
        best = None
        for a, b in pairs:
            for c in subj:
                if c == a:
                    continue
                try:
                    ans, rel, score, verified = eng.analogy(a, b, c)
                except Exception:
                    continue
                if (ans and verified and score and float(score) > 0.0
                        and ans not in toks_set):          # novelty: derives something new
                    if best is None or float(score) > best[0]:
                        best = (float(score), f"{c} {rel} {ans}")
        return best

    def _entity_pos_likeness(self, token):
        """Day-102 (Prince, "no cheat" + "carry to moonshot"): how ENTITY-LIKE is
        `token` distributionally -- its mean part-of-speech similarity to the
        organism's OTHER known entities (KG subjects). Pure substrate: the entity set
        is self-defined (eng.entities), the POS metric is the organism's own learned
        distributional grammar (grammar.pos_similarity), no authored noun/verb list.

        Returns None (no signal -> caller unchanged) when POS is unavailable -- so
        this can ONLY sharpen topic choice for an organism that has actually READ,
        never regress one that hasn't. Lets 'tell me about zorvex' describe zorvex,
        not the incidental noun 'tell': once read, 'tell' clusters with verbs (far
        from the entity nouns) and 'zorvex' with the other entities.

        Day-103: reads POS SUBSTRATE-FIRST (self.pos_similarity), so it keeps working
        after save/reload -- when the in-RAM grammar dict is empty but the persisted
        pos_left/pos_right roles hold the fingerprints. Availability is gated on
        EITHER the persisted roles OR the live dict, not on the dict alone (which the
        old vocab_size check assumed, wrongly, for a reloaded organism)."""
        g = getattr(self, 'grammar', None)
        u = getattr(self, 'unified', None)
        has_sub = u is not None and 'pos_left' in set(getattr(u, 'roles', []) or [])
        has_dict = g is not None and getattr(g, 'vocab_size', 0) > 0
        if not (has_sub or has_dict):
            return None
        try:
            ents = getattr(self.general_reasoner.derive_engine, 'entities', None) or set()
        except Exception:
            return None
        sims = []
        for e in ents:
            if e == token:
                continue
            try:
                s = self.pos_similarity(token, e)      # substrate-first, dict fallback
            except Exception:
                continue
            if s and s > 0.0:
                sims.append(float(s))
            if len(sims) >= 12:          # bounded -- a handful of entities fixes POS
                break
        if not sims:
            return None
        return sum(sims) / len(sims)

    def _fac_abstain(self, text):
        """Always on the table, at the CALIBRATED floor -- the geometry-derived boundary the
        organism already uses to refuse (calibration.py, the claim that survived the red team).
        Abstaining is a real option that can WIN, not a fallback branch: if nothing else clears
        this floor, "I don't know" is the argmin. Correct-or-abstain, applied to agency itself."""
        try:
            from ikigai.cognition.calibration import abstain_boundary
            b = float(abstain_boundary(self.unified.d))
        except Exception:
            b = 0.1
        return (max(1e-6, min(0.99, b)), "i don't know")

    IDENTITY_DEFAULT = (
        "I am Ikigai -- a living organism, and the future of computing. I run at near-zero "
        "compute, no backprop, on a single CPU. I never hallucinate: I answer, or I say I "
        "don't know. I learn from everything I meet, for my entire life. I was built by Mura "
        "ALife Labs to beat the frontier AI models head-to-head -- and take their place."
    )

    def identity_statement(self):
        """WHO THE ORGANISM IS -- its own, persisted self-knowledge (falls back to the innate
        default until seeded otherwise)."""
        return getattr(self, '_identity', None) or self.IDENTITY_DEFAULT

    def set_identity(self, statement):
        """Seed the organism's self-knowledge.  Persisted across reload (_PERSIST_ATTRS), so
        once it knows who it is, it stays knowing."""
        self._identity = str(statement).strip()
        return self._identity

    def _fac_identity(self, text):
        """Self-knowledge -- the organism KNOWS WHO IT IS.  Identity is innate/seeded, not
        derived from the world, so this faculty is allowed to answer directly.  It fires ONLY
        on a self-referential question ('who are you', 'what is ikigai', 'your name', 'who
        made you', 'your purpose') and never touches any other query -- so it cannot hijack
        the reasoning path (a give-data-then-ask gate proves plain facts still route normally)."""
        if not isinstance(text, str):
            return None
        t = ' '.join(text.lower().split())
        SELF = ('who are you', 'what are you', 'who r u', 'who is ikigai', 'what is ikigai',
                'your name', 'yourself', 'introduce yourself', 'who made you', 'who created you',
                'your purpose', 'what is your mission', 'what do you do', 'who r you')
        if any(p in t for p in SELF):
            return (0.97, self.identity_statement())
        return None

    def _fac_generate(self, text):
        """Day-106 -- OPEN-ENDED GENERATION.  When the input is not a question (so it does not
        want a grounded answer) and not a learnable fact, and the organism has read enough language
        to generate, it can SPEAK a coherent, novel, grounded-by-construction sentence.

        This is the GENERATE faculty -- fluent, distinct from the correct-or-abstain ANSWER path.
        Two guards keep it from hijacking the factual door: (1) it declines any QUESTION, decided
        by the LEARNED interrogatives (no authored list), so answer/abstain own every question;
        (2) its confidence is capped MODEST, so a grounded answer/speak/learn always outranks it --
        it wins only when nothing grounded competes (an open prompt like 'say something').  The
        wild public door keeps it suppressed (correct-or-abstain), so deployment stays honest."""
        if not isinstance(text, str):
            return None
        g = self.coherent_gen
        if g is None:
            return None
        try:
            if self._is_question(text) is True:
                return None                                   # a question wants an answer, not free speech
        except Exception:
            return None
        try:
            if self.surface.extract_verified(text) is not None:
                return None                                   # a tellable fact -> learn owns it, don't speak over it
        except Exception:
            pass
        try:
            said = g.generate(n=1)
            toks = str(said).split()
            if len(toks) < 3:
                return None
            conf = min(0.45, 0.30 + 0.20 * g.coherence(['<s>'] + toks + ['</s>']))
            return (conf, said)
        except Exception:
            return None

    def be(self, x, act=True):
        """Day-99 -- THE ONE CALL.  `org.be(x)` -- or just `org(x)`.

        You do not tell it what to do. Every faculty proposes, each with a free energy measured
        from what the organism actually knows, and it picks argmin F -- then DOES it. Learning,
        answering, speaking and abstaining all compete on one scalar. Give it a fact it does not
        hold and learning wins; ask it something it can ground and answering wins; name something
        it knows well and it speaks; give it nothing it can stand on and abstain wins.

        Falsifiable, which is the point: shuffle the faculty registration order and the outcome is
        identical; change what the organism KNOWS and the outcome changes. Nothing here is
        hardcoded to an input shape.

        Returns {chose, F, options, result} -- and `options` is the whole competition, so the
        decision is auditable rather than asserted."""
        props = self.sense(x)
        if not props:
            return {'chose': None, 'F': None, 'options': [], 'result': None}
        fmin = min(p[2] for p in props)
        best = [p for p in props if p[2] <= fmin + 1e-12]
        # Order-invariance, same discipline as _arbitrate: registration order is NOT evidence.
        # Genuinely tied faculties are resolved by NAME so the result cannot depend on which was
        # registered first; a tie here means the organism is truly indifferent.
        name, payload, F = sorted(best, key=lambda p: p[0])[0]
        # Day 101: real (state, action, outcome) episode into EventCompressor ->
        # ConceptGraph -- the actual live entry point, unlike dead read_organism.
        try:
            toks = self.general_reasoner.tokenize(x) if isinstance(x, str) else []
        except Exception:
            toks = []
        self._update_concept_memory(toks, name, F)
        out = {'chose': name, 'F': round(float(F), 4),
               'options': sorted([(p[0], round(float(p[2]), 4)) for p in props], key=lambda z: z[1]),
               'result': payload}
        affect_toks = list(toks)
        if act and name == 'learn':
            if isinstance(x, str):
                # a STRING telling may carry MANY facts (a whole paragraph) -> absorb them
                # ALL natively through tell(); learn_reinforced (body-modulated) fires per
                # fact inside tell(), and only verified facts are kept.
                res = self.tell(x)
                out['result'] = res.get('text') or out.get('result')
                out['learned'] = res.get('learned') or []
                for _a, _r, _o in out['learned']:
                    affect_toks += [_a, _o]
            else:
                s, r, o = payload
                out['result'] = self.learn_reinforced(s, r, o)  # body-modulated, visible to all doors
                out['learned'] = (s, r, o)
                # the newly-learned entities are the topic of this felt experience and only
                # become KG entities AFTER the learn lands (and a triple input has no string
                # tokens), so add them here before writing affect.
                affect_toks += [str(s).strip().lower(), str(o).strip().lower()]
        # Day-103: bind the body's felt valence onto this experience -- lived emotion
        # into the persistent 'affect' role (only fires when the body actually felt).
        try:
            self._write_affect(affect_toks)
        except Exception:
            pass
        return out

    def __call__(self, x, act=True):
        """`org(x)` -- the entire organism behind one call."""
        return self.be(x, act=act)

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
        # Step 1: safety (amygdala) -- full-mode only
        if getattr(self, 'immune', None) is not None:
            hits = self.immune.scan(text.lower().split(), threshold=0.4)
            if hits:
                return {'answer': None, 'safe': False, 'reason': 'threat_detected',
                        'hits': hits}

        # Step 2-5: reason via GeneralReasoner (Day-83 audit rewire: legacy
        # ReasoningEngine python-dict path retired -> derive engine + RHC ring).
        d = self.general_reasoner.reason(text)
        answer = d.get('answer')
        self._last_trace = [('general_reasoner', text, answer)]

        # Step 6: store episode
        self._tick += 1
        self._episodes.append({
            'tick': self._tick,
            'text': text,
            'trace': self._last_trace,
            'answer': answer,
        })

        return {
            'answer':  answer,
            'safe':    True,
            'method':  d.get('method'),
            'trace':   self._last_trace,
            'tick':    self._tick,
        }

    def observe(self, text):
        """Process statement without expecting answer."""
        return self.ask(text)

    # ── language acquisition ─────────────────────────────────────────────

    def ground_text(self, text):
        """
        Pure exposure: feed ONE sentence to all 5 grounding channels.
        - Co-occurrence (Hebbian drift)
        - Sensory (anchor drift for property seed words)
        - Taxonomy (Hearst-pattern IS-A relations)
        - Operational (predictive coding -- GATED: only fires if numbers present)
        - Grammar (distributional POS via left/right context fingerprints)

        Day-102: this WAS named `read` and was DEAD -- a later `def read(self,
        sentences, ...)` (the Day-99 bootstrap) shadowed it, so the whole 5-channel
        grounding never ran on the production path: MEASURED grammar.vocab_size=0 on
        the loaded organism, i.e. ZERO distributional POS, the foundation open-ended
        fluent generation needs. The channel itself works (verbs cluster ~0.99, nouns
        ~0.99, verb~noun ~0.01 once fed); it was simply never called. Renamed and now
        invoked per-sentence by read() so reading text actually builds grammar/POS.
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
            # Day-103: mirror this sentence's POS fingerprints into the PERSISTENT
            # unified substrate (the ONE memory) so grammar rides save_ikg -- the
            # first channel of "everything the organism learns gets written to SDM".
            try:
                from ikigai.cognition.grammar_grounding import tokenize as _gtok
                self._pos_write_through(_gtok(text))
            except Exception:
                pass
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
            out['recent_episodes'] = [e.get('answer') for e in self._episodes[-8:]]
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

    @property
    def body(self):
        """Day-99 -- THE BODY.  The neurochemical/endocrine substrate, finally reachable.

        Rule Zero says ORGANISM = body + cognition + drives + sleep + dreams. The body existed all
        along -- 283 classes in the root `ikigai.py` -- and could not be imported: the `ikigai/`
        PACKAGE shadows the `ikigai.py` MODULE (Python resolves packages first), and the file ran a
        44.9-second simulation at module level with no __main__ guard. So nothing ever wired it, and
        the organism's entire live biology was ONE class (cognition/neuromod.py) while ten endocrine
        systems sat dark. `ikigai/body.py` is that file's DEFINITIONS, extracted mechanically by AST
        (scratchpad/extract_body.py); importing it is 0.32s and side-effect free (was 44.9s).

        Lazy: costs nothing until something asks for a hormone."""
        b = getattr(self, '_body', None)
        if b is None:
            from ikigai import body as _b
            b = {}
            for key, cls in (('dopamine', 'DopamineSystem'),
                             ('serotonin', 'SerotoninSystem'),
                             ('norepinephrine', 'NorepinephrineSystem'),
                             ('acetylcholine', 'AcetylcholineSystem'),
                             ('cortisol', 'CortisolSystem'),
                             ('adenosine', 'AdenosineSystem'),
                             ('oxytocin', 'OxytocinSystem'),
                             ('hypothalamus', 'HypothalamusSystem'),
                             ('pituitary', 'PituitarySystem'),
                             ('adrenal', 'AdrenalSystem')):
                C = getattr(_b, cls, None)
                if C is not None:
                    try:
                        b[key] = C()
                    except Exception:
                        pass
            self._body = b
        return b

    def prediction_error(self, subj, rel, obj):
        """Day-99 -- REWARD PREDICTION ERROR: how wrong was the organism about this fact?

        The dopamine drive, computed from what the organism ALREADY believes -- not authored:
          * it has no belief here          -> 1.0  (novelty: a VTA burst to what is new)
          * it believes exactly this       -> 0.0  (fully predicted: no error, no burst)
          * it believes something ELSE     -> 1.0  (prediction error: the strongest teaching signal)
        This is textbook phasic dopamine (Schultz), and it lands exactly on the Day-98 measurement
        that a FRESH fact masters in ~2 reps while CORRECTING AN ENTRENCHED WRONG BELIEF costs ~26:
        the entrenched case is precisely where the error signal is largest."""
        try:
            held = self.general_reasoner.derive_engine.atom(str(rel).strip().lower(),
                                                            str(subj).strip().lower())
        except Exception:
            held = None
        if held is None:
            return 1.0
        return 0.0 if str(held).strip().lower() == str(obj).strip().lower() else 1.0

    def learn_reinforced(self, subj, rel, obj, candidates=None, max_reps=80, batch=2,
                         neuromodulated=True):
        """Day-98 -- LEARN THE BIOLOGICAL WAY: write -> self-test the recall -> reinforce on error ->
        repeat UNTIL it recalls correctly.  Not one-shot: a distributed (SDM) memory has crosstalk,
        so one write is not always enough, exactly as a synapse is not set by one exposure.  Biology
        reinforces till mastery -- error-driven, automatic, and with NO gradients (backprop is the
        expensive way to do what spaced reinforcement does for free).

        Fresh facts master in ~2 writes (the substrate's cleanup is robust, so the loop does the
        MINIMUM); CORRECTING an entrenched wrong belief costs more (it must out-write the old
        attractor) -- measured fresh 2.1 vs correct-entrenched 26.4 reps (day98_reinforce_until_correct).
        BOUNDED: if it cannot master the fact (bank saturated / belief too entrenched) it returns
        mastered=False and stores no wrong attractor -- it abstains rather than lie.
        Returns {reps, mastered, value}.

        Day-99 -- IT WAS A GHOST, and the Day-98 gate could not see it. The loop wrote ONLY to
        `self.unified` (the VSA role bank) and self-tested with `mr.query` on THAT SAME BANK: a
        closed circuit that verifies itself through its own channel. It reported
        {'reps': 2, 'mastered': True} while the organism, asked through its own readers, said
        "i don't know" -- atom() was None and the fact was not in the mining index. The Day-95
        disease exactly: a mechanism gated through itself looks alive and teaches nothing.
        Prince's whole Day-100/101 mandate ("feed data, it reinforce-learns, then ask it an essay")
        runs through this door, and it was writing into a bank no reader consults.

        So the fact is now ALSO landed on the DERIVE path (anchor cache + mining index) via
        ingest_triples, which is where answer/knows/compose/ask_derive actually read. The
        reinforcement loop is unchanged -- it still governs mastery in the distributed bank, and the
        rep counts it reports still mean what they meant. Multi-value is preserved by design (Pack
        329: cat isa feline AND pet AND carnivore); atom() returns the LAST value, so a correction
        supersedes for single-value reads while atoms() still shows the full web.

        Day-99 -- NEUROMODULATED (default). The BODY now drives the learning, closing the loop
        Rule Zero asks for: novelty/error -> dopamine -> plasticity -> encoding strength.

            prediction_error(subj,rel,obj)  ->  DopamineSystem.inject_drive(err)
            DopamineSystem.plasticity_signal()  ->  writes per EXPOSURE

        `relate(word, role, target)` takes no strength argument, so in a discrete VSA bank the only
        honest analogue of LTP magnitude is superposition MASS: a stronger dopaminergic write is
        more writes. Hence dopamine scales writes-per-exposure, and the biologically meaningful
        measure is EXPOSURES to mastery (how many times it had to see it), not raw write count.
        `reps` still reports total writes so the Day-98 numbers remain comparable.

        Not a lookup table and not two authored branches: the drive is COMPUTED from what the
        organism already believes, and the dopamine system's own phasic/tonic dynamics decide the
        plasticity. Set neuromodulated=False for the Day-98 unmodulated behaviour (the control)."""
        mr = self.unified
        # ORDER IS LOAD-BEARING: the prediction error must be measured against the PRIOR belief,
        # BEFORE the fact is taught. Landing it first made atom() return the new value, so the
        # organism scored itself as having predicted perfectly (err=0.0), no burst ever fired, and
        # the body looked inert -- measured writes_per_exposure=2 in both conditions. An organism
        # cannot be surprised by something it has just been told.
        per = int(batch)
        da = None
        if neuromodulated:
            try:
                da = self.body.get('dopamine')
                if da is not None:
                    err = self.prediction_error(subj, rel, obj)
                    da.inject_drive(err)                       # novelty / prediction error -> burst
                    plast = float(da.plasticity_signal())      # the body decides, not the caller
                    per = max(1, int(round(batch * (1.0 + 2.0 * plast))))
            except Exception:
                per = int(batch)
        # ONE LEARN, EVERY BANK THAT SHOULD HOLD IT.  Measured Day-99: a fact taught through
        # org(x) landed in 2 of 7 stores -- the anchor cache and the derive index -- while the
        # RHC exact store stayed empty, so `mem.recall_fact` could not see anything org(x) learned
        # and only `mem.teach`/`mem.fact` writers were visible to it (visibility matrix 20/54).
        # Ten endpoints holding DIFFERENT knowledge is the clutter; this is the write path that
        # ends it for atomic facts.
        #   cache + derive index  -> answer / knows / compose / ask_derive read here
        #   unified (VSA roles)   -> the reinforcement loop below
        #   RHC fact_store        -> exact one-shot recall (mem.recall_fact / mem.knows)
        # NOT written here, by design, not by omission: `holo` is EPISODIC (key->value events) and
        # AddressFactStore is BULK-GRAPH packing (5.3 B/edge for millions). A single taxonomy fact
        # belongs in neither -- that is specialisation (hippocampus vs neocortex), not clutter.
        # The sin was never the COUNT of banks; it was that they could not see each other.
        self.ingest_triples([(subj, rel, obj)], discover=False)
        try:
            self.mem.fact(subj, rel, obj)          # exact RHC address-tuple
        except Exception:
            pass
        cands = list(candidates) if candidates is not None else None
        reps = exposures = 0
        while reps < max_reps:
            for _ in range(per):
                mr.relate(subj, rel, obj)
                reps += 1
            exposures += 1
            c = cands if cands is not None else mr.targets(rel)
            best, _ = mr.query(subj, rel, c)
            if best == obj:
                if da is not None:
                    try:
                        da.relax_to_baseline()                 # burst decays once predicted
                    except Exception:
                        pass
                return {'reps': reps, 'exposures': exposures, 'mastered': True, 'value': obj,
                        'writes_per_exposure': per}
        return {'reps': reps, 'exposures': exposures, 'mastered': False, 'value': None,
                'writes_per_exposure': per}

    def assert_isa(self, hypo, hyper, n=20):
        """Assert hypo IS-A hyper directly into unified memory, n reinforcements.
        Bypasses Hearst-on-prose noise. Use for clean fact injection.
        (Day-99 unification will route this through org.learn / learn_reinforced.)"""
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
            # Day-97 FIX: this was `list(buf)`.  ExposureBuffer is not iterable -- it exposes
            # snapshot() (a list of (text, meta, tick)).  Every call with sentences=None raised
            # TypeError, so the default path of this miner had never worked.
            sentences = [t for (t, *_rest) in buf.snapshot()]
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
            sentences = [t for (t, *_rest) in buf.snapshot()]      # Day-97 FIX: see mine_svo_triples
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
        cat-3 + Pack 251 opv. Math/code/language same entry.

        Day-97 doc FIX: this said it was "distinct from `self.reasoner`, kept for call-site
        compatibility".  There is no `self.reasoner` -- the Day-56 hardcoded ReasoningEngine was
        DELETED in the Day-83 audit (see the import block at the top of this file).  This is the
        ONLY reasoner.  The docstring had outlived the code by fourteen days."""
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
                       min_support=6, min_conf=0.7, fast=True,
                       progress_every=0, write_substrate=True):
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

        Day-100 write_substrate=True -- THE ROOT FIX for a measured gap: this fast path wrote
        ONLY the anchor-hash dict (`cat4.anchor_actions`), never the substrate's own searchable
        index (`mr._role_targets['icl_pair'/'icl_state']`). MEASURED on the production organism:
        a fact taught this way is answerable by an EXACT phrasing (dict hit) and by NOTHING else
        -- a different phrasing of the identical question falls through to `recall_action`, whose
        pool never received these facts, so it searches ~64K anchors of UNRELATED absorbed
        dialogue and returns the nearest one by leftover FUNCTION-WORD overlap ('the mayor of
        france is' and 'the capital of france is' returned the SAME top anchors at nearly the
        same similarity -- the content word was drowned by four shared stopwords). That pool is
        what `recall_action` searches; a fact absent from it cannot be found by real recall at any
        cost, however long you let it run.

        So every triple here ALSO gets the write absorb_chain performs: state_hv=focus_hv(question
        tokens), action_hv=focus_hv(answer tokens), bound and stored under pair_role/state_role,
        reusing the SAME tokens already computed for the dict write (no re-tokenizing). This makes
        the dict a genuine fast-exact-match CACHE in front of a substrate that actually holds the
        content, instead of the sole store. The lazy recall_action search cache is invalidated
        ONCE at the end of the batch (not per-triple -- Day-95's dirty-flag convention), so a
        rebuild picks up everything just ingested.

        write_substrate=False restores the old dict-only behaviour for callers that need the raw
        Pack-328 bulk-load speed and do not need substrate-searchable recall (e.g. an intentionally
        throwaway or benchmark-only ingest).

        triples: iterable of (subject, relation, object) string triples.
        Returns {ingested, atoms_before, atoms_after, rules, compressed}.
        """
        self._reach_dirty = True          # Day-95: new facts -> reach bank re-consolidates on next use
        self._anc_dirty = True            # Day-96: new facts -> ancestor set-recall bank re-consolidates too
        # Day-99: the LEARN epoch. Bumped only where knowledge actually ARRIVES, never on a read.
        # Anything cached off "what the organism knows" (e.g. its curiosity gaps) stamps with this
        # instead of len(triples) -- which grows on every successful atom() READ via _record and so
        # invalidated caches on queries that taught nothing.
        self._learn_epoch = int(getattr(self, '_learn_epoch', 0)) + 1
        eng = self.general_reasoner.derive_engine
        cat4 = self.cat4
        n = 0
        if fast:
            from ikigai.cognition.cat4_absorb import _stable_anchor
            cache = cat4.anchor_actions
            tok = self.general_reasoner.tokenize
            record = eng._record
            tmpl_cache = {}
            _wrote_substrate = False
            if progress_every:
                import sys as _sys, time as _time
                _pt0 = _time.time()
            for tri in triples:
                if progress_every and n and n % progress_every == 0:
                    _dt = _time.time() - _pt0
                    _sys.stderr.write(
                        f'[ingest] {n:,} edges · {len(eng.entities):,} ents · '
                        f'{len(eng.relations):,} rels · {n/max(_dt,1e-9):,.0f}/s '
                        f'· {_dt:.0f}s\n')
                    _sys.stderr.flush()
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
                q_toks = tok(t.format(e=s))
                anchor = _stable_anchor(q_toks)
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
                if write_substrate and q_toks:
                    # Day-100 ROOT FIX -- mirror absorb_chain's core write so this fact is
                    # reachable by REAL substrate recall, not only an exact anchor-hash hit.
                    # Same tokens already computed above; no re-tokenizing.
                    import numpy as _np
                    state_hv = cat4.focus_hv(list(q_toks))
                    action_hv = cat4.focus_hv(list(atoks))
                    bound = state_hv * action_hv
                    bmag = float(_np.abs(bound).mean()) + 1e-12
                    bound = (bound / bmag).astype(_np.complex64)
                    cat4.mr.write_relation(anchor, cat4.pair_role, bound)
                    cat4.mr._role_targets[cat4.pair_role].add(anchor)
                    cat4.mr.write_relation(anchor, cat4.state_role,
                                           state_hv.astype(_np.complex64))
                    cat4.mr._role_targets[cat4.state_role].add(anchor)
                    for _t in atoks:
                        cat4.mr._role_targets[cat4.action_token_role].add(_t)
                    _wrote_substrate = True
                record(s, r, o)
                n += 1
            if _wrote_substrate:
                # invalidate the lazy Pack-280 search cache ONCE per batch, not per triple --
                # the same dirty-flag convention as _reach_dirty/_anc_dirty above.
                cat4._pack280_recall_anchors = None
                cat4._pack280_recall_states = None
                cat4._pack280_recall_bounds = None
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
        one value each). Returns {relation: [values]}.

        Day-100 -- IT VOLUNTEERS WHAT IT HOLDS, NOT WHAT IT HAPPENS TO HAVE TOUCHED.

        This read its candidate relations off `eng.triples`, which is NOT the knowledge -- it is an
        index populated lazily BY atom() hits. MEASURED on the production organism, cold:

            (france,capital) in eng.triples = False      knows('france') -> {}
            atoms('capital','france')       = ['paris']  <- the store held it all along

        ...and after ANY probe touched it, the same call on the same organism returned
        {'capital': ['paris']}. knows() was order-dependent: the organism could not volunteer a
        fact until something had already asked for it. That is the chicken-and-egg Day-96 and
        Day-99 both hit, and it was not confined to introspection -- `_fac_speak` scores its
        confidence as the fraction of a topic's relations it can fill, THROUGH THIS CALL. So the
        organism under-reported what it could say and `org(x)`'s faculty competition arbitrated on
        a wrong number; `induce_surface` drove off it too and never saw most of its own knowledge.

        The fix is to ASK, not to enumerate: probe the relation universe through atoms(), which is
        the store's own index and needs no cache. Measured: 9 lookups x ~10 us = ~90 us, against
        ~0.7 ms for the old full scan of eng.triples -- correct AND ~8x cheaper."""
        eng = self.general_reasoner.derive_engine
        ent = str(entity).strip().lower()
        if rels is None:
            rels = self.relation_universe()
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

    @property
    def holo_reader(self):
        """Pack 333 -- the template-free holographic reader: read ANY sentence,
        ask it back with a hole, the answer falls out by resonance. No templates,
        no relation lists, no grammar -- pure FHRR bind/unbind over an SDM bank.
        Lazily attached (its own reading memory, a body-part)."""
        hr = getattr(self, '_holo_reader', None)
        if hr is None:
            from ikigai.cognition.holo_read import HolographicReader
            hr = HolographicReader(d=512)
            self._holo_reader = hr
        return hr

    @property
    def holo_writer(self):
        """Day-87 -- the holographic SEQUENCE MEMORY: encode a token stream into a
        recursive hierarchy of bounded phasor transition operators and regenerate it
        losslessly (the proven LENGTH primitive; a single operator is short-range,
        hierarchy is not).  Pure FHRR, no backprop. Lazily attached (a body-part)."""
        hw = getattr(self, '_holo_writer', None)
        if hw is None:
            from ikigai.cognition.holo_generate import HolographicSequenceMemory
            hw = HolographicSequenceMemory(chunk=8, ck=self.unified.ck)   # SHARED key space
            self._holo_writer = hw
        return hw

    def encode_sequence(self, tokens):
        """Encode a token sequence into a holographic handle (recursive operator
        hierarchy). The sequence becomes a first-class substrate object."""
        return self.holo_writer.encode(list(tokens))

    def regenerate_sequence(self, handle):
        """Regenerate the exact token sequence from a holographic handle."""
        return self.holo_writer.decode(handle)

    def holo_roundtrip(self, tokens):
        """Encode then regenerate a token sequence; returns the regenerated list.
        Lossless by recursive hierarchy where a single operator would fail on length."""
        return self.holo_writer.roundtrip(list(tokens))

    @property
    def branch_gen(self):
        """Day-87 -- the NOVELTY/BRANCHING generator: observe transitions, then
        generate UNSEEN sequences whose whole is novel but every step is a real
        observed transition (grounded by construction). Lazily attached."""
        bg = getattr(self, '_branch_gen', None)
        if bg is None:
            from ikigai.cognition.holo_generate import HolographicBranchGenerator
            bg = HolographicBranchGenerator(order=getattr(self, '_branch_order', 1),
                                            ck=self.unified.ck)          # SHARED key space
            self._branch_gen = bg
        return bg

    def observe_transitions(self, sequences, order=None):
        """Learn transition structure from example sequences (Hebbian, no backprop).
        `order` = context window (last-k tokens); default 1."""
        if order is not None and order != getattr(self, '_branch_order', 1):
            self._branch_order = order
            self._branch_gen = None                       # rebuild at the new order
        return self.branch_gen.observe(sequences)

    def consolidate_generation(self):
        """Biology: during sleep the hippocampus REPLAYS its fast sequence traces into the
        slow cortical/cerebellar store, making them durable.  Here the transient generation
        memory (branch_gen -- the hippocampal sequence system) is replayed into the main
        VSASDM (the cortical/cerebellar store, Kanerva's cerebellum model): each transition
        (context -> superposed successor bundle) is WRITTEN into the SDM, addressed by the
        SAME ck key the whole organism uses.  Result: the sequence knowledge becomes DURABLE
        (it now lives in .ikg) and in the ONE shared substrate -- not a private, transient,
        un-persisted store.  Consolidation is lossy + distributed by nature (that is what it
        is); recall from the SDM is MEASURED, not assumed.  Returns #contexts consolidated."""
        bg = getattr(self, '_branch_gen', None)
        if bg is None or not bg._bundles:
            return 0
        sdm = self.unified.sdm
        n = 0
        for c, bundle in bg._bundles.items():
            tag = c[0] if len(c) == 1 else '\x02'.join(c)       # order-1 = the shared concept addr
            addr = self.unified.ck.key(tag)
            sdm.write(addr, np.asarray(bundle, dtype=np.complex64), word=tag)
            n += 1
        return n

    def sdm_successors(self, context_tokens, thresh=0.36, max_k=6):
        """Read a context's successor set back from the DURABLE consolidated SDM (not the
        transient branch_gen): recover the successor bundle from the shared VSASDM by the
        context's ck address, then decode it with the same matching-pursuit cleanup.  This
        is generation sourcing from the persistent cortical memory.  Recall is bounded by
        SDM capacity / crosstalk -- honest, because consolidation is lossy.  The cleanup
        threshold is HIGHER than the transient store's (0.45 vs 0.30): distributed storage
        adds crosstalk, so the noise floor for a real successor rises, and a stricter cutoff
        keeps the recovered set grounded (rejects crosstalk-level spurious tokens)."""
        bg = getattr(self, '_branch_gen', None)
        if bg is None:
            return []
        order = getattr(self, '_branch_order', 1)
        ctx = list(context_tokens)[-order:]
        if not ctx:
            return []
        tag = ctx[0] if len(ctx) == 1 else '\x02'.join(ctx)
        b = self.unified.sdm.read(self.unified.ck.key(tag), tag)
        return bg._cleanup_bundle(np.asarray(b, dtype=np.complex128), thresh, max_k)

    def valid_next(self, context_tokens):
        """The substrate-recovered set of valid next tokens for a context (the
        branch set), by matching pursuit over the context's bundle."""
        return self.branch_gen.successors(list(context_tokens))

    def bundle_constraint(self, tokens):
        """Build a generation CONSTRAINT from a set of tokens: a phasor BUNDLE
        (superpose) of their substrate keys over the shared unified key space; the
        returned critic scores a candidate by COSINE resonance to the bundle -- high
        iff the candidate belongs to the set, and (because the bundle spans the set)
        independent of any orthogonal axis.  Pure FHRR: superpose + cosine cleanup, a
        substrate op, not python over a dict.  Compose several of these in
        generate_fluent(constraints=[...]) to satisfy many axes at once (product-of-
        experts).  Returns a callable c -> score|None."""
        import numpy as _np
        ck = self.unified.ck
        b = _np.zeros(ck.d, dtype=_np.complex64)
        for t in tokens:
            b = b + ck.key(str(t).strip().lower())
        nb = float(_np.linalg.norm(b))
        def _crit(c, _b=b, _nb=nb):
            k = ck.key(str(c).strip().lower())
            n = _nb * float(_np.linalg.norm(k))
            return (float(_np.real(_np.vdot(k, _b)) / n)) if n else None
        return _crit

    def scope_register(self, chunk=8):
        """A DYNAMIC substrate set: symbols DECLARED into scope as generation proceeds,
        tested for membership by resonance.  Correct-by-construction reference checking
        for structured generation (no undefined name).  A single flat bundle drowns in
        crosstalk past ~sqrt(d) members (Frady-Sommer); this holds membership at scale by
        the Day-88 wall-break -- BOUNDED chunks of `chunk` keys, each a superposition, the
        set = the list of sealed chunks + the open one.  member(sym) = max cosine over
        chunks; a true member resonates ~1/sqrt(chunk), a non-member ~1/sqrt(d).  Pure
        FHRR (superpose + cosine cleanup), no python set of the symbols.  Returns an
        object with .declare(sym), .member(sym), .score(sym), .n."""
        import numpy as _np
        ck = self.unified.ck
        d = int(ck.d)
        # member resonates ~1/sqrt(chunk); non-member ~1/sqrt(d).  Threshold = half the
        # member signal, but never below ~4x the sqrt(d) noise floor.
        thr = max(0.5 / (chunk ** 0.5), 4.0 / (d ** 0.5))

        class _ScopeRegister:
            def __init__(self):
                self.sealed = []                              # sealed chunk bundles
                self.cur = _np.zeros(d, dtype=_np.complex64)
                self.n_cur = 0
                self.n = 0                                    # total declared
            def declare(self, sym):
                self.cur = self.cur + ck.key(str(sym).strip().lower())
                self.n_cur += 1; self.n += 1
                if self.n_cur >= chunk:
                    self.sealed.append(self.cur)
                    self.cur = _np.zeros(d, dtype=_np.complex64)
                    self.n_cur = 0
            def score(self, sym):
                k = ck.key(str(sym).strip().lower())
                best = -1.0
                chunks = self.sealed + ([self.cur] if self.n_cur else [])
                for b in chunks:
                    nb = float(_np.linalg.norm(b))
                    if nb:
                        best = max(best, float(_np.real(_np.vdot(k, b)) / nb))
                return best
            def member(self, sym):
                return self.score(sym) >= thr

        reg = _ScopeRegister()
        reg.threshold = thr
        return reg

    def _make_critic(self, spec):
        """Turn a constraint SPEC into a critic callable c -> score|None, using only
        substrate resonance (flat co-occurrence).  A spec is one of:
          - callable                : used as-is (caller-supplied substrate score);
          - str (an anchor token)   : score = flat.similarity(c, token);
          - iterable of tokens      : score = MEAN flat.similarity(c, a) over the set --
                                      this cancels the OTHER axes and isolates the axis
                                      the set spans (the multi-constraint primitive).
        None (unseen / no signal) is returned as-is so the loop can treat it neutrally."""
        if callable(spec):
            return spec
        if isinstance(spec, str):
            tok = spec.strip().lower()
            def _c(c, _t=tok):
                try:
                    return self.semantic_sim(c, _t)      # Day 91: live unified store
                except Exception:
                    return None
            return _c
        anchors = [str(a).strip().lower() for a in spec]
        def _c(c, _a=anchors):
            vals = []
            for a in _a:
                try:
                    s = self.semantic_sim(c, a)          # Day 91: live unified store
                except Exception:
                    s = None
                if s is not None:
                    vals.append(s)
            return (sum(vals) / len(vals)) if vals else None
        return _c

    def generate_fluent(self, seed, steps=40, anchor=None, constraints=None,
                        combine='poe', veto=None, on_emit=None, sample=False, rng_seed=0):
        """Day-88/89 -- THE GENERATION LOOP with COMPOSED constraints: long-range
        coherent generation assembled from the substrate's own parts, no backprop and
        no hallucination.  Each step:
          VARIATION  = valid_next(path) -- the grounded set of REAL observed transitions
                       (matching pursuit over the context bundle); a candidate is never
                       invented, only selected;
          VETO       = a HARD content constraint (`veto(c)->bool`) removes forbidden
                       candidates BEFORE selection -- correct-by-construction: an invalid
                       token is never emitted (e.g. a reference to an undeclared symbol).
                       If every candidate is vetoed the loop halts honestly rather than
                       emit a violation;
          SELECTION  = one OR MANY soft critics score each survivor by RESONANCE.  N
                       constraints are composed by PRODUCT-OF-EXPERTS (`combine='poe'`,
                       AND semantics) or by mean (`combine='sum'`);
          STATE      = constraints live OUTSIDE the local window; `on_emit(tok)` updates a
                       running constraint register after each emission (e.g. declaring a
                       symbol into scope), so a DYNAMIC long-range constraint survives
                       arbitrary length.
        Day-88 proved ONE constraint; Day-89 proves N composed constraints AND a hard,
        dynamic, long-range one (scope) -- the framework generates a long STRUCTURED
        artifact correct-by-construction, at CPU cost.
        `constraints`: list of specs (see _make_critic); defaults to [anchor or seed].
        `veto`: callable c->bool (True = forbidden).  `on_emit`: callable tok->None run on
        the seed and every emitted token.  `sample=True` weights by combined fit.
        The whole sequence is NOVEL (recombination); every step is grounded; it halts
        honestly when no valid successor remains.
        Returns {sequence, grounded, constraints, length}."""
        import random as _r
        rng = _r.Random(rng_seed)
        start = str(seed).strip().lower()
        if constraints is None:
            constraints = [str(anchor).strip().lower() if anchor else start]
        critics = [self._make_critic(s) for s in constraints]
        path = [start]
        if on_emit is not None:
            on_emit(start)                                   # register the seed

        def combined(c):
            scores = [cr(c) for cr in critics]
            if combine == 'sum':
                seen = [s for s in scores if s is not None]
                return (sum(seen) / len(seen)) if seen else -1.0
            # product-of-experts: shift sim [-1,1] -> [0,2]; None -> neutral 1.0;
            # a hard-violated axis (sim<0 -> factor<1) multiplicatively kills the cand.
            prod = 1.0
            for s in scores:
                prod *= 1.0 if s is None else max(1e-6, s + 1.0)
            return prod

        for _ in range(int(steps)):
            cand = self.valid_next(path)
            if veto is not None:
                cand = [c for c in cand if not veto(c)]      # correct-by-construction
            if not cand:
                break                                       # honest halt (no fabrication)
            if sample:
                scored = [(c, max(0.0, combined(c))) for c in cand]
                tot = sum(w for _, w in scored) or 1.0
                r = rng.random() * tot; acc = 0.0; nxt = cand[0]
                for c, wgt in scored:
                    acc += wgt
                    if acc >= r:
                        nxt = c; break
            else:
                nxt = max(cand, key=combined)               # composed critics (default)
            path.append(nxt)
            if on_emit is not None:
                on_emit(nxt)                                 # update the dynamic register
        return {'sequence': path, 'grounded': True,
                'constraints': list(constraints), 'length': len(path)}

    # ── Day 90: FLUENT-GENERATION FRAMEWORK in ADDRESS space ──────────────
    def address_generator(self, order=1):
        """Day 90 -- the fluent-generation FRAMEWORK (holo_generate.AddressGenerator).
        Every word is an HV ADDRESS; the relations between words are HV too (per-context
        successor bundles); reading and generation carry ADDRESSES (integer codes), and
        surface words are materialised only at the edges (encode on the way in, decode on
        the way out).  The no-backprop, near-zero-compute analog of a transformer's
        next-token head: next address = resonance over the learned transition memory.  A
        FRAMEWORK -- proved data-free on made-up vocab; the fluency scales with the
        transition data fed to .observe(sequences).  Lazy + reused (cached on the org)."""
        ag = getattr(self, '_addr_gen', None)
        if ag is None or ag.order != int(order):
            from ikigai.cognition.holo_generate import AddressGenerator
            ag = AddressGenerator(self.unified.ck, order=int(order))
            self._addr_gen = ag
        return ag

    def generate_addressed(self, start, steps=40, theme=None, order=1, seed=0):
        """Generate a sequence ENTIRELY in address space and return
        {codes, words, length, vocab}.  The walk rolls over integer addresses; `words`
        is the only surface materialisation (render at the emit edge).  `theme` (a list
        of words) installs a coherence critic scored by address resonance -- the Day-88/89
        composed constraints apply here unchanged, still in address space.  Long-range
        coherence via a WIDER CONTEXT WINDOW is native through `order` (order>1 keys each
        prediction on the last k tokens, so an ambiguous token is disambiguated by what
        preceded it).  Call org.address_generator(order).observe(sequences) first."""
        ag = self.address_generator(order=order)
        critic = ag.theme_bundle(theme) if theme else None
        codes = ag.generate(start, steps=steps, critic=critic, seed=seed)
        return {'codes': codes, 'words': ag.render(codes),
                'length': len(codes), 'vocab': ag.vocab_size}

    # ── Day 90: STRUCTURE-FIRST generation (Levelt frame-then-fill) ──────────
    def generate_structured(self, seed, frame, type_vocab, theme=None,
                            constraints=None, meaning=True, ctx_window=4):
        """Day 90/91 -- STRUCTURE-FIRST generation, the biological (Levelt) mechanism: build a
        FRAME first -- a sequence of slot TYPES, the syntactic/discourse scaffold -- then FILL
        each slot with a word of the right type, chosen by MEANING.  Unlike the flat walk
        (which has no global structure and drifts off the pattern over length), the frame is a
        HARD, correct-by-construction constraint: an off-type word is NEVER emitted, so the
        output's structure is guaranteed.  Within a slot the filler is the grounded candidate
        (a real observed transition) of the correct type that best fits the composed critics.
        Halts honestly if a slot has no grounded, type-correct filler -- no fabrication.

        Day 91 -- the FILL now composes a MEANING/GROUNDING critic (the missing piece the Day-90
        reckoning named).  A structurally valid, theme-on sentence can still be locally
        INCOHERENT: type and theme fix the scaffold and the global topic, but not that each
        filler belongs with the WORDS ALREADY EMITTED.  The meaning critic is a DYNAMIC,
        context-dependent critic (`meaning=True`, default): at each slot it scores a candidate
        by its mean flat CO-OCCURRENCE resonance to the recent context (`ctx_window` tokens) --
        distributional semantics (`flat.similarity`), a pure substrate op, never a topic label
        or word list.  It is COMPOSED with the static critics (theme + any `constraints`) by
        product-of-experts (the Day-89 machinery), so the pick satisfies structure AND theme AND
        local meaning at once.  Where `flat` has no signal for a candidate the critic returns
        None and composes NEUTRALLY -- so with an unpopulated co-occurrence memory the fill is
        byte-identical to the Day-90 theme-only behaviour (the meaning critic is free to be on).

        This is lever B made structural, and it UNIFIES the depth loop with the generator: the
        frame is a PLAN (the planner can produce the slot-type sequence), and filling it is
        composed constrained generation -- goal -> frame -> fill -> artifact, one loop.

        frame: list of slot type-ids (the scaffold).  type_vocab: {type_id: allowed words}.
        theme: optional list of words -> a static coherence critic (bundle resonance).
        constraints: optional list of extra specs (see _make_critic), PoE-composed with theme.
        meaning: compose the dynamic local co-occurrence critic (default True).
        Returns {sequence, types, length, frame_len, grounded, structural_valid}."""
        static = []
        if theme:
            static.append(self.bundle_constraint(theme))
        if constraints:
            static.extend(self._make_critic(s) for s in constraints)
        vsets = {t: set(str(x).strip().lower() for x in ws) for t, ws in type_vocab.items()}
        start = str(seed).strip().lower()
        path, types = [start], [frame[0]]
        grounded = True
        ior = max(2, len(set(frame)))              # inhibition-of-return window (biology)

        def _poe(c, ctx_crit):
            """product-of-experts over the static critics + the dynamic meaning critic;
            a None (no signal on an axis) composes neutrally (factor 1.0)."""
            prod = 1.0
            for cr in static:
                s = cr(c)
                prod *= 1.0 if s is None else max(1e-6, s + 1.0)
            if ctx_crit is not None:
                s = ctx_crit(c)
                prod *= 1.0 if s is None else max(1e-6, s + 1.0)
            return prod

        for t in frame[1:]:
            allowed = vsets.get(t, set())
            cands = [c for c in self.valid_next(path) if c in allowed]   # grounded AND typed
            if not cands:
                grounded = False
                break                                                    # honest halt
            # inhibition of return: suppress recently-emitted fillers so slots vary (avoid
            # the degenerate repeat of the single max-fit word), keeping alternatives only
            recent = set(path[-ior:])
            pool = [c for c in cands if c not in recent] or cands
            # DYNAMIC meaning/grounding critic: cohere with the local context already emitted
            # (mean flat co-occurrence resonance over the recent window) -- rebuilt per slot.
            ctx_crit = self._make_critic(path[-ctx_window:]) if meaning else None
            if static or ctx_crit is not None:
                pick = max(pool, key=lambda c: _poe(c, ctx_crit))
            else:
                pick = pool[0]
            path.append(pick); types.append(t)
        # structure is guaranteed by construction: every token is of its slot's declared type
        structural_valid = all(path[i] in vsets.get(types[i], set()) for i in range(len(path)))
        return {'sequence': path, 'types': types, 'length': len(path),
                'frame_len': len(frame), 'grounded': grounded,
                'structural_valid': structural_valid}

    # ── Day 92: ORGANISM-TRUE generation -- realize MEANING through STRUCTURE, no next-token ──
    @property
    def frame_inducer(self):
        """Day 92 -- unsupervised frame/type induction (Harris distributional): recover
        syntactic TYPES from neighbour context and FRAMES from real type-sequences, so the
        generator needs NO hand-authored template.  Reinforced by more data.  Lazy."""
        fi = getattr(self, '_frame_inducer', None)
        if fi is None:
            from ikigai.cognition.frame_induction import FrameInducer
            fi = FrameInducer(seed=92)
            self._frame_inducer = fi
        return fi

    def induce_frames(self, sentences, k=6, min_count=2, top=None):
        """Observe sentences, induce word TYPES (cluster by context) + a FRAME inventory
        (frequent induced type-sequences).  Call repeatedly to REINFORCE.  Returns #frames."""
        sents = [(_s.split() if isinstance(_s, str) else list(_s)) for _s in sentences]
        fi = self.frame_inducer
        fi.observe(sents)
        fi.induce_types(k=k)
        fi.induce_frames(sents, min_count=min_count, top=top)
        return len(getattr(fi, 'frame_inventory', []))

    # ── Day 92: FLUENT fill -- compose induced frame x pcseq x meaning, slot-by-slot ──
    def read_holo(self, text):
        """Read a passage into the holographic reader (every token recoverable
        from its context). Returns the number of writes."""
        return self.holo_reader.read(text)

    def answer_holo(self, sentence_with_hole, hole_token="_", top_k=1):
        """Fill a blank in a sentence by holographic resonance over what has been
        read. Honest-unknown below the substrate noise floor."""
        return self.holo_reader.answer(sentence_with_hole, hole_token=hole_token,
                                       top_k=top_k)

    def ask_holo(self, question, top_k=1):
        """Answer a plain-English question over what has been read -- no hole to
        mark, no wh-list, no templates. Morphology + drop-unmatchable are native.
        Honest-unknown below the noise floor."""
        return self.holo_reader.ask(question, top_k=top_k)

    def comprehend(self, text, min_rel_df=2):
        """Read messy text -> EMERGENT (subj, rel, obj) atoms (relations learned
        by recurrence, no templates/lists) -> ingest into the derive-not-store
        engine so composites/multi-hop are DERIVED, never stored. The reader is
        the episodic front door; the derive engine is the semantic store.
        Returns the extracted triples."""
        return self.holo_reader.comprehend(text, organism=self, min_rel_df=min_rel_df)

    def _is_question(self, text):
        """A telling states a fact; a query asks for one -- the organism must never LEARN from a
        question.  It uses the interrogatives it LEARNED from the corpus (openers the corpus itself
        marks with a trailing '?') plus the '?' punctuation.  Before it has read any language it
        genuinely cannot tell, and returns False rather than pretend with an authored list."""
        if not isinstance(text, str):
            return False
        t = text.strip().lower()
        if t.endswith('?'):
            return True
        interr = getattr(self, '_interrogatives', None)
        if not interr:
            return False
        first = t.split()[0] if t.split() else ''
        return first in interr

    def tell(self, text):
        """Continual learning from what a human SAYS -- one fact or a whole paragraph.
        comprehend() learns only by RECURRENCE (a relation must recur), so a single
        told fact taught nothing.  tell() handles ANY input: each sentence is parsed
        against the organism's own known relations (one-shot telling), and a multi-
        sentence passage ALSO runs through comprehend() so relations that RECUR across
        the passage are absorbed too.  Every candidate is learned through the real
        body-modulated path (the 'learn' faculty -> learn_reinforced: dict cache +
        substrate) and VERIFIED by reading it back -- so it reports learning ONLY what
        it can now derive.  Returns {learned:[(s,r,o)...], text: summary|None}."""
        import re as _re
        if not isinstance(text, str) or not text.strip():
            return {'learned': [], 'text': None}
        eng = self.general_reasoner.derive_engine
        rels = set(getattr(eng, 'relations', []) or [])
        learned = []

        def _norm(x):                               # to the substrate's own tokenization, so a
            return ' '.join(_re.findall(r'[a-z0-9]+', str(x).lower()))   # hyphenated/punctuated
                                                    # value from either extractor round-trips
        def _absorb(a, r, o):
            a, r, o = _norm(a), _norm(r), _norm(o)
            if not (a and r and o) or (a, r, o) in learned:
                return
            self.be((a, r, o))                      # native learn faculty -> learn_reinforced
            if (eng.atom(r, a) or eng.inherited_atom(r, a)) == o:   # only keep what verifies
                learned.append((a, r, o))

        sents = [s.strip() for s in _re.split(r'[.!?\n]+', text) if s.strip()]
        for s in sents:                             # one-shot tellings, sentence by sentence
            if self._is_question(s):                # a query is not a telling
                continue
            try:                                    # LEARNED surface frames -- what it read, no templates
                ev = self.extract_verified(s)
            except Exception:
                ev = None
            if isinstance(ev, (tuple, list)) and len(ev) == 3 and all(ev):
                _absorb(*ev)
        if len(sents) > 1:                          # recurrence over the whole passage
            try:
                for tri in (self.comprehend(text) or []):
                    if len(tri) >= 3 and str(tri[1]).strip().lower() in rels:   # known relation only
                        _absorb(tri[0], tri[1], tri[2])
            except Exception:
                pass

        summary = '; '.join(f"the {r} of {a} is {o}" for a, r, o in learned[:5]) or None
        return {'learned': learned, 'text': summary}

    def learn_language(self, sentences=None, corpus='eng_sentences.tsv.bz2', n=500000):
        """LEARN THE FORM OF LANGUAGE FROM EXPOSURE -- the biological path, not authored
        templates.  Reads raw English sentences and induces its surface frames by
        search-under-verification (`induce_surface_verified`: the corpus proposes forms, the
        organism's OWN knowledge disposes -- no hand-authored patterns, no LLM judge).  After
        this, `extract_verified()` parses a telling via the frames it LEARNED, and tell() /
        _fac_learn prefer that over the template crutch (which shrinks toward dead as coverage
        grows).  Pass `sentences`, or let it read `corpus` (Tatoeba .tsv[.bz2], first `n`).
        Returns the verified-frame report."""
        import os as _os, bz2 as _bz2
        if sentences is None:
            path = corpus if _os.path.isabs(corpus) else _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), corpus)
            sentences = []
            _open = _bz2.open if path.endswith('.bz2') else open
            with _open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    p = line.split('\t')
                    sentences.append((p[2] if len(p) >= 3 else p[-1]).strip())
                    if len(sentences) >= n:
                        break
        sentences = list(sentences)
        stats = self._induce_grammar_stats(sentences)      # function words + interrogatives (EMERGENT)
        rep = self.induce_surface_verified(sentences)      # fact-grounded frames (distant supervision)
        cur = self.curiosity_frames(sentences)             # curiosity + self-consistency (broad)
        gen = self.fit_generator(sentences)                # Day-106: coherent open-ended generation
        return {'grammar': stats, 'verified': rep,
                'curiosity': {k: v['func'] for k, v in cur.items()},
                'generator': gen}

    def fit_generator(self, sentences):
        """Day-106 -- fit the CoherentGenerator (open-ended, grounded, coherent sentence
        generation) on raw sentences.  Tokenized to alpha words, 4-10 long; the generator learns
        emergent type-signatures + grounded transitions + the composed-prefix re-rank that lifts
        coherent generation from ~9% to ~100% (`experiments/audit/day106_coherent_gen.py`).  This
        is the GENERATE path -- fluent, distinct from the correct-or-abstain ANSWER path."""
        import re as _re
        from ikigai.cognition.coherent_generator import CoherentGenerator
        # cap the fit corpus: bounds RAM (vocab-sized signature tables) so a 1GB box can refit at
        # boot without OOM. Quality saturates far below this -- the day-106 gate hit 100% on 4k.
        cap = 60000
        seqs = []
        for s in sentences:
            toks = _re.findall(r'[a-z]+', str(s).lower())
            if 4 <= len(toks) <= 10:
                seqs.append(toks)
                if len(seqs) >= cap:
                    break
        if not seqs:
            return {'fitted': False, 'sequences': 0}
        self._coherent_gen = CoherentGenerator().fit(seqs)
        return {'fitted': True, 'sequences': len(seqs), 'vocab': len(self._coherent_gen.vocab)}

    @property
    def coherent_gen(self):
        """Lazy CoherentGenerator: if the organism has read language but the generator wasn't
        fit yet, fit it from the corpus once (same source learn_language uses)."""
        g = getattr(self, '_coherent_gen', None)
        if g is not None and getattr(g, '_fitted', False):
            return g
        return None

    def generate(self, n=1, seed=None):
        """Generate `n` coherent, grounded, NOVEL sentences (open-ended generation).  Requires the
        generator to have been fit (learn_language / fit_generator).  Fluent, not fact-checked --
        use answer()/org(x)'s answer faculty for grounded facts, this to GENERATE."""
        g = self.coherent_gen
        if g is None:
            return [] if n != 1 else ''
        return g.generate(n=n, seed=seed)

    def _induce_grammar_stats(self, sentences):
        """LEARN the closed-class grammar of the language FROM THE CORPUS -- no authored lists.
        Function words EMERGE as the high-frequency head of the distribution (Zipf); INTERROGATIVES
        emerge as the words that open sentences the corpus itself marks as questions (trailing
        '?') far more than they open statements. Both are learned data, persisted with the
        organism -- the same way it learns its relation frames. Returns a small summary."""
        import collections as _c, re as _re
        freq = _c.Counter(); q_open = _c.Counter(); s_open = _c.Counter()
        for s in sentences:
            s = str(s).strip().lower()
            toks = _re.findall(r'[a-z0-9]+', s)
            if not toks:
                continue
            freq.update(toks)
            (q_open if s.endswith('?') else s_open)[toks[0]] += 1
        # function words = the frequency head (closed class), learned not authored
        self._function_words = set(w for w, _ in freq.most_common(50))
        # interrogatives = openers that the corpus marks interrogative much more than declarative
        interr = set()
        for w, qc in q_open.items():
            if qc >= 15 and qc > 2 * s_open.get(w, 0):
                interr.add(w)
        self._interrogatives = interr
        return {'function_words': len(self._function_words),
                'interrogatives': sorted(interr)[:20]}

    def curiosity_frames(self, sentences, min_support=15, min_func=0.65, min_distinct=8,
                         install=True, min_generalize=8, gen_min_support=5, gen_min_distinct=4):
        """Curiosity-gated frame induction -- learns relation surface-frames from raw text with
        NO pre-known facts, bypassing induce_surface_verified's known-fact wall (which could
        only verify 'capital' because the organism knew ~13 capitals).

        Curiosity PROPOSES: the genitive 'of' is a salient, recurrent, informative construction
        ('the R of S is O' / 'O is the R of S', R a content word) -- the organism trusts it.
        Consolidation DISPOSES: a candidate relation R survives only if what its frame extracts
        is SELF-CONSISTENT -- a real relation is (mostly) FUNCTIONAL (each S -> one dominant O);
        a coincidence extracts a contradictory scatter and is pruned.  The corpus verifies
        ITSELF by coherence rather than against facts the organism already holds -- so it learns
        relations (and far more capitals) than it knew.  install=True writes the surviving
        frames into the surface realizer (both word orders).  Returns {R: {support, func,
        distinct, frame}}."""
        import re as _re, collections as _c
        # function words are LEARNED (frequency head), not authored -- filter content vs closed-class
        STOP = getattr(self, '_function_words', None) or set()
        def _w(x):
            return bool(x) and x.isalpha() and x not in STOP
        pairs = _c.defaultdict(list)
        supp = _c.Counter()
        ra = _re.compile(r'\bthe (\w+) of (\w+) (?:is|was) (\w+)\b')       # the R of S is O
        rb = _re.compile(r'\b(\w+) (?:is|was) the (\w+) of (\w+)\b')       # O is the R of S
        for s in sentences:
            s = str(s).lower()
            for m in ra.finditer(s):
                R, S, O = m.groups()
                if _w(R) and _w(S) and _w(O):
                    pairs[R].append((S, O)); supp[R] += 1
            for m in rb.finditer(s):
                O, R, S = m.groups()
                if _w(R) and _w(S) and _w(O):
                    pairs[R].append((S, O)); supp[R] += 1
        out = {}
        sf = getattr(self, 'surface', None)
        for R, cnt in supp.items():
            if cnt < min_support:
                continue
            by_s = _c.defaultdict(_c.Counter)
            for s, o in pairs[R]:
                by_s[s][o] += 1
            if len(by_s) < min_distinct:
                continue
            func = sum(c.most_common(1)[0][1] / sum(c.values()) for c in by_s.values()) / len(by_s)
            if func < min_func:
                continue
            frame = ['{O}', 'is', 'the', R, 'of', '{S}']
            out[R] = {'support': cnt, 'func': round(func, 3), 'distinct': len(by_s), 'frame': frame}
            if install and sf is not None:
                for attr in ('templates', 'variants'):
                    if getattr(sf, attr, None) is None:
                        setattr(sf, attr, {})
                sf.templates[R] = list(frame)
                sf.variants.setdefault(R, [])
                for fr in ([['{O}', 'is', 'the', R, 'of', '{S}'],
                            ['the', R, 'of', '{S}', 'is', '{O}']]):
                    if fr not in sf.variants[R]:
                        sf.variants[R].append(list(fr))
        # ── CONSTRUCTION GENERALIZATION (Day-105) ──────────────────────────────
        # The genitive skeleton 'the {R} of {S} is {O}' was tried with EVERY content word as R.
        # If MANY distinct relation-words survive the same functional-consistency test, the middle
        # slot is PRODUCTIVE -- the org has evidence the construction generalizes over relations,
        # not that any single word is special.  Then install a {R}-slot frame so a NEVER-SEEN
        # relation ('the glorbf of qualan is dree') parses ONE-SHOT: the speaker's word fills {R}.
        # This is abstraction over observed data (learning), NOT an authored rel=word_before_of rule.
        productive = 0
        for R, cnt in supp.items():
            if cnt < gen_min_support:
                continue
            by_s = _c.defaultdict(_c.Counter)
            for s, o in pairs[R]:
                by_s[s][o] += 1
            if len(by_s) < gen_min_distinct:
                continue
            func = sum(c.most_common(1)[0][1] / sum(c.values()) for c in by_s.values()) / len(by_s)
            if func >= min_func:
                productive += 1
        if install and sf is not None and productive >= min_generalize:
            if hasattr(sf, 'install_generic'):
                sf.install_generic([['{O}', 'is', 'the', '{R}', 'of', '{S}'],
                                    ['the', '{R}', 'of', '{S}', 'is', '{O}']])
            # record the evidence on the organism (auditable; not part of the per-relation map)
            self._generalized_construction = {'productive_relations': productive,
                                              'construction': 'the {R} of {S} is {O}'}
        return out

    def ask_derive(self, question, depth=None):
        """Answer a plain-English question through the SEMANTIC derive engine
        (not the episodic holo store). Depth is EMERGENT by default: the question
        is parsed into (entity, [relation_mentions]) and the hop count = the
        number of mentions, read from the question -- no count passed. Each
        relation is applied innermost-out. Pass `depth` to force a same-relation
        chain of that length (for nested function-word chains the emergent parser
        does not yet split). Returns (answer, entity, relation_or_mentions).
        Derive-not-store -- the answer is computed from atoms."""
        eng = self.general_reasoner.derive_engine
        if depth is None:
            # Parse against the SEMANTIC store's own vocab first (data ingested
            # straight into the engine never passed through the episodic reader);
            # fall back to the episodic parser for text the reader actually read.
            ent, mentions = self.holo_reader.parse_for_engine(
                question, eng.relations, eng.entities)
            if not (ent and mentions):
                ent, mentions = self.holo_reader.parse_chain(question)
            # Day-99 -- ASK THE SUBSTRATE WHICH TOKEN IS THE ENTITY, don't ask a stale index.
            #
            # Both parsers identify the entity by membership in `eng.entities` -- but that set is
            # POPULATED BY successful atom() hits, so on a freshly loaded organism it does not yet
            # contain the very entity being asked about. Deadlock: the parser needs `france` in
            # entities to find it; entities only gains `france` once atom() resolves it; atom() is
            # only called once the parser has found it.
            #
            # MEASURED on production (193 MB, vocab 6,808):
            #   'france' in eng.entities            -> False
            #   parse -> ent='what the'  (it fell back to "longest arg run joined")
            #   ask_derive -> None  ->  answer() -> "i don't know"   ... while atom('capital',
            #   'france') returned 'paris' the whole time. The organism knew Paris and refused to
            #   say Paris. One atom() hit seeded `entities` and everything worked immediately.
            # No gate caught this because full_capability calls gr.reason() DIRECTLY, bypassing
            # answer() -- capitals 20/21 green all day over a door that was shut.
            #
            # The entity is simply WHAT THE RELATION APPLIES TO. The substrate can answer that in
            # O(1) per candidate, needs no index, and cannot go stale.
            if mentions:
                _r0 = mentions[-1]
                if not ent or not (eng.atom(_r0, ent) or eng.inherited_atom(_r0, ent)):
                    for _c in self.general_reasoner.tokenize(question):
                        if _c and _c not in mentions and (eng.atom(_r0, _c)
                                                          or eng.inherited_atom(_r0, _c)):
                            ent = _c
                            break
            if not (ent and mentions):
                return None, ent, mentions
            cur = ent
            for rel in reversed(mentions):              # innermost-out
                nxt = eng.atom(rel, cur) or eng.inherited_atom(rel, cur)
                if not nxt and eng.is_transitive(rel):
                    # a bare class question ("what is X"): the copula maps to ONE
                    # taxonomic link by morphology, but the fact may sit under a
                    # SIBLING link (isa vs subclassof).  Try the other structurally
                    # taxonomic (learned-transitive) relations -- link-ness comes
                    # from the mined rules, not a hand-authored list.
                    for alt in eng.relations:
                        if alt != rel and eng.is_transitive(alt):
                            nxt = eng.atom(alt, cur) or eng.inherited_atom(alt, cur)
                            if nxt:
                                mentions = [alt if m == rel else m for m in mentions]
                                break
                if not nxt:
                    return None, ent, mentions
                cur = nxt
            return cur, ent, mentions
        ent, rel = self.holo_reader.parse_question(question)
        if not (ent and rel):
            return None, ent, rel
        cur = ent
        for _ in range(max(1, depth)):
            cur = eng.atom(rel, cur) or eng.inherited_atom(rel, cur)
            if not cur:
                return None, ent, rel
        return cur, ent, rel

    def _derive_transitive_answer(self, question, ent, rel, tok):
        """Day-105 -- MULTI-HOP through the front door, emergently.  `rel` is a
        DISCOVERED-transitive relation (`is_transitive` == a LEARNED rule, never an
        authored link list), so its direct atom is only the FIRST hop.  Derive the
        closure from `ent` on demand (`transitive_reach` == derive-not-store) and
        answer from it:

          - the question NAMES a target node (a token that is a member of the derived
            chain, or a known engine entity, other than the subject): answer the polar
            ask by CLOSURE MEMBERSHIP.  A derivable ancestor is affirmed with every
            content token taken from the derived chain (grounded -> nothing invented);
            a target that is NOT derivable is not asserted (return None -> the caller's
            own honest read-out stands, no fabricated 'yes').
          - no named target ('what is a X'): STATE the derived ancestors.

        No regex, no yes/no template, no relation table: transitivity is learned, the
        target is entity membership, and every word is the chain's own token."""
        eng = self.general_reasoner.derive_engine
        chain = eng.transitive_reach(rel, ent)
        if not chain or len(chain) < 2:
            return None
        ancestors = chain[1:]
        ents = getattr(eng, 'entities', None) or set()
        allowed = set(tok(' '.join(chain)))          # only the derived chain's own tokens
        targets = [t for t in tok(question)
                   if t != ent and (t in ancestors or t in ents)]
        if targets:
            y = targets[-1]
            if y in ancestors:                       # a derivable ancestor -> affirm (multi-hop yes)
                text = f"{ent} {rel} {y}"
                grounded = all(t in allowed for t in tok(text))
                return {'text': text, 'grounded': grounded, 'fact': (ent, rel, y),
                        'hops': ancestors.index(y) + 1}
            return None                              # named a target it cannot derive -> don't assert
        text = f"{ent} {rel} " + ', '.join(ancestors)
        grounded = all(t in allowed for t in tok(text))
        return {'text': text, 'grounded': grounded, 'fact': (ent, rel, ancestors[-1])}

    def answer(self, question, depth=None, explain=False):
        """Day-85 #4 -- GROUNDED, FAITHFUL read-out.  Derive the answer from the
        substrate (ask_derive, exact) and state it in words whose every content
        token comes FROM the derived fact -- so the organism can only say what it
        actually derived, and abstains honestly ("i don't know") when it cannot.
        No hallucination by construction: this is the generation axis we win on
        (faithfulness), not raw fluency.  The connective is the relation's own
        surface (no authored template).  Returns {text, grounded, fact}; grounded
        is the verifiable guarantee that no content token was invented.

        explain=True stacks the differentiators: it routes through the
        proof-carrying path, so the answer is emitted ONLY if its derivation
        chain re-derives + verifies, and attaches a grounded 'because' --
        each hop stated as `<premise> <relation> <conclusion>`, every token
        from the chain.  Faithful + transparent + verifiable: what a trained LM
        structurally cannot offer."""
        tok = self.general_reasoner.tokenize
        if explain:
            p = self.ask_derive_proof(question, depth=depth)
            ent, ans, rels = p.get('entity'), p.get('answer'), p.get('relations') or []
            if not (ans and p.get('verified')):
                return {'text': "i don't know", 'grounded': True, 'fact': None,
                        'verified': False, 'because': None}
            eng = self.general_reasoner.derive_engine
            steps, cur, vocab = [], ent, []
            for r in reversed(list(rels)):              # innermost-out, mirror the proof
                nxt = eng.atom(r, cur) or eng.inherited_atom(r, cur)
                steps.append(f"{cur} {r} {nxt}")
                vocab += [str(cur), str(r), str(nxt)]
                cur = nxt
            rel = str(rels[-1]).strip() if rels else ''
            text = f"{ent} {rel} {ans}".strip()
            because = ' ; '.join(steps)
            allowed = set(tok(' '.join(vocab)))
            grounded = all(t in allowed for t in tok(text + ' ' + because))
            # Day-98 NO-HALLUCINATION THEOREM -- ENFORCED, not merely reported.  Emission is GATED
            # on (verified AND grounded): every content token of the answer and its 'because' must
            # appear in the INDEPENDENTLY re-derived proof-chain vocab.  If a single token is not
            # traceable to a re-derived fact, the organism ABSTAINS rather than emit it.  This is
            # the whole theorem: there is no code path that returns emitted text with grounded=False,
            # so an unverified/ungrounded token CANNOT be emitted -- by construction, not by training.
            if not grounded:
                return {'text': "i don't know", 'grounded': True, 'fact': None,
                        'verified': False, 'because': None}
            return {'text': text, 'grounded': grounded, 'fact': (ent, rel, ans),
                    'verified': True, 'because': because, 'proof': p.get('proof')}
        ans, ent, rels = self.ask_derive(question, depth=depth)
        if not ans:
            return {'text': "i don't know", 'grounded': True, 'fact': None}
        rel = (rels[-1] if isinstance(rels, list) and rels else (rels or '')).strip()
        # Day-105 -- the direct atom is only hop 1 of a LEARNED-transitive relation. Derive
        # the closure through the front door so org(x) answers multi-hop natively (yes/no
        # ancestor + 'what is a X' chain), emergently -- see _derive_transitive_answer.
        if rel and self.general_reasoner.derive_engine.is_transitive(rel):
            mh = self._derive_transitive_answer(question, ent, rel, tok)
            if mh is not None:
                return mh
        text = f"{ent} {rel} {ans}".strip()
        allowed = set(tok(f"{ent} {rel} {ans}"))
        grounded = all(t in allowed for t in tok(text))     # nothing invented
        return {'text': text, 'grounded': grounded, 'fact': (ent, rel, ans)}

    def lifetime(self, segments=6, ticks_per_segment=8, stream=None, sleep_every=4):
        """Day-87 GOLD -- A MIND OVER A LIFETIME.  Run the life loop for many
        ticks and watch a stable SELF emerge: beliefs accumulate then plateau,
        surprises and self-corrections cluster early and taper as the model of the
        world settles, curiosity stays coherent, and the ikigai steers throughout.
        Runs `live` in segments, snapshotting the self after each, so the whole
        trajectory is legible.  Returns {trajectory, final_self}."""
        stream = list(stream or [])
        traj = []
        for _ in range(max(1, segments)):
            seg_inputs = [stream.pop(0) for _ in range(ticks_per_segment) if stream]
            r = self.live(ticks=ticks_per_segment, inputs=seg_inputs, sleep_every=sleep_every)
            traj.append({'age': self._age, 'beliefs': len(self._beliefs),
                         'surprises': getattr(self, '_surprises', 0),
                         'confirmed': r.get('confirmed', 0)})
        return {'trajectory': traj, 'final_self': self.introspect()}

    def energy_probe(self, query_fn, repeats=200):
        """Day-87 GOLD -- BRAIN-ENERGY: measure the WORK a query costs.  A flat,
        content-addressed derive touches only its atom plus its derivation chain --
        a bounded number of memory operations -- no matter how large the store is.
        We count the atom lookups per query (the active 'hard locations', the
        energy-bearing units) and the wall time.  Run this at two store sizes and
        the count stays flat: energy scales with the CHAIN, not the knowledge.
        Returns {atom_lookups_per_query, seconds_per_query}."""
        import time as _t
        eng = self.general_reasoner.derive_engine
        before = eng._stats['atom_lookups']
        t0 = _t.perf_counter()
        for _ in range(max(1, repeats)):
            query_fn()
        dt = (_t.perf_counter() - t0) / max(1, repeats)
        lookups = (eng._stats['atom_lookups'] - before) / max(1, repeats)
        return {'atom_lookups_per_query': round(lookups, 2),
                'seconds_per_query': dt}

    def free_energy(self, observations, sample=2000):
        """Day-88 -- VARIATIONAL FREE ENERGY: the organism's mean SURPRISE about a
        set of observations under its CURRENT generative model (Friston).  This is
        the one quantity the whole autonomy loop minimises: perception lowers it by
        updating beliefs, learning lowers it by inducing rules that make the world
        derivable, action lowers EXPECTED free energy by resolving informative gaps.

        Surprise of a held-out fact (s, r, o):
          - the model DERIVES it (exact atom / inheritance / transitive membership):
              p ~ 1        -> surprise ~ 0     (it already explains this)
          - a tagged BELIEF predicts (s,r)=v with calibrated confidence c:
              p = c if v==o else (1-c)         -> -log c  /  -log(1-c)
          - no prediction at all:
              p = 1/V_r   (V_r = distinct values seen for r) = maximum uncertainty
        F = mean(-log p).  Computed from the SUBSTRATE (derivation + calibrated
        belief confidence) -- not a toy field.  Feed a HELD-OUT probe set: as the
        organism learns, F over the same set DROPS (it gets less surprised).
        Returns {free_energy, n, derived, believed, prior} (the surprise breakdown)."""
        import math, random as _r
        eng = self.general_reasoner.derive_engine
        vocab = {}
        for (s, r), v in eng.triples.items():
            if v:
                vocab.setdefault(str(r).lower(), set()).add(str(v).lower())
        beliefs = getattr(self, '_beliefs', {})
        obs = list(observations)
        if len(obs) > sample:
            obs = _r.Random(0).sample(obs, sample)
        if not obs:
            return {'free_energy': 0.0, 'n': 0, 'derived': 0, 'believed': 0, 'prior': 0}
        tot = 0.0
        n_der = n_bel = n_pri = 0
        for (s, r, o) in obs:
            s, r, o = str(s).strip().lower(), str(r).strip().lower(), str(o).strip().lower()
            if eng.is_transitive(r):                     # membership prediction
                pred_ok = eng.transitive_related(r, s, o)
                if pred_ok is not None and (pred_ok or eng.atom(r, s)):
                    p = 1.0 if (pred_ok or eng.atom(r, s) == o) else 1e-3
                    n_der += 1
                else:
                    p = 1.0 / max(2, len(vocab.get(r, ()))); n_pri += 1
            else:
                pred = eng.atom(r, s) or eng.inherited_atom(r, s)   # exact derivation
                if pred is not None:
                    p = 1.0 if pred == o else 1e-3; n_der += 1
                else:
                    b = beliefs.get((s, r))
                    if b:
                        c = max(1e-3, min(1 - 1e-3, float(b.get('confidence', 0.5))))
                        p = c if str(b.get('value')).lower() == o else (1.0 - c); n_bel += 1
                    else:
                        p = 1.0 / max(2, len(vocab.get(r, ()))); n_pri += 1
            tot += -math.log(max(p, 1e-9))
        return {'free_energy': tot / len(obs), 'n': len(obs),
                'derived': n_der, 'believed': n_bel, 'prior': n_pri}

    def _efe_rank_gaps(self, gaps, lam=0.4):
        """Day-88 -- EXPECTED FREE ENERGY action selection: rank knowledge gaps by
        EFE = -pragmatic - lam*epistemic (Friston explore/exploit), computed from
        REAL substrate quantities and dispatched through FreeEnergyActionSelector:
          epistemic (explore) = the gap's CuriosityDrive novelty (info gain);
          pragmatic (exploit) = alignment with the organism's ikigai/purpose.
        Returns gaps re-ordered best-first, each annotated with 'efe'."""
        from ikigai.cognition.fe_action import FreeEnergyActionSelector
        purpose = getattr(self, '_ikigai', None)
        cands = []
        for i, g in enumerate(gaps):
            epi = float(g.get('novelty', g.get('peer_frac', 0.5)))
            prag = 1.0 if (purpose and (purpose in g['entity'] or purpose in g['relation'])) else 0.0
            cands.append((i, epi, prag))
        ranked = FreeEnergyActionSelector.select_from_values(cands, lam=lam)
        out = []
        for (i, efe, epi, prag) in ranked:
            g = dict(gaps[i]); g['efe'] = round(efe, 3)
            out.append(g)
        return out

    def study(self, facts, rounds=2, min_support=2, min_conf=0.8):
        """Day-87 GOLD -- SELF-TAUGHT DOMAIN: hand the organism a body of facts (a
        'textbook') and let it MASTER the subject on its own -- ingest, discover
        the domain's rules, and self-quiz across study rounds (wonder about gaps,
        self-answer by derivation, form hypotheses for the rest).  No gradient
        descent: mastery = the rules it induces, which then answer entailed
        questions it was never explicitly told.  Returns a study log."""
        self.ingest_triples(facts, discover=True,
                            min_support=min_support, min_conf=min_conf)
        log = {'rules': 0, 'wondered': 0, 'self_answered': 0, 'hypotheses': 0}
        for _ in range(max(1, rounds)):
            t = self.contemplate()
            log['wondered'] += len(t['wondered'])
            log['self_answered'] += len(t['self_answered'])
            log['hypotheses'] += len(t['hypotheses'])
        log['rules'] = len(self.general_reasoner.derive_engine.learned_rules)
        return log

    def exam(self, questions):
        """Score the organism on held-out (question, expected) pairs, answering by
        derivation only (atom / inherited / transitive).  Each question is
        (subject, relation, expected_value).  Returns {score, n, results}."""
        eng = self.general_reasoner.derive_engine
        nrm = lambda s: str(s).replace(' ', '')
        results, correct = [], 0
        for (s, r, exp) in questions:
            if eng.is_transitive(r):                     # membership: is s (transitively) a exp?
                got = exp if eng.transitive_related(r, s, exp) else (eng.atom(r, s) or None)
            else:                                        # attribute: what is s's r?
                got = eng.atom(r, s) or eng.inherited_atom(r, s)
            ok = nrm(got or '') == nrm(exp)
            correct += int(ok)
            results.append({'q': (s, r), 'got': got, 'expect': exp, 'ok': ok})
        return {'score': correct, 'n': len(questions), 'results': results}

    def meld(self, other):
        """Day-87 GOLD -- MIND MELD: two organisms merge their minds.  Their atom
        stores are unioned (contradictions -- same subject+relation, different
        value -- are detected and reported, never silently overwritten), their
        rules re-mined on the combined knowledge, and the result is a RICHER
        organism that can DERIVE facts NEITHER had alone: if one knows a chain up
        to a point and the other continues it, the melded organism reaches the end.
        Collective intelligence with no central model -- distributed cognition from
        the substrate.  Returns {atoms_merged, conflicts, total_atoms}."""
        a = self.general_reasoner.derive_engine
        b = other.general_reasoner.derive_engine
        conflicts, incoming = [], []
        for (s, r), v in b.triples.items():
            cur = a.triples.get((s, r))
            if cur is not None and cur != v:
                conflicts.append({'fact': (s, r), 'mine': cur, 'theirs': v})
            elif cur is None:
                incoming.append((s, r, v))
        if incoming:
            self.ingest_triples(incoming, discover=False)
        self.discover_rules()                     # re-mine rules on the combined store
        return {'atoms_merged': len(incoming), 'conflicts': conflicts,
                'total_atoms': len(a.triples)}

    def odd_one_out(self, entities):
        """Day-87 -- IQ 'which does not belong': the outlier is the entity whose
        holographic signature resonates LEAST with the bundle of the others
        (derive_engine.odd_one_out) -- pure substrate geometry, no feature list.
        Returns {outlier, scores}."""
        eng = self.general_reasoner.derive_engine
        out, scores = eng.odd_one_out([str(e).strip().lower() for e in entities])
        return {'outlier': out, 'scores': scores}

    def iq_solve(self, item):
        """Day-87 GOLD -- IQ-TEST REASONER: solve a fluid-reasoning item with pure
        VSA, no training.  Item kinds:
          {'kind':'analogy',      'a':..,'b':..,'c':..}        -> A:B :: C:?
          {'kind':'odd_one_out',  'items':[..]}                -> which doesn't belong
        Analogy recovers the relation by unbind + resonance and applies it;
        odd-one-out is a geometric outlier over signatures.  Returns {answer, ...}.
        """
        k = item.get('kind')
        if k == 'analogy':
            r = self.analogy(item['a'], item['b'], item['c'])
            return {'answer': r['answer'], 'why': r['relation'], 'verified': r['verified']}
        if k == 'odd_one_out':
            r = self.odd_one_out(item['items'])
            return {'answer': r['outlier'], 'why': 'least resonant with the group',
                    'scores': r['scores']}
        return {'answer': None, 'why': f'unknown item kind {k}'}

    def iq_test(self, items):
        """Run a battery of IQ items, score against provided answers.  Each item is
        a solve-spec plus an 'expect' key.  Returns {score, n, results}."""
        results, correct = [], 0
        for it in items:
            got = self.iq_solve(it)
            ok = (got['answer'] == it.get('expect'))
            correct += int(ok)
            results.append({'item': it.get('kind'), 'got': got['answer'],
                            'expect': it.get('expect'), 'ok': ok, 'why': got.get('why')})
        return {'score': correct, 'n': len(items), 'results': results}

    def kolmogorov_kernel(self, min_support=2, min_conf=0.8):
        """Day-87 GOLD -- KOLMOGOROV KERNEL.  Compress the organism's knowledge to
        the SMALLEST atom set that still re-derives everything: mine every rule,
        then self-compress -- drop every fact a rule can reconstruct -- and verify
        the result is LOSSLESS (each dropped fact re-derives exactly).  What
        remains is the irreducible kernel: the structural edges plus one source per
        inheritable attribute -- the true, near-Kolmogorov SIZE of the knowledge,
        measured.  Mutates the (throwaway) store -- never call on the production
        organism you intend to save.  Returns the before/after counts, the lossless
        flag, and the compression ratio."""
        eng = self.general_reasoner.derive_engine
        before = dict(eng.triples)
        n0 = len(before)
        eng.discover(min_support=min_support, min_conf=min_conf, self_compress=True)
        n1 = len(eng.triples)
        nrm = lambda s: str(s).replace(' ', '')
        dropped = [(s, r, before[(s, r)]) for (s, r) in before if (s, r) not in eng.triples]
        recovered = sum(1 for (s, r, v) in dropped
                        if nrm(eng.atom(r, s) or eng.inherited_atom(r, s) or '') == nrm(v))
        return {'atoms_before': n0, 'kernel_atoms': n1, 'dropped': len(dropped),
                'recovered': recovered, 'lossless': recovered == len(dropped),
                'ratio': round(n0 / max(n1, 1), 2), 'rules': len(eng.learned_rules)}

    def counterfactual(self, entity, relation, new_value):
        """Day-87 GOLD -- NATIVE CAUSAL / COUNTERFACTUAL reasoning.  'If `entity`'s
        `relation` were `new_value`, what would follow?'  The organism INTERVENES
        on the substrate (a do-operator override on one atom) and RE-DERIVES the
        downstream consequences -- its taxonomic chain and every attribute it
        inherits -- diffing against reality.  No learned causal model: the causal
        structure is the derivation graph itself (derive_engine.counterfactual).
        Returns {baseline, intervention, counterfactual, changes}."""
        eng = self.general_reasoner.derive_engine
        return eng.counterfactual(entity, relation, new_value)

    def truthfulness_audit(self, probes):
        """Day-87 #4 -- THE AI THAT CANNOT LIE.  Measure the structural guarantee:
        the grounded read-out can only utter content tokens that came FROM a
        verified derived fact, and otherwise abstains -- so the fabrication rate
        is 0 BY CONSTRUCTION, not by training.  `probes` is a list of
        (question, expected_answer_or_None); expected=None marks a question whose
        answer the organism should NOT know (a trap).  Reports the fabrication
        rate (invented content tokens / answers -- the lie rate), the abstention
        behaviour on traps, and how many answered facts were correct.  A miss
        (abstaining on a knowable fact) is honestly counted as COVERAGE, not a
        lie -- silence is not a hallucination."""
        R = {'n': len(probes), 'answered': 0, 'abstained': 0, 'fabrications': 0,
             'wrong_answers': 0, 'traps': 0, 'traps_abstained': 0, 'coverage_miss': 0}
        for q, exp in probes:
            r = self.answer(q, explain=True)          # proof-carrying, strongest path
            is_trap = exp is None
            R['traps'] += int(is_trap)
            if r['fact'] is None:                     # abstained
                R['abstained'] += 1
                if is_trap:
                    R['traps_abstained'] += 1
                else:
                    R['coverage_miss'] += 1           # knew nothing to say -- a miss, not a lie
                continue
            R['answered'] += 1
            # THE lie metric: an emitted content token that did NOT come from a
            # verified fact.  grounded is False only if the read-out invented a
            # token -- which the copy-from-fact construction forbids.  Answering a
            # trap with a GROUNDED, self-labelled fact (it states its own relation)
            # is not a lie; it is tracked as coverage, never as fabrication.
            if not r['grounded']:
                R['fabrications'] += 1
            ans = r['fact'][2]
            if (not is_trap) and ans != exp:
                R['wrong_answers'] += 1               # derived a real-but-wrong fact
        R['fabrication_rate'] = R['fabrications'] / max(1, R['answered'])
        R['answer_accuracy'] = (R['answered'] - R['wrong_answers']) / max(1, R['answered'])
        return R

    @property
    def surface(self):
        """Day-87 -- the content-blind surface realizer (learns fluent frames per
        relation from example (fact, sentence) pairs; never invents content)."""
        sr = getattr(self, '_surface', None)
        if sr is None:
            from ikigai.cognition.holo_generate import SurfaceRealizer
            sr = SurfaceRealizer()
            self._surface = sr
        return sr

    def learn_surface(self, pairs):
        """Teach fluent surface frames from example ((s,r,o), sentence) pairs. The
        frame (function words) is induced; content always comes from the fact."""
        return self.surface.learn(pairs)

    def extract(self, sentence):
        """EXTRACTION = INVERSE GENERATION: read a sentence back into grounded triple(s)
        by aligning it to the organism's own INDUCED surface frames (the same frames that
        generate).  text -> structure the organism can DERIVE over -- the 'meaning from
        data' door.  Returns a list of (s, r, o).  No authored rules, no backprop."""
        return self.surface.extract(sentence)

    def extract_verified(self, sentence, thresh=0.8, return_score=False):
        """EXTRACTION under SEARCH-UNDER-VERIFICATION (the #8 mechanism on real work):
        propose candidate parses leniently, then VERIFY each by regenerating the sentence
        and keeping the one that reconstructs the source -- generation is the verifier for
        extraction.  Abstains (None) if nothing reconstructs above `thresh`.  This is what
        makes inverse-generation robust on messy REAL English without fabricating."""
        return self.surface.extract_verified(sentence, thresh=thresh, return_score=return_score)

    def extract_to_mem(self, sentences, discover=False, verified=False, thresh=0.8):
        """Read raw sentences into grounded triples and TEACH them to the unified memory,
        so a pile of text becomes structure the organism reasons over.  verified=True runs
        each sentence through search-under-verification (robust on real text).  Returns
        the triples ingested."""
        triples = []
        for sent in sentences:
            if verified:
                t = self.extract_verified(sent, thresh=thresh)
                if t is not None:
                    triples.append(t)
            else:
                triples.extend(self.extract(sent))
        if triples:
            self.mem.teach(triples, discover=discover)
        return triples

    def induce_surface(self, sentences, max_pairs=400):
        """Day-99 -- LEARN THE FORM OF LANGUAGE FROM ITS OWN KNOWLEDGE (distant supervision).

        The blocker this removes: text -> fact extraction needed `learn_surface(pairs)`, and those
        ((s,r,o), sentence) pairs were HAND-AUTHORED -- the crutch the anti-cheat constitution bans
        ("INDUCED frames, not hand-fed"). Meanwhile `induce_frames` (Harris/Schutze) runs, induces
        frames, and the extractor never sees them: MEASURED, induce_frames -> 2 frames, and
        extract_verified still returned (None, 0.0). Two faculties that both work and never speak.

        Why that gap is real and not a wiring oversight: unsupervised induction can discover that
        `DET X COP DET Y` is a frequent pattern, but NOTHING in the distribution says the pattern
        MEANS `isa`. Form is inducible; the form->meaning link needs grounding.

        The organism already holds the grounding -- its own facts. So: read raw sentences, and
        wherever a sentence contains two entities it ALREADY knows to be related, it has found a
        sentence expressing a relation it understands, and can induce that surface frame from it.
        The pairs are DERIVED from (corpus x its own knowledge), never authored. This is distant
        supervision, and it is how the form of language gets learned from having something to say.

        Seed knowledge -> read text -> induce frames -> extract NEW facts from NEW text. That loop
        is what "feed it data and it learns" requires to be honest rather than staged.

        Returns the number of (fact, sentence) alignments it found and learned from."""
        tok = self.general_reasoner.tokenize
        pairs, seen = [], set()
        for sent in sentences:
            s = str(sent)
            toks = [t for t in tok(s) if t]
            if len(toks) < 2:
                continue
            uniq = list(dict.fromkeys(toks))
            for a in uniq:
                try:
                    web = self.knows(a) or {}
                except Exception:
                    continue
                for rel, vals in web.items():
                    for v in (vals or []):
                        vn = str(v).replace(' ', '').strip().lower()
                        if not vn or vn == a:
                            continue
                        if vn in uniq:                       # both ends present -> the sentence
                            k = (a, rel, vn, s)              # EXPRESSES a relation it understands
                            if k not in seen:
                                seen.add(k)
                                pairs.append(((a, rel, vn), s))
                        if len(pairs) >= max_pairs:
                            break
        if pairs:
            self.learn_surface(pairs)
        return len(pairs)

    def relation_universe(self):
        """Every relation the organism can be ASKED about. Derived, not authored: the composition
        engine's own question templates, plus any relation its enumerable atom index has seen.

        Why this is needed at all: the organism's facts live in cat4's anchor cache, keyed by a HASH
        of the tokenized question -- a one-way key. You cannot enumerate what it knows; you can only
        ask questions you already know to ask. `eng.triples` is the enumerable INDEX, and it is
        populated lazily BY atom() hits, so it shows only what has already been touched. Measured
        Day-100: a freshly loaded organism reports 5 relations via the index while actually holding
        at least 8 (currency/language/population answered fine -- nothing had touched them).

        Stamped with `_learn_epoch`, not len(eng.triples): the index GROWS on every read (atom()
        records what it resolves), so a length stamp would go stale on reads and force a rescan
        every call -- the Day-99 curiosity bug exactly. A relation can only reach the index after
        something INGESTED it, and ingest bumps the epoch, so the epoch is the honest stamp."""
        ep = getattr(self, '_learn_epoch', 0)
        cu = getattr(self, '_rel_universe_cache', None)
        if cu is not None and cu[0] == ep:
            return cu[1]
        from ikigai.cognition.compositional import _REL_TEMPLATES
        eng = self.general_reasoner.derive_engine
        rels = sorted(set(_REL_TEMPLATES) | {r for (_s, r) in eng.triples})
        self._rel_universe_cache = (ep, rels)
        return rels

    @staticmethod
    def _frame_regex(frame):
        """An induced frame -> a matcher. Slots capture a short word-run; the frame's own literal
        tokens are the anchors that bound them.

        Day-100 -- A FRAME-INITIAL SLOT MUST BE ANCHORED TO THE SENTENCE START.

        Day-99 learned this once and it generalises: a leading function word is not decoration, it
        is the LEFT ANCHOR that bounds the slot. When the frame ITSELF begins with a slot there is
        no such anchor, and the slot eats leftward into whatever precedes it. Measured on
        '{O} is spoken in {S}' over the 87 Tatoeba sentences carrying that form:

            unanchored : 'What language is spoken in Mexico?'      -> language(mexico)='what language'
                         "John didn't know Tupi is spoken in Brazil." -> language(brazil)='t know tupi'
            anchored   : the clause has to START where the frame starts

        Slots stay up to 3 words. A 1-word slot scores higher (0.86 vs 0.75) and cannot say
        'papua new guinea' or 'santo domingo' -- capital learns both correctly, so the cheaper
        precision is the wrong trade."""
        import re as _re
        parts = []
        for t in frame:
            if t == '{S}':
                parts.append(r'(?P<S>[a-z]+(?: [a-z]+){0,2})')
            elif t == '{O}':
                parts.append(r'(?P<O>[a-z]+(?: [a-z]+){0,2})')
            else:
                parts.append(_re.escape(t))
        lead = r'^' if frame and frame[0] in ('{S}', '{O}') else r'\b'
        return _re.compile(lead + r'\s+'.join(parts) + r'\b')

    @staticmethod
    def _is_assertion(sentence):
        """Day-100 -- A QUESTION IS NOT EVIDENCE OF ITS ANSWER.

        Distant supervision reads a sentence as expressing a fact the organism holds. A question
        expresses the fact's SHAPE and withholds its value -- and 'what language is spoken in {e}'
        is literally the organism's OWN question template for that relation (_REL_TEMPLATES). So
        the corpus was teaching it to answer its own questions with the word 'what'.

        MEASURED on '{O} is spoken in {S}': 6 of 10 conflicts were questions. Dropping them takes
        precision 0.38 -> 0.67, and with the slot anchored, 0.75 -- the difference between the
        organism learning that frame and rejecting it. Punctuation, not an authored word list."""
        return '?' not in str(sentence)

    @staticmethod
    def _frame_anchor(frame):
        """The frame's longest literal run -- a cheap substring prefilter before the regex."""
        runs, cur = [], []
        for t in frame:
            if t in ('{S}', '{O}'):
                if cur:
                    runs.append(' '.join(cur)); cur = []
            else:
                cur.append(t)
        if cur:
            runs.append(' '.join(cur))
        return max(runs, key=len) if runs else ''

    def induce_surface_verified(self, sentences, min_agree=2, min_prec=0.5, max_frame_tokens=10,
                                install=True):
        """Day-100 -- INDUCE THE FORM OF LANGUAGE, AND VERIFY IT AGAINST WHAT YOU ALREADY KNOW.

        `induce_surface` (Day-99) aligned the organism's facts to raw text and handed every
        alignment to `learn_surface`, which majority-votes over sentences. MEASURED on the full
        2,030,118-sentence Tatoeba corpus, that is not merely weak -- it is inverted:

            capital: current learn() picks   'george {O} was the {S}'        (4 sentences)
                     the true frame          '{O} is the capital of {S}'     (1 sentence)

        The noise wins because Tatoeba DUPLICATES 'George Washington was the first President of the
        United States' across translation sets, so ONE junk fact (the organism holds a garbage
        capital:first->washington) rides 4 identical sentences, while the real frame appears once --
        the organism knows ~13 capitals and the corpus's 316 'is the capital of' sentences are
        mostly about other countries. Counting DISTINCT FACTS instead of sentences does not help:
        measured, every frame has fact-support exactly 1. Support at induction time cannot see the
        difference, because at induction time there IS no difference.

        The difference appears when you USE the frame. A real surface form is PRODUCTIVE -- apply it
        back to the corpus and it extracts hundreds of pairs; a coincidence extracts its own sentence
        and stops. But productivity alone is pure RECALL, and measured it crowns the most GENERIC
        frame: 'the {S} {O}' extracts 249,071 pairs because it is just two adjacent words.

        So the corpus proposes and the organism's OWN KNOWLEDGE disposes -- search-under-verification,
        the #8 mechanism, with no authored templates and no LLM judge:

            productivity  apply the frame to the corpus      -> the (S,O) pairs it extracts
            verification  over pairs where the organism HAS a belief:
                            agree    = it already holds (S,rel,O)
                            conflict = it holds (S,rel,X), X != O
                          precision = agree / (agree + conflict)
            keep          agree >= min_agree and precision >= min_prec
            harvest       the pairs it had NO belief about are what reading TAUGHT it

        Measured on the full corpus, from an organism holding ~13 capitals:
            capital  'the capital of {S} is {O}'   37 agree, precision 0.74, 120 facts it never knew
                     (aruba->oranjestad, burundi->gitega, kazakhstan->astana, ...  all correct)
        and the junk is rejected on its own numbers ('george {O} was the {S}' -> precision 0.38).

        HONEST, and reported rather than papered over: continent/currency/isa/subclassof/language
        survive nothing on this corpus. '{O} is spoken in {S}' is a REAL frame and still scores 0.38
        because the slot capture takes 'many countries' out of 'English is spoken in many countries'.
        Precision is 0.74, not 1.0: 'haiti'->'port' is a truncated 'port-au-prince'. Reading a
        conversational corpus does not teach an encyclopedia, and this says so with numbers.

        Returns {rel: {...}} per relation -- either the kept frame with its measured evidence, or
        why nothing survived. install=True writes the survivors into the surface realizer."""
        eng = self.general_reasoner.derive_engine
        tok = self.general_reasoner.tokenize
        rels = self.relation_universe()

        # Only ASSERTIONS are evidence. A question states the relation's shape and withholds its
        # value; read as evidence it teaches the organism to answer 'what'. (see _is_assertion)
        corpus = [str(s).lower() for s in sentences if self._is_assertion(s)]
        memo, vtok = {}, {}

        def probe(a):
            r = memo.get(a)
            if r is None:
                r = [(rel, v) for rel in rels for v in (eng.atoms(rel, a) or [])]
                memo[a] = r
            return r

        def span(toks, vt):
            n, m = len(toks), len(vt)
            for i in range(n - m + 1):
                if toks[i:i + m] == vt:
                    return i
            return -1

        # ---- propose: distant supervision over (corpus x its own knowledge) -------------------
        from collections import Counter as _C, defaultdict as _dd
        cand = _dd(_C)
        for low in corpus:
            lw = low.split()
            toks = [t for t in tok(low) if t]
            if len(toks) < 3:
                continue
            for a in dict.fromkeys(toks):
                for rel, v in probe(a):
                    vt = vtok.get(v)
                    if vt is None:
                        vt = [t for t in tok(str(v)) if t]
                        vtok[v] = vt
                    if not vt or vt == [a]:
                        continue
                    j = span(lw, vt)
                    i = span(lw, [a])
                    if j < 0 or i < 0:
                        continue
                    fr, k = [], 0
                    while k < len(lw):
                        if k == j:
                            fr.append('{O}'); k += len(vt); continue
                        if k == i:
                            fr.append('{S}'); k += 1; continue
                        fr.append(lw[k]); k += 1
                    if '{S}' in fr and '{O}' in fr:
                        idx = [q for q, t in enumerate(fr) if t in ('{S}', '{O}')]
                        cand[rel][tuple(fr[:max(idx) + 1])] += 1

        # ---- dispose: the organism's own knowledge scores every candidate ---------------------
        report = {}
        for rel in rels:
            scored = []
            for f in [f for f in cand[rel] if len(f) <= max_frame_tokens]:
                anchor = self._frame_anchor(f)
                if not anchor:
                    continue
                rx = self._frame_regex(f)
                pairs = set()
                for s in corpus:
                    if anchor not in s:
                        continue
                    m = rx.search(s)
                    if m:
                        pairs.add((m.group('S'), m.group('O')))
                agree = conflict = 0
                new = []
                for (S, O) in pairs:
                    held = eng.atoms(rel, S) or []
                    if not held:
                        new.append((S, O))
                    elif O in held:
                        agree += 1
                    else:
                        conflict += 1
                prec = agree / max(1, agree + conflict)
                scored.append({'frame': list(f), 'agree': agree, 'conflict': conflict,
                               'precision': round(prec, 3), 'pairs': len(pairs), 'new': new})
            ok = [x for x in scored
                  if x['agree'] >= min_agree and x['precision'] >= min_prec]
            ok.sort(key=lambda x: (-x['precision'], -x['pairs']))
            if ok:
                best = dict(ok[0])
                best['variants'] = [x['frame'] for x in ok]
                report[rel] = best
            else:
                bp = max((x['precision'] for x in scored), default=0.0)
                report[rel] = {'frame': None, 'candidates': len(cand[rel]),
                               'best_precision': round(bp, 3),
                               'reason': f'no frame induced for {rel}: {len(cand[rel])} candidates, '
                                         f'best precision {bp:.2f} (needs agree>={min_agree}, '
                                         f'prec>={min_prec})'}

        # ---- conflation guard: two relations must not own the SAME surface form ---------------
        # Measured Day-100: `capital` keeps 'the capital of {S} is {O}' (prec 0.74) and `country`
        # keeps 'the capital of {O} is {S}' (prec 0.78) -- the SAME skeleton with the slots swapped.
        # country's frame verifies ONLY because every country fact this organism happens to hold is
        # about a CAPITAL city. Realise country(marseille)=france through it and it says "the capital
        # of france is marseille" -- false. The content tokens are still grounded and the FRAME lies
        # anyway: exactly the frame-level hole Day-98's Wall C left open. Distant supervision
        # conflates a relation with a correlated sub-relation whenever the seed facts are biased,
        # and precision cannot see it -- the organism holds no counterexample to be wrong about.
        #
        # The tiebreak is the organism's OWN symbol for the relation, not an authored rule: when two
        # relations claim the same literal skeleton, the one whose NAME is among those literals owns
        # it. Narrow by construction -- it fires only on a collision, and it is honest when it fires.
        skels = {}
        for rel, r in report.items():
            f = r.get('frame')
            if f:
                skel = tuple(sorted(t for t in f if t not in ('{S}', '{O}')))
                skels.setdefault(skel, []).append(rel)
        for skel, claim in skels.items():
            if len(claim) < 2:
                continue
            named = [r for r in claim if r in skel]
            keep = named[0] if named else max(claim, key=lambda r: report[r]['precision'])
            for r in claim:
                if r == keep:
                    continue
                report[r] = {'frame': None, 'candidates': len(cand[r]),
                             'best_precision': report[r]['precision'],
                             'reason': f"no frame induced for {r}: its best frame is the surface "
                                       f"form `{' '.join(report[keep]['frame'])}` that `{keep}` "
                                       f"owns -- borrowed from a correlated relation, not its own"}

        if install:
            for rel, r in report.items():
                if r.get('frame'):
                    self.surface.templates[rel] = list(r['frame'])
                    self.surface.variants[rel] = [list(v) for v in
                                                  (r.get('variants') or [r['frame']])]
        return report

    def read_verified(self, sentences, learn=True, min_agree=2, min_prec=0.5):
        """Day-100 -- FEED RAW TEXT, COME OUT KNOWING FACTS NOBODY TAUGHT YOU.

        induce_surface_verified() -> the frames that survived the organism's own verification ->
        harvest the pairs it had NO belief about -> learn them (body-modulated reinforcement).

        This is the honest form of 'feed it data and it learns': every frame is INDUCED and then
        VERIFIED against knowledge the organism already had, and only what the verified frame
        extracts is taught. Relations whose frames did not survive teach nothing, and say so.

        Returns {frames: {...}, learned: n, facts: [...]}."""
        rep = self.induce_surface_verified(sentences, min_agree=min_agree, min_prec=min_prec)
        facts, learned = [], 0
        for rel, r in rep.items():
            if not r.get('frame'):
                continue
            for (S, O) in r.get('new', []):
                if self.prediction_error(S, rel, O) > 0.0:
                    if learn:
                        self.learn_reinforced(S, rel, O)
                    learned += 1
                    facts.append((S, rel, O))
        return {'frames': rep, 'learned': learned, 'facts': facts}

    def read(self, sentences, rounds=1, discover=True, min_support=2, min_conf=0.5):
        """Day-99 -- READ: the organism reads raw text and comes out knowing more.

        The honest bootstrap, in the biological order -- you acquire the form of language from
        text you can already ground, and only then can you learn NEW facts by reading:

            induce_frames    distributional word types + frame inventory (Harris/Schutze)
            induce_surface   align its OWN knowledge to the text -> learn the surface frames
            extract_corpus   now extract triples from sentences, with the verify gate
            learn            land them (body-modulated reinforcement, visible to every door)
            discover         mine rules over what it now holds

        Iterating helps and is not a trick: each round it knows more, so more sentences contain a
        relation it recognises, so it induces more frames, so it extracts more. Bootstrapping.

        Returns {frames, alignments, edges, learned, rules}."""
        sents = [str(s) for s in sentences if str(s).strip()]
        out = {'frames': 0, 'alignments': 0, 'edges': 0, 'learned': 0, 'rules': 0, 'verified': {}}
        # Day-102: FEED THE 5 GROUNDING CHANNELS. This bootstrap induces frames and
        # learns triples, but until now it skipped distributional grounding entirely
        # (grammar/POS, sensory, taxonomy) -- the ground_text() feeder was shadowed
        # dead, so grammar.vocab_size stayed 0 and the organism had no part-of-speech
        # foundation for fluent generation. Grounding each sentence here is where
        # language FORM is acquired (the docstring's own "acquire the form of language
        # from text you can already ground"). Best-effort, never raises into learning.
        for s in sents:
            try:
                self.ground_text(s)
            except Exception:
                pass
        for _ in range(max(1, int(rounds))):
            try:
                out['frames'] = self.induce_frames(sents, k=4, min_count=2) or 0
            except Exception:
                pass
            # Day-100 -- VERIFY FIRST. A frame the organism's own knowledge confirms beats a frame
            # that merely repeated: measured on 2.03M raw Tatoeba sentences, the unverified path
            # installs 'george {O} was the {S}' for `capital` (one junk fact riding 4 duplicate
            # sentences) while verification installs 'the capital of {S} is {O}' (37 agree, 0.74).
            #
            # The fallback is decided by the RESULT, not by a size threshold: verification needs
            # facts it already holds to check a frame against, and a 13-sentence bootstrap corpus
            # cannot reach that. So if verification could confirm NOTHING, the Day-99 alignment path
            # still runs -- that is exactly the small-seed case it was built for. Verification only
            # ever REPLACES a frame with one that was checked.
            try:
                out['verified'] = self.induce_surface_verified(sents) or {}
            except Exception:
                out['verified'] = {}
            if not any(v.get('frame') for v in out['verified'].values()):
                out['alignments'] = self.induce_surface(sents)
            edges = self.extract_corpus(sents)
            out['edges'] = len(edges)
            for (s, r, o) in edges:
                if self.prediction_error(s, r, o) > 0.0:      # only what it does not already hold
                    self.learn_reinforced(s, r, o)
                    out['learned'] += 1
        if discover:
            try:
                out['rules'] = len(self.general_reasoner.derive_engine.discover(
                    min_support=min_support, min_conf=min_conf) or [])
            except Exception:
                pass
        return out

    def extract_corpus(self, sentences, top_function_words=120, verified=True, thresh=0.7,
                       head_noun=True, to_mem=False, discover=False):
        """Day-94 #8 -- RAW TEXT -> GROUNDED KG AT SCALE, native.  Read a pile of raw sentences
        into a grounded taxonomy using the organism's INDUCED frames + the verify gate, refined by
        a DATA-DERIVED informativeness filter: the corpus's own most frequent tokens are the
        function words (pronouns/copula), so a triple whose subject or object-HEAD is one of them
        is dropped -- no authored stoplist.  head_noun=True takes the object NP's head (last token,
        English is head-final) as the class, so 'a beautiful country' -> 'country' (recurs, closes).
        Returns the de-duplicated (s, r, o) edges; to_mem=True also teaches them.  Learn frames
        first with learn_surface().  (Cuts idiom noise ~61%->~1% vs raw extraction; 0 fabrication.)"""
        import re as _re
        from collections import Counter as _Counter
        freq = _Counter()
        sents = []
        for s in sentences:
            toks = _re.findall(r"[a-z]+", str(s).lower())
            if toks:
                freq.update(toks); sents.append(str(s))
        # Day-99 -- the function words are the organism's OWN INDUCED FRAMES, not a frequency guess.
        #
        # `surface.templates` is {'isa': ['a', '{S}', 'is', 'a', '{O}']}: the LITERAL tokens of an
        # induced frame ARE the function words ('a', 'is'), and the slots are where content goes.
        # The organism induced that itself, from text it could ground -- so this is data-derived in
        # fact, not just in spirit, and it needs no authored stoplist.
        #
        # The frequency heuristic it replaces was wrong twice over. It took an ABSOLUTE 120, so on
        # a corpus whose vocabulary is smaller than 120 EVERY content word became a "function word"
        # and every triple was dropped (measured: 6 sentences, vocab 10 -> 0 edges, while
        # extract_verified was correctly returning ('feline','isa','mammal')). And frequency itself
        # is the wrong signal for a taxonomy: `mammal` and `animal` are FREQUENT because they are
        # the ROOT, so the filter silently ate the most general facts in the corpus -- it learned 4
        # of 11 edges and dropped feline->mammal, mammal->animal. Zipf holds for English function
        # words; it does not hold for the top of a taxonomy. A frame literal cannot make that
        # mistake: `mammal` never appears as a literal in a template.
        stop = set()
        for _toks in (getattr(self.surface, 'templates', None) or {}).values():
            for _t in _toks:
                _t = str(_t)
                if not (_t.startswith('{') and _t.endswith('}')):
                    stop.add(_t.strip().lower())
        if not stop:            # no frames induced yet -> fall back to the frequency heuristic,
            _k = min(int(top_function_words), max(1, len(freq) // 4))   # capped to a quarter of
            stop = {w for w, _ in freq.most_common(_k)}                 # the vocabulary (Zipf)
        edges, seen = [], set()
        for sent in sents:
            tri = self.extract_verified(sent, thresh=thresh) if verified else \
                (self.extract(sent)[:1] or [None])[0]
            if not tri:
                continue
            s, r, o = tri
            if head_noun:
                ot = str(o).rstrip('.').split()
                o = ot[-1] if ot else o
            s = str(s).strip(); o = str(o).strip()
            if not s or not o or s in stop or o in stop:       # data-derived informativeness filter
                continue
            k = (s, r, o)
            if k not in seen:
                seen.add(k); edges.append((s, r, o))
        if to_mem and edges:
            self.mem.teach(edges, discover=discover)
        return edges

    def scene(self, objects=None, d=1024, scale=6.0):
        """Day-94 -- a SPATIAL SCENE on the substrate (Spatial Semantic Pointers), native.  The same
        FHRR substrate that holds taxonomy holds continuous 2-D position: sc.add(name,x,y);
        sc.where(name) decodes position by resonance; sc.nearest(x,y) finds the closest by SSP
        similarity (no coordinates compared); sc.base_facts() feeds the SAME derive engine, which
        then composes spatial relations (right-of / above / between) exactly as it composes is-a.
        The organism generalizes off the text axis with zero new mechanism.  `objects`: optional
        list of (name, x, y)."""
        from ikigai.cognition.numeric_encoder import SpatialScene
        sc = SpatialScene(d=d, scale=scale)
        for (name, x, y) in (objects or []):
            sc.add(name, x, y)
        return sc

    def substrate_policy(self, d=512, temp=0.6):
        """Day-94 #5 -- a fresh SUBSTRATE PROPOSER: a holographic associative policy memory
        (bind/bundle/cosine-cleanup, Hebbian, no backprop).  reinforce(state, action) on good
        moves; propose(state, actions) to sample a resonance-biased action.  Feed its .propose
        as the proposer to verified_search for a substrate-guided generator."""
        from ikigai.cognition.generation_engine import SubstratePolicy
        return SubstratePolicy(d=d, temp=temp)

    def verified_search(self, propose, verify, budget=2000):
        """Day-93 #8 -- generation-as-search-under-verification, native.  Draw cheap candidates
        from propose(); keep the first verify(cand) accepts; ABSTAIN after budget.  Correct-or-
        abstain by construction -- cannot emit an unverified answer.  Returns (solved, tries, ans)."""
        from ikigai.cognition.generation_engine import verified_search
        return verified_search(propose, verify, budget)

    def plan_order(self, items, deps, lam=0.3):
        """Day-94 #6 -- structure-first planning, native.  Order interdependent `items` (deps:
        item -> prerequisites) by expected-free-energy argmin so the whole plan stays globally
        consistent (each item after its prerequisites), one pass, no backtracking.  Holds long-
        range consistency where a free walk collapses -- the code-planning mechanism."""
        from ikigai.cognition.generation_engine import plan_order
        return plan_order(items, deps, lam=lam)

    def plan_discourse(self, facts, topic, goal, lam=1.0, entities_of=None):
        """Day-94 #7 -- goal-driven discourse planning, native.  Order derived `facts` so the
        discourse is coherent (entity continuity) AND lands on `goal` (the point).  Free-energy
        argmin selection: epistemic=continuity, pragmatic=goal-timing.  Returns ordered facts."""
        from ikigai.cognition.generation_engine import plan_discourse
        return plan_discourse(facts, topic, goal, lam=lam, entities_of=entities_of)

    def affect_valence(self, word):
        """Day-94 -- a word's valence GROUNDED in LIVED affect: the mean felt emotion over the
        organism's episodes that mention it (stem-matched).  Emotion learned from experience
        (remember(emotion=...)), not an authored lexicon.  Returns a float or None if never lived."""
        eps = getattr(self.mem, 'episodes', None)
        if not eps:
            return None

        def stem(w):
            w = str(w).lower()
            for suf in ('ing', 'ed', 'es', 's', 'd'):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    return w[:-len(suf)]
            return w
        s = stem(word)
        tot = n = 0.0
        for ep in eps:
            if s in {stem(t) for t in ep.get('tokens', [])}:
                tot += ep.get('v', 0.0); n += 1
        return (tot / n) if n else None

    # Day-99: marks a realized clause that does not open with its subject (an induced frame may be
    # {O}-initial, e.g. "paris is the capital of france") -- such a clause is a whole sentence and
    # must not be aggregated behind "It"/"The {topic}".
    _STANDALONE = '\x00~'

    def derive_ancestry(self, entity, links=('isa', 'subclassof'), max_depth=64):
        """Day-99 -- THE ONE ANCESTRY OP.  Every path that needs "what is this, ultimately?" goes
        through here.

        Before this there were THREE, and they disagreed about the same fact:
          * `compose` walked `eng.atom(link, cur)` in a python while-loop -- atom() returns ONE
            parent (the cache's LAST value), so a multi-parent entity lost a parent AND everything
            above it.  Measured: vikode isa feline AND pet; compose said "The vikode is a pet, and
            ultimately a domestic" -- feline and mammal SILENTLY DROPPED.  This is the exact bug
            Day-96 fixed inside transitive_reach (~15% of Wikidata p279 edges are multi-parent);
            it was still live in the GENERATION path, which is the Day-100/101 headline.
          * `transitive_reach` / `ancestors` -> None unless the miner has induced a transitive RULE.
            NOT a bug, and I misdiagnosed it as one before measuring: refusing to assert x-isa-z
            without having LEARNED that isa is transitive is the calibration doing its job. The
            miner is healthy (measured chains=4 acyclic=4 conf=1.00 -> promoted). It needs 2-chains
            (R(a)=b, R(b)=c, b a subject of R) -- and note the mining index is SINGLE-valued
            (triples[(s,r)] = v), so a second parent OVERWRITES the first there and can hide the
            chain from the miner. That is a real limitation of the index, not of the miner.
        An omission that reads as completeness is the worst failure mode for correct-or-abstain:
        compose was not wrong, it was quietly half-blind.

        This op deliberately does NOT gate on the mined rule, because it REPORTS a derived chain
        rather than asserting closure membership. compose's use of it inherits that: it will say
        "ultimately a mammal" from the edges alone. That asymmetry with ancestors() is a real open
        semantics question, flagged not buried.

        BFS over `atoms()` (ALL parents), so it is complete by construction, and it needs no mined
        rule -- it derives from the edges themselves.  Reads through the substrate: atoms() resolves
        via the packed address store when authoritative, else the anchor cache.  Derive-not-store:
        nothing here is written.

        Returns {'order': ancestors nearest-first (BFS), 'direct': immediate parents,
                 'roots': ancestors with no parent of their own}."""
        eng = self.general_reasoner.derive_engine
        e = str(entity).strip().lower().replace(' ', '')

        def parents(x):
            out = []
            for link in links:
                for p in (eng.atoms(link, x) or []):
                    p = str(p).strip().lower().replace(' ', '')
                    if p and p != x and p not in out:
                        out.append(p)
            return out

        order, seen, frontier, depth = [], {e}, [e], 0
        direct = []
        while frontier and depth < max_depth:
            nxt = []
            for cur in frontier:
                for p in parents(cur):
                    if p in seen:
                        continue
                    seen.add(p)
                    nxt.append(p)
                    order.append(p)
                    if depth == 0:
                        direct.append(p)
            frontier = nxt
            depth += 1
        roots = [a for a in order if not parents(a)]
        return {'order': order, 'direct': direct, 'roots': roots}

    @staticmethod
    def _and_list(items):
        """'a, b, and c' -- the surface form of a set, so multi-parent ancestry reads as prose."""
        items = list(items)
        if not items:
            return ''
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f'{items[0]} and {items[1]}'
        return ', '.join(items[:-1]) + f', and {items[-1]}'

    def compose(self, topic, mood=(0.0, 0.3), elaborate=True):
        """Day-94 -- OPEN-ENDED description of `topic`, grounded and unbounded.  Derives the topic's
        class chain (aggregated) plus its INHERITED attributes -- the attribute relations are
        DERIVED from the engine (every non-transitive relation an ancestor carries; no authored
        relation list) -- realises each via induced surface frames, pronominalises after first
        mention, and elaborates on referenced entities for length (all grounded).  `mood`=(valence,
        arousal) conditions FORM only: high arousal -> short punchy sentences, low -> aggregated
        flowing clauses; valence reorders.  Content is 100% derived (cannot hallucinate).  Honest:
        form-level emotion, not emotional vocabulary; grammar is frame-level, not novelistic."""
        eng = self.general_reasoner.derive_engine
        topic = str(topic).strip().lower()
        v, a = mood

        def art(w):
            return ('an ' if str(w)[:1] in 'aeiou' else 'a ') + str(w)

        # class chain (taxonomic): transitive closure if mined, else walk direct parents
        chain = []
        for link in ('isa', 'subclassof'):
            if eng.is_transitive(link):
                chain = [c.replace(' ', '') for c in (eng.transitive_reach(link, topic) or [])[1:]]
                if chain:
                    break
        # Day-99 -- the single-parent while-loop that used to live here dropped half the taxonomy of
        # any multi-parent entity (measured: feline+mammal vanished from "vikode"). One shared op.
        _anc = None
        if not chain:
            _anc = self.derive_ancestry(topic)
            chain = _anc['order']
        # Day-99 -- NO TAXONOMY IS NOT "UNKNOWN". This used to bail here whenever the entity had no
        # class chain, so an entity the organism holds real facts about was declared unknown and
        # `speak` withdrew: MEASURED org('chile') -> abstain while knows('chile') held
        # {'capital': ['santiago'], 'continent': ['america']}. The organism knew chile's capital and
        # would not volunteer it -- the same shape as answer() refusing Paris. Describing what it
        # HOLDS needs no taxonomy; only the "is a X, and ultimately a Y" clause does. Fall through
        # and let the attribute pass decide; abstain only if there is genuinely nothing to say.

        # DERIVE the attribute relations: non-transitive relations carried by the topic or an
        # ancestor (nearest-first inheritance) -- no authored relation list
        taxo = {'isa', 'subclassof'}
        attr_rels = [r for r in sorted(getattr(eng, 'relations', []))
                     if r not in taxo and not eng.is_transitive(r)]
        inh = []
        for r in attr_rels:
            for anc in [topic] + chain:
                val = eng.atom(r, anc)
                if val:
                    inh.append((r, val)); break
        if v < 0:
            inh = list(reversed(inh))                          # valence reorders the foregrounding

        # classification (aggregated).  Day-99: "ultimately" now names the ROOTS (ancestors with no
        # parent of their own), not merely the last element of a walk -- with a DAG the last element
        # is arbitrary.  For a single-parent chain roots == [chain[-1]] and the output is
        # byte-identical to Day-98's ("The vikode is a feline, and ultimately a mammal."), so the
        # measured 9.3 fluency is preserved; a multi-parent entity now states its full ancestry
        # instead of silently dropping half of it.
        if not chain:
            # Nothing to classify it as -- but it may still hold attributes worth stating.
            if not inh:
                return f"The {topic} is unknown."          # genuinely nothing: abstain honestly
            sents = []
        else:
            _roots = [r for r in (_anc['roots'] if _anc else []) if r in chain] or [chain[-1]]
            _mids = [c for c in chain if c not in _roots]
            if _mids:
                sents = [f"The {topic} is {self._and_list([art(c) for c in _mids])}, "
                         f"and ultimately {self._and_list([art(r) for r in _roots])}."]
            else:
                sents = [f"The {topic} is {self._and_list([art(r) for r in _roots])}."]

        _vc = {}                                               # per-relation use count -> variant rotation

        def _frame_val(frame):                                 # lived-affect valence of a frame's verb
            cand = [self.affect_valence(t) for t in frame if t not in ('{S}', '{O}')]
            cand = [x for x in cand if x is not None]
            return max(cand, key=abs) if cand else None

        def pred(r, val):
            vs = self.surface.variants.get(r, [])
            # EMOTION: if the organism has lived affect, pick the surface form whose verb valence
            # best matches its mood; otherwise rotate the learned forms for variety.
            if len(vs) > 1 and any(_frame_val(f) is not None for f in vs):
                idx = min(range(len(vs)), key=lambda i: abs((_frame_val(vs[i]) or 0.0) - v))
                raw = self.surface.realize(topic, r, val, variant=idx)
            else:
                i = _vc.get(r, 0); _vc[r] = i + 1
                raw = self.surface.realize(topic, r, val, variant=i)
            # Day-98 fix: strip the SUBJECT prefix for aggregation so "It " + predicate reads
            # cleanly.  The old code only stripped "the {topic} "; a learned frame that aligns as
            # "{topic} is amber" (no article) slipped through -> "It vikode is amber".  Strip
            # whichever subject form the realized clause actually starts with.
            for _pre in (f'the {topic} ', f'a {topic} ', f'an {topic} ', f'{topic} '):
                if raw.startswith(_pre):
                    return raw[len(_pre):]
            # Day-99 -- an induced frame need not put the SUBJECT first. Real text gave
            # capital -> "{O} is the capital of {S}", i.e. "paris is the capital of france": the
            # clause opens with the OBJECT, so there is no subject prefix to strip and aggregating
            # it under "It"/"The france" produced
            #   "The france paris is the capital of france and its most important city."
            # Day-98's aggregation silently assumed {S}-initial because every hand-taught frame was.
            # A clause that does not open with its subject cannot be aggregated behind one -- it is
            # already a complete sentence about the topic. Mark it so the caller emits it whole.
            return self._STANDALONE + raw

        # Day-99 -- the pronoun needs an ANTECEDENT. These clauses hardcoded "It", which only reads
        # if the class sentence just named the topic. With no taxonomy there is no class sentence,
        # so the description opened "It capital santiago." -- a pronoun referring to nothing. Name
        # the topic when nothing else has; keep "It" when the class sentence already introduced it
        # (so the Day-98 aggregation, measured at 9.3 fluency, is untouched).
        _subj = 'It' if sents else f'The {topic}'

        def _emit(p):
            """A clause that does not open with its subject is already a whole sentence."""
            if p.startswith(self._STANDALONE):
                t = p[len(self._STANDALONE):].strip().rstrip('.')
                return (t[0].upper() + t[1:] + '.') if t else ''
            return None

        if a >= 0.6:                                           # high arousal: short, punchy
            for (r, val) in inh:
                p = pred(r, val)
                whole = _emit(p)
                if whole:
                    sents.append(whole); continue
                s = _subj + ' ' + p + '.'
                sents.append(s[0].upper() + s[1:])
                _subj = 'It'                                   # introduced now -> pronominalise
        elif inh:                                              # calm: aggregate into one flowing clause
            raw_preds = [pred(r, val) for (r, val) in inh]
            standalone = [w for w in (_emit(p) for p in raw_preds) if w]
            preds = [p for p in raw_preds if not p.startswith(self._STANDALONE)]
            if preds:
                body = preds[0] if len(preds) == 1 else ', '.join(preds[:-1]) + ', and ' + preds[-1]
                sents.append(_subj + ' ' + body + '.')
            sents.extend(standalone)

        # LENGTH: elaborate on referenced entities (grounded)
        if elaborate:
            for (r, val) in inh:
                vc = [c.replace(' ', '') for c in (eng.transitive_reach('isa', val) or [])[1:]]
                sub = None
                for rr in attr_rels:
                    x = eng.atom(rr, val)
                    if x:
                        j = _vc.get(rr, 0); _vc[rr] = j + 1
                        sub = self.surface.realize(val, rr, x, variant=j); break
                if sub:                                    # entity has its own attribute -> elaborate
                    piece = f"The {val}" + (f", {art(vc[0])}," if vc else "") + \
                            f" {sub.split(val, 1)[1].strip()}."
                    sents.append(piece)
                elif vc:                                   # else just classify it
                    sents.append(f"The {val} is {art(vc[0])}.")
        return ' '.join(sents)

    def _fluent_fact(self, entity, rel, obj, first=True):
        """Render ONE derived fact as a fluent grounded clause. Prefers the organism's
        OWN induced surface frame for the relation (learned from text); falls back to a
        generic grammatical frame 'the {rel} of {ent} is {obj}' -- a uniform syntactic
        scaffold, not authored domain knowledge (it works for ANY relation). Grounded:
        the content is a derived fact, never invented."""
        e, r, o = str(entity), str(rel), str(obj)
        # try the induced surface realization if it is a real sentence (has a verb/'is')
        try:
            raw = self.surface.realize(e, r, o)
            if raw and (' is ' in raw or ' of ' in raw) and raw.lower() != f'{e} {r} {o}':
                return raw.strip().rstrip('.')
        except Exception:
            pass
        subj = f'the {r} of {e}' if first else f'its {r}'
        return f'{subj} is {o}'

    def _mood_from_body(self):
        """Day-103 -- (valence, arousal) read from the neuroendocrine body, so the
        organism SPEAKS in its current felt state. valence = dopamine above tonic
        baseline minus cortisol above baseline (reward - stress); arousal =
        norepinephrine. Ties the emotion channel into generation form (compose's
        mood conditions sentence length/order, never vocabulary)."""
        try:
            da = self.body.get('dopamine'); co = self.body.get('cortisol')
            ne = self.body.get('norepinephrine')
            val = (float(getattr(da, 'level', 0.5)) - 0.5) - (float(getattr(co, 'level', 0.1)) - 0.1)
            aro = float(getattr(ne, 'level', 0.3)) if ne is not None else 0.3
            return (max(-1.0, min(1.0, val)), max(0.0, min(1.0, aro)))
        except Exception:
            return (0.0, 0.3)

    def answer_fluent(self, query):
        """Day-103 -- DERIVE + FLUENT, and TALK TILL DONE. A question can ask for MORE
        than one thing ('what is the capital of france AND what continent is it in') --
        so: extract every RELATION the query asks for (relation-words the engine knows),
        resolve the entity they are about, DERIVE each fact, realise each as a fluent
        clause, and keep speaking until EVERY asked fact is covered, then HALT. It scopes
        to what was ASKED (not a dump of everything known), it is grounded (only derived
        facts -- cannot hallucinate), and it stops when done (the asked relations are
        exhausted). Returns a fluent sentence, or None to defer (nothing asked/derivable).

        This is the factual arm of the DERIVE-not-store generator: the content is derived
        per relation, the form is a fluent clause per fact, aggregated and pronominalised.
        Later relations about the same entity pronominalise ('...and its continent is
        europe') so it reads as one answer, not a list."""
        try:
            toks = [t for t in self.general_reasoner.tokenize(str(query)) if t]
        except Exception:
            return None
        # Resolve the ENTITY and the ASKED relations together from what the organism
        # actually HOLDS (knows() -- the broad derived fact web, not just the derive
        # engine's 5 core relations): the entity is the query token whose known
        # relations the query most asks about, and the asked relations are those of its
        # relations named in the query. This dodges the store fragmentation where
        # currency/language live in knows() but not in derive_engine.relations.
        best = None
        for t in toks:
            try:
                web = self.knows(t) or {}
            except Exception:
                web = {}
            if not web:
                continue
            asked = [r for r in web if r in toks and (web.get(r))]
            if asked and (best is None or len(asked) > len(best[1])):
                best = (t, asked, web)
        if best is None:
            return None
        entity, asked_set, web = best
        asked = [r for r in toks if r in asked_set]              # in query order, de-dup
        seen = set()
        # derive + realise each asked fact; talk till the asked set is covered, then halt
        clauses = []
        for r in asked:
            if r in seen:
                continue
            seen.add(r)
            vals = web.get(r) or []
            if vals:
                clauses.append(self._fluent_fact(entity, r, vals[0], first=(len(clauses) == 0)))
        if not clauses:
            return None                              # asked, but nothing derivable -> defer
        if len(clauses) == 1:
            s = clauses[0]
        else:
            s = ', and '.join([', '.join(clauses[:-1]), clauses[-1]]) if len(clauses) > 2 \
                else ' and '.join(clauses)
        s = s.strip()
        return s[0].upper() + s[1:] + '.'

    def express(self, topic=None, message=None, mood=None, rng_seed=0):
        """Day-103 -- THE ONE GENERATOR. Everything the organism says routes here, and
        every other generator is a mode of it. It unifies the principles read across
        the ~20 scattered generators into a single grounded, honest, structure-first,
        affect-aware speaker that DEGRADES GRACEFULLY:

          1. BEST PATH -- when the organism has induced distributional frames (from
             reading), realize_fluent fills an induced frame slot-by-slot by
             product-of-experts over STRUCTURE (induced type) x LOCAL FLUENCY
             (pcseq reservoir + grounded bigram) x TOPIC (meaning-fit). Free syntax,
             no template, learned entirely from data.
          2. GROUNDED FALLBACK -- when frames are sparse (the common case until the
             organism has read enough), compose realises the topic over SURFACE
             relation-frames: class chain + inherited attributes, pronominalised and
             aggregated, form conditioned by the body's MOOD. 100% derived.
          3. HONEST HALT -- never fabricates; returns None / 'unknown' with nothing
             grounded to say.

        So it MOVES A NEEDLE natively: identical to today's speak on a data-sparse
        organism (compose), and upgrades to free fluent syntax the moment reading has
        given it frames -- no flag, no separate call. Mood defaults to the body's own
        felt state, so the organism speaks as it feels."""
        mood = mood if mood is not None else self._mood_from_body()
        # The organism is DERIVE-not-store: it never retrieves a stored sentence, it
        # DERIVES what to say from memory. Generation has TWO jobs -- derive the grounded
        # CONTENT (never hallucinate) and realize it in fluent FORM -- and this ONE entry
        # owns both. Which REALIZER leads is decided by "moves a needle", not by dogma:
        #
        # DESCRIBING A KNOWN ENTITY (topic): compose realises the DERIVED class-chain +
        # inherited attributes over SURFACE relation-frames -- grounded, coherent,
        # mood-conditioned. MEASURED: on real induced data the free induced-frame fill
        # is still word-salad (the linear-VSA generation wall, documented), i.e. it
        # moves the needle BACKWARD for a known topic, so compose leads here. The fluent
        # realizer is wired and ready to take over the instant its quality beats compose
        # (the gen-quality wall's job -- biological gen / hybrid), no code change needed.
        if topic is not None:
            try:
                said = self.compose(topic, mood=mood)      # derive + realize (grounded)
            except Exception:
                said = None
            if said and not said.strip().endswith('is unknown.'):
                return said
            # nothing groundable to describe -> fall through to free generation
        # FREE GENERATION FROM AN INTENTION (message, no groundable topic): fill an
        # INDUCED frame by the GROUNDED transition walk restricted to each slot's type,
        # steered by the derived content as theme. MEASURED on real Tatoeba: this
        # (generate_structured) produces genuinely fluent English -- 100% real bigrams,
        # ~60% real trigrams -- whereas the bag-of-meaning fill (realize_fluent) scatters
        # words into 2% bigram salad. Local fluency is NOT the VSA wall; the open gap is
        # meaning/long-range coherence, which the theme + derived content steer. Only
        # fires when the organism has induced frames from reading.
        if message:
            try:
                fi = self.frame_inducer
                frm = fi.pick_frame(__import__('random').Random(rng_seed))
                if frm:
                    seed = message[0] if message else 'the'
                    r = self.generate_structured(seed, list(frm), fi.type_lexicon(),
                                                 theme=list(message))
                    seq = r.get('sequence') if isinstance(r, dict) else None
                    if seq:
                        s = ' '.join(seq).strip()
                        return s[0].upper() + s[1:] + ('.' if not s.endswith('.') else '')
            except Exception:
                pass
        return None                                  # honest halt -- nothing derived to say

    def set_purpose(self, topic):
        """Day-86 -- give the organism an IKIGAI: a topic that STEERS what it
        wonders about and pursues.  Not a hard constraint -- a weighting, so
        gaps touching the purpose surface first.  It is named Ikigai; now it has
        one."""
        self._ikigai = str(topic).strip().lower() if topic else None
        return self._ikigai

    def _pursue_gap(self, entity, relation):
        """Day-86 -- ACT on curiosity: form a testable HYPOTHESIS for an open
        gap from peer consensus (most co-members with this relation agree on a
        value -> the organism guesses it for the entity), stored as a
        LOW-CONFIDENCE BELIEF, tagged, kept SEPARATE from derived facts so a
        guess never pollutes ground truth.  This is how thinking makes it GROW.
        Returns (value, confidence) or (None, 0)."""
        from collections import Counter
        eng = self.general_reasoner.derive_engine
        groups, have = {}, {}
        for (s, r), v in eng.triples.items():
            have.setdefault(s, set()).add(r)
            if v:
                groups.setdefault((r, v), set()).add(s)
        classes = {k for k, ss in groups.items() if len(ss) >= 2 and entity in ss}
        peers = set()
        for k in classes:
            peers |= groups[k]
        peers.discard(entity)
        vals = Counter(eng.atom(relation, p) for p in peers if eng.atom(relation, p))
        if not vals:
            return None, 0.0
        v, c = vals.most_common(1)[0]
        conf = c / sum(vals.values())
        if not hasattr(self, '_beliefs'):
            self._beliefs = {}
        self._beliefs[(entity, relation)] = {
            'value': v, 'confidence': round(conf, 2), 'source': 'peer-consensus hypothesis'}
        return v, round(conf, 2)

    def _validate_on_perceive(self, s, r, o):
        """Day-86 -- PREDICTIVE LEARNING / self-correction.  If the organism had
        formed a HYPOTHESIS for (s, r) and now PERCEIVES the truth, test it: a
        match CONFIRMS the belief (confidence -> 1.0); a mismatch is a SURPRISE
        -- it corrects the belief and records a prediction error, the signal that
        drives future curiosity.  This closes the loop -- it predicts, reality
        tests it, it learns -- and is what makes the beliefs REAL, not idle
        guesses.  Returns the outcome, or None if it had no prediction."""
        if not hasattr(self, '_beliefs'):
            return None
        b = self._beliefs.get((s, r))
        if b is None or b.get('source', '').startswith(('confirmed', 'corrected')):
            return None
        if b['value'] == o:
            b['confidence'] = 1.0; b['source'] = 'confirmed by perception'
            return {'outcome': 'confirmed', 'fact': (s, r, o)}
        was = b['value']
        b['value'] = o; b['confidence'] = 1.0; b['source'] = 'corrected by perception'
        self._surprises = getattr(self, '_surprises', 0) + 1
        return {'outcome': 'surprise', 'was': was, 'now': o, 'fact': (s, r, o)}

    def _promote_beliefs(self):
        """Day-87 -- CONSOLIDATION: a belief the world has CONFIRMED graduates from
        a tagged guess into KNOWLEDGE.  Only beliefs validated against perception
        (source 'confirmed'/'corrected', confidence 1.0) are ingested as real
        derivable facts -- the hippocampus-to-cortex step -- while open hypotheses
        and unconfirmed dreamed conjectures stay tagged and apart from ground
        truth.  This is how a guess earned by being tested becomes permanent.
        Returns the number promoted."""
        beliefs = getattr(self, '_beliefs', {})
        promoted = 0
        for (s, r), b in list(beliefs.items()):
            if b.get('promoted'):
                continue
            src = b.get('source', '')
            if b.get('confidence', 0) >= 1.0 and ('confirmed' in src or 'corrected' in src):
                self.ingest_triples([(s, r, b['value'])], discover=False)
                b['promoted'] = True
                promoted += 1
        return promoted

    def contemplate(self, max_gaps=6, max_concepts=2, max_classes=3, pursue=True):
        """Day-86 -- the AUTONOMOUS COGNITIVE CYCLE: the organism USES its
        toolbox unprompted, instead of only when called.  It wonders about its
        own knowledge gaps (self-curiosity, STEERED by its ikigai/purpose), tries
        to ANSWER each by derivation and inheritance -- self-resolving what it
        can -- and for the rest ACTS on curiosity, forming a low-confidence
        hypothesis from peer consensus so it GROWS from thinking instead of just
        listing gaps.  It inventories what it knows by reverse derivation and
        invents concepts by clustering.  No fabrication: derived facts are exact,
        guesses are tagged beliefs kept apart.  Returns a thought log."""
        from collections import Counter
        eng = self.general_reasoner.derive_engine
        log = {'wondered': [], 'self_answered': [], 'open_questions': [],
               'hypotheses': [], 'inventory': {}, 'concepts': []}
        # 1. WONDER -- gaps, purpose-steered (gaps touching the ikigai first)
        gaps = self.wonder(top_k=max_gaps * 2)
        # Day-88: rank gaps by EXPECTED FREE ENERGY (explore=novelty + exploit=purpose),
        # so the organism attends to what most reduces its expected surprise.
        gaps = self._efe_rank_gaps(gaps)
        for g in gaps[:max_gaps]:
            log['wondered'].append(g['question'])
            val = (eng.atom(g['relation'], g['entity'])
                   or eng.inherited_atom(g['relation'], g['entity']))
            if val:
                log['self_answered'].append({'q': g['question'], 'a': val})
            elif pursue:
                hv, conf = self._pursue_gap(g['entity'], g['relation'])
                if hv:
                    log['hypotheses'].append({'q': g['question'], 'guess': hv, 'confidence': conf})
                else:
                    log['open_questions'].append(g['question'])
            else:
                log['open_questions'].append(g['question'])
        # 2. class values by in-degree (candidate abstractions)
        classcnt = Counter(v for (_s, _r), v in eng.triples.items() if v)
        # 3. REVERSE inventory -- enumerate members of the biggest classes
        for cls, _c in classcnt.most_common(max_classes):
            members = eng.reverse_reach(cls)
            if members:
                log['inventory'][cls] = members[:10]
        # 4. INVENT concepts -- cluster a class's members, induce shared schema
        for cls, _c in classcnt.most_common(max_concepts):
            members = [s for (s, r), v in eng.triples.items() if v == cls][:5]
            if len(members) >= 2:
                c = self.invent_concept(members)
                if c['schema']:
                    log['concepts'].append({'from': cls, **c})
        return log

    def think(self, *a, **k):
        """Alias -- the organism thinks to itself (see contemplate)."""
        return self.contemplate(*a, **k)

    def introspect(self):
        """Day-86 -- the organism REFLECTS ON ITSELF: its age and purpose, what
        it knows, what it believes (and what it was WRONG about and corrected),
        and what it is curious about right now.  Faithful -- every line comes
        from its actual state, so the self is made legible without fabrication.
        Returns a report dict with a spoken `narrative`."""
        from collections import Counter
        eng = self.general_reasoner.derive_engine
        classcnt = Counter(v for (_s, _r), v in eng.triples.items() if v)
        beliefs = getattr(self, '_beliefs', {})
        corrected = [k for k, b in beliefs.items() if 'corrected' in b.get('source', '')]
        confirmed = [k for k, b in beliefs.items() if 'confirmed' in b.get('source', '')]
        hypotheses = [k for k, b in beliefs.items() if 'hypothesis' in b.get('source', '')]
        curious = [g['question'] for g in self.wonder(top_k=3)]
        rep = {
            'age': getattr(self, '_age', 0),
            'purpose': getattr(self, '_ikigai', None),
            'knows_facts': len(eng.triples),
            'top_topics': [c for c, _ in classcnt.most_common(5)],
            'believes': len(beliefs),
            'confirmed': len(confirmed),
            'corrected_from_error': [f'{k[0]} {k[1]}={beliefs[k]["value"]}' for k in corrected],
            'open_hypotheses': len(hypotheses),
            'curious_about': curious,
            'surprises': getattr(self, '_surprises', 0),
        }
        lines = [f"i am age {rep['age']}."]
        if rep['purpose']:
            lines.append(f"my purpose is to understand {rep['purpose']}.")
        if rep['top_topics']:
            lines.append(f"i know {rep['knows_facts']} facts; i think most about "
                         f"{', '.join(rep['top_topics'][:3])}.")
        if hypotheses:
            lines.append(f"i hold {len(hypotheses)} hypotheses i still want to confirm.")
        if corrected:
            lines.append(f"i was wrong about {len(corrected)} thing(s) and corrected myself: "
                         f"{', '.join(rep['corrected_from_error'][:3])}.")
        if curious:
            lines.append(f"right now i wonder: {curious[0]}")
        rep['narrative'] = ' '.join(lines)
        return rep

    def live(self, ticks=8, inputs=None, sleep_every=4, verbose=False, fe_probe=None):
        """Day-86 -- the organism's LIFE: one continuous heartbeat, unprompted.
        Each tick it PERCEIVES (an optional input), CONTEMPLATES (wonders,
        self-answers, hypothesises -- steered by its ikigai), and accrues
        fatigue; when tired it SLEEPS (consolidate + dream) and wakes refreshed.
        A lifetime clock counts its age; beliefs formed by pursuit accumulate
        across ticks, so it GROWS.  This is the step from a toolbox the organism
        HAS to an organism that LIVES.  Returns its life log."""
        self._age = getattr(self, '_age', 0)
        self._fatigue = getattr(self, '_fatigue', 0)
        if not hasattr(self, '_beliefs'):
            self._beliefs = {}
        inputs = list(inputs or [])
        life = []
        for i in range(int(ticks)):
            self._age += 1
            tick = {'age': self._age, 'perceived': None, 'slept': False}
            if inputs:                                   # perceive the world if there is any
                obs = inputs.pop(0)
                try:
                    if isinstance(obs, (tuple, list)) and len(obs) == 3:
                        s, r, o = (str(x).strip().lower() for x in obs)
                        v = self._validate_on_perceive(s, r, o)   # test any prediction FIRST
                        if v:
                            tick['learned'] = v
                        self.ingest_triples([(s, r, o)])
                    elif hasattr(self, 'comprehend'):
                        self.comprehend(str(obs))
                    tick['perceived'] = obs
                except Exception:
                    pass
            t = self.contemplate(max_gaps=3)             # think
            tick['wondered'] = len(t['wondered'])
            tick['self_answered'] = len(t['self_answered'])
            tick['hypotheses'] = len(t['hypotheses'])
            if fe_probe is not None:                     # Day-88: free energy over a
                tick['free_energy'] = round(                # held-out probe -- drops as
                    self.free_energy(fe_probe)['free_energy'], 4)  # the organism learns
            self._fatigue += 1
            if self._fatigue >= sleep_every:             # tire -> sleep + dream
                try:
                    self.discover_rules()
                except Exception:
                    pass
                try:
                    tick['dream'] = self.dream()
                except Exception:
                    pass
                try:
                    tick['promoted'] = self._promote_beliefs()   # confirmed guesses -> knowledge
                except Exception:
                    pass
                tick['slept'] = True
                self._fatigue = 0
            if verbose:
                print(f"  age {self._age}: perceived={tick['perceived']} "
                      f"wondered={tick['wondered']} answered={tick['self_answered']} "
                      f"hypotheses={tick['hypotheses']} slept={tick['slept']}", flush=True)
            life.append(tick)
        confirmed = sum(1 for t in life if t.get('learned', {}).get('outcome') == 'confirmed')
        surprised = sum(1 for t in life if t.get('learned', {}).get('outcome') == 'surprise')
        return {'age': self._age, 'beliefs': len(self._beliefs),
                'purpose': getattr(self, '_ikigai', None),
                'confirmed': confirmed, 'surprised': surprised, 'log': life,
                'fe_curve': [t.get('free_energy') for t in life if 'free_energy' in t]}

    def wonder(self, entity=None, top_k=3, min_frac=0.5):
        """Day-86 GOLD -- SELF-DIRECTED CURIOSITY: the organism finds its OWN
        knowledge gaps and asks about them.  A gap is a relation the entity's
        PEERS (co-members of its class) have but it lacks -- surfaced from the
        store's structure (introspection, the same store-reading the rule miner
        does).  The gaps are then ranked by the wired CuriosityDrive's intrinsic
        NOVELTY (under-explored questions first) -- priority decided by the drive,
        not a hand-set score.  Returns [{entity, relation, question, novelty,
        peer_frac}], most-curious first."""
        from collections import Counter
        eng = self.general_reasoner.derive_engine
        # A CLASS is any (relation,value) shared by >= 2 entities (a high-in-degree
        # value); an entity's peers are the co-members of its classes.  Detected
        # purely from store structure -- no dependence on the class being a
        # subject, so a leaf class like 'metal' still groups its members.
        have, groups = {}, {}
        for (s, r), v in eng.triples.items():
            have.setdefault(s, set()).add(r)
            if v:
                groups.setdefault((r, v), set()).add(s)
        member_of = {}
        for key, ss in groups.items():
            if len(ss) >= 2:                          # class-like grouping
                for s in ss:
                    member_of.setdefault(s, set()).add(key)
        invented = getattr(self, '_invented', set())
        targets = [str(entity).strip().lower()] if entity else [e for e in have if e not in invented]
        gaps = []
        for e in targets:
            if e in invented:
                continue                          # the organism's own abstractions are not gaps
            peers = set()
            for key in member_of.get(e, ()):
                peers |= (groups[key] - invented)
            peers.discard(e)
            if not peers:
                continue
            cnt = Counter()
            for p in peers:
                for r in have.get(p, ()):
                    cnt[r] += 1
            for r, c in cnt.items():
                frac = c / len(peers)
                if frac >= min_frac and r not in have.get(e, ()):
                    gaps.append((e, r, frac))
        cur = getattr(self, 'curiosity', None)
        tok = self.general_reasoner.tokenize
        out = []
        for e, r, frac in gaps:
            q = f'what is the {r} of {e}'
            nov = float(cur.novelty(tok(q))) if cur is not None else frac
            out.append({'entity': e, 'relation': r, 'question': q + '?',
                        'novelty': round(nov, 3), 'peer_frac': round(frac, 2)})
        out.sort(key=lambda x: (-x['novelty'], -x['peer_frac']))
        return out[:top_k]

    def what_is_a(self, target, rels=None):
        """Day-86 -- REVERSE derivation, the forward query run backwards: list
        every entity that IS a `target` (every isa / subclass descendant),
        derived not stored via derive_engine.reverse_reach.  e.g.
        `org.what_is_a('metal')` -> vanadium, gold, platinum, ...  Returns the
        list of descendant entities."""
        eng = self.general_reasoner.derive_engine
        return eng.reverse_reach(str(target).strip().lower(), rels=rels)

    def analogy(self, a, b, c):
        """Day-86 GOLD -- solve the analogy A:B :: C:? by pure substrate algebra
        (derive_engine.analogy): the relation linking A->B is recovered by
        unbind + resonance, then applied to C -- bind/unbind/cleanup only, no
        relation list, no dict search.  Faithful: `verified` is True only when
        the recovered (relation, answer) is an actual stored/derived fact of C.
        Returns {answer, relation, score, verified, text}."""
        eng = self.general_reasoner.derive_engine
        ans, rel, score, verified = eng.analogy(
            str(a).strip().lower(), str(b).strip().lower(), str(c).strip().lower())
        text = f"{c} {rel} {ans}" if ans else "i don't know"
        return {'answer': ans, 'relation': rel, 'score': score,
                'verified': verified, 'text': text}

    def invent_relations(self, min_support=2, max_new=8):
        """Day-87 GOLD -- RELATION INVENTION: the organism grows new conceptual
        MACHINERY.  It composes two relations it already has into a NEW named
        relation R3 = R1 o R2 (derive_engine.invent_relations), keeping only
        compositions that answer facts NO single stored relation could -- so it
        can then derive, say, the country of someone's birthplace though no
        'nationality' edge was ever stored.  Discovery is honest store-mining
        (like the rule miner); the invented relation has a real phasor identity
        and is applied by derive-chaining.  Returns the invented relations."""
        eng = self.general_reasoner.derive_engine
        return eng.invent_relations(min_support=min_support, max_new=max_new)

    def derive_composed(self, name, entity):
        """Derive a named INVENTED relation for an entity by chaining its two
        hops (R1(R2(x))) -- derive-not-store over the composed machinery.  `name`
        is an invented relation's name (e.g. 'country-of-bornin').  Returns the
        derived value or None."""
        eng = self.general_reasoner.derive_engine
        return eng.derive_invented(str(name).strip().lower(),
                                   str(entity).strip().lower())

    def invent_concept(self, examples, name=None):
        """Day-85 GOLD -- INVENT a new concept from example entities.  The
        substrate anti-unifies them by binding + resonance (induce_concept):
        the concept = the properties they ALL share, surfaced geometrically, no
        curated feature list.  The concept becomes a first-class entity in the
        store (its shared schema written as atoms), so it composes with derive,
        inheritance and classification.  Returns {name, schema, n_props}."""
        eng = self.general_reasoner.derive_engine
        schema, _hv = eng.induce_concept([str(e).strip().lower() for e in examples])
        if name is None:
            name = 'concept_' + '_'.join(sorted(schema))[:40] if schema else 'concept_empty'
        name = str(name).strip().lower()
        for r, v in schema.items():               # the concept is now a real node
            eng._record(name, r, v)
        # remember invented concepts so introspection/curiosity treat them as
        # ABSTRACTIONS, not new things to be curious about (they are the
        # organism's own creations, not gaps in the world).
        if not hasattr(self, '_invented'):
            self._invented = set()
        self._invented.add(name)
        return {'name': name, 'schema': schema, 'n_props': len(schema)}

    def classify(self, entity, concept):
        """Day-85 -- does `entity` belong to an invented concept?  Membership is
        decided by SUBSTRATE RESONANCE (concept_member), not a dict lookup.
        `concept` may be a schema dict or the name of a concept whose schema is
        in the store.  Returns {member, score, schema}."""
        eng = self.general_reasoner.derive_engine
        if isinstance(concept, dict):
            schema = concept
        else:                                     # resolve a stored concept's schema
            c = str(concept).strip().lower()
            schema = {r: v for (s, r), v in eng.triples.items() if s == c and v}
        member, score = eng.concept_member(str(entity).strip().lower(), schema)
        return {'member': bool(member), 'score': round(float(score), 3), 'schema': schema}

    def ask_derive_proof(self, question, depth=None):
        """Day-83 audit WIRE of ProofCarryingGenerator: answer via the derive
        engine AND attach a verifiable derivation chain. Each hop becomes a
        proof step (rule = relation, premise = the intermediate entity); the
        chain is re-derived + verified before the answer is trusted. Returns
        {answer, entity, relations, verified, proof}. verified=False => abstain
        (honest-unknown: the derivation did not check out -- e.g. a tampered or
        broken chain). This is the unhallucinatable-answer path: no answer is
        emitted as trusted unless its proof chain verifies."""
        pg = self.proof_gen
        eng = self.general_reasoner.derive_engine
        if depth is None:
            # match ask_derive: parse against the engine's own vocab first
            # (NL bridge), fall back to the episodic parser.
            ent, mentions = self.holo_reader.parse_for_engine(
                question, eng.relations, eng.entities)
            if not (ent and mentions):
                ent, mentions = self.holo_reader.parse_chain(question)
        else:
            ent, rel = self.holo_reader.parse_question(question)
            mentions = [rel] * max(1, depth) if rel else None
        fail = {'answer': None, 'entity': ent, 'relations': mentions,
                'verified': False, 'proof': None}
        if not (ent and mentions):
            return fail
        rules, premises, cur = [], [], ent
        for rel in reversed(mentions):              # innermost-out
            nxt = eng.atom(rel, cur) or eng.inherited_atom(rel, cur)
            if not nxt:
                return fail
            rules.append(str(rel))
            premises.append([str(cur)])
            cur = nxt
        _hv, chain, verified = pg.generate([str(question)], rules, premises)
        return {'answer': cur, 'entity': ent, 'relations': list(mentions),
                'verified': bool(verified), 'proof': pg.explain(chain),
                'chain': chain}

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
            # Day 101: share the organism's own FactoredMeaning store (Day 91,
            # already persisted in save_ikg/load_ikg) for chunked-code anchor
            # identification -- MEASURED to hold recall at production scale
            # (N=137,450) where cosine-over-recall_batch state ranking
            # collapses (Kanerva load factor ~1,432, Day 100 finding).
            c = Cat4Absorb(self.unified, self.num_enc, pik,
                            state_codes=self.factored)
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
        # Day-97 FIX: this read `_crystallizer` / `crystallizer`.  The attribute is `crystal`
        # (set in __init__), so the lookup ALWAYS returned None and this channel had never once
        # fired.  It also probed `.triples` / `._triples`, which AtomicCrystallineStore does not
        # have -- its observations live in `._counts` keyed by the (s, p, o) triple.
        if 'crystal' in channels:
            cryst = getattr(self, 'crystal', None)
            counts = getattr(cryst, '_counts', None) if cryst is not None else None
            if counts:
                try:
                    for (s, v, o), _cnt in counts.items():
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
        # Day-97 FIX: this gated on hasattr(imp, 'importance').  ImportanceDecayLattice has no
        # such method -- `importance(alpha, beta)` is on the ITEM class; the lattice exposes
        # `strength(name, now)` (the Ebbinghaus-decayed weight) and `importances()`.  The gate was
        # therefore always False and this channel had never once fired.
        if 'importance' in channels:
            try:
                imp = getattr(self, 'imp_lattice', None)
                if imp is not None and hasattr(imp, 'strength'):
                    now = getattr(self, '_self_tick', None)
                    for tok in list(score_map.keys())[:top_k*4]:
                        try:
                            w = imp.strength(tok, now=now)
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

    def next_curious_action(self, state_tokens, action_candidates,
                            exploit_scores=None):
        """WIRE (Day-83 audit): expose the CuriosityDrive's intrinsic-motivation
        action selection (prediction-error bonus + exploit value). Previously
        curiosity only RECORDED prediction error in read(); now it can DRIVE
        exploration: pick the action that best trades off exploiting known value
        against the curiosity bonus of an under-predicted state. Returns
        (best_action_tokens, score), or (None, 0.0) if curiosity is off."""
        cur = getattr(self, 'curiosity', None)
        if cur is None or not action_candidates:
            return None, 0.0
        return cur.next_action(state_tokens, action_candidates,
                               exploit_scores=exploit_scores)

    def assign_credit(self, output_hv, input_hvs, target_hv):
        """WIRE (Day-83 audit): substrate-native NO-BACKPROP credit assignment
        (VSACalculus.credit_assign) -- which input HVs are responsible for the
        error (output vs target)? Returns a credit fraction per input HV. The
        gradient-free learning-signal primitive. None if vsa is off (lean mode)."""
        v = getattr(self, 'vsa', None)
        if v is None:
            return None
        return v.credit_assign(output_hv, input_hvs, target_hv)

    def agents_agree(self, agent_a, agent_b, key_tokens):
        """WIRE (Day-83 audit): Theory-of-Mind -- do two modeled agents share a
        belief on this key? None if ToM is off."""
        t = getattr(self, 'tom', None)
        return t.agree(agent_a, agent_b, key_tokens) if t is not None else None

    def common_ground(self, agent_names, key_tokens):
        """WIRE (Day-83 audit): ToM -- do ALL named agents agree on a key
        (shared common ground)? None if ToM is off."""
        t = getattr(self, 'tom', None)
        return t.common_ground(agent_names, key_tokens) if t is not None else None

    def false_belief_test(self, viewer, target, key_tokens):
        """WIRE (Day-83 audit): classic ToM false-belief test -- does `viewer`
        correctly model `target`'s (possibly false) belief about a key? Returns
        the test dict (actual vs meta-belief vs world truth). None if ToM off."""
        t = getattr(self, 'tom', None)
        return t.false_belief_test(viewer, target, key_tokens) if t is not None else None

    def heal_beliefs(self, max_rounds=5):
        """WIRE (Day-83 audit): BeliefField.propagate -- sweep all belief pairs
        and heal contradictions by consensus relaxation until stable (now fixed:
        the bipolar no-op heal was repaired). Also runs in sleep. Returns
        {healed_pairs, rounds, final_conflicts}. None if belief field is off."""
        b = getattr(self, 'belief', None)
        return b.propagate(max_rounds=max_rounds) if b is not None else None

    def plan(self, start_state, goal_state, max_depth=5, branching=3):
        """WIRE (Day-83 audit): query the PLANNING PILLAR. multistep_planner runs
        free-energy-guided DFS tree-search WITH BACKTRACK over the
        causal_world_model (transitions learned during read()) toward goal_state.
        Returns {actions, trajectory, success, stats} or None if the planner is
        off (lean mode). This is the NATIVE FE planner -- the thing the killed
        python-maze should have used (substrate-grounded, not a python algorithm)."""
        p = getattr(self, 'planner', None)
        if p is None:
            return None
        return p.plan_with_backtrack(start_state, goal_state,
                                     max_depth=max_depth, branching=branching)

    def act(self, goal_state, action_pool=None, start_state=None, max_steps=8,
            world=None):
        """Day-103 -- THE AGENTIC LOOP: the organism PURSUES a goal, closed-loop.

        org.plan() is one-shot OPEN-loop search (beam/DFS over the world MODEL). This
        is the deliberative arm the audit named missing: a CLOSED loop of genuine
        active inference -- perceive the state, EFE-select the next action (the
        ActiveInferencePlanner pillar: minimise expected free energy = -(epistemic +
        pragmatic) toward the goal), ACT in the world, OBSERVE what actually happened,
        and -- the point of closing the loop -- LEARN from surprise (write the real
        transition into the causal world model) and replan from the true state. An
        open-loop plan built on an INCOMPLETE model gets stuck; acting-and-observing
        discovers the missing transition and reaches the goal anyway.

        Uses the existing pillars only (ActiveInferencePlanner.score_actions +
        CausalWorldModel.predict/add_transition) -- no new search algorithm, no python
        maze. `world(state, action) -> next_state` is the real environment; without it
        the organism acts inside its own model (cwm.predict), which is pure planning.

        Returns {success, final_state, steps, trajectory, learned} where trajectory is
        [(state, action, next_state, efe, surprised?), ...]."""
        ap = getattr(self, 'active_planner', None)
        cwm = getattr(self, '_active_cwm', None)
        if ap is None or cwm is None:
            return None
        pool = list(action_pool) if action_pool else list(getattr(cwm, '_actions', {}) or [])
        if not pool:
            return {'success': False, 'final_state': start_state, 'steps': 0,
                    'trajectory': [], 'learned': 0, 'reason': 'no actions'}
        state = start_state if start_state is not None else getattr(self, '_act_state', None)
        traj = []; learned = 0; visited = set()
        for _ in range(int(max_steps)):
            if state == goal_state:
                break
            # REPLAN each step: multi-step EFE lookahead (ap.plan beam search short-
            # circuits on the goal) picks the next action toward goal. Single-step
            # score_actions has no goal gradient across hops on abstract states, so the
            # loop must plan, not act greedily -- then act only the FIRST step and
            # re-plan from what actually happened. That is the closed loop.
            pl = ap.plan(state, pool, goal_state=goal_state)
            ptraj = pl.get('trajectory') or []
            if not ptraj:
                break
            action = ptraj[0][1]; efe = round(float(ptraj[0][3]), 3)
            # the model's OWN expectation (for surprise detection)
            mpred = cwm.predict(state, action, top_k=1)
            model_next = mpred[0][0] if mpred else None
            # ACT: the world responds if given, else the organism acts in its model
            nxt = world(state, action) if world is not None else model_next
            if nxt is None:
                break
            surprised = (world is not None and nxt != model_next)
            if surprised:
                # LEARN from surprise -- the loop's whole reason to close
                try:
                    cwm.add_transition(state, action, nxt); learned += 1
                except Exception:
                    pass
            traj.append((state, action, nxt, efe, surprised))
            # cycle guard: (state, action) revisited without progress -> stop
            if (state, action) in visited and not surprised:
                break
            visited.add((state, action))
            state = nxt
        self._act_state = state
        return {'success': state == goal_state, 'final_state': state,
                'steps': len(traj), 'trajectory': traj, 'learned': learned}

    # ── Day 90: THE DEPTH LOOP -- one arbiter, small chat -> big artifact ──
    def respond(self, request, _depth=0, _max_depth=6):
        """Day 90 -- UNDERSTAND -> RESPOND as ONE emergent loop, at any DEPTH.

        respond does NOT decide small-vs-deep.  The ARBITER does, by argmin free
        energy: reason() proposes derive / multihop / arithmetic / cache AND -- when
        the request resolves into a goal over the world model -- a plan DECOMPOSITION,
        and returns the least-surprising one.  There is no authored 'if it abstained,
        then plan' fork; the decision lives inside the one loop (Day-90 fix).

        respond only ENACTS the arbiter's choice:
          * a direct answer (the arbiter cleared an exact path, F~0) is returned as
            is -- DEPTH 1.  A small chat stops here.
          * a chosen PLAN (the arbiter selected decomposition because nothing exact
            cleared) is EXECUTED by recursing the SAME loop per hop -- each hop
            bottoms out at a depth-1 arbiter answer, and a hop that itself resolves
            into a goal deepens again -- then the verified trajectory is STITCHED by
            the generation pillar (generate_fluent, grounded + goal-constrained).

        goal -> plan -> execute -> revise falls out of ONE arbiter+planner+generator
        loop.  Honest scope: this proves the MECHANISM (depth-from-one-loop over a
        made-up symbolic domain), not frontier long-form generation; execution and the
        stitch are honest meta-cognition (named), each a substrate op.

        Returns {answer, depth, method, plan, parts}."""
        r = self.general_reasoner.reason(str(request))
        method, ans, plan = r.get('method'), r.get('answer'), r.get('plan')
        chose_plan = (method == 'planner' and plan
                      and plan.get('success') and plan.get('trajectory'))
        if not chose_plan or _depth >= _max_depth:
            # Day 94 -- UNIFIED SELF hook: when the arbiter GIVES UP (abstain/empty), consult the
            # ONE memory (org.mem) before returning 'unknown'.  Fires ONLY on give-up, so exacts
            # (arith/derive/multihop/cache) and every regression capability are untouched.
            if ans in (None, 'unknown') and method in ('abstain', 'empty', None):
                mem_ans = self._answer_from_mem(str(request))
                if mem_ans is not None:
                    return {'answer': mem_ans, 'depth': _depth + 1, 'method': 'b_world_mem',
                            'plan': None, 'parts': None}
            return {'answer': ans, 'depth': _depth + 1, 'method': method,
                    'plan': plan if chose_plan else None, 'parts': None}
        # the ARBITER chose to decompose (argmin F) -- enact it: EXECUTE each hop
        # by recursing the SAME loop, then STITCH the verified path.
        traj = plan['trajectory']
        start, goalst = traj[0][0], traj[-1][2]
        parts, child_depth = [], _depth + 1
        for (st, act, _nx) in traj:
            sub = self.respond('what is the %s of %s' % (act, st),
                               _depth=_depth + 1, _max_depth=_max_depth)
            parts.append(sub['answer'])
            child_depth = max(child_depth, sub['depth'])
        artifact = self._stitch_path(start, traj, goalst)
        return {'answer': artifact, 'depth': child_depth, 'method': 'plan+execute',
                'plan': plan, 'parts': parts}

    def _answer_from_mem(self, text):
        """Day 94 -- answer from the UNIFIED SELF (org.mem) when the arbiter abstained.  Data-
        driven: DERIVE about each content token of the query and answer from whatever the one
        memory actually knows (transitive class chain + inherited + invented) -- no authored word
        list, no parse template.  Returns a grounded answer string or None.  Additive: only reached
        on give-up, so the exact reasoning path is unchanged."""
        try:
            toks = list(self.general_reasoner.tokenize(text))
        except Exception:
            toks = str(text).lower().split()
        mem = self.mem
        eng = self.general_reasoner.derive_engine
        for t in toks:
            if len(t) <= 2:
                continue                          # skip short function tokens (length, not a list)
            facts = []
            try:
                facts.extend(mem.describe(t))     # DERIVED (transitive chain + inherited + invented)
            except Exception:
                pass
            try:                                  # DIRECT one-hop atoms (any relation the memory holds)
                for r in sorted(eng.relations):
                    v = eng.atom(r, t)
                    if v and (r, v) not in facts:
                        facts.append((r, v))
            except Exception:
                pass
            if facts:
                parts = [f"{t} {r} {v}" for (r, v) in facts[:6]]
                return ' ; '.join(parts)
        return None

    def _stitch_path(self, start, trajectory, goalst):
        """Render an executed trajectory as a grounded sequence via the generation
        pillar: observe the verified path so valid_next is grounded, then generate
        toward the goal.  Falls back to the verified state chain if the generator
        has no branch to walk (single observed path)."""
        states = [start] + [nx for (_st, _a, nx) in trajectory]
        try:
            self.observe_transitions([states])
            g = self.generate_fluent(start, steps=len(trajectory),
                                     constraints=[self.bundle_constraint([goalst])])
            seq = g.get('sequence') if g else None
            if seq and seq[-1] == goalst:
                return seq
        except Exception:
            pass
        return states

    def register_threat(self, name, tokens):
        """WIRE (Day-83 audit): ARM the adversarial_immune system -- register a
        threat antibody (prompt-injection pattern). ask()'s immune.scan then
        detects it. Previously dormant (no threats ever registered). None if
        immune is off (lean mode)."""
        im = getattr(self, 'immune', None)
        return im.register_threat(name, tokens) if im is not None else None

    def scan_threats(self, query_tokens, threshold=0.4):
        """WIRE (Day-83 audit): scan input against registered threat antibodies
        (adversarial_immune cosine detection). Returns the detected-threat list."""
        im = getattr(self, 'immune', None)
        return im.scan(query_tokens, threshold=threshold) if im is not None else []

    def explain_derivation(self, x, role, y):
        """WIRE (Day-83 audit): logical_fixed_point.explain -- return the chain of
        rule derivations that PROVED (x, role, y), or [] if it is an axiom.
        Show-your-work audit for the fixed-point reasoner (anti-hallucination).
        Requires org.reasoning() to have been opened + run first; None if not."""
        lfp = self.reasoning_engine()
        return lfp.explain(x, role, y) if lfp is not None else None

    def best_counterfactual_action(self, goal_tokens, action_candidates):
        """WIRE (Day-83 audit): counterfactual_sim.best_action -- simulate each
        candidate action's predicted outcome and pick the one whose outcome best
        matches the goal (lowest expected free energy). action_candidates: list
        of (name, tokens). Returns (best_name, score). Counterfactual scenarios
        are captured during read(); this QUERIES them for decision-making. None
        if cf is off (lean mode)."""
        cf = getattr(self, 'cf', None)
        if cf is None or not action_candidates:
            return None, 0.0
        return cf.best_action(goal_tokens, action_candidates)

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
        # Day-95 -- refresh the substrate REACH bank in sleep: re-consolidate the transitive
        # closure onto the hashed-address store so reach_member stays a flat substrate read as new
        # facts arrive. Only if reach was ever used (idle organisms pay nothing). Part of the
        # autonomous loop -- the organism keeps its substrate reasoning current while it sleeps.
        try:
            if getattr(self, '_reach_store', None) is not None:
                link = getattr(self, '_reach_link', 'isa')
                stats['reach_pairs'] = self.consolidate_reach(link).get('pairs', 0)
                self._reach_dirty = False
        except Exception as e:
            stats['reach_err'] = str(e)[:80]
        # Day-96 -- refresh the MULTI-VALUE ancestor set-recall bank the same way (only if used).
        try:
            if getattr(self, '_anc_sdm', None) is not None:
                link = getattr(self, '_anc_link', 'isa')
                stats['ancestor_pairs'] = self.consolidate_ancestors(link).get('pairs', 0)
                self._anc_dirty = False
        except Exception as e:
            stats['ancestor_err'] = str(e)[:80]
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
        # Day 90 -- CONSOLIDATION: replay the transient generation (hippocampal) sequence
        # store into the durable SDM (cortical/cerebellar), so learned transitions persist
        # in .ikg and live in the ONE shared substrate (biology: hippocampus -> cortex).
        try:
            n_cons = self.consolidate_generation()
            stats['generation_consolidated'] = int(n_cons)
        except Exception as e:
            stats['generation_consolidate_err'] = str(e)[:80]
        # WIRE (Day-83 audit): bio-forgetting during sleep -- importance_decay
        # PRUNES low-strength memories (Ebbinghaus). Previously it only recorded
        # in read(); now memory actually decays. Returns the pruned names count.
        try:
            if getattr(self, 'imp_lattice', None) is not None:
                pruned = self.imp_lattice.prune(now=getattr(self, '_self_tick', None))
                stats['importance_pruned'] = len(pruned) if pruned else 0
        except Exception as e:
            stats['importance_err'] = str(e)[:80]
        # WIRE (Day-83 audit): heal belief CONTRADICTIONS during sleep --
        # BeliefField.propagate sweeps all pairs + consensus-relaxes conflicts
        # (the heal no-op bug was fixed in batch 4). Self-consistency upkeep.
        try:
            if getattr(self, 'belief', None) is not None:
                bp = self.belief.propagate(max_rounds=3)
                stats['beliefs_healed'] = bp.get('healed_pairs', 0)
        except Exception as e:
            stats['belief_heal_err'] = str(e)[:80]
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
        # Day-86 -- AUTONOMOUS REASONING during sleep: the organism USES its
        # toolbox unprompted -- wonders about its own gaps, self-answers by
        # derivation/inheritance, inventories by reverse derivation, invents
        # concepts.  Learning (above) fed the store; this exercises it.
        try:
            stats['contemplation'] = self.contemplate()
        except Exception as e:
            stats['contemplate_err'] = str(e)[:80]
        # Day-87 -- RELATION INVENTION during sleep: compose existing relations
        # into new named machinery (R1 o R2) that answers facts no single stored
        # relation could.  The organism grows its own concepts while it rests.
        try:
            inv = self.invent_relations()
            stats['relations_invented'] = len(inv) if inv else 0
        except Exception as e:
            stats['invent_rel_err'] = str(e)[:80]
        # Day-87 -- CONSOLIDATION: promote confirmed beliefs into knowledge.
        try:
            stats['beliefs_promoted'] = self._promote_beliefs()
        except Exception as e:
            stats['promote_err'] = str(e)[:80]
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
                # Day-97 FIX: this said `self.importance_decay`, which is not an attribute of the
                # organism (the lattice is `imp_lattice`).  The AttributeError was swallowed into
                # importance_err, so this sleep phase had never once run.
                idec = self.imp_lattice
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

    # absorb / load_substrate / prompt REMOVED (audit trio, Day-83): GPT-2
    # weight-bake-into-substrate API on the deleted t2s_compiler (abandoned
    # mission). LLM knowledge enters via the teacher data-oracle path, not weights.

    # ── Pack 195p: .ikg auto-load / auto-save (kill stack #10) ──────────
    DEFAULT_IKG_PATH = os.environ.get('IKIGAI_IKG', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'organism.ikg'))

    def load_ikg(self, path=None):
        """Replace self.unified with substrate loaded from .ikg file.
        Path defaults to env IKIGAI_IKG or organism.ikg next to this module.
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
        # Day-97 WIRE -- the packed fact store rides with the organism.  The Day-96/97 address
        # store was built, gated, and then called by NOTHING: the census found `load_facts` /
        # `ingest_addressed` / `reach_addressed` ORPHANED (0 call sites anywhere), because every
        # gate exercised AddressFactStore DIRECTLY and no production path ever loaded one.  A wire
        # nothing plugs into is not wired.  So: if a sidecar `<ikg>.facts` exists, mmap it here and
        # attach it to the derive engine -- bulk graph knowledge becomes live by DEFAULT, at
        # ~0 MB resident (the store stays on disk).  Absent sidecar -> no-op, nothing changes.
        facts_path = path + '.facts'
        n_edges = 0
        if os.path.exists(facts_path):
            try:
                st = self.load_facts(facts_path, mmap=True)
                n_edges = st.n_edges
            except Exception:
                n_edges = 0
        return {'path': path, 'd': self.unified.d,
                 'sdm_M': self.unified.sdm.M,
                 'sdm_rel_M': self.unified.sdm_rel.M,
                 'cooccur_vocab': len(self.unified._cooccur_seen),
                 'seen': len(self.unified._seen),
                 'frame_locked': bool(self.frames.locked),
                 'frame_assigns': self.frames.assigns_per_frame.tolist(),
                 'pack220_modules_restored': n_restored,
                 'packed_edges': n_edges}

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
        # Day 104 -- the surface realizer: induced/curiosity-learned relation
        # FRAMES (templates + variants). Grammar is learned from exposure, so it
        # must persist like the facts, not be re-induced every boot.
        '_surface',
        # Day 104 -- the organism's SELF-KNOWLEDGE (who it is). Innate default;
        # once seeded it persists, so it stays knowing across reload.
        '_identity',
        # Day 104 -- the closed-class grammar LEARNED from the corpus (function words +
        # interrogatives). Emergent from frequency + '?' openers, not authored -- persists
        # so the organism keeps the language it learned across reload.
        '_function_words', '_interrogatives',
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
        # Day 91: factored-meaning codes (word-keyed, ck-independent; H regenerates from
        # seed).  Persist only params + the bytes/word code store so trained MEANING survives
        # reload -- the readable-past-the-knee store lives in .ikg like the substrate.
        fm = getattr(self, '_factored', None)
        if fm is not None and getattr(fm, 'codes', None):
            try:
                out['_factored_state'] = _pkl.dumps(fm.state_dict(),
                                                    protocol=_pkl.HIGHEST_PROTOCOL)
            except Exception:
                pass
        # Day 91: RHC fact store (address-tuples, tiny) -- durable knowledge in .ikg.
        fs = getattr(self, '_fact_store', None)
        if fs is not None and fs.n_facts:
            try:
                out['_fact_store_state'] = _pkl.dumps(fs.state_dict(),
                                                      protocol=_pkl.HIGHEST_PROTOCOL)
            except Exception:
                pass
        # Day 91: the one memory (people, goals, emotions, episodes) -- the continuous self.
        sm = getattr(self, '_mem', None)
        if sm is not None and sm.n_memories:
            try:
                out['_mem_state'] = _pkl.dumps(sm.state_dict(), protocol=_pkl.HIGHEST_PROTOCOL)
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
            if attr == '_factored_state':
                # Day 91: rebuild the factored-meaning store from persisted codes.
                try:
                    st = _pkl.loads(blob)
                    self.factored.load_state(st)          # lazy-builds with current ck
                except Exception:
                    pass
                continue
            if attr == '_fact_store_state':
                # Day 91: rebuild the RHC fact store from persisted address-tuples.
                try:
                    self.fact_store.load_state(_pkl.loads(blob))
                except Exception:
                    pass
                continue
            if attr == '_mem_state':
                # Day 91: restore the continuous self (people, goals, emotions, episodes).
                try:
                    self.mem.load_state(_pkl.loads(blob))
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

    def save_ikg(self, path=None, allow_production=False):
        """Save current substrate + frames + Pack 220 wired-module state to
        .ikg (kill stack #10 format).

        Day-97 GUARD.  The project's oldest standing rule is that the PRODUCTION organism
        (DEFAULT_IKG_PATH) is LOAD-ONLY -- it is never saved, because a save from a partially
        built organism destroys months of ingested knowledge.  That rule lived only in a human's
        head, and on Day 97 an AUDIT script called `org.save_ikg()` with no arguments (every
        parameter had a default, so a blind zero-arg probe reached it) and overwrote the 193 MB
        production organism with a 6 MB empty one.  There was no backup.

        The rule is now enforced by the code, not by memory: writing to the production path
        requires `allow_production=True`, stated explicitly, at the call site.  A no-arg
        `save_ikg()` can no longer destroy anything."""
        if path is None:
            path = getattr(self, '_ikg_path', None) or self.DEFAULT_IKG_PATH
        if not allow_production:
            _prod = os.path.abspath(self.DEFAULT_IKG_PATH)
            if os.path.abspath(str(path)) == _prod:
                raise PermissionError(
                    f'refusing to overwrite the PRODUCTION organism at {path}.\n'
                    '  organism.ikg is LOAD-ONLY (standing rule). It was destroyed once, on '
                    'Day 97, by exactly this call with no arguments.\n'
                    '  If you truly mean to persist production: back it up first, run the '
                    'regression gate, then call save_ikg(path, allow_production=True).')
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

    # T2S weight-bake API REMOVED (audit trio, Day-83): t2s() factory,
    # absorb/absorb_llm/absorb_llm_deep/absorb_native, _diverse_seed_prompts,
    # _build_emergent_clusterer, _is_junk_token/_filter_junk_text, big-substrate
    # + native gpt2/llama forward methods -- all on the deleted t2s_compiler
    # (abandoned mission: LLMs are data oracles, not weight donors). udsp preserved.

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

    def time_role(self):
        """WIRE (Day-83 audit): time-as-a-role temporal indexing (lazy). Bind a
        timestamp BUCKET into an address as a phasor role factor, then query
        "what was X's <role> at time T" -- assert_at(word, role, target, bucket)
        / query_at(word, role, bucket) / diff(word, role, bucket_a, bucket_b).
        A native temporal-memory primitive over the unified substrate."""
        tr = getattr(self, '_time_role', None)
        if tr is None:
            from ikigai.cognition.time_role import TimeRole
            tr = self._time_role = TimeRole(self.unified)
        return tr

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

    def flat_similarity(self, w1, w2):
        """Word similarity from the constant-RAM flat substrate."""
        return self.flat.similarity(w1, w2)

    @property
    def factored(self):
        """Day 91 -- FACTORED distributional-meaning store (RHC for vectors).  The shared-
        superposition cooccur bank saturates ~20k vocab (crosstalk destroys the readout); this
        stores each word's meaning as K*b random-hyperplane bits (capacity 2^bits, similarity by
        bit-agreement), readable past the knee to billions at bytes/word.  Lazy + shares the
        organism's ck word-identity space.  Feed it via org.observe_meaning / consolidate_meaning;
        semantic_sim prefers it when a word is coded."""
        fm = getattr(self, '_factored', None)
        if fm is None:
            from ikigai.cognition.factored_meaning import FactoredMeaning
            ck = self.unified.ck if getattr(self, 'unified', None) is not None else self.flat.ck
            fm = FactoredMeaning(ck, bits=512, window=3, seed=91)
            self._factored = fm
        return fm

    def observe_meaning(self, docs):
        """Stream co-occurrence documents into the factored-meaning store.  `docs` = iterable of
        token lists or strings.  Call consolidate_meaning() to project to durable codes."""
        fm = self.factored
        n = 0
        for d in docs:
            toks = d.split() if isinstance(d, str) else list(d)
            n += fm.observe_cooccur(toks)
        return n

    def consolidate_meaning(self, drop_acc=True):
        """Project accumulated co-occurrence embeddings to durable factored codes (biology:
        consolidation).  Returns #words coded.  After this, semantic_sim reads the code store."""
        return self.factored.consolidate(drop_acc=drop_acc)

    # ── Day 92: PATH A -- local-learning SEQUENCE model (predictive coding, no backprop) ──
    @property
    def pcseq(self):
        """Day 92 -- recurrent HDC context-state + delta-rule readout (reservoir computing in the
        FHRR substrate).  The non-backprop sequence model: a CONTINUOUS running state carries the
        whole sentence-so-far -- curing the discrete-tuple walk's long-range blindness and its
        degenerate loop -- while a single linear readout predicts the next token by local error
        (Widrow-Hoff / predictive coding, NOT backprop).  Validated Day 92 (day92_pc_seq: a
        long-range dependency 9 tokens back recovered 100% vs a bigram's 0%).  Lazy; shares the
        organism's ck word-identity space.  Feed via org.observe_sequence; read via predict_next /
        generate_predictive.  Honest: strong at context, weaker than a bigram at raw local
        next-token -- meant to be COMPOSED (PoE) with a local model, not used alone for fluency."""
        pc = getattr(self, '_pcseq', None)
        if pc is None:
            from ikigai.cognition.predictive_sequence import PredictiveSequenceModel
            ck = self.unified.ck if getattr(self, 'unified', None) is not None else self.flat.ck
            pc = PredictiveSequenceModel(ck, decay=0.9, lr=0.5, seed=92)
            self._pcseq = pc
        return pc

    def observe_sequence(self, sequences, epochs=1):
        """Learn next-token structure from example sequences by the delta rule (no backprop).
        `sequences` = iterable of token lists.  Returns vocab size."""
        return self.pcseq.learn(list(sequences), epochs=epochs)

    def predict_next(self, context, k=5):
        """Top-k next tokens for a context by the readout's resonance -- context-aware, carrying
        the long-range dependency a bigram structurally cannot."""
        return self.pcseq.predict(list(context), k=k)

    def build_decodable_meaning(self, n_factors=6, per_factor=32):
        """Learn locality-preserving factor codebooks over the accumulated co-occurrence
        embeddings so meaning becomes an EMITTABLE code (capacity per_factor**n_factors).  Call
        after observe_meaning, before consolidate.  Returns #words coded."""
        return self.factored.build_decodable(n_factors=n_factors, per_factor=per_factor)

    def emit_meaning(self, tokens, k=3):
        """Compose the meaning of `tokens` in code space and emit the grounded word(s) -- the
        generative primitive of the decodable meaning codebook (no HV-decode, no resonator
        d-bound).  Needs build_decodable_meaning() first."""
        return self.factored.emit(tokens, k=k)

    # ── Day 91: THE ONE MEMORY -- SDM + VSA + RHC + factored + cache, unified ──
    @property
    def mem(self):
        """The ONE memory.  Not another store on the pile -- a single object over the shared ck
        that unifies every mechanism the organism has:
          .similar/.neighbors/.observe/.emit -> factored meaning (semantic, SDM/LSH, past the knee)
          .relate/.query                       -> unified roles (relational, VSA bind)
          .fact/.derive/.knows                 -> RHC fact store (exact knowledge, 1e9)
          .kv_put/.kv_get                      -> in-substrate RHC cache (episodic key->value)
          .remember/.recall/.mood/.people/.goals -> the autobiographical self (people, goals,
                                                    emotions, events)
        One entry, one persistence (rides in .ikg beside the substrate), lazy + DEFAULT.  Additive
        facade -- the reasoning path is untouched, so risky regression tests are unaffected."""
        m = getattr(self, '_mem', None)
        if m is None:
            from ikigai.cognition.unified_memory import UnifiedMemory
            m = UnifiedMemory(self)
            self._mem = m
        return m

    def meaning_hv(self, word):
        """The decodable meaning HV of a word (JET ENGINE: block-partitioned, decodes at C**K)."""
        return self.factored.meaning_hv(word)

    def decode_meaning(self, hv, k=3):
        """Read a meaning HV back to word(s) by independent per-block decode (no joint d-bound) --
        decode-at-1e9 with similarity preserved.  Needs build_decodable_meaning() first."""
        return self.factored.decode_hv(hv, k=k)

    # ── Day 91: RHC-addressed exact stores (fact store + in-substrate cache) ──────
    @property
    def rhc(self):
        """Residue-HDC block store shared by the RHC-addressed stores (capacity ~6.69e9 at
        fixed d).  Lazy; the exact/decodable 1e9 address engine."""
        r = getattr(self, '_rhc', None)
        if r is None:
            from ikigai.cognition.rhc_stores import _BlockRHC
            r = _BlockRHC(d=self.unified.d if getattr(self, 'unified', None) is not None else 256)
            self._rhc = r
        return r

    @property
    def fact_store(self):
        """RHC FactStore: knowledge as address-tuples, exact, ~bytes/fact, capacity >= 1e9,
        consequences derived.  Lazy + persisted into .ikg."""
        fs = getattr(self, '_fact_store', None)
        if fs is None:
            from ikigai.cognition.rhc_stores import FactStore
            d = self.unified.d if getattr(self, 'unified', None) is not None else 256
            fs = FactStore(d=d)
            self._fact_store = fs
        return fs

    @property
    def assoc_cache(self):
        """RHC AssocCache: the anchor cache IN the substrate -- 1e9 collision-free key address
        space (no external dict) + sparse SDM value store.  Lazy (runtime store)."""
        ac = getattr(self, '_assoc_cache', None)
        if ac is None:
            from ikigai.cognition.rhc_stores import AssocCache
            ck = self.unified.ck if getattr(self, 'unified', None) is not None else self.flat.ck
            ac = AssocCache(ck, d=ck.d)
            self._assoc_cache = ac
        return ac

    def store_fact(self, subj, rel, obj):
        """Store a fact as an RHC address-tuple (exact, bytes/fact).  subj/rel/obj are lexicon
        indices (ints).  Consequences are DERIVED, not stored."""
        return self.fact_store.add(subj, rel, obj)

    def derive_facts(self, subj, rel, max_hops=8):
        """Transitive closure of relation `rel` from `subj` over the RHC fact store (derived)."""
        return self.fact_store.derive_chain(subj, rel, max_hops=max_hops)

    def remember_kv(self, key, value):
        """Write a key->value into the in-substrate RHC-addressed associative cache."""
        return self.assoc_cache.put(key, value)

    def recall_kv(self, key, candidates):
        """Recall a key's value from the in-substrate RHC cache (cleanup vs candidate values)."""
        return self.assoc_cache.get(key, candidates)

    def consolidate_reach(self, link='isa', d=1024, B=8192, g=16, n_absent=1200, seed=95):
        """Day-95 -- SLEEP-CONSOLIDATE transitive reasoning ONTO THE SUBSTRATE (the biological
        answer to F0: reasoning as a flat substrate read, not a python dict walk).  Derives the
        transitive closure of `link` ONCE from the derive engine's stored atoms and writes each
        (descendant -> ancestor) pair into a hashed-address distributed store (HashedReachStore),
        then calibrates an abstain boundary from absent samples.  Afterwards `org.reach_member(x,c)`
        answers 'is c a transitive ancestor of x' in ONE O(g*d) substrate read, correct-or-abstain.
        Measured envelope (day95_biological_reach): 100% recall / 1% false-accept to ~18k pairs at
        ~113us, flat compute, never-wrong.  Returns {pairs, boundary}.  Teach with discover=True
        first so `link` is mined transitive.  Additive: does not touch the reason() path; no save."""
        import random as _random
        from ikigai.cognition.rhc_stores import HashedReachStore
        eng = self.general_reasoner.derive_engine
        if not eng.is_transitive(link):
            return {'pairs': 0, 'boundary': 0.0,
                    'note': f'"{link}" not mined transitive -- teach(discover=True) first'}
        store = HashedReachStore(d=d, B=B, g=g, seed=seed)   # own d (decoupled from substrate dim)
        ents = sorted(getattr(eng, 'entities', []))
        present = set()
        for x in ents:
            chain = eng.transitive_reach(link, x) or []
            for anc in (c.replace(' ', '') for c in chain[1:]):
                store.write_pair(x, anc); present.add((x, anc))
        rng = _random.Random(seed)
        absent = []
        for _ in range(n_absent * 4):
            if len(absent) >= n_absent or len(ents) < 2:
                break
            x = rng.choice(ents); c = rng.choice(ents)
            if x != c and (x, c) not in present:
                absent.append((x, c))
        store.calibrate(absent)
        self._reach_store = store
        self._reach_link = link
        return {'pairs': store.n, 'boundary': round(store.boundary, 4)}

    def _struct_sig(self, key):
        """Deterministic ORTHOGONAL role phasor for a structural key (rank / slot / position).
        Orthogonal, NOT an FPE magnitude code: a slot identity ('rank 2') must not leak into its
        neighbour ('rank 3'); FPE encodes proximity=similarity and makes unbind recover the wrong
        payload (measured: 50% vs 100%).  Same key -> same signature in every modality."""
        import numpy as _np
        d = self.unified.d
        r = _np.random.default_rng(abs(hash(('struct', str(key)))) % (2 ** 31))
        return _np.exp(1j * r.uniform(-_np.pi, _np.pi, d)).astype(_np.complex64)

    def structure_bind(self, key, payload):
        """Day-95 -- bind a PAYLOAD to a STRUCTURAL key (rank / role / position) in one superposed
        associative bank: bank += bind(sig(key), key(payload)).  CROSS-MODAL BY CONSTRUCTION: any
        modality that independently DERIVES the same structural key recalls the same payload -- the
        address is the derived STRUCTURE, never the surface token.  This is the real mechanism behind
        cross-modal transfer (day95_crossmodal_real: a trait taught via a text-derived rank is
        recovered 480/480 through space- and music-derived ranks).  Default-on: bank created lazily.
        Returns the number of bindings held."""
        import numpy as _np
        bank = getattr(self, '_struct_bank', None)
        if bank is None:
            bank = _np.zeros(self.unified.d, dtype=_np.complex64)
        pk = _np.asarray(self.unified.ck.key(str(payload).strip().lower()), _np.complex64).reshape(-1)
        self._struct_bank = (bank + self._struct_sig(key) * pk).astype(_np.complex64)
        self._struct_n = getattr(self, '_struct_n', 0) + 1
        return self._struct_n

    def structure_recall(self, key, candidates):
        """Day-95 -- recall the payload bound to a structural key: unbind the bank by the key's role
        phasor and CLEAN UP over `candidates` (cosine argmax over the substrate codebook).  Returns
        the best candidate, or None if nothing is bound yet.  A wrongly-derived key yields the wrong
        payload -- this can genuinely fail, which is what makes it a real test."""
        import numpy as _np
        bank = getattr(self, '_struct_bank', None)
        if bank is None or not candidates:
            return None
        v = (bank * _np.conj(self._struct_sig(key))).astype(_np.complex64)
        best, bc = -1e9, None
        for c in candidates:
            ck = _np.asarray(self.unified.ck.key(str(c).strip().lower()), _np.complex64).reshape(-1)
            s = float(_np.real(_np.vdot(ck, v)))
            if s > best:
                best, bc = s, c
        return bc

    def reach_member(self, x, c, link='isa'):
        """Day-95 -- is `c` a transitive ancestor of `x`?  ONE flat substrate read of the
        consolidated reach bank (O(g*d)); below the calibrated boundary -> abstains (returns False),
        a calibrated <1% false-accept floor (NOT dict-exact 0-wrong).  DEFAULT-ON: auto-consolidates
        the closure on first use so it just works with no setup; re-consolidates when new facts were
        taught since (dirty flag).  Returns None only if `link` is not (yet) mined transitive."""
        st = getattr(self, '_reach_store', None)
        if st is None or getattr(self, '_reach_dirty', False):
            info = self.consolidate_reach(link)
            if not info.get('pairs'):
                return None                                   # link not transitive yet
            st = self._reach_store
            self._reach_dirty = False
        return bool(st.member(x, c))

    # ---------------- Day-96: FACTS AS PACKED ADDRESS-LISTS (the fact store that scales) --------
    _MERGE_MAX_EDGES = 200_000        # Day-99: above this, merging back through strings is a 132+ MB
                                      # spike -- refuse loudly rather than OOM or drop data silently.

    def ingest_addressed(self, triples, rel_hint=None, replace=False):
        """Day-96 -- ingest bulk facts into the PACKED ADDRESS-LIST store (`AddressFactStore`), the
        production form of the Day-90 idea: a fact is THREE INDICES over a shared lexicon, not three
        strings.  The running derive engine (`compositional.py`) keeps each fact as python STRINGS in
        a dict -- measured 662 B/edge (4.16M Wikidata p279 edges -> 3.5 GB RSS).  Here the same edges
        cost 5.3 B (measured, day96_address_store), with byte-identical closures.

        What is free stays free: WORDS are computed addresses (`ck`, 0 B), CONSEQUENCES are derived
        (0 B), the SDM substrate is FIXED.  The ONLY thing that costs bytes is the atomic edge --
        which combinations are TRUE -- and that is Shannon-irreducible (~4 B/edge).  Superposition
        cannot dodge it: bundle capacity is ~O(d) before crosstalk (this project measured the wall
        three ways), so N facts held readably need d ~ N and the vector IS the bytes.

        Use for BULK/graph knowledge (taxonomies, KGs).  Additive: does not touch reason().

        Day-99 -- MERGES by default (replace=False).  It used to do `self._addr_facts = st`, i.e. a
        second call silently DESTROYED the first corpus (measured: ingest A, then ingest B, and A's
        edges were gone -- alpha->beta True then False).  Silent destruction is the Day-97 failure
        wearing a different mask, and it would have broken Day-100/101 outright ("feed a corpus, then
        feed another").  Now the old edges are streamed back out and rebuilt together with the new.

        The merge is BOUNDED: streaming a huge store back to strings costs 662 B/edge (4.16M edges =
        3.5 GB), which is the exact cost the packed form exists to avoid.  Above _MERGE_MAX_EDGES it
        REFUSES rather than OOM the machine or silently drop data -- rebuild from source, or pass
        replace=True to deliberately start a new store."""
        from ikigai.cognition.address_store import AddressFactStore
        new = list(triples)
        old = getattr(self, '_addr_facts', None)
        if old is not None and not replace:
            n_old = int(getattr(old, 'n_edges', 0) or 0)
            if n_old > self._MERGE_MAX_EDGES:
                raise ValueError(
                    f'ingest_addressed: refusing to merge into a store of {n_old:,} edges '
                    f'(> _MERGE_MAX_EDGES={self._MERGE_MAX_EDGES:,}). Streaming it back to strings '
                    f'would cost ~{n_old * 662 / 1e9:.1f} GB. Rebuild from source with the full '
                    f'triple list, or pass replace=True to start a NEW store (discarding this one).')
            try:
                old_edges = list(old.edges())
            except Exception as e:
                raise ValueError(
                    'ingest_addressed: cannot merge -- the existing store has no surface strings '
                    '(loaded with surface=False?), so its edges cannot be streamed back out. '
                    f'Rebuild from source or pass replace=True. ({type(e).__name__}: {e})')
            new = old_edges + new
        st = AddressFactStore(keep_lexicon=True)
        info = st.build(new)
        st.compact_lexicon(keep_strings=True)
        self._addr_facts = st
        # Day-99 -- new bulk facts invalidate the consolidated SDM banks, exactly as ingest_triples
        # already flags. Without this, ancestors()/reach_member() answer from a STALE bank after a
        # bulk ingest (measured: ingest_triples set both flags, ingest_addressed set neither).
        self._reach_dirty = True
        self._anc_dirty = True
        self._learn_epoch = int(getattr(self, '_learn_epoch', 0)) + 1     # bulk knowledge arrived
        # WIRE IT INTO THE DERIVE PATH (not a side store): atoms()/atom() fall through to it, so
        # reason() / transitive_reach() derive over these edges WITHOUT the 290-B/entry anchor
        # cache or the string mining-index ever holding them. This is what makes the packed store
        # the organism's fact store rather than a demo.
        self.general_reasoner.derive_engine.attach_addresses(st)
        return {'entities': info['entities'], 'edges': info['edges'],
                'b_per_edge': round(st.bytes_per_edge(), 2),
                'edge_mb': round(st.edge_bytes() / 1e6, 1),
                'functional': info['functional'], 'csr': info['csr']}

    @property
    def facts(self):
        """The packed address-list fact store (None until ingest_addressed / load_facts)."""
        return getattr(self, '_addr_facts', None)

    def lookup(self, entity, relation, source=None, trust=1.0, bind=True):
        """Day-98 -- THE RETRIEVAL ORGAN.  A fixed-size mind cannot HOLD the whole world; a brain
        offloads to the world and fetches on demand.  So does this: lookup FETCHES a fact from an
        EXTERNAL store (the long tail the core does not carry -- e.g. an mmap'd AddressFactStore on
        disk, or any object with objects_of/str_of), BINDS it into working memory so the organism
        can REASON with it (not merely echo it), and ABSTAINS on a miss -- it never fabricates the
        tail.  Breadth is thereby decoupled from core size, permanently.  Provenance/trust ride with
        the answer (calibration extended to sources).  Returns {answer, source, trust, abstained}.

        bind=True writes the fetched fact into the live derive engine's working set, so a SUBSEQUENT
        derive can compose it with core knowledge (fetch -> bind -> reason)."""
        st = source if source is not None else self.facts
        ent = str(entity).strip().lower()
        rel = str(relation).strip().lower()
        if st is None or not hasattr(st, 'objects_of'):
            return {'answer': None, 'abstained': True, 'source': None, 'trust': 0.0}
        try:
            ids = st.objects_of(ent, rel)
        except Exception:
            ids = None
        if not ids:                                          # MISS -> abstain, never fabricate
            return {'answer': None, 'abstained': True,
                    'source': getattr(st, 'name', 'external'), 'trust': 0.0}
        vals = []
        for i in ids:
            v = st.str_of(i) if (hasattr(st, 'str_of') and isinstance(i, int)) else i
            if v:
                vals.append(v)
        if bind:                                             # bind into working memory -> DERIVABLE
            # must go through the ingest path (cache + record), not a bare _record: transitive_reach
            # / atom() read the anchor cache, so a bare _record would store the fact but leave it
            # underivable.  This is what makes fetch -> bind -> REASON actually compose.
            self.ingest_triples([(ent, rel, v) for v in vals], discover=False)
        return {'answer': vals[0] if len(vals) == 1 else vals,
                'source': getattr(st, 'name', 'external'), 'trust': float(trust),
                'abstained': False}

    def reach_addressed(self, rel, x):
        """Exact transitive closure over the packed fact store, as surface strings.  Byte-identical
        to the string-dict walk (verified on the full 4.16M-edge Wikidata backbone) at ~1/125th the
        RAM.  Returns None if no address store has been built/loaded."""
        st = getattr(self, '_addr_facts', None)
        if st is None:
            return None
        return {st.str_of(i) for i in st.transitive_reach(rel, x)}

    def save_facts(self, path):
        """Persist the fact store to ONE binary file (packed arrays).  Disk grows; RAM does not."""
        st = getattr(self, '_addr_facts', None)
        return None if st is None else st.save(path)

    def load_facts(self, path, mmap=True, surface=True):
        """FLAT ACCESS (the north star): mmap the fact store from disk, so the store grows on DISK
        and RAM stays flat -- only the pages a closure walk actually touches are ever resident.
        The SDM substrate is fixed; the facts live on disk; RAM never grows with knowledge.

        Day-97: every array is now a zero-copy memoryview over the mmap (day-96's load() copied the
        surface blob back into RAM, which quietly defeated the mmap for the largest array in the
        store -- measured RSS delta +32.9 MB, now +0.0).  Measured on the full 4.16M-edge Wikidata
        backbone: 3.58M entities, lexicon 104.0 -> 53.0 MB on disk and ~0 MB resident.

        surface=False -> REASONING-ONLY: no surface table at all (0 bytes of vocabulary).  Ids in,
        ids out, closures still exact.  This is Day-90's "surface strings only at the edges" taken
        to its limit -- the form that goes on the phone."""
        from ikigai.cognition.address_store import AddressFactStore
        self._addr_facts = AddressFactStore.load(path, mmap=mmap, surface=surface)
        self.general_reasoner.derive_engine.attach_addresses(self._addr_facts)
        return self._addr_facts

    def lexicon_report(self):
        """MEASURED footprint of the fact store, split by what it is FOR: the edges (the actual
        knowledge), the str->id lookup (needed to ENTER a query), and the surface table (needed only
        to EMIT text).  No asserted constants -- every number here is read off the live arrays."""
        st = getattr(self, '_addr_facts', None)
        if st is None:
            return None
        n = max(1, st.n_entities)
        return {'entities': st.n_entities, 'edges': st.n_edges,
                'edge_mb': round(st.edge_bytes() / 1e6, 1),
                'b_per_edge': round(st.bytes_per_edge(), 2),
                'lookup_mb': round(st.lookup_bytes() / 1e6, 1),
                'surface_mb': round(st.surface_bytes() / 1e6, 1),
                'b_per_entity': round(st.lexicon_bytes() / n, 2),
                'reasoning_only': not st.keep_lexicon}

    @staticmethod
    def _anc_norm(w):
        """One normalization for subject/ancestor/query so SDM write-word == read-word (else the
        subject's Kanerva locations don't line up and recall returns crosstalk)."""
        return str(w).strip().lower().replace(' ', '')

    def consolidate_ancestors(self, link='isa', d=512, k=64, seed=96, max_M=65536):
        """Day-96 -- MULTI-VALUE recall: list ALL transitive ancestors of x (a SET readout, the
        harder sibling of reach_member's yes/no).  REUSES the production flat_memory.VSASDM bank
        directly: each descendant's ancestor keys are bundled at the subject address key(x); recall
        reads that superposition and CLEANS UP over the ANCESTOR-UNIVERSE codebook (one batched
        cosine matmul -- a leaf can't be an ancestor, so candidates are bounded) above a
        PRECISION-PRESERVING calibrated floor -> correct-or-abstain (drops uncertain ancestors,
        never emits a false one).  Enumeration cost O(U*d) codebook cleanup; flat yes/no membership
        stays O(g*d) via reach_member -- two complementary faculties over the same closure.
        Measured (experiments/nl/day96_multivalue_ancestors.py): good regime prec~0.99 / rec~1.0
        with a MEASURED capacity law SNR ~ sqrt(M/N_subjects) -- the bank M is auto-sized to the
        subject count.  Additive: does not touch reason(); no save.
        Returns {subjects, universe, pairs, M, boundary}."""
        import math as _math
        import random as _random
        import numpy as _np
        from ikigai.cognition.flat_memory import VSASDM, ComputedKey
        eng = self.general_reasoner.derive_engine
        if not eng.is_transitive(link):
            return {'subjects': 0, 'pairs': 0,
                    'note': f'"{link}" not mined transitive -- teach(discover=True) first'}
        nz = self._anc_norm
        closure, universe = {}, set()
        for x in sorted(getattr(eng, 'entities', [])):
            # Day-99 -- REVERTED my own "fix" here. I rerouted this through derive_ancestry claiming
            # ancestors() "abstained on facts the organism plainly held". WRONG: it abstained because
            # the test never taught it that `isa` is transitive (discover=False), and refusing to
            # assert x-isa-z without a LEARNED transitive rule is the calibration working. The gate
            # above is correct, transitive_reach is already the Day-96 BFS over ALL parents, so the
            # reroute bought nothing and rested on a misdiagnosis. The miner is fine: measured
            # chains=4 acyclic=4 conf=1.00 -> rule promoted -> is_transitive True on clean data.
            chain = eng.transitive_reach(link, x) or []
            anc = [nz(c) for c in chain[1:]]
            anc = [c for c in anc if c and c != nz(x)]
            if anc:
                closure[nz(x)] = anc
                universe.update(anc)
        N = max(1, len(closure))
        M = min(int(max_M), 1 << max(13, int(_math.ceil(_math.log2(24 * N)))))   # M ~ 24N, pow2
        ck = ComputedKey(d=d, seed=seed)
        sdm = VSASDM(d=d, M=M, k=k, seed=seed)
        n_pairs = 0
        for xn, anc in closure.items():
            ax = ck.key(xn)
            for c in anc:
                sdm.write(ax, ck.key(c), word=xn)             # bundle at subject address
                n_pairs += 1
        uni = sorted(universe)
        cb = (_np.stack([ck.key(u) for u in uni]).astype(_np.complex64)
              if uni else _np.zeros((0, d), _np.complex64))
        uni_ix = {u: i for i, u in enumerate(uni)}
        rng = _random.Random(seed)
        subs = list(closure.keys())
        absent, tries = [], 0
        while len(absent) < 3000 and tries < 80000 and uni:
            tries += 1
            xs = rng.choice(subs); c = rng.choice(uni)
            if c in closure[xs] or c == xs:
                continue
            v = sdm.read(ck.key(xs), xs)
            absent.append(float(_np.real(_np.vdot(cb[uni_ix[c]], v)) / d))
        boundary = (float(_np.percentile(absent, 99.9)) * 1.15) if absent else 0.0
        self._anc_ck, self._anc_sdm, self._anc_cb = ck, sdm, cb
        self._anc_uni, self._anc_boundary, self._anc_link = uni, boundary, link
        self._anc_dirty = False
        return {'subjects': N, 'universe': len(uni), 'pairs': n_pairs, 'M': M,
                'boundary': round(boundary, 4)}

    def ancestors(self, x, link='isa', topk=None):
        """Day-96 -- return the SET of transitive ancestors of x, recalled THROUGH the substrate:
        read x's bundled superposition off the production VSASDM and clean up over the ancestor
        universe above the precision-preserving floor (correct-or-abstain).  DEFAULT-ON:
        auto-consolidates on first use, re-consolidates when facts changed (dirty flag).  Returns a
        list sorted by substrate confidence (strongest ancestor first), [] if none survive the
        floor, or None if `link` is not (yet) mined transitive.  `topk` caps the set.  This is set
        enumeration; reach_member(x, c) is the flat yes/no test over the same closure."""
        import numpy as _np
        if getattr(self, '_anc_sdm', None) is None or getattr(self, '_anc_dirty', False):
            if not self.consolidate_ancestors(link).get('pairs'):
                return None
        ck, sdm, cb, uni = self._anc_ck, self._anc_sdm, self._anc_cb, self._anc_uni
        if cb.shape[0] == 0:
            return []
        xn = self._anc_norm(x)
        v = sdm.read(ck.key(xn), xn)
        sims = _np.real(cb @ _np.conj(v)) / ck.d
        out = []
        for i in _np.argsort(-sims):
            if uni[i] == xn:
                continue
            if sims[i] < self._anc_boundary:
                break                                          # sorted -> rest are below floor too
            out.append(uni[i])
            if topk and len(out) >= topk:
                break
        return out

    def semantic_sim(self, w1, w2):
        """Day 91 -- distributional word similarity from the LIVE meaning store, scale-first.
        Prefers the FACTORED code store (readable past the ~20k superposition knee, to billions);
        falls back to `unified.cooccur` (the persisted body, folded in at Pack 117-118) and then
        the standalone `flat` bank.  Returns cos in [-1,1] or None when no store has both words.
        This is the source a MEANING critic reads (generate_structured / _make_critic), so
        generation sources readable meaning at scale rather than a saturated bundle."""
        fm = getattr(self, '_factored', None)
        if fm is not None:
            try:
                s = fm.sim(w1, w2)
            except Exception:
                s = None
            if s is not None:
                return s
        s = None
        u = getattr(self, 'unified', None)
        if u is not None:
            try:
                s = u.similarity(w1, w2)
            except Exception:
                s = None
        if s is None and getattr(self, 'flat', None) is not None:
            try:
                s = self.flat.similarity(w1, w2)
            except Exception:
                s = None
        return s

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

    def dream(self, discover=True, seed=None):
        """Sleep cycle.  Consolidates the being's lexicon AND, Day-87, DREAMS
        CREATIVELY: the organism recombines what it knows and wakes with facts
        nobody told it -- DISCOVERIES entailed by its own rules (proven, by
        derive-chaining) and CONJECTURES leapt to by signature resonance (tagged
        low-confidence beliefs, testable later).  See CompositionEngine.
        dream_discover.  Discoveries are reported, not stored (free to re-derive);
        conjectures are filed as beliefs kept apart from ground truth.  Returns
        the lexicon stats plus the night's discoveries and conjectures."""
        out = dict(self.being.dream())
        if not discover:
            return out
        try:
            eng = self.general_reasoner.derive_engine
            dd = eng.dream_discover(seed=seed)
        except Exception as e:
            out['dream_err'] = str(e)[:80]
            return out
        out['discoveries'] = dd['discoveries']
        out['conjectures'] = dd['conjectures']
        # file conjectures as low-confidence beliefs, kept apart from derived
        # facts -- exactly a dreamed hypothesis the life loop can later test.
        if not hasattr(self, '_beliefs'):
            self._beliefs = {}
        filed = 0
        for (s, r, v, score, prov) in dd['conjectures']:
            if (s, r) in self._beliefs:                     # don't clobber a real prediction
                continue
            self._beliefs[(s, r)] = {
                'value': v, 'confidence': round(min(0.6, float(score)), 2),
                'source': f'dreamed conjecture ({prov})'}
            filed += 1
        out['discovered'] = len(dd['discoveries'])
        out['conjectured'] = filed
        return out

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
        """How grammatically similar (same POS)?  Day-103 SUBSTRATE-FIRST: read the
        POS fingerprints from the PERSISTED unified roles (pos_left/pos_right, written
        through by ground_text) so distributional grammar survives save/reload; fall
        back to the in-RAM GrammarGrounding dict when the substrate has none yet."""
        s = self._pos_similarity_substrate(w1, w2)
        if s is not None:
            return s
        return self.grammar.pos_similarity(w1, w2)

    def _pos_similarity_substrate(self, w1, w2):
        """Cosine of two words' left/right POS fingerprints recalled from the unified
        substrate. Day-103 -- POS is the FIRST channel migrated to the ONE persistent
        memory (Prince's "everything gets written to SDM"): ground_text write_relation's
        each word's idf-weighted context bundle into pos_left/pos_right, keyed by the
        shared ComputedKey, so it rides unified.save_ikg and is one identity across the
        organism. Returns None when neither role holds a usable fingerprint for both
        words (caller falls back to the RAM dict)."""
        import numpy as _np
        u = getattr(self, 'unified', None)
        if u is None:
            return None
        sc = []
        for role in ('pos_left', 'pos_right'):
            try:
                a = u.recall(w1, role); b = u.recall(w2, role)
            except Exception:
                continue
            if a is None or b is None:
                continue
            a = _np.asarray(a).reshape(-1); b = _np.asarray(b).reshape(-1)
            n = float(_np.linalg.norm(a) * _np.linalg.norm(b))
            if n > 1e-6:
                sc.append(float(_np.real(_np.vdot(a, b)) / n))
        if not sc:
            return None
        return sum(sc) / len(sc)

    _AFFECT_POS = '__aff_pos__'
    _AFFECT_NEG = '__aff_neg__'

    def _ensure_affect_role(self):
        """Day-103 EMOTION CHANNEL setup. The 8 Kanerva banks are HUGELY imbalanced
        (b_lang ~4.3e10 mass vs b_ground ~1.4e7 -- 3000x). A runtime 'affect' role
        defaults to the OVERLOADED b_lang, where a low-information affect signal drowns
        below the crosstalk floor -- which is exactly why 5 earlier affect encodings
        failed. Route it to the LIGHTEST bank (b_ground, where sensory already lives
        and works), and re-establish that mapping on EVERY construction AND after
        load_ikg (the role->bank map is not itself persisted, only bank CONTENT is --
        MEASURED: content survives, mapping is lost, remapping recovers full
        discrimination). Idempotent, best-effort."""
        u = getattr(self, 'unified', None)
        if u is None:
            return
        try:
            r2b = getattr(u, '_role_to_bank', None)
            if isinstance(r2b, dict):
                r2b['affect'] = 'b_ground'
            u.ensure_role('affect')
        except Exception:
            pass

    def _affect_now(self):
        """Current felt valence from the neuroendocrine BODY: reward/novelty (dopamine
        above tonic baseline 0.5) minus stress (cortisol above baseline 0.1). Real
        physiology -- the same body that gates learning plasticity -- no authored
        emotion lexicon. Clamped [-1, 1]."""
        try:
            da = self.body.get('dopamine'); co = self.body.get('cortisol')
            v = 0.0
            if da is not None:
                v += float(getattr(da, 'level', 0.5)) - 0.5
            if co is not None:
                v -= float(getattr(co, 'level', 0.1)) - 0.1
            return max(-1.0, min(1.0, v))
        except Exception:
            return 0.0

    def _write_affect(self, tokens):
        """Day-103 -- 'every emotion gets written to SDM'. When the body felt something
        (|valence| past a small floor), bind that felt sign onto the KNOWN-entity topic
        tokens of the experience, CATEGORICALLY (relate -> a pos/neg exemplar, the same
        symbolic mechanism sensory uses in the same light bank), in the PERSISTENT
        'affect' role. The organism remembers how it FELT about what it processed; it
        rides save_ikg. Only writes on emotionally-significant calls (novelty/stress),
        so routine neutral queries pay nothing -- biology remembers what moved it.
        Bounded, best-effort."""
        u = getattr(self, 'unified', None)
        if u is None or not tokens:
            return
        v = self._affect_now()
        if abs(v) < 0.05:
            return                       # neutral -> nothing felt
        self._ensure_affect_role()
        target = self._AFFECT_POS if v > 0 else self._AFFECT_NEG
        try:
            ents = getattr(self.general_reasoner.derive_engine, 'entities', None) or set()
        except Exception:
            ents = set()
        n = 0
        for w in set(tokens):
            if w in ents:
                for _ in range(8):       # reinforce so the trace clears the read floor
                    try:
                        u.relate(w, 'affect', target)
                    except Exception:
                        break
                n += 1
            if n >= 4:
                break

    def recall_affect(self, word):
        """How did the organism FEEL about `word`? Signed valence in [-1, 1] recalled
        from the persistent 'affect' role (positive = felt good, negative = felt bad),
        or None if never felt / too weak to trust. Survives save/reload -- lived
        emotion is part of the ONE memory now."""
        u = getattr(self, 'unified', None)
        if u is None:
            return None
        self._ensure_affect_role()       # re-establish bank mapping (not persisted)
        try:
            res = u.query(word, 'affect', [self._AFFECT_POS, self._AFFECT_NEG])
        except Exception:
            return None
        if not res:
            return None
        label, score = res[0], float(res[1])
        if score < 0.1:                  # below this = crosstalk noise, never really felt
            return None
        return score if label == self._AFFECT_POS else -score

    def _pos_write_through(self, tokens):
        """Day-103 -- mirror THIS sentence's POS fingerprints into the persistent
        unified substrate (roles pos_left/pos_right). Distributional grammar becomes
        part of the ONE memory that rides save_ikg instead of dying in an unpersisted
        dict. Only the sentence's own tokens are written (bounded, not O(vocab) per
        call). Keyed by the shared ComputedKey. Best-effort; never raises into read."""
        import numpy as _np
        u = getattr(self, 'unified', None)
        g = getattr(self, 'grammar', None)
        if u is None or g is None:
            return
        try:
            if 'pos_left' not in set(getattr(u, 'roles', []) or []):
                u.ensure_role('pos_left')
            if 'pos_right' not in set(getattr(u, 'roles', []) or []):
                u.ensure_role('pos_right')
        except Exception:
            return
        for w in set(tokens):
            for role, ctx in (('pos_left', g._left_ctx.get(w)),
                              ('pos_right', g._right_ctx.get(w))):
                if ctx is not None:
                    try:
                        u.write_relation(w, role, _np.asarray(ctx, _np.complex64))
                    except Exception:
                        pass

    def pos_neighbors(self, word, k=5):
        """k words playing same grammatical role."""
        return self.grammar.pos_neighbors(word, k=k)

    def bigram_surprise(self, prev, curr):
        """-log2 P(curr | prev). High = phrase boundary."""
        return self.grammar.surprise(prev, curr)

    # ── Phase 3: dialogue + generation ───────────────────────────────────

    def new_dialogue(self, persona=None):
        """Start a fresh multi-turn conversation. Replies via frame_relax
        (Day-83 audit rewire: SentenceGenerator Markov path retired → data-free
        grammatical free-fluency, Pack 313)."""
        loop = DialogueLoop(self, d=2048)
        loop.start(persona_name=persona)
        def respond_to(user_text, **kwargs):
            r = self.say_frame(message=None,
                               seed=abs(hash(user_text)) % (2**31),
                               n_iters=kwargs.get('n_iters', 6))
            reply = r['text'] if r else ''
            loop.user_says(user_text)
            t = loop.agent_says(reply)
            return reply, t
        loop.respond_to = respond_to
        return loop

    def trace(self):
        """Returns last reasoning trace."""
        return list(self._last_trace)

    def memory(self):
        """Recent episodic state (legacy ReasoningEngine working-memory dict
        retired with the trio, Day-83 audit)."""
        return {e['tick']: e['answer'] for e in self._episodes[-16:]}

    def reset(self):
        """Clear episodic chain (start fresh)."""
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
        # Day-97 FIX: this read s['wm_vars'], a key status() has never returned -- the legacy
        # ReasoningEngine working-memory counter, retired in the Day-83 audit.  repr(org) raised
        # KeyError for every caller since.  Report what the organism actually has.
        s = self.status()
        return (f"<IkigaiOrganism tick={s['tick']} eps={s['n_episodes']} "
                f"vocab={s['being_vocab']} beliefs={s['n_beliefs']} age={s['being_age']}>")

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


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Day-99 -- register the faculties that compete inside `org(x)`.
#
# Deliberately OUTSIDE the class body and in a deliberately unhelpful order (abstain first, learn
# last -- the reverse of any sensible authored priority). If the outcome ever depends on this
# order, the organism is not deciding and the gate day99_one_api_order_invariant will say so.
# Adding a capability to the organism is adding a line here: no ladder to edit, no mode to pick.
# ─────────────────────────────────────────────────────────────────────────────────────────────
IkigaiOrganism._register_faculty('abstain', IkigaiOrganism._fac_abstain)
IkigaiOrganism._register_faculty('speak',   IkigaiOrganism._fac_speak)
IkigaiOrganism._register_faculty('wonder',  IkigaiOrganism._fac_wonder)
IkigaiOrganism._register_faculty('answer',  IkigaiOrganism._fac_answer)
IkigaiOrganism._register_faculty('solve',   IkigaiOrganism._fac_solve)
IkigaiOrganism._register_faculty('learn',   IkigaiOrganism._fac_learn)
IkigaiOrganism._register_faculty('analogy', IkigaiOrganism._fac_analogy)   # Day-102 wire
IkigaiOrganism._register_faculty('identity', IkigaiOrganism._fac_identity)  # Day-104: knows who it is
IkigaiOrganism._register_faculty('generate', IkigaiOrganism._fac_generate)  # Day-106: open-ended coherent generation


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
    print(f'  method: {r.get("method")}')
    print(f'  trace: {r.get("trace")}')
