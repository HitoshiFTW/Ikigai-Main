"""ikigai.cognition — high-level reasoning capabilities atop the organism."""

from ikigai.cognition.routers import ToolRouter
from ikigai.cognition.memory import TransitionMemory
from ikigai.cognition.moe import MoERouter, Codebook
from ikigai.cognition.wake_sleep import WakeSleepCompressor
from ikigai.cognition.reasoning import (
    RelationAlgebra,
    SemanticRoleMemory,
    EventSequenceMemory,
    detect_pattern,
    compose,
    synthesize_composition,
    generate_composition_code,
)
from ikigai.cognition.code_gen import (
    CodeIndex,
    encode_function,
    encode_query,
    generate_function,
    generate_with_verifier,
    generate_with_index,
    build_common_index,
    rename_function,
    compile_ok,
    exec_and_test,
)
from ikigai.cognition.verifier import (
    solve_greedy,
    solve_verifier,
    solve_chain,
    verify_trajectory,
    trajectory_hv,
    extract_numbers,
    extract_numbers_smart,
    preprocess_relational,
    _detect_final_op,
)
from ikigai.cognition.conversation import (
    Conversation,
    IntentRouter,
    EpisodicBuffer,
    build_conversation,
)
from ikigai.cognition.phasor_state import (
    PhaseLockedHolographicBuffer,
    random_phasor, normalize_phase, bind as phasor_bind, unbind as phasor_unbind,
    rotate as phasor_rotate, cosine as phasor_cosine, superpose as phasor_superpose,
    HV_DIM, OMEGA_DEFAULT,
)
from ikigai.cognition.dssc_coupling import (
    ParallelSemSynCoupling,
    GrammarCFG,
    build_default_cfg,
)
from ikigai.cognition.persona_manifold import (
    BeliefProjectionManifold,
    VALENCE, AROUSAL, CERTAINTY, FORMALITY, TECHNICALITY,
)
from ikigai.cognition.free_energy_drive import (
    ConversationalVariationalFreeEnergyField,
    ACTIONS as CVFEF_ACTIONS,
)
from ikigai.cognition.crystallizer import (
    AtomicCrystallineStore,
    WILDCARD as ACCI_WILDCARD,
)
from ikigai.cognition.hebbian_tuner import HebbianVocabTuner
from ikigai.cognition.ngram_expander import NGramExpander
from ikigai.cognition.belief_expander import BeliefConditionedExpander
from ikigai.cognition.skill_crystal import SkillCrystal
from ikigai.cognition.multiresngram import MultiResNGram
from ikigai.cognition.goal_decomposer import GoalDecomposer, PlanStep, ATOMIC_ACTIONS
from ikigai.cognition.self_verifier import SelfVerifier
from ikigai.cognition.world_model import SymbolicWorldModel
from ikigai.cognition.schema_inducer import SchemaInducer, anti_unify, apply_schema, SLOT
from ikigai.cognition.cross_modal_binder import CrossModalBinder
from ikigai.cognition.schema_refiner import SchemaRefiner
from ikigai.cognition.metacognitive_mirror import MetacognitiveHVMirror
from ikigai.cognition.generation_pipeline import GenerationPipeline, RunResult
from ikigai.cognition.self_modifying_refiner import SelfModifyingRefiner, SelfModifyingSchema
from ikigai.cognition.benchmark_runner import BenchmarkRunner, FrozenBaseline
from ikigai.cognition.substrate_adapter import SubstrateAdapter
from ikigai.cognition.no_forgetting_proof import NoForgettingProof
from ikigai.cognition.hot_loader import CognitionHotLoader
from ikigai.cognition.holographic_memory import HolographicMemory, _encode_semantic
from ikigai.cognition.vsa_calculus import VSACalculus
from ikigai.cognition.belief_field import BeliefField
from ikigai.cognition.cross_time_resonance import CrossTimeResonator
from ikigai.cognition.cross_modal_space import (
    CrossModalSpace, encode_text as cm_encode_text,
    encode_vision, encode_audio,
)
from ikigai.cognition.concept_atomizer import ConceptAtomizer
from ikigai.cognition.proof_carrying_gen import ProofChain, ProofCarryingGenerator
from ikigai.cognition.counterfactual_sim import CounterfactualField
from ikigai.cognition.inverse_compile import InverseCompiler
from ikigai.cognition.importance_decay import ImportanceDecayLattice
from ikigai.cognition.adversarial_immune import AdversarialImmune
from ikigai.cognition.theory_of_mind import TheoryOfMindSandbox, AgentMind
from ikigai.cognition.algebraic_closure import AlgebraicClosure
from ikigai.cognition.unified_organism import UnifiedOrganism
from ikigai.cognition.fe_action import FreeEnergyActionSelector
from ikigai.cognition.causal_world_model import CausalWorldModel
from ikigai.cognition.multistep_planner import MultiStepPlanner
from ikigai.cognition.curiosity_drive import CuriosityDrive
from ikigai.cognition.persona_fe_coupling import PersonaFEC
from ikigai.cognition.benchmark_harness_v2 import (
    Problem, SolverResult, BenchmarkReport, BenchmarkHarness,
    numeric_scorer, exact_match_scorer, multiple_choice_scorer, code_exec_scorer,
)
# Day 75: gsm8k_solver_v2-v4 retired -- replaced by Pack 254 MathEval
# (engine='auto', Pack 257 RHC) + Pack 253 emergent operator + word-magnitude
# grounding. Old solvers were hand-curated handler stacks.
from ikigai.cognition.cgpsp_encoder import CGPSPEncoder
from ikigai.cognition.pi_k_algebra import PiK
from ikigai.cognition.pgmw import PersonaGrid
from ikigai.cognition.sac_field import SACField, BasinField

__all__ = [
    'ToolRouter',
    'TransitionMemory',
    'MoERouter', 'Codebook',
    'WakeSleepCompressor',
    'RelationAlgebra', 'SemanticRoleMemory', 'EventSequenceMemory',
    'detect_pattern', 'compose', 'synthesize_composition', 'generate_composition_code',
    'CodeIndex', 'encode_function', 'encode_query',
    'generate_function', 'generate_with_verifier', 'generate_with_index',
    'build_common_index', 'rename_function', 'compile_ok', 'exec_and_test',
    'solve_greedy', 'solve_verifier', 'solve_chain', 'verify_trajectory',
    'trajectory_hv', 'extract_numbers', 'extract_numbers_smart',
    'Conversation', 'IntentRouter', 'EpisodicBuffer', 'build_conversation',
    'PhaseLockedHolographicBuffer',
    'random_phasor', 'normalize_phase',
    'phasor_bind', 'phasor_unbind', 'phasor_rotate', 'phasor_cosine', 'phasor_superpose',
    'HV_DIM', 'OMEGA_DEFAULT',
    'ParallelSemSynCoupling', 'GrammarCFG', 'build_default_cfg',
    'BeliefProjectionManifold',
    'VALENCE', 'AROUSAL', 'CERTAINTY', 'FORMALITY', 'TECHNICALITY',
    'ConversationalVariationalFreeEnergyField', 'CVFEF_ACTIONS',
    'AtomicCrystallineStore', 'ACCI_WILDCARD',
    'HebbianVocabTuner',
    'NGramExpander',
    'BeliefConditionedExpander',
    'SkillCrystal',
    'MultiResNGram',
    'GoalDecomposer', 'PlanStep', 'ATOMIC_ACTIONS',
    'SelfVerifier',
    'SymbolicWorldModel',
    'SchemaInducer', 'anti_unify', 'apply_schema', 'SLOT',
    'CrossModalBinder',
    'SchemaRefiner',
    'MetacognitiveHVMirror',
    'GenerationPipeline', 'RunResult',
    'SelfModifyingRefiner', 'SelfModifyingSchema',
    'BenchmarkRunner', 'FrozenBaseline',
    'SubstrateAdapter',
    'NoForgettingProof',
    'CognitionHotLoader',
    'HolographicMemory',
    'VSACalculus',
    'BeliefField',
    'CrossTimeResonator',
    'CrossModalSpace', 'cm_encode_text', 'encode_vision', 'encode_audio',
    'ConceptAtomizer',
    'ProofChain', 'ProofCarryingGenerator',
    'CounterfactualField',
    'InverseCompiler',
    'ImportanceDecayLattice',
    'AdversarialImmune',
    'TheoryOfMindSandbox', 'AgentMind',
    'AlgebraicClosure',
    'UnifiedOrganism',
    'FreeEnergyActionSelector',
    'CausalWorldModel',
    'MultiStepPlanner',
    'CuriosityDrive',
    'PersonaFEC',
    'Problem', 'SolverResult', 'BenchmarkReport', 'BenchmarkHarness',
    'numeric_scorer', 'exact_match_scorer', 'multiple_choice_scorer', 'code_exec_scorer',
    'solve_gsm8k_v2', 'solve_gsm8k_v3', 'solve_gsm8k_v4',
    'CGPSPEncoder',
    'PiK',
    'PersonaGrid',
    'SACField', 'BasinField',
]
