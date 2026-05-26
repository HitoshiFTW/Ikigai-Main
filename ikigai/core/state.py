# ===========================================================================
# CANONICAL STATE TYPES  (Day 32 Pack 4 -- Controlled Modular Extraction)
# Extracted from ikigai.py. ikigai.py keeps _IKG = IkigaiContext() so each
# exec() call gets its own fresh instance -- no module-cache singleton leak.
# ===========================================================================


class IkigaiState:
    """Per-tick state carrier. Pass via system.update(state). Zero drift."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class IkigaiContext:
    """Named runtime state replacing anonymous globals().get() accesses."""
    def __init__(self):
        self.sleeping = True   # safe default: same semantics as old globals().get('sleeping', True)
        self.l23      = None   # SystemRigimeTracker (cortical L2/3 energy)
        self.ado      = None   # AdenosineSystem
        # Day 33 Pack 1 -- Temporal Credit Bridge (read-only signal, default OFF)
        self.enable_temporal_bridge  = False
        self.temporal_credit_signal  = 0.0
        self.temporal_depth          = 0
        # Day 33 Pack 2 -- Temporal Integration Layer (adjusted credit, default OFF)
        self.enable_temporal_integration = False
        self.adjusted_credit_signal      = 0.0
        # Day 33 Pack 3 -- Selective Ignition Layer (ignited confidence, default OFF)
        self.enable_selective_ignition   = False
        self.ignited_confidence_signal   = 0.0
        # Day 33 Pack 4 -- Decision Bridge Layer (override bind signal, default OFF)
        self.enable_decision_bridge      = False
        self.override_bind_signal        = False
        # Day 34 Pack 1 -- Confidence Variance Injection (dampening, default OFF)
        self.enable_confidence_variance  = False
        # Day 34 Pack 2 -- Candidate-level Confidence Differentiation (default OFF)
        self.enable_candidate_variance   = False
        # Day 34 Pack 3 -- Action Routing Correction (windowed fallback, default OFF)
        self.enable_routing_correction   = False
        # Day 34 Pack 4 -- Background Policy Diversification (default OFF)
        self.enable_bg_policy = False
        self.bg_energy        = 0.5  # l23 mean energy; 0.5 = safe default
        self.bg_pe            = 0.0  # pp.error; 0.0 = use confidence-gap proxy
        # Day 35 Pack 1 -- Action-State Coupling Layer (default OFF)
        self.enable_action_state_coupling = False
        # Day 35 Pack 2 -- Confidence Compression (near-threshold regime, default OFF)
        self.enable_confidence_compression = False
        # Day 35 Pack 3 -- BG Policy Decomposition (anti-monopoly, default OFF)
        self.enable_bg_decomposition = False
        # Day 36 Pack 1 -- Credit-to-Decision Coupling (action-conditioned, default OFF)
        self.enable_credit_coupling = False
        self.last_action_type       = None   # previous tick's selected action; read by ACE
        # Day 36 Pack 2 -- BG Credit Modulation (default OFF)
        self.enable_bg_credit       = False
        # Day 36 Pack 2 -- Execution Diversity (BG->exec type remap, default OFF)
        self.enable_exec_diversity  = False
        self.exec_type              = None   # remapped execution type; set when enabled
        # Day 36 Pack 3 -- Score-Based BG Selection (continuous pressure, default OFF)
        self.enable_bg_scoring      = False
        # Day 36 Pack 4 -- BG Score Rebalancing (equal-scale bases, default OFF)
        self.enable_bg_rebalance    = False
        # Day 36 Pack 5 -- Credit-to-Candidate Coupling (candidate-level only, default OFF)
        # Credit shifts candidate ranking; top_confidence NEVER modified.
        # Keeps TC natural (~0.53) so BG routing remains reachable.
        self.enable_credit_selection = False
        # Day 36 Pack 6 -- PE Exploration Pressure inside BG (default OFF)
        # High PE boosts explore, suppresses approach. Only active inside BG scoring path.
        self.enable_pe_exploration   = False
        # Day 37 Pack 1 -- Novelty Pressure (default OFF)
        # Less-visited coarse states boost explore; no randomness; BG scoring only.
        self.enable_novelty  = False
        self.novelty_counts  = {}
        # Day 37 Pack 3 -- Contextual Curiosity (default OFF)
        # Explore only when novelty AND credit-vs-PE mismatch are both high.
        self.enable_contextual_curiosity = False
        # Day 37 Pack 4 -- Curiosity Reinforcement (default OFF)
        # Exploration that yields good outcomes is remembered and repeated.
        self.enable_curiosity_reinforcement = False
        self.curiosity_value = {}   # state key -> learned exploration value [0,1]
        # Day 37 V2 -- Proto-Representation: Transition Counts (always active, no flag)
        # Records (prev_state_key -> curr_state_key) pair frequencies. Pure observation.
        # No decay, no scoring use. Feeds future prediction / pattern learning.
        self.transition_counts = {}
        self._prev_state_key   = None
        # Day 43 Pack 1 -- Controlled Environment Layer (default OFF).
        # Small deterministic state space with stable corridors and branch/merge regions.
        # OFF means no env_step calls, no env state mutation, and no transition-key substitution.
        self.enable_env = False
        # Day 43 Pack 2 -- Action -> State Coupling (default OFF).
        # When ON: action adds a small integer shift to the visit-based branch index.
        # When OFF: env_step() is bit-identical to Pack 1 (visit % n_options only).
        # K_ENV_ACT is an integer shift, NOT a float scale.
        self.enable_env_action = False
        self.K_ENV_ACT = 1
        # Day 43 Pack 3 -- Outcome Diversity Guarantee (default OFF).
        # Converts absolute TE into a relative outcome by normalizing within the
        # observed [te_min, te_max] range.  OFF = bit-identical to Pack 2.
        # vsa_te_min / vsa_te_max track the running TE range seen at S_real events.
        self.enable_env_outcome = False
        self.vsa_te_min = 1.0   # initialised high; shrinks toward observed minimum
        self.vsa_te_max = 0.0   # initialised low;  grows  toward observed maximum
        # Day 43 Pack 4 -- Outcome-Conditioned Action Bias (default OFF).
        # Positive vsa_outcome boosts approach/wait; negative boosts explore.
        # K_OUTCOME_BIAS < K_OUTCOME (0.005). OFF = bit-identical to Pack 3.
        self.enable_env_outcome_bias = False
        self.K_OUTCOME_BIAS = 0.004
        # Day 43 Pack 6 -- Shift-Gated Outcome Calibration (default OFF).
        # Tiny calibration nudge on Pack 4 bias when Pack 5 shift is weak.
        # K_SHIFT_CAL < K_OUTCOME_BIAS (0.004). OFF = bit-identical to Pack 5.
        self.enable_env_shift_cal = False
        self.K_SHIFT_CAL = 0.002
        self.env_shift_target = 0.5
        self.env_shift_cal = 0.0
        # Day 43 Pack 5 -- Adaptive Shift Detection (default OFF).
        # Observation-only: tracks whether outcome sign co-occurs with action type.
        # No behavior change; zero drift when OFF.
        self.enable_env_shift = False
        self.env_shift_count = 0
        self.env_shift_pos = 0
        self.env_shift_neg = 0
        self.env_shift_corr = 0.0
        self.env_shift_rate = 0.0
        self.env_shift_neg_rate = 0.0
        self.env_shift_summary = 0.0
        # Day 43 Pack 7 -- Regime-Split Exposure (default OFF).
        # Second deterministic transition graph gated by shift saturation.
        # OFF = bit-identical to Pack 6. Switching is threshold-triggered, never random.
        self.enable_env_regime = False
        self.env_regime = 0
        self.env_regime_count = 0
        self.env_regime_switches = 0
        self.env_regime_trigger = -150.0
        # Day 44 Pack 1 -- Minimal Task Loop (default OFF).
        # Deterministic symbolic task graph. Prediction via transition memory.
        # Action influences branch vs corridor traversal. No reward injection.
        # OFF = bit-identical to Day 43 Pack 12.
        self.enable_task_loop = False
        self.task_state = 0
        self.task_step = 0
        self.task_input = None
        self.task_target = None
        self.task_correct = 0
        self.task_incorrect = 0
        self.task_total = 0
        self.task_transition_counts = {}
        # Day 44 Pack 2 -- Context-Dependent Task Ambiguity (default OFF).
        # State 3 ambiguous: branch-path context (ctx_a==5) -> 3->1; else 3->0.
        # Two-step local context window. Zero drift when OFF.
        self.enable_task_ambiguity = False
        self.task_ctx_a = None   # task state 2 steps back
        self.task_ctx_b = None   # task state 1 step back
        self.task_ctx_score = 0  # count of mode-1 (branch-context) resolutions
        self.task_ctx_mode = 0   # 0=normal, 1=ambiguous branch-context resolution
        # Day 44 Pack 3 -- Deterministic Difficulty Cycle (default OFF).
        # Hard phase (step % cycle_len >= cycle_len//2): state 1 -> 4 (forces branch path).
        # Exposes ambiguous state-3 more often; interacts with Pack 2 context ambiguity.
        # Zero drift when OFF. No randomness. All fields have write AND read paths.
        self.enable_task_difficulty_cycle = False
        self.task_cycle_len = 64
        self.task_cycle_phase = 0
        self.task_hard_mode = False
        self.task_cycle_easy_ticks = 0
        self.task_cycle_hard_ticks = 0
        self.task_cycle_switches = 0
        # Day 44 Pack 4 -- Deterministic Recovery Window (default OFF).
        # Failure streak in hard mode (non-resetting on pass) triggers 16-tick recovery.
        # Recovery: state-1 routes to easy path (2) instead of hard path (4).
        # All 7 fields have write AND read paths. Zero drift when OFF.
        self.enable_task_recovery = False
        self.task_fail_streak = 0
        self.task_recovery_mode = False
        self.task_recovery_timer = 0
        self.task_recovery_count = 0
        self.task_recovery_easy_ticks = 0
        self.task_recovery_hard_ticks = 0
        # Day 44 Pack 5 -- Held-Out Generalization Probe (default OFF).
        # Every 128 task steps, first 16 = probe window. Probe: state 2 routes to 5
        # (transition 2->5 never appears in Pack 1-4 normal/hard/recovery flow).
        # Probe ticks counted separately; transition_counts not updated during probe.
        # Pack 4 fail_streak gated off during probe. Zero drift when OFF.
        self.enable_task_holdout_probe = False
        self.task_holdout_len = 16
        self.task_holdout_phase = 0
        self.task_holdout_active = False
        self.task_holdout_switches = 0
        self.task_holdout_total = 0
        self.task_holdout_correct = 0
        self.task_holdout_incorrect = 0
        self.task_holdout_seen = 0
        # Day 44 Pack 6 -- Probe Aftereffect Tracking (default OFF).
        # Measures normal task acc and TE in 8-tick windows before/after each probe.
        # Pre-window: phases 120-127 (last 8 steps before probe). Post-window: phases 16-23.
        # Measurement only. Zero drift when OFF. All 14 fields write+read.
        self.enable_task_probe_aftereffect = False
        self.task_probe_pre_ticks = 0
        self.task_probe_post_ticks = 0
        self.task_probe_window_len = 8
        self.task_probe_pre_acc = 0.0
        self.task_probe_post_acc = 0.0
        self.task_probe_pre_te = 0.0
        self.task_probe_post_te = 0.0
        self.task_probe_pre_correct = 0
        self.task_probe_post_correct = 0
        self.task_probe_pre_total = 0
        self.task_probe_post_total = 0
        self.task_probe_acc_gap = 0.0
        self.task_probe_te_gap = 0.0
        # Day 44 Pack 7 -- Probe-Conditioned Cycle Tilt (default OFF).
        # Reads Pack 6 acc_gap/te_gap; computes tiny bounded tilt for hard/easy split.
        # task_cycle_hard_len = baseline split (READ by Pack 3 modified logic).
        # task_cycle_easy_len = derived value updated each cycle (WRITE proof-of-life).
        # Tilt range: [-2, +2]; split clamped to [24, 40] out of 64-tick cycle.
        # Zero drift when OFF. All 7 fields write+read.
        self.enable_task_aftereffect_tilt = False
        self.task_aftereffect_tilt = 0
        self.task_cycle_hard_len = 32
        self.task_cycle_easy_len = 32
        self.task_aftereffect_tilt_updates = 0
        self.task_aftereffect_tilt_pos = 0
        self.task_aftereffect_tilt_neg = 0
        # Day 44 Pack 8 -- Cycle Stability Measurement (default OFF).
        # Observation-only: acc/TE per early/middle/late segment of the cycle.
        # OFF = bit-identical to Pack 7.
        self.enable_task_cycle_stability = False
        self.task_cycle_seg_a_correct = 0
        self.task_cycle_seg_b_correct = 0
        self.task_cycle_seg_c_correct = 0
        self.task_cycle_seg_a_total = 0
        self.task_cycle_seg_b_total = 0
        self.task_cycle_seg_c_total = 0
        self.task_cycle_seg_a_te = 0.0
        self.task_cycle_seg_b_te = 0.0
        self.task_cycle_seg_c_te = 0.0
        self.task_cycle_seg_a_acc = 0.0
        self.task_cycle_seg_b_acc = 0.0
        self.task_cycle_seg_c_acc = 0.0
        self.task_cycle_stability_gap = 0.0
        # Day 45 Pack 1 -- Task-to-Neural Feedback Bridge (default OFF).
        # First bidirectional coupling: task acc -> neural action bias.
        # OFF = bit-identical to Day 44 Pack 8.
        self.enable_task_feedback = False
        self.task_feedback_signal = 0.0
        self.task_feedback_ema = 0.0
        self.K_TASK_FB = 0.003
        self.task_feedback_updates = 0
        self.task_feedback_pos = 0
        self.task_feedback_neg = 0
        # Day 45 Pack 2 -- Boundary-Sensitive Task Feedback (default OFF).
        # Amplified feedback ONLY when winner/runner-up margin < 0.03.
        # K_TASK_FB (0.003) < K_TASK_FB_BOUNDARY (0.012). OFF = bit-identical Pack 1.
        self.enable_task_feedback_boundary = False
        self.task_fb_boundary_events = 0
        self.task_fb_boundary_flips = 0
        self.task_fb_boundary_delta = 0.0
        self.K_TASK_FB_BOUNDARY = 0.025
        # Day 45 Pack 3 -- Task-Graph Exploration Bias (default OFF).
        # Reuses P1 EMA. At states 0/3, biases _tl_explore direction.
        # Positive EMA -> favor stable (suppress explore branch).
        # Negative EMA -> favor exploratory (encourage explore branch).
        # OFF = bit-identical to Pack 2.
        self.enable_task_feedback_branch = False
        self.task_feedback_branch_signal = 0.0
        self.task_feedback_branch_ema = 0.0
        self.K_TASK_FB_BRANCH = 0.008
        self.task_feedback_branch_updates = 0
        self.task_feedback_branch_pos = 0
        self.task_feedback_branch_neg = 0
        self.task_feedback_branch_flips = 0
        self.task_feedback_branch_events = 0
        # Day 46 Pack 1 -- Deterministic Token Encoder (default OFF).
        # Binary hypervectors derived from deterministic hashing of token strings.
        # Same token -> same vector across all runs; never touches global random state.
        # Parallel prediction machinery mirrors state-level system (isolated; no shared dicts).
        # OFF = bit-identical to Day 45 Pack 3.
        self.enable_token_encoder = False
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_hv = {}
        self.token_seq = []
        self.token_seq_vec = None
        self.token_hv_dim = 400
        # Day 52 Pack 5 -- Semantic HVs via Random Indexing.
        # Each token accumulates co-occurrence HV by bundling index HVs of context tokens.
        # Tokens appearing in similar contexts -> similar semantic HVs.
        # enable_semantic_hv=True -> token_hv_for() returns semantic HV instead of random.
        self.enable_semantic_hv   = False  # gate: use co-occurrence HVs
        self.semantic_hv_window   = 3      # context window radius (±window tokens)
        self.semantic_hv_counts   = {}     # {token: list[int]} -- running bit counts for majority vote
        self.semantic_hv_n        = {}     # {token: int} -- total context HVs bundled
        self.semantic_hv          = {}     # {token: bytearray} -- finalized binary semantic HV
        self.semantic_hv_ready    = False  # True after semantic_hv_finalize() called
        # Day 53 Pack 5 -- TF-IDF weighted BoW HV.
        # idf(t) = log(N / df(t)); tokens absent from corpus get idf=log(N).
        self.token_idf            = {}     # {token: float} -- inverse document frequency
        self.token_transition_counts = {}
        self.token_transition_probs = {}
        self.token_transition_error = 0.0
        self.token_prediction_conf = 0.0
        self._tok_pred_last_rebuild = -1
        self.token_pred_rebuild_interval = 50
        # Day 50 Pack 4 -- Trigram Transition Model (always co-built alongside bigram).
        # Key: (tok_a, tok_b) -> {tok_c: prob}. generate() prefers trigram; falls back to bigram.
        # Fixes ':' ambiguity: bigram ':' -> {return,yield,...} but trigram ('items',':') -> {yield:1.0}.
        self.token_transition_counts_3 = {}   # {(a,b): {c: count}}
        self.token_transition_probs_3  = {}   # {(a,b): {c: float}}
        self.token_transition_error_3  = 0.0  # TE using trigram probs (last triple)
        self.token_prediction_conf_3   = 0.0  # trigram max-prob confidence
        # Day 52 Pack 1 -- 4-gram Transition Model (always co-built; gated in generate by enable_fourgram).
        # Key: (tok_a, tok_b, tok_c) -> {tok_d: prob}. Reduces collision weight by adding one more context token.
        self.token_transition_counts_4 = {}   # {(a,b,c): {d: count}}
        self.token_transition_probs_4  = {}   # {(a,b,c): {d: float}}
        self.token_transition_error_4  = 0.0  # TE using 4-gram probs (last 4-tuple)
        self.token_prediction_conf_4   = 0.0  # 4-gram max-prob confidence
        self.enable_fourgram           = False  # gate: generate() tries 4-gram before trigram
        # Day 52 Pack 2 -- Variable-Order Prediction (always uses highest confident order).
        # enable_variorder: at each step pick highest N-gram whose max_prob >= variorder_threshold.
        # Falls back through 4->3->2 then fires VSA guide if still ambiguous.
        self.enable_variorder          = False  # gate: variable-order confidence selection
        self.variorder_threshold       = 0.50   # min confidence to trust an N-gram's pick
        self.gen_order_4g_count        = 0      # how many steps used 4-gram
        self.gen_order_3g_count        = 0      # how many steps used trigram
        self.gen_order_2g_count        = 0      # how many steps used bigram
        # Day 53 Pack 22 -- Kneser-Ney N-gram Smoothing.
        # Replaces raw frequency backoff with continuation-count-based distribution.
        # Tokens appearing in MANY diverse contexts get higher backoff probability
        # than tokens that appear many times in ONE context (e.g. "if" after "A").
        # kn_probs_1: continuation-weighted unigram (replaces raw frequency 1-gram).
        # kn_probs_2: discounted bigram + kn_probs_1 backoff (replaces raw bigram).
        self.enable_kn_smoothing   = False
        self.kn_discount           = 0.75    # absolute discounting constant D
        self.kn_probs_1            = {}      # token -> KN unigram probability
        self.kn_probs_2            = {}      # context_tok -> {next_tok: KN prob}
        self.kn_fallback_count     = 0       # times KN 1-gram was used as final fallback
        self.token_total = 0
        self.token_unique = 0
        self.token_seq_updates = 0
        # Day 46 Pack 2 -- Deterministic Code Token Curriculum (default OFF).
        # Cycles three sequences (seen_0, held_out, seen_1) by phase = token_seq_updates % 3.
        # Held-out sequence reuses same vocabulary but novel last-pair transition (=->1
        # conflicts with seen_1's =->x), producing measurably higher TE than seen.
        # OFF = bit-identical to Pack 1 (token_seq stays fixed probe).
        self.enable_token_curriculum = False
        self.token_curriculum_idx = 0
        self.token_curriculum_phase = 0
        self.token_curriculum_repeat = 0
        self.token_curriculum_total = 0
        self.token_curriculum_seen = 0
        self.token_curriculum_heldout = 0
        self.token_curriculum_correct = 0
        self.token_curriculum_incorrect = 0
        self.token_curriculum_te_seen = 0.0
        self.token_curriculum_te_heldout = 0.0
        # Day 46 Pack 3 -- Token-State Behavioral Coupling (default OFF).
        # Classifies token_seq into symbolic categories (control-flow, math/ops, structural).
        # Signal = 0.5*control_density - 0.25*math_density -> EMA for 1-tick-lagged apply.
        # Apply gate: vsa_sw_timer > 0. K_TOKEN_BEHAVIOR < K_MARGIN. OFF = bit-identical P2.
        self.enable_token_behavior = False
        self.token_behavior_signal = 0.0
        self.token_behavior_ema = 0.0
        self.token_behavior_structural = 0
        self.token_behavior_control = 0
        self.token_behavior_math = 0
        self.K_TOKEN_BEHAVIOR = 0.004
        self.token_behavior_updates = 0
        self.token_behavior_pos = 0
        self.token_behavior_neg = 0
        # Day 46 Pack 4 -- Token-Conditioned Task Feedback (default OFF).
        # Secondary EMA of Pack 3 token_behavior_ema (double-smoothed, 2-tick lag).
        # Extra delta = K_TOKEN_TASK_FB * task_feedback_ema * tt_ema applied to _bg_scores.
        # Gate: enable_task_feedback AND vsa_sw_timer > 0. K < K_TASK_FB_BRANCH. OFF = P3.
        self.enable_token_task_feedback = False
        self.token_task_feedback_signal = 0.0
        self.token_task_feedback_ema = 0.0
        self.K_TOKEN_TASK_FB = 0.003
        self.token_task_feedback_updates = 0
        self.token_task_feedback_pos = 0
        self.token_task_feedback_neg = 0
        # Day 46 Pack 5 -- Token Replay Consolidation (default OFF).
        # Deterministic replay during sleep: priority = held-out first (if heldout avg TE > seen avg TE),
        # else replay high-TE seen_1. Uses Pack 1 token_process_seq; no backprop, no new predictor.
        # Gate: sleeping == True (complement of Packs 1-4 not-sleeping gate).
        self.enable_token_replay = False
        self.token_replay_idx = 0
        self.token_replay_phase = 0
        self.token_replay_mode = 0
        self.token_replay_total = 0
        self.token_replay_seen = 0
        self.token_replay_heldout = 0
        self.token_replay_correct = 0
        self.token_replay_incorrect = 0
        self.token_replay_hard = 0
        self.token_replay_easy = 0
        self.token_replay_te_total = 0.0
        self.token_replay_te_hard = 0.0
        self.token_replay_te_easy = 0.0
        # Day 46 Pack 6 -- Replay Retention Measurement (default OFF).
        # Measures TE and accuracy before/after each sleep cycle to detect consolidation.
        # Pre-window = last window_len waking ticks before sleep (committed at sleep onset).
        # Post-window = first window_len waking ticks after sleep (gated by _rr_had_sleep).
        # Observation-only; zero behavior change. OFF = bit-identical to Pack 5.
        self.enable_token_replay_retention = False
        self.token_replay_pre_ticks = 0
        self.token_replay_post_ticks = 0
        self.token_replay_window_len = 8
        self.token_replay_pre_acc = 0.0
        self.token_replay_post_acc = 0.0
        self.token_replay_pre_te = 0.0
        self.token_replay_post_te = 0.0
        self.token_replay_pre_correct = 0
        self.token_replay_post_correct = 0
        self.token_replay_pre_total = 0
        self.token_replay_post_total = 0
        self.token_replay_acc_gap = 0.0
        self.token_replay_te_gap = 0.0
        # internal: circular pre-buffer + sleep transition state (strictly required)
        self._rr_prev_sleeping = False
        self._rr_wake_count = 0
        self._rr_had_sleep = False
        self._rr_pre_buf = []
        # Day 46 Pack 7 -- Deterministic Code Completion Micro-Benchmark (default OFF).
        # Read-only: uses existing token_transition_probs, no token_process_seq calls.
        # Seen templates: context tokens from curriculum -> P(x|ctx)=1.0 -> TE=0.0.
        # Held-out templates: masked token is novel "y" -> P(y|ctx)=0 -> TE=1.0.
        # OFF = bit-identical to Pack 6.
        self.enable_code_benchmark = False
        self.code_bench_idx = 0
        self.code_bench_phase = 0
        self.code_bench_total = 0
        self.code_bench_seen = 0
        self.code_bench_heldout = 0
        self.code_bench_correct = 0
        self.code_bench_incorrect = 0
        self.code_bench_te_seen = 0.0
        self.code_bench_te_heldout = 0.0
        self.code_bench_acc_seen = 0.0
        self.code_bench_acc_heldout = 0.0
        # internal: per-class correct counts (strictly required to compute per-class accuracy)
        self._cb_correct_seen = 0
        self._cb_correct_heldout = 0
        # Day 46 Pack 8 -- In-Vocabulary Structural Generalization Probe (default OFF).
        # All tokens from trained vocabulary: {if, x, ==, 1, :, return, 0, =, while, +}.
        # Seen: mask at dominant transitions (P=1.0 -> TE=0). No OOV.
        # Held-out: novel structural recombinations, mask at ambiguous positions (TE~0.67).
        # Read-only: queries token_transition_probs only; no token_process_seq calls.
        # OFF = bit-identical to Pack 7.
        self.enable_code_struct_probe = False
        self.code_struct_idx = 0
        self.code_struct_phase = 0
        self.code_struct_total = 0
        self.code_struct_seen = 0
        self.code_struct_heldout = 0
        self.code_struct_correct = 0
        self.code_struct_incorrect = 0
        self.code_struct_te_seen = 0.0
        self.code_struct_te_heldout = 0.0
        self.code_struct_acc_seen = 0.0
        self.code_struct_acc_heldout = 0.0
        # internal: per-class correct counts (strictly required for per-class accuracy)
        self._cs_correct_seen = 0
        self._cs_correct_heldout = 0
        # Day 47 Pack 1 -- Deterministic String-to-Hypervector Encoder (default OFF).
        # Converts arbitrary strings to stable VSA HVs via position-bound chars + bigrams.
        # Connects to string-level transition predictor for TE generation.
        # OFF = bit-identical to Pack 8.
        self.enable_string_encoder = False
        self.string_to_hv = {}
        self.string_seq = []
        self.string_hv_dim = 400
        self.string_total = 0
        self.string_unique = 0
        self.string_seq_updates = 0
        self.string_char_total = 0
        self.string_char_unique = 0
        # internal: char / position HV caches + transition predictor (strictly required)
        self._string_char_hv = {}
        self._string_pos_hv = {}
        self.string_transition_counts = {}
        self.string_transition_probs = {}
        self.string_transition_error = 0.0
        self.string_prediction_conf = 0.0
        self.string_seq_vec = None
        self._str_pred_last_rebuild = 0
        self._str_rebuild_interval = 50
        self._string_prev = None
        # Day 47 Pack 2 -- String Curriculum with Held-Out Generalization (default OFF).
        # 10-phase cycle: S0 S1 S2 S0 S1 S2 S0 S1 S2 H. Seen TE->0, held-out TE~0.67.
        # Reuses Pack 1 string prediction machinery. OFF = bit-identical to Pack 1.
        self.enable_string_curriculum = False
        self.string_curriculum_idx = 0
        self.string_curriculum_phase = 0
        self.string_curriculum_total = 0
        self.string_curriculum_seen = 0
        self.string_curriculum_heldout = 0
        self.string_curriculum_correct = 0
        self.string_curriculum_incorrect = 0
        self.string_curriculum_te_seen = 0.0
        self.string_curriculum_te_heldout = 0.0
        self.string_curriculum_acc_seen = 0.0
        self.string_curriculum_acc_heldout = 0.0
        self._scurr_correct_seen = 0
        self._scurr_correct_heldout = 0
        self._scurr_prev = None
        # Day 47 Pack 3 -- Cross-Modal String<->Token Alignment (default OFF).
        # Measures vsa_similarity(string_encode(s), vsa_bundle(token_hvs)) per waking tick.
        # Gate: enable_string_encoder + enable_token_encoder. OFF = bit-identical to Pack 2.
        self.enable_cross_modal_align = False
        self.cross_modal_sim = 0.0
        self.cross_modal_sim_ema = 0.0
        self.cross_modal_total = 0
        self.cross_modal_sim_sum = 0.0
        self._cma_alpha = 0.1
        # Day 47 Pack 4 -- String-Level Held-Out Replay (Sleep Consolidation) (default OFF).
        # Replays [S2,H] during sleep if heldout TE > seen TE, else [S0,S1].
        # Calls string_process_seq -- same machinery as Pack 2, no new predictor.
        # OFF = bit-identical to Pack 3 (routing/TC unchanged).
        self.enable_string_replay = False
        self.string_replay_idx = 0
        self.string_replay_phase = 0
        self.string_replay_mode = 0
        self.string_replay_total = 0
        self.string_replay_seen = 0
        self.string_replay_heldout = 0
        self.string_replay_correct = 0
        self.string_replay_incorrect = 0
        self.string_replay_hard = 0
        self.string_replay_easy = 0
        self.string_replay_te_total = 0.0
        self.string_replay_te_hard = 0.0
        self.string_replay_te_easy = 0.0
        # Day 47 Pack 5 -- String Replay Retention Measurement (default OFF).
        # Snapshots heldout TE at sleep onset; computes te_gap post-sleep.
        # Fires every tick. OFF = bit-identical to Pack 4.
        self.enable_string_replay_retention = False
        self.string_rr_window_len = 5
        self.string_rr_pre_avg = 0.0
        self.string_rr_post_avg = 0.0
        self.string_rr_te_gap = 0.0
        self._srr_prev_sleeping = False
        self._srr_had_sleep = False
        self._srr_pre_te_snap = 0.0
        self._srr_pre_cnt_snap = 0
        self._srr_post_window_done = False
        # Day 47 Pack 6 -- Cross-Modal TE Coupling (Pearson r) (default OFF).
        # Measures running Pearson r(string_te, token_te) over waking ticks.
        # Gate: not sleeping. OFF = bit-identical to Pack 5.
        self.enable_cross_modal_te_coupling = False
        self.cmt_n = 0
        self.cmt_sum_x = 0.0
        self.cmt_sum_y = 0.0
        self.cmt_sum_xx = 0.0
        self.cmt_sum_yy = 0.0
        self.cmt_sum_xy = 0.0
        self.cmt_pearson_r = 0.0
        # Day 47 Pack 7 -- Multi-Cycle Sleep Accumulation (default OFF).
        # Observes Pack 4 replay stats across multiple sleep cycles (5000t run).
        # Tracks P(H|S2) and replay_correct evolution. OFF = bit-identical to Pack 6.
        self.enable_string_multi_cycle = False
        self.string_multi_cycle_idx = 0
        self.string_multi_cycle_phase = 0
        self.string_multi_cycle_count = 0
        self.string_multi_cycle_ph_total = 0
        self.string_multi_cycle_ph_correct = 0
        self.string_multi_cycle_ph_incorrect = 0
        self.string_multi_cycle_ph_te = 0.0
        self.string_multi_cycle_ph = 0.0
        self.string_multi_cycle_replay_total = 0
        self.string_multi_cycle_replay_correct = 0
        self.string_multi_cycle_replay_incorrect = 0
        self.string_multi_cycle_replay_te = 0.0
        self.string_multi_cycle_ph_start = 0.0
        self.string_multi_cycle_ph_end = 0.0
        self.string_multi_cycle_ph_trend = 0.0
        self._smc_prev_sleeping = False
        self._smc_ph_start_set = False
        self._smc_replay_base_total = 0
        self._smc_replay_base_correct = 0
        self._smc_replay_base_te = 0.0
        # Day 47 Pack 8 -- String Vocabulary Expansion (default OFF).
        # Tier-2 curriculum (3 novel strings, same char vocab) after tier-1 convergence.
        # Measures HV structural similarity and whether transfer > cold-start.
        # OFF = bit-identical to Pack 7.
        self.enable_string_vocab_expand = False
        self.string_vocab_tier = 0
        self.string_vocab_tier_switch = 60
        self.string_vocab_tier2_idx = 0
        self.string_vocab_tier2_total = 0
        self.string_vocab_tier2_seen = 0
        self.string_vocab_tier2_heldout = 0
        self.string_vocab_tier2_correct = 0
        self.string_vocab_tier2_incorrect = 0
        self.string_vocab_tier2_te_sum = 0.0
        self.string_vocab_tier2_te_avg = 0.0
        self.string_vocab_tier2_acc = 0.0
        self.string_vocab_hv_sim_t0 = 0.0
        self.string_vocab_hv_sim_t1 = 0.0
        self.string_vocab_transfer_ratio = 0.0
        self._svx_prev = None
        # Day 47 Pack 9 -- Tier-2 String Transfer Probe (default OFF).
        # Interleaves tier-1 and tier-2 strings via existing Pack 1 prediction machinery.
        # Measures whether cross-tier context helps predict tier-2 targets.
        # OFF = bit-identical to Pack 8.
        self.enable_string_tier2 = False
        self.string_tier2_idx = 0
        self.string_tier2_phase = 0
        self.string_tier2_mode = 0
        self.string_tier2_total = 0
        self.string_tier2_seen = 0
        self.string_tier2_heldout = 0
        self.string_tier2_correct = 0
        self.string_tier2_incorrect = 0
        self.string_tier2_te_seen = 0.0
        self.string_tier2_te_heldout = 0.0
        self.string_tier2_transfer_ratio = 0.0
        self.string_tier2_cold_start_te = 0.0
        self.string_tier2_warm_start_te = 0.0
        self.string_tier2_only = False
        self._st2_prev = None
        # Day 47 Pack 10 -- Character-Level Transition Predictor (default OFF).
        # Char bigram predictor generalizes across string boundaries via shared chars.
        # Expected: warm_te < cold_te (positive transfer) because tier-1 and tier-2 share bigrams.
        # char_tier2_only=True for cold-start run (no tier-1 warmup).
        # OFF = bit-identical to Pack 9.
        self.enable_char_predictor = False
        self.char_transition_counts = {}
        self.char_transition_probs = {}
        self.char_te = 0.0
        self.char_te_sum_seen = 0.0
        self.char_te_sum_heldout = 0.0
        self.char_seen_total = 0
        self.char_heldout_total = 0
        self.char_correct = 0
        self.char_incorrect = 0
        self.char_transfer_ratio = 0.0
        self.char_tier2_only = False
        self._char_rebuild_interval = 50
        self._char_pred_last_rebuild = 0
        self._char_seq_updates = 0
        # Day 47 Pack 11 -- Syntax-Level VSA Binding Predictor (default OFF).
        # Learns global binding op = majority(prev_hv XOR next_hv) from all transitions.
        # Predicts next_hv = bind(curr_hv, op). Measures sim(predicted, actual).
        # Expected: sim > 0.5 because tier swap (if<->while) is consistent across pairs.
        # OFF = bit-identical to Pack 10.
        self.enable_vsa_predictor = False
        self.vsa_pred_sim = 0.0
        self.vsa_pred_sim_ema = 0.0
        self.vsa_pred_sim_sum = 0.0
        self.vsa_pred_total = 0
        self.vsa_pred_seen_total = 0
        self.vsa_pred_heldout_total = 0
        self.vsa_pred_seen_sim_sum = 0.0
        self.vsa_pred_heldout_sim_sum = 0.0
        self.vsa_binding_op = None
        self._vsp_op_counts = None
        self._vsp_op_total = 0
        self._vsp_rebuild_interval = 10
        self._vsp_last_rebuild = 0
        self._vsp_prev_hv = None
        self._vsp_alpha = 0.05
        self._vsp_step = 0
        # Day 48 Pack 1 -- Nearest-Neighbor Associative Retrieval (default OFF).
        # Novel context HV retrieves nearest stored context; inherits its per-context
        # transition distribution. nn_te_inherited < nn_te_blind proves structural generalization.
        # OFF = bit-identical to Pack 11.
        self.enable_nn_retrieval = False
        self.nn_sim_threshold = 0.5
        self.nn_registry = {}       # bytes(hv) -> {prev_tok: {next_tok: count}}
        self.nn_hv_store = []       # list of bytearray HVs for sim search
        self.nn_hv_keys = []        # bytes keys parallel to nn_hv_store
        self.nn_query_count = 0
        self.nn_hit_count = 0
        self.nn_miss_count = 0
        self.nn_last_sim = 0.0
        self.nn_sim_sum = 0.0
        self.nn_te_blind = 0.0
        self.nn_te_inherited = 0.0
        self.nn_te_blind_sum = 0.0
        self.nn_te_inherited_sum = 0.0
        self.nn_te_count = 0
        self.nn_te_ratio = 0.0
        self.nn_te_ratio_ema = 0.0
        # Day 48 Pack 2 -- Hierarchical Context HV (default OFF).
        # Encodes contexts at pair-level (bigrams) AND phrase-level (trigrams) simultaneously.
        # Both levels must independently retrieve the correct nearest neighbor.
        # Proves VSA geometry holds across representational hierarchy.
        # OFF = bit-identical to Pack 1.
        self.enable_nn_hierarchy = False
        self.nn_phrase_registry = {}    # bytes(phrase_hv) -> {prev_tok: {next_tok: count}}
        self.nn_phrase_hv_store = []    # trigram-level HV store
        self.nn_phrase_hv_keys = []
        self.nn_hier_step = 0           # curriculum step (mod 6)
        self.nn_hier_query_count = 0
        self.nn_hier_coherent_count = 0  # both levels agree on nearest neighbor
        self.nn_hier_coherence = 0.0
        self.nn_pair_te_h_sum = 0.0     # TE via pair-level retrieval
        self.nn_phrase_te_h_sum = 0.0   # TE via phrase-level retrieval
        self.nn_hier_te_count = 0
        self.nn_combined_correct = 0    # combined sim picks correct context
        self.nn_hier_last_pair_sim = 0.0
        self.nn_hier_last_phrase_sim = 0.0
        # Day 48 Pack 3 -- Compositional Role-Content Binding (default OFF).
        # Minimal symbolic units: each token bound to a semantic role HV via XOR.
        # Bundle units into compositional HV. Test partial retrieval, role recovery,
        # and VSA collapse under nested symbolic composition.
        # OFF = bit-identical to Pack 2.
        self.enable_nn_composition = False
        self.nn_role_hv = {}             # {role_name: bytearray} -- lazy-init deterministic
        self.nn_comp_registry = {}       # bytes(full_hv) -> label str
        self.nn_comp_hv_store = []       # full compositional HVs for retrieval
        self.nn_comp_hv_keys = []
        self.nn_comp_phrase_registry = {} # return-phrase HVs (2-unit sub-structure)
        self.nn_comp_phrase_hv_store = []
        self.nn_comp_phrase_hv_keys = []
        self.nn_comp_step = 0
        self.nn_comp_query_count = 0
        self.nn_comp_full_correct = 0    # full-HV retrieval correct
        self.nn_comp_partial_correct = 0 # partial-HV (no VARIABLE) retrieval correct
        self.nn_comp_phrase_correct = 0  # phrase-HV (RET_TYPE+RET_VAL) retrieval correct
        self.nn_comp_full_sim_sum = 0.0
        self.nn_comp_partial_sim_sum = 0.0
        self.nn_comp_phrase_sim_sum = 0.0
        self.nn_comp_recovery_sim_sum = 0.0  # role recovery: XOR(bundle,role) ~ content
        self.nn_comp_leakage_sim_sum = 0.0   # leakage: should be ~0.5
        self.nn_comp_recovery_count = 0
        # Day 48 Pack 4 -- Generative Probe: Retrieve-then-Decode (default OFF).
        # Partial query (RET_VAL unit missing) -> retrieve nearest complete context
        # -> decode missing unit via XOR(retrieved_bundle, ROLE_RET_VAL).
        # Measures whether decoded token HV identifies correct RET_VAL above chance.
        # OFF = bit-identical to Pack 3.
        self.enable_nn_generate = False
        self.nn_gen_query_count = 0
        self.nn_gen_correct = 0        # decoded token sim to correct > sim to all wrong
        self.nn_gen_sim_correct_sum = 0.0   # sim(decoded, correct_hv)
        self.nn_gen_sim_wrong_sum = 0.0     # sim(decoded, best_wrong_hv)
        self.nn_gen_margin_sum = 0.0        # correct - best_wrong
        self.nn_gen_step = 0
        # Day 48 Pack 5 -- Out-of-Vocab Generalization & Graceful Degradation (default OFF).
        # 5-level novelty ladder: inject 0..5 novel units per query.
        # Measures: retrieval sim curve, margin vs correctness correlation (EEIL viability),
        # abstraction-layer stability, entropy, catastrophic vs graceful failure.
        # OFF = bit-identical to Pack 4.
        self.enable_nn_degrade = False
        self.nn_deg_step = 0
        # Per-level counters (index 0-4 = novelty level 0-4)
        self.nn_deg_query   = [0]*5
        self.nn_deg_correct = [0]*5
        self.nn_deg_sim_sum    = [0.0]*5
        self.nn_deg_margin_sum = [0.0]*5
        self.nn_deg_entropy_sum= [0.0]*5
        # Phrase-level stability per level
        self.nn_deg_phrase_query   = [0]*5
        self.nn_deg_phrase_correct = [0]*5
        # Correlation lists for Pearson r(margin, correctness)
        self.nn_deg_margin_list  = []   # margin per query (all levels)
        self.nn_deg_correct_list = []   # 1.0/0.0 correctness per query
        self.nn_deg_level_list   = []   # novelty level per query
        # Day 48 Pack 6 -- Sequential Retrieval Stability (default OFF).
        # Autoregressive chain with confidence-gated phrase-layer stabilizer.
        # Context = accumulated XOR-pair bundle (grows with each emitted token).
        # Emit predicted token (not GT) to measure realistic novelty accumulation.
        # Compares L2-novel chains WITH vs WITHOUT phrase stabilizer.
        # OFF = bit-identical to Pack 5.
        self.enable_nn_autoregress = False
        self.nn_ar_step = 0
        self.nn_ar_margin_threshold = 0.04  # margin below this -> phrase-layer fallback
        self.nn_ar_chain_count = 0
        self.nn_ar_total_steps = 0
        self.nn_ar_correct_steps = 0
        self.nn_ar_phrase_fired = 0
        self.nn_ar_phrase_helped = 0
        # Level indices: 0=seen, 1=L0(1novel), 2=L1(2novel), 3=L2+stab, 4=L2-stab
        self.nn_ar_level_chains  = [0]*5
        self.nn_ar_level_correct = [0]*5
        self.nn_ar_level_steps   = [0]*5
        self.nn_ar_level_phrase  = [0]*5
        # Series across all chains for Pearson r(margin, correctness)
        self.nn_ar_sim_list     = []
        self.nn_ar_margin_list  = []
        self.nn_ar_correct_list = []
        self.nn_ar_entropy_list = []
        # Day 48 Pack 7 -- Retrieval-Layer Uncertainty (default OFF).
        # Measures Pearson r(retrieval_sim, chain_correctness).
        # Retrieval sim is the correct EEIL signal for sparse-data autoregression.
        # Attractor curve: per-level, per-step-index sim average shows context growth.
        # Confidence gate: sim < threshold -> uncertain; tests uncertain_rate < confident_rate.
        # OFF = bit-identical to Pack 6.
        self.enable_nn_ret_uncertainty = False
        self.nn_ru_step = 0
        self.nn_ru_sim_threshold = 0.65
        # Per-level, per-step-index accumulators (5 levels x 6 steps)
        self.nn_ru_level_step_sim     = [[0.0] * 6 for _ in range(5)]
        self.nn_ru_level_step_count   = [[0]   * 6 for _ in range(5)]
        self.nn_ru_level_step_correct = [[0]   * 6 for _ in range(5)]
        # Confidence gate counters
        self.nn_ru_confident_correct = 0
        self.nn_ru_confident_total   = 0
        self.nn_ru_uncertain_correct = 0
        self.nn_ru_uncertain_total   = 0
        # Per-step series for Pearson r(sim, correctness)
        self.nn_ru_sim_list      = []
        self.nn_ru_correct_list  = []
        self.nn_ru_level_list    = []
        self.nn_ru_step_idx_list = []
        # Day 49 Pack 1 -- Vocabulary Coverage Gate (default OFF).
        # Direct-hit gate: abstain when prev_tok not found directly in retrieved
        # context entry (fallback decoding = uncovered transition = uncertain).
        # Oracle continuation after abstention (GT token advances chain).
        # Compares convergent evidence with Pack 7 sim-gate abstention set.
        # OFF = bit-identical to Pack 7.
        self.enable_nn_vocab_gate = False
        self.nn_vg_step = 0
        self.nn_vg_vocab = set()         # all registered next-tokens
        # Per-level (0=seen, 1=L0, 2=L1, 3=L2+stab, 4=L2-stab)
        self.nn_vg_level_emit    = [0]*5  # steps emitted (direct hit)
        self.nn_vg_level_abstain = [0]*5  # steps abstained (fallback / uncovered)
        self.nn_vg_level_correct = [0]*5  # correct among emitted
        self.nn_vg_level_would   = [0]*5  # would-be-correct among abstained
        self.nn_vg_total_emit    = 0
        self.nn_vg_total_abstain = 0
        self.nn_vg_total_correct = 0
        # Day 49 Pack 2 -- Cross-Level Structural Transfer (default OFF).
        # Train on "if" family only (SEEN_A, SEEN_B). Generate for "while" family.
        # Surface: bigram encode_ctx (live chain). Role: analytical comparison in unit tests.
        # OFF = bit-identical to Pack 1.
        self.enable_nn_xfer = False
        self.nn_xfer_step = 0
        # 4 levels: 0=baseline(if,x), 1=XL0(while,x), 2=XL1(while,y), 3=XL2(while,z)
        self.nn_xfer_level_chains  = [0]*4
        self.nn_xfer_level_steps   = [0]*4
        self.nn_xfer_level_correct = [0]*4
        self.nn_xfer_sim_list      = []
        self.nn_xfer_correct_list  = []
        self.nn_xfer_level_list    = []
        # Per-level, per-step-index sim for attractor curve (4 levels x 6 steps)
        self.nn_xfer_step_sim   = [[0.0] * 6 for _ in range(4)]
        self.nn_xfer_step_count = [[0]   * 6 for _ in range(4)]
        # Day 49 Pack 3 -- Holographic Interference Phase Transition (default OFF).
        # Direct superposition model: M = vsa_bundle(p1,...,pN).
        # Measures sim(pi, M) degradation as N grows. Identifies graceful vs sharp transition.
        # Two cohorts: role-bound (nn_compose_unit) vs random. Noise robustness at 5% flip.
        self.enable_nn_holo   = False
        self.nn_holo_dim      = 400        # must match vsa_dim
        self.nn_holo_memory   = None       # bytearray(400): current superposition
        self.nn_holo_count    = 0
        self.nn_holo_patterns = {}         # name -> bytearray(400)
        self.nn_holo_cap_role = {}         # N -> (mean_dsim, corr0, corr5, mean_margin)
        self.nn_holo_cap_rand = {}         # N -> same
        self.nn_holo_phase_role = 0        # N where mean_dsim first < 0.65 (role-bound)
        self.nn_holo_phase_rand = 0        # N where mean_dsim first < 0.65 (random)
        # Day 49 Pack 4 -- Flat Majority Holographic Storage (default OFF).
        # Stores all patterns; rebuilds bundle as vsa_bundle(ALL) after each addition.
        # Avoids cascaded AND collapse. Gives gradual degradation vs sharp cliff.
        self.enable_nn_holo_flat = False
        self.nn_holo_flat_vecs   = []    # all stored bytearray(400) in order
        self.nn_holo_flat_names  = []    # parallel names
        self.nn_holo_flat_memory = None  # bytearray: flat majority of all stored
        # Day 49 Pack 5 -- Attractor-Assisted Holographic Cleanup (default OFF).
        # Uses nn_holo_flat memory. Iteratively blends probe toward recalled attractor.
        # Measures basin radius and convergence speed under heavy noise.
        self.enable_nn_holo_attract = False
        self.nn_holo_attract_max_iters = 5
        # Day 49 Pack 8 -- Position-Sensitive Bigram Binding (default OFF).
        # Fixes XOR commutativity bug: vsa_bind(a,b)==vsa_bind(b,a).
        # token_bigram_pos_hv(a,b) = XOR(hv_a, rotate(hv_b, pos_shift)).
        # Guarantees (a,b) != (b,a) for distinct tokens. Critical for code ordering.
        self.enable_nn_holo_pos = False
        self.nn_holo_pos_shift  = 100    # bit rotation for pos-1 token (400/4 = quarter cycle)
        # Day 49 Pack 9 -- Holographic Working Memory (default OFF).
        # Sliding-window flat majority over recent token sequence HVs.
        # Novelty = 1 - dsim(current_seq, WM_before_add). Routes to explore pressure.
        self.enable_nn_holo_wm   = False
        self.nn_holo_wm_size     = 8     # max sequences in WM (sliding window)
        self.nn_holo_wm_vecs     = []    # bytearray list, oldest first
        self.nn_holo_wm_names    = []    # parallel name list
        self.nn_holo_wm_memory   = None  # flat majority of current WM window
        self.nn_holo_wm_novelty  = 0.0   # 1 - dsim(latest, WM_before): high = novel
        self.nn_holo_wm_familiar = 0.0   # dsim(latest, WM_before): high = familiar
        self.nn_holo_wm_count    = 0     # total sequences processed
        self.K_WM_NOVELTY        = 0.003 # novelty -> explore pressure scale (< K_TOKEN_BEHAVIOR)
        self.nn_holo_wm_explore_bias = 0.0  # K_WM_NOVELTY * novelty, applied to explore score
        # Day 49 Pack 11 -- Holographic WM Replay during Sleep (default OFF).
        # Extends Pack 10: stores novelty + token seq alongside HV for sleep replay.
        # During sleep: replays highest-novelty sequence via token_process_seq each tick.
        self.enable_nn_holo_wm_replay  = False
        self.nn_holo_wm_novelties      = []   # novelty at store time (parallel to wm_vecs/names)
        self.nn_holo_wm_seqs           = []   # token seq at store time (parallel)
        self.nn_holo_wm_replay_count   = 0    # total sleep replay ticks
        self.nn_holo_wm_replay_te_sum  = 0.0
        self.nn_holo_wm_replay_te_n    = 0
        self.nn_holo_wm_replay_avg_te  = 0.0  # mean TE during replay (consolidation signal)
        self.nn_holo_wm_waking_te_sum  = 0.0  # waking TE baseline (for comparison)
        self.nn_holo_wm_waking_te_n    = 0
        self.nn_holo_wm_waking_avg_te  = 0.0
        # Day 49 Pack 12 -- TE-Guided Replay Priority (default OFF).
        # Extends Pack 11: stores token_transition_error at encode time.
        # Priority = novelty * TE: targets structurally novel AND temporally hard sequences.
        # Orthogonal to novelty-only (Pack 11): selects different replay target.
        self.enable_nn_holo_wm_te_priority     = False
        self.nn_holo_wm_tes                    = []   # TE at store time (parallel to wm_vecs)
        self.nn_holo_wm_priority_replay_count  = 0
        self.nn_holo_wm_priority_replay_te_sum = 0.0
        self.nn_holo_wm_priority_replay_te_n   = 0
        self.nn_holo_wm_priority_replay_avg_te = 0.0
        self.nn_holo_wm_priority_top_te        = 0.0  # TE of last selected seq
        self.nn_holo_wm_priority_top_nov       = 0.0  # novelty of last selected seq
        self.nn_holo_wm_priority_top_prio      = 0.0  # novelty * TE of last selected seq
        # Day 52 Pack 7 -- Semantic BoW WM (parallel to position-sensitive WM).
        self.nn_holo_wm_bow_vecs               = []   # BoW semantic HVs (parallel to wm_vecs)
        self.nn_holo_wm_bow_sim                = 0.0  # sim from last nn_holo_wm_nearest_bow call
        # Day 52 Pack 9 -- Domain-tagged WM.
        self.nn_holo_wm_domains                = []   # domain tag per WM slot ("code","language","")
        # Day 52 Pack 12 -- Question-part PoS HVs (tokens before "A" delimiter only).
        # Eliminates answer-token noise from PoS routing: short query vs long stored sequence.
        self.nn_holo_wm_qpos_vecs              = []   # question-part PoS HVs (parallel to wm_vecs)
        # Day 52 Pack 10 -- Reservoir + RLS output layer.
        # VSA binary HVs as fixed random reservoir; online RLS learns linear output weights.
        # No backprop. No GPU. Learns from 5 examples. Never forgets via sleep replay.
        self.enable_reservoir    = False
        self.rls_alpha           = 0.50    # EMA update rate for reservoir state (0.5 = balanced history)
        self.rls_lambda          = 0.98    # RLS forgetting factor
        self.rls_delta           = 1.0     # initial P diagonal (prior precision)
        self.rls_reservoir       = None    # np.float64 (dim,) -- current reservoir state
        self.rls_P               = None    # np.float64 (dim,dim) -- RLS precision matrix
        self.rls_W               = None    # np.float64 (dim,vocab) -- output weight matrix
        self.rls_vocab           = {}      # {token: int}
        self.rls_vocab_inv       = {}      # {int: token}
        self.rls_train_count     = 0       # total RLS update steps
        self.rls_last_loss       = 0.0     # cross-entropy loss on last training step
        self.rls_train_loss_sum  = 0.0
        # Day 52 Pack 14 -- Multi-turn conversation memory.
        # chat_reservoir persists across chat() calls instead of resetting per turn.
        # Steps 3/4 of rls_hybrid_generate start from chat_reservoir (not zero).
        self.enable_multiturn    = False
        self.chat_reservoir      = None    # np.float64 (dim,) -- persistent conv state
        self.chat_turn_count     = 0       # number of completed chat() turns
        # Day 52 Pack 15 -- Episodic WM: store each chat() exchange as WM entry.
        # Ikigai learns from its own outputs mid-conversation without sleep replay.
        self.enable_episodic_wm  = False
        self.episodic_wm_count   = 0       # number of exchanges stored in WM
        # Day 52 Pack 16 -- Online RLS fine-tuning from conversation.
        # After each chat() exchange, train RLS on prompt+response (no sleep needed).
        # Permanently integrates novel patterns mid-conversation. Never forgets.
        self.enable_online_rls   = False
        self.online_rls_count    = 0       # total online RLS training steps
        # Day 52 Pack 21 -- Operation Pattern Table (OPT).
        # Learns "Q [op] [...] A [resp]" -> op_token: first_resp_token mapping.
        # Enables argument-independent generalization: after 5 shots of "scale X -> 2",
        # ANY novel arg "scale zeta" also generates "2" as first token.
        self.enable_op_pattern   = False
        self.op_pattern_table    = {}      # op_token -> first_response_token
        self.op_pattern_count    = {}      # op_token -> evidence count
        # Day 53 Pack 22 -- VSA Analogy Completion.
        # Generative inference via HV arithmetic: a:b :: c:? = hv[a] XOR hv[b] XOR hv[c].
        # Enables answering questions never seen in training by structural analogy.
        # Also integrates into rls_hybrid_generate as Step 2.5 fallback.
        self.enable_vsa_analogy      = False
        self.vsa_analogy_count       = 0      # times analogy attempted
        self.vsa_analogy_hits        = 0      # times sim >= threshold
        self.vsa_analogy_threshold   = 0.58   # min similarity to accept result
        self.vsa_analogy_last_sim    = 0.0    # sim from last call
        self.vsa_analogy_last_result = None   # token from last call
        # Day 50 Pack 1 -- Generation Engine (default OFF).
        # generate(prompt_tokens, max_len, temperature) samples from token_transition_probs.
        # Greedy (temp=0) or temperature-scaled sampling (temp>0).
        # Returns full sequence (prompt + generated). No exec() injection needed.
        self.enable_generation  = False
        self.gen_last_prompt    = []
        self.gen_last_output    = []
        self.gen_last_len       = 0
        # Day 50 Pack 2 -- Scale Memory (default: standard dims; call scale_for_benchmark() to upgrade).
        # vsa_dim=1000 raises holographic capacity from N*~56 to N*~140.
        # nn_holo_wm_size=256 retains all 20 CCGB patterns simultaneously (vs 8-slot eviction).
        # token_hv_dim must match vsa_dim for consistent HV arithmetic.
        self.scaled_for_benchmark = False
        # Day 50 Pack 5 -- Continual Learning Harness (CCGB infrastructure).
        # run_ccgb() presents patterns sequentially with sleep-cycle consolidation.
        # No forgetting: token_transition_counts is cumulative, never evicts transitions.
        self.ccgb_results       = {}   # {pattern_name: accuracy} after full run
        self.ccgb_pattern_count = 0    # patterns learned so far
        # Day 51 Pack 1 -- VSA-Guided Generation (default OFF).
        # When ambiguous (max continuation prob < threshold), encodes current tokens as HV,
        # retrieves nearest WM template by per-entry sim, blends template-local distribution
        # at K_VSA_GUIDE weight. Fixes shared-context interference without retraining.
        self.enable_vsa_guided_gen  = False
        self.K_VSA_GUIDE            = 0.9    # template weight in blend (1-K = global weight)
        self.vsa_guide_threshold    = 0.51   # max_prob below this = ambiguous -> guide fires
        self.vsa_guide_count        = 0      # total ambiguous steps encountered
        self.vsa_guide_hits         = 0      # steps where template found and blending applied
        # Day 43 Pack 12 -- Regime Trend Detection (default OFF).
        # Observation-only: short-horizon EMA trend per regime + gap. No behavior change.
        # OFF = bit-identical to Pack 11.
        self.enable_env_regime_trend = False
        self.env_regime0_trend_ema = 0.0
        self.env_regime1_trend_ema = 0.0
        self.env_regime0_trend = 0.0
        self.env_regime1_trend = 0.0
        self.env_regime_trend_gap = 0.0
        # Day 43 Pack 11 -- Regime Dwell-Time Guard (default OFF).
        # Tiny trigger adjustment based on dwell time. K_REGIME_DWELL < K_REGIME_PREF.
        # Better regime: trigger += push; worse regime: trigger -= push. OFF = bit-identical to Pack 10.
        self.enable_env_regime_dwell = False
        self.K_REGIME_DWELL = 0.15
        self.env_regime_dwell = 0
        self.env_regime_dwell_ema = 0.0
        self.env_regime_dwell_adjust = 0.0
        # Day 43 Pack 10 -- Regime-Conditioned Outcome Bias (default OFF).
        # Tiny scaling factor on Pack 4 outcome bias: 1.0 (better) or <1.0 (worse).
        # K_REGIME_BIAS < K_OUTCOME_BIAS. OFF = bit-identical to Pack 9.
        self.enable_env_regime_bias = False
        self.K_REGIME_BIAS = 0.002
        self.env_regime_bias = 1.0
        # Day 43 Pack 9 -- Regime Preference Nudge (default OFF).
        # Tiny threshold adjustment toward better regime. K_REGIME_PREF < 50.
        # OFF = bit-identical to Pack 8.
        self.enable_env_regime_pref = False
        self.K_REGIME_PREF = 0.5
        self.env_regime_pref = 0.0
        # Day 43 Pack 8 -- Regime Outcome Profiling (default OFF).
        # Observation-only: tracks per-regime outcome mean and gap.
        # No behavior change; OFF = bit-identical to Pack 7.
        self.enable_env_regime_profile = False
        self.env_regime0_count = 0
        self.env_regime1_count = 0
        self.env_regime0_outcome_sum = 0.0
        self.env_regime1_outcome_sum = 0.0
        self.env_regime0_outcome_mean = 0.0
        self.env_regime1_outcome_mean = 0.0
        self.env_regime_outcome_gap = 0.0
        self.env_states = list(range(15))
        self.env_current = 0
        self.env_transitions = {
            0:  [1],
            1:  [2],
            2:  [3],
            3:  [4],
            4:  [5, 6, 7],
            5:  [8],
            6:  [8],
            7:  [8],
            8:  [9],
            9:  [10, 11, 12],
            10: [13],
            11: [13],
            12: [13],
            13: [14],
            14: [4],
        }
        # Day 43 Pack 7 -- Regime 1: reversed branch ordering at nodes 4 and 9.
        # Same 15 states, same determinism; different predictable/unpredictable pockets.
        self.env_transitions_r1 = {
            0:  [1],
            1:  [2],
            2:  [3],
            3:  [4],
            4:  [7, 6, 5],    # reversed vs Regime 0 [5, 6, 7]
            5:  [8],
            6:  [8],
            7:  [8],
            8:  [9],
            9:  [12, 10, 11],  # shifted vs Regime 0 [10, 11, 12]
            10: [13],
            11: [13],
            12: [13],
            13: [14],
            14: [4],
        }
        self.env_visit_counts = {s: 0 for s in self.env_states}
        self.env_transition_counts = {}
        self.env_visited = [0]
        self.env_step_count = 0
        self.env_last_state = None
        self.env_last_action = None
        self.env_last_branch_index = 0
        # Day 38 Pack 1 -- Transition Prediction (Proto-Expectation Layer, always active)
        # Pure inference over transition_counts. No behavior influence, no smoothing,
        # no decay, no learning feedback. Recomputed periodically.
        # Structure: {prev_state_key -> {next_state_key -> probability}}
        self.transition_probs        = {}
        self._pred_last_rebuild_tick = -1
        self.pred_rebuild_interval   = 100
        # Day 38 Pack 2 -- Transition Prediction Error (signal only, always active)
        # error = 1 - P(actual_next | prev). 1.0 if prev not yet in transition_probs.
        # No smoothing, no thresholding, no behavior influence.
        self.transition_error      = 0.0
        self.predicted_next_state  = None
        self.actual_next_state     = None
        self._te_sum   = 0.0
        self._te_sumsq = 0.0
        self._te_count = 0
        # Day 38 Pack 3 -- Error-Gated Modulation (bounded score nudge, default OFF)
        # K_PE=0.10 strictly < novelty K_N=0.20 (constraint: not a primary driver).
        # No routing/TC/ACE impact; only score_explore += and score_approach -= inside BG.
        self.enable_prediction_modulation = False
        # Day 38 Pack 4 -- Prediction Confidence (signal only, always active)
        # confidence = max P(next | current_state). 0.0 if current state unknown.
        # Independent of error: high conf + high error = "sure and wrong" (real surprise).
        # No smoothing, no thresholding, no behavior influence.
        self.prediction_confidence = 0.0
        self._pc_sum   = 0.0
        self._pc_sumsq = 0.0
        self._pc_count = 0
        # Day 38 Pack 5 -- Confidence-Aware Surprise (selective refinement, default OFF)
        # real_surprise = TE * PC. Boost score_explore only when sure-and-wrong.
        # K_CS=0.01 (= 0.5 * K_PE). Sub-PE refinement; same 3 gates as Pack 3.
        self.enable_confidence_surprise = False
        # Day 39 Pack 1 -- State Abstraction (always active, observation only)
        # Online clustering of coarse states into <=16 reusable concepts.
        # Periodic update; no behavior coupling.
        self.state_embeddings              = {}    # state_key -> feature tuple
        self.state_clusters                = {}    # state_key -> cluster_id
        self.cluster_stats                 = {}    # cluster_id -> dict
        self._state_runtime                = {}    # state_key -> {te_sum, te_count}
        self._cluster_last_update_tick     = -1
        self.cluster_update_interval       = 100
        self.max_clusters                  = 16
        self.cluster_distance_threshold    = 3.0
        # Day 39 Pack 2 -- Concept-Conditioned Modulation (default OFF, weak context)
        # K_CTX_CL=0.005 strictly < K_CS=0.01 < K_PE=0.02 < K_N=0.20.
        # Cluster context guides; never drives.
        self.enable_concept_modulation     = False
        # Day 39 Pack 3 -- Concept Transition Learning (always active, observation only)
        # Records (cid_prev -> cid_curr) frequencies; periodic rebuild to probs.
        # Bounded by max_clusters^2 (256). No behavior coupling.
        self.cluster_transition_counts    = {}    # (cid_prev, cid_curr) -> int count
        self.cluster_transition_probs     = {}    # cid_prev -> {cid_curr: prob}
        self._cluster_prev_id             = None
        self._cluster_last_rebuild_tick   = -1
        self.cluster_rebuild_interval     = 100
        # Day 39 Pack 4 -- Concept-Level Error & Confidence (signal only, always active)
        # TE_c = 1 - P(cid_actual | cid_prev). PC_c = max P(cid_next | cid_prev).
        # Unknown prev -> TE_c = 1.0, PC_c = 0.0. No behavior influence.
        self.cluster_transition_error = 0.0
        self.cluster_prediction_conf  = 0.0
        self._ce_sum   = 0.0
        self._ce_sumsq = 0.0
        self._ce_count = 0
        self._cpc_sum   = 0.0
        self._cpc_sumsq = 0.0
        self._cpc_count = 0
        # Day 39 Pack 5 -- Cross-Level Conflict (meta-consistency, signal only)
        # conflict = |state_TE*state_PC - concept_TE*concept_PC|. Signed variant retains direction.
        # No behavior influence; pure observation of disagreement between micro and macro models.
        self.cross_level_conflict = 0.0
        self.cross_level_signed   = 0.0
        self._cc_sum   = 0.0
        self._cc_sumsq = 0.0
        self._cc_count = 0
        # Day 39 Pack 6 -- Conflict-Gated Attention (default OFF, smallest K in stack)
        # Gated: signed<0 AND conflict>=0.6 AND cooldown elapsed.
        # K_CF=0.002 strictly < K_CTX_CL=0.005 < K_CS=0.01 < K_PE=0.02 < K_N=0.20.
        self.enable_conflict_attention = False
        self.conflict_cooldown_ticks   = 5
        self._conflict_last_tick       = -1
        # Day 39 Pack 7 -- Attention Persistence (default OFF)
        # Conflict event triggers temporal focus window (5 ticks, exponential decay).
        # K_AP=0.001 strictly < K_CF=0.002. Activated by Pack 6 fire only.
        self.enable_attention_persistence = False
        self.attention_timer              = 0
        self.attention_strength           = 0.0
        # Day 39 Pack 8 -- Safe-Probing under Conflict (default OFF, energy-gated)
        # K_SP=0.0015. Active only when attention window open AND energy safe margin.
        # Investigates anomalies without breaking energy stability.
        self.enable_safe_probe = False
        # Day 39 Pack 9 -- Targeted Exploration (default OFF, signal-gated)
        # K_TE=0.0012 < K_SP=0.0015. Probes only where model is meaningfully wrong:
        # max(state_TE * state_PC, concept_TE * concept_PC) > 0.3.
        self.enable_targeted_explore = False
        # Day 39 Pack 10 -- Selective Override (default OFF, ultra-strict gate)
        # K_OV=0.0008 < K_TE=0.0012. Active only when target_signal > 0.7 AND
        # attention window open AND energy safe. Reduces approach/wait dominance
        # rather than boosting explore directly.
        self.enable_selective_override = False
        # Day 39 Pack 11 -- Micro-Strategy Persistence (default OFF, smallest K)
        # K_MS=0.0006 < K_OV=0.0008. Brief commit (3 ticks) to last action when
        # attention high + target_signal > 0.6. No long-term lock; max 3 ticks.
        self.enable_strategy_persistence = False
        self.strategy_timer              = 0
        self.strategy_action             = None
        # Day 40 Pack 1 -- VSA Foundation (default OFF, storage-only)
        # Vector Symbolic Architecture: binary hypervectors with XOR binding,
        # majority bundling, hamming similarity. Item memory is fixed (no growth).
        # NOT connected to BG / routing / scoring / energy. Pure representation layer.
        self.enable_vsa  = False
        self.vsa_dim     = 400          # match cortical population scale
        self.vsa_items   = {}           # name -> bytearray(0/1) of length vsa_dim
        self.vsa_current = None         # working register (lPFC analog)
        # Day 40 Pack 2 -- Observational VSA Encoding (default OFF, fixed buffer)
        # Per-tick (cid, action) bound events accumulate in a circular buffer.
        # Multi-scale similarity (sim_1, sim_5, sim_20) measures internal
        # representation stability. NO behavior coupling (signal only).
        self.vsa_event       = None
        self.vsa_prev_event  = None
        self.vsa_buffer_size = 128
        self.vsa_buffer      = [None] * 128
        self.vsa_index       = 0
        self.vsa_count       = 0     # total events recorded (for warmup gating)
        # Multi-scale running aggregates.
        self.vsa_sim1_sum    = 0.0
        self.vsa_sim1_count  = 0
        self.vsa_sim5_sum    = 0.0
        self.vsa_sim5_count  = 0
        self.vsa_sim20_sum   = 0.0
        self.vsa_sim20_count = 0
        # Day 40 Pack 3 -- Semantic Validation (default OFF, signal only)
        # Spike threshold raised to 0.5; classification of spikes by what changed
        # (cluster vs action vs both); separation metric for same vs diff cluster
        # similarity; pattern reuse via buffer match counts.
        self.vsa_spike_threshold = 0.5     # raised from Pack 2 (was 0.3)
        self.vsa_spike_count     = 0       # canonical Pack 3 name
        self.vsa_sim_spikes      = 0       # alias for back-compat with Pack 2 harness
        # Spike classification (sum to >= vsa_spike_count; both is a subset)
        self.vsa_spike_cluster_change = 0
        self.vsa_spike_action_change  = 0
        self.vsa_spike_both_change    = 0
        # Stability tracking -- same vs different cluster similarity
        self.vsa_same_cluster_sim_sum = 0.0
        self.vsa_same_cluster_count   = 0
        self.vsa_diff_cluster_sim_sum = 0.0
        self.vsa_diff_cluster_count   = 0
        # Pattern reuse -- buffer matches above threshold per event
        self.vsa_pattern_match_count = 0
        self.vsa_pattern_total       = 0
        self.vsa_pattern_threshold   = 0.8
        # Previous action carrier (needed for spike action-change classification)
        self.vsa_prev_action = None
        # Day 40 Pack 4 -- Representation Manipulation (default OFF, signal only)
        # Decomposition: bind(E, CID) recovers ACT; bind(E, ACT) recovers CID.
        # Chain: bundle(E_t, E_{t-1}) preserves both components above 0.5 cosine.
        # All metrics aggregated; no behavior coupling.
        self.vsa_recover_act_sim = 0.0
        self.vsa_recover_cid_sim = 0.0
        self.vsa_recover_count   = 0
        self.vsa_chain_event = None
        self.vsa_chain_sim   = 0.0
        self.vsa_chain_count = 0
        # Day 41 Pack 1 -- VSA-Prediction Diagnostic Link (default OFF, observation only)
        # Correlates VSA spike events with transition_error (TE) and
        # prediction_confidence (PC) at the same tick.
        # No behavior coupling. Gated by enable_vsa_prediction_link.
        # vsa_last_spike: set by vsa_record_event; read by ikigai hook same tick.
        self.enable_vsa_prediction_link = False
        self.vsa_last_spike    = False  # True if sim_1 < vsa_spike_threshold last event
        self.vsa_spike_te_high = 0      # spike ticks where TE >= vsa_te_threshold
        self.vsa_spike_te_low  = 0      # spike ticks where TE <  vsa_te_threshold
        self.vsa_spike_pc_high = 0      # spike ticks where PC >= vsa_pc_threshold
        self.vsa_spike_pc_low  = 0      # spike ticks where PC <  vsa_pc_threshold
        self.vsa_total_spikes  = 0      # total spike ticks recorded by prediction link
        self.non_spike_te_high = 0      # non-spike ticks where TE >= vsa_te_threshold
        self.non_spike_te_low  = 0      # non-spike ticks where TE <  vsa_te_threshold
        self.non_spike_count   = 0      # total non-spike ticks recorded by prediction link
        self.vsa_te_threshold  = 0.5    # TE boundary: >=0.5 = "high error"
        self.vsa_pc_threshold  = 0.7    # PC boundary: >=0.7 = "high confidence"
        # Day 41 Pack 2 -- Real-Surprise Gated Modulation (default OFF)
        # First controlled coupling: S_real = spike AND TE_high AND PC_high.
        # Minimal BG nudge when system was confidently wrong and structure changed.
        # K_RS=0.001 strictly < K_CF=0.002 << all primary drivers.
        self.enable_vsa_real_modulation = False
        self.vsa_real_surprise          = False  # set at VSA hook; read by BG next tick
        self.vsa_real_trigger_count     = 0      # times S_real fired and was applied
        self.vsa_real_applied_count     = 0      # redundant check: should equal trigger
        self.K_RS                       = 0.001  # explore nudge; 0.5*K_RS suppresses approach
        # Day 41 Pack 3 -- Surprise Persistence Window (temporal extension of S_real)
        # Short-lived internal state: K_SW explore nudge for vsa_sw_duration ticks post S_real.
        # No stacking: timer resets to vsa_sw_duration on each new S_real.
        # K_SW=0.0005 < K_RS=0.001 -- softest layer in the behavioral stack.
        self.enable_vsa_surprise_window = False
        self.vsa_sw_timer        = 0       # ticks remaining in active window
        self.vsa_sw_duration     = 3       # hard bound; window cannot exceed this
        self.K_SW                = 0.0005  # explore nudge per active tick
        self.vsa_sw_activations  = 0       # times S_real started a window
        self.vsa_sw_ticks        = 0       # total waking ticks with window active
        self.vsa_sw_active_tick  = False   # set True this tick if window applied; reset each tick
        # Day 41 Pack 4 -- Pattern-Guided Exploration (context-aware adjustment)
        # Uses recent VSA event similarity to bias direction WITHIN the surprise window.
        # K_PG=0.0003 < K_SW=0.0005 -- weakest modulation layer in behavioral stack.
        self.enable_vsa_pattern_guidance = False
        self.K_PG              = 0.0003  # directional nudge; K_PG / 0.5*K_PG pair
        self.vsa_pg_applied    = 0       # window ticks where guidance was evaluated
        self.vsa_pg_ticks      = 0       # spec alias for applied
        self.vsa_pg_high_ticks = 0       # ticks where sim > 0.7 (stable -> exploit)
        self.vsa_pg_low_ticks  = 0       # ticks where sim < 0.3 (transition -> explore)
        # Day 41 Pack 5 -- Micro-Strategy Commitment (short-lived directional hold)
        # Converts Pack 4 directional signal into a bounded commitment within window.
        # Reset when sw_timer == 0; no stacking (activate only when ms_timer == 0).
        # K_MS=0.0002 < K_PG=0.0003 -- weakest behavioral layer.
        self.enable_vsa_micro_strategy = False
        self.vsa_ms_timer       = 0       # ticks remaining in active commitment
        self.vsa_ms_duration    = 2       # hard bound on hold duration
        self.vsa_ms_action      = None    # 'explore' or 'approach'
        self.K_MS               = 0.0002  # commitment nudge strength
        self.vsa_ms_activations = 0       # times a commitment was started
        self.vsa_ms_ticks       = 0       # total ticks commitment was applied
        self.vsa_ms_active_tick = False   # set True this tick if commitment applied; reset each tick
        # Day 41 Pack 6 -- Cross-Window Continuity Signal
        # Detects repeated S_real contexts via 4-event buffer; K_CT fires only inside window.
        # K_CT=0.0001 < K_MS=0.0002 -- weakest behavioral layer.
        self.enable_vsa_continuity  = False
        self.vsa_ct_buffer          = []     # last vsa_ct_max S_real event vectors
        self.vsa_ct_max             = 4      # hard cap on buffer size
        self.vsa_ct_similarity      = 0.0    # max sim to recent S_real events (set at S_real time)
        self.K_CT                   = 0.0001 # modulation nudge strength
        self.vsa_ct_events          = 0      # S_real events processed by continuity
        self.vsa_ct_matches         = 0      # S_real events with sim > 0.7 (repeat detected)
        self.vsa_ct_mod_ticks       = 0      # window ticks where continuity modulation applied
        # Day 41 Pack 7 -- Repetition-Weighted Modulation (loop-aware adjustment)
        # Scales explore nudge by normalized similarity; fires only in window + repeat context.
        # K_RW=0.00005 < K_CT=0.0001 -- absolute weakest behavioral layer.
        self.enable_vsa_repetition_weighting = False
        self.vsa_rw_strength = 0.0    # normalized intensity in [0, 1]; 0 outside repeat context
        self.K_RW            = 0.00005 # base coefficient; scaled by vsa_rw_strength
        self.vsa_rw_ticks    = 0      # window ticks where repetition modulation applied
        self.vsa_rw_applied  = 0      # alias for vsa_rw_ticks (spec-defined redundant counter)
        # Day 42 Pack 1 -- Cross-Event Direction Accumulation (Trajectory Signal)
        # Accumulates direction (+1 explore, -1 approach, 0 neutral) across S_real events.
        # Short memory (max 6 events); mean score in [-1, 1]; K_TR < K_RW. Window-only apply.
        self.enable_vsa_trajectory = False
        self.vsa_tr_buffer = []       # last vsa_tr_max event directions (+1/-1/0)
        self.vsa_tr_max    = 6        # hard cap; pop(0) when exceeded
        self.vsa_tr_score  = 0.0     # mean direction in [-1, 1]
        self.K_TR          = 0.00003  # K_TR < K_RW=0.00005 -- weakest behavioral layer
        self.vsa_tr_events = 0        # S_real events processed by trajectory
        # Day 43 Pack 2 -- Contextual Trajectory Gating
        # Modulates trajectory influence by vsa_ct_similarity: low sim -> explore boost, high sim -> damp.
        # OFF = bit-identical to Pack 1. K_TR_CTX < K_TR (weaker layer).
        self.enable_vsa_tr_context = False
        self.K_TR_CTX = 0.000015  # < K_TR=0.00003
        # Day 42 Pack 3 -- Trajectory Persistence Stabilization
        # Temporal EMA smoothing of trajectory score across S_real events.
        # OFF = bit-identical to Pack 2. K_TR_PERSIST < K_TR_CTX < K_TR.
        self.enable_vsa_tr_persist = False
        self.K_TR_PERSIST = 0.00001   # < K_TR_CTX=0.000015 < K_TR=0.00003
        self.vsa_tr_persist = 0.0     # EMA of vsa_tr_score (updated at S_real only)
        # Day 42 Pack 4 -- Conflict Resolution Layer
        # Trajectory applied only in low-confidence / conflict states (top two scores close).
        # OFF = bit-identical to Pack 3. K_TR_CONFLICT <= K_TR_PERSIST.
        self.enable_vsa_tr_conflict = False
        self.K_TR_CONFLICT = 0.000008  # <= K_TR_PERSIST=0.00001
        # Day 42 Pack 5 -- Uncertainty-Scaled Trajectory Amplification
        # Scales trajectory by decision uncertainty (inverse of score gap).
        # OFF = bit-identical to Pack 4. K_TR_PERSIST < K_TR_UNCERT < K_TR.
        self.enable_vsa_tr_uncertainty = False
        self.K_TR_UNCERT = 0.00002  # K_TR_PERSIST(0.00001) < K_TR_UNCERT < K_TR(0.00003)

        # Day 42 Pack 6 -- Decision Margin Shaping
        # Multiplicative softening: top1 reduced, top2 lifted when gap < 0.05 and window > 0.
        # OFF = bit-identical to Pack 5. K_MARGIN intentionally larger (multiplicative, not additive).
        self.enable_vsa_margin = False
        self.K_MARGIN = 0.03

        # Day 42 Pack 7 -- Trajectory-Consistent Action Bias
        # Weak multiplicative boost toward repeating the last BG direction.
        # OFF = bit-identical to Pack 6. K_CONSIST < K_MARGIN.
        self.enable_vsa_consistency = False
        self.K_CONSIST = 0.01
        self.vsa_last_action = None

        # Day 42 Pack 8 -- Outcome Trace (Short-Horizon Performance Signal)
        # EMA of recent outcome quality from prediction error TE.
        # OFF = bit-identical to Pack 7. K_OUTCOME < K_CONSIST. Event-level only.
        self.enable_vsa_outcome = False
        self.K_OUTCOME = 0.005
        self.vsa_outcome = 0.0

        # Day 42 Pack 9 -- Outcome Delta (Relative Performance Signal)
        # Detects improvement vs deterioration vs Pack 8's absolute signal.
        # OFF = bit-identical to Pack 8. K_OUTCOME_DELTA < K_OUTCOME.
        self.enable_vsa_outcome_delta = False
        self.K_OUTCOME_DELTA = 0.003
        self.vsa_outcome_prev = 0.0

        # Day 42 Pack 10 -- Signal Alignment Boost
        # Amplifies decision when >=2 of {vsa_tr_persist, vsa_outcome, outcome_delta} agree in sign.
        # OFF = bit-identical to Pack 9. K_ALIGN < K_OUTCOME_DELTA.
        self.enable_vsa_alignment = False
        self.K_ALIGN = 0.002

        # Day 42 Pack 11 -- Exploration Perturbation
        # Rare uniform score perturbation when outcome persistently < -0.2 and in surprise window.
        # OFF = bit-identical to Pack 10. P_PERTURB <= 5% of eligible ticks.
        self.enable_vsa_perturb = False
        self.K_PERTURB = 0.02
        self.P_PERTURB = 0.05

        # Day 42 Pack 12 -- Boundary-Sensitive Perturbation
        # Targeted nudge at decision boundaries when in persistent negative outcome.
        # OFF = bit-identical to Pack 11. P_PERTURB2 <= 3% of eligible ticks.
        self.enable_vsa_perturb2 = False
        self.K_PERTURB2 = 0.015
        self.P_PERTURB2 = 0.03

    # -------------------------------------------------------------------
    # Day 40 Pack 1 -- VSA operators (binding / bundling / similarity)
    # All operators are O(N), allocation-bounded, and pure (no side effects
    # except vsa_init_items which is idempotent). Behavior unaffected when
    # enable_vsa = False (these methods are simply never called).
    # -------------------------------------------------------------------
    def vsa_init_items(self, names=None, seed=42):
        """Generate fixed near-orthogonal binary hypervectors. Idempotent."""
        if self.vsa_items:
            return
        if names is None:
            # Pack 1 atomic concepts plus Pack 2 vocabulary
            # (CID_0..CID_15 cluster ids + raw BG action types + exec-diversity mapped types).
            names = ["A", "B", "STATE", "ACTION", "CONTEXT"]
            for i in range(self.max_clusters if hasattr(self, "max_clusters") else 16):
                names.append(f"CID_{i}")
            for a in ("approach", "explore", "wait", "idle_recover",
                      "edit_code", "run_experiment"):
                names.append(f"ACT_{a}")
        import random as _r
        rng = _r.Random(seed)
        n = self.vsa_dim
        for nm in names:
            self.vsa_items[nm] = bytearray(rng.getrandbits(1) for _ in range(n))

    def vsa_bind(self, a, b):
        """XOR binding. Self-inverse: bind(bind(A,B), A) == B."""
        return bytearray(x ^ y for x, y in zip(a, b))

    def vsa_bundle(self, vectors):
        """Majority-rule bundling. Threshold > half of input count."""
        if not vectors:
            return bytearray(self.vsa_dim)
        n = len(vectors[0])
        half = len(vectors) / 2.0
        out = bytearray(n)
        for i in range(n):
            s = 0
            for v in vectors:
                s += v[i]
            out[i] = 1 if s > half else 0
        return out

    def vsa_similarity(self, a, b):
        """Hamming match ratio. Identical=1.0, random~0.5, orthogonal-ish~0.5."""
        if a is None or b is None:
            return 0.0
        n = len(a)
        if n == 0:
            return 0.0
        matches = 0
        for x, y in zip(a, b):
            if x == y:
                matches += 1
        return matches / n

    def vsa_cosine(self, a, b):
        """Cosine similarity over -1/+1 mapped binary vectors.
        Equivalent to 2 * hamming_match_ratio - 1. Range [-1,+1]; random ~ 0.0."""
        if a is None or b is None:
            return 0.0
        n = len(a)
        if n == 0:
            return 0.0
        matches = 0
        for x, y in zip(a, b):
            if x == y:
                matches += 1
        return 2.0 * (matches / n) - 1.0

    def vsa_encode_event(self, cid, action_type):
        """Build event vector E = bind(CID_<cid>, ACT_<action>). None if missing item."""
        if cid is None or action_type is None:
            return None
        cid_key = f"CID_{cid}"
        act_key = f"ACT_{action_type}"
        cid_v = self.vsa_items.get(cid_key)
        act_v = self.vsa_items.get(act_key)
        if cid_v is None or act_v is None:
            return None
        return self.vsa_bind(cid_v, act_v)

    def vsa_record_event(self, event, curr_cid=None, prev_cid=None,
                          curr_action=None, prev_action=None):
        """Append event to circular buffer; update sim_1 / sim_5 / sim_20 aggregates.

        Day 40 Pack 3: when (curr_cid, prev_cid, curr_action, prev_action) provided,
        also update spike classification, same/diff cluster separation, and
        buffer-wide pattern reuse counters. Classification is signal-only.
        """
        if event is None:
            return
        # ---- sim_1 against most recent prior event ----
        s1 = None
        if self.vsa_event is not None:
            s1 = self.vsa_cosine(event, self.vsa_event)
            self.vsa_sim1_sum   += s1
            self.vsa_sim1_count += 1
            if s1 < self.vsa_spike_threshold:
                self.vsa_spike_count += 1
                self.vsa_sim_spikes  += 1   # legacy alias
                # Classification -- what changed at this transition?
                cluster_changed = (prev_cid is not None and curr_cid is not None
                                   and prev_cid != curr_cid)
                action_changed  = (prev_action is not None and curr_action is not None
                                   and prev_action != curr_action)
                if cluster_changed:
                    self.vsa_spike_cluster_change += 1
                if action_changed:
                    self.vsa_spike_action_change += 1
                if cluster_changed and action_changed:
                    self.vsa_spike_both_change += 1
            # Same/diff cluster separation -- only meaningful when both ids known
            if prev_cid is not None and curr_cid is not None:
                if prev_cid == curr_cid:
                    self.vsa_same_cluster_sim_sum += s1
                    self.vsa_same_cluster_count   += 1
                else:
                    self.vsa_diff_cluster_sim_sum += s1
                    self.vsa_diff_cluster_count   += 1
        # Day 41 Pack 1: communicate spike status to calling hook (same tick).
        self.vsa_last_spike = (s1 is not None and s1 < self.vsa_spike_threshold)
        # ---- sim_5 / sim_20 windowed means + pattern reuse ----
        n_buf = self.vsa_buffer_size
        cnt   = self.vsa_count
        if cnt > 0:
            for window, sum_attr, cnt_attr in (
                (5,  'vsa_sim5_sum',  'vsa_sim5_count'),
                (20, 'vsa_sim20_sum', 'vsa_sim20_count'),
            ):
                k = window if cnt >= window else cnt
                if k <= 0:
                    continue
                acc = 0.0
                for j in range(1, k + 1):
                    idx = (self.vsa_index - j) % n_buf
                    prev = self.vsa_buffer[idx]
                    if prev is None:
                        continue
                    acc += self.vsa_cosine(event, prev)
                setattr(self, sum_attr, getattr(self, sum_attr) + acc / k)
                setattr(self, cnt_attr, getattr(self, cnt_attr) + 1)
            # Pattern reuse -- count buffer entries with similarity above threshold.
            matches = 0
            scan_k = min(cnt, n_buf)
            for j in range(1, scan_k + 1):
                idx = (self.vsa_index - j) % n_buf
                prev = self.vsa_buffer[idx]
                if prev is None:
                    continue
                if self.vsa_cosine(event, prev) > self.vsa_pattern_threshold:
                    matches += 1
            self.vsa_pattern_match_count += matches
            self.vsa_pattern_total       += 1
        # ---- Append to buffer + advance index ----
        self.vsa_buffer[self.vsa_index] = event
        self.vsa_index  = (self.vsa_index + 1) % n_buf
        self.vsa_count += 1
        self.vsa_prev_event  = self.vsa_event
        self.vsa_event       = event
        if curr_action is not None:
            self.vsa_prev_action = curr_action

    def vsa_test_recovery(self, event, cid_vec, act_vec):
        """Day 40 Pack 4: bind(E, CID) -> ACT'; bind(E, ACT) -> CID'. Pure measurement."""
        if event is None or cid_vec is None or act_vec is None:
            return
        rec_act = self.vsa_bind(event, cid_vec)
        rec_cid = self.vsa_bind(event, act_vec)
        self.vsa_recover_act_sim += self.vsa_cosine(rec_act, act_vec)
        self.vsa_recover_cid_sim += self.vsa_cosine(rec_cid, cid_vec)
        self.vsa_recover_count   += 1

    def vsa_compose_chain(self, e_curr, e_prev):
        """Day 40 Pack 4: chain = bundle(E_t, E_{t-1}). Stores result; tracks coherence."""
        if e_curr is None or e_prev is None:
            return None
        chain = self.vsa_bundle([e_curr, e_prev])
        self.vsa_chain_event  = chain
        self.vsa_chain_sim   += (self.vsa_cosine(chain, e_curr) +
                                  self.vsa_cosine(chain, e_prev)) / 2.0
        self.vsa_chain_count += 1
        return chain

    @property
    def vsa_sim1_mean(self):
        return self.vsa_sim1_sum / max(self.vsa_sim1_count, 1)

    @property
    def vsa_sim5_mean(self):
        return self.vsa_sim5_sum / max(self.vsa_sim5_count, 1)

    @property
    def vsa_sim20_mean(self):
        return self.vsa_sim20_sum / max(self.vsa_sim20_count, 1)

    @property
    def vsa_recover_act_mean(self):
        return self.vsa_recover_act_sim / max(self.vsa_recover_count, 1)

    @property
    def vsa_recover_cid_mean(self):
        return self.vsa_recover_cid_sim / max(self.vsa_recover_count, 1)

    @property
    def vsa_chain_mean(self):
        return self.vsa_chain_sim / max(self.vsa_chain_count, 1)

    # -------------------------------------------------------------------
    # Day 46 Pack 1 -- Deterministic Token Encoder operators
    # -------------------------------------------------------------------
    def token_hv_for(self, token):
        """Deterministic binary HV for token. Cached. Local-seeded RNG; no global state.
        If enable_semantic_hv=True and semantic_hv_ready, returns co-occurrence HV instead."""
        if self.enable_semantic_hv and self.semantic_hv_ready and token in self.semantic_hv:
            return self.semantic_hv[token]
        if token in self.token_hv:
            return self.token_hv[token]
        import hashlib, random as _r
        h = int(hashlib.md5(token.encode('utf-8', errors='replace')).hexdigest(), 16)
        rng = _r.Random(h & 0xFFFFFFFF)
        hv = bytearray(rng.getrandbits(1) for _ in range(self.token_hv_dim))
        self.token_hv[token] = hv
        if token not in self.token_to_id:
            tid = len(self.token_to_id)
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
            self.token_unique = len(self.token_to_id)
        return hv

    def semantic_hv_update(self, tokens):
        """Update co-occurrence counts for all tokens in sequence using window context.
        Call this on each training sequence BEFORE semantic_hv_finalize().
        Uses token_hv (random index HVs) as the basis vectors -- NOT semantic_hv."""
        import numpy as np
        w = self.semantic_hv_window
        dim = self.token_hv_dim
        # Pre-convert all token HVs in this sequence to numpy arrays (avoid repeated bytearray indexing)
        hv_arrs = []
        for tok in tokens:
            raw = self.token_hv[tok] if tok in self.token_hv else self.token_hv_for(tok)
            hv_arrs.append(np.frombuffer(raw, dtype=np.uint8).astype(np.int32))
        for i, tok in enumerate(tokens):
            if tok not in self.semantic_hv_counts:
                self.semantic_hv_counts[tok] = np.zeros(dim, dtype=np.int32)
                self.semantic_hv_n[tok] = 0
            counts = self.semantic_hv_counts[tok]
            start = max(0, i - w)
            end   = min(len(tokens), i + w + 1)
            for j in range(start, end):
                if j == i:
                    continue
                counts += hv_arrs[j]
                self.semantic_hv_n[tok] += 1

    def semantic_hv_finalize(self):
        """Convert running bit counts to binary majority-vote HVs. Call once after all training.
        Tokens with no co-occurrence data keep their random index HV."""
        import numpy as np
        for tok, counts in self.semantic_hv_counts.items():
            n = max(self.semantic_hv_n[tok], 1)
            threshold = n / 2.0
            if isinstance(counts, np.ndarray):
                self.semantic_hv[tok] = bytearray((counts > threshold).astype(np.uint8).tobytes())
            else:
                self.semantic_hv[tok] = bytearray(1 if c > threshold else 0 for c in counts)
        self.semantic_hv_ready = True

    def online_semantic_update(self, tokens):
        """Incremental semantic HV update for tokens in a new sequence (mid-conversation).
        Updates co-occurrence counts for affected tokens, then re-finalizes only those tokens.
        Allows novel tokens to accumulate semantic context without full re-training."""
        if not self.enable_semantic_hv:
            return
        self.semantic_hv_update(tokens)   # accumulate counts (creates entries for new tokens)
        # Re-finalize only the tokens that appear in this sequence (cheap, O(seq_len * dim))
        dim = self.token_hv_dim
        import numpy as np
        for tok in set(tokens):
            if tok in self.semantic_hv_counts:
                n = max(self.semantic_hv_n[tok], 1)
                threshold = n / 2.0
                counts = self.semantic_hv_counts[tok]
                if isinstance(counts, np.ndarray):
                    self.semantic_hv[tok] = bytearray((counts > threshold).astype(np.uint8).tobytes())
                else:
                    self.semantic_hv[tok] = bytearray(1 if c > threshold else 0 for c in counts)
        self.semantic_hv_ready = True

    def token_encode_seq_bow(self, tokens):
        """Bag-of-Words semantic HV: bundle token_hv_for() for each token, unordered.
        When enable_semantic_hv=True and semantic_hv_ready, token_hv_for returns semantic HVs,
        so related tokens across domains share bits -> cross-domain retrieval works."""
        dim = self.token_hv_dim
        if not tokens:
            return bytearray(dim)
        hvs = [self.token_hv_for(tok) for tok in tokens]
        return bytearray(self.vsa_bundle(hvs))

    def compute_idf(self, corpus):
        """Compute IDF for every token across corpus (list of token lists).
        idf(t) = log(N / df(t)) where df = number of seqs containing t.
        Stores result in self.token_idf. Call once after loading full corpus."""
        import math
        N = len(corpus)
        if N == 0:
            return
        df = {}
        for seq in corpus:
            for tok in set(seq):
                df[tok] = df.get(tok, 0) + 1
        log_N = math.log(N)
        self.token_idf = {tok: log_N - math.log(count) for tok, count in df.items()}

    def tfidf_bow_hv(self, tokens, alpha=1.0):
        """TF-IDF weighted BoW semantic HV.
        w(t) = idf(t) ** alpha.  alpha=1.0 = full IDF; alpha=0.5 = sqrt-IDF (softer).
        Falls back to uniform weighting (alpha=0 or token_idf empty).
        Returns bytearray of dim bits."""
        import numpy as np
        dim = self.token_hv_dim
        if not tokens:
            return bytearray(dim)
        counts = np.zeros(dim, dtype=np.float64)
        total_weight = 0.0
        for tok in tokens:
            hv = self.token_hv_for(tok)
            raw_idf = self.token_idf.get(tok, 1.0)
            w = raw_idf ** alpha if alpha != 1.0 else raw_idf
            arr = np.frombuffer(hv, dtype=np.uint8).astype(np.float64)
            counts += arr * w
            total_weight += w
        if total_weight == 0:
            return bytearray(dim)
        threshold = total_weight / 2.0
        return bytearray((counts > threshold).astype(np.uint8).tobytes())

    def tokenize(self, text):
        """Deterministic tokenizer: identifiers, integers, two-char ops, punctuation."""
        import re
        return re.findall(
            r'[A-Za-z_]\w*|[0-9]+|==|!=|<=|>=|\*\*|//|[+\-*/=<>():\[\]{},;.]',
            text
        )

    def tokenize_python(self, text):
        """Python tokenizer: handles multi-char ops, type hints, aug-assign, strips comments."""
        import re
        text = re.sub(r'#[^\n]*', ' ', text)   # strip line comments
        # Collapse string literals to STR token
        text = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\'', ' STR ', text)
        tokens = re.findall(
            r'[A-Za-z_]\w*'                         # identifiers / keywords
            r'|[0-9]+\.[0-9]*|[0-9]+'               # floats, integers
            r'|->'                                    # return type annotation
            r'|\*\*=|\*\*|//=|//|<<=|>>='            # 3-char then 2-char special ops
            r'|==|!=|<=|>=|<<|>>'                    # comparison / shift
            r'|\+=|-=|\*=|/=|%=|&=|\|=|\^='          # aug-assign
            r'|[+\-*/%@&|^~<>=!():\[\]{},.]',        # single chars (! for standalone)
            text
        )
        return [t for t in tokens if t.strip()]

    def token_encode_seq(self, tokens):
        """XOR-bind adjacent token pairs; majority-bundle all pairs into one sequence vector."""
        if not tokens:
            return None
        if len(tokens) == 1:
            return self.token_hv_for(tokens[0])
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append(self.vsa_bind(
                self.token_hv_for(tokens[i]),
                self.token_hv_for(tokens[i + 1])
            ))
        return self.vsa_bundle(pairs)

    def token_process_seq(self, tokens, tick=None):
        """Update token transition counts, rebuild probs, compute TE for last pair.

        tick=None -> uses token_seq_updates as clock (predictable rebuild schedule).
        tick=int  -> uses absolute tick (for external sync).
        """
        if not tokens:
            return
        for tok in tokens:
            self.token_hv_for(tok)
            self.token_total += 1
        for i in range(len(tokens) - 1):
            pk, ck = tokens[i], tokens[i + 1]
            if pk not in self.token_transition_counts:
                self.token_transition_counts[pk] = {}
            self.token_transition_counts[pk][ck] = (
                self.token_transition_counts[pk].get(ck, 0) + 1
            )
        for i in range(len(tokens) - 2):
            key = (tokens[i], tokens[i + 1])
            nxt = tokens[i + 2]
            if key not in self.token_transition_counts_3:
                self.token_transition_counts_3[key] = {}
            self.token_transition_counts_3[key][nxt] = (
                self.token_transition_counts_3[key].get(nxt, 0) + 1
            )
        for i in range(len(tokens) - 3):
            key4 = (tokens[i], tokens[i + 1], tokens[i + 2])
            nxt4 = tokens[i + 3]
            if key4 not in self.token_transition_counts_4:
                self.token_transition_counts_4[key4] = {}
            self.token_transition_counts_4[key4][nxt4] = (
                self.token_transition_counts_4[key4].get(nxt4, 0) + 1
            )
        t = self.token_seq_updates if tick is None else tick
        if t - self._tok_pred_last_rebuild >= self.token_pred_rebuild_interval:
            for pk, nexts in self.token_transition_counts.items():
                total = sum(nexts.values())
                if total > 0:
                    self.token_transition_probs[pk] = {
                        k: v / total for k, v in nexts.items()
                    }
            for key, nexts in self.token_transition_counts_3.items():
                total = sum(nexts.values())
                if total > 0:
                    self.token_transition_probs_3[key] = {
                        k: v / total for k, v in nexts.items()
                    }
            for key4, nexts in self.token_transition_counts_4.items():
                total = sum(nexts.values())
                if total > 0:
                    self.token_transition_probs_4[key4] = {
                        k: v / total for k, v in nexts.items()
                    }
            self._tok_pred_last_rebuild = t
        if len(tokens) >= 2:
            lp, lc = tokens[-2], tokens[-1]
            probs = self.token_transition_probs.get(lp, {})
            if probs:
                self.token_transition_error = 1.0 - probs.get(lc, 0.0)
                self.token_prediction_conf = max(probs.values())
            else:
                self.token_transition_error = 1.0
                self.token_prediction_conf = 0.0
        if len(tokens) >= 3:
            lpp, lp, lc = tokens[-3], tokens[-2], tokens[-1]
            probs3 = self.token_transition_probs_3.get((lpp, lp), {})
            if probs3:
                self.token_transition_error_3 = 1.0 - probs3.get(lc, 0.0)
                self.token_prediction_conf_3  = max(probs3.values())
            else:
                self.token_transition_error_3 = self.token_transition_error
                self.token_prediction_conf_3  = self.token_prediction_conf
        if len(tokens) >= 4:
            lp3, lp2, lp1, lc = tokens[-4], tokens[-3], tokens[-2], tokens[-1]
            probs4 = self.token_transition_probs_4.get((lp3, lp2, lp1), {})
            if probs4:
                self.token_transition_error_4 = 1.0 - probs4.get(lc, 0.0)
                self.token_prediction_conf_4  = max(probs4.values())
            else:
                self.token_transition_error_4 = self.token_transition_error_3
                self.token_prediction_conf_4  = self.token_prediction_conf_3
        self.token_seq_updates += 1

    def predict_next_cluster(self, cid):
        """Read-only: argmax next cluster for given cid; None if unknown."""
        nxt = self.cluster_transition_probs.get(cid)
        if not nxt:
            return None
        return max(nxt.items(), key=lambda kv: kv[1])[0]

    def predict_next_state(self, state_key):
        """Read-only: return argmax next_state for given current state, or None."""
        nxt = self.transition_probs.get(state_key)
        if not nxt:
            return None
        return max(nxt.items(), key=lambda kv: kv[1])[0]

    # Day 47 Pack 1 -- Deterministic String-to-Hypervector Encoder operators
    # -------------------------------------------------------------------
    def string_char_hv_for(self, char):
        """Deterministic binary HV for character. Cached. Local-seeded RNG."""
        if char in self._string_char_hv:
            return self._string_char_hv[char]
        import hashlib, random as _r
        h = int(hashlib.md5(f'char:{char}'.encode('utf-8', errors='replace')).hexdigest(), 16)
        rng = _r.Random(h & 0xFFFFFFFF)
        hv = bytearray(rng.getrandbits(1) for _ in range(self.string_hv_dim))
        self._string_char_hv[char] = hv
        self.string_char_unique = len(self._string_char_hv)
        return hv

    def string_pos_hv_for(self, pos):
        """Deterministic binary HV for position index. Cached."""
        if pos in self._string_pos_hv:
            return self._string_pos_hv[pos]
        import hashlib, random as _r
        h = int(hashlib.md5(f'pos:{pos}'.encode()).hexdigest(), 16)
        rng = _r.Random(h & 0xFFFFFFFF)
        hv = bytearray(rng.getrandbits(1) for _ in range(self.string_hv_dim))
        self._string_pos_hv[pos] = hv
        return hv

    def string_normalize(self, text):
        """Strip ends, collapse whitespace. Deterministic."""
        import re
        return re.sub(r'\s+', ' ', str(text).strip())

    def string_encode(self, text):
        """Encode text: position-bound chars + char bigrams, bundled into 1 HV."""
        s = self.string_normalize(text)
        if s in self.string_to_hv:
            return self.string_to_hv[s]
        if not s:
            hv = bytearray(self.string_hv_dim)
            self.string_to_hv[s] = hv
            self.string_unique = len(self.string_to_hv)
            return hv
        components = []
        for i, c in enumerate(s):
            components.append(self.vsa_bind(
                self.string_char_hv_for(c),
                self.string_pos_hv_for(i)
            ))
        for i in range(len(s) - 1):
            components.append(self.vsa_bind(
                self.string_char_hv_for(s[i]),
                self.string_char_hv_for(s[i + 1])
            ))
        hv = self.vsa_bundle(components)
        self.string_to_hv[s] = hv
        self.string_unique = len(self.string_to_hv)
        return hv

    def string_process_seq(self, strings):
        """Update string transition counts, rebuild probs, compute TE for last pair."""
        if not strings:
            return
        for s in strings:
            ns = self.string_normalize(s)
            self.string_char_total += len(ns)
            self.string_encode(ns)
            self.string_total += 1
        for i in range(len(strings) - 1):
            pk = self.string_normalize(strings[i])
            ck = self.string_normalize(strings[i + 1])
            if pk not in self.string_transition_counts:
                self.string_transition_counts[pk] = {}
            self.string_transition_counts[pk][ck] = (
                self.string_transition_counts[pk].get(ck, 0) + 1
            )
        t = self.string_seq_updates
        if t - self._str_pred_last_rebuild >= self._str_rebuild_interval:
            for pk, nexts in self.string_transition_counts.items():
                total = sum(nexts.values())
                if total > 0:
                    self.string_transition_probs[pk] = {
                        k: v / total for k, v in nexts.items()
                    }
            self._str_pred_last_rebuild = t
        if len(strings) >= 2:
            lp = self.string_normalize(strings[-2])
            lc = self.string_normalize(strings[-1])
            probs = self.string_transition_probs.get(lp, {})
            if probs:
                self.string_transition_error = 1.0 - probs.get(lc, 0.0)
                self.string_prediction_conf = max(probs.values())
            else:
                self.string_transition_error = 1.0
                self.string_prediction_conf = 0.0
        self.string_seq_updates += 1

    # Day 48 Pack 1 -- Nearest-Neighbor Associative Retrieval operators

    def nn_register(self, context_hv, prev_tok, next_tok):
        """Store prev_tok->next_tok under context_hv in per-context registry."""
        key = bytes(context_hv)
        if key not in self.nn_registry:
            self.nn_registry[key] = {}
            self.nn_hv_store.append(bytearray(context_hv))
            self.nn_hv_keys.append(key)
        entry = self.nn_registry[key]
        if prev_tok not in entry:
            entry[prev_tok] = {}
        entry[prev_tok][next_tok] = entry[prev_tok].get(next_tok, 0) + 1

    def nn_retrieve(self, query_hv):
        """Return (max_sim, nearest_key). Returns (0.0, None) if registry empty."""
        if not self.nn_hv_store:
            return 0.0, None
        dim = len(query_hv)
        best_sim = -1.0
        best_key = None
        for hv, key in zip(self.nn_hv_store, self.nn_hv_keys):
            matches = sum(a == b for a, b in zip(query_hv, hv))
            sim = matches / dim
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_sim, best_key

    # Day 48 Pack 2 -- Hierarchical Context HV operators

    def nn_phrase_encode(self, tokens):
        """Trigram HV: majority_bundle of bind(bind(tok[i], tok[i+1]), tok[i+2])."""
        if len(tokens) < 3:
            return bytearray(self.token_hv_dim)
        components = []
        for i in range(len(tokens) - 2):
            pair_hv = self.vsa_bind(self.token_hv_for(tokens[i]),
                                    self.token_hv_for(tokens[i + 1]))
            tri_hv = self.vsa_bind(pair_hv, self.token_hv_for(tokens[i + 2]))
            components.append(tri_hv)
        return self.vsa_bundle(components)

    def nn_phrase_register(self, phrase_hv, prev_tok, next_tok):
        """Store prev_tok->next_tok transition under phrase_hv in phrase-level registry."""
        key = bytes(phrase_hv)
        if key not in self.nn_phrase_registry:
            self.nn_phrase_registry[key] = {}
            self.nn_phrase_hv_store.append(bytearray(phrase_hv))
            self.nn_phrase_hv_keys.append(key)
        entry = self.nn_phrase_registry[key]
        if prev_tok not in entry:
            entry[prev_tok] = {}
        entry[prev_tok][next_tok] = entry[prev_tok].get(next_tok, 0) + 1

    def nn_phrase_retrieve(self, query_hv):
        """Return (max_sim, nearest_key) from phrase-level store. (0.0, None) if empty."""
        if not self.nn_phrase_hv_store:
            return 0.0, None
        dim = len(query_hv)
        best_sim = -1.0
        best_key = None
        for hv, key in zip(self.nn_phrase_hv_store, self.nn_phrase_hv_keys):
            matches = sum(a == b for a, b in zip(query_hv, hv))
            sim = matches / dim
            if sim > best_sim:
                best_sim = sim
                best_key = key
        return best_sim, best_key

    # Day 48 Pack 3 -- Compositional Role-Content Binding operators

    def nn_role_hv_for(self, role_name):
        """Deterministic role HV seeded from role name. Never touches global random."""
        if role_name not in self.nn_role_hv:
            import hashlib as _h, random as _r
            seed_int = int.from_bytes(
                _h.md5(f'ROLE:{role_name}'.encode()).digest()[:4], 'big')
            rng = _r.Random(seed_int)
            self.nn_role_hv[role_name] = bytearray(
                rng.randint(0, 1) for _ in range(self.token_hv_dim))
        return self.nn_role_hv[role_name]

    def nn_compose_unit(self, role_name, content_tok):
        """Single compositional unit: XOR(role_hv, content_hv)."""
        return self.vsa_bind(self.nn_role_hv_for(role_name),
                             self.token_hv_for(content_tok))

    def nn_recover_content(self, bundle_hv, role_name):
        """Attempt content recovery: XOR(bundle, role_hv). Inverse of nn_compose_unit."""
        return self.vsa_bind(bundle_hv, self.nn_role_hv_for(role_name))

    def nn_ar_encode_ctx(self, tokens):
        """Accumulated XOR-pair bundle context for autoregressive generation."""
        if not tokens:
            return bytearray(self.token_hv_dim)
        if len(tokens) == 1:
            return bytearray(self.token_hv_for(tokens[0]))
        pairs = [self.vsa_bind(self.token_hv_for(tokens[i]),
                               self.token_hv_for(tokens[i + 1]))
                 for i in range(len(tokens) - 1)]
        return self.vsa_bundle(pairs)

    def nn_ar_decode(self, key, prev_tok):
        """Decode next token from nn_registry key + prev token.
        Falls back to all prev_toks if prev_tok not found.
        Returns (best_tok, margin, entropy)."""
        import math as _m
        if key is None or key not in self.nn_registry:
            return None, 0.0, 1.0
        entry = self.nn_registry[key]
        dist = dict(entry.get(prev_tok, {}))
        if not dist:
            for nd in entry.values():
                for t, c in nd.items():
                    dist[t] = dist.get(t, 0) + c
        if not dist:
            return None, 0.0, 1.0
        total = float(sum(dist.values()))
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        p1 = ranked[0][1] / total
        p2 = ranked[1][1] / total if len(ranked) > 1 else 0.0
        ent = max(0.0, -sum((c / total) * _m.log2(c / total + 1e-10) for _, c in ranked))
        return ranked[0][0], p1 - p2, ent

    def nn_vg_is_direct(self, key, prev_tok):
        """True if prev_tok has a direct entry in nn_registry[key] (not fallback)."""
        if key is None or key not in self.nn_registry:
            return False
        return prev_tok in self.nn_registry[key]

    def nn_vg_build_vocab(self):
        """Collect all registered next-tokens into nn_vg_vocab."""
        vocab = set()
        for entry in self.nn_registry.values():
            for next_dist in entry.values():
                vocab.update(next_dist.keys())
        self.nn_vg_vocab = vocab

    def nn_ar_run_chain(self, prefix, target, use_stabilizer):
        """Run autoregressive chain emitting predicted (not GT) tokens.
        Returns list of (sim, margin, entropy, correct_float, phrase_fired)."""
        tokens = list(prefix)
        results = []
        for tgt in target:
            ctx = self.nn_ar_encode_ctx(tokens)
            sim, key = self.nn_retrieve(ctx)
            best, margin, ent = self.nn_ar_decode(key, tokens[-1])
            fired = False
            if use_stabilizer and margin < self.nn_ar_margin_threshold and len(tokens) >= 3:
                ph = self.nn_phrase_encode(tokens)
                ph_sim, ph_key = self.nn_phrase_retrieve(ph)
                if ph_key is not None:
                    pb, pm, pe = self.nn_ar_decode(ph_key, tokens[-1])
                    if pb is not None:
                        best, margin, ent, sim, fired = pb, pm, pe, ph_sim, True
            correct = 1.0 if best == tgt else 0.0
            results.append((sim, margin, ent, correct, fired))
            tokens.append(best if best is not None else tgt)
        return results

    def nn_holo_reset(self):
        """Clear holographic memory for a fresh capacity test."""
        self.nn_holo_memory   = bytearray(self.nn_holo_dim)
        self.nn_holo_count    = 0
        self.nn_holo_patterns = {}

    def nn_holo_store(self, name, vec):
        """Add vec to holographic superposition (majority-vote bundle grows with each add)."""
        vec = bytearray(vec)
        if self.nn_holo_count == 0:
            self.nn_holo_memory = bytearray(vec)
        else:
            self.nn_holo_memory = bytearray(
                self.vsa_bundle([self.nn_holo_memory, vec]))
        self.nn_holo_patterns[name] = vec
        self.nn_holo_count += 1

    def nn_holo_direct_sim(self, vec):
        """sim(vec, M): overlap of vec with holographic superposition. 0.0 if empty."""
        if self.nn_holo_count == 0 or self.nn_holo_memory is None:
            return 0.0
        return self.vsa_similarity(bytearray(vec), self.nn_holo_memory)

    def nn_holo_recall(self, query):
        """k-NN cleanup: argmax sim(query, stored_pi). Returns (name, sim, margin)."""
        if not self.nn_holo_patterns:
            return None, 0.0, 0.0
        query = bytearray(query)
        best_name, best_sim, second_sim = None, -1.0, -1.0
        for name, vec in self.nn_holo_patterns.items():
            s = self.vsa_similarity(query, vec)
            if s > best_sim:
                second_sim = best_sim
                best_sim   = s
                best_name  = name
            elif s > second_sim:
                second_sim = s
        margin = best_sim - second_sim if second_sim >= 0.0 else 1.0
        return best_name, best_sim, margin

    def nn_holo_test_capacity(self, noise_frac=0.05, seed=42):
        """Test recall for all stored patterns with exact and noisy probes.
        Returns (mean_direct_sim, correct_rate_exact, correct_rate_noisy, mean_margin_exact)."""
        if not self.nn_holo_patterns:
            return 0.0, 0.0, 0.0, 0.0
        import random as _r
        rng = _r.Random(seed)
        dsims, margins, correct0, correct5 = [], [], 0, 0
        for name, vec in self.nn_holo_patterns.items():
            dsims.append(self.nn_holo_direct_sim(vec))
            r0, _, m0 = self.nn_holo_recall(vec)
            if r0 == name:
                correct0 += 1
            margins.append(m0)
            noisy = bytearray(vec)
            nflip = max(1, int(len(noisy) * noise_frac))
            flipped = set()
            while len(flipped) < nflip:
                b = rng.randrange(len(noisy))
                if b not in flipped:
                    flipped.add(b)
                    noisy[b] ^= 1
            r5, _, _ = self.nn_holo_recall(noisy)
            if r5 == name:
                correct5 += 1
        n = len(self.nn_holo_patterns)
        return (sum(dsims) / n, correct0 / n, correct5 / n, sum(margins) / n)

    # ---- Pack 4: Flat Majority Holographic Storage ----

    def nn_holo_flat_reset(self):
        self.nn_holo_flat_vecs   = []
        self.nn_holo_flat_names  = []
        self.nn_holo_flat_memory = bytearray(self.nn_holo_flat_dim if hasattr(self, 'nn_holo_flat_dim') else self.vsa_dim)

    def nn_holo_flat_store(self, name, vec):
        """Add vec; rebuild flat majority over ALL stored patterns."""
        self.nn_holo_flat_vecs.append(bytearray(vec))
        self.nn_holo_flat_names.append(name)
        self.nn_holo_flat_memory = bytearray(self.vsa_bundle(self.nn_holo_flat_vecs))

    def nn_holo_flat_direct_sim(self, vec):
        if not self.nn_holo_flat_vecs:
            return 0.0
        return self.vsa_similarity(bytearray(vec), self.nn_holo_flat_memory)

    def nn_holo_flat_recall(self, query):
        if not self.nn_holo_flat_vecs:
            return None, 0.0, 0.0
        query = bytearray(query)
        best_name, best_sim, second_sim = None, -1.0, -1.0
        for name, vec in zip(self.nn_holo_flat_names, self.nn_holo_flat_vecs):
            s = self.vsa_similarity(query, vec)
            if s > best_sim:
                second_sim = best_sim; best_sim = s; best_name = name
            elif s > second_sim:
                second_sim = s
        margin = best_sim - second_sim if second_sim >= 0.0 else 1.0
        return best_name, best_sim, margin

    def nn_holo_flat_test_capacity(self, noise_frac=0.05, seed=42):
        if not self.nn_holo_flat_vecs:
            return 0.0, 0.0, 0.0, 0.0
        import random as _r
        rng = _r.Random(seed)
        dsims, margins, c0, c5 = [], [], 0, 0
        for name, vec in zip(self.nn_holo_flat_names, self.nn_holo_flat_vecs):
            dsims.append(self.nn_holo_flat_direct_sim(vec))
            r0, _, m0 = self.nn_holo_flat_recall(vec)
            if r0 == name: c0 += 1
            margins.append(m0)
            noisy = bytearray(vec)
            nflip = max(1, int(len(noisy) * noise_frac))
            flipped = set()
            while len(flipped) < nflip:
                b = rng.randrange(len(noisy))
                if b not in flipped: flipped.add(b); noisy[b] ^= 1
            r5, _, _ = self.nn_holo_flat_recall(noisy)
            if r5 == name: c5 += 1
        n = len(self.nn_holo_flat_vecs)
        return sum(dsims)/n, c0/n, c5/n, sum(margins)/n

    # ---- Pack 5: Attractor-Assisted Holographic Cleanup ----

    def nn_holo_attract_run(self, target_idx, noise_frac, max_iters=5, seed=42):
        """Iterative blend cleanup using nn_holo_flat memory.
        Returns list of (iter, recalled_name, sim_to_target, correct(0/1)) per step.
        Blend step: majority([probe, recalled_vec]) pulls probe toward recalled attractor.
        Requires nn_holo_flat_vecs populated with target at target_idx."""
        import random as _r
        if not self.nn_holo_flat_vecs or target_idx >= len(self.nn_holo_flat_vecs):
            return []
        target_name = self.nn_holo_flat_names[target_idx]
        target_vec  = self.nn_holo_flat_vecs[target_idx]
        rng = _r.Random(seed)
        probe = bytearray(target_vec)
        nflip = max(1, int(len(probe) * noise_frac))
        flipped = set()
        while len(flipped) < nflip:
            b = rng.randrange(len(probe))
            if b not in flipped: flipped.add(b); probe[b] ^= 1
        results = []
        for it in range(max_iters):
            recalled, sim_r, margin = self.nn_holo_flat_recall(probe)
            sim_to_tgt = self.vsa_similarity(probe, target_vec)
            correct = 1 if recalled == target_name else 0
            results.append((it, recalled, sim_to_tgt, correct, margin))
            if correct and it > 0:
                break
            if recalled is None:
                break
            r_idx = self.nn_holo_flat_names.index(recalled)
            r_vec = self.nn_holo_flat_vecs[r_idx]
            probe = bytearray(self.vsa_bundle([probe, r_vec]))
        return results

    # ---- Pack 8: Position-Sensitive Bigram Binding ----

    def vsa_rotate(self, vec, shift):
        """Cyclic left-rotation of vector bits by shift positions. Deterministic, invertible."""
        vec = bytearray(vec)
        n = len(vec)
        if n == 0 or shift == 0:
            return vec
        shift = shift % n
        return bytearray(vec[shift:] + vec[:shift])

    def token_bigram_pos_hv(self, tok_a, tok_b, shift=None):
        """Position-sensitive bigram HV: XOR(hv_a, rotate(hv_b, shift)).
        Unlike vsa_bind, this is NOT commutative: (a,b) != (b,a) for distinct tokens."""
        if shift is None:
            shift = self.nn_holo_pos_shift
        return self.vsa_bind(
            bytearray(self.token_hv_for(tok_a)),
            self.vsa_rotate(self.token_hv_for(tok_b), shift)
        )

    def token_encode_seq_pos(self, tokens, shift=None):
        """Like token_encode_seq but uses position-sensitive pairs via token_bigram_pos_hv.
        Fixes commutativity: bigrams (a,b) and (b,a) are now distinct in sequence HVs."""
        if not tokens:
            return None
        if len(tokens) == 1:
            return bytearray(self.token_hv_for(tokens[0]))
        if shift is None:
            shift = self.nn_holo_pos_shift
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append(self.token_bigram_pos_hv(tokens[i], tokens[i + 1], shift))
        return bytearray(self.vsa_bundle(pairs))

    # ---- Pack 9: Holographic Working Memory ----

    def nn_holo_wm_update(self, seq_hv, name=None, tokens=None, te=0.0, domain=""):
        """Add seq_hv to sliding-window WM. Computes novelty BEFORE adding.
        Evicts oldest if window full. Updates memory, novelty, familiar, explore_bias.
        tokens: optional token list for sleep replay (Pack 11).
        te: token_transition_error at encode time for priority replay (Pack 12).
        Returns novelty score (0=familiar, 1=completely novel)."""
        if seq_hv is None:
            return 0.0
        seq_hv = bytearray(seq_hv)
        if name is None:
            name = f"seq_{self.nn_holo_wm_count}"
        if self.nn_holo_wm_memory is not None and len(self.nn_holo_wm_vecs) > 0:
            dsim = self.vsa_similarity(seq_hv, self.nn_holo_wm_memory)
        else:
            dsim = 1.0   # first entry: no prior context, treat as familiar
        self.nn_holo_wm_familiar = dsim
        self.nn_holo_wm_novelty  = 1.0 - dsim
        if len(self.nn_holo_wm_vecs) >= self.nn_holo_wm_size:
            self.nn_holo_wm_vecs.pop(0)
            self.nn_holo_wm_names.pop(0)
            if self.nn_holo_wm_novelties: self.nn_holo_wm_novelties.pop(0)
            if self.nn_holo_wm_seqs: self.nn_holo_wm_seqs.pop(0)
            if self.nn_holo_wm_tes: self.nn_holo_wm_tes.pop(0)
            if self.nn_holo_wm_bow_vecs: self.nn_holo_wm_bow_vecs.pop(0)
            if self.nn_holo_wm_domains: self.nn_holo_wm_domains.pop(0)
            if self.nn_holo_wm_qpos_vecs: self.nn_holo_wm_qpos_vecs.pop(0)
        self.nn_holo_wm_vecs.append(seq_hv)
        self.nn_holo_wm_names.append(name)
        self.nn_holo_wm_novelties.append(self.nn_holo_wm_novelty)
        self.nn_holo_wm_seqs.append(list(tokens) if tokens is not None else None)
        self.nn_holo_wm_tes.append(float(te))
        bow_hv = (self.token_encode_seq_bow(list(tokens))
                  if (self.enable_semantic_hv and self.semantic_hv_ready and tokens is not None)
                  else bytearray(len(seq_hv)))
        self.nn_holo_wm_bow_vecs.append(bow_hv)
        self.nn_holo_wm_domains.append(domain)
        # Question-part PoS: tokens before "A" delimiter (or full seq if no "A").
        # Eliminates answer-token noise when routing short novel queries.
        if tokens is not None:
            q_toks = list(tokens)
            if "A" in q_toks:
                q_toks = q_toks[:q_toks.index("A")]
            qpos_hv = self.token_encode_seq_pos(q_toks) if q_toks else bytearray(len(seq_hv))
        else:
            qpos_hv = bytearray(len(seq_hv))
        self.nn_holo_wm_qpos_vecs.append(qpos_hv)
        self.nn_holo_wm_memory = bytearray(self.vsa_bundle(self.nn_holo_wm_vecs))
        self.nn_holo_wm_count += 1
        self.nn_holo_wm_explore_bias = self.K_WM_NOVELTY * self.nn_holo_wm_novelty
        return self.nn_holo_wm_novelty

    def nn_holo_wm_query(self, seq_hv):
        """Query WM WITHOUT adding. Returns (familiar, novelty). Pure read."""
        if self.nn_holo_wm_memory is None or len(self.nn_holo_wm_vecs) == 0:
            return 1.0, 0.0
        seq_hv = bytearray(seq_hv)
        dsim = self.vsa_similarity(seq_hv, self.nn_holo_wm_memory)
        return dsim, 1.0 - dsim

    def nn_holo_wm_replay_best(self):
        """Return token seq with highest stored novelty in current WM window.
        Returns None if WM empty, or if the best entry has no token seq stored."""
        if not self.nn_holo_wm_novelties:
            return None
        idx = max(range(len(self.nn_holo_wm_novelties)),
                  key=lambda i: self.nn_holo_wm_novelties[i])
        if idx < len(self.nn_holo_wm_seqs) and self.nn_holo_wm_seqs[idx] is not None:
            return self.nn_holo_wm_seqs[idx]
        return None

    def nn_holo_wm_replay_priority_best(self):
        """Return token seq with highest novelty*TE priority in current WM window.
        Returns None if WM empty, no tokens stored, or all priorities == 0.
        Updates priority_top_nov/te/prio with selected entry's metrics."""
        n = min(len(self.nn_holo_wm_novelties), len(self.nn_holo_wm_tes))
        if n == 0:
            return None
        priorities = [self.nn_holo_wm_novelties[i] * self.nn_holo_wm_tes[i]
                      for i in range(n)]
        max_prio = max(priorities)
        if max_prio <= 0.0:
            return None  # all sequences are either familiar or trivially predicted
        idx = priorities.index(max_prio)
        self.nn_holo_wm_priority_top_nov  = self.nn_holo_wm_novelties[idx]
        self.nn_holo_wm_priority_top_te   = self.nn_holo_wm_tes[idx]
        self.nn_holo_wm_priority_top_prio = max_prio
        if idx < len(self.nn_holo_wm_seqs) and self.nn_holo_wm_seqs[idx] is not None:
            return self.nn_holo_wm_seqs[idx]
        return None

    def nn_holo_wm_nearest_seq(self, query_hv, target_domain=None):
        """Find WM entry with highest per-entry similarity to query_hv.
        Unlike nn_holo_wm_query (which uses the bundle), this compares against individual
        stored vectors to find the single best-matching template.
        target_domain: if set, only considers entries with matching domain tag.
        Returns (seq, sim) of best match, or (None, 0.0) if WM is empty."""
        if not self.nn_holo_wm_vecs:
            return None, 0.0
        query_hv = bytearray(query_hv)
        best_sim = -1.0
        best_seq = None
        for i, vec in enumerate(self.nn_holo_wm_vecs):
            if target_domain is not None:
                d = self.nn_holo_wm_domains[i] if i < len(self.nn_holo_wm_domains) else ""
                if d != target_domain:
                    continue
            s = self.vsa_similarity(query_hv, bytearray(vec))
            if s > best_sim:
                best_sim = s
                best_seq = self.nn_holo_wm_seqs[i] if i < len(self.nn_holo_wm_seqs) else None
        return best_seq, best_sim

    def nn_holo_wm_nearest_qpos(self, query_qpos_hv, target_domain=None):
        """Find WM entry with highest question-part PoS similarity to query_qpos_hv.
        Uses nn_holo_wm_qpos_vecs (PoS HV of tokens before 'A') for cleaner routing:
        short novel queries match against question parts only, eliminating answer-token noise.
        target_domain: if set, only considers entries with matching domain tag.
        Returns (seq, sim) of best match, or (None, 0.0) if qpos_vecs empty."""
        if not self.nn_holo_wm_qpos_vecs:
            return None, 0.0
        query_qpos_hv = bytearray(query_qpos_hv)
        best_sim = -1.0
        best_seq = None
        for i, qpos in enumerate(self.nn_holo_wm_qpos_vecs):
            if not qpos:
                continue
            if target_domain is not None:
                d = self.nn_holo_wm_domains[i] if i < len(self.nn_holo_wm_domains) else ""
                if d != target_domain:
                    continue
            s = self.vsa_similarity(query_qpos_hv, bytearray(qpos))
            if s > best_sim:
                best_sim = s
                best_seq = self.nn_holo_wm_seqs[i] if i < len(self.nn_holo_wm_seqs) else None
        return best_seq, best_sim

    def nn_holo_wm_nearest_bow(self, query_bow_hv, target_domain=None):
        """Find WM entry with highest BoW semantic similarity to query_bow_hv.
        Uses nn_holo_wm_bow_vecs (unordered semantic bundles) instead of position-sensitive vecs.
        target_domain: if set (e.g. "code"), only considers WM entries with matching domain tag.
        Returns (seq, sim) of best match, or (None, 0.0) if bow_vecs empty.
        Updates nn_holo_wm_bow_sim with the best similarity found."""
        if not self.nn_holo_wm_bow_vecs:
            return None, 0.0
        query_bow_hv = bytearray(query_bow_hv)
        best_sim = -1.0
        best_seq = None
        for i, bow in enumerate(self.nn_holo_wm_bow_vecs):
            if not bow:
                continue
            if target_domain is not None:
                d = self.nn_holo_wm_domains[i] if i < len(self.nn_holo_wm_domains) else ""
                if d != target_domain:
                    continue
            s = self.vsa_similarity(query_bow_hv, bytearray(bow))
            if s > best_sim:
                best_sim = s
                best_seq = self.nn_holo_wm_seqs[i] if i < len(self.nn_holo_wm_seqs) else None
        self.nn_holo_wm_bow_sim = best_sim
        return best_seq, best_sim

    def _vsa_template_probs(self, template_seq, prev_tok, curr_tok):
        """Extract transition distribution from template_seq at (prev, curr) context.
        Tries trigram (prev, curr)->next first; falls back to bigram curr->next.
        Returns probability dict or {} if current token absent from template."""
        if prev_tok is not None:
            counts = {}
            for i in range(len(template_seq) - 2):
                if template_seq[i] == prev_tok and template_seq[i + 1] == curr_tok:
                    nxt = template_seq[i + 2]
                    counts[nxt] = counts.get(nxt, 0) + 1
            if counts:
                total = sum(counts.values())
                return {k: v / total for k, v in counts.items()}
        counts = {}
        for i in range(len(template_seq) - 1):
            if template_seq[i] == curr_tok:
                nxt = template_seq[i + 1]
                counts[nxt] = counts.get(nxt, 0) + 1
        if counts:
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}
        return {}

    # ---- Pack 1 Day 50: Generation Engine ----

    def generate(self, prompt_tokens, max_len=50, temperature=0.0):
        """Generate tokens. Prefers trigram context when available; falls back to bigram.

        Greedy (temperature=0.0): always pick highest-prob next token.
        Sampled (temperature>0.0): sample from prob^(1/T) distribution.
        Returns full sequence (prompt + generated tokens).
        Stops when no transitions known or max_len reached.
        """
        import random
        tokens = list(prompt_tokens)
        # Seed seen 4-grams from prompt so detection catches repeats that span prompt boundary.
        seen_fourgrams = set()
        for _j in range(len(tokens) - 3):
            seen_fourgrams.add(tuple(tokens[_j:_j + 4]))
        for _ in range(max_len):
            # prefer 4-gram > trigram > bigram when available
            probs = {}
            if self.enable_variorder:
                # variable-order: use highest N-gram whose confidence >= variorder_threshold
                thr = self.variorder_threshold
                if self.enable_fourgram and len(tokens) >= 3:
                    p4 = self.token_transition_probs_4.get(
                        (tokens[-3], tokens[-2], tokens[-1]), {})
                    if p4 and max(p4.values()) >= thr:
                        probs = p4
                        self.gen_order_4g_count += 1
                if not probs and len(tokens) >= 2:
                    p3 = self.token_transition_probs_3.get((tokens[-2], tokens[-1]), {})
                    if p3 and max(p3.values()) >= thr:
                        probs = p3
                        self.gen_order_3g_count += 1
                if not probs:
                    p2 = (self.kn_probs_2.get(tokens[-1], {})
                          if self.enable_kn_smoothing else {})
                    if not p2:
                        p2 = self.token_transition_probs.get(tokens[-1], {})
                    if p2:
                        probs = p2
                        self.gen_order_2g_count += 1
            else:
                if self.enable_fourgram and len(tokens) >= 3:
                    probs = self.token_transition_probs_4.get(
                        (tokens[-3], tokens[-2], tokens[-1]), {})
                if not probs and len(tokens) >= 2:
                    probs = self.token_transition_probs_3.get((tokens[-2], tokens[-1]), {})
                if not probs:
                    p2 = (self.kn_probs_2.get(tokens[-1], {})
                          if self.enable_kn_smoothing else {})
                    if not p2:
                        p2 = self.token_transition_probs.get(tokens[-1], {})
                    probs = p2
            # KN 1-gram final fallback: richer than stopping when all N-grams miss
            if not probs and self.enable_kn_smoothing and self.kn_probs_1:
                probs = self.kn_probs_1
                self.kn_fallback_count += 1
            if not probs:
                break
            # VSA-guided disambiguation: when ambiguous, retrieve nearest WM template
            if self.enable_vsa_guided_gen and probs:
                max_p = max(probs.values())
                if max_p < self.vsa_guide_threshold:
                    self.vsa_guide_count += 1
                    # Pack 8: try BoW semantic retrieval first; fall back to PoS
                    template = None
                    if (self.enable_semantic_hv and self.semantic_hv_ready
                            and self.nn_holo_wm_bow_vecs):
                        _bow_q = self.token_encode_seq_bow(tokens)
                        _bow_tmpl, _bow_sim = self.nn_holo_wm_nearest_bow(_bow_q)
                        if _bow_sim >= self.vsa_guide_threshold:
                            template = _bow_tmpl
                    if template is None:
                        q_hv = self.token_encode_seq_pos(tokens)
                        template, _ = self.nn_holo_wm_nearest_seq(q_hv)
                    if template is not None:
                        prev = tokens[-2] if len(tokens) >= 2 else None
                        curr = tokens[-1]
                        tprobs = self._vsa_template_probs(template, prev, curr)
                        if tprobs:
                            self.vsa_guide_hits += 1
                            all_keys = set(probs) | set(tprobs)
                            probs = {
                                k: self.K_VSA_GUIDE * tprobs.get(k, 0.0) +
                                   (1.0 - self.K_VSA_GUIDE) * probs.get(k, 0.0)
                                for k in all_keys
                            }
            if temperature == 0.0:
                next_tok = max(probs, key=probs.get)
            else:
                inv_t = 1.0 / temperature
                scaled = {k: v ** inv_t for k, v in probs.items()}
                total = sum(scaled.values())
                r = random.random() * total
                cumsum = 0.0
                next_tok = next(iter(scaled))
                for tok, w in scaled.items():
                    cumsum += w
                    if r <= cumsum:
                        next_tok = tok
                        break
            tokens.append(next_tok)
            # Loop detection: stop if the last 4 tokens form a 4-gram seen anywhere before.
            # Seeded with prompt 4-grams -> catches repeats that span prompt boundary.
            if len(tokens) >= 4:
                fg = tuple(tokens[-4:])
                if fg in seen_fourgrams:
                    tokens.pop()   # discard the looping token
                    break
                seen_fourgrams.add(fg)
        self.gen_last_prompt = list(prompt_tokens)
        self.gen_last_output = tokens[len(prompt_tokens):]
        self.gen_last_len    = len(tokens) - len(prompt_tokens)
        return tokens

    # ---- Pack 2 Day 50: Scale Memory ----

    def scale_for_benchmark(self, vsa_dim=1000, wm_size=256):
        """Upgrade to benchmark-scale parameters. Call BEFORE any HV generation.

        vsa_dim=1000 raises capacity N*~140 (vs 56 at 400-dim).
        wm_size=256 retains all 20 CCGB patterns simultaneously.
        token_hv_dim synced to vsa_dim for consistent HV arithmetic.
        Safe to call only on a fresh IkigaiContext before any HVs are encoded.
        """
        self.vsa_dim         = vsa_dim
        self.token_hv_dim    = vsa_dim
        self.nn_holo_dim     = vsa_dim
        self.nn_holo_wm_size = wm_size
        self.scaled_for_benchmark = True

    # ---- Pack 5 Day 50: Continual Learning Harness ----

    def _rebuild_token_probs(self):
        """Force-rebuild bigram, trigram, and 4-gram probs from current counts. Bypasses interval gate."""
        for pk, nexts in self.token_transition_counts.items():
            total = sum(nexts.values())
            if total > 0:
                self.token_transition_probs[pk] = {k: v / total for k, v in nexts.items()}
        for key, nexts in self.token_transition_counts_3.items():
            total = sum(nexts.values())
            if total > 0:
                self.token_transition_probs_3[key] = {k: v / total for k, v in nexts.items()}
        for key4, nexts in self.token_transition_counts_4.items():
            total = sum(nexts.values())
            if total > 0:
                self.token_transition_probs_4[key4] = {k: v / total for k, v in nexts.items()}
        self._tok_pred_last_rebuild = self.token_seq_updates
        if self.enable_kn_smoothing:
            self.kn_rebuild()

    def kn_rebuild(self):
        """Build Kneser-Ney smoothed distributions from current bigram counts.

        KN unigram (kn_probs_1): continuation count / total unique bigrams.
          Tokens appearing in many diverse contexts (e.g. 'the') score higher
          than tokens appearing many times in one context (e.g. 'if' after 'A').

        KN bigram (kn_probs_2): discounted count / total + lambda * KN_unigram.
          Smooths seen bigrams and provides meaningful backoff for unseen ones."""
        D = self.kn_discount
        # Continuation counts: for each token, how many unique preceding tokens?
        cont_sets = {}   # next_tok -> set of unique prev_toks
        for prev, nexts in self.token_transition_counts.items():
            for nxt in nexts:
                if nxt not in cont_sets:
                    cont_sets[nxt] = set()
                cont_sets[nxt].add(prev)
        total_unique = sum(len(s) for s in cont_sets.values())
        if total_unique == 0:
            return
        # KN unigram: proportion of unique bigrams ending in each token
        self.kn_probs_1 = {t: len(s) / total_unique for t, s in cont_sets.items()}
        # KN bigram: discounted + continuation backoff
        self.kn_probs_2 = {}
        for prev, nexts in self.token_transition_counts.items():
            total = sum(nexts.values())
            if total == 0:
                continue
            n_unique = len(nexts)        # unique tokens following prev
            lambda_w = D * n_unique / total   # backoff weight
            probs = {}
            for t, cnt in nexts.items():
                p_kn = self.kn_probs_1.get(t, 1e-10)
                probs[t] = max(cnt - D, 0.0) / total + lambda_w * p_kn
            self.kn_probs_2[prev] = probs

    def ccgb_sleep_cycle(self, n_replays=3):
        """Simulate sleep consolidation: replay highest-priority WM sequences n_replays times."""
        for _ in range(n_replays):
            seq = self.nn_holo_wm_replay_priority_best()
            if seq is None:
                seq = self.nn_holo_wm_replay_best()
            if seq is not None:
                self.token_process_seq(seq)
        self._rebuild_token_probs()

    def ccgb_learn_pattern(self, sequences, n_each=5, n_sleep=3):
        """Learn a pattern from a list of sequences then sleep-consolidate.

        For each sequence: encode HV, update WM with current TE, update transition counts.
        After all sequences: sleep cycle (replay + rebuild probs).
        """
        for seq in sequences * n_each:
            self.token_process_seq(seq)
            hv = self.token_encode_seq_pos(seq)
            self.nn_holo_wm_update(hv, tokens=seq, te=self.token_transition_error_3, domain="code")
        self._rebuild_token_probs()
        self.ccgb_sleep_cycle(n_replays=n_sleep)
        self.ccgb_pattern_count += 1

    def ccgb_token_accuracy(self, prompt, target, max_len=None):
        """Exact token match accuracy: fraction of target tokens correctly generated from prompt."""
        if not target:
            return 1.0
        if max_len is None:
            max_len = len(target) + 2
        gen = self.generate(prompt, max_len=max_len, temperature=0.0)
        generated = gen[len(prompt):]
        matches = sum(
            1 for i, t in enumerate(target)
            if i < len(generated) and generated[i] == t
        )
        return matches / len(target)

    def run_ccgb(self, patterns, examples_each=5, n_sleep=3):
        """Run Continual Code Generation Benchmark.

        patterns: list of dicts with keys:
          'name'      -- string identifier
          'sequences' -- list of token sequences (training examples)
          'prompt'    -- token list (test input)
          'target'    -- token list (expected continuation)

        Returns dict {name: accuracy} for all patterns after sequential learning.
        """
        for pat in patterns:
            self.ccgb_learn_pattern(pat['sequences'], n_each=examples_each, n_sleep=n_sleep)
        results = {
            pat['name']: self.ccgb_token_accuracy(pat['prompt'], pat['target'])
            for pat in patterns
        }
        self.ccgb_results = results
        return results

    # ---- Pack 10 Day 52: Reservoir + RLS output layer ----

    def rls_init(self):
        """Initialize reservoir state, precision matrix P, and output weights W.
        Must be called after scale_for_benchmark() so token_hv_dim is set.
        Safe to call multiple times -- re-initializes from scratch."""
        import numpy as np
        dim = self.token_hv_dim
        self.rls_reservoir = np.zeros(dim, dtype=np.float64)
        self.rls_P = np.eye(dim, dtype=np.float64) * self.rls_delta
        self.rls_W = np.zeros((dim, max(1, len(self.rls_vocab))), dtype=np.float64)
        self.rls_train_count = 0
        self.rls_last_loss = 0.0
        self.rls_train_loss_sum = 0.0

    def rls_register_token(self, token):
        """Ensure token has an RLS vocab index. Returns its index."""
        if token not in self.rls_vocab:
            idx = len(self.rls_vocab)
            self.rls_vocab[token] = idx
            self.rls_vocab_inv[idx] = token
            if self.rls_W is not None:
                import numpy as np
                new_col = np.zeros((self.rls_W.shape[0], 1), dtype=np.float64)
                self.rls_W = np.hstack([self.rls_W, new_col])
        return self.rls_vocab[token]

    def rls_update_reservoir(self, token):
        """EMA update of reservoir state using the token's HV.
        r_t = alpha * r_{t-1} + (1 - alpha) * hv(token)"""
        import numpy as np
        hv = self.token_hv_for(token)
        hv_f = np.array(list(hv), dtype=np.float64)
        if self.rls_reservoir is None:
            self.rls_reservoir = (1.0 - self.rls_alpha) * hv_f
        else:
            self.rls_reservoir = (self.rls_alpha * self.rls_reservoir
                                  + (1.0 - self.rls_alpha) * hv_f)

    def rls_train_step(self, target_token):
        """One RLS update: given current reservoir state, predict target_token.
        Updates W and P using recursive least squares with forgetting factor lambda.
        W_new = W_old + k * (y - W_old^T r)^T
        where k = P r / (lambda + r^T P r)
        Returns prediction error (0.0 if perfect)."""
        import numpy as np
        if self.rls_reservoir is None or self.rls_W is None:
            return 1.0
        self.rls_register_token(target_token)
        r = self.rls_reservoir                        # (dim,)
        vocab_size = len(self.rls_vocab)
        if self.rls_W.shape[1] < vocab_size:
            extra = np.zeros((self.rls_W.shape[0],
                               vocab_size - self.rls_W.shape[1]), dtype=np.float64)
            self.rls_W = np.hstack([self.rls_W, extra])
        # Kalman gain
        Pr = self.rls_P @ r                           # (dim,)
        denom = self.rls_lambda + r @ Pr
        k = Pr / denom                                # (dim,)
        # Prediction: logits for all vocab tokens
        logits = self.rls_W.T @ r                     # (vocab,)
        # Target: one-hot
        y = np.zeros(vocab_size, dtype=np.float64)
        y[self.rls_vocab[target_token]] = 1.0
        # Error
        e = y - logits[:vocab_size]                   # (vocab,)
        # Weight update
        self.rls_W[:, :vocab_size] += np.outer(k, e)
        # P update
        self.rls_P = (self.rls_P - np.outer(k, Pr)) / self.rls_lambda
        # Loss (cross-entropy approximation via squared error norm)
        loss = float(np.dot(e, e))
        self.rls_last_loss = loss
        self.rls_train_loss_sum += loss
        self.rls_train_count += 1
        return loss

    def rls_predict(self, top_k=1):
        """Predict next token(s) from current reservoir state.
        Returns list of (token, score) sorted by descending score, length top_k."""
        import numpy as np
        if self.rls_reservoir is None or self.rls_W is None or not self.rls_vocab:
            return []
        logits = self.rls_W.T @ self.rls_reservoir    # (vocab,)
        vocab_size = len(self.rls_vocab)
        pairs = [(self.rls_vocab_inv[i], float(logits[i]))
                 for i in range(min(vocab_size, len(logits)))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def rls_learn_seq(self, tokens):
        """Feed a token sequence to the reservoir + RLS system.
        Resets reservoir before each sequence to match rls_generate inference behavior.
        For each token: update reservoir, train to predict next token.
        Builds vocab automatically."""
        import numpy as np
        self.rls_reservoir = np.zeros(self.token_hv_dim, dtype=np.float64)
        for tok in tokens:
            self.rls_register_token(tok)
        for i in range(len(tokens) - 1):
            self.rls_update_reservoir(tokens[i])
            self.rls_train_step(tokens[i + 1])
        if tokens:
            self.rls_update_reservoir(tokens[-1])

    def rls_generate(self, prompt_tokens, max_len=20, temperature=0.0):
        """Generate tokens using RLS predictions from reservoir state.
        Resets reservoir, feeds prompt, then generates greedily (temp=0) or sampled (temp>0).
        Returns full sequence (prompt + generated)."""
        import numpy as np, random as _rng
        if self.rls_W is None or not self.rls_vocab:
            return list(prompt_tokens)
        # Multi-turn: start from persistent chat_reservoir; stateless: start from zero.
        if self.enable_multiturn and self.chat_reservoir is not None:
            self.rls_reservoir = self.chat_reservoir.copy()
        else:
            self.rls_reservoir = np.zeros(self.token_hv_dim, dtype=np.float64)
        for tok in prompt_tokens:
            self.rls_update_reservoir(tok)
        tokens = list(prompt_tokens)
        for _ in range(max_len):
            preds = self.rls_predict(top_k=len(self.rls_vocab))
            if not preds:
                break
            if temperature == 0.0:
                next_tok = preds[0][0]
            else:
                scores = np.array([s for _, s in preds], dtype=np.float64)
                scores = scores - scores.max()
                probs = np.exp(scores / temperature)
                probs /= probs.sum()
                idx = _rng.choices(range(len(preds)), weights=probs.tolist())[0]
                next_tok = preds[idx][0]
            tokens.append(next_tok)
            self.rls_update_reservoir(next_tok)
        return tokens

    # ---- Pack 22 Day 53: VSA Analogy Completion ----

    def vsa_analogy(self, a_tok, b_tok, c_tok, exclude=None):
        """Solve a:b :: c:? using VSA binding arithmetic.
        result_hv = hv[a] XOR hv[b] XOR hv[c]; find nearest token.
        Returns (best_token, similarity). Uses semantic HVs when available."""
        ha = self.token_hv_for(a_tok)
        hb = self.token_hv_for(b_tok)
        hc = self.token_hv_for(c_tok)
        result = bytearray(x ^ y ^ z for x, y, z in zip(ha, hb, hc))
        excl = set(exclude or []) | {a_tok, b_tok, c_tok}
        best_tok, best_sim = None, -1.0
        # Search over all registered tokens (semantic HVs when available)
        for tok in self.token_to_id:
            if tok in excl:
                continue
            sim = self.vsa_similarity(result, self.token_hv_for(tok))
            if sim > best_sim:
                best_sim = sim
                best_tok = tok
        self.vsa_analogy_count += 1
        self.vsa_analogy_last_sim = best_sim
        self.vsa_analogy_last_result = best_tok
        if best_sim >= self.vsa_analogy_threshold:
            self.vsa_analogy_hits += 1
        return best_tok, best_sim

    def vsa_analogy_generate(self, prompt_tokens):
        """Derive first response token for a novel prompt via WM analogy.
        Finds nearest stored Q-A pattern, identifies the pivot token that
        differs between stored and novel query, applies:
            stored_pivot : stored_first_answer :: novel_pivot : ?
        Returns list with one token (the inferred answer start) or [].
        Requires enable_vsa_analogy=True and populated WM."""
        if not self.enable_vsa_analogy or not self.nn_holo_wm_bow_vecs:
            return []
        p = list(prompt_tokens)
        if "A" not in p or p[0] != "Q":
            return []
        a_idx = p.index("A")
        q_tokens = p[1:a_idx]
        if not q_tokens:
            return []
        # Find nearest stored Q-A template via BoW semantic similarity
        bow_q = self.token_encode_seq_bow(p)
        tmpl, _sim = self.nn_holo_wm_nearest_bow(bow_q)
        if tmpl is None or "A" not in tmpl:
            return []
        a_tmpl = tmpl.index("A")
        tmpl_q = tmpl[1:a_tmpl]
        tmpl_a = tmpl[a_tmpl + 1:]
        if not tmpl_a or not tmpl_q:
            return []
        # Pivot: most distinctive token in template-Q not in novel-Q
        q_set = set(q_tokens)
        tmpl_set = set(tmpl_q)
        pivots = [(t, self.vsa_analogy_last_sim) for t in tmpl_set - q_set
                  if t in self.token_to_id]
        novels = [t for t in q_set - tmpl_set if t in self.token_to_id]
        if not pivots or not novels:
            return []
        # Use longest content token as pivot/novel (heuristic for most informative)
        pivot = max(tmpl_set - q_set, key=len, default=None)
        novel = max(q_set - tmpl_set, key=len, default=None)
        if pivot is None or novel is None:
            return []
        # Analogy: pivot:tmpl_first :: novel:answer
        tmpl_first = tmpl_a[0]
        result_tok, result_sim = self.vsa_analogy(pivot, tmpl_first, novel,
                                                   exclude={pivot, novel, tmpl_first, "Q", "A"})
        if result_tok is None or result_sim < self.vsa_analogy_threshold:
            return []
        return [result_tok]

    # ---- Pack 21 Day 52: Operation Pattern Table ----

    def op_pattern_learn(self, seq):
        """Extract op->first_response mapping from a Q-A sequence.
        Pattern: ["Q", op_token, ..., "A", first_resp, ...]
        After N calls with same op, op_pattern_table[op] = first_resp.
        Used by rls_hybrid_generate for argument-independent generalization."""
        if not self.enable_op_pattern or len(seq) < 4:
            return
        if seq[0] != "Q" or "A" not in seq:
            return
        op = seq[1]
        a_idx = seq.index("A")
        if a_idx + 1 >= len(seq):
            return
        first_resp = seq[a_idx + 1]
        self.op_pattern_count[op] = self.op_pattern_count.get(op, 0) + 1
        self.op_pattern_table[op] = first_resp

    # ---- Pack 11 Day 52: Hybrid RLS+NGram generation ----

    def rls_hybrid_generate(self, prompt_tokens, max_len=40, temperature=0.0,
                             target_domain=None):
        """Hybrid generation: N-gram for seen contexts, BoW WM for novel.

        Pipeline:
        1. SEEN context: if prompt[-2:] or prompt[-3:] already has N-gram coverage,
           generate directly with N-gram+VSA. Perfect accuracy. No RLS needed.
        2. NOVEL context: N-gram has no coverage for this prompt ending.
           a. RLS predicts top-5 tokens. For each: if (prompt[-1], pred) IS in
              trigram or 4-gram table, extend and use N-gram from there.
           b. BoW WM fallback: find nearest template (domain-filtered).
              Scan template for the prompt as a sub-sequence; return its suffix.
              If no sub-sequence match: return template tokens after same offset.
        3. Pure RLS fallback: worst case, always returns something.

        Key invariant: N-gram continuation only fires when trigram/4-gram
        confirms the transition is in training data (no spurious bigram extension).
        This prevents RLS-predicted tokens from hijacking N-gram into wrong paths."""
        import numpy as np

        # Step 1: Direct N-gram coverage (ONLY for domain-agnostic generation).
        # When target_domain is set, N-gram is domain-blind -- skip to BoW WM
        # which carries explicit domain tags for correct routing.
        p = list(prompt_tokens)
        if target_domain is None:
            last2 = (p[-2], p[-1]) if len(p) >= 2 else None
            last3 = (p[-3], p[-2], p[-1]) if len(p) >= 3 else None
            has_direct = (
                (self.enable_fourgram and last3 is not None and
                 last3 in self.token_transition_probs_4) or
                (last2 is not None and last2 in self.token_transition_probs_3)
            )
            if has_direct:
                return self.generate(p, max_len=max_len, temperature=temperature)

        # Step 1.5: Op-pattern table lookup (argument-independent generalization).
        # Fires for "Q [op] [any_arg] A" when op is in op_pattern_table.
        # Overrides N-gram gap for held-out argument tokens.
        if (self.enable_op_pattern and len(p) >= 3
                and p[0] == "Q" and p[-1] == "A" and p[1] in self.op_pattern_table):
            first_tok = self.op_pattern_table[p[1]]
            if max_len <= 1:
                return p + [first_tok]
            return self.generate(p + [first_tok], max_len=max_len, temperature=temperature)

        # Step 2: WM template retrieval (domain-filtered).
        # Two-stage: position-sensitive first (exact seen queries, high precision),
        # then BoW (novel queries, semantic bridging).
        if self.nn_holo_wm_vecs:
            plen = len(p)
            a_tok = "A"

            # 2a: Question-part PoS retrieval -- matches only the question tokens (before "A").
            # Eliminates answer-token dilution when short novel queries face long stored templates.
            q_part = p[:p.index("A")] if "A" in p else p
            q_part = q_part if q_part else p
            pos_q = self.token_encode_seq_pos(q_part)
            pos_tmpl, pos_sim = self.nn_holo_wm_nearest_qpos(
                pos_q, target_domain=target_domain)
            if pos_tmpl is not None and pos_sim >= 0.55:
                # Good position-sensitive match: find exact sub-sequence
                for start in range(len(pos_tmpl) - plen + 1):
                    if pos_tmpl[start:start + plen] == p:
                        return p + pos_tmpl[start + plen:]
                # No sub-sequence: return after "A" delimiter
                if a_tok in pos_tmpl:
                    aidx = pos_tmpl.index(a_tok)
                    return p + pos_tmpl[aidx + 1:]

            # 2b: BoW retrieval -- for novel prompts with different vocabulary.
            if self.nn_holo_wm_bow_vecs:
                bow_q = self.token_encode_seq_bow(p)
                bow_tmpl, _bow_sim = self.nn_holo_wm_nearest_bow(
                    bow_q, target_domain=target_domain)
                if bow_tmpl is not None:
                    for start in range(len(bow_tmpl) - plen + 1):
                        if bow_tmpl[start:start + plen] == p:
                            return p + bow_tmpl[start + plen:]
                    if a_tok in bow_tmpl:
                        aidx = bow_tmpl.index(a_tok)
                        return p + bow_tmpl[aidx + 1:]
                    if len(bow_tmpl) > plen:
                        return p + bow_tmpl[plen:]

        # Step 2.5: VSA Analogy fallback -- derive first response token via structural analogy.
        # Fires when WM retrieval found no usable template. Genuinely generative:
        # answers novel queries by analogy from nearest stored Q-A pattern.
        if self.enable_vsa_analogy and p[-1] == "A":
            analogy_resp = self.vsa_analogy_generate(p)
            if analogy_resp:
                ext = p + analogy_resp
                if max_len <= 1:
                    return ext
                return self.generate(ext, max_len=max_len, temperature=temperature)

        # Step 3: RLS routing -- extend with predicted token if trigram covers it.
        # Multi-turn: start from persistent chat_reservoir (conversation history blends in).
        if self.enable_multiturn and self.chat_reservoir is not None:
            self.rls_reservoir = self.chat_reservoir.copy()
        else:
            self.rls_reservoir = np.zeros(self.token_hv_dim, dtype=np.float64)
        for tok in p:
            if tok in self.rls_vocab:
                self.rls_update_reservoir(tok)
        top_preds = self.rls_predict(top_k=8)
        for pred_tok, _score in top_preds:
            ext = p + [pred_tok]
            ext_last2 = (ext[-2], ext[-1])
            ext_last3 = (ext[-3], ext[-2], ext[-1]) if len(ext) >= 3 else None
            has_4g = (self.enable_fourgram and ext_last3 is not None and
                      ext_last3 in self.token_transition_probs_4)
            has_3g = ext_last2 in self.token_transition_probs_3
            if has_4g or has_3g:
                return self.generate(ext, max_len=max_len, temperature=temperature)

        # Step 4: pure RLS fallback
        return self.rls_generate(p, max_len=max_len, temperature=temperature)

    def chat(self, text, max_len=25, temperature=0.0):
        """High-level chat interface: tokenize text, generate response, return string.
        Wraps input with Q/A delimiters, calls rls_hybrid_generate, returns token string.
        No domain flag -- semantic similarity routes to English or code automatically.
        When enable_multiturn=True, rls_reservoir persists across calls via chat_reservoir."""
        import numpy as np
        tokens = self.tokenize(text)
        prompt = ["Q"] + tokens + ["A"]
        gen = self.rls_hybrid_generate(prompt, max_len=max_len, temperature=temperature)
        response = gen[len(prompt):]
        if self.enable_multiturn:
            # Update persistent reservoir with this turn's tokens so next turn inherits context.
            if self.chat_reservoir is None:
                self.chat_reservoir = np.zeros(self.token_hv_dim, dtype=np.float64)
            # rls_reservoir was updated during generation; absorb full turn (Q + response).
            saved = self.rls_reservoir.copy() if self.rls_reservoir is not None else None
            self.rls_reservoir = self.chat_reservoir.copy()
            for tok in prompt + response:
                if tok in self.rls_vocab:
                    self.rls_update_reservoir(tok)
            self.chat_reservoir = self.rls_reservoir.copy()
            if saved is not None:
                self.rls_reservoir = saved
        # Online RLS: fine-tune on this exchange immediately (no sleep replay needed).
        # Registers any new tokens, then runs rls_learn_seq on full Q+A sequence.
        # Also updates semantic HVs incrementally so novel tokens gain context over time.
        if self.enable_online_rls and response:
            episode = prompt + response
            for tok in episode:
                self.rls_register_token(tok)
            self.rls_learn_seq(episode)
            self.online_rls_count += len(episode) - 1
            self.online_semantic_update(episode)   # Pack 18: pull novel tokens toward context
        # Episodic WM: store this exchange so future turns can retrieve it.
        # Full sequence = prompt + response; qpos HV = question part (before "A").
        if self.enable_episodic_wm and response:
            episode = prompt + response
            ep_hv = self.token_encode_seq_pos(episode)
            self.nn_holo_wm_update(ep_hv, tokens=episode,
                                   te=0.5, domain="")   # domain="" -- unified
            self.episodic_wm_count += 1
        self.chat_turn_count += 1
        return " ".join(response)

    def chat_reset(self):
        """Clear multi-turn conversation state. Next chat() call starts fresh."""
        import numpy as np
        self.chat_reservoir = np.zeros(self.token_hv_dim, dtype=np.float64)
        self.chat_turn_count = 0

    def env_action_index(self, action):
        """Deterministic action bucket for Day 43 environment branching."""
        key = str(action) if action is not None else "idle_recover"
        return {
            "idle_recover": 0,
            "wait": 0,
            "approach": 1,
            "edit_code": 1,
            "update_memory": 1,
            "explore": 2,
            "run_experiment": 2,
        }.get(key, 0)

    def env_branch_degree(self, state=None):
        """Return the number of outgoing environment options for observation keys."""
        st = self.env_current if state is None else state
        return len(self.env_transitions.get(st, []))

    def env_step(self, action):
        """Advance the controlled Day 43 environment by one deterministic step.

        The branch cursor is based on state-local visit count plus an explicit
        action index. No randomness or Python hash salt participates.
        """
        if not self.enable_env:
            return self.env_current
        # Day 43 Pack 7: regime switch check (deterministic, threshold-gated).
        if self.enable_env_regime:
            self.env_regime_count += 1
            # Day 43 Pack 9: tiny threshold nudge toward better regime (before switch check).
            if self.enable_env_regime_pref and self.env_regime_count > 0:
                if abs(self.env_regime_outcome_gap) > 0.0:
                    _p9_sign = -1 if self.env_regime_outcome_gap < 0 else 1
                    self.env_regime_pref = _p9_sign * min(1.0, abs(self.env_regime_outcome_gap))
                else:
                    self.env_regime_pref = 0.0
                if self.env_regime_outcome_gap < 0:
                    self.env_regime_trigger += self.K_REGIME_PREF * self.env_regime_pref
                elif self.env_regime_outcome_gap > 0:
                    self.env_regime_trigger -= self.K_REGIME_PREF * self.env_regime_pref
            # Day 43 Pack 11: dwell-time guard (after Pack 9, before switch check).
            if self.enable_env_regime_dwell:
                self.env_regime_dwell += 1
                _p11_factor = min(1.0, self.env_regime_dwell / 200.0)
                self.env_regime_dwell_ema = 0.9 * self.env_regime_dwell_ema + 0.1 * _p11_factor
                if self.env_regime_outcome_gap < 0:
                    _p11_better = 0
                elif self.env_regime_outcome_gap > 0:
                    _p11_better = 1
                else:
                    _p11_better = -1
                if _p11_better != -1:
                    _p11_push = self.K_REGIME_DWELL * (1.0 - self.env_regime_dwell_ema)
                    self.env_regime_dwell_adjust = _p11_push * abs(self.env_regime_outcome_gap)
                    if self.env_regime == _p11_better:
                        self.env_regime_trigger += self.env_regime_dwell_adjust
                    else:
                        self.env_regime_trigger -= self.env_regime_dwell_adjust
                else:
                    self.env_regime_dwell_adjust = 0.0
            if self.env_shift_count > 0 and self.env_shift_corr <= self.env_regime_trigger:
                self.env_regime = 1 - self.env_regime
                self.env_regime_switches += 1
                self.env_regime_trigger -= 50.0
                if self.enable_env_regime_dwell:
                    self.env_regime_dwell = 0
        # Day 43 Pack 10: compute regime bias factor (near 1.0; worse regime slightly less).
        if self.enable_env_regime_bias:
            if self.env_regime_outcome_gap == 0.0:
                self.env_regime_bias = 1.0
            else:
                _p10_better = 0 if self.env_regime_outcome_gap < 0 else 1
                if self.env_regime == _p10_better:
                    self.env_regime_bias = 1.0
                else:
                    self.env_regime_bias = 1.0 - min(0.1, self.K_REGIME_BIAS * abs(self.env_regime_outcome_gap))
        curr = self.env_current if self.env_current in self.env_transitions else 0
        # Pack 7: select active transition graph based on current regime.
        _active_trans = self.env_transitions_r1 if (self.enable_env_regime and self.env_regime == 1) else self.env_transitions
        options = _active_trans.get(curr, [0])
        action_index = self.env_action_index(action)
        visits = self.env_visit_counts.get(curr, 0)
        if len(options) <= 1:
            branch_index = 0
        else:
            # Pack 2 -- gated action bias.
            # Base is purely visit-driven (Pack 1 behaviour when OFF).
            base_idx = visits % len(options)
            if self.enable_env_action:
                # action_index already computed above; reuse it, no second call.
                # Small deterministic integer shift; no randomness, no direct control.
                branch_index = (base_idx + self.K_ENV_ACT * action_index) % len(options)
            else:
                branch_index = base_idx
        next_state = options[branch_index]

        self.env_visit_counts[curr] = visits + 1
        pair = (curr, next_state)
        self.env_transition_counts[pair] = self.env_transition_counts.get(pair, 0) + 1
        self.env_last_state = curr
        self.env_last_action = action
        self.env_last_branch_index = branch_index
        self.env_current = next_state
        self.env_step_count += 1
        if next_state not in self.env_visited:
            self.env_visited.append(next_state)
        return next_state

    def reset(self):
        """Reset experiment-controlled flags. Call after exec() in test harnesses."""
        self.__init__()
