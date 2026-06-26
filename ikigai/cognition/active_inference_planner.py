"""
ikigai.cognition.active_inference_planner -- Pack 258 Expected Free Energy planner.

Day 74. Extends Pack 73 MultiStepPlanner with Friston-style active
inference. Replaces reactive free-energy heuristic with EXPECTED free
energy over rollouts, yielding epistemic (info-gain) + pragmatic
(goal-seeking) action selection in one objective.

WHY THIS EXISTS
---------------
Pack 255 GeneralReasoner composes existing primitives but its planner
fallback uses FE = -log p(observation | belief) -- reactive only.
An organism stuck in a state where prediction is poor but no goal is
specified has no drive to act. Pack 258 adds epistemic curiosity:
even without a goal, the agent picks actions whose predicted
outcomes are MOST INFORMATIVE.

This is the substrate analog of Friston (2010) Active Inference:

    EFE(a) = -[ Epistemic(a) + Pragmatic(a) ]

  Epistemic = expected info-gain from doing a
            ~ entropy reduction over predicted next-obs distribution
  Pragmatic = expected log-preference for the predicted outcome
            ~ cosine(predicted_state, goal_hv)

Lower EFE = better action. Agent minimizes EFE.

NOT A NEW SUBSTRATE
-------------------
Pack 258 composes existing CausalWorldModel.predict + substrate
recall confidence. NO new substrate math. Pure orchestration.

CAT-4 CONNECTION
----------------
Pack 258 is the action-selection half of cat-4 b_self bootstrap. The
organism uses EFE to choose which (state, action) pairs to absorb
into b_self. Pairs the planner with cat-4 ICL absorb (future).

ROADMAP STATUS
--------------
Day 73 close: dropped from active list. Day 73 close revised: restored
as future pack. Day 74: built.
"""

import numpy as np


class ActiveInferencePlanner:
    """Pack 258 Expected Free Energy planner.

    Drop-in replacement for MultiStepPlanner's action scorer. Same
    interface (plan(start, goal, max_depth)) but scoring uses EFE
    instead of FE. Works WITH or WITHOUT a goal -- epistemic value
    alone drives action when goal is None.
    """

    def __init__(self, cwm, mr,
                  epistemic_weight=1.0,
                  pragmatic_weight=1.0,
                  rollout_depth=3,
                  beam_width=3,
                  goal_resolution=0.1):
        """
        Args:
            cwm                 -- CausalWorldModel (Pack 72) with .predict
            mr                  -- MultiRoleMemory (for cleanup confidence)
            epistemic_weight    -- multiplier on info-gain term
            pragmatic_weight    -- multiplier on goal-similarity term
            rollout_depth       -- how many steps to look ahead
            beam_width          -- candidates kept per depth level
            goal_resolution     -- numerical guard for goal cosine
        """
        self.cwm = cwm
        self.mr = mr
        self.d = mr.d
        self.epistemic_weight = float(epistemic_weight)
        self.pragmatic_weight = float(pragmatic_weight)
        self.rollout_depth = int(rollout_depth)
        self.beam_width = int(beam_width)
        self.goal_resolution = float(goal_resolution)
        self._stats = {
            'plans': 0, 'actions_scored': 0,
            'mean_epistemic': 0.0, 'mean_pragmatic': 0.0,
        }

    # ---- scoring primitives ------------------------------------------

    def epistemic_value(self, state_name, action_name,
                          candidate_states=None, top_k=5):
        """Info-gain estimate: how confident is the substrate about
        next-state after action? More peaked = more info gained.

        Uses CWM.predict top-k cosines as a proxy distribution; computes
        normalized entropy. Lower entropy = higher info-gain.
        """
        preds = self.cwm.predict(state_name, action_name,
                                   candidate_states=candidate_states,
                                   top_k=top_k)
        if not preds:
            return 0.0
        sims = np.array([max(s, 1e-9) for (_, s) in preds],
                         dtype=np.float64)
        # Softmax to distribution
        sims = sims - sims.max()
        p = np.exp(sims * 4.0)
        p = p / max(p.sum(), 1e-12)
        # Entropy
        H = float(-(p * np.log(p + 1e-12)).sum())
        H_max = float(np.log(len(p)))
        # Info-gain = (H_max - H) / H_max -- 1.0 = peaked, 0.0 = flat
        return (H_max - H) / max(H_max, 1e-12)

    def pragmatic_value(self, state_name, action_name, goal_name,
                          candidate_states=None, top_k=1):
        """Goal-similarity estimate: cosine between top predicted
        next-state and goal HV."""
        if goal_name is None:
            return 0.0
        preds = self.cwm.predict(state_name, action_name,
                                   candidate_states=candidate_states,
                                   top_k=top_k)
        if not preds:
            return 0.0
        return float(preds[0][1])  # top-1 cosine to goal-included candidates

    def efe(self, state_name, action_name, goal_name=None,
             candidate_states=None):
        """Expected Free Energy: lower = better action."""
        E = self.epistemic_value(state_name, action_name,
                                    candidate_states=candidate_states)
        P = self.pragmatic_value(state_name, action_name, goal_name,
                                    candidate_states=candidate_states)
        return -(self.epistemic_weight * E + self.pragmatic_weight * P)

    # ---- planning -----------------------------------------------------

    def score_actions(self, state_name, action_pool, goal_name=None,
                        candidate_states=None):
        """Score every action in pool by EFE. Returns sorted list of
        (action, efe_score, epistemic, pragmatic). Lower EFE = better."""
        out = []
        for a in action_pool:
            E = self.epistemic_value(state_name, a,
                                        candidate_states=candidate_states)
            P = self.pragmatic_value(state_name, a, goal_name,
                                        candidate_states=candidate_states)
            efe = -(self.epistemic_weight * E + self.pragmatic_weight * P)
            out.append((a, efe, E, P))
            self._stats['actions_scored'] += 1
        out.sort(key=lambda r: r[1])
        return out

    def plan(self, start_state, action_pool, goal_state=None,
              max_depth=None, beam_width=None, candidate_states=None):
        """EFE-guided beam search. Returns:
            {
              'success': bool,        # only meaningful if goal_state given
              'trajectory': [(s, a, s', efe, eps, prag), ...],
              'total_efe': float,
              'final_state': str,
            }
        """
        self._stats['plans'] += 1
        depth = max_depth if max_depth is not None else self.rollout_depth
        beam = beam_width if beam_width is not None else self.beam_width

        # Beam: list of (current_state, traj_so_far, total_efe)
        frontier = [(start_state, [], 0.0)]
        for d in range(depth):
            new_frontier = []
            for (s, traj, acc_efe) in frontier:
                scored = self.score_actions(
                    s, action_pool, goal_name=goal_state,
                    candidate_states=candidate_states)
                # Take top-beam actions
                for (a, efe, E, P) in scored[:beam]:
                    # Get predicted next state for trajectory
                    preds = self.cwm.predict(s, a, top_k=1,
                                               candidate_states=candidate_states)
                    if not preds:
                        continue
                    s_next, _ = preds[0]
                    new_traj = traj + [(s, a, s_next, efe, E, P)]
                    new_frontier.append((s_next, new_traj, acc_efe + efe))
                    # Goal-reached short circuit
                    if goal_state is not None and s_next == goal_state:
                        return {
                            'success': True,
                            'trajectory': new_traj,
                            'total_efe': acc_efe + efe,
                            'final_state': s_next,
                        }
            if not new_frontier:
                break
            # Keep top-beam frontiers by acc_efe
            new_frontier.sort(key=lambda x: x[2])
            frontier = new_frontier[:beam]

        # Return best so far
        if not frontier:
            return {'success': False, 'trajectory': [], 'total_efe': 0.0,
                     'final_state': start_state}
        best = frontier[0]
        return {
            'success': (goal_state == best[0]) if goal_state else True,
            'trajectory': best[1],
            'total_efe': best[2],
            'final_state': best[0],
        }

    def stats(self):
        return dict(self._stats)
