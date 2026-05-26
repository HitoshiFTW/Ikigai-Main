"""
ikigai.cognition.multistep_planner -- Multi-step Planner w/ Backtrack.

Day 55 Pack 73 -- Phase B: goal-directed planning via CausalWorldModel + FE.

Algorithm: bounded-depth DFS w/ FE-guided heuristic.
    plan(start, goal, max_depth, beam_width)
        For depth d in 1..max_depth:
            DFS from start. At each node:
                - rank candidate actions by FE w.r.t. goal
                - take top-beam_width actions
                - recurse
            Stop when goal reached or max_depth exhausted.
        Backtrack on dead ends (node has no actions that reduce FE below threshold).

    Returns: (action_sequence, state_trajectory, total_fe_drop, success_bool)

FE-guided heuristic:
    cost(state, action) = expected_FE(action, goal | current_state)
                        = -cos(predicted_next_state, goal) - lam * cos(action, belief)

Beam search variant:
    Keep top-beam_width plans per depth. Bounded memory growth.

Bio analog: prefrontal cortex tree search + BG action gating.

vs LLM: chain-of-thought = single trajectory. No backtrack.
        MultiStepPlanner: explicit tree search with goal-guided pruning.
"""

import numpy as np

from ikigai.cognition.causal_world_model import CausalWorldModel
from ikigai.cognition.fe_action import FreeEnergyActionSelector


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MultiStepPlanner:
    """
    Plans an action sequence from start_state to goal_state.

    plan(start, goal, max_depth, beam_width, score_min)
        Returns dict with:
            actions       -- list of action names
            trajectory    -- [(state, action, next_state), ...]
            success       -- bool (reached goal)
            depth         -- length of plan
            total_score   -- sum of per-step scores

    plan_with_backtrack(start, goal, max_depth, branching)
        Variant: DFS with backtrack on dead ends.

    score_plan(actions) -- score a candidate plan by chained predictions.
    """

    def __init__(self, world_model, action_selector=None):
        self.world  = world_model
        self.fea    = action_selector

    #  score a plan

    def score_plan(self, start_state, actions):
        """Walk plan, return (final_state, total_score, trajectory)."""
        current = start_state
        total   = 0.0
        traj    = []
        for action in actions:
            pred = self.world.predict(current, action, top_k=1)
            if not pred:
                return None, 0.0, traj
            next_state, score = pred[0]
            traj.append((current, action, next_state))
            total += score
            current = next_state
        return current, total, traj

    #  beam search plan

    def plan(self, start_state, goal_state, max_depth=5, beam_width=3, score_min=0.1):
        """
        Bounded beam search. At each depth, expand top beam_width frontiers.
        Returns best plan found within depth limit.
        """
        candidate_actions = list(self.world._actions.keys())
        if not candidate_actions:
            return self._empty_plan(start_state)

        # Frontier: list of (current_state, action_path, traj, total_score)
        frontier = [(start_state, [], [], 0.0)]

        for depth in range(max_depth):
            new_frontier = []
            for (state, path, traj, total) in frontier:
                if state == goal_state:
                    new_frontier.append((state, path, traj, total))
                    continue
                # Try each action, rank by predicted score (toward goal)
                ranked = []
                for action in candidate_actions:
                    pred = self.world.predict(state, action, top_k=1)
                    if not pred:
                        continue
                    next_state, score = pred[0]
                    if score < score_min:
                        continue
                    # Heuristic: prefer next_state == goal, else use score
                    goal_match = 1.0 if next_state == goal_state else 0.0
                    rank_value = score + 10.0 * goal_match
                    ranked.append((action, next_state, score, rank_value))
                ranked.sort(key=lambda x: -x[3])
                # Beam: keep top beam_width
                for (action, next_state, score, _) in ranked[:beam_width]:
                    new_frontier.append((
                        next_state,
                        path + [action],
                        traj + [(state, action, next_state)],
                        total + score,
                    ))

            # Check if any frontier reached goal
            goals_found = [f for f in new_frontier if f[0] == goal_state]
            if goals_found:
                # Pick best
                best = max(goals_found, key=lambda f: f[3])
                return {
                    'actions':     best[1],
                    'trajectory':  best[2],
                    'success':     True,
                    'depth':       len(best[1]),
                    'total_score': best[3],
                }

            # Prune to beam_width * max-frontier
            new_frontier.sort(key=lambda f: -f[3])
            frontier = new_frontier[: beam_width * 4]
            if not frontier:
                break

        # No goal reached; return best partial plan
        if frontier:
            best = max(frontier, key=lambda f: f[3])
            return {
                'actions':     best[1],
                'trajectory':  best[2],
                'success':     False,
                'depth':       len(best[1]),
                'total_score': best[3],
            }
        return self._empty_plan(start_state)

    #  DFS with backtrack

    def plan_with_backtrack(self, start_state, goal_state, max_depth=5, branching=3):
        """DFS with explicit backtrack tracking."""
        candidate_actions = list(self.world._actions.keys())
        stats = {'expansions': 0, 'backtracks': 0}

        def dfs(state, depth, path, traj, score):
            stats['expansions'] += 1
            if state == goal_state:
                return path, traj, score, True
            if depth >= max_depth:
                return None
            # Rank actions
            ranked = []
            for action in candidate_actions:
                pred = self.world.predict(state, action, top_k=1)
                if not pred:
                    continue
                next_state, s = pred[0]
                if s < 0.1:
                    continue
                # Avoid revisits (cycle prevention)
                if next_state in [t[0] for t in traj] + [traj[-1][2] if traj else None]:
                    continue
                ranked.append((action, next_state, s))
            ranked.sort(key=lambda x: -x[2])

            for (action, next_state, s) in ranked[:branching]:
                result = dfs(
                    next_state,
                    depth + 1,
                    path + [action],
                    traj + [(state, action, next_state)],
                    score + s,
                )
                if result is not None and result[3]:
                    return result
            stats['backtracks'] += 1
            return None

        outcome = dfs(start_state, 0, [], [], 0.0)
        if outcome is not None:
            actions, traj, score, success = outcome
            return {
                'actions':     actions,
                'trajectory':  traj,
                'success':     success,
                'depth':       len(actions),
                'total_score': score,
                'expansions':  stats['expansions'],
                'backtracks':  stats['backtracks'],
            }
        return {
            'actions':     [],
            'trajectory':  [],
            'success':     False,
            'depth':       0,
            'total_score': 0.0,
            'expansions':  stats['expansions'],
            'backtracks':  stats['backtracks'],
        }

    #  helpers

    def _empty_plan(self, start_state):
        return {
            'actions':     [],
            'trajectory':  [],
            'success':     False,
            'depth':       0,
            'total_score': 0.0,
        }
