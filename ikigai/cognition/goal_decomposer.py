"""
ikigai.cognition.goal_decomposer -- Goal Decomposition Engine.

Day 55 Pack 44 -- Layer 3 Reasoning (LLM replacement stack).

Problem: current system responds turn-by-turn with no planning.
         Complex tasks (5000-line codegen, 20-step reasoning) need
         hierarchical task decomposition before execution.

Fix: GoalDecomposer maintains a plan corpus (goal_hv → plan steps).
     decompose(goal_tokens) = nearest-plan retrieval + step binding.
     Leaves are atomic actions: recall, codegen, math, kb_query, verify, respond.

Plan corpus built in two ways:
  1. Built-in templates (6 task archetypes, registered at init)
  2. register_plan(name, intent_tokens, steps) -- user-defined

Retrieval: cosine(query_hv, template_intent_hv) -- same as SkillCrystal.
Execution: execute_plan(plan) -> ordered (step_name, action_type) list.

No forgetting: plan corpus additive (new plans never overwrite old).
"""

import numpy as np

ATOMIC_ACTIONS = ['recall', 'codegen', 'math', 'kb_query', 'verify', 'respond', 'clarify', 'learn']

BUILTIN_PLANS = [
    {
        'name':   'code_generation',
        'intent': ['generate', 'code', 'function', 'algorithm', 'implement', 'write'],
        'steps':  [('parse_requirements', 'recall'),
                   ('generate_code',      'codegen'),
                   ('verify_output',      'verify')],
    },
    {
        'name':   'information_retrieval',
        'intent': ['find', 'search', 'retrieve', 'look', 'query', 'fetch'],
        'steps':  [('parse_query',     'kb_query'),
                   ('filter_results',  'recall'),
                   ('present_answer',  'respond')],
    },
    {
        'name':   'arithmetic_reasoning',
        'intent': ['calculate', 'compute', 'solve', 'math', 'arithmetic', 'number'],
        'steps':  [('parse_problem', 'recall'),
                   ('compute',       'math'),
                   ('verify_answer', 'verify')],
    },
    {
        'name':   'skill_learning',
        'intent': ['learn', 'teach', 'show', 'example', 'pattern', 'demonstrate'],
        'steps':  [('encode_intent',  'learn'),
                   ('store_skill',    'learn'),
                   ('confirm_stored', 'respond')],
    },
    {
        'name':   'skill_recall',
        'intent': ['remember', 'apply', 'use', 'execute', 'run', 'recall', 'invoke'],
        'steps':  [('query_skills', 'recall'),
                   ('retrieve',     'recall'),
                   ('apply',        'codegen')],
    },
    {
        'name':   'conversation',
        'intent': ['answer', 'explain', 'discuss', 'respond', 'tell', 'describe'],
        'steps':  [('understand', 'recall'),
                   ('formulate',  'respond'),
                   ('deliver',    'respond')],
    },
]


def _hv(word, d):
    seed = hash(f'goal::{word}') & 0x7FFFFFFF
    rng  = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=d) * 2 - 1).astype(np.float32)


def _encode(tokens, d):
    s = np.zeros(d, dtype=np.float32)
    for t in tokens:
        s += _hv(t, d)
    out = np.sign(s).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _sim(a, b, d):
    return float(np.dot(a, b)) / d


class PlanStep:
    """One step in a decomposed plan."""
    __slots__ = ['name', 'action_type', 'sub_goal']

    def __init__(self, name, action_type, sub_goal=None):
        self.name        = name
        self.action_type = action_type
        self.sub_goal    = sub_goal or []

    def __repr__(self):
        return f'PlanStep({self.name!r}, action={self.action_type!r})'


class GoalDecomposer:
    """
    Hierarchical goal → plan decomposition via HV nearest-plan retrieval.

    register_plan(name, intent_tokens, steps)  -- add plan template
    decompose(goal_tokens, depth=1)            -- retrieve + instantiate plan
    execute_plan(plan)                         -- (step_name, action_type) list
    nearest_plan(goal_tokens)                  -- (plan_name, score)

    No forgetting: plan corpus is append-only.
    Built-in: 6 archetypes (code_generation, retrieval, arithmetic,
              skill_learning, skill_recall, conversation).
    """

    def __init__(self, d=400):
        self.d      = d
        self._plans = {}   # name → {intent_hv, steps, count}

        for p in BUILTIN_PLANS:
            self.register_plan(p['name'], p['intent'], p['steps'])

    # ── registration ──────────────────────────────────────────────────────

    def register_plan(self, name, intent_tokens, steps):
        """
        steps: list of (step_name, action_type) tuples.
        Idempotent: re-registering bundles intent HVs.
        """
        intent_hv = _encode(intent_tokens, self.d)
        plan_steps = [
            PlanStep(sname, atype)
            for sname, atype in steps
        ]
        if name in self._plans:
            e = self._plans[name]
            bundled = np.sign(e['intent_sum'] + intent_hv).astype(np.float32)
            bundled[bundled == 0] = 1.0
            e['intent_hv']  = bundled
            e['intent_sum'] += intent_hv
            e['count']      += 1
        else:
            self._plans[name] = {
                'intent_hv':     intent_hv,
                'intent_sum':    intent_hv.copy(),
                'steps':         plan_steps,
                'intent_tokens': list(intent_tokens),
                'count':         1,
            }

    # ── decomposition ─────────────────────────────────────────────────────

    def nearest_plan(self, goal_tokens):
        """Return (plan_name, score) of best matching plan."""
        if not self._plans:
            return None, 0.0
        q = _encode(goal_tokens, self.d)
        best_name, best_score = None, -2.0
        for name, e in self._plans.items():
            s = _sim(q, e['intent_hv'], self.d)
            if s > best_score:
                best_score, best_name = s, name
        return best_name, best_score

    def decompose(self, goal_tokens, depth=1):
        """
        Retrieve nearest plan, return list of PlanSteps.
        depth > 1: recursively decompose each step's sub_goal.
        Returns [] if no plans registered.
        """
        name, score = self.nearest_plan(goal_tokens)
        if name is None:
            return []
        plan = self._plans[name]
        steps = list(plan['steps'])

        if depth > 1:
            for step in steps:
                sub = self.decompose([step.name], depth=depth - 1)
                step.sub_goal = sub

        return steps

    def execute_plan(self, steps):
        """
        Flatten plan into ordered [(step_name, action_type)] sequence.
        Depth-first traversal of sub_goal trees.
        """
        result = []
        for step in steps:
            if step.sub_goal:
                result.extend(self.execute_plan(step.sub_goal))
            else:
                result.append((step.name, step.action_type))
        return result

    def all_plans(self, top_k=3, query_tokens=None):
        """Return top_k plans by intent cosine to query (or all if no query)."""
        if query_tokens:
            q = _encode(query_tokens, self.d)
            scored = [(n, _sim(q, e['intent_hv'], self.d)) for n, e in self._plans.items()]
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]
        return list(self._plans.keys())[:top_k]

    # ── stats ─────────────────────────────────────────────────────────────

    @property
    def plan_count(self):
        return len(self._plans)

    @property
    def total_registrations(self):
        return sum(e['count'] for e in self._plans.values())
