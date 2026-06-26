# ===========================================================================
# TOOL ROUTER  (Day 32 Pack 4 — Controlled Modular Extraction)
# Extracted from ikigai.py lines 4424-4565.
# Only dependency: collections.deque — zero coupling to runtime state.
# ===========================================================================

from collections import deque


class ToolRouter:
    """
    Intent routing layer: maps cognitive tasks to specific internal operations.

    Reads the active task from TaskFramework and converts it into a structural
    intent. This layer acts as the proto-API boundary: it decides *what* tool
    or operation profile the organism would theoretically execute (e.g. inspect,
    continue, expand) before any actual execution engine exists.

    Operation type mapping
    ----------------------
    'high_priority_stabilization'  -> 'inspect_and_adjust'
    'focused_continuation'         -> 'continue_thread'
    'exploratory_probe'            -> 'compare_paths'
    'background_monitoring'        -> 'observe_only'
    'research_expansion'           -> 'hypothesis_expand'

    Confidence Blend
    ----------------
    route_confidence = (task_priority * 0.80) + (recent narrative continuity * 0.20)
    Clamped to [0.0, 1.0].

    Route schema
    ------------
    tick             : int
    operation_type   : str
    task_origin      : str
    route_confidence : float [0, 1]
    route_summary    : str         — always exactly 2 sentences
    """

    def __init__(self, maxlen=128):
        self._routes = deque(maxlen=maxlen)
        self._maxlen = maxlen

    # ------------------------------------------------------------------
    # Internal helpers  (read-only)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_operation(task_type):
        """Map task_type to operation_type."""
        mapping = {
            'high_priority_stabilization': 'inspect_and_adjust',
            'focused_continuation':        'continue_thread',
            'exploratory_probe':           'compare_paths',
            'background_monitoring':       'observe_only',
            'research_expansion':          'hypothesis_expand',
        }
        return mapping.get(task_type, 'observe_only')

    @staticmethod
    def _compute_confidence(priority, narrative_memory):
        """
        Blend task priority (0.80) with simple recency continuity (0.20).
        Continuity assumes stable context if the recent narrative arc's
        dominant concept roughly persists or confidence is stable.
        For exact match to requirements, we'll use recent mean_confidence
        as the continuity proxy.
        """
        arcs = narrative_memory.recent(1)
        continuity = arcs[0].get('mean_confidence', 0.50) if arcs else 0.50
        raw = (priority * 0.80) + (continuity * 0.20)
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _compose_route_summary(operation_type):
        """Compose a 2-sentence operation intent note based on mapping."""
        s1_map = {
            'inspect_and_adjust': "The active cognitive task directs structural modifications via targeted inspection operations.",
            'continue_thread':    "The active cognitive task should continue through thread-preserving operations on the current dominant motif.",
            'compare_paths':      "The active cognitive task initiates parallel comparisons across divergent conceptual paths.",
            'observe_only':       "The active cognitive task requires purely observational processing with no structural interventions.",
            'hypothesis_expand':  "The active cognitive task launches generative hypothesis formulation to expand the semantic frontier.",
        }
        s1 = s1_map.get(operation_type, s1_map['observe_only'])

        s2 = "Recent task persistence and elevated priority suggest this route remains the most coherent continuation path."
        return f"{s1} {s2}"

    # ------------------------------------------------------------------
    # Main route call  (waking-branch only)
    # ------------------------------------------------------------------

    def update(self, state):
        """Unified interface (Day 32)."""
        return self._route(
            tick             = state.tick,
            task_framework   = state.task_framework,
            narrative_memory = state.narrative_memory,
        )

    def _route(self, tick, task_framework, narrative_memory):
        """
        Derive an operation intent from the current active task.

        MUST be called only from 'if not sleeping:' branch, right after
        TaskFramework.spawn().
        """
        task = task_framework.latest()
        if task is None:
             return None

        task_type  = task.get('task_type', 'background_monitoring')
        priority   = task.get('priority', 0.0)

        operation  = self._resolve_operation(task_type)
        confidence = self._compute_confidence(priority, narrative_memory)
        summary    = self._compose_route_summary(operation)

        record = {
            'tick':             int(tick),
            'operation_type':   operation,
            'task_origin':      task_type,
            'route_confidence': float(round(confidence, 4)),
            'route_summary':    summary,
        }
        self._routes.append(record)
        return record

    def latest(self):
        return self._routes[-1] if self._routes else None

    def recent(self, n=10):
        buf = list(self._routes)
        return buf[-n:] if n < len(buf) else buf

    def __len__(self):
        return len(self._routes)

    def to_dict(self):
        return {
            'maxlen': self._maxlen,
            'routes': list(self._routes),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(maxlen=int(d.get('maxlen', 128)))
        for rec in d.get('routes', []):
            obj._routes.append(rec)
        return obj
