"""
DEPRECATED (Day 77 Pack 278 v0).  Substrate-native reasoning is now Pack
254 RHC (math) + Pack 255 GeneralReasoner (compositional) + Pack 273
anchor-action cache + Pack 276 Cat4Dopamine.  This module's hardcoded
OPERATOR_LEXICON violates the "no hardcoding" rule (Day 75 close);
Phase 5 of integrate.read_statement still calls into it but is queued
for deletion in Pack 278 v1 (Day 78) after bench gate confirms
removal does not regress 25/25.  DO NOT add new call-sites.  DO NOT
extend OPERATOR_LEXICON.  See `c:/neuroseed/MEMORY.md` Pack 278 entry
and the Day 77 research log for the migration plan.

ikigai.cognition.reasoning_engine -- Generative Reasoning Core.

Day 56 Pack 93 -- the missing capability: ACTUALLY REASON, not retrieve.

When organism reads "Janet has 5 apples. She ate 2.", brain must:
    1. PARSE: extract entities, verbs, quantities, objects
    2. BIND:  store variables in working memory
    3. RESOLVE: pronouns -> last-mentioned entity
    4. EXECUTE: apply verb's operator to variable
    5. OUTPUT: respond to query

Architecture maps to brain regions:
    Wernicke's area  -- parse_statement: SVO extraction
    Hippocampus      -- episodic chain (statements remembered in order)
    Prefrontal       -- variable working memory (PiK-bound)
    Basal ganglia    -- operator lexicon (verb -> operator)
    Cerebellum       -- multi-statement chaining (sequential update)
    Broca's          -- generate(query) -> answer

This is the generative core that lets Ikigai THINK, not just retrieve.
"""

import re
from collections import deque

# Pack 278 v1 (Day 77): OPERATOR_LEXICON purged.  Substrate-native
# operator semantics now emerge via Pack 253 cat-3 absorb + Pack 254
# RHC + Pack 291 multiplicative ⋆ binding.  Day 75 "no hardcoding"
# rule.  Empty dict preserved so existing call-sites in this module
# (`tok in OPERATOR_LEXICON`) degrade to "False" without raising.
OPERATOR_LEXICON = {}

# Query phrases
QUERY_MARKERS = [
    'how many', 'how much', 'what is', "what's", 'how old',
    'find', 'calculate', 'compute', 'total',
]

# Pronouns -> resolve to last named entity
PRONOUNS = {'she', 'he', 'it', 'they', 'her', 'him', 'them', 'his', 'hers'}


# ── token utilities ──────────────────────────────────────────────────────────

def tokenize(text):
    """Lowercase, strip punct except $, %, ?."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[\?\$%]", text.lower())


def extract_number(tokens, i):
    """Return float at tokens[i] or None."""
    if i < len(tokens):
        try: return float(tokens[i])
        except ValueError: return None
    return None


# ── Working memory ───────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Holds variable bindings: (entity, object) -> value.

    Mental model: 'Janet has 5 apples' -> ('janet', 'apples') = 5.
    'She ate 2' -> ('janet', 'apples') -= 2.
    """

    def __init__(self):
        self._vars = {}              # (entity, obj) -> value
        self._last_entity = None     # for pronoun resolution
        self._last_object = None     # for elided object resolution
        self._history    = []        # episodic chain of events

    def set(self, entity, obj, value):
        self._vars[(entity, obj)] = float(value)
        self._last_entity = entity
        self._last_object = obj
        self._history.append(('SET', entity, obj, value))

    def add(self, entity, obj, value):
        key = (entity, obj)
        if key not in self._vars:
            self._vars[key] = 0.0
        self._vars[key] += float(value)
        self._last_entity = entity
        self._last_object = obj
        self._history.append(('ADD', entity, obj, value))

    def sub(self, entity, obj, value):
        key = (entity, obj)
        if key not in self._vars:
            return None
        self._vars[key] -= float(value)
        self._last_entity = entity
        self._last_object = obj
        self._history.append(('SUB', entity, obj, value))
        return self._vars[key]

    def mul(self, entity, obj, value):
        key = (entity, obj)
        if key not in self._vars:
            self._vars[key] = 1.0
        self._vars[key] *= float(value)
        self._last_entity = entity
        self._last_object = obj
        self._history.append(('MUL', entity, obj, value))

    def get(self, entity, obj):
        return self._vars.get((entity, obj))

    def resolve_pronoun(self, token):
        if token in PRONOUNS:
            return self._last_entity
        return token

    def resolve_object(self, obj):
        return obj if obj else self._last_object

    def all_values(self):
        return dict(self._vars)

    def history(self):
        return list(self._history)


# ── Sentence parser ──────────────────────────────────────────────────────────

class Statement:
    """One parsed statement: entity, op, value, obj."""
    __slots__ = ('entity', 'op', 'valence', 'value', 'obj', 'raw')

    def __init__(self, entity, op, valence, value, obj, raw):
        self.entity   = entity
        self.op       = op
        self.valence  = valence
        self.value    = value
        self.obj      = obj
        self.raw      = raw

    def __repr__(self):
        return (f"<Statement entity={self.entity!r} op={self.op} "
                f"value={self.value} obj={self.obj!r}>")


class ReasoningParser:
    """
    Parse natural-language statements into Statement structs.
    Uses operator lexicon + simple positional rules.

    Limitations: handles SVO-ish English of GSM8K complexity.
    Not full parser. Genuine generative reasoning over parsed bindings.
    """

    def __init__(self):
        self._known_entities = set()    # learned during parsing

    def parse_statement(self, sentence):
        tokens = tokenize(sentence)
        if not tokens:
            return None

        # Find first capitalized-style entity (or pronoun)
        # We lowercased everything, so use first noun-like token
        # Detect entity: first non-stopword token that's not number/op
        STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'so', 'if',
                     'then', 'now', 'at', 'in', 'on', 'of', 'for', 'with',
                     'more', 'less', 'fewer', 'away', 'to', 'from', 'by',
                     'friend', 'friends'}
        entity = None
        for tok in tokens:
            if tok in PRONOUNS:
                entity = tok
                break
            if (tok not in STOPWORDS and not tok.isdigit() and
                tok not in OPERATOR_LEXICON and tok not in QUERY_MARKERS and
                len(tok) > 1):
                entity = tok
                self._known_entities.add(tok)
                break

        # Find operator verb
        op = None
        op_type = None
        valence = None
        op_pos = -1
        for i, tok in enumerate(tokens):
            if tok in OPERATOR_LEXICON:
                op, (op_type, valence) = tok, OPERATOR_LEXICON[tok]
                op_pos = i
                break

        # Find quantity (first number)
        value = None
        value_pos = -1
        for i, tok in enumerate(tokens):
            try:
                value = float(tok)
                value_pos = i
                break
            except ValueError:
                continue

        # Find object (noun after value or after op, NOT entity NOR known-entity)
        obj = None
        if value_pos >= 0:
            search_start = value_pos + 1
        elif op_pos >= 0:
            search_start = op_pos + 1
        else:
            search_start = 0
        for tok in tokens[search_start:]:
            if (tok not in STOPWORDS and tok not in PRONOUNS and
                tok != entity and tok not in OPERATOR_LEXICON and
                not tok.replace('.', '').isdigit() and
                tok not in QUERY_MARKERS and
                tok not in {'?', '$', '%'} and
                tok not in self._known_entities):   # NEW: skip known entities
                obj = tok
                break

        if op is None or value is None:
            return None

        return Statement(entity, op_type, valence, value, obj, sentence)

    def parse_query(self, sentence):
        """Detect query. Returns (target_obj, target_entity) or None."""
        tokens = tokenize(sentence)
        text_lower = ' '.join(tokens)
        if not any(qm in text_lower for qm in QUERY_MARKERS):
            return None
        # Pick last meaningful noun as target object
        STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'does', 'have', 'has',
                     'left', 'remain', 'now', 'is', 'are', 'how', 'many',
                     'much', 'what', 'does'}
        target_obj = None
        target_entity = None
        for tok in tokens:
            if tok in self._known_entities:
                target_entity = tok
            elif (tok not in STOPWORDS and not tok.replace('.', '').isdigit() and
                  tok not in OPERATOR_LEXICON and len(tok) > 1 and
                  tok not in PRONOUNS and tok not in {'?', '$', '%'}):
                target_obj = tok
        return target_obj, target_entity


# ── Reasoning engine (the core) ──────────────────────────────────────────────

class ReasoningEngine:
    """
    Generative reasoning core. Pipeline:
        sentences -> parsed Statements -> apply to WorkingMemory -> query -> answer

    Brain-region mapping:
        parser        -> Wernicke's area (language comprehension)
        working_memory -> prefrontal cortex (active variable bindings)
        operator_apply -> basal ganglia (action selection from verb)
        episodic_log  -> hippocampus (event order)
        query         -> Broca's area (production)
    """

    def __init__(self):
        self.parser = ReasoningParser()
        self.wm     = WorkingMemory()

    def reset(self):
        self.parser = ReasoningParser()
        self.wm     = WorkingMemory()

    def read_statement(self, sentence):
        """Parse + apply one statement. Returns (Statement, updated_value)."""
        stmt = self.parser.parse_statement(sentence)
        if stmt is None:
            return None, None

        # Resolve entity (pronoun -> last named)
        entity = self.wm.resolve_pronoun(stmt.entity)

        # Resolve object (None -> last named)
        obj = self.wm.resolve_object(stmt.obj)

        # Apply operator
        if stmt.op == 'SET':
            self.wm.set(entity, obj, stmt.value)
        elif stmt.op == 'ADD':
            self.wm.add(entity, obj, stmt.value)
        elif stmt.op == 'SUB':
            self.wm.sub(entity, obj, stmt.value)
        elif stmt.op == 'MUL':
            self.wm.mul(entity, obj, stmt.value)

        return stmt, self.wm.get(entity, obj)

    def answer_query(self, sentence):
        """Parse query, retrieve from working memory."""
        result = self.parser.parse_query(sentence)
        if result is None:
            return None
        target_obj, target_entity = result

        # If both target obj and entity specified: return exact binding
        if target_obj and target_entity:
            return self.wm.get(target_entity, target_obj)

        # If only obj specified: return value across any entity that has it
        if target_obj:
            for (ent, ob), v in self.wm.all_values().items():
                if ob == target_obj:
                    return v

        # If only entity: return any binding for that entity
        if target_entity:
            for (ent, ob), v in self.wm.all_values().items():
                if ent == target_entity:
                    return v

        # Last fallback: most recently updated value
        if self.wm._last_entity and self.wm._last_object:
            return self.wm.get(self.wm._last_entity, self.wm._last_object)
        return None

    def reason(self, text):
        """
        End-to-end: split text into sentences, process each.
        Returns (trace, answer).
        trace = list of (sentence, parsed_stmt, post_value).
        answer = result of last query (or last variable value if no query).
        """
        # Split into sentences (simple period split)
        sentences = [s.strip() for s in re.split(r'[\.\!\?]+', text) if s.strip()]

        trace = []
        answer = None
        for sent in sentences:
            # Is it a query?
            qry = self.parser.parse_query(sent)
            if qry is not None:
                # Process as query
                answer = self.answer_query(sent)
                trace.append((sent, ('QUERY', qry), answer))
                continue
            # Process as statement
            stmt, val = self.read_statement(sent)
            trace.append((sent, stmt, val))
            if val is not None:
                answer = val

        return trace, answer
