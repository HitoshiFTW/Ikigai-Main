"""
ikigai.conversation_kernel_io -- Text I/O Bridge for ConversationKernel.

Day 55 Pack 37 -- bridges raw text to/from the conversational substrate.

Pipeline per turn:
    text -> tokenize -> ConversationKernel.step() -> decode B_U -> grammar prefix -> text

Encoding (text -> kernel):
    tokenize(text) -> [tokens]
    kernel.step(tokens) -> KernelOutput (B_U, grammar_name, action, F_t, ...)

Decoding (kernel -> text):
    B_U = semantic belief vector (d_sem = 64)
    vocab[word] = deterministic bipolar HV per word, L2-normalized
    content = top_k words by cosine(B_U, word_hv)
    response = GRAMMAR_PREFIX[grammar_name] + " " + " ".join(content[:3])

Vocabulary:
    - BASE_VOCAB: 100 pre-seeded common/domain words
    - Online: every user word added automatically on first mention
    - Storage: O(vocab_size * d_sem) bytes

No gradient. No LLM. No templates looked up in a database.
Every response is geometrically determined from user belief state.
"""

import re
import numpy as np

from ikigai.conversation_kernel import ConversationKernel


# -----------------------------------------------------------------------
# Grammar-typed response prefixes (16 dialogue-act nodes)
# -----------------------------------------------------------------------

GRAMMAR_PREFIX = {
    'GREET':      'Hello,',
    'IDENTIFY':   'I am',
    'QUERY':      'Can you tell me about',
    'CLARIFY':    'To clarify,',
    'STATEMENT':  'The',
    'AFFIRM':     'Yes,',
    'NEGATE':     'No,',
    'CONJUNCT':   'And also,',
    'CAUSE':      'Because',
    'EFFECT':     'Therefore,',
    'EXAMPLE':    'For example,',
    'COMPARE':    'Similarly,',
    'CONCLUDE':   'In conclusion,',
    'QUESTION':   'Do you think',
    'ACKNOWLEDGE':'I understand.',
    'FAREWELL':   'Goodbye.',
}

# -----------------------------------------------------------------------
# Base vocabulary (pre-seeded, 100 words)
# -----------------------------------------------------------------------

BASE_VOCAB = [
    # greetings / closings
    'hello', 'hi', 'goodbye', 'bye', 'welcome', 'greet', 'hey',
    # interrogatives
    'what', 'how', 'why', 'when', 'where', 'who', 'which',
    'can', 'could', 'would', 'should', 'does', 'did', 'will',
    # affirmations / negations
    'yes', 'correct', 'right', 'agree', 'true', 'certainly', 'exactly',
    'no', 'not', 'wrong', 'incorrect', 'disagree', 'false', 'never',
    # connectives / discourse
    'and', 'but', 'because', 'therefore', 'so', 'also', 'however',
    'thus', 'hence', 'furthermore', 'moreover', 'meanwhile', 'then',
    # epistemic
    'think', 'know', 'understand', 'believe', 'feel', 'seem', 'appear',
    'consider', 'assume', 'expect',
    # tech/AI domain
    'neural', 'network', 'vector', 'symbolic', 'cognitive', 'algorithm',
    'code', 'function', 'system', 'architecture', 'model', 'learning',
    'data', 'pattern', 'memory', 'compute', 'inference', 'output',
    'layer', 'embedding', 'attention', 'transformer', 'token', 'context',
    'gradient', 'weight', 'tensor', 'matrix', 'dimension',
    # generic content
    'time', 'information', 'process', 'result', 'state', 'value',
    'problem', 'solution', 'method', 'approach', 'example', 'case',
    'structure', 'behaviour', 'property', 'relation', 'concept', 'idea',
    # conversational meta
    'question', 'answer', 'response', 'topic', 'meaning', 'intent',
    'message', 'context', 'turn', 'conversation', 'dialogue',
]


# -----------------------------------------------------------------------
# Text I/O Bridge
# -----------------------------------------------------------------------

def _word_hv(word, d):
    """Deterministic L2-normalized bipolar HV per word.
    MUST use same hash prefix as persona_manifold._token_hv ('bspm::')
    so vocab HVs live in the same space as B_U for valid cosine decoding.
    """
    seed = hash(f'bspm::{word}') & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    v = (rng.randint(0, 2, size=d) * 2 - 1).astype(np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _cosine_dot(a, b):
    """Fast cosine when both vectors are already L2-normalized."""
    return float(np.dot(a, b))


class ConversationKernelIO:
    """
    Text-level I/O wrapper around ConversationKernel.

    Vocab is built online (every new user token added automatically).
    BASE_VOCAB pre-seeded at init for cold-start decoding quality.

    respond(text) -> (response_str, KernelOutput)
    """

    def __init__(self, seed=42, top_k=5):
        self.kernel = ConversationKernel(seed=seed)
        self.top_k  = top_k
        self._d     = self.kernel.bspm.d   # semantic dimension (=64 default)
        self._vocab = {}                    # word -> L2-normalized HV
        self.log    = []                    # list of turn dicts
        self._turn  = 0

        # Pre-seed base vocabulary
        for w in BASE_VOCAB:
            self._register(w)

    def _register(self, word):
        if word not in self._vocab:
            self._vocab[word] = _word_hv(word, self._d)
        return self._vocab[word]

    # ─── tokenizer ───────────────────────────────────────────────

    def tokenize(self, text):
        """Lowercase, strip punctuation, split on non-alpha-numeric."""
        return re.findall(r"[a-z0-9']+", text.lower())

    # ─── decoder ─────────────────────────────────────────────────

    def decode_to_words(self, belief_hv, top_k=None):
        """
        Find top_k vocab words with highest cosine to belief_hv.
        belief_hv should be L2-normalized (from BSPM.B_U).
        """
        k = top_k if top_k is not None else self.top_k
        # dot product = cosine when both normalized
        scores = [(w, _cosine_dot(belief_hv, hv)) for w, hv in self._vocab.items()]
        scores.sort(key=lambda x: -x[1])
        return [w for w, _ in scores[:k]]

    # ─── respond ─────────────────────────────────────────────────

    def respond(self, user_text, role='user'):
        """
        Full round-trip: raw text -> kernel -> decoded text response.

        Returns (response_str, KernelOutput).
        """
        tokens = self.tokenize(user_text)
        if not tokens:
            tokens = ['empty']

        # Register new tokens online
        for t in tokens:
            self._register(t)

        # Run kernel
        out = self.kernel.step(tokens, role=role)

        # Decode belief to content words (exclude stopwords if desired)
        content_words = self.decode_to_words(out.B_U, top_k=self.top_k)

        # Grammar prefix
        prefix = GRAMMAR_PREFIX.get(out.grammar_name, '')

        # Assemble response: prefix + up to 3 content words
        response_parts = [prefix] + content_words[:3] if prefix else content_words[:3]
        response = ' '.join(response_parts).strip()

        self._turn += 1
        entry = {
            'turn':          self._turn,
            'input':         user_text,
            'tokens':        tokens,
            'grammar':       out.grammar_name,
            'action':        out.action,
            'F_t':           out.F_t,
            'content_words': content_words,
            'response':      response,
            'belief_drift':  self.kernel.bspm.belief_drift(),
            'delta_cert':    out.delta_cert,
        }
        self.log.append(entry)
        return response, out

    # ─── multi-turn demo ─────────────────────────────────────────

    def run_dialogue(self, exchanges):
        """
        exchanges: list of (role, text) pairs.
        Returns list of (text, response, KernelOutput) tuples.
        """
        results = []
        for role, text in exchanges:
            response, out = self.respond(text, role=role)
            results.append((text, response, out))
        return results

    # ─── state ───────────────────────────────────────────────────

    @property
    def vocab_size(self):
        return len(self._vocab)

    def summary(self):
        kernel_s = self.kernel.state_summary()
        return {
            **kernel_s,
            'vocab_size': self.vocab_size,
            'io_turns':   self._turn,
        }

    def reset(self, seed=42):
        self.kernel.reset(seed=seed)
        self._turn = 0
        self.log.clear()
        # Keep vocab (online learning persists)
