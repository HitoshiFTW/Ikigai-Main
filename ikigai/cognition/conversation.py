"""
ikigai.cognition.conversation — General-purpose conversational pipeline.

Binary routing: code queries -> CodeAlpaca kNN, everything else -> General corpus kNN.
No synthetic domain data. Responses come from real corpus retrieval + local n-gram.
Multi-turn context via EpisodicBuffer (conversation history shifts retrieval).

Usage:
    from ikigai.cognition.conversation import Conversation, build_conversation
    conv = build_conversation(ctx, code_corpus_path, general_corpus_path)
    response = conv.chat("what is machine learning")
    response = conv.chat("write a function to sort a list")
"""

import re, math, json, time
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokenizer (mirrors ctx.tokenize but standalone for use before ctx built)
# ---------------------------------------------------------------------------
_SIMPLE_TOK = re.compile(r"[a-z0-9']+|[.,!?;:()\[\]{}/\\\"<>=+\-*&|^%$#@~`]")

def _tokenize(text):
    return _SIMPLE_TOK.findall(text.lower())


# ---------------------------------------------------------------------------
# Code intent keywords (fast check before HV comparison)
# ---------------------------------------------------------------------------
_CODE_WORDS = {
    'write', 'implement', 'create', 'build', 'code', 'function', 'def',
    'class', 'algorithm', 'program', 'script', 'python', 'javascript',
    'sql', 'loop', 'sort', 'search', 'parse', 'debug', 'fix', 'snippet',
    'method', 'module', 'library', 'api', 'recursive', 'iterate',
}


# ---------------------------------------------------------------------------
# Arithmetic (safe eval for math queries)
# ---------------------------------------------------------------------------
_SAFE_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
_SAFE_NS.update({'abs': abs, 'round': round, 'min': min, 'max': max,
                 'int': int, 'float': float, 'pow': pow})

_EXPR_RE = re.compile(
    r'(?<![a-zA-Z_])'
    r'(?:'
    r'[0-9]+(?:\.[0-9]+)?(?:\s*[\+\-\*\/\%\^]\s*[0-9]+(?:\.[0-9]+)?)+'
    r'|(?:sqrt|log|sin|cos|tan|floor|ceil|abs)\s*\([^)]+\)'
    r'|\([^)]+\)\s*[\+\-\*\/\%]\s*[0-9]+'
    r')',
    re.IGNORECASE
)

def _safe_eval(expr):
    # DEPRECATED Pack 278 v0.  Math now resolved substrate-native via
    # Pack 254 RHC (additive + Pack 291 multiplicative ⋆ planned).
    # Kept while IntentRouter math branch in this file still falls
    # through; both routes scheduled for deletion in Pack 278 v1.
    expr = expr.strip().replace('^', '**')
    for b in ['import', '__', 'open(', 'exec(', 'eval(', 'globals(', 'locals(']:
        if b in expr:
            return None
    try:
        r = eval(expr, {"__builtins__": {}}, _SAFE_NS)
        return float(r) if isinstance(r, (int, float)) and math.isfinite(r) else None
    except Exception:
        return None

def _detect_math(text):
    """Return (expr_str, value) or (None, None)."""
    for m in _EXPR_RE.finditer(text):
        v = _safe_eval(m.group())
        if v is not None:
            return m.group(), v
    for m in re.finditer(r'\b(\d+)\s*([\+\-\*\/\%])\s*(\d+)\b', text):
        v = _safe_eval(m.group().replace(' ', ''))
        if v is not None:
            return m.group(), v
    return None, None


# ---------------------------------------------------------------------------
# kNN retrieval + local N-gram generation
# ---------------------------------------------------------------------------
def _retrieve(query_hv, matrix, seqs, k=20):
    """Hamming-distance kNN. Returns [(sim, seq), ...]."""
    q = np.frombuffer(query_hv, dtype=np.uint8)
    sims = 1.0 - (matrix != q).mean(axis=1)
    k = min(k, len(sims) - 1)
    idx = np.argpartition(-sims, k)[:k + 1]
    idx = idx[np.argsort(-sims[idx])][:k]
    return [(float(sims[i]), seqs[i]) for i in idx]


def _build_ngram(retrieved):
    """Build bigram + trigram dicts from retrieved sequences."""
    bg, tg = {}, {}
    for _, seq in retrieved:
        for i in range(len(seq) - 1):
            p, c = seq[i], seq[i+1]
            bg.setdefault(p, {})[c] = bg.get(p, {}).get(c, 0) + 1
            if i >= 1:
                key = (seq[i-1], p)
                tg.setdefault(key, {})[c] = tg.get(key, {}).get(c, 0) + 1
    return bg, tg


def _generate(bg, tg, q_context, seed=None, max_len=25, thresh=2):
    """Generate answer tokens from local n-gram, optionally seeded."""
    prefix = list(q_context[-3:]) if len(q_context) >= 3 else list(q_context)
    result = []
    seen = set()
    for i in range(len(q_context) - 3):
        seen.add(tuple(q_context[i:i+4]))

    if seed:
        if len(prefix) >= 3:
            seen.add(tuple(prefix[-3:]) + (seed,))
        result.append(seed)
        prefix.append(seed)

    for _ in range(max_len - len(result)):
        cands = {}
        if len(prefix) >= 2:
            key = (prefix[-2], prefix[-1])
            if key in tg and sum(tg[key].values()) >= thresh:
                cands = tg[key]
        if not cands and prefix:
            prev = prefix[-1]
            if prev in bg and sum(bg[prev].values()) >= thresh:
                cands = bg[prev]
        if not cands:
            break
        tok = None
        for cand, _ in sorted(cands.items(), key=lambda x: -x[1]):
            if cand in ('A', 'Q'):
                continue
            if len(prefix) >= 3:
                fg = tuple(prefix[-3:]) + (cand,)
                if fg in seen:
                    continue
            tok = cand
            break
        if tok is None:
            break
        if len(prefix) >= 3:
            seen.add(tuple(prefix[-3:]) + (tok,))
        result.append(tok)
        prefix.append(tok)
    return result


# ---------------------------------------------------------------------------
# IntentRouter — binary: code vs general
# ---------------------------------------------------------------------------
class IntentRouter:
    """
    DEPRECATED Pack 278 v0.  Hardcoded code/general split.  Substrate
    routes via Pack 255 GeneralReasoner which composes math/code/lang
    uniformly without branching.  Scheduled for deletion Pack 278 v1.

    Classifies query as 'code' or 'general' using two signals:
    1. Keyword check (fast): any _CODE_WORDS token in query -> code
    2. HV similarity (fallback): TF-IDF query HV vs domain archetype HVs
    """

    _CODE_ARCHETYPES = [
        "write function python implement algorithm",
        "def class method return loop",
        "code snippet program script",
        "sort search binary tree recursion",
        "debug fix bug error exception",
        "sql query database select insert",
        "javascript html css frontend",
        "api endpoint request response",
    ]
    _GEN_ARCHETYPES = [
        "what is explain describe overview",
        "how does it work science history",
        "tell me about nature universe",
        "why is reason cause effect",
        "define meaning concept theory",
        "compare difference between",
        "summarize summary key points",
        "opinion advice recommendation",
    ]

    def __init__(self, ctx):
        self.ctx = ctx
        self._has_idf = bool(ctx.token_idf)
        self._code_hv = self._bundle(self._CODE_ARCHETYPES)
        self._gen_hv  = self._bundle(self._GEN_ARCHETYPES)

    def _embed(self, tokens):
        if self._has_idf:
            return self.ctx.tfidf_bow_hv(tokens, alpha=0.5)
        dim = self.ctx.token_hv_dim
        counts = np.zeros(dim, dtype=np.int32)
        for t in tokens:
            hv = self.ctx.token_hv_for(t)
            counts += np.frombuffer(hv, dtype=np.uint8).astype(np.int32)
        n = len(tokens)
        if n == 0:
            return bytearray(dim)
        return bytearray((counts > n / 2).astype(np.uint8).tobytes())

    def _bundle(self, phrases):
        hvs = [self._embed(_tokenize(p)) for p in phrases]
        return bytearray(self.ctx.vsa_bundle(hvs))

    def classify(self, text):
        """Returns ('code'|'general', {'code': sim, 'general': sim})."""
        toks = _tokenize(text)
        # Fast keyword check
        if any(t in _CODE_WORDS for t in toks):
            q_hv = self._embed(toks)
            code_sim = self.ctx.vsa_cosine(q_hv, self._code_hv)
            gen_sim  = self.ctx.vsa_cosine(q_hv, self._gen_hv)
            # Only classify as code if code similarity is meaningfully higher
            if code_sim >= gen_sim - 0.02:
                return 'code', {'code': code_sim, 'general': gen_sim}
        q_hv = self._embed(toks)
        code_sim = self.ctx.vsa_cosine(q_hv, self._code_hv)
        gen_sim  = self.ctx.vsa_cosine(q_hv, self._gen_hv)
        domain = 'code' if code_sim > gen_sim else 'general'
        return domain, {'code': code_sim, 'general': gen_sim}


# ---------------------------------------------------------------------------
# EpisodicBuffer — conversation history shifts retrieval
# ---------------------------------------------------------------------------
class EpisodicBuffer:
    """
    Stores last N turns as HVs. context_retrieve() blends query sim with
    context sim so follow-up questions (e.g. "what about venus?") pull
    toward the conversation domain even without explicit keywords.
    """

    def __init__(self, ctx, capacity=8):
        self.ctx = ctx
        self.capacity = capacity
        self.dim = ctx.token_hv_dim
        self.turns = []
        self.context_hv = bytearray(self.dim)

    def add_turn(self, q_toks, a_toks):
        q_hv = self.ctx.tfidf_bow_hv(q_toks, alpha=0.5)
        a_hv = self.ctx.tfidf_bow_hv(a_toks, alpha=0.5)
        turn_hv = bytearray(self.ctx.vsa_bind(q_hv, a_hv))
        self.turns.append(turn_hv)
        if len(self.turns) > self.capacity:
            self.turns.pop(0)
        if self.turns:
            self.context_hv = bytearray(self.ctx.vsa_bundle(self.turns))

    def retrieve(self, q_toks, matrix, seqs, k=20, blend=0.25):
        q_hv = self.ctx.tfidf_bow_hv(q_toks, alpha=0.5)
        q = np.frombuffer(q_hv, dtype=np.uint8)
        q_sims = 1.0 - (matrix != q).mean(axis=1)
        if blend > 0.0 and any(b != 0 for b in self.context_hv):
            c = np.frombuffer(self.context_hv, dtype=np.uint8)
            c_sims = 1.0 - (matrix != c).mean(axis=1)
            scores = (1.0 - blend) * q_sims + blend * c_sims
        else:
            scores = q_sims
        k = min(k, len(scores) - 1)
        idx = np.argpartition(-scores, k)[:k + 1]
        idx = idx[np.argsort(-scores[idx])][:k]
        return [(float(scores[i]), seqs[i]) for i in idx]

    def reset(self):
        self.turns = []
        self.context_hv = bytearray(self.dim)

    @property
    def turn_count(self):
        return len(self.turns)


# ---------------------------------------------------------------------------
# Conversation — the full pipeline
# ---------------------------------------------------------------------------
class Conversation:
    """
    General-purpose multi-turn conversational agent.

    Architecture per turn:
      query -> IntentRouter.classify()
            -> EpisodicBuffer.retrieve(appropriate corpus)
            -> _build_ngram(retrieved)
            -> _generate(local ngram, optional seed)
            -> buffer.add_turn(q, a)

    Code queries: retrieve from CodeAlpaca, generate code-like responses.
    General queries: retrieve from general corpus, generate natural language.
    Math queries: arithmetic eval -> seed numeric token -> general generation.
    """

    def __init__(self, ctx, router,
                 code_matrix, code_seqs,
                 gen_matrix,  gen_seqs,
                 capacity=8, k=20, blend=0.25, thresh=2):
        self.ctx         = ctx
        self.router      = router
        self.code_matrix = code_matrix
        self.code_seqs   = code_seqs
        self.gen_matrix  = gen_matrix
        self.gen_seqs    = gen_seqs
        self.buffer      = EpisodicBuffer(ctx, capacity=capacity)
        self.history     = []
        self.k           = k
        self.blend       = blend
        self.thresh      = thresh

    def chat(self, text, max_len=25, verbose=False):
        """
        Process one turn. Returns response string.
        Automatically routes to code or general corpus.
        Math expressions are computed and seeded as first token.
        """
        t0 = time.perf_counter()
        q_toks = _tokenize(text)
        q_ctx  = ['Q'] + q_toks + ['A']

        # 1. Classify intent
        domain, scores = self.router.classify(text)

        # 2. Arithmetic seed (for math expressions in any domain)
        seed = None
        _, math_val = _detect_math(text)
        if math_val is not None:
            seed = str(int(round(math_val)))
            domain = 'general'  # answer in natural language, not code

        # 3. Retrieve from appropriate corpus
        if domain == 'code':
            retrieved = self.buffer.retrieve(
                q_toks, self.code_matrix, self.code_seqs,
                k=self.k, blend=self.blend)
        else:
            retrieved = self.buffer.retrieve(
                q_toks, self.gen_matrix, self.gen_seqs,
                k=self.k, blend=self.blend)

        # 4. Build local n-gram + generate
        bg, tg = _build_ngram(retrieved)
        answer_toks = _generate(bg, tg, q_ctx, seed=seed,
                                max_len=max_len, thresh=self.thresh)

        # 5. Update episodic buffer
        self.buffer.add_turn(q_toks, answer_toks or ['none'])

        # 6. Record turn
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.history.append({
            'query':   text,
            'domain':  domain,
            'scores':  scores,
            'seed':    seed,
            'answer':  answer_toks,
            'ms':      elapsed_ms,
        })

        if verbose:
            ans = ' '.join(answer_toks[:15])
            q_short = repr(text[:40])
            a_short = repr(ans[:50])
            print(f"  [{domain}] {q_short} -> {a_short}  ({elapsed_ms:.1f}ms)")

        return ' '.join(answer_toks)

    def reset(self):
        """Clear conversation history and episodic buffer."""
        self.buffer.reset()
        self.history.clear()

    @property
    def turn_count(self):
        return len(self.history)


# ---------------------------------------------------------------------------
# Builder — loads corpora, builds HV matrices, wires everything
# ---------------------------------------------------------------------------
def build_conversation(ctx, code_corpus_path, general_corpus_path,
                       max_code=None, max_gen=None,
                       capacity=8, k=20, blend=0.25):
    """
    Load corpora, build HV matrices, construct Conversation.

    Args:
        ctx: IkigaiContext (call scale_for_benchmark + enable_semantic_hv before this)
        code_corpus_path: path to codealpaca_ikigai.jsonl
        general_corpus_path: path to general_ikigai.jsonl
        max_code: max sequences from code corpus (None = all)
        max_gen:  max sequences from general corpus (None = all)
    Returns:
        Conversation instance, ready to .chat()
    """
    def _load(path, maxn):
        seqs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                seqs.append(json.loads(line)['tokens'])
                if maxn and len(seqs) >= maxn:
                    break
        return seqs

    print(f"[Conversation] Loading code corpus ({code_corpus_path})...")
    code_seqs = _load(code_corpus_path, max_code)
    print(f"  {len(code_seqs)} code sequences")

    print(f"[Conversation] Loading general corpus ({general_corpus_path})...")
    gen_seqs = _load(general_corpus_path, max_gen)
    print(f"  {len(gen_seqs)} general sequences")

    # Build IDF on combined corpus for better TF-IDF weights
    print("[Conversation] Building IDF table...")
    ctx.compute_idf(code_seqs + gen_seqs)

    # Build HV matrices
    print("[Conversation] Building code HV matrix...")
    code_hvs = []
    for seq in code_seqs:
        toks = [t for t in seq if t not in ('Q', 'A')]
        hv = ctx.tfidf_bow_hv(toks, alpha=0.5)
        code_hvs.append(np.frombuffer(hv, dtype=np.uint8))
    code_matrix = np.stack(code_hvs, axis=0)

    print("[Conversation] Building general HV matrix...")
    gen_hvs = []
    for seq in gen_seqs:
        toks = [t for t in seq if t not in ('Q', 'A')]
        hv = ctx.tfidf_bow_hv(toks, alpha=0.5)
        gen_hvs.append(np.frombuffer(hv, dtype=np.uint8))
    gen_matrix = np.stack(gen_hvs, axis=0)

    # Build router (uses IDF already loaded into ctx)
    print("[Conversation] Building IntentRouter...")
    router = IntentRouter(ctx)

    print(f"[Conversation] Ready. code={code_matrix.shape} gen={gen_matrix.shape}")
    return Conversation(ctx, router, code_matrix, code_seqs,
                        gen_matrix, gen_seqs,
                        capacity=capacity, k=k, blend=blend)
