"""
ikigai.cognition.generation_engine -- flat-memory generation engine.

Day 58 Pack 135. A novel generation architecture that:
  - decouples THINK from SPEAK (LLMs conflate them in one forward pass)
  - lives entirely on the flat-memory substrate (no context window)
  - has O(1) per-token RAM cost (no growing KV cache)
  - exposes chain-of-thought as a first-class observable trace
  - integrates new facts mid-generation (online learning)

Core mechanics:
  THOUGHT-STATE  T_t in C^d:
    a single phasor HV that evolves over generation steps.

  THINK step (associative walk through flat memory):
    addr  = T_t (*) ROLE_cooccur          (treat thought as substrate query)
    resp  = sdm.read(addr)                (superposed cooccur context)
    T_{t+1} = renorm( momentum * T_t + (1-momentum) * resp )

  SPEAK step (emit next token):
    cands = next_word_candidates(last_token, top_k)
    score_w = base_w * exp(gamma * cos(key(w), T_t))
    sample by softmax(scores / temperature)

  LOOP:
    for _ in range(think_steps): think()
    speak() -> append token
    repeat for max_tokens

The thought is the running semantic anchor.  Bigram alone drifts after
6-8 tokens; thought-weighted bigram stays anchored across hundreds.

Multi-modal extension: any HV (image, sensor, scalar) can be mixed into
thought via the same momentum equation -> the engine thinks across
modalities natively.
"""

import re
import random
import numpy as np


def _tokenize(text):
    return [t for t in re.sub(r"[^a-z0-9'\s]", ' ', text.lower()).split() if t]


def _is_junk_token(tok):
    """Pack 192 v1.1 -- reject code-y / non-linguistic candidates.
    Heuristics (no English vocab): digit, length>14, low alpha ratio,
    mid-word uppercase (camel/PascalCase)."""
    if not tok or len(tok) > 14:
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    n_alpha = sum(1 for ch in tok if ch.isalpha())
    if n_alpha == 0 or n_alpha / len(tok) < 0.7:
        return True
    if any(c.isupper() for c in tok[1:]):
        return True
    return False


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class GenerationEngine:
    """
    Flat-memory generation engine.  Construct via IkigaiOrganism.cogitate(...)
    or instantiate directly with an organism reference.

    Parameters:
      organism             - IkigaiOrganism with populated unified memory
      think_steps          - thought-walk steps per emitted token (default 3)
      momentum             - thought retention vs new association (0..1, default 0.7)
      thought_gamma        - alignment boost in speak (higher = thought dominates)
      temperature          - softmax temperature on speak sampling
      top_k                - bigram candidates considered per speak step
      remove_common        - if True, mean-remove the global discourse direction
                              from speak scoring (sharper topic preservation)
      ngram_weights        - (w_bi, w_tri, w_4gram) weights for combined
                              n-gram backoff scoring (Pack 136). Set to
                              (1.0, 0.0, 0.0) to fall back to bigram-only.
      ngram_ctx            - how many trailing tokens to feed the combined
                              n-gram scorer (default 3 = up to 4-gram).
      goal_gamma           - alignment strength to a FIXED goal HV (Pack 142).
                              Goal is the initial prompt HV and does NOT drift.
                              Thought drifts via momentum; goal anchors the
                              speak step toward prompt-topic across long gen.
                              0.0 = off (pre-Pack-142 behaviour).
      grounded_gamma       - Pack 146: strength of grounded-semantics boost.
                              Pulls speak toward candidates that share isa
                              parents / properties with the last token, when
                              such meaning is stored in the substrate. Falls
                              back gracefully for words with no meaning data.
                              0.0 = off (pre-Pack-146 behaviour).
      grounded_roles       - which roles to consult for grounded scoring.
                              Default ('isa', 'property'). Each role's
                              recall is summed into a meaning context HV.
    """

    def __init__(self, organism, think_steps=3, momentum=0.7,
                 thought_gamma=4.0, temperature=0.7, top_k=20,
                 remove_common=True, ngram_weights=(0.2, 0.4, 0.4),
                 ngram_ctx=3, goal_gamma=0.0,
                 grounded_gamma=0.0, grounded_roles=('isa', 'property')):
        self.org = organism
        self.think_steps = int(think_steps)
        self.momentum = float(momentum)
        self.thought_gamma = float(thought_gamma)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.remove_common = bool(remove_common)
        self.ngram_weights = tuple(ngram_weights)
        self.ngram_ctx = int(ngram_ctx)
        self.goal_gamma = float(goal_gamma)
        self.grounded_gamma = float(grounded_gamma)
        self.grounded_roles = tuple(grounded_roles)
        self.thought = None
        self.goal = None
        self.thought_trace = []
        self.history = []
        self._rng = random.Random()

    # ── initialization ────────────────────────────────────────────────────
    def _init_thought(self, prompt_tokens):
        d = self.org.unified.d
        if not prompt_tokens:
            self.thought = np.ones(d, dtype=np.complex64) / np.sqrt(d)
            self.goal = self.thought.copy()
            return
        # Pack 149: if the organism has SelfDefiningConcepts built, prefer
        # concept[t] over raw key(t) for prompt-HV construction. The
        # concept HV fuses every channel's meaning for the word, so the
        # goal anchor is pulled toward semantic meaning instead of raw symbol.
        # Falls back per-token to key(t) when no concept exists.
        cs = getattr(self.org, '_concepts', None)
        accum = np.zeros(d, dtype=np.complex64)
        for t in prompt_tokens:
            c = cs.concept_of(t) if cs is not None else None
            accum = accum + (c if c is not None else self.org.unified.ck.key(t))
        self.thought = _renorm(accum)
        # Pack 142: goal is the FIXED initial prompt HV; never drifts.
        self.goal = self.thought.copy()

    # ── one think step: thought-state walks the substrate ─────────────────
    def think_step(self):
        ROLE = self.org.unified.roles['cooccur']
        addr = (self.thought * ROLE).astype(np.complex64)
        resp = self.org.unified.sdm.read(addr)
        self.thought = _renorm(
            self.momentum * self.thought + (1.0 - self.momentum) * resp)
        self.thought_trace.append(self.thought.copy())

    # ── Pack 146: build meaning-context HV from recent meaningful tokens ──
    def _grounded_meaning_hv(self, last_token, lookback=6):
        """
        Recall stored meaning across grounded_roles for the most recent
        tokens in history. Walks back up to `lookback` tokens (newest first),
        sums their role-recalls (with decay), and returns a renormalized HV.
        Returns None if no token in the lookback window has any meaning data.
        """
        unified = self.org.unified
        accum = None
        hits = 0
        # Build candidate list: last_token first, then walk back through history.
        cands = [last_token]
        for t in reversed(self.history[:-1] if self.history else []):
            if t == last_token: continue
            cands.append(t)
            if len(cands) >= lookback:
                break
        decay = 1.0
        for tok in cands:
            tok_hit = False
            for role in self.grounded_roles:
                targets = unified._role_targets.get(role, set())
                if tok not in targets:
                    continue
                try:
                    hv = unified.recall(tok, role)
                except Exception:
                    continue
                if hv is None:
                    continue
                contrib = (hv * decay).astype(np.complex64)
                accum = contrib if accum is None else (accum + contrib)
                hits += 1
                tok_hit = True
            decay *= 0.6   # exponential decay across lookback distance
            if hits >= 4:    # stop once we have enough meaning signal
                break
        if hits == 0 or accum is None:
            return None
        return _renorm(accum)

    # ── one speak step: n-gram + thought + goal + grounded-meaning softmax ─
    def speak_step(self, last_token):
        ctx = self.history[-self.ngram_ctx:] if self.history else [last_token]
        unified = self.org.unified
        if hasattr(unified, 'combined_ngram_candidates') and len(ctx) >= 2:
            cands = unified.combined_ngram_candidates(
                ctx, top_k=self.top_k, weights=self.ngram_weights)
        else:
            cands = unified.next_word_candidates(last_token, top_k=self.top_k)
        # Pack 192 v1.1 junk filter: drop code-y candidates (digits, camelCase,
        # long-non-alpha). The substrate may hold poisoned tokens from prior
        # absorbs; this filters them at READ time without touching writes.
        cands = [(w, s) for w, s in cands if not _is_junk_token(w)]
        # Pack 192 v1.2 vocab gate: candidate must be in absorbed cooccur
        # vocab. Plus the alpha-only ASCII filter on the candidate itself.
        seen_vocab = getattr(unified, '_cooccur_seen', None)
        if seen_vocab:
            cands = [(w, s) for w, s in cands
                      if w in seen_vocab
                      and len(w) >= 2 and len(w) <= 14
                      and all(ord(ch) < 128 and ch.isalpha() for ch in w)]
        # Pack 198: frame-scoped vocab gate. If a frame is active for this
        # generation, candidates must be in THAT frame's vocab. Keeps English
        # generation in English, Italian in Italian, etc.
        fidx = getattr(self, '_active_frame_idx', None)
        if fidx is not None and hasattr(self.org, 'frames'):
            fv = self.org.frames.frame_vocab[fidx] if 0 <= fidx < self.org.frames.K else None
            if fv:
                strict = [(w, s) for w, s in cands if w in fv]
                if strict:
                    cands = strict
        # Pack 199 NEW1: grammar-FSM filter. Look up prev-token's frame, get
        # next-frame distribution, drop candidates whose frame has near-zero
        # transition prob from prev's frame. Soft filter (preserves diversity
        # via dist mass, not hard cut).
        ff = getattr(self.org, 'frames', None)
        if ff is not None and ff.locked and ff.frame_bigram.sum() > 0:
            pf = ff.frame_of_word(last_token)
            if pf >= 0:
                next_dist = ff.next_frame_probs(pf)
                if next_dist is not None and next_dist.max() > 0:
                    boosted = []
                    for w, s in cands:
                        cf = ff.frame_of_word(w)
                        if cf >= 0:
                            boost = float(next_dist[cf])
                        else:
                            boost = 1.0 / max(ff.K, 1)
                        boosted.append((w, s * (0.1 + boost)))
                    cands = boosted
        # Pack 199 P2: stop-loop detection. If last 2 emitted tokens are the
        # same as a candidate, suppress that candidate to break repetition.
        if len(self.history) >= 2:
            recent = self.history[-2:]
            if recent[0] == recent[1]:
                cands = [(w, s * 0.05) if w == recent[-1] else (w, s) for w, s in cands]
        if not cands:
            return None
        d = self.org.unified.d
        thought = self.thought
        if self.remove_common:
            self.org.unified._refresh_dirs()
            for v in self.org.unified._dirs:
                thought = thought - np.vdot(v, thought) * v
            thought = _renorm(thought)
        # Pack 142: optional goal anchor (fixed prompt HV). Same projection
        # as thought (mean-removal applied to keep comparable).
        goal = self.goal
        if self.goal_gamma > 0.0 and goal is not None and self.remove_common:
            for v in self.org.unified._dirs:
                goal = goal - np.vdot(v, goal) * v
            goal = _renorm(goal)
        # Pack 146: optional grounded-meaning HV from last token's isa/property.
        meaning = None
        if self.grounded_gamma > 0.0:
            meaning = self._grounded_meaning_hv(last_token)
        scores = []
        for w, base in cands:
            kw = self.org.unified.ck.key(w)
            align = float(np.real(np.vdot(kw, thought))) / d
            boost = self.thought_gamma * align
            if self.goal_gamma > 0.0 and goal is not None:
                g_align = float(np.real(np.vdot(kw, goal))) / d
                boost += self.goal_gamma * g_align
            if self.grounded_gamma > 0.0 and meaning is not None:
                m_align = float(np.real(np.vdot(kw, meaning))) / d
                boost += self.grounded_gamma * m_align
            scores.append((w, max(float(base), 1e-6) * float(np.exp(boost))))
        vals = np.array([s for _, s in scores], dtype=np.float64)
        vals = vals / max(self.temperature, 1e-3)
        vals = vals - vals.max()
        ev = np.exp(vals)
        probs = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev) / len(ev)
        idx = self._rng.choices(range(len(scores)), weights=probs)[0]
        return scores[idx][0]

    # ── primary interface ──────────────────────────────────────────────────
    def generate(self, prompt='', max_tokens=100, return_trace=False, seed=None,
                  auto_frame=True):
        """Generate text from prompt. Pack 198: if `auto_frame` and the
        organism has frames, the prompt is routed to a frame, that frame is
        activated for the duration of generation, and candidates are
        constrained to the frame's vocab."""
        if seed is not None:
            self._rng.seed(seed)
        tokens = _tokenize(prompt)
        self._init_thought(tokens)
        self.history = list(tokens)
        self.thought_trace = [self.thought.copy()]

        # Pack 198: auto-route prompt to a frame, lock substrate to it.
        prev_frame_hv = None
        prev_frame_tag = None
        active_frame_idx = None
        if auto_frame and hasattr(self.org, 'frames') and self.org.frames is not None \
                and self.org.frames.locked:
            try:
                idx, fhv, _ = self.org.frames.route_prompt(tokens, self.org.unified.ck)
                if fhv is not None and hasattr(self.org.unified, 'set_frame'):
                    prev_frame_hv = self.org.unified.current_frame_hv
                    prev_frame_tag = getattr(self.org.unified, '_current_frame_tag', None)
                    self.org.unified.set_frame(fhv, frame_tag=f'f{idx}')
                    active_frame_idx = idx
                    self._active_frame_idx = idx
            except Exception:
                pass

        try:
            for _ in range(int(max_tokens)):
                for _ in range(self.think_steps):
                    self.think_step()
                last = self.history[-1] if self.history else ''
                nxt = self.speak_step(last) if last else None
                if nxt is None:
                    break
                self.history.append(nxt)
        finally:
            # Restore frame state.
            if hasattr(self.org.unified, 'set_frame'):
                if prev_frame_hv is None:
                    self.org.unified.clear_frame()
                else:
                    self.org.unified.set_frame(prev_frame_hv, frame_tag=prev_frame_tag)
            self._active_frame_idx = None

        out = ' '.join(self.history)
        if return_trace:
            return out, self.thought_trace
        return out

    # ── inject new knowledge mid-generation ──────────────────────────────
    def inject_fact(self, hypo, role, target, n=20):
        """Online learning during generation. Subsequent tokens see this fact."""
        self.org.unified.assert_relation(hypo, role, target) if hasattr(
            self.org.unified, 'assert_relation') else None
        for _ in range(n):
            self.org.unified.relate(hypo, role, target)
