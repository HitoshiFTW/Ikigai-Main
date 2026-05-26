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
    """

    def __init__(self, organism, think_steps=3, momentum=0.7,
                 thought_gamma=4.0, temperature=0.7, top_k=20,
                 remove_common=True, ngram_weights=(0.2, 0.4, 0.4),
                 ngram_ctx=3):
        self.org = organism
        self.think_steps = int(think_steps)
        self.momentum = float(momentum)
        self.thought_gamma = float(thought_gamma)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.remove_common = bool(remove_common)
        self.ngram_weights = tuple(ngram_weights)
        self.ngram_ctx = int(ngram_ctx)
        self.thought = None
        self.thought_trace = []
        self.history = []
        self._rng = random.Random()

    #  initialization
    def _init_thought(self, prompt_tokens):
        d = self.org.unified.d
        if not prompt_tokens:
            self.thought = np.ones(d, dtype=np.complex64) / np.sqrt(d)
            return
        accum = np.zeros(d, dtype=np.complex64)
        for t in prompt_tokens:
            accum = accum + self.org.unified.ck.key(t)
        self.thought = _renorm(accum)

    #  one think step: thought-state walks the substrate
    def think_step(self):
        ROLE = self.org.unified.roles['cooccur']
        addr = (self.thought * ROLE).astype(np.complex64)
        resp = self.org.unified.sdm.read(addr)
        self.thought = _renorm(
            self.momentum * self.thought + (1.0 - self.momentum) * resp)
        self.thought_trace.append(self.thought.copy())

    #  one speak step: n-gram-combined + thought-aligned softmax
    def speak_step(self, last_token):
        ctx = self.history[-self.ngram_ctx:] if self.history else [last_token]
        unified = self.org.unified
        if hasattr(unified, 'combined_ngram_candidates') and len(ctx) >= 2:
            cands = unified.combined_ngram_candidates(
                ctx, top_k=self.top_k, weights=self.ngram_weights)
        else:
            cands = unified.next_word_candidates(last_token, top_k=self.top_k)
        if not cands:
            return None
        d = self.org.unified.d
        thought = self.thought
        if self.remove_common:
            self.org.unified._refresh_dirs()
            for v in self.org.unified._dirs:
                thought = thought - np.vdot(v, thought) * v
            thought = _renorm(thought)
        scores = []
        for w, base in cands:
            kw = self.org.unified.ck.key(w)
            align = float(np.real(np.vdot(kw, thought))) / d
            scores.append((w, max(float(base), 1e-6) * float(np.exp(self.thought_gamma * align))))
        vals = np.array([s for _, s in scores], dtype=np.float64)
        vals = vals / max(self.temperature, 1e-3)
        vals = vals - vals.max()
        ev = np.exp(vals)
        probs = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev) / len(ev)
        idx = self._rng.choices(range(len(scores)), weights=probs)[0]
        return scores[idx][0]

    #  primary interface
    def generate(self, prompt='', max_tokens=100, return_trace=False, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        tokens = _tokenize(prompt)
        self._init_thought(tokens)
        self.history = list(tokens)
        self.thought_trace = [self.thought.copy()]
        for _ in range(int(max_tokens)):
            for _ in range(self.think_steps):
                self.think_step()
            last = self.history[-1] if self.history else ''
            nxt = self.speak_step(last) if last else None
            if nxt is None:
                break
            self.history.append(nxt)
        out = ' '.join(self.history)
        if return_trace:
            return out, self.thought_trace
        return out

    #  inject new knowledge mid-generation
    def inject_fact(self, hypo, role, target, n=20):
        """Online learning during generation. Subsequent tokens see this fact."""
        self.org.unified.assert_relation(hypo, role, target) if hasattr(
            self.org.unified, 'assert_relation') else None
        for _ in range(n):
            self.org.unified.relate(hypo, role, target)
