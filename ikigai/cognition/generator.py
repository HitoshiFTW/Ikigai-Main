"""
ikigai.cognition.generator -- SentenceGenerator.

Day 56 Pack 107 -- turn READER into SPEAKER.

Pure HDC + statistics. No gradient, no autoregressive transformer.
Uses what already exists in organism:
    - bigram counts (Channel 5 grammar)
    - lexicon HVs (Channel 1 co-occurrence + drift)
    - context HV (running thought)
    - persona vectors (Channel 3 / PGMW)

Generation = Markov walk with HV-context bias.

API:
    gen = SentenceGenerator(organism)
    text = gen.generate(prompt='the cat', max_len=15)
    text = gen.respond(user_text, max_len=15)
    text = gen.generate(prompt='', context_hv=..., persona='formal')

Modes:
    bigram_argmax   -- pick most likely next word (deterministic, dull)
    bigram_sample   -- sample from bigram distribution (varied, sometimes wild)
    context_biased  -- mix bigram prob w/ context-cosine bias (coherent + varied)
"""

import re
import math
import random
from collections import Counter

import numpy as np


def tokenize(text):
    cleaned = re.sub(r"[^a-z0-9'\s]", ' ', text.lower())
    return [t for t in cleaned.split() if t]


def _cos_c(a, b, d):
    return float(np.real(np.vdot(a, b))) / d


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


class SentenceGenerator:
    """
    Generate text from the organism's learned bigram distribution + HV space.

    organism: IkigaiOrganism instance (must have being.lexicon + grammar._bigram).
    """

    def __init__(self, organism, d=2048):
        self.org = organism
        self.d   = int(d)
        self._rng = random.Random()

    # ── candidates from bigrams ───────────────────────────────────────────

    def _next_word_candidates(self, prev_word, top_k=20):
        """Return top-k candidate next-words from bigram counts, with probs."""
        bigram = self.org.grammar._bigram
        # All (prev, c) starting with prev_word
        candidates = Counter()
        for (p, c), n in bigram.items():
            if p == prev_word:
                candidates[c] += n
        if not candidates:
            return []
        total = sum(candidates.values())
        ranked = [(w, c / total) for w, c in candidates.most_common(top_k)]
        return ranked

    # ── word selection w/ context bias + persona ─────────────────────────

    def _select_next(self, prev_word, mode='context_biased',
                     context_hv=None, persona=None,
                     trajectory_hv=None,
                     temperature=0.7, context_weight=0.4,
                     persona_gamma=4.0, syntax_weight=0.5):
        """
        Pick next word given prev_word.

        Modes:
            bigram_argmax    -- argmax bigram prob
            bigram_sample    -- sample from bigram distribution
            context_biased   -- bigram x exp(gamma*persona_cos) x context + syntax

        New (Pack 109 patches):
            Fix 1: contrastive softmax persona warp via exp(gamma * cos)
            Fix 2: syntactic L/R context feedback via Channel 5 left_ctx
        """
        cands = self._next_word_candidates(prev_word, top_k=20)
        if not cands:
            return None

        # Argmax: deterministic, ignore biases
        if mode == 'bigram_argmax':
            for w, _ in cands:
                if w != self.org.grammar.SENT_EOS:
                    return w
            return cands[0][0]

        # Compute final score per candidate
        scores = []
        for word, prob in cands:
            if word == self.org.grammar.SENT_EOS:
                base = prob * 0.3
            else:
                base = max(prob, 1e-6)

            # ── Fix 1: persona contrastive boost ──
            persona_boost = 1.0
            if persona is not None and persona in self.org.persona._personas \
               and word in self.org.being.lexicon:
                p_hv, _ = self.org.persona._personas[persona]
                cos_p = _cos_c(self.org.being.lexicon[word], p_hv, self.d)
                # Exponential boost: exp(gamma * cos). gamma=4 -> 50x boost at cos=1.
                persona_boost = float(np.exp(persona_gamma * cos_p))

            # ── Fix 2: syntactic L/R context feedback ──
            # candidate's left_ctx should match the trajectory we have so far
            syntax_boost = 1.0
            if trajectory_hv is not None and word != self.org.grammar.SENT_EOS:
                left_ctx = self.org.grammar.left_context(word)
                if left_ctx is not None:
                    cos_s = _cos_c(left_ctx, trajectory_hv, self.d)
                    # Soft boost: 1 + syntax_weight * max(0, cos)
                    syntax_boost = 1.0 + syntax_weight * max(0.0, cos_s)

            # Context-HV bias (existing)
            ctx_score = 0.0
            if context_hv is not None and word in self.org.being.lexicon:
                ctx_score = max(0.0, _cos_c(self.org.being.lexicon[word],
                                              context_hv, self.d))

            if mode == 'context_biased':
                # Multiplicative combination: persona + syntax shape distribution
                # then additive context bias modulates final score
                combined = base * persona_boost * syntax_boost + \
                           context_weight * ctx_score
            else:
                combined = base
            scores.append((word, combined))

        # Sample with temperature
        if mode == 'bigram_sample' or mode == 'context_biased':
            vals = np.array([s for _, s in scores], dtype=np.float64)
            vals = vals / max(temperature, 1e-3)
            vals = vals - vals.max()
            exp_v = np.exp(vals)
            total = exp_v.sum()
            if total < 1e-12:
                return scores[0][0]
            probs = exp_v / total
            idx = self._rng.choices(range(len(scores)), weights=probs)[0]
            return scores[idx][0]
        else:
            return max(scores, key=lambda x: x[1])[0]

    # ── primary interface ───────────────────────────────────────────────

    def generate(self, prompt='', max_len=15, mode='context_biased',
                 context_hv=None, persona=None,
                 temperature=0.7, context_weight=0.4,
                 persona_gamma=4.0, syntax_weight=0.5,
                 stop_at_eos=True, seed=None):
        """
        Generate continuation given a prompt.

        persona_gamma: exponential boost strength for persona contrastive warp
        syntax_weight: linear boost strength for L/R context syntactic feedback
        """
        if seed is not None:
            self._rng.seed(seed)

        # Bootstrap from prompt
        prompt_tokens = tokenize(prompt) if prompt else []
        if not prompt_tokens:
            current = self.org.grammar.SENT_BOS
        else:
            current = prompt_tokens[-1]

        # If no context_hv provided, derive from prompt's lexicon-bag
        if context_hv is None and prompt_tokens:
            accum = np.zeros(self.d, dtype=np.complex64)
            n = 0
            for tok in prompt_tokens:
                if tok in self.org.being.lexicon:
                    accum = accum + self.org.being.lexicon[tok]
                    n += 1
            if n > 0:
                context_hv = _renorm(accum / n)

        # Trajectory HV = running average of generated tokens' lexicon HVs.
        # Used for Fix 2 (syntactic L/R context feedback).
        trajectory_hv = context_hv.copy() if context_hv is not None else None

        output = list(prompt_tokens)
        for _ in range(max_len):
            nxt = self._select_next(current, mode=mode,
                                     context_hv=context_hv,
                                     persona=persona,
                                     trajectory_hv=trajectory_hv,
                                     temperature=temperature,
                                     context_weight=context_weight,
                                     persona_gamma=persona_gamma,
                                     syntax_weight=syntax_weight)
            if nxt is None:
                break
            if stop_at_eos and nxt == self.org.grammar.SENT_EOS:
                break
            if nxt == self.org.grammar.SENT_BOS:
                continue
            output.append(nxt)
            current = nxt

            # Update context HV + trajectory with newly-added token
            if nxt in self.org.being.lexicon:
                tok_hv = self.org.being.lexicon[nxt]
                if context_hv is not None:
                    a_ctx = 0.2
                    context_hv = _renorm(
                        (1 - a_ctx) * context_hv + a_ctx * tok_hv
                    )
                if trajectory_hv is None:
                    trajectory_hv = tok_hv.copy()
                else:
                    a_tr = 0.3
                    trajectory_hv = _renorm(
                        (1 - a_tr) * trajectory_hv + a_tr * tok_hv
                    )

        return ' '.join(output)

    # ── dialogue helper ──────────────────────────────────────────────────

    def respond(self, user_text, dialogue_loop=None, max_len=12,
                temperature=0.6, persona=None, seed=None):
        """
        Generate a response to user_text.
        If dialogue_loop is provided: use its context_hv as bias.
        """
        if dialogue_loop is not None:
            context_hv = dialogue_loop.context_hv()
            if persona is None:
                persona = dialogue_loop.active_persona()
        else:
            context_hv = None
        # Use a few last tokens of user as prompt seed
        tokens = tokenize(user_text)
        prompt = ' '.join(tokens[-3:]) if tokens else ''
        return self.generate(prompt=prompt, max_len=max_len,
                             mode='context_biased',
                             context_hv=context_hv,
                             persona=persona,
                             temperature=temperature,
                             seed=seed)
