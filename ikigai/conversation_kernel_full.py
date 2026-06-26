"""
ikigai.conversation_kernel_full -- Fully Integrated Conversational Agent.

Day 55 Pack 40 -- single chat() entry point, all 7 primitives active:

    IO Bridge      -- tokenize / decode / grammar prefix
    PLHB           -- phasor state, constant 3200 B
    PSSTC          -- grammar-locked emit
    BSPM           -- per-user belief B_U + cognitive axes
    CVFEF          -- free energy action selection
    ACCI           -- crystalline triple store
    HebbianTuner   -- online T_couple -> B_U alignment
    NGramExpander  -- geometric seed -> fluent phrase

Per-turn pipeline:
    1. tokenize(text)
    2. register tokens in vocab
    3. expander.train_on_turn(tokens)          [n-gram online update]
    4. kernel.step(tokens)                     [PLHB+PSSTC+BSPM+CVFEF+ACCI]
    5. tuner.tune(S, G_id, B_U)               [Hebbian T_couple update]
    6. seed = decode_to_words(B_U, top_k=3)   [cosine nearest words]
    7. expanded = expander.expand(seed)        [n-gram continuation]
    8. response = prefix + expanded            [grammar-typed output]

Quality metrics tracked per turn:
    emit_cos   cosine(emit_vec, B_U)          rises as Hebbian tunes
    vocab_hit  fraction of expanded words in user vocab
    F_t        free energy (falls as conversation stabilises)
"""

import numpy as np

from ikigai.conversation_kernel_io import (
    ConversationKernelIO, GRAMMAR_PREFIX, _word_hv,
)
from ikigai.cognition.hebbian_tuner import HebbianVocabTuner
from ikigai.cognition.ngram_expander  import NGramExpander


class ChatOutput:
    """Per-turn output from ConversationKernelFull.chat()."""
    __slots__ = [
        'turn', 'response', 'grammar', 'action', 'F_t',
        'emit_cos', 'vocab_hit', 'expanded_words',
        'cos_before', 'cos_after', 'B_U', 'C_U',
    ]
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return (f'ChatOutput(turn={self.turn}, grammar={self.grammar}, '
                f'action={self.action}, F={self.F_t:.3f}, '
                f'emit_cos={self.emit_cos:.3f}, vocab_hit={self.vocab_hit:.2f})')


class ConversationKernelFull:
    """
    Fully integrated conversational agent. No LLM. No transformer.

    chat(text) -> (response_str, ChatOutput)

    All 7 primitives active. Quality improves with every turn:
        - Hebbian tuning aligns emit_vec to user vocabulary (emit_cos rises)
        - N-gram learns user phrasing patterns (expansion improves)
        - BSPM belief tracks semantic intent (B_U converges to topic)
        - CVFEF free energy falls as conversation stabilises
    """

    def __init__(self, seed=42, top_k=5, eta=0.1, n=3, max_expand=10):
        self.io       = ConversationKernelIO(seed=seed, top_k=top_k)
        self.tuner    = HebbianVocabTuner(self.io.kernel.psstc, eta=eta)
        self.expander = NGramExpander(n=n, max_expand=max_expand)
        self._turn    = 0

        # Quality log: list of dicts per turn
        self.quality_log = []

        # Running user vocab (all tokens seen so far)
        self._user_vocab = set()

    # ─── core chat ───────────────────────────────────────────────

    def chat(self, text, role='user'):
        """
        Full per-turn pipeline.
        text: raw natural language string
        role: 'user' or 'self'
        Returns (response_str, ChatOutput).
        """
        # 1. Tokenize
        tokens = self.io.tokenize(text)
        if not tokens:
            tokens = ['empty']

        # 2. Register in vocab + track user vocab
        for t in tokens:
            self.io._register(t)
        if role == 'user':
            self._user_vocab.update(tokens)

        # 3. N-gram online training
        self.expander.train_on_turn(tokens)

        # 4. Kernel step (PLHB + PSSTC + BSPM + CVFEF + ACCI)
        out = self.io.kernel.step(tokens, role=role)

        # 5. Hebbian tune T_couple[:, G_id, :] toward B_U
        S = self.io.kernel.psstc.encode_semantic(tokens)
        _, cos_before, cos_after = self.tuner.tune(S, out.G_id, out.B_U)

        # 6. Decode B_U to seed words
        seed_words = self.io.decode_to_words(out.B_U, top_k=3)

        # 7. N-gram expansion
        expanded = self.expander.expand(seed_words, max_len=self.expander.max_expand)
        if not expanded:
            expanded = seed_words[:3]

        # 8. Grammar prefix + expanded content
        prefix = GRAMMAR_PREFIX.get(out.grammar_name, '')
        parts  = ([prefix] + expanded) if prefix else expanded
        response = ' '.join(parts[:10]).strip()

        # Quality metrics
        vocab_hit  = (len(set(expanded) & self._user_vocab) / max(1, len(expanded))
                      if self._user_vocab else 0.0)

        self._turn += 1
        q = {
            'turn':          self._turn,
            'grammar':       out.grammar_name,
            'action':        out.action,
            'F_t':           out.F_t,
            'emit_cos':      cos_after,
            'cos_before':    cos_before,
            'vocab_hit':     vocab_hit,
            'expanded_words': expanded,
        }
        self.quality_log.append(q)

        return response, ChatOutput(
            turn=self._turn,
            response=response,
            grammar=out.grammar_name,
            action=out.action,
            F_t=out.F_t,
            emit_cos=cos_after,
            vocab_hit=vocab_hit,
            expanded_words=expanded,
            cos_before=cos_before,
            cos_after=cos_after,
            B_U=out.B_U,
            C_U=out.C_U,
        )

    # ─── quality trends ──────────────────────────────────────────

    def emit_cos_trend(self):
        """(mean early 5 turns, mean late 5 turns) of emit_cos."""
        if len(self.quality_log) < 2:
            return 0.0, 0.0
        early = [q['emit_cos'] for q in self.quality_log[:5]]
        late  = [q['emit_cos'] for q in self.quality_log[-5:]]
        return float(np.mean(early)), float(np.mean(late))

    def vocab_hit_trend(self):
        """(mean early, mean late) of vocab_hit."""
        if len(self.quality_log) < 2:
            return 0.0, 0.0
        early = [q['vocab_hit'] for q in self.quality_log[:5]]
        late  = [q['vocab_hit'] for q in self.quality_log[-5:]]
        return float(np.mean(early)), float(np.mean(late))

    def F_trend(self):
        """(mean early, mean late) of F_t."""
        if len(self.quality_log) < 2:
            return 0.0, 0.0
        early = [q['F_t'] for q in self.quality_log[:5]]
        late  = [q['F_t'] for q in self.quality_log[-5:]]
        return float(np.mean(early)), float(np.mean(late))

    # ─── pass-through accessors ──────────────────────────────────

    @property
    def turn_count(self):
        return self._turn

    @property
    def plhb_bytes(self):
        return self.io.kernel.plhb.state_size_bytes()

    @property
    def vocab_size(self):
        return self.io.vocab_size

    @property
    def ngram_entries(self):
        return self.expander.ngram_entries

    @property
    def crystal_unique(self):
        return self.io.kernel.acci.unique_triples()

    def summary(self):
        e_early, e_late = self.emit_cos_trend()
        v_early, v_late = self.vocab_hit_trend()
        f_early, f_late = self.F_trend()
        return {
            'turns':          self._turn,
            'plhb_bytes':     self.plhb_bytes,
            'vocab_size':     self.vocab_size,
            'ngram_entries':  self.ngram_entries,
            'crystal_unique': self.crystal_unique,
            'emit_cos_trend': (e_early, e_late),
            'vocab_hit_trend':(v_early, v_late),
            'F_trend':        (f_early, f_late),
        }

    def reset(self, seed=42):
        self.io.reset(seed=seed)
        self.tuner.reset_stats()
        self.expander.reset()
        self._turn = 0
        self.quality_log.clear()
        self._user_vocab.clear()
