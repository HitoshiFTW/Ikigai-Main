"""
ikigai.cognition.cat4_dopamine -- Pack 276 dopamine + noisy-retry loop.

Day 76 (2 PM session). Biologically-motivated supervised RL on top
of cat-4 substrate ICL.

WHY THIS EXISTS
---------------
Substrate alone hits the cleanup wall (16/16 only because Pack 273
cache bypassed cleanup for the 16 known capitals). For unseen
queries, substrate's first prediction may be junk from the noise
floor. Pack 276 lets the organism RETRY:

    1. Predict (current substrate state)
    2. If correct (verifier says YES): fire dopamine -> reinforce
       the (state, action) binding via cat4.absorb_chain. Substrate
       gets *more* confident on this anchor for next time.
    3. If wrong (verifier says NO): inject phase noise into the
       state HV recall and try again. Noise = exploration. Continue
       until correct or budget exhausted.

BIOLOGICAL GROUNDING
--------------------
- Schultz 1997 -- dopamine RPE (Pack 251 already uses this at write)
- Fremaux + Gerstner 2016 -- DA-gated STDP
- Aston-Jones + Cohen 2005 -- noradrenergic gain modulation = noise
  injection enabling exploration when prediction is wrong
- Williams 1992 REINFORCE -- score-function gradient via random
  perturbations as an unbiased estimator

This module composes existing primitives (reason() + absorb_chain)
into a closed-loop teaching mechanism. Nothing new at the substrate
level. Pack 276 = the LOOP.

USAGE
-----
    teach = Cat4Dopamine(org)

    # Single query with ground-truth answer
    result = teach.reason_until(
        query='What is the capital of Germany',
        expected='berlin',
        max_iters=10,
        noise_scale=0.05,
    )
    # result.predicted, result.iters_used, result.converged

    # Batch (teaches many at once, organism stays loaded)
    pairs = [('What is the capital of France', 'paris'), ...]
    stats = teach.teach_batch(pairs)
"""

import hashlib
import re
import time
import unicodedata
import numpy as np


def _ascii_fold(s):
    """Pack 287.5 v4 -- fold accented/unicode letters to ASCII (n~ -> n,
    a' -> a) via NFKD so the ASCII word regex doesn't FRAGMENT accented
    words ('espanol' from 'espa' + 'ol' -> last-word parser returned
    'ol').  General Unicode normalisation, not a word list."""
    if not s:
        return s
    return (unicodedata.normalize('NFKD', str(s))
            .encode('ascii', 'ignore').decode('ascii'))


class TeacherOracle:
    """Wraps a RemoteLLMTeacher so the dopamine loop can fetch
    ground-truth answers from an external oracle instead of relying
    on a hardcoded test set.

    Biologically: the teacher = environment / parent / sensory input.
    Substrate = organism. Dopamine fires when organism's prediction
    matches the environment signal.
    """

    # Pack 287/289: more aggressive system prompt to discourage
    # restated-question answers ("Greece's capital is athens" -> we
    # want 'athens', not 'greece').  Combined with the smarter parser
    # below, we get the trailing content word out of either bare-token
    # or restated-form responses.
    _WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']*")   # digits: Pack 296 numeric answers ('0')
    _STRIP_FIRST_WORD = _WORD_RE          # legacy alias for Pack 287 v0
    _ECHO_STOPWORDS = frozenset({
        'a', 'an', 'the', 'is', 'was', 'are', 'be', 'been',
        'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
        'that', 'this', 'it', 'its', 'capital', 'city',
        'answer', 'response', 'word', 'lowercase',
    })

    def __init__(self, teacher, system_prefix=None,
                  max_tokens=128, temperature=0.0, multiword=False):
        # Pack 289 v0 default bump: 24 -> 128.  DeepSeek-R1 wraps
        # answers in <think>...</think> CoT chains; 24 tokens runs out
        # before the actual answer emits.  128 covers typical
        # one-line answers + their thinking trace without bloating
        # latency by more than ~50ms.
        self.teacher = teacher
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        # Pack 307 (Day 80 #1) -- multi-word answer mode.  Single-word is
        # the default (protects the 30/30 capital baseline + every existing
        # caller).  When multiword=True the parser keeps the FULL content
        # phrase ('south america', 'buenos aires', 'may 16 2015') instead of
        # collapsing to the last content token -- the storage path already
        # stores the action token tuple whole and rejoins it on recall, so
        # the only gap was the parse.
        self.multiword = bool(multiword)
        # Pack 309 (Day 80 #2 follow-up) -- ABSTENTION PROTOCOL.  The
        # teacher self-reports ignorance via a single sentinel token the
        # PROMPT defines ("if you don't know, reply exactly idk").  This is
        # an instruction/protocol -- like "answer in one word" -- NOT a
        # hardcoded list of human abstention phrases ("i'm not sure", "no
        # information", ...), which would be the forbidden English-heuristic
        # word list.  Answers == sentinel are dropped (return None) so the
        # cache never absorbs a "don't know" -> recall returns unknown
        # instead of a noise-floor token.  Also dodges the 287.5 sycophancy
        # trap: this detects abstention, it does NOT verify answers.
        self.abstain_token = 'idk'
        # short, deterministic prompt -- want a one-token answer (or, in
        # multiword mode, just the answer phrase, still no explanation).
        _abstain = (f'If you do not know the answer, reply with exactly '
                    f'{self.abstain_token} and nothing else. ')
        self.system_prefix = system_prefix or (
            ('Answer with only the answer, lowercase, no explanation, '
             'no restating the question. ' + _abstain) if self.multiword else
            ('Answer in one word, lowercase, no explanation, '
             'no extra punctuation. ' + _abstain))
        self.cache = {}     # in-memory cache of (query -> answer)

    def _parse_answer(self, text, query=None):
        """Pack 289 parser -- pick the salient answer word from the
        teacher's response, robust to restated-question forms.

        Strategy:
          1. Extract all word tokens (`_WORD_RE`).
          2. Drop tokens that appear verbatim (case-insensitive) in the
             query -- this kills "Greece" in "Greece's capital is
             Athens" without a separate parser.
          3. Drop stopwords ("the", "is", "capital", ...).
          4. Return the LAST surviving word.  R1 puts the salient noun
             at the end ("X is Y" / "the capital is Y").  Last-wins
             beats first-wins on every restated form measured.
          5. Fall back to first word if all tokens get filtered (handles
             bare single-word responses like 'paris').
        """
        if not text:
            return None
        # Pack 287.5 v2 -- R1-Distill blurts the one-word answer, then a
        # blank line, then reverts to a CoT ramble ("tokyo\n\nOkay, so I
        # need to figure out...") that max_tokens truncates mid-sentence.
        # Keep only the first non-empty line so "last salient word" sees
        # just the answer, not the trailing ramble.  Single-line restated
        # forms ("Greece's capital is Athens") are unaffected.
        for _line in text.splitlines():
            if _line.strip():
                text = _line
                break
        # Pack 287.5 v4 -- fold accents BEFORE the ASCII word regex so
        # 'espanol'/'bogota' stay whole instead of fragmenting to 'ol'.
        text = _ascii_fold(text)
        words = [w.lower() for w in self._WORD_RE.findall(text)]
        if not words:
            return None
        query_toks = set()
        if query:
            for q in self._WORD_RE.findall(str(query).lower()):
                query_toks.add(q)
        # Pass 1: keep only content tokens (not echoed, not stopwords)
        content = [w for w in words
                     if w not in query_toks and w not in self._ECHO_STOPWORDS]
        if content:
            # Pack 307 -- multiword keeps the whole content phrase in order
            # ('buenos aires', 'south america', 'may 16 2015'); single-word
            # mode keeps the last salient token (R1 puts it at the end).
            return ' '.join(content) if self.multiword else content[-1]
        # Pass 2: drop only stopwords (preserve echo if necessary)
        content2 = [w for w in words if w not in self._ECHO_STOPWORDS]
        if content2:
            return ' '.join(content2) if self.multiword else content2[-1]
        # Pass 3: bare first word
        return words[0]

    def ask(self, query):
        """Return the oracle's one-word answer for `query`. Cached
        per process so repeated calls during a loop are free.

        Backwards-compatible single-query path.  For N>1 queries prefer
        `ask_batch` -- one HTTP roundtrip + vLLM continuous batching.
        """
        results = self.ask_batch([query])
        return results.get(str(query).strip())

    def ask_with_logprobs(self, queries, top_n=10):
        """Pack 287.5 v2 -- batched ask returning an ATOMIC-token first
        answer distribution per query.

        Returns dict[query, (answer_token, [(atomic_token, logprob), ...])].
        v2 reassembles BPE subwords across generated positions into whole
        answer words ('to'+'kyo' -> 'tokyo') and sums their logprobs, so
        the distribution's tokens line up with the substrate's atomic
        action vocabulary.  Falls back to the position-0 token list when
        serve.py ships a single position (older deploy) or reassembly
        misses.  Older serve.py without logprobs returns None.
        """
        results = {}
        if not queries:
            return results
        prompts = [
            f'{self.system_prefix}Q: {str(q).strip()}\nA:'
            for q in queries]
        saved_t = self.teacher.temperature
        saved_n = self.teacher.max_new_tokens
        try:
            self.teacher.temperature = self.temperature
            self.teacher.max_new_tokens = self.max_tokens
            data = self.teacher._batch(prompts, logprobs=int(top_n))
        finally:
            self.teacher.temperature = saved_t
            self.teacher.max_new_tokens = saved_n
        if not data or 'items' not in data:
            for q in queries:
                results[str(q).strip()] = (None, None)
            return results
        items = data['items']
        for q, item in zip(queries, items):
            text = self.teacher._post(item.get('text', ''))
            answer = self._parse_answer(text, query=q)
            # Pack 287.5 v2 -- collapse BPE subwords across positions into
            # an atomic-token distribution aligned to the substrate vocab.
            dist = _atomic_logprob_dist(item.get('logprobs'), answer)
            results[str(q).strip()] = (answer, dist)
        return results

    def ask_batch(self, queries, batch_size=16):
        """Pack 281 -- batch oracle calls so the 3090 vLLM scheduler
        can run continuous batching across N prompts in one request.

        Throughput sustained at batch=32: ~50 q/s vs ~2 q/s sequential
        (measured Day 73 absorb runs).  Returns dict[query, answer]
        covering every input -- cache hits served locally without
        re-hitting the remote.
        """
        results = {}
        uncached = []
        for q in queries:
            qs = str(q).strip()
            if qs in self.cache:
                results[qs] = self.cache[qs]
            else:
                uncached.append(qs)
        if not uncached:
            return results
        # Save + override teacher params for deterministic output
        saved_t = self.teacher.temperature
        saved_n = self.teacher.max_new_tokens
        try:
            self.teacher.temperature = self.temperature
            self.teacher.max_new_tokens = self.max_tokens
            # Chunk to batch_size so the 3090 doesn't OOM on very large
            # absorb runs (Pack 282+ pipeline can pass 1000+ at a time)
            for i in range(0, len(uncached), int(batch_size)):
                chunk = uncached[i:i + int(batch_size)]
                prompts = [
                    f'{self.system_prefix}Q: {q}\nA:' for q in chunk]
                data = self.teacher._batch(prompts)
                if not data or 'items' not in data:
                    # Mark all in this chunk as None so callers can
                    # decide to retry; do not cache failures.
                    for q in chunk:
                        results[q] = None
                    continue
                items = data['items']
                for q, item in zip(chunk, items):
                    text = self.teacher._post(item.get('text', ''))
                    # Pack 289: query-aware parser drops echoed query
                    # tokens + stopwords + prefers last salient word
                    # (handles "Greece's capital is Athens" -> 'athens')
                    answer = self._parse_answer(text, query=q)
                    # Pack 309 -- abstention: teacher emitted the sentinel
                    # (whole answer == 'idk', not merely contains it) ->
                    # treat as "don't know", DON'T cache, return None so the
                    # caller skips absorption and recall stays unknown.
                    if answer and answer.strip().lower() == self.abstain_token:
                        results[q] = None
                        continue
                    if answer:
                        self.cache[q] = answer
                        results[q] = answer
                    else:
                        results[q] = None
        finally:
            self.teacher.temperature = saved_t
            self.teacher.max_new_tokens = saved_n
        return results


def _atomic_logprob_dist(positions, answer):
    """Pack 287.5 v2 -- turn vLLM per-position top-N logprobs into an
    atomic-token distribution aligned to the substrate vocabulary.

    `positions` is a list (one per generated token position) of entries
    [{'token': str, 'logprob': float}, ...] sorted logprob-desc, so
    entry[0] is the greedy token (the oracle runs temperature=0).

    Returns [(atomic_token, logprob), ...] or None.

    Strategy: take the greedy token at each position, concatenate the
    lowercased strings into the generated text, locate the parsed
    `answer` word in it, and SUM the logprobs of the positions whose
    characters compose it -> joint logprob of the whole answer word.
    This collapses subword splits ('to'+'kyo') into one atomic entry
    ('tokyo') whose token matches the substrate's action vocabulary.

    Degrades gracefully: a single-position payload (older serve.py) is
    returned as the first-token distribution unchanged; a reassembly
    miss (think-block / restated form) falls back to the position-0
    greedy token so the caller still gets a signal.
    """
    if not positions:
        return None
    pos0 = positions[0] or []
    pos0_dist = [(str(e['token']).strip().lower(), float(e['logprob']))
                  for e in pos0 if e.get('token')]
    # Single position -> first-token distribution (back-compat path).
    if len(positions) == 1:
        return pos0_dist or None
    # Multi-position: greedy token per position = entry[0].
    pieces = []   # (lowercased_raw, logprob)
    for pos in positions:
        if not pos:
            continue
        top = pos[0]
        pieces.append((str(top.get('token', '')).lower(),
                        float(top.get('logprob', 0.0))))
    target = (answer or '').strip().lower()
    if not pieces or not target:
        return pos0_dist or None
    concat = ''.join(p for p, _ in pieces)
    start = concat.find(target)
    if start < 0:
        # Reassembly miss -> fall back to position-0 greedy token.
        first_tok = pieces[0][0].strip()
        return [(first_tok, pieces[0][1])] if first_tok else (pos0_dist or None)
    end = start + len(target)
    joint = 0.0
    off = 0
    for piece, lp in pieces:
        p_start, p_end = off, off + len(piece)
        off = p_end
        if p_start < end and p_end > start:    # char-span overlap
            joint += lp
    return [(target, joint)]


def _matches_expected(predicted, expected):
    """Forgiving match: substring either direction, lowercased."""
    p = (predicted or '').strip().lower()
    e = (expected or '').strip().lower()
    if not p or not e:
        return False
    return e in p or p in e


# Pack 283 -- atomic vs compositional classifier.  The cache is for
# the VOCABULARY of facts; the substrate is for the GRAMMAR of
# composition.  Math queries ("5 + 3", "100 times 99") cost nothing
# to recompute (Pack 254/291 RHC ⋆ at ~50ms per query) -- caching
# them duplicates substrate behavior without semantic gain, and at
# 10^9-fact scale every avoided cache entry buys back ~40 bytes
# (Pack 282 LMDB).
#
# Heuristic: a query is COMPOSITIONAL if its tokenization contains
# any of {arithmetic operator token, two or more pure-number tokens}.
# Otherwise it is ATOMIC (a specific fact lookup).

_COMPOSITIONAL_OPS = frozenset({
    '+', '-', '*', '/', '=',
    'plus', 'minus', 'times', 'multiplied', 'product', 'into',
    'divided', 'over', 'sum', 'difference', 'quotient',
})

_NUM_RE = re.compile(r'^-?\d+$')


def is_compositional_query(text):
    """True when the query carries arithmetic structure that the
    substrate can recompute from primitives without cache help."""
    if not text:
        return False
    tokens = re.findall(r"[a-z]+|-?\d+|[+\-*/=]", str(text).lower())
    num_count = 0
    for t in tokens:
        if t in _COMPOSITIONAL_OPS:
            return True
        if _NUM_RE.match(t):
            num_count += 1
            if num_count >= 2:
                return True
    return False


def is_atomic_query(text):
    return not is_compositional_query(text)


class Cat4Dopamine:
    """Pack 276 dopamine reinforcement + noradrenergic noise retry."""

    def __init__(self, organism):
        self.org = organism
        self.stats = {
            'queries': 0,
            'first_try_correct': 0,
            'corrected_after_retry': 0,
            'failed': 0,
            'total_iters': 0,
            'dopamine_fires': 0,
            'noise_fires': 0,
        }

    # ---- inner pieces -----------------------------------------------

    def _dopamine_reinforce(self, query, expected_answer):
        """Write (query, expected_answer) into substrate via the
        standard absorb_chain path. OPV gate fires KNOWN (weak
        reinforce if already there) or SURPRISE (unlearn wrong +
        learn correct). Also writes into Pack 273 anchor-action
        cache (atomic facts only -- Pack 283 routes compositional
        queries to substrate recompute, avoiding cache bloat).

        Uses explicit \n\n segment boundaries so split_pairs
        guarantees a clean (state, action) pair from the chain.
        """
        # Pack 283: compositional queries (math) cost nothing to
        # recompute via Pack 254/291 RHC; their cache entry duplicates
        # substrate behavior.  Skip the cache write for them.
        if is_compositional_query(query):
            self.stats.setdefault('cache_writes_skipped', 0)
            self.stats['cache_writes_skipped'] += 1
            self.stats['dopamine_fires'] += 1
            return
        chain = f'{query}\n\n{expected_answer}\n\n'
        self.org.cat4.absorb_chain(chain)
        self.stats['dopamine_fires'] += 1

    def _swarm_predict(self, query_toks, n_particles=8, sigma=0.05, top_k=5):
        """Pack 288 -- parallel particle-swarm cleanup.

        Stack N noise perturbations into a single (N, d) query matrix
        and route through Pack 280's vectorized recall cache in one
        matmul: cost = O(N * N_anchors * d) batched vs O(N) sequential
        recall_action calls.

        Two-tier resolution (mirrors Pack 287 _substrate_distribution):
          1. Pack 273 cache fast-path -- query's deterministic anchor
             present?  Delta consensus at cached token, no swarm.
          2. Otherwise, swarm: N noisy focus_hv variants, vote on
             top-1 cleanup per particle.

        Cache fast-path is the dominant production behavior for
        well-taught queries; swarm primarily fires during the teaching
        loop when substrate has no anchor yet.

        Returns dict {token: vote_count} aggregated across particles +
        per-particle (token, sim) detail.  Caller decides whether to
        argmax on votes or accept any HIT.
        """
        import numpy as np
        cat4 = self.org.cat4
        anchor_actions = getattr(cat4, 'anchor_actions', {}) or {}
        # Tier 1 -- Pack 273 fast path
        from ikigai.cognition.cat4_absorb import _stable_anchor
        q_anchor = _stable_anchor(list(query_toks))
        cache_hit = anchor_actions.get(q_anchor)
        if cache_hit and cache_hit[0]:
            tok = str(cache_hit[0][-1]).lower()
            return {
                'votes': {tok: int(n_particles)},
                'particles': [(tok, 1.0)] * int(n_particles),
                'top_token': tok,
                'consensus_sim': 1.0,
                'cache_hit': True,
            }
        # Tier 2 -- swarm
        if cat4._pack280_recall_states is None:
            cat4._pack280_build_recall_cache()
        if not cat4._pack280_recall_anchors:
            return {'votes': {}, 'particles': [],
                     'top_token': None, 'consensus_sim': 0.0,
                     'cache_hit': False}
        base = cat4.focus_hv(query_toks).astype(np.complex64)
        d = cat4.mr.d
        # Generate N deterministic per-particle phase twists.
        # Particle 0 = no noise (substrate's clean prediction).
        salt = int(time.perf_counter_ns() & 0xffff_ffff)
        rng = np.random.default_rng(salt)
        twists = [np.ones(d, dtype=np.complex64)]    # particle 0 clean
        for p in range(1, int(n_particles)):
            phase_noise = rng.normal(0.0, float(sigma), size=d).astype(
                np.float32)
            twists.append(np.exp(1j * phase_noise).astype(np.complex64))
        # Stack as (N, d) query batch
        queries = np.stack([base * t for t in twists], axis=0)  # (N, d)
        # Single matmul: (N, d) @ (N_anchors, d).conj().T
        # =>  (N, N_anchors) state_sims per particle
        stored = cat4._pack280_recall_states
        dots = queries @ stored.conj().T
        sims_all = (np.real(dots) / float(d)).astype(np.float32)
        # Per particle, take top-1 prediction
        votes = {}
        particles = []
        anchor_actions = getattr(cat4, 'anchor_actions', {}) or {}
        anchors_list = cat4._pack280_recall_anchors
        for p in range(int(n_particles)):
            sims = sims_all[p]
            top_idx = int(np.argmax(sims))
            anchor = anchors_list[top_idx]
            entries = anchor_actions.get(anchor)
            if entries and entries[-1]:               # Pack 330: last-wins value
                tok = str(entries[-1][-1]).lower()
            else:
                tok = ''
            particles.append((tok, float(sims[top_idx])))
            votes[tok] = votes.get(tok, 0) + 1
        # Consensus: argmax over vote counts; tie-break by mean sim
        sorted_tokens = sorted(votes.items(),
                                  key=lambda kv: (-kv[1],
                                                    -float(np.mean(
                                                        [s for t, s in particles
                                                         if t == kv[0]]))))
        top_token = sorted_tokens[0][0] if sorted_tokens else None
        return {
            'votes': votes,
            'particles': particles,
            'top_token': top_token,
            'consensus_sim': float(np.mean(
                [s for t, s in particles if t == top_token])) if top_token else 0.0,
            'cache_hit': False,
        }

    def _reason_with_noise(self, query, noise_scale):
        """Same as gr.reason(query) but inject phase noise into the
        focus_hv state encoding. Implements 'noradrenergic gain
        modulation' that re-explores cleanup when first try wrong.

        noise_scale=0 -> identical to gr.reason(query).
        noise_scale=0.1 -> small phase perturbations per dim.
        noise_scale=0.5 -> heavy exploration; substrate may collapse
                            to nearest different anchor.

        Implementation: temporarily monkeypatch cat4.focus_hv to
        multiply each phase by `exp(i * noise)`. Restored after
        the reason() call.
        """
        if noise_scale <= 0:
            return self.org.general_reasoner.reason(query)

        cat4 = self.org.cat4
        original_focus_hv = cat4.focus_hv
        d = self.org.unified.d
        # Deterministic noise per (query, attempt) so retries with
        # the same scale don't trivially repeat the same wrong path.
        # We mix in nanoseconds so iterative retries diverge.
        salt = int(time.perf_counter_ns() & 0xffff_ffff)
        rng = np.random.default_rng(salt)
        phase_noise = rng.normal(0.0, float(noise_scale), size=d).astype(
            np.float32)
        twist = np.exp(1j * phase_noise).astype(np.complex64)

        def noisy_focus_hv(tokens):
            hv = original_focus_hv(tokens)
            return (hv * twist).astype(np.complex64)

        try:
            cat4.focus_hv = noisy_focus_hv
            result = self.org.general_reasoner.reason(query)
        finally:
            cat4.focus_hv = original_focus_hv
        return result

    # ---- public API -------------------------------------------------

    def reason_until(self, query, expected,
                       max_iters=10,
                       noise_scale_init=0.05,
                       noise_growth=1.4,
                       verbose=False):
        """Predict, check, reinforce / retry until correct or budget
        exhausted. Returns dict with the final state.

        On success: ALWAYS fires dopamine on the FINAL successful
        pair, so even first-try-correct strengthens the binding.

        Skips silently when `expected` is None (no oracle answer
        available) -- nothing to teach against.
        """
        if expected is None or not str(expected).strip():
            return {
                'query': query, 'expected': expected,
                'predicted': '', 'iters_used': 0,
                'converged': False, 'final_noise': 0.0,
                'skipped': True,
            }
        self.stats['queries'] += 1
        last = None
        noise = 0.0      # first attempt = NO noise (substrate's clean prediction)
        for it in range(int(max_iters)):
            result = self._reason_with_noise(query, noise)
            pred = (result.get('icl_action')
                      or result.get('answer') or '')
            last = result
            ok = _matches_expected(pred, expected)
            if verbose:
                print(f'    iter {it+1:>2d} noise={noise:.3f} '
                       f'pred={pred!r:<25s} ok={ok}')
            if ok:
                if it == 0:
                    self.stats['first_try_correct'] += 1
                else:
                    self.stats['corrected_after_retry'] += 1
                # Dopamine: reinforce the (query, expected) binding
                self._dopamine_reinforce(query, expected)
                self.stats['total_iters'] += it + 1
                return {
                    'query': query, 'expected': expected,
                    'predicted': pred, 'iters_used': it + 1,
                    'converged': True, 'final_noise': noise,
                }
            # Wrong: ramp noise and retry
            self.stats['noise_fires'] += 1
            noise = noise_scale_init if noise == 0.0 else (
                noise * noise_growth)
        self.stats['failed'] += 1
        self.stats['total_iters'] += int(max_iters)
        # Pre-commitment: even on failure, write the expected pair so
        # next encounter has substrate signal. This is the "the teacher
        # told me, I should learn it" finalize.
        self._dopamine_reinforce(query, expected)
        return {
            'query': query, 'expected': expected,
            'predicted': (last and (last.get('icl_action')
                                       or last.get('answer'))) or '',
            'iters_used': int(max_iters),
            'converged': False, 'final_noise': noise,
        }

    def teach_with_oracle(self, queries, oracle,
                            max_iters=8,
                            noise_scale_init=0.05,
                            noise_growth=1.4,
                            oracle_batch_size=16,
                            verbose=False):
        """Teacher-supervised loop. For each query, fetch ground truth
        from the oracle (RemoteLLMTeacher wrapper), then run the
        dopamine + noisy-retry loop until substrate matches.

        Substrate predicts -> oracle says truth -> if match, dopamine;
        else noisy retry until match. After exhausting retries, the
        teacher's answer gets persisted via pre-commit dopamine write.

        Pack 281: oracle answers are pre-fetched in a single batched
        HTTP call (vLLM continuous batching).  ~25x throughput on the
        oracle leg, dominant cost shifts to substrate dopamine loop
        (which Pack 280 vectorize brought from 28s -> ~80ms/iter).
        """
        # Pack 281 batched oracle pre-fetch
        truths = {}
        if queries:
            ask_batch = getattr(oracle, 'ask_batch', None)
            if callable(ask_batch):
                truths = ask_batch(queries, batch_size=oracle_batch_size)
                if verbose:
                    hits = sum(1 for v in truths.values() if v)
                    print(f'  oracle batch: {hits}/{len(queries)} answers')
            else:
                # Old single-shot oracle fallback
                for q in queries:
                    truths[str(q).strip()] = oracle.ask(q)
        results = []
        for q in queries:
            truth = truths.get(str(q).strip())
            if truth is None:
                if verbose:
                    print(f'  oracle silent on {q!r}; skipping')
                continue
            if verbose:
                print(f'\n  Q: {q!r}\n  oracle answer: {truth!r}')
            r = self.reason_until(q, truth,
                                    max_iters=max_iters,
                                    noise_scale_init=noise_scale_init,
                                    noise_growth=noise_growth,
                                    verbose=verbose)
            r['oracle_answer'] = truth
            results.append(r)
        return {
            'queries': len(queries),
            'taught': len(results),
            'first_try_correct': self.stats['first_try_correct'],
            'corrected': self.stats['corrected_after_retry'],
            'failed': self.stats['failed'],
            'avg_iters':
                self.stats['total_iters'] / max(self.stats['queries'], 1),
            'dopamine_fires': self.stats['dopamine_fires'],
            'noise_fires': self.stats['noise_fires'],
            'results': results,
        }

    def teach_batch(self, pairs, **kwargs):
        """Run reason_until on each (query, expected) pair. Returns
        a dict with per-pair results + aggregate stats."""
        results = []
        for q, e in pairs:
            r = self.reason_until(q, e, **kwargs)
            results.append(r)
        return {
            'pairs': len(pairs),
            'first_try_correct': self.stats['first_try_correct'],
            'corrected': self.stats['corrected_after_retry'],
            'failed': self.stats['failed'],
            'avg_iters':
                self.stats['total_iters'] / max(self.stats['queries'], 1),
            'dopamine_fires': self.stats['dopamine_fires'],
            'noise_fires': self.stats['noise_fires'],
            'results': results,
        }


# Pack 287 -- KL-graded RPE.  Replaces substring _matches_expected with a
# continuous fire signal = exp(-KL(teacher || substrate_softmax)).  v0 here
# approximates teacher as a one-hot mass at the expected token, reducing
# KL to -log(P_sub[expected]) and fire to P_sub[expected] directly.  Full
# v1 with vLLM logprobs cross-vocab alignment is Pack 287.5 follow-up.
#
# Schultz 1997 measured graded VTA firing rates -- bool substring match
# was the biological violation.  exp(-KL) gives [0, 1] continuous reward.


import re as _re

_TOK_SPLIT_RE = _re.compile(r'[a-z]+|-?\d+')


class KLDopamine(Cat4Dopamine):
    """Pack 287 v0 -- continuous KL-graded dopamine fire on top of Pack
    276's noisy-retry loop.

    Adds:
      * `_substrate_distribution(toks, top_k)` -- softmax over cleanup
        state_sims with calibrated temperature tau.
      * `_kl_fire(toks, expected)` -- exp(-KL(teacher_one_hot || P_sub))
        = P_sub at the expected action token.
      * `reason_until_graded(query, expected)` -- like reason_until but
        records fire signal at each iter; converged when fire > fire_thr.

    tau calibration: held-out set should give P_sub entropy matched to
    teacher entropy.  Default tau=0.5 is a starting point; Pack 287.5
    will calibrate from data.
    """

    def __init__(self, organism, tau=0.5, fire_threshold=0.30):
        super().__init__(organism)
        self.tau = float(tau)
        self.fire_threshold = float(fire_threshold)
        self.stats.update({
            'graded_fires': 0,
            'sum_fire': 0.0,
            'max_fire_seen': 0.0,
        })

    def _substrate_distribution(self, query_toks, top_k=10):
        """Return (P, tokens): substrate's prediction distribution.

        Two-tier resolution (mirrors the reason() runtime path):
        1. Stable-anchor cache lookup -- if the query's deterministic
           anchor sits in `anchor_actions`, return P as a delta at the
           cached token.  This is the Pack 273 fast path.
        2. Otherwise softmax over recall_action top-K state_sims.
           Tokens come from anchor_actions on the returned anchors when
           present, or fall back to cleanup against the action codebook
           when not.
        """
        import numpy as np
        cat4 = self.org.cat4
        anchor_actions = getattr(cat4, 'anchor_actions', {}) or {}
        # Pack 273 fast path -- deterministic anchor on the query
        from ikigai.cognition.cat4_absorb import _stable_anchor
        q_anchor = _stable_anchor(list(query_toks))
        cache_hit = anchor_actions.get(q_anchor)
        if cache_hit and cache_hit[0]:
            tok = str(cache_hit[0][-1]).lower()
            p = np.array([1.0], dtype=np.float32)
            return p, [tok]
        # Slow path -- cleanup top-K with cached tokens
        results = cat4.recall_action(query_toks, top_k=top_k)
        if not results:
            return None, []
        sims = np.array([float(r[2]) for r in results], dtype=np.float32)
        z = sims / max(self.tau, 1e-6)
        z = z - float(z.max())
        p = np.exp(z)
        p = p / float(p.sum())
        # Fallback to action codebook cleanup for tokens whose anchor
        # isn't in the cache (most of the 63K Pack 262 v5 anchors).
        tokens = []
        try:
            vocab, K = cat4.action_codebook()
        except Exception:
            vocab, K = [], None
        for anchor, action_hv, _ in results:
            entries = anchor_actions.get(anchor)
            if entries and entries[-1]:               # Pack 330: last-wins value
                tokens.append(str(entries[-1][-1]).lower())
                continue
            if K is not None and len(vocab) > 0:
                # cosine vs codebook -- pick argmax token
                scores = np.real(K.conj() @ action_hv).astype(np.float32)
                tokens.append(str(vocab[int(scores.argmax())]).lower())
            else:
                tokens.append('')
        return p, tokens

    def _kl_fire(self, query_toks, expected, top_k=10,
                   teacher_distribution=None):
        """Fire signal = exp(-KL(teacher || P_sub)).

        v0 (default, teacher_distribution=None): teacher modeled as
        a delta at the expected token; KL reduces to -log(P_sub[exp])
        and fire = substrate mass at the matching token.

        v1 (Pack 287.5, teacher_distribution=[(tok, logprob), ...]):
        full KL using the teacher's first-token distribution from
        vLLM logprobs.  KL(T||P) = sum_t T(t) * log(T(t)/P(t)) where
        P(t) is the substrate softmax mass on the matching token and
        T(t) is the teacher's probability.  Fire = exp(-KL).
        """
        if not expected:
            return 0.0
        target = str(expected).strip().lower()
        toks = _TOK_SPLIT_RE.findall(target)
        target_tok = toks[-1] if toks else target
        p_sub, sub_tokens = self._substrate_distribution(
            query_toks, top_k=top_k)
        if p_sub is None:
            return 0.0
        # Build substrate prob lookup dict
        import math
        import numpy as np
        sub_lookup = {}
        for tok, prob in zip(sub_tokens, p_sub):
            if not tok:
                continue
            sub_lookup[tok] = sub_lookup.get(tok, 0.0) + float(prob)
        if teacher_distribution is None:
            # v0 path -- one-hot at expected
            mass = 0.0
            for tok, prob in sub_lookup.items():
                if tok == target_tok or target_tok in tok or tok in target_tok:
                    mass += prob
            return float(mass)
        # v1 path -- continuous KL against teacher distribution
        # Normalize teacher distribution into probs that sum to <= 1
        # (logprobs above are top-N, missing mass goes to "other").
        teacher_probs = {}
        for tok, lp in teacher_distribution:
            t = str(tok).strip().lower()
            teacher_probs[t] = teacher_probs.get(t, 0.0) + math.exp(float(lp))
        # Compute KL(teacher || sub) over the teacher's support
        kl = 0.0
        for t, tp in teacher_probs.items():
            if tp <= 0.0:
                continue
            sp = max(sub_lookup.get(t, 0.0), 1e-6)
            kl += tp * math.log(tp / sp)
        # Pack 287.5 v2 -- the teacher distribution is partial (top-N,
        # missing mass -> "other") and the substrate is often a delta
        # (cache fast path), which can drive KL negative.  Clamp at 0 so
        # fire stays in [0, 1] and v1 fire <= v0 fire as designed.
        kl = max(kl, 0.0)
        return float(math.exp(-kl))

    def reason_until_graded(self, query, expected,
                              max_iters=10,
                              noise_scale_init=0.05,
                              noise_growth=1.4,
                              verbose=False):
        """Pack 287 graded analogue of reason_until.

        Converged condition: fire >= fire_threshold (vs bool substring).
        Dopamine reinforcement scaled by fire (graded VTA firing).
        """
        if expected is None or not str(expected).strip():
            return {
                'query': query, 'expected': expected,
                'predicted': '', 'iters_used': 0,
                'converged': False, 'final_noise': 0.0,
                'final_fire': 0.0, 'skipped': True,
            }
        self.stats['queries'] += 1
        # Tokenize query the same way cat4 does so the cache anchor
        # lines up
        q_toks = _TOK_SPLIT_RE.findall(str(query).lower())
        last_result = None
        last_fire = 0.0
        noise = 0.0
        for it in range(int(max_iters)):
            result = self._reason_with_noise(query, noise)
            last_result = result
            pred = (result.get('icl_action')
                      or result.get('answer') or '')
            fire = self._kl_fire(q_toks, expected)
            last_fire = fire
            self.stats['sum_fire'] += fire
            if fire > self.stats['max_fire_seen']:
                self.stats['max_fire_seen'] = fire
            if verbose:
                print(f'    iter {it+1:>2d} noise={noise:.3f} '
                       f'pred={pred!r:<22s} fire={fire:.3f}')
            if fire >= self.fire_threshold:
                if it == 0:
                    self.stats['first_try_correct'] += 1
                else:
                    self.stats['corrected_after_retry'] += 1
                self.stats['graded_fires'] += 1
                self._dopamine_reinforce(query, expected)
                self.stats['total_iters'] += it + 1
                return {
                    'query': query, 'expected': expected,
                    'predicted': pred, 'iters_used': it + 1,
                    'converged': True, 'final_noise': noise,
                    'final_fire': fire,
                }
            self.stats['noise_fires'] += 1
            noise = noise_scale_init if noise == 0.0 else (
                noise * noise_growth)
        self.stats['failed'] += 1
        self.stats['total_iters'] += int(max_iters)
        # Pre-commit dopamine even on graded failure
        self._dopamine_reinforce(query, expected)
        pred = ''
        if last_result is not None:
            pred = (last_result.get('icl_action')
                      or last_result.get('answer') or '')
        return {
            'query': query, 'expected': expected,
            'predicted': pred,
            'iters_used': int(max_iters),
            'converged': False, 'final_noise': noise,
            'final_fire': last_fire,
        }


# Pack 289 -- classifier-free trait conditioning (from Faldor + Cully
# 2024, Zhang HADES 2024).  Adds trait targets on top of Pack 287 KL
# fire so the dopamine signal scales by both (a) substrate confidence
# in the answer token AND (b) whether the predicted answer matches the
# expected TRAIT class (digit vs word, single token vs sentence,
# yes-no vs city vs number).
#
# Currently substrate's icl_action is short by construction (single
# token from anchor_actions cache).  The trait gate is more useful
# when wired against the teacher's predicted distribution (Pack 287.5
# vLLM logprobs) -- here it serves as a sanity guard against
# misclassified responses.


import string as _string

# Trait classes -- exhaustively covers Day 76 bench + future absorb.
TRAIT_DIGIT = 'digit'          # answer is integer / numeric
TRAIT_YES_NO = 'yes_no'         # answer is 'yes' or 'no'
TRAIT_LOWERCASE_WORD = 'lower_word'   # bare lowercase content word
TRAIT_UNKNOWN = 'unknown'


def classify_trait(answer):
    """Bucket a teacher / substrate answer string by its TRAIT class.

    >>> classify_trait('8')
    'digit'
    >>> classify_trait('berlin')
    'lower_word'
    >>> classify_trait('YES')
    'yes_no'
    """
    if answer is None:
        return TRAIT_UNKNOWN
    a = str(answer).strip().lower()
    if not a:
        return TRAIT_UNKNOWN
    if a in ('yes', 'no'):
        return TRAIT_YES_NO
    if all(c in _string.digits + '-.' for c in a):
        return TRAIT_DIGIT
    if all(c in _string.ascii_lowercase + '-' for c in a):
        return TRAIT_LOWERCASE_WORD
    return TRAIT_UNKNOWN


class TraitConditionalDopamine(KLDopamine):
    """Pack 289 -- KL fire signal weighted by trait match.

    Composite fire = KL_fire × trait_match where trait_match in {0, 1}.
    Mismatched traits zero the fire even if substrate happens to score
    the right token in its top-K cleanup.  Catches misclassified
    responses ("paris" expected, substrate returns '42').
    """

    def _kl_fire(self, query_toks, expected, top_k=10):
        base = super()._kl_fire(query_toks, expected, top_k=top_k)
        if base <= 0.0:
            return 0.0
        expected_trait = classify_trait(expected)
        # Find substrate's top predicted token (highest P)
        import numpy as np
        p, tokens = self._substrate_distribution(query_toks, top_k=top_k)
        if p is None or len(tokens) == 0:
            return 0.0
        top_idx = int(np.argmax(p))
        predicted_top = tokens[top_idx]
        predicted_trait = classify_trait(predicted_top)
        # Hard gate (Pack 289 v0): same trait or skip
        if expected_trait != TRAIT_UNKNOWN and predicted_trait != expected_trait:
            # Substrate confident on the wrong trait -- zero fire,
            # noisy retry will fire on the next iter
            return 0.0
        return base
