"""
ikigai.cognition.self_verifier -- Self-Verification Loop.

Day 55 Pack 45 -- Layer 4 Coherence + Layer 3 Reasoning completion.

Problem: system emits responses without checking quality.
         Low-coherence responses accepted as-is.
         No mechanism to detect incoherence and regenerate.

Fix: SelfVerifier wraps expansion step.
     verify_coherence(response_tokens, B_U) -> (pass, score)
     verify_and_select(candidates, B_U)     -> best candidate by belief cosine
     verify_consistency(tokens, crystal)    -> (pass, contradictions)

Integration into chat() pipeline:
  After ngram expansion, generate top-k candidates via belief-steered
  expander with different seeds, verify_and_select picks best.

Improvement guarantee: cosine(selected, B_U) >= cosine(argmax, B_U)
                       by definition (we pick maximum).
No overhead if top-k=1 (falls back to direct expansion).
"""

import numpy as np

COHERENCE_THRESHOLD = 0.25  # minimum cosine(response_hv, B_U) to pass


def _hv(word, d):
    seed = hash(f'bspm::{word}') & 0x7FFFFFFF
    rng  = np.random.RandomState(seed)
    v    = (rng.randint(0, 2, size=d) * 2 - 1).astype(np.float32)
    n    = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def _encode_phrase(tokens, d):
    """L2-normalized sum of word HVs (same space as BSPM B_U)."""
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    s = np.zeros(d, dtype=np.float32)
    for t in tokens:
        s += _hv(t, d)
    n = float(np.linalg.norm(s))
    return s / (n + 1e-12)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SelfVerifier:
    """
    Self-verification loop for response quality control.

    verify_coherence(tokens, B_U)             -- cosine(resp_hv, B_U) >= threshold
    verify_consistency(tokens, crystal)        -- no contradictions in ACCI
    verify_and_select(candidates, B_U)         -- pick best-coherence candidate
    verify_beam(seed, B_U, expander, k=5)      -- generate k candidates, select best

    Quality guarantee: selected response cosine >= any single-draw response cosine.
    """

    def __init__(self, d=400, threshold=COHERENCE_THRESHOLD):
        self.d         = d
        self.threshold = threshold

        # Stats
        self.total_verified   = 0
        self.total_passed     = 0
        self.total_selections = 0   # times non-first candidate won
        self.coherence_log    = []  # (score_before, score_after) per beam select

    # ── coherence verification ─────────────────────────────────────────────

    def verify_coherence(self, response_tokens, B_U):
        """
        Check cosine(phrase_hv(response), B_U) >= threshold.
        Returns (pass: bool, score: float).
        """
        d_actual = B_U.shape[0]
        resp_hv = _encode_phrase(response_tokens, d_actual)
        score   = _cosine(resp_hv, B_U)
        ok      = score >= self.threshold
        self.total_verified += 1
        if ok:
            self.total_passed += 1
        return ok, float(score)

    # ── consistency verification ───────────────────────────────────────────

    def verify_consistency(self, response_tokens, crystal):
        """
        Check response against ACCI crystal for contradictions.
        A contradiction: (w, 'NOT', w2) in crystal AND w2 in response_tokens.
        Returns (pass: bool, contradictions: list).
        """
        token_set = set(response_tokens)
        contradictions = []
        for key in crystal._counts:
            s, p, o = key
            if p == 'NOT' and s in token_set and o in token_set:
                contradictions.append(key)
        ok = len(contradictions) == 0
        return ok, contradictions

    # ── beam selection ─────────────────────────────────────────────────────

    def verify_and_select(self, candidates, B_U):
        """
        Pick candidate with highest cosine(phrase_hv, B_U).
        candidates: list of token lists.
        Returns (best_tokens, best_score, winning_idx).
        """
        best_tokens = candidates[0]
        best_score  = -2.0
        best_idx    = 0

        score_first = None
        for i, tokens in enumerate(candidates):
            _, score = self.verify_coherence(tokens, B_U)
            if score_first is None:
                score_first = score
            if score > best_score:
                best_score  = score
                best_tokens = tokens
                best_idx    = i

        if best_idx > 0:
            self.total_selections += 1

        self.coherence_log.append((score_first, best_score))
        return best_tokens, best_score, best_idx

    def verify_beam(self, seed_words, B_U, expander, k=5, max_len=8):
        """
        Generate k candidates from expander with perturbed seeds.
        Select best by coherence. Returns (best_tokens, score, gains).
        Requires expander with expand_belief() method.
        """
        candidates = []

        # Candidate 0: base argmax
        base = expander.expand(seed_words, max_len=max_len)
        candidates.append(base)

        # Candidates 1..k-1: belief-steered from B_U variants
        d_actual = B_U.shape[0]
        for i in range(1, k):
            # Perturb B_U slightly: add small noise, renormalize
            noise = np.random.RandomState(i).standard_normal(d_actual).astype(np.float32) * 0.1
            B_perturb = B_U + noise
            n = float(np.linalg.norm(B_perturb))
            B_perturb = B_perturb / (n + 1e-12)
            cand = expander.expand_belief(seed_words, B_perturb, max_len=max_len)
            candidates.append(cand)

        best_tokens, best_score, best_idx = self.verify_and_select(candidates, B_U)

        # Gain = best_score - score(candidate_0)
        _, score_0 = self.verify_coherence(candidates[0], B_U)
        gain = best_score - score_0
        return best_tokens, best_score, gain

    # ── stats ──────────────────────────────────────────────────────────────

    def pass_rate(self):
        if self.total_verified == 0:
            return 0.0
        return self.total_passed / self.total_verified

    def selection_rate(self):
        """Fraction of beam calls where non-first candidate won."""
        n = len(self.coherence_log)
        if n == 0:
            return 0.0
        return self.total_selections / n

    def mean_gain(self):
        """Mean (best_score - first_score) across all beam calls."""
        if not self.coherence_log:
            return 0.0
        return float(np.mean([b - a for a, b in self.coherence_log]))

    def reset_stats(self):
        self.total_verified   = 0
        self.total_passed     = 0
        self.total_selections = 0
        self.coherence_log    = []
