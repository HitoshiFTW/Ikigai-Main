"""
ikigai.cognition.calibration -- Pack 310 (Day 80, framework line #11).

CALIBRATION: the organism knows what it does NOT know.

This is the framework's differentiator vs LLMs.  An LLM hallucinates
confidently; a tiny edge brain that returns "unknown" -- and is RIGHT
about not knowing -- is more trustworthy than a 400 GB model that bluffs.
Calibration is not a feature, it is the line that makes "beats LLMs"
credible, and it is a pure, DATA-FREE mathematical line.

THE BOUNDARY (derived from substrate geometry, not tuned)
---------------------------------------------------------
In FHRR every stored item is a unit-magnitude phasor HV in C^d.  A clean
cleanup similarity is sim = Re(<a, b>) / d.  For two INDEPENDENT random
HVs (i.e. a query for an ABSENT fact -- the noise floor) each of the d
terms is cos(theta) with theta uniform on [0, 2pi):
    E[cos theta]   = 0
    Var[cos theta] = 1/2
so sim is a mean of d such terms ->
    mean(sim)  = 0
    std(sim)   = sqrt(d * 1/2) / d = 1 / sqrt(2 d)         <-- noise floor

A PRESENT fact recalls far above this floor (exact anchor hit -> 1.0;
structured soft hit -> 0.7-0.9).  So the abstain boundary is k standard
deviations above the zero-mean floor:
    boundary(d, k) = k / sqrt(2 d)

k is a STATISTICAL CONFIDENCE LEVEL (a sigma-multiple), not a magic
threshold: under the Gaussian approximation k=4 gives a ~3e-5 false-accept
probability per candidate.  sigma comes purely from d.  Nothing here is
tuned to any dataset -- it is the geometry of the space.  This replaces
the old hand-tuned icl_min_state_sim=0.10 with a derived value
(boundary(400, 4) ~= 0.141).
"""

import math

# Default sigma-multiple.  4-sigma: under the Gaussian noise-floor model a
# random/absent query clears the boundary with probability ~3.2e-5.  Raise
# for stricter abstention (fewer false answers), lower for more recall.
DEFAULT_K = 4.0


def noise_floor_sigma(d):
    """Std-dev of the cleanup similarity for an ABSENT (random) query in a
    d-dimensional FHRR space.  Pure geometry: 1 / sqrt(2 d)."""
    d = int(d)
    if d <= 0:
        return 0.0
    return 1.0 / math.sqrt(2.0 * d)


def abstain_boundary(d, k=DEFAULT_K):
    """Similarity below which a recall is statistically indistinguishable
    from querying empty memory -> the organism must abstain.

    boundary = k * sigma, sigma = 1/sqrt(2 d).  k = sigma-multiple
    (confidence level), NOT a tuned magic number."""
    return float(k) * noise_floor_sigma(d)


def false_accept_prob(k=DEFAULT_K):
    """Gaussian-tail probability that an absent query clears a k-sigma
    boundary -- the calibration's stated false-accept rate PER CANDIDATE."""
    return 0.5 * math.erfc(float(k) / math.sqrt(2.0))


# Pack 319 (Day 80) -- MULTIPLE-COMPARISON correction.
#
# abstain_boundary(d, k) is the per-comparison floor: correct when the recall
# is scored against ONE candidate (e.g. a single ICL state-sim, Pack 310).
# But cleanup over a vocabulary takes the ARGMAX over N candidates, and the
# maximum of N independent noise sims is much larger than one draw:
#     E[max of N]  ~=  sigma * sqrt(2 ln N)
# so with a big vocab the noise MAX clears the per-comparison boundary and an
# ABSENT subject looks present (limit-test finding: ~98% false-accept at
# N=1000).  The argmax-safe boundary applies an extreme-value / Bonferroni
# correction: to hold the family-wise false-accept at alpha over N candidates,
#     theta(d, N, alpha) = sigma * sqrt( 2 * ln(N / alpha) )
# This is still derived geometry (sigma = 1/sqrt(2d)); alpha is the target
# false-accept rate (a confidence level, not a tuned threshold).

DEFAULT_ALPHA = 1e-3


def abstain_boundary_n(d, n_candidates, alpha=DEFAULT_ALPHA):
    """Argmax-safe abstain boundary for cleanup over `n_candidates`.

    theta = sigma * sqrt(2 * ln(N / alpha)), sigma = 1/sqrt(2 d).  Use this
    (not abstain_boundary) whenever the recall sim is the MAX over a candidate
    vocabulary, so the noise-floor maximum does not masquerade as a hit."""
    n = max(int(n_candidates), 1)
    sigma = noise_floor_sigma(d)
    z = math.sqrt(2.0 * math.log(max(n / float(alpha), math.e)))
    return float(sigma * z)


# Pack 320 (Day 80) -- EMPIRICAL calibration.
#
# The theoretical floors above assume absent queries read PURE Gaussian noise.
# On a populated SDM the absent read is CROSSTALK from the real stored vectors,
# which is structured and larger than the Gaussian model (limit-test 319: a
# residual false-accept floor the theory misses).  The fix is substrate-native:
# MEASURE the absent-query similarity distribution on THIS bank and set the
# boundary from it -- same k-sigma form, but sigma is the bank's OWN measured
# noise (crosstalk included), not 1/sqrt(2d).  Nothing is hand-tuned; the
# number is read off the substrate.

def empirical_boundary(absent_sims, k=DEFAULT_K):
    """Abstain boundary measured from a sample of ABSENT-query top-sims.

    boundary = mean(absent_sims) + k * std(absent_sims)
    k is the same confidence multiple; mean/std are the bank's real noise +
    crosstalk floor.  Returns the theoretical-free, substrate-measured
    threshold below which a recall is indistinguishable from this bank's
    empty-memory response."""
    a = [float(s) for s in absent_sims]
    if not a:
        return 0.0
    n = len(a)
    mu = sum(a) / n
    var = sum((s - mu) ** 2 for s in a) / max(n - 1, 1)
    return float(mu + float(k) * math.sqrt(var))


# Day 102 -- DISTRIBUTION-FREE FINITE-SAMPLE abstention (conformal / selective
# risk control), the calibrated complement to the geometry boundaries above.
#
# WHY, on top of the geometry/empirical boundaries: once recall goes through an
# APPROXIMATE top-k index (Day-102 SPANN-style code index -> factored_meaning),
# the retriever ALWAYS returns a nearest item, even for an out-of-store query,
# and a k-sigma geometry threshold gives no finite-sample GUARANTEE on the
# false-accept rate over that index.  Conformal prediction turns ANY score
# (code agreement, cleanup cosine, unbinding residue) into a threshold tau with
# a certified risk bound, assuming only exchangeability -- no distributional
# model, no neural head, no gradient.
#
# (An external assessment framed this as 'SCRC-I'; that specific acronym is
# unverified -- but conformal prediction, selective risk control, and the
# Hoeffding upper confidence bound below are all standard, corroborated methods.)

class ConformalAbstain:
    """Inductive conformal abstention over a retriever confidence score.

    THE GUARANTEE (the honest, achievable one -- a FALSE-ALARM bound):
    calibrate(scores, correct) picks a single scalar threshold tau so that, for
    an exchangeable OUT-OF-STORE / unanswerable query (one where answering would
    be wrong), P(score >= tau) <= alpha.  I.e. the rate at which the organism
    ANSWERS something it should have abstained on is bounded by alpha, with EXACT
    finite-sample validity under exchangeability alone -- no distribution model,
    no neural head, no gradient.  accept(score) answers iff score >= tau, else
    abstains.  (This bounds the false-ANSWER rate on unknowns -- P(accept|wrong);
    it does NOT by itself bound P(wrong|accept), which is base-rate dependent.
    False-alarm control is exactly what 'correct-or-abstain over a top-k index'
    needs: an ANN index always returns a nearest item, and this caps how often a
    spurious one is trusted.)

    METHOD: one-sided conformal quantile of the should-abstain (wrong) scores --
    tau = the ceil((n+1)(1-alpha))-th smallest of the n wrong-example scores.
    Standard split-conformal; exact, no CI, no multiplicity search (an earlier
    Hoeffding-selective-risk scan abstained on everything -- wrong tool).

    (An external assessment framed this as 'SCRC-I'; that acronym is unverified,
    but split-conformal prediction and one-sided conformal p-values are standard,
    corroborated methods.)"""

    def __init__(self, alpha=0.05, delta=0.05):
        self.alpha = float(alpha)
        self.delta = float(delta)          # kept for API/compat; conformal is exact, unused
        self.tau = float('inf')            # uncalibrated -> abstain on everything (safe)
        self.answer_rate = 0.0
        self.n_calib = 0

    def calibrate(self, scores, correct):
        """scores: retriever confidence per calibration query (higher = more
        confident).  correct: truthy iff ANSWERING that query would be right (an
        absent/out-of-store query is always False -- answering it is a false
        alarm).  Calibrates tau on the wrong (should-abstain) scores.  Returns tau."""
        scores = [float(s) for s in scores]
        neg = sorted(s for s, c in zip(scores, correct) if not c)   # wrong-example scores, ascending
        self.n_calib = len(neg)
        n = len(neg)
        if n == 0:
            self.tau = float('-inf')       # nothing should be rejected -> answer all
        else:
            rank = math.ceil((n + 1) * (1.0 - self.alpha))   # one-sided conformal quantile
            self.tau = neg[rank - 1] if rank <= n else float('inf')  # alpha < 1/(n+1) -> abstain all
        self.answer_rate = (sum(1 for s in scores if s > self.tau) / len(scores)
                            if scores else 0.0)
        return self.tau

    def accept(self, score):
        """Answer iff confident enough; else abstain.  STRICT '>' -- the scores
        are discretized (code agreement is k/bits), so ties pile up at tau; strict
        acceptance excludes that tie mass and keeps the false-alarm bound
        conservative (FAR <= alpha) instead of letting ties over-accept it."""
        return float(score) > self.tau

    def state_dict(self):
        return {'alpha': self.alpha, 'delta': self.delta, 'tau': self.tau,
                'answer_rate': self.answer_rate, 'n_calib': self.n_calib}

    def load_state(self, st):
        self.alpha = float(st.get('alpha', self.alpha))
        self.delta = float(st.get('delta', self.delta))
        self.tau = float(st.get('tau', float('inf')))
        self.answer_rate = float(st.get('answer_rate', 0.0))
        self.n_calib = int(st.get('n_calib', 0))
        return self.tau
