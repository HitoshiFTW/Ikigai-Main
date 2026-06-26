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
