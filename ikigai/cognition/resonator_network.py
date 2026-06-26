"""
ikigai.cognition.resonator_network -- Pack 256 hierarchical Resonator Network.

Day 74. Multi-factor decomposition for bound FHRR phasor vectors.
Implementation of Frady & Sommer (2020) "Resonator Networks for
Factoring Distributed Representations of Data Structures."

WHY THIS EXISTS
---------------
The existing single-pass `MultiRoleMemory.resonator_recall` (Pack 224,
Day 67) does ONE-factor cleanup: given a noisy HV, snap to nearest
codebook entry. It cannot decompose:

    C = bind(A1, B3, C7)   ->   recover (A1, B3, C7) separately.

Pack 256 fills that gap. Each factor maintains a belief estimate;
factors update by mean-field iteration -- each unbinds C by the
current beliefs of all OTHER factors, then snaps to its own
codebook. Converges to the correct factorization with high
probability when codebook size is below the resonator capacity bound.

USE CASES
---------
- Pack 255 GeneralReasoner: decompose superposed state into role+token
- Pack 252 NumericEncoder: factor a magnitude*role binding into pieces
- Pack 253 cat-3 absorb: separate (subject, predicate, object) triples
- Invention 1 HPE: per-level decode lives here

NOT A NEW SUBSTRATE
-------------------
Pack 256 composes existing operations. Adds factorization capability
that single-pass cleanup cannot.

PACK 224 vs PACK 256
--------------------
Pack 224 (single-pass) -- snap noisy HV to one codebook. Used by
                          fsm.step everywhere.
Pack 256 (this)        -- decompose multi-bind into N factors using
                          N codebooks simultaneously.
"""

import numpy as np


class ResonatorNetwork:
    """Pack 256 Resonator Network (Frady & Sommer 2020).

    Factors a bound FHRR phasor HV into its N constituent codebook
    entries via mean-field iteration.

    USAGE
    -----
        codebooks = [{'A1': hv, 'A2': hv, ...},  # factor 1
                     {'B1': hv, 'B2': hv, ...},  # factor 2
                     ...]
        rn = ResonatorNetwork(d=400, codebooks=codebooks)
        result = rn.decode(bound_hv)
        # result = [('A1', 0.97), ('B3', 0.95), ('C7', 0.92)]
    """

    def __init__(self, d, codebooks, max_iters=30, beta=8.0,
                  momentum=0.5, conv_tol=1e-3):
        """
        Args:
            d            -- vector dimension
            codebooks    -- list of dicts {name: complex64[d]} per factor
            max_iters    -- max mean-field iterations
            beta         -- softmax sharpness for codebook cleanup
            momentum     -- old/new blend per iter (0.5 = balanced)
            conv_tol     -- convergence threshold (mean abs change)
        """
        self.d = int(d)
        self.codebooks = list(codebooks)
        self.n_factors = len(self.codebooks)
        self.max_iters = int(max_iters)
        self.beta = float(beta)
        self.momentum = float(momentum)
        self.conv_tol = float(conv_tol)
        # Precompute codebook matrices (N_i, d) per factor
        self._K = []
        self._names = []
        for cb in self.codebooks:
            names = list(cb.keys())
            K = np.stack([np.asarray(cb[n], dtype=np.complex64)
                           for n in names])
            self._K.append(K)
            self._names.append(names)
        self.stats = {'decodes': 0, 'mean_iters': 0.0,
                       'converged_runs': 0}

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _normalize(hv):
        """Renormalize phasor to unit average magnitude."""
        mag = float(np.abs(hv).mean()) + 1e-12
        return (hv / mag).astype(np.complex64)

    def _clean_to_codebook(self, target_hv, K):
        """Soft-projection: weighted sum of codebook entries by softmax
        cosine similarity to target. Returns reshaped factor estimate."""
        sims = np.real(K @ np.conj(target_hv)) / self.d  # (N,)
        logits = self.beta * sims
        logits -= logits.max()
        w = np.exp(logits).astype(np.float32)
        w /= (w.sum() + 1e-12)
        # weighted reconstruction
        new = (w[:, None] * K).sum(axis=0).astype(np.complex64)
        return self._normalize(new)

    def _argmax_codebook(self, target_hv, K, names):
        """Hard-decode: cosine vs codebook, return (name, score)."""
        sims = np.real(K @ np.conj(target_hv)) / self.d
        idx = int(np.argmax(sims))
        return names[idx], float(sims[idx])

    # ---- decode ------------------------------------------------------

    def decode(self, bound_hv, init=None, return_traces=False):
        """Factor bound_hv into N codebook entries.

        Args:
            bound_hv     -- complex64[d] target
            init         -- optional list of N HVs for initial estimates
                            (default = codebook mean per factor)
            return_traces -- if True also return per-iter state for
                              inspection

        Returns:
            list of (name, score) per factor in codebook order
        """
        self.stats['decodes'] += 1
        c = np.asarray(bound_hv, dtype=np.complex64)
        c = self._normalize(c)
        # Init: each factor = mean of its codebook OR provided
        if init is not None:
            factors = [self._normalize(np.asarray(h, dtype=np.complex64))
                        for h in init]
        else:
            factors = [self._normalize(self._K[i].mean(axis=0))
                        for i in range(self.n_factors)]

        traces = []
        it_used = 0
        converged = False
        for it in range(self.max_iters):
            it_used = it + 1
            max_delta = 0.0
            new_factors = []
            for i in range(self.n_factors):
                # Compute "what factor i should be" by unbinding bound_hv
                # by ALL other current factor estimates
                divisor = None
                for j in range(self.n_factors):
                    if j == i:
                        continue
                    divisor = factors[j] if divisor is None else (divisor * factors[j])
                if divisor is None:
                    # Single-factor case
                    target = c
                else:
                    divisor = self._normalize(divisor)
                    target = c * np.conj(divisor)
                target = self._normalize(target)
                # Snap to codebook i
                new_f = self._clean_to_codebook(target, self._K[i])
                # Momentum blend
                blended = (self.momentum * factors[i]
                            + (1.0 - self.momentum) * new_f)
                blended = self._normalize(blended)
                delta = float(np.abs(blended - factors[i]).mean())
                max_delta = max(max_delta, delta)
                new_factors.append(blended)
            factors = new_factors
            if return_traces:
                traces.append([f.copy() for f in factors])
            if max_delta < self.conv_tol:
                converged = True
                break

        # Hard decode each factor
        result = []
        for i in range(self.n_factors):
            name, sc = self._argmax_codebook(factors[i], self._K[i],
                                               self._names[i])
            result.append((name, sc))

        # Stats
        prev = self.stats['mean_iters'] * (self.stats['decodes'] - 1)
        self.stats['mean_iters'] = (prev + it_used) / self.stats['decodes']
        if converged:
            self.stats['converged_runs'] += 1

        if return_traces:
            return result, traces
        return result

    def stats_summary(self):
        n = max(self.stats['decodes'], 1)
        return {
            'decodes': self.stats['decodes'],
            'mean_iters': self.stats['mean_iters'],
            'convergence_rate': self.stats['converged_runs'] / n,
        }
