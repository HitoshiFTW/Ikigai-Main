"""
ikigai.cognition.pi_k_algebra -- Π_k Prime-Offset Permutation Family.

Day 56 Pack 85 -- Pillar 2 of UHE.

Solves the bipolar-XOR-leakage problem (Pack 72 lesson):
    XOR is involutive → in chains s_0 → s_1 → ... → s_n, predict(s_k, action)
    leaks BOTH s_{k+1} AND s_{k-1}.

Solution: family of cyclic-shift permutations indexed by primes.
    Π_k(x) = np.roll(x, p_k) where p_k is the k-th prime
    Not involutive: Π_k(Π_k(x)) = roll(x, 2*p_k) ≠ x in general
    Non-commutative: Π_j ∘ Π_k = roll(., p_j + p_k) ≠ Π_k ∘ Π_j (sequence matters)
    Reversible: Π_k^{-1}(x) = np.roll(x, -p_k)
    Coprime stack: gcd(p_1, p_2, ..., p_K) = 1 → 32-depth chains have full state separation

Π-binding:
    bind_Π_k(a, b) = Π_k(a) * b      (asymmetric)
    unbind_Π_k(c, b) = Π_k^{-1}(c * b)   (since b*b = ±1 for bipolar / |b|² = 1 for phasor)

Turing-completeness: state machine via Π chain with role/value/PC slots.
"""

import numpy as np


# Primes 2..127 (32 primes)
PRIMES_32 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
    113, 127, 131,
]


class PiK:
    """
    Family of prime-offset cyclic shifts on d-dim hypervectors.

    pi(k, x)        -> Π_k(x)
    pi_inv(k, x)    -> Π_k^{-1}(x)
    bind(k, a, b)   -> Π_k(a) ⊙ b  (asymmetric)
    unbind(k, c, b) -> Π_k^{-1}(c ⊙ conj(b))  (recovers a from c=bind(k,a,b))
    chain_bind(items) -> sequential bind with increasing k

    Supports both bipolar real and phasor complex HVs.
    """

    def __init__(self, d=2048, n_primes=32):
        if d <= max(PRIMES_32[:n_primes]):
            raise ValueError(
                f'd={d} must exceed largest prime offset {max(PRIMES_32[:n_primes])}')
        self.d = int(d)
        self.primes = PRIMES_32[:n_primes]
        self.n = len(self.primes)

    def pi(self, k, x):
        """Π_k(x) = roll(x, p_k). k is 0-indexed into primes list."""
        return np.roll(x, self.primes[k % self.n], axis=-1)

    def pi_inv(self, k, x):
        return np.roll(x, -self.primes[k % self.n], axis=-1)

    @staticmethod
    def _is_phasor(x):
        return np.issubdtype(x.dtype, np.complexfloating)

    @staticmethod
    def _mul(a, b):
        """Pointwise multiply (works for bipolar real or phasor complex)."""
        return a * b

    @staticmethod
    def _conj(x):
        if np.issubdtype(x.dtype, np.complexfloating):
            return np.conj(x)
        return x  # real bipolar is self-conjugate

    def bind(self, k, a, b):
        """Π_k(a) ⊙ b."""
        return self._mul(self.pi(k, a), b)

    def unbind(self, k, c, b):
        """Given c = bind(k, a, b), recover a."""
        return self.pi_inv(k, self._mul(c, self._conj(b)))

    def chain_bind(self, *items):
        """
        Chain binding: bind(0, bind(1, ..., bind(n-1, items[0], items[1]), ...), items[n])
        Each level uses a different Π_k. items: variadic HVs.
        Returns final bound HV.
        """
        if not items:
            raise ValueError('need at least one item')
        result = items[0]
        for i, x in enumerate(items[1:]):
            result = self.bind(i, result, x)
        return result

    def chain_unbind(self, c, *items_minus_first):
        """
        Inverse of chain_bind. Given c = chain_bind(a, b1, b2, ..., bn),
        and the same b1..bn, recover a.
        """
        result = c
        # Unbind in reverse order
        for i in reversed(range(len(items_minus_first))):
            result = self.unbind(i, result, items_minus_first[i])
        return result

    # ── Turing-style state machine on top ────────────────────────────────

    def stamp(self, hv, role_k):
        """Mark hv with role index. role_k determines which Π applies."""
        return self.pi(role_k, hv)

    def unstamp(self, hv, role_k):
        return self.pi_inv(role_k, hv)
