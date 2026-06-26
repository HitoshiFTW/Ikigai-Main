"""
ikigai.cognition.residue_hdc -- Pack 257 Residue Hyperdimensional Computing.

Day 74. Multi-modulus phase encoding for compositional integers. Based
on Frady, Kymn, Anandkumar (2023) "Variable Binding for Sparse
Distributed Representations" + Chinese Remainder Theorem.

WHY THIS EXISTS
---------------
Pack 252 NumericEncoder uses a SINGLE Fractional Power Encoding at
scale=10. Capacity ~100 distinct integers before phase wrap collides.
Adequate for grade-school arithmetic (0..100); insufficient for general
math (1000+), GSM8K word problems, or compositional integers.

RHC encodes x across K residue moduli (m_1, m_2, ..., m_K) and stores
the bound product. CRT recovers x from residues. Range = product of
moduli, error-correcting via redundancy.

Example: moduli (7, 11, 13) -> range 1001
         moduli (7, 11, 13, 17) -> range 17,017
         moduli (7, 11, 13, 17, 19, 23) -> range 7,436,429

HPE CONNECTION (Invention 1)
----------------------------
Per the Day 73 inventions roadmap, RHC was folded into Invention 1
HPE as the error-correction layer. Each HPE level can carry one
residue code, providing redundancy across levels. Pack 257 is the
standalone primitive; HPE will use it internally.

NOT A NEW SUBSTRATE OP
----------------------
Composes Pack 252 NumericEncoder primitives. Adds compositional
capacity via residue arithmetic. No new substrate math.

CAPACITY vs SINGLE FPE
----------------------
Single FPE at scale=S: ~S distinct decodes before noise collision
RHC at moduli (m_1..m_K): up to product(m_i) decodes via CRT
For matched substrate cost (1 bound HV), RHC wins by factor
    product(m_i) / S
e.g. moduli (7, 11, 13, 17) and S=100 -> 170x range lift.
"""

import math
import numpy as np


def _coprime(a, b):
    return math.gcd(a, b) == 1


def _all_coprime(moduli):
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if not _coprime(moduli[i], moduli[j]):
                return False
    return True


def _crt_recover(residues, moduli):
    """Chinese Remainder Theorem: solve x mod m_i = r_i for all i."""
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        # Modular inverse Mi^-1 mod m
        inv = pow(Mi, -1, m)
        x = (x + r * Mi * inv) % M
    return x


class ResidueHDC:
    """Pack 257 Residue HDC encoder.

    Encodes integers via K-modulus residue codes bound into a single
    FHRR phasor HV. Decoding via per-modulus cleanup + CRT recovery.

    USAGE
    -----
        rhc = ResidueHDC(d=400, moduli=(7, 11, 13, 17), seed=257)
        hv5 = rhc.encode(5)
        hv3 = rhc.encode(3)
        hv8 = rhc.add(hv5, hv3)     # Hadamard mul ~= FPE add
        decoded, score = rhc.decode(hv8)   # -> 8
    """

    def __init__(self, d, moduli, seed=257):
        if not _all_coprime(moduli):
            raise ValueError(
                f'moduli must be pairwise coprime; got {moduli}')
        self.d = int(d)
        self.moduli = tuple(int(m) for m in moduli)
        self.K = len(self.moduli)
        self.range = 1
        for m in self.moduli:
            self.range *= m
        # INTEGER multipliers per modulus in [1, m_k). Integer
        # multipliers preserve modular semantics under Hadamard mul:
        # exp(i*2pi*(a+b)*k/m) = exp(i*2pi*((a+b) mod m)*k/m) since
        # k * floor((a+b)/m) is integer -> exp(i*2pi*int) = 1.
        # Real-valued phases break this (composition diverges).
        rng = np.random.default_rng(int(seed))
        self.phases = []
        for k in range(self.K):
            ph = rng.integers(1, self.moduli[k], self.d).astype(np.float32)
            self.phases.append(ph)
        # Precompute base phasors -- encode(x) for x in [0, m_k)
        # Cache only on demand to keep memory minimal at startup.
        self._codebooks = [{} for _ in range(self.K)]

    # ---- per-modulus FPE -------------------------------------------

    def _encode_residue(self, k, r):
        """Encode r mod m_k as FPE phasor in modular phase space.
        theta = 2*pi * r * basis[k] / m_k  with INTEGER basis[k] in [1, m_k).
        Composition holds: encode_residue(a) * encode_residue(b) =
        encode_residue((a+b) mod m_k).
        """
        m = self.moduli[k]
        r = int(r) % m
        cache = self._codebooks[k]
        if r in cache:
            return cache[r]
        theta = (2.0 * np.pi * r / m) * self.phases[k]
        hv = np.exp(1j * theta).astype(np.complex64)
        cache[r] = hv
        return hv

    def _modulus_codebook(self, k):
        """Return full codebook for modulus k -- dict {r: HV}."""
        cb = self._codebooks[k]
        for r in range(self.moduli[k]):
            if r not in cb:
                self._encode_residue(k, r)
        return cb

    # ---- public API ------------------------------------------------

    def encode(self, x):
        """Encode integer x as bound product of per-modulus FPEs."""
        x = int(x)
        hv = self._encode_residue(0, x % self.moduli[0])
        for k in range(1, self.K):
            hv = hv * self._encode_residue(k, x % self.moduli[k])
        return hv.astype(np.complex64)

    def add(self, hv_a, hv_b):
        """Compositional add: hv(a+b) = hv(a) * hv(b) (Hadamard mul,
        FPE semantics under each modulus)."""
        return (hv_a * hv_b).astype(np.complex64)

    def sub(self, hv_a, hv_b):
        """Compositional sub: hv(a-b) = hv(a) * conj(hv(b))."""
        return (hv_a * np.conj(hv_b)).astype(np.complex64)

    def decode(self, hv, mode='brute', x_min=0, x_max=None):
        """Recover integer x from bound HV.

        mode='brute' (default): iterate x in [x_min, x_max), cosine
            against encode(x). Exact but O(range * d). Fine for ranges
            up to ~100K.
        mode='resonator': Pack 256 ResonatorNetwork iterative decode.
            Faster for large ranges but resonator init on per-modulus
            codebooks with mean~0 is unreliable; needs tuning. TODO.

        Returns (x, score).
        """
        if x_max is None:
            x_max = self.range
        hv_n = self._normalize(hv)
        if mode == 'brute':
            best_x = 0
            best_score = -1e9
            # Vectorize: build encode batch in chunks to bound memory
            BATCH = 2048
            for start in range(int(x_min), int(x_max), BATCH):
                end = min(start + BATCH, int(x_max))
                xs = list(range(start, end))
                K = np.stack([self._normalize(self.encode(x)) for x in xs])
                sims = np.real(K @ np.conj(hv_n)) / self.d
                idx = int(np.argmax(sims))
                if sims[idx] > best_score:
                    best_score = float(sims[idx])
                    best_x = xs[idx]
            return best_x, best_score
        elif mode == 'resonator':
            from ikigai.cognition.resonator_network import ResonatorNetwork
            codebooks = []
            for k in range(self.K):
                cb = self._modulus_codebook(k)
                codebooks.append({f'r{r}': cb[r]
                                    for r in range(self.moduli[k])})
            if (not hasattr(self, '_resonator')
                or self._resonator_K != self.K):
                self._resonator = ResonatorNetwork(
                    d=self.d, codebooks=codebooks,
                    max_iters=40, beta=8.0, momentum=0.3)
                self._resonator_K = self.K
            result = self._resonator.decode(hv_n)
            residues = []
            scores = []
            for k in range(self.K):
                name, sc = result[k]
                r = int(name[1:])
                residues.append(r)
                scores.append(sc)
            x = _crt_recover(residues, self.moduli)
            return x, float(np.mean(scores))
        else:
            raise ValueError(f'unknown mode: {mode!r}')

    @staticmethod
    def _normalize(hv):
        mag = float(np.abs(hv).mean()) + 1e-12
        return (hv / mag).astype(np.complex64)

    # ---- Pack 291 multiplicative binding ⋆ -------------------------
    # Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen (2023)
    # arXiv:2311.04872v1 §4.1.2.  Requires all moduli prime so that
    # every integer multiplier u_j in [1, m_k) has a modular inverse.
    # NeuroSeed's default moduli (7, 11, 13, 17) and the production
    # 8-moduli set (5, 7, 11, 13, 17, 19, 23, 29) satisfy this.

    @staticmethod
    def _is_prime(n):
        n = int(n)
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def _ensure_antibases(self):
        """Build per-modulus modular-inverse anti-base phasor arrays.
        Cached on the instance after first call.  Raises if any
        modulus is not prime."""
        if getattr(self, '_antibases', None) is not None:
            return
        antibases = []
        antibase_resids = []
        for k in range(self.K):
            m = int(self.moduli[k])
            if not self._is_prime(m):
                raise ValueError(
                    f'multiplicative bind requires prime moduli; '
                    f'modulus {m} at index {k} is composite')
            u = self.phases[k].astype(np.int64) % m
            # pow(u, -1, m) needs Python ints; vectorize with comprehension
            v = np.array([pow(int(uu), -1, m) for uu in u],
                          dtype=np.int64)
            theta = 2.0 * np.pi * v / m
            antibases.append(np.exp(1j * theta).astype(np.complex64))
            antibase_resids.append(v)
        self._antibases = antibases
        self._antibase_resids = antibase_resids

    def encode_factored(self, x):
        """Return per-modulus list of HVs (length K) -- the factor
        encoding required by ⋆.  Use `bundle_factored` to recover
        the standard bundled HV."""
        return [self._encode_residue(k, int(x) % self.moduli[k])
                  for k in range(self.K)]

    def bundle_factored(self, factors):
        """Multiply per-modulus HVs into a single bundled HV."""
        hv = factors[0]
        for f in factors[1:]:
            hv = hv * f
        return hv.astype(np.complex64)

    def mul_factored(self, factors_a, factors_b):
        """⋆ multiplicative bind on per-modulus factor lists.

        Algebra (Kymn 2023 §4.1.2):
            a_mk[j] = e^(i * 2π/m * u_j * x1)
            b_mk[j] = e^(i * 2π/m * u_j * x2)
            f(a, b)[j] phase = 2π/m * (u_j*x1) * (u_j*x2) mod m
                              = 2π/m * u_j² * x1*x2 mod m
            y_mk[j] = e^(i * 2π/m * v_j) where u_j * v_j ≡ 1 (mod m)
            f(f(a,b), y_mk)[j] phase = 2π/m * (u_j²*x1*x2) * v_j mod m
                                      = 2π/m * u_j * x1*x2 mod m
            which is exactly z_mk(x1*x2)[j].
        """
        self._ensure_antibases()
        result = []
        for k in range(self.K):
            m = int(self.moduli[k])
            # Extract residues from per-modulus phasors by reading angle.
            # angle(e^(i*theta)) returns theta in (-π, π]; multiply by
            # m/(2π) and round to recover the integer residue.
            ang_a = np.angle(factors_a[k])
            ang_b = np.angle(factors_b[k])
            r = np.round(ang_a * m / (2.0 * np.pi)).astype(np.int64) % m
            s = np.round(ang_b * m / (2.0 * np.pi)).astype(np.int64) % m
            # f(a, b): phasor with residue (r * s) mod m
            rs = (r * s) % m
            # f(f(a, b), y_mk): residue becomes (rs * v_j) mod m, which
            # by the algebra above equals u_j * x1 * x2 mod m.
            v = self._antibase_resids[k]
            final_r = (rs * v) % m
            theta = 2.0 * np.pi * final_r / m
            result.append(np.exp(1j * theta).astype(np.complex64))
        return result

    def multiply(self, hv_a_factors, hv_b_factors):
        """Convenience alias -- multiplicative bind on factored inputs.
        Returns the BUNDLED HV of x1 * x2.
        For bundled-HV inputs, factor recovery via resonator first."""
        return self.bundle_factored(
            self.mul_factored(hv_a_factors, hv_b_factors))

    def mul_int(self, x1, x2):
        """End-to-end substrate-native multiplication for unit tests:
        encode each, ⋆, decode."""
        fa = self.encode_factored(x1)
        fb = self.encode_factored(x2)
        out_factors = self.mul_factored(fa, fb)
        bundled = self.bundle_factored(out_factors)
        decoded, _ = self.decode(bundled)
        return int(decoded)

    # ---- Pack 291.8 ⋆-inverse (exact integer division) -------------
    # Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen (2023)
    # arXiv:2311.04872v1 §4.1.3 (modular multiplicative-inverse path).
    # For EXACT division q = x1 / x2 (x1 = q * x2) recover q per modulus
    # via the modular inverse of x2's residue:
    #     z(x1)[j] = u_j * x1,   z(x2)[j] = u_j * x2     (mod m)
    #     u_j * q = u_j * (x1 * x2^{-1}) = u_j * r * s^{-1}   (mod m)
    # where r = z(x1)[j], s = z(x2)[j], s^{-1} the modular inverse of s.
    # A modulus is VALID iff x2 != 0 (mod m); when m | x2 the residue
    # field has no inverse (the gcd(x2, m) != 1 case from the memo) so
    # that modulus is dropped and q is CRT-recovered from the remaining
    # valid moduli.  Exact whenever product(valid moduli) > q.
    #
    # This is substrate-native: q is produced by phasor algebra + CRT,
    # not by encoding a Python-computed quotient (the Pack 291.7 path).

    @staticmethod
    def _modinv_vec(arr, m):
        """Per-element modular inverse mod prime m.  Elements are
        reduced mod m first; a zero element maps to 0 (callers only
        pass nonzero residues on the valid path)."""
        out = np.empty(arr.shape, dtype=np.int64)
        for i, a in enumerate(arr):
            a = int(a) % m
            out[i] = pow(a, -1, m) if a else 0
        return out

    def div_factored(self, factors_a, factors_b):
        """⋆-inverse division on per-modulus factor lists.

        Returns (out_factors, valid): out_factors[k] is the canonical
        phasor for residue (u_j * q) mod m_k; valid[k] is True when
        modulus k could invert (x2 != 0 mod m_k).  Invalid moduli carry
        a placeholder and MUST be skipped on decode.
        """
        self._ensure_antibases()
        out, valid = [], []
        for k in range(self.K):
            m = int(self.moduli[k])
            r = np.round(np.angle(factors_a[k]) * m / (2.0 * np.pi)
                          ).astype(np.int64) % m
            s = np.round(np.angle(factors_b[k]) * m / (2.0 * np.pi)
                          ).astype(np.int64) % m
            if not np.any(s):            # x2 == 0 (mod m): no inverse
                out.append(factors_a[k])
                valid.append(False)
                continue
            s_inv = self._modinv_vec(s, m)
            u = self.phases[k].astype(np.int64) % m
            final_r = (r * s_inv % m) * u % m        # = u_j * q (mod m)
            theta = 2.0 * np.pi * final_r / m
            out.append(np.exp(1j * theta).astype(np.complex64))
            valid.append(True)
        return out, valid

    def _residue_of(self, factor, k):
        """Recover the integer residue (x mod m_k) carried by a single
        canonical per-modulus phasor: invert the u_j multiplier and take
        the modal value across dims (exact for clean phasors)."""
        m = int(self.moduli[k])
        canon = np.round(np.angle(factor) * m / (2.0 * np.pi)
                          ).astype(np.int64) % m        # = u_j * x
        v = self._antibase_resids[k]                     # u_j^{-1}
        per_dim = (canon * v) % m                         # = x mod m
        return int(np.bincount(per_dim, minlength=m).argmax())

    def div_int(self, x1, x2):
        """End-to-end substrate-native EXACT division: encode each,
        ⋆-inverse, CRT-decode over valid moduli.

        Requires x2 != 0 and x1 % x2 == 0 -- the exact-division regime
        the ⋆-inverse path covers.  Inexact division is the Pack 291.7
        direct-quotient fallback's job (floor semantics)."""
        x1, x2 = int(x1), int(x2)
        if x2 == 0:
            raise ZeroDivisionError('div_int by zero')
        if x1 % x2 != 0:
            raise ValueError(
                f'div_int covers exact division only; {x1} % {x2} != 0')
        self._ensure_antibases()
        fa = self.encode_factored(x1)
        fb = self.encode_factored(x2)
        out, valid = self.div_factored(fa, fb)
        residues, mods = [], []
        for k in range(self.K):
            if not valid[k]:
                continue
            residues.append(self._residue_of(out[k], k))
            mods.append(int(self.moduli[k]))
        if not mods:
            raise ValueError(
                f'no valid modulus for {x1}/{x2}: x2 divisible by every '
                f'modulus {self.moduli}')
        prod = 1
        for mm in mods:
            prod *= mm
        return int(_crt_recover(residues, mods) % prod)

    # ---- Pack 305 fast factored add/sub + CRT decode ---------------
    # The default decode() is O(range * d) brute cosine -- ~300 ms per
    # call.  Staying in FACTORED form and decoding per-modulus via
    # _residue_of + CRT is O(K * d) (~us), exact for clean RHC factors.

    @staticmethod
    def add_factored(factors_a, factors_b):
        """Per-modulus residue addition: factor phase (u_j*a)+(u_j*b) =
        u_j*(a+b), still a canonical factor.  Hadamard per modulus."""
        return [(fa * fb).astype(np.complex64)
                for fa, fb in zip(factors_a, factors_b)]

    @staticmethod
    def sub_factored(factors_a, factors_b):
        """Per-modulus residue subtraction: u_j*(a-b)."""
        return [(fa * np.conj(fb)).astype(np.complex64)
                for fa, fb in zip(factors_a, factors_b)]

    def decode_factored(self, factors, valid=None, signed=False):
        """Decode a per-modulus factor list to an integer via per-modulus
        residue recovery + CRT.  O(K * d) -- the fast path replacing brute
        decode.  `valid` (from div_factored) skips dropped moduli.

        signed=True interprets the modular ring as signed: a result in the
        upper half (> prod/2) maps to its negative (x - prod).  The ring
        natively represents -k as prod-k, so subtraction a-b<0 decodes
        correctly with no Python sign handling.

        Returns (x, score=1.0)."""
        self._ensure_antibases()
        residues, mods = [], []
        for k in range(self.K):
            if valid is not None and not valid[k]:
                continue
            residues.append(self._residue_of(factors[k], k))
            mods.append(int(self.moduli[k]))
        if not mods:
            return 0, 0.0
        prod = 1
        for mm in mods:
            prod *= mm
        x = int(_crt_recover(residues, mods) % prod)
        if signed and x > prod // 2:
            x -= prod
        return x, 1.0

    # ---- diagnostics -----------------------------------------------

    def capacity(self):
        return self.range

    def summary(self):
        return {
            'd': self.d,
            'moduli': self.moduli,
            'K': self.K,
            'range': self.range,
            'codebook_size_total': sum(self.moduli),
        }
