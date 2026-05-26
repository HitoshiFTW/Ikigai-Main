"""
ikigai.cognition.vsa_calculus -- VSA Differential Calculus.

Day 55 Pack 58 -- complete #1: unbind as gradient, chain rule, no backprop.

VSA Algebra (bipolar +-1, bind = elementwise multiply):
    bind(a, b)   = a * b          self-inverse: bind(bind(a,b), a) = b
    bundle(hvs)  = sign(sum(hvs)) majority vote
    unbind(c, b) = bind(c, b)     same as bind (self-inverse)

Differential interpretation:
    c = bind(f, g)
    dc/df = g   (perturb f by df -> c changes by bind(df, g))
    dc/dg = f

Chain rule:
    d bind(f(x), g) / dx = bind(df/dx, g)
    Composition: bind(bind(bind(x, h1), h2), h3)
    gradient w.r.t. x = bind(h1, bind(h2, h3))

Credit assignment (no backprop):
    output = bind(input_a, input_b)
    target != output -> error_mask = positions where output differs from target
    credit[i] = fraction of error positions where input_i is decisive in bundle
    Adjust: input_new = sign(input + lr * unbind(target, other))

vs LLM backprop:
    LLM: O(params) per backward pass, approximate float gradients
    VSACalculus: exact integer operations, O(d) per grad/chain call
    No numerical precision issues. Algebraically exact.
"""

import numpy as np


_HV_CACHE = {}


def _hv_for(key, d):
    if (key, d) not in _HV_CACHE:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
        _HV_CACHE[(key, d)] = (rng.integers(0, 2, size=d) * 2 - 1).astype(np.float32)
    return _HV_CACHE[(key, d)]


def _encode(tokens, d):
    if not tokens:
        return np.zeros(d, dtype=np.float32)
    accum = np.zeros(d, dtype=np.int32)
    for i, tok in enumerate(tokens):
        accum += _hv_for(f'{tok}@{i}', d).astype(np.int32)
    out = np.sign(accum).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _bind(a, b):
    """Elementwise multiply. Self-inverse: bind(bind(a,b),a) = b for +-1 HVs."""
    out = np.sign(a * b).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _bundle(hvs):
    if not hvs:
        return np.zeros(len(hvs[0]) if hvs else 0, dtype=np.float32)
    accum = np.zeros(len(hvs[0]), dtype=np.int32)
    for h in hvs:
        accum += h.astype(np.int32)
    out = np.sign(accum).astype(np.float32)
    out[out == 0.0] = 1.0
    return out


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VSACalculus:
    """
    Exact algebraic differentiation over VSA bind/bundle operations.
    No floating-point gradients. No backpropagation. O(d) per operation.
    """

    def __init__(self, d=400):
        self.d = d

    #  bind differentiation

    def grad_bind(self, f_hv, g_hv):
        """
        dc/df where c = bind(f, g).
        Gradient of bind w.r.t. f is g (the other operand).
        Perturbing f by df gives c change: bind(df, g).
        """
        return g_hv.copy()

    def chain_bind(self, grad_out, *intermediate_hvs):
        """
        Chain rule through a sequence of binds.
        If c = bind(bind(bind(x, h1), h2), h3, ...):
        dc/dx = bind(h1, bind(h2, bind(h3, ...))) = bind of all intermediates.
        Apply grad_out on top: result = bind(grad_out, h1, h2, ...).
        """
        result = grad_out.copy()
        for h in intermediate_hvs:
            result = _bind(result, h)
        return result

    #  bundle differentiation

    def grad_bundle(self, hvs, target_idx):
        """
        ∂bundle(hvs)/∂hvs[target_idx].
        Position p is decisive: removing hvs[target_idx] at p changes the bundle output.
        Computed exactly: compare bundle_with vs bundle_without at each position.
        Returns binary mask (1.0 where decisive, 0.0 elsewhere).
        """
        d = len(hvs[0])
        accum_others = np.zeros(d, dtype=np.int32)
        for i, h in enumerate(hvs):
            if i != target_idx:
                accum_others += h.astype(np.int32)
        accum_all = accum_others + hvs[target_idx].astype(np.int32)

        def _bsign(x):
            s = np.sign(x)
            s[s == 0] = 1  # tie -> +1 convention
            return s

        with_target    = _bsign(accum_all)
        without_target = _bsign(accum_others)
        return (with_target != without_target).astype(np.float32)

    #  solve / unbind

    def solve_bind(self, c_hv, b_hv):
        """
        Given c = bind(a, b), recover a = unbind(c, b) = bind(c, b).
        Exact for bipolar +-1 HVs (self-inverse property).
        """
        return _bind(c_hv, b_hv)

    def solve_chain(self, output_hv, *forward_hvs):
        """
        Recover input x from: output = bind(bind(bind(x, h1), h2), h3).
        x = unbind(output, h1, h2, h3) = bind(output, h1, h2, h3).
        Exact for +-1 HVs.
        """
        result = output_hv.copy()
        for h in forward_hvs:
            result = _bind(result, h)
        return result

    #  credit assignment

    def credit_assign(self, output_hv, input_hvs, target_hv):
        """
        Which input HVs are responsible for the error (output != target)?
        error_mask: positions where output differs from target.
        credit[i]: fraction of error positions where input_hvs[i] is decisive.
        Higher credit = input_i caused more of the error.
        Returns list of floats, one per input HV.
        """
        error_mask = (output_hv != target_hv).astype(np.float32)
        n_errors   = float(np.sum(error_mask))
        if n_errors == 0.0:
            return [0.0] * len(input_hvs)
        credits = []
        for i in range(len(input_hvs)):
            decisive = self.grad_bundle(input_hvs, i)
            credit   = float(np.dot(error_mask, decisive)) / n_errors
            credits.append(credit)
        return credits

    #  update step

    def update_toward(self, hv, target_hv, lr=0.5):
        """
        Nudge hv toward target_hv (Hebbian VSA step).
        hv_new = sign(hv + lr * target_hv).
        Equivalent to gradient descent on 1 - cosine(hv, target) in VSA algebra.
        """
        updated = hv.astype(np.float32) + lr * target_hv.astype(np.float32)
        out = np.sign(updated).astype(np.float32)
        out[out == 0.0] = 1.0
        return out

    def converge(self, hv, target_hv, lr=0.5, max_steps=20, tol=0.99):
        """
        Repeat update_toward until cosine(hv, target) >= tol or max_steps.
        Returns (final_hv, n_steps, final_cosine).
        """
        curr = hv.copy()
        for step in range(max_steps):
            sim = _cosine(curr, target_hv)
            if sim >= tol:
                return curr, step, sim
            curr = self.update_toward(curr, target_hv, lr)
        return curr, max_steps, _cosine(curr, target_hv)

    #  error energy

    def error_energy(self, output_hv, target_hv):
        """
        Hamming distance (fraction of positions that disagree).
        Equivalent to (1 - cosine) / 2 for normalized +-1 HVs.
        """
        return float(np.mean(output_hv != target_hv))

    def xor_gradient(self, a_hv, b_hv):
        """
        Positions where a and b disagree = 'conflict vector'.
        This is the gradient direction to reduce disagreement.
        Returns mask (1.0 = conflict position, 0.0 = agreement).
        """
        return (a_hv != b_hv).astype(np.float32)
