"""
ikigai.cognition.on_policy_eval -- Pack 251 On-Policy Evaluation Engine.

Day 73. Teacher-gated substrate write. RPE-style three-factor plasticity:
the organism predicts via its own substrate (fsm.step), teacher provides
ground truth (LLM next token), the gap modulates the write to the k=64
SDM locations that addr(cur, role) activates.

Biological grounding:
    - Schultz 1997 dopamine RPE: phasic DA spike on positive surprise
    - Fremaux + Gerstner 2016: neuromodulated STDP -- DA gates synaptic
      plasticity within an eligibility-trace window
    - Williams 1992 REINFORCE: reward x eligibility_trace = update rule
    - Agarwal 2023 GKD: on-policy student-generated trajectories fix
      train/inference mismatch in distillation

Substrate mapping:
    - "neurons" for binding (cur, role)->actual = the k=64 hard locations
      that bank._activate(addr(cur, role)) returns
    - "eligibility trace" = those k locations stay active for the duration
      of this observation (no temporal trace -- substrate writes are
      instantaneous)
    - "DA signal" = (predict == actual) gate
    - YES signal => weak reinforce (already known, low write strength)
    - NO signal  => unlearn wrong prediction + learn correct + optional
                    noise injection (DA-driven exploration)
    - NOVEL signal => full reinforce (no prior signal to compare)

This module composes primitives that already exist in multirole_memory.py
and vs_fsm.py. Nothing new needed at the substrate level.

Pack 251 is the natural absorb gate for Pack 248 LLM-teacher capability
transfer pipeline -- raw mr.relate over already-absorbed corpus causes
saturation (Day 73 smoke result). Gated absorb writes ONLY where the
organism is wrong, preserving existing structure.
"""

import numpy as np


class OnPolicyEvaluator:
    """On-policy teacher-gated substrate write.

    Usage:
        opv = OnPolicyEvaluator(mr, fsm, alpha=1.0, beta=0.5)
        for (prev, cur, actual) in teacher_stream:
            opv.gated_observe(prev, cur, actual, role='next',
                              candidates=cands)
        # opv.stats has correct/wrong/novel/total
    """

    def __init__(self, mr, fsm, alpha=1.0, beta=0.5, gamma_noise=0.0,
                  reinforce_factor=0.30, novel_factor=1.0,
                  n_iters=2, fsm_beta=8.0, seed=251):
        """
        Args:
            mr               -- MultiRoleMemory
            fsm              -- VSFSM (for org.vs_fsm)
            alpha            -- positive write strength
            beta             -- negative (unlearn wrong) strength
            gamma_noise      -- noise injection on wrong (0 = no noise)
            reinforce_factor -- multiplier on alpha when YES (already known)
            novel_factor     -- multiplier on alpha when no prediction
            n_iters          -- fsm.step resonator iters (default 2 for speed)
            fsm_beta         -- fsm.step softmax beta
            seed             -- rng for noise injection
        """
        self.mr = mr
        self.fsm = fsm
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma_noise = float(gamma_noise)
        self.reinforce_factor = float(reinforce_factor)
        self.novel_factor = float(novel_factor)
        self.n_iters = int(n_iters)
        self.fsm_beta = float(fsm_beta)
        self._rng = np.random.default_rng(int(seed))
        self.stats = {'correct': 0, 'wrong': 0, 'novel': 0, 'total': 0,
                       'top5_hit': 0}

    def _predict(self, prev, cur, candidates):
        try:
            res = self.fsm.step(cur, prev_token=prev,
                                  candidates=candidates,
                                  n_iters=self.n_iters,
                                  beta=self.fsm_beta, top_k=5)
            if not res:
                return None, []
            preds = [t for (t, _) in res if t]
            return (preds[0] if preds else None), preds[:5]
        except Exception:
            return None, []

    def gated_observe(self, prev, cur, actual, role='next', candidates=None):
        """Predict -> compare to actual -> gated write to the activated k=64
        locations for addr(cur, role). Returns (predict, actual)."""
        mr = self.mr
        # On-policy prediction (organism's current guess)
        predict, top5 = self._predict(prev, cur, candidates)

        # Activate the SDM locations for this binding
        bank = mr._bank(role)
        addr = mr._addr(cur, role)
        locs = bank.locs(addr, word=mr._slot(cur, role))
        target_hv = mr.ck.key(actual).astype(np.complex64)

        if predict is None:
            # Novel: full strength write (no prior signal)
            bank.C[locs] += (self.alpha * self.novel_factor) * target_hv
            self.stats['novel'] += 1
        elif predict == actual:
            # YES: weak reinforce (substrate already correct)
            bank.C[locs] += (self.alpha * self.reinforce_factor) * target_hv
            self.stats['correct'] += 1
        else:
            # NO: unlearn wrong + learn correct + optional noise
            wrong_hv = mr.ck.key(predict).astype(np.complex64)
            bank.C[locs] -= self.beta * wrong_hv
            bank.C[locs] += self.alpha * target_hv
            if self.gamma_noise > 0:
                ph = self._rng.uniform(-np.pi, np.pi, mr.d).astype(np.float32)
                noise = np.exp(1j * ph).astype(np.complex64)
                bank.C[locs] += self.gamma_noise * noise
            self.stats['wrong'] += 1

        # bookkeeping for cleanup-candidate registry
        mr._role_targets.setdefault(role, set()).add(actual)
        mr._seen.add(cur)
        mr._dirty = True
        self.stats['total'] += 1
        if actual in top5:
            self.stats['top5_hit'] += 1
        return predict, actual

    def summary(self):
        s = self.stats
        n = max(s['total'], 1)
        return {
            'total': s['total'],
            'correct': s['correct'],
            'wrong': s['wrong'],
            'novel': s['novel'],
            'top1_acc': s['correct'] / n,
            'top5_acc': s['top5_hit'] / n,
            'novel_rate': s['novel'] / n,
            'wrong_rate': s['wrong'] / n,
        }

    def reset(self):
        self.stats = {'correct': 0, 'wrong': 0, 'novel': 0, 'total': 0,
                       'top5_hit': 0}
