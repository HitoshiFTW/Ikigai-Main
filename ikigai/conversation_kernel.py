"""
ikigai.conversation_kernel -- Master Conversational Substrate Kernel.

Day 55 -- integrates all 5 conversational substrate primitives:

    Pack 32: PhaseLockedHolographicBuffer (PLHB)
             Single phasor state Phi in C^d. Constant 3.2 KB. Infinite history.

    Pack 33: ParallelSemSynCoupling (PSSTC)
             Third-order tensor coupling. Grammar-locked emit. 100% CFG conformance.

    Pack 34: BeliefProjectionManifold (BSPM)
             Per-user belief B_U, cognitive axes C_U[5], frozen persona P_self.

    Pack 35: ConversationalVariationalFreeEnergyField (CVFEF)
             F_t = KL_surprise + w_k*contradiction + w_g*gap.
             Action selection: a* = argmin_a E[F|a].

    Pack 36: AtomicCrystallineStore (ACCI)
             Lock-free triple counter. WAL persistence. Zero catastrophic forgetting.

Master execution equation (per turn t):
    Phi_t    = PLHB.add_turn(tokens, role)
    B_U_t    = BSPM.update(tokens)            --> belief manifold
    emit_t   = PSSTC.step(tokens)             --> grammar-locked vector
    F_t      = CVFEF.ingest(B_U_t, u_hv_t)  --> free energy
    a_t      = CVFEF.select_action()          --> action
    count_t  = ACCI.observe(role, G_t, state) --> crystallization
    resp_t   = BSPM.align(emit_t)             --> persona-aligned response HV

State footprint: 3.2 KB (Phi) + 256 B (B_U) + 256 KB (T_couple) = ~263 KB total.
No gradient. No backprop. No LLM. No transformer.
"""

import numpy as np

from ikigai.cognition.phasor_state import PhaseLockedHolographicBuffer, HV_DIM, OMEGA_DEFAULT
from ikigai.cognition.dssc_coupling import ParallelSemSynCoupling, build_default_cfg
from ikigai.cognition.persona_manifold import BeliefProjectionManifold, CERTAINTY
from ikigai.cognition.free_energy_drive import ConversationalVariationalFreeEnergyField
from ikigai.cognition.crystallizer import AtomicCrystallineStore


class KernelOutput:
    """Single-turn output from ConversationKernel.step()."""
    __slots__ = [
        'phi', 'B_U', 'C_U', 'delta_cert',
        'emit_vec', 'G_id', 'grammar_name',
        'F_t', 'action',
        'resp_aligned',
        'turn',
    ]

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f'KernelOutput(turn={self.turn}, grammar={self.grammar_name}, '
            f'action={self.action}, F={self.F_t:.4f}, '
            f'delta_cert={self.delta_cert:.4f})'
        )


class ConversationKernel:
    """
    Conversational substrate kernel. No LLM. No transformer. No forgetting.

    All 5 primitives integrated via master execution equation.
    Per-turn cost: O(d) phasor + O(d*|V|) emit + O(d) belief + O(1) FE + O(1) crystal.
    """

    def __init__(
        self,
        plhb_dim=HV_DIM,
        dssc_d_sem=64,
        dssc_d_emit=64,
        bspm_d=64,
        bspm_alpha=0.3,
        bspm_gamma=0.5,
        cvfef_d=64,
        cvfef_w_k=0.4,
        cvfef_w_g=0.3,
        seed=42,
    ):
        cfg = build_default_cfg()

        self.plhb  = PhaseLockedHolographicBuffer(dim=plhb_dim, omega=OMEGA_DEFAULT, seed=seed)
        self.psstc = ParallelSemSynCoupling(cfg=cfg, d_sem=dssc_d_sem, d_emit=dssc_d_emit, seed=seed)
        self.bspm  = BeliefProjectionManifold(d=bspm_d, alpha=bspm_alpha, gamma=bspm_gamma, seed=seed)
        self.cvfef = ConversationalVariationalFreeEnergyField(
            d=cvfef_d, w_k=cvfef_w_k, w_g=cvfef_w_g, seed=seed
        )
        self.acci  = AtomicCrystallineStore()

        self._turn = 0

    #  master step

    def step(self, tokens, role='user'):
        """
        Full conversational step.

        tokens: list[str]  -- tokenized utterance
        role:   str        -- 'user' or 'self'
        Returns KernelOutput.
        """
        # Pack 32: holographic turn ingestion
        phi = self.plhb.add_turn(tokens, role)

        # Pack 34: belief projection update
        B_U, C_U, delta_cert = self.bspm.update(tokens)

        # Pack 33: grammar-locked emit
        emit_vec, G_id = self.psstc.step(tokens)
        grammar_name = self.psstc.grammar_name(G_id)

        # Pack 35: free energy + action selection
        u_hv = self.bspm.encode_utterance(tokens)
        F_t = self.cvfef.ingest(B_U, utterance_hv=u_hv)
        action = self.cvfef.select_action()

        # Pack 36: crystallize observation triple
        cert_state = 'high_cert' if float(C_U[CERTAINTY]) > 0.1 else 'low_cert'
        self.acci.observe(role, grammar_name, cert_state)

        # Pack 34: align emit to persona + user belief
        resp_aligned = self.bspm.align(emit_vec)

        self._turn += 1
        return KernelOutput(
            phi=phi,
            B_U=B_U,
            C_U=C_U,
            delta_cert=delta_cert,
            emit_vec=emit_vec,
            G_id=G_id,
            grammar_name=grammar_name,
            F_t=F_t,
            action=action,
            resp_aligned=resp_aligned,
            turn=self._turn,
        )

    #  recall

    def recall(self, k_back, role='user'):
        """Retrieve turn HV from k_back turns ago via PLHB reverse rotation."""
        return self.plhb.recall_turn(k_back, role)

    def recall_fidelity(self, k_back, role='user'):
        return self.plhb.recall_fidelity(k_back, role)

    #  mining

    def mine_schemas(self, min_support=2):
        """Anti-unification schema mining over crystallized triples."""
        return self.acci.mine_schemas(min_support=min_support)

    def save_wal(self, path):
        return self.acci.save_wal(path)

    def load_wal(self, path):
        self.acci.load_wal(path)

    #  state summary

    def state_summary(self):
        return {
            'turns':          self._turn,
            'plhb_bytes':     self.plhb.state_size_bytes(),
            'free_energy':    self.cvfef.free_energy(),
            'action':         self.cvfef.action_log[-1] if self.cvfef.action_log else None,
            'grammar':        self.psstc.grammar_name(self.psstc.current_node_id),
            'belief_drift':   self.bspm.belief_drift(),
            'crystal_unique': self.acci.unique_triples(),
            'crystal_total':  self.acci.total_observations(),
        }

    def reset(self, seed=None):
        s = seed if seed is not None else 42
        self.plhb.reset(seed=s)
        self.psstc.reset()
        self.bspm.reset(seed=s)
        self.cvfef.reset()
        self.acci.reset()
        self._turn = 0

    @property
    def turn_count(self):
        return self._turn
