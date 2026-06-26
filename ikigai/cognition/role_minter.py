"""
ikigai.cognition.role_minter -- Pack 247w abductive role induction.

Day 72. Extends Pack 226 (schema_inducer anti-unification) from token-shape
schemas to RELATION-LABEL roles.

Problem: substrate has 44 hardcoded roles, 39 never written during absorb
because no upstream code emits triples for those labels. Pre-Pack-247w
dictionary absorb would need every WordNet relation (hyponym, meronym,
troponym, derivationally_related, ...) pre-declared. Hardcoded list = rigid.

Fix: observe-then-mint. Buffer triples whose relation_label is not in
mr.roles. When the same label recurs >= min_support times, mint a new
role HV, register in mr.roles, route to a bank (by keyword), flush
buffered triples through the new role.

Hand-coded roles stay as biological scaffold (genome encodes cortex
layers + cell types). Emergent roles grow the ceiling without breaking
the scaffold.

Persistence: mint mutates mr.roles + mr._role_to_bank (multi-bank) or
just mr.roles (legacy 2-bank). Both are persisted by MultiRoleMemory
save_ikg. In multi-bank mode the role name is also appended to
mr._bank_assignment[bank_id]['roles'] so bank_assignment_json reflects
the new role on round-trip.

Buffer is volatile. Sub-threshold triples drop on save by design --
mint represents conviction, not bookkeeping.
"""

import numpy as np


# Keyword router: relation-label -> bank id. Used only when running in
# multi-bank mode. Order matters (first match wins).
_DEFAULT_BANK_ROUTES = (
    ('b_sem',    ('syn', 'def', 'hyp', 'hyper', 'hypo', 'mer', 'holo',
                    'isa', 'similar', 'antonym', 'concept', 'gloss')),
    ('b_aff',    ('verb', 'action', 'agent', 'patient', 'do', 'affordance',
                    'tropo', 'entail')),
    ('b_world',  ('cause', 'effect', 'prevent', 'enable', 'world',
                    'counterfactual', 'transition')),
    ('b_epi',    ('before', 'after', 'during', 'time', 'episode',
                    'example', 'event', 'temporal')),
    ('b_ground', ('see', 'hear', 'touch', 'color', 'shape', 'sensory',
                    'modality', 'motor', 'image')),
    ('b_self',   ('belief', 'intent', 'goal', 'persona', 'self', 'meta',
                    'i_', '_i', 'mind')),
    ('b_gram',   ('schema', 'rule', 'template', 'gram', 'syntax',
                    'deriv', 'morpho', 'inflect', 'pos_')),
    ('b_lang',   ('next', 'prev', 'co', 'pos_', 'token', 'word')),
)


def _route_by_keyword(label, bank_ids):
    """Map a relation label to a bank id by keyword. Falls back to b_sem
    if available else first declared bank."""
    low = str(label).lower()
    for bank_id, keys in _DEFAULT_BANK_ROUTES:
        if bank_id not in bank_ids:
            continue
        if any(k in low for k in keys):
            return bank_id
    return 'b_sem' if 'b_sem' in bank_ids else next(iter(bank_ids))


class RoleMinter:
    """
    Abductive role induction over relation labels.

    Use:
        mr = MultiRoleMemory(...)
        # observe(s, r, o) for every triple; pre-known r passes through
        # directly to mr.relate, unknown r buffers until min_support, then
        # mint + flush.
        mr.role_minter.observe('cat', 'hyponym', 'animal')
        mr.role_minter.observe('dog', 'hyponym', 'animal')
        ...  # after >= min_support calls, role 'hyponym' is minted.

    Stats:
        status()   -> dict with mint count + per-pattern buffer size
    """

    def __init__(self, mr, min_support=8, seed=247014, bank_router=None,
                 collision_policy='skip', max_buffer_per_label=4096):
        """
        Args:
            mr                       -- MultiRoleMemory instance (mutated on mint)
            min_support              -- triples needed to mint (default 8)
            seed                     -- rng for phasor minting (deterministic)
            bank_router              -- (label, bank_ids) -> bank_id (optional)
            collision_policy         -- 'skip' (default): if minted name matches
                                          existing role, route through existing
                                          and do NOT remint
            max_buffer_per_label     -- cap on pre-mint buffer (cheap DoS guard)
        """
        self.mr = mr
        self.min_support = int(min_support)
        self.max_buffer_per_label = int(max_buffer_per_label)
        self.collision_policy = str(collision_policy)
        self._rng = np.random.default_rng(int(seed))
        self._bank_router = bank_router or _route_by_keyword
        # label -> list[(subj, obj)] pre-mint buffer
        self._buffer = {}
        # label -> int. Total observations including post-mint flushes.
        self._support = {}
        # set of labels we have minted in this session (for status())
        self._minted = set()
        # set of labels we refused to mint due to collision skip
        self._collided = set()

    # ---- public API --------------------------------------------------

    def observe(self, subj, relation_label, obj):
        """Observe a triple. Routes to mr.relate immediately if the role
        already exists; otherwise buffers until min_support is hit, then
        mints + flushes.

        Returns the mode taken: 'direct', 'buffered', or 'minted'.
        """
        label = str(relation_label)
        self._support[label] = self._support.get(label, 0) + 1
        if label in self.mr.roles:
            self.mr.relate(subj, label, obj)
            return 'direct'
        # buffer
        buf = self._buffer.setdefault(label, [])
        if len(buf) < self.max_buffer_per_label:
            buf.append((subj, obj))
        if len(buf) >= self.min_support:
            self._mint(label)
            return 'minted'
        return 'buffered'

    def flush(self):
        """Force-mint every label whose buffer has any entries (regardless
        of support). Returns list of minted labels. Useful for end-of-corpus
        commit when you want partial-support roles persisted."""
        minted = []
        for label in list(self._buffer.keys()):
            if self._buffer[label]:
                self._mint(label)
                minted.append(label)
        return minted

    def discard(self, label=None):
        """Drop buffered triples without minting. label=None drops all."""
        if label is None:
            self._buffer.clear()
        else:
            self._buffer.pop(label, None)

    def status(self):
        """Snapshot for logs / debugging."""
        return {
            'min_support': self.min_support,
            'minted_count': len(self._minted),
            'minted_labels': sorted(self._minted),
            'collided_labels': sorted(self._collided),
            'buffered_labels': sorted(self._buffer.keys()),
            'buffer_sizes': {l: len(b) for l, b in self._buffer.items()},
            'support_per_label': dict(self._support),
        }

    # ---- mint --------------------------------------------------------

    def _mint(self, label):
        """Mint a phasor HV for `label`, route to a bank, flush buffer
        through mr.relate. Idempotent: if label is already a role, just
        flush + drop buffer."""
        mr = self.mr
        # collision branch: hardcoded role with same name already exists.
        if label in mr.roles:
            if self.collision_policy == 'skip':
                self._collided.add(label)
            buf = self._buffer.pop(label, [])
            for (s, o) in buf:
                mr.relate(s, label, o)
            return

        # fresh mint: phasor with rng-init phases
        ph = self._rng.uniform(-np.pi, np.pi, mr.d).astype(np.float32)
        mr.roles[label] = np.exp(1j * ph).astype(np.complex64)

        # bank routing
        if mr._bank_assignment is not None:
            bank_ids = list(mr._bank_assignment.keys())
            bank_id = self._bank_router(label, bank_ids)
            mr._role_to_bank[label] = bank_id
            # persist by mutating the bank_assignment dict that gets saved
            roles_list = mr._bank_assignment[bank_id].setdefault('roles', [])
            if label not in roles_list:
                roles_list.append(label)
        # else legacy 2-bank mode: role auto-routes via DENSE_ROLES check
        # (unknown -> sdm_rel). No mutation needed.

        # init empty target set
        mr._role_targets.setdefault(label, set())

        # flush buffer
        buf = self._buffer.pop(label, [])
        for (s, o) in buf:
            mr.relate(s, label, o)

        self._minted.add(label)

    # ---- introspection ------------------------------------------------

    def __repr__(self):
        return (f'<RoleMinter minted={len(self._minted)} '
                f'buffered={sum(len(b) for b in self._buffer.values())} '
                f'min_support={self.min_support}>')
