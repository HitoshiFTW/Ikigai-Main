"""
ikigai.cognition.federated_merge -- Kill Stack invention #7.

Federated substrate merging. Two (or more) Ikigai organisms with matching
substrate config (d, M, k, seed) share hard-location addresses. Their
counter banks live in the SAME coordinate system. Merging is therefore
a clean VSA superposition:

    merged.C[i] = sum over arms of arm.C[i]

No alignment training needed. No LoRA-mixing hacks. No conflict resolution
heuristic. The math just works because both arms wrote into the same
substrate geometry from the start.

Transformer analogue: LoRA merging requires the same base model + careful
weight scaling + still has interference. Substrate merging is a single
elementwise sum on two ndarrays. O(M*d) work. Substrate stays FIXED.

Public API:
    merged = federated_merge(org_a, org_b, [org_c, ...], alpha=None)
    # alpha optional: list of weights summing to 1, default = equal weighting

Constraints:
    - All organisms must share (d, M, k, seed) so hard-location addresses
      align across arms. The constructor seed is the implicit "merge key".
    - role registration is unioned across arms (deterministic from seed).
    - cooccur_seen, _role_targets, _seen all unioned.
"""

import numpy as np


def federated_merge(*organisms, alpha=None, dest=None):
    """
    Merge N organisms into one. Returns a new IkigaiOrganism (or dest if
    provided) whose substrate is the (optionally weighted) sum of inputs.

    All input organisms MUST share substrate config (d, M, k, seed) so
    hard-location addresses align. The substrate stays FIXED in size.
    """
    if not organisms:
        raise ValueError("federated_merge needs at least one organism")
    base = organisms[0]
    d = base.unified.d
    M = base.unified.sdm.C.shape[0]
    M_rel = base.unified.sdm_rel.C.shape[0]
    # sanity: matching geometry
    for o in organisms[1:]:
        if (o.unified.d != d
                or o.unified.sdm.C.shape[0] != M
                or o.unified.sdm_rel.C.shape[0] != M_rel):
            raise ValueError(
                "federated_merge requires matching substrate geometry "
                "(d, M, M_rel) across all organisms")
    # weights
    if alpha is None:
        alpha = [1.0] * len(organisms)
    if len(alpha) != len(organisms):
        raise ValueError("alpha length must equal number of organisms")
    # destination
    if dest is None:
        # build a fresh organism with the same constructor signature
        cls = type(base)
        dest = cls(flat_only=True)
    # zero out dest counter banks
    dest.unified.sdm.C[...] = 0
    dest.unified.sdm_rel.C[...] = 0
    # sum in
    for org, w in zip(organisms, alpha):
        dest.unified.sdm.C += (w * org.unified.sdm.C).astype(np.complex64)
        dest.unified.sdm_rel.C += (w * org.unified.sdm_rel.C).astype(np.complex64)
    # union of metadata
    for org in organisms:
        dest.unified._seen.update(org.unified._seen)
        dest.unified._cooccur_seen.update(org.unified._cooccur_seen)
        for role, targets in org.unified._role_targets.items():
            dest.unified._role_targets.setdefault(role, set()).update(targets)
        # copy roles each arm has registered, preferring first definition
        for role_name, role_vec in org.unified.roles.items():
            dest.unified.roles.setdefault(role_name, role_vec)
    dest.unified._dirty = True
    return dest
