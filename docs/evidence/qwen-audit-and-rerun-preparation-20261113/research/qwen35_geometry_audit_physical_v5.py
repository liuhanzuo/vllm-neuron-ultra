"""Future audit-only physical pass boundaries; no execution or IO on import.

Original full-tree physical bodies and pair validators remain authoritative.
Only the complete pair collector's per-pass begin/end guards force discovery.
"""
import copy
from qwen35_package_preflight_history_v3 import History as OriginalHistory
from qwen35_sampler_runtime_history_v12 import LOCK_IDENTITY, require, process_api


def _collect_pair(done, guard, physical, read_properties, read_empty_group, *, known_units=None):
    """Future admitted caller only; no invocation during source preparation.

    All callbacks remain caller-owned and source-pinned. Two complete full-tree
    reads plus exact missing-field supplements occur under caller's original
    lock and existing resource guard. Bounds are not extended if work overruns.
    """
    from copy import deepcopy
    from research.minicpm5_sdk_physical_evidence_v3 import (
        require, unit_name, MAX_UNITS, _tree_header, inspect_tree, GROUP_PREFIX,
        observation, validate_observation, SCHEMA, validate_pair)
    require(callable(guard) and callable(physical) and callable(read_properties)
            and callable(read_empty_group), 'Explicit original admitted callbacks')
    required = {unit_name(row['unit']) for row in done['nemotron']['history']['rows']}
    require(len(required) == 1308, 'Exact original full history before observations')
    known = required if known_units is None else {unit_name(x) for x in known_units}
    require(required <= known and len(known) <= MAX_UNITS, 'Authenticated original plus exact saved supplemental unit set')
    values = []
    for _ in range(2):
        guard(force=True)
        tree = deepcopy(physical(done, guard))
        _tree_header(tree)
        require(tree['all_additional_owners_checked'] == done['owners']
                and tree['lock_identity'] == done['lock_identity'], 'Fresh original owner/lock-bound full tree')
        scan = inspect_tree(tree, known)
        require(required <= scan['units'], 'No original historical unit omitted')
        supplemental = []
        for name in sorted(scan['needs_supplement']):
            guard()
            # name derives only from this pinned original tree's validated
            # trn1 unit/Id/group evidence, never from a caller-supplied pathname.
            props = read_properties(name)
            group = read_empty_group(GROUP_PREFIX+name+'.service')
            row = dict(unit=name, properties=props, group=group)
            observation(row, name)
            supplemental.append(row)
        value = dict(tree=tree, supplemental_observations=supplemental)
        validate_observation(value, known)
        values.append(value)
        guard(force=True)
    pair = dict(schema=SCHEMA, first=values[0], second=values[1])
    validate_pair(pair, required, done['owners'], done['lock_identity'], known_units=known)
    return pair


def collect_pair(done, guard, authority):
    """Original current collector wiring to the boundary-aware exact pair body."""
    from research import minicpm5_native_current_physical_v1 as original
    authority.disjoint(done['complete_known_units'])
    return _collect_pair(done, guard, lambda value, check: original.physical(value, check, authority),
        original.p.properties, original.p.empty_group, known_units=done['complete_known_units'])


class History(OriginalHistory):
    """Inherit full resolve/identities; change only physical collector import."""
    def physical_pair(self, authority):
        import qwen35_geometry_audit_physical_v5 as physical
        from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
        from research.audit_minicpm5_layer1_operator_sdk_result_v3 import packed_physical
        self.guard(); pair = physical.collect_pair(self.scope, self.guard, authority)
        required = {row["unit"] for row in self.scope["nemotron"]["history"]["rows"]}
        known = set(self.scope["complete_known_units"])
        require(completed.physical.validate_pair(pair, required, self.scope["owners"], LOCK_IDENTITY,
                                                known_units=known) == known, "entire original two-pass physical scope")
        from qwen35_sampler_history_supplement_v1 import observe
        retirement = observe(self.supplement, process_api(), self.guard)
        return {"schema": "qwen35_sampler_full_history_pair_v1", "history_supplement": retirement, "current_serial_catalog": self.spec["current_serial_catalog"],
                "history_overlay": self.spec["history_overlay"], "resolved_plan_sha256": self.resolved["plan_sha256"],
                "physical_pair_schema": completed.physical.PAIR_SCHEMA, "paired_physical": packed_physical(pair),
                "full_physical_passes": 2, "complete_known_units": sorted(known),
                "owners": copy.deepcopy(self.scope["owners"]), "lock_identity": copy.deepcopy(LOCK_IDENTITY),
                "original_failed_model_exit_code": 1, "full_model_qualified": False}
