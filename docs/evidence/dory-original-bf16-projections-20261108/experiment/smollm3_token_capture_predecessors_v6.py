"""Metadata-only successor catalog builder; final binding requires root evidence.

The current253 baseline is reconstructed from actual Phi v10 admission/audit.
No future Mini units or process identities are invented here. A later root
review must supply every intervening workflow and its final exact unit order.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import smollm3_token_capture_protocol_v6 as p
import smollm3_token_capture_predecessors_v3 as old

q = old.q
BASE = p.ROOT / 'research/minicpm5-layer0-complete-predecessors-20261107-v1.json'
BASE_SHA = '1f3e010a589373738b9533ca4544bd58ae39c919277ca0ee7e2972bf45a7dd9c'
PHI_AUDIT = p.ROOT / 'research/phi4-saved-boundary-cpu-v10-final-root-audit-20261107-v1.json'
PHI_AUDIT_SHA = '8249f8960d3d018d7e3f88f9996fff4f1f97ffba3694f75ada173ddc7893eb53'
CATALOG = p.ROOT / 'research/smollm3-engine-token-capture-predecessors-20261107-v6.json'
NULLSTART_CATALOG = dict(path=str(p.ROOT / 'research/minicpm5-layer0-complete-predecessors-20261107-v2.json'),
    sha256='d04ee13cb7b5a101e88c2aa684d500a786be8b4173ef61cc7f851259d6e9fb52')
NULLSTART_AUDIT = dict(path=str(p.ROOT / 'research/minicpm5-layer0-v3-failed-final-root-audit-20261107-v1.json'),
    sha256='0929ea8d5e353adaf9c1ddd7f3942a79193d8c725004fce6acfd0d0219a3d75e')
NULLSTART_RELEASE = dict(path=str(p.ROOT / 'minicpm5-layer0-control-v3/root-release.json'),
    sha256='06d228d7b3eefa1439a816b7793899e4795cef62ecb624b9e002b066222bb27f')
NULLSTART_UNIT = 'trn1-latest-verify-minicpm5-layer0-v3-controller'


def unknown_start_owner(row):
    """Only the entire already-audited failed Mini v3 row may lack start ticks.

    This authenticates historical metadata only. The runtime caller must require
    PID absence, even if that PID has been reused, plus ordinary service/drain
    checks. No live owner or successful execution is inferred.
    """
    owner = row['owner']
    p.require(isinstance(owner, dict) and row['unit'] == NULLSTART_UNIT and
        owner['unit'] == NULLSTART_UNIT and type(owner['pid']) is int and owner['pid'] == 1432134 and
        owner['start_ticks'] is None and owner.get('start_ticks_observed') is False and
        row.get('captured_live_owner') is False and row['never_launched'] is False and
        row['invocation_id'] == '30de9e643d7a4af08445e7d96ed1d644', 'Only exact audited null-start controller')
    catalog = metadata(NULLSTART_CATALOG); audit = metadata(NULLSTART_AUDIT); release = metadata(NULLSTART_RELEASE)
    recorded = [x for x in catalog['rows'] if x['unit'] == NULLSTART_UNIT]
    p.require(len(recorded) == 1 and row == recorded[0], 'Unknown-start row differs from fixed root catalog')
    p.require(audit['controller_pid'] == owner['pid'] and audit['controller_invocation'] == row['invocation_id'] and
        audit['controller_live_start_observed'] is False and audit['all_actual_groups_drained'] is True and
        audit['root_failed_release'] == NULLSTART_RELEASE and audit['runtime_successful'] is False,
        'Exact root failed audit required')
    p.require(release['released'] is True and release['successful'] is False and release['failed'] is True and
        release['explicit_failed_release'] is True and release['all_current_groups_recursively_drained'] is True,
        'Exact failed release required')
    return owner


def metadata(ref):
    p.performance_guard(); path = Path(ref['path'])
    p.require(path.is_absolute() and path.resolve() == path and not any(x.is_symlink() for x in (path, *path.parents)), 'Fixed catalog metadata path')
    p.require(path.suffix in ('.json', '.jsonl', '.py', '.md') and path.stat().st_size <= 8*2**20, 'Catalog metadata-only type/cap')
    p.require(p.sha(path) == ref['sha256'], 'Catalog evidence hash changed')
    return p.read(path) if path.suffix == '.json' else None


def validate_rows(rows, processes):
    names = [row['unit'] for row in rows]
    p.require(len(names) == len(set(names)), 'Duplicate predecessor unit')
    identities = set()
    for row in rows:
        unit = row['unit']
        p.require(re.fullmatch(r'trn1-[A-Za-z0-9_.@:-]+', unit) is not None and not unit.endswith('.service'), 'Exact safe service unit')
        groups = row['recorded_cgroups']
        p.require(groups and len(groups) == len(set(groups)), 'Missing/duplicate recorded groups')
        for group in groups:q.group_contract(unit, group)
        owner = row['owner']
        if owner is None:
            p.require(row['never_launched'] is True and row['invocation_id'] is None and row['installed_argv'] is None, 'Explicit never-launched state')
            p.require(row['absent_records'] and all(Path(x).is_absolute() for x in row['absent_records']), 'Complete never-launched absence evidence')
        else:
            checked = unknown_start_owner(row) if owner.get('start_ticks') is None else q.owner_key(owner)
            p.require(checked == owner and owner['unit'] == unit and row['never_launched'] is False and row['absent_records'] == [], 'Historical owner identity')
            key = owner['pid'], owner['start_ticks']
            p.require(key not in identities, 'Duplicate historical process identity'); identities.add(key)
            if row['historical_invocation_unknown']:
                p.require(row['invocation_id'] is None and row['installed_argv'] is None, 'Explicit missing historical invocation only')
            else:p.require(old.invocation(row['invocation_id']) and isinstance(row['installed_argv'], str) and row['installed_argv'], 'Actual captured command/invocation required')
    for owner in processes:
        p.require(type(owner['pid']) is int and owner['pid'] > 0 and type(owner['start_ticks']) is str and owner['start_ticks'].isdigit(), 'Actual standalone owner')
        p.require(isinstance(owner['argv'], list) and owner['argv'] and all(type(x) is str for x in owner['argv']), 'Actual standalone command')
        key = owner['pid'], owner['start_ticks'];p.require(key not in identities, 'Duplicate standalone identity');identities.add(key)
    return names


def current253():
    """Reconstruct actual current catalog from retained metadata, never services."""
    base = metadata(dict(path=str(BASE), sha256=BASE_SHA))
    audit = metadata(dict(path=str(PHI_AUDIT), sha256=PHI_AUDIT_SHA))
    p.require(audit['reviewer'] == '/root' and audit['passed'] is True and audit['diagnostic_complete'] is True, 'Actual successful Phi diagnostic root audit')
    pins = dict(base['source_sha256'])
    pins[str(BASE)] = BASE_SHA; pins[str(PHI_AUDIT)] = PHI_AUDIT_SHA
    for ref in (NULLSTART_CATALOG, NULLSTART_AUDIT, NULLSTART_RELEASE):
        metadata(ref); pins[ref['path']] = ref['sha256']
    for path, wanted in audit['source_evidence_sha256'].items():
        p.require(path not in pins or pins[path] == wanted, 'Conflicting actual Phi evidence');pins[path] = wanted
    for path, wanted in pins.items():metadata(dict(path=path, sha256=wanted))
    admission_path = p.ROOT / 'research/phi4-saved-boundary-alignment-cpu-v10-root-runtime-admission.json'
    admission = metadata(dict(path=str(admission_path), sha256=pins[str(admission_path)]))
    external_path = p.ROOT / 'research/phi4-saved-boundary-cpu-v10-root-v1/external-audit.json'
    external = metadata(dict(path=str(external_path), sha256=pins[str(external_path)]))
    release_path = p.ROOT / 'research/phi4-saved-boundary-cpu-v10-root-v1/release.json'
    release = metadata(dict(path=str(release_path), sha256=pins[str(release_path)]))
    p.require(external['passed'] is external['terminal_contract_passed'] is True and release['all_groups_drained'] is True, 'Actual Phi audit/release')
    p.require(release['runtime_admission_sha256'] == external['runtime_admission_sha256'] == pins[str(admission_path)], 'Phi admission binding')
    rows = copy.deepcopy(admission['predecessors'])
    p.require(len(rows) == 251, 'Actual Phi251 admission')
    for key in ('controller_terminal', 'worker_terminal'):
        proof = external['external'][key];owner = proof['owner'];samples = proof['samples']
        p.require(proof['successful'] is proof['physically_drained'] is proof['identity_verified'] is True and len(samples) == 2 and samples[0] == samples[1], 'Retained Phi terminal proof')
        props = samples[0]['properties'];unit = owner['unit'];q.terminal_state(props)
        p.require(props['InvocationID'] == owner['invocation_id'] and props['ExecMainPID'] == str(owner['pid']) and props['Result'] == 'success' and props['ExecMainCode'] == '1' and props['ExecMainStatus'] == '0', 'Actual Phi terminal result identity')
        argv = re.findall(r'argv\[\]=(.*?)\s*;', props['ExecStart']);p.require(len(argv) == 1, 'Exact installed Phi argv')
        rows.append(dict(unit=unit, owner=q.owner_key(dict(owner, script=str(p.ROOT/'phi4_saved_boundary_cpu_v10.py'))), invocation_id=owner['invocation_id'], installed_argv=argv[0], recorded_cgroups=['/sys/fs/cgroup'+owner['cgroup']], historical_invocation_unknown=False, never_launched=False, absent_records=[]))
    p.require(rows == base['rows'] and len(rows) == 253, 'Authoritative253 row ordering/identities changed')
    names = validate_rows(rows, base['processes'])
    # Accepted v3 scope is preserved after removing the four actually launched
    # Phi units; reuse authoritative current ordering rather than reshuffle it.
    added = [name for name in names if '-phi4-saved-boundary-alignment-cpu-v' in name]
    exact_added = ['trn1-latest-verify-phi4-saved-boundary-alignment-cpu-v'+v+'-'+kind for v in ('8', '10') for kind in ('controller', 'worker')]
    p.require(added == exact_added and [n for n in names if n not in added] == old.expected(), 'Original249 plus exact four actual Phi units')
    return dict(rows=rows, process_predecessors=copy.deepcopy(base['processes']), source_sha256=pins, complete_count=253)


def build(root_review_ref):
    """In-memory final union, callable only with explicit post-terminal review."""
    base = current253();review = metadata(root_review_ref)
    p.require(review['schema'] == 'smollm3_capture_v6_final_catalog_review_v1' and review['reviewer'] == '/root' and review['passed'] is True and review['source_only'] is True, 'Independent final catalog review')
    p.require(review['blocking_workflows_terminal'] is True and review['actual_state_checked_by_root'] is True and review['all_planned_launches_accounted_for'] is True, 'Intervening workflow must be terminal and fully accounted for')
    p.require(review['base_catalog_sha256'] == BASE_SHA and review['base_unit_count'] == 253, 'Exact authoritative baseline')
    rows = copy.deepcopy(base['rows']);processes = copy.deepcopy(base['process_predecessors']);pins = dict(base['source_sha256'])
    pins[root_review_ref['path']] = root_review_ref['sha256']
    for ref in review['append_manifests']:
        value = metadata(ref)
        p.require(value['schema'] == 'smollm3_capture_v5_terminal_append_manifest_v1' and value['reviewer'] == '/root' and value['source_only'] is True and value['terminal'] is True and value['all_planned_launches_accounted_for'] is True, 'Root reviewed actual terminal append')
        p.require(value['root_terminal_audits'], 'No terminal root audit evidence')
        audited_rows = []; audited_processes = []
        for auditref in value['root_terminal_audits']:
            evidence = metadata(auditref)
            p.require(evidence['reviewer'] == '/root' and evidence['actual_identity_verified'] is True and evidence['all_groups_drained'] is True and evidence['all_planned_launches_accounted_for'] is True, 'Intervening run identity/drain audit')
            audited_rows.extend(evidence['predecessor_catalog_rows']);audited_processes.extend(evidence['process_predecessors'])
            pins[auditref['path']] = auditref['sha256']
        p.require(audited_rows == value['rows'] and audited_processes == value['process_predecessors'], 'Appended owners differ from exact terminal audit identities')
        for path, wanted in value['source_sha256'].items():
            metadata(dict(path=path, sha256=wanted));p.require(path not in pins or pins[path] == wanted, 'Conflicting appended source');pins[path] = wanted
        pins[ref['path']] = ref['sha256'];rows.extend(value['rows']);processes.extend(value['process_predecessors'])
    units = validate_rows(rows, processes)
    p.require(units == review['expected_units'] and processes == review['expected_process_predecessors'], 'Missing/extra/reordered final predecessor')
    actual_digest = hashlib.sha256(json.dumps(dict(rows=rows, process_predecessors=processes), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    p.require(actual_digest == review['expected_full_catalog_sha256'], 'Root reviewed full owner catalog digest')
    return dict(schema='smollm3_capture_v6_bound_predecessors_v1', source_only=True, rows=rows, process_predecessors=processes,
                source_sha256=pins, final_root_catalog_review=root_review_ref, all_planned_launches_accounted_for=True,
                runtime_physical_drain_required=True, physical_absence_is_not_success=True, complete_count=len(rows))


def bound(plan):
    ref = plan['successor_predecessor_catalog'];value = metadata(ref)
    p.require(Path(ref['path']) == CATALOG and value['schema'] == 'smollm3_capture_v6_bound_predecessors_v1', 'Exact final catalog path/schema')
    p.require(build(value['final_root_catalog_review']) == value, 'Final catalog differs from reviewed terminal evidence')
    p.require([row['unit'] for row in value['rows']] == plan['planned_predecessors'], 'Plan full ordered catalog')
    p.require(all(plan['source_sha256'].get(path) == digest for path, digest in value['source_sha256'].items()), 'Complete catalog evidence source closure')
    return value


def canonical_tail(rows):
    return old.canonical_tail(rows)
