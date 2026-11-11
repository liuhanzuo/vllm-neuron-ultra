"""Reuse completed serial metadata; no native/source-payload IO or admission.

The old Mini scientific plan is read only by its existing completion validator.
Its library/model source map is historical provenance, never this LFM worker's
execution dependency map. Fresh physical observations use the exact derived
unit/owner union, including the actual current Mini run after its external exit.
"""
import copy
from pathlib import Path
from research import minicpm5_layer1_operator_sdk_completion_v2 as sdk
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as serial
from research import minicpm5_sdk_physical_evidence_v3 as physical
import minicpm5_layer0_execution_protocol_v4 as p

ROOT = Path(__file__).resolve().parent
Q = ROOT / 'research'
MINIMUM = dict(path=str(Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261111-v45.json'),
               sha256='bda6d214cf0a77e379f1aeba68c669d89f6977049bcea4e04bb79c0dfe91a426')
MINI_LIVE = dict(path=str(Q / 'minicpm5-layer1-operator-native-r2-catalog256-actual-live-20261111-v1.json'),
                 sha256='3b911911696eb151e797a0083f449f8df1dc78ac7f8eefe0fd9521e5e6656164')
LOCK = dict(device=66306, inode=8294421)


def retained_runs(current, minimum, live):
    """Pure check; an actual failed exit is retained as failure, not promoted."""
    rows = {row['terminal']['path']: row for row in current['runs']}
    p.require(len(rows) == len(current['runs']), 'One row per actual serial terminal')
    p.require(all(rows.get(row['terminal']['path']) == row for row in minimum['runs']),
              'Every original current catalog row retained verbatim')
    matches = [row for row in current['runs'] if row['tool'].get('kind') == 'tool_session'
               and row['tool'].get('session_id') == live['actual_execution_handle']['session_id']]
    p.require(len(matches) == 1, 'Actual current Mini exit must be catalogued; live-only is insufficient')
    row = matches[0]
    p.require(type(row['tool']['exit_code']) is int and 0 <= row['tool']['exit_code'] <= 255,
              'Real successful OR failed Mini tool exit required')
    keys = ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')
    for owner in (live['owner'], live['coordinator']):
        p.require(any(all(candidate.get(key) == owner[key] for key in keys) for candidate in row['owners']),
                  'Current Mini outer and service identities retained')
    p.require(live['coordinator']['unit'] in row['units'], 'Actual Mini service in complete drain scope')
    return row


def resolve(catalog_ref, terminal_ref, read, guard):
    """Existing SDK completed metadata + all later root-observed serial rows."""
    guard()
    model = read(sdk.BINDING['model_exit'])
    old_spec = read(model['spec'])
    historical_plan = copy.deepcopy(read(old_spec['plan']))
    ended = read(sdk.BINDING['audit_exit'])
    supplement = read(ended['post_exit_source_supplement'])
    for path, wanted in supplement['source_sha256'].items():
        p.require(path not in historical_plan['source_sha256'] or historical_plan['source_sha256'][path] == wanted,
                  'Unchanged historical source provenance')
        historical_plan['source_sha256'][path] = wanted
    historical_plan['layer1_operator_boundary'] = copy.deepcopy(sdk.BINDING)
    historical_plan['sdk_operator_manifest'] = sdk.manifest.manifest()
    original = sdk.completed(historical_plan, read, guard)
    p.require(original['lock_identity'] == LOCK, 'Original history lock identity')
    catalog, minimum, live = read(catalog_ref), read(MINIMUM), read(MINI_LIVE)
    required_run = retained_runs(catalog, minimum, live)
    p.require(catalog['current_handoff'] == terminal_ref, 'Root binds the actual current idle terminal')
    provenance = {}
    for row in catalog['runs']:
        for path, wanted in row['source_sha256'].items():
            p.require(path not in provenance or provenance[path] == wanted, 'Historical source conflict')
            provenance[path] = wanted
    view = dict(latest_completion=historical_plan['latest_completion'], serial_predecessors=catalog_ref,
                source_sha256=provenance)
    later = serial.serial_catalog(view, read, guard)
    freshness = serial.require_latest_terminal(view, read, guard)
    owners = copy.deepcopy(original['owners']) + copy.deepcopy(later['owners'])
    units = set(original['complete_known_units']) | set(later['units'])
    for owner in owners:
        group = owner.get('cgroup', '')
        if group.startswith('0::' + p.GROUP_PREFIX):
            units.add(physical.group_unit(group[3:]))
    p.require(live['coordinator']['unit'] in units and len(units) <= physical.MAX_UNITS,
              'Whole historical service scope, including latest Mini')
    return dict(owners=owners, units=sorted(units), lock_identity=LOCK, catalog=catalog_ref,
                terminal=terminal_ref, freshness=freshness, required_run=required_run,
                historical_sdk=sdk.BINDING, minimum_catalog=MINIMUM,
                historical_source_provenance=provenance, runtime_admitted=False)


def collect_drain(scope, guard):
    """Two complete fresh passes; never reuse old physical evidence as current."""
    samples = []
    for _ in range(2):
        guard()
        for owner in scope['owners']:
            guard(); p.gone(owner)
        rows = []
        for unit in scope['units']:
            guard()
            row = dict(unit=unit, properties=p.properties(unit),
                       group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
            physical.observation(row, unit)
            rows.append(row)
        samples.append(rows)
        guard()
    p.require(samples[0] == samples[1], 'Every historical service stable across both complete passes')
    return dict(complete=True, physical_checks_performed=True, lock_identity=LOCK,
                owners_checked=scope['owners'], units=scope['units'], samples=samples,
                current_catalog=scope['catalog'], current_terminal=scope['terminal'],
                current_mini_actual_tool=scope['required_run']['tool'])
