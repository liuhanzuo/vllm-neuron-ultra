"""Strict complete historical drain evidence, metadata-only successor.

Supports the four actual saved cgroup return shapes from low.empty_group and
life.cgroup_inventory. Never calls systemctl/proc/cgroup APIs on import or in
validation. Future collect_pair receives explicitly injected admitted callbacks.
Raw trees are retained verbatim; missing legacy Job values are never invented.
"""
from copy import deepcopy
import re
from pathlib import PurePosixPath

SCHEMA = 'minicpm5_sdk_complete_physical_pair_v3'
PAIR_SCHEMA = SCHEMA
GROUP_PREFIX = '/user.slice/user-1000.slice/user@1000.service/app.slice/'
LEGACY_PROPERTIES = {'ActiveState', 'ControlPID', 'ExecMainCode', 'ExecMainPID',
    'ExecMainStatus', 'Id', 'InvocationID', 'LoadState', 'MainPID', 'Result', 'SubState'}
MAX_NODES, MAX_DEPTH, MAX_UNITS = 500000, 80, 4096


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def unit_name(value):
    require(type(value) is str and re.fullmatch(r'trn1-[A-Za-z0-9_.@:-]{1,240}(?:\.service)?', value)
            and '/' not in value and '..' not in value, 'Bounded exact trn1 service name')
    name = value.removesuffix('.service')
    require(not name.endswith('.service'), 'One optional service suffix only')
    return name


def group_unit(value):
    require(type(value) is str and value.startswith(GROUP_PREFIX) and value.endswith('.service'),
            'Exact owned app.slice service cgroup path')
    name = unit_name(value[len(GROUP_PREFIX):])
    require(value == GROUP_PREFIX+name+'.service', 'Canonical original unit cgroup')
    return name


def empty_group(value):
    require(type(value) is dict and 'group' in value and type(value.get('absent')) is bool,
            'Complete known cgroup evidence object')
    name = group_unit(value['group'])
    keys = set(value)
    if keys == {'group', 'absent'}:
        require(value['absent'] is True, 'Summary absent form must actually be absent')
    elif keys == {'group', 'absent', 'cgroups'}:
        require(value['absent'] is False and type(value['cgroups']) is int
                and 1 <= value['cgroups'] <= 256, 'Original complete recursive empty-group summary')
    elif keys == {'group', 'absent', 'pids', 'directories'}:
        require(value['pids'] == [] and type(value['pids']) is list
                and type(value['directories']) is list, 'No actual process in detailed cgroup inventory')
        paths = value['directories']
        if value['absent']:
            require(paths == [], 'Absent detailed group cannot contain directories')
        else:
            expected = '/sys/fs/cgroup'+value['group']
            require(1 <= len(paths) <= 256 and paths == sorted(set(paths)) and expected in paths,
                    'Every existing detailed group includes exact root and bounded unique directories')
            for path in paths:
                require(type(path) is str and str(PurePosixPath(path)) == path
                        and '..' not in PurePosixPath(path).parts
                        and (path == expected or path.startswith(expected+'/')),
                        'All detailed descendants stay inside exact owned root')
    else:
        raise RuntimeError('Unknown or contradictory cgroup evidence schema')
    return name


def properties(value, expected=None, *, legacy_missing_job=False):
    require(type(value) is dict and {'Id', 'MainPID', 'ControlPID', 'ActiveState', 'SubState'} <= set(value),
            'Complete terminal properties and exact service Id')
    name = unit_name(value['Id'])
    require(value['Id'] == name+'.service' and (expected is None or name == expected),
            'Property Id matches its own exact unit')
    require(value['MainPID'] == value['ControlPID'] == '0'
            and (value['ActiveState'], value['SubState']) in {
                ('inactive', 'dead'), ('failed', 'failed'), ('active', 'exited')},
            'No main/control process and a legitimate terminal state pair')
    if 'Job' in value:
        require(value['Job'] == '', 'No pending systemd Job')
        missing_job = False
    else:
        require(legacy_missing_job and set(value) == LEGACY_PROPERTIES,
                'Only exact original legacy property schema may lack Job')
        missing_job = True
    if 'LoadState' in value:
        require(value['LoadState'] in ('loaded', 'not-found'), 'Loaded or collected original service')
        if value['LoadState'] == 'not-found':
            require(value.get('InvocationID') == '' and value.get('ExecMainPID') == '0'
                    and (value['ActiveState'], value['SubState']) == ('inactive', 'dead'),
                    'Collected service has no invocation or former main identity')
    if 'ControlGroup' in value:
        require(value['ControlGroup'] in ('', GROUP_PREFIX+name+'.service'), 'Property cgroup belongs to same unit')
    return name, missing_job


def observation(sample, expected=None, *, legacy_missing_job=False):
    require(type(sample) is dict and 'properties' in sample
            and ('group' in sample) != ('groups' in sample), 'One original property/group observation')
    name, missing = properties(sample['properties'], expected, legacy_missing_job=legacy_missing_job)
    groups = [sample['group']] if 'group' in sample else sample['groups']
    require(type(groups) is list and len(groups) == 1, 'One exact unique owned group per unit sample')
    require(empty_group(groups[0]) == name, 'Observed cgroup path binds the property Id')
    return name, missing


def inspect_tree(tree, known_units=None):
    """Validate every existing evidence field; identify only real missing data.

    Some pinned old producers omit Job, or retain only model_group summaries.
    These need a complete successor observation; neither is silently accepted
    as a full service terminal. Complete service owners identify supplemental scope; bare unit rows without
    process identities are invalid. Owner identities never imply terminal state.
    """
    require(type(tree) is dict, 'Complete historical physical tree')
    units, grouped, complete, incomplete, observations = set(), set(), set(), set(), {}
    owner_units, owner_records = set(), {}
    nodes = 0

    def track_sample(sample, expected=None):
        name, missing = observation(sample, expected, legacy_missing_job=True)
        units.add(name); grouped.add(name)
        observations.setdefault(name, []).append(deepcopy(sample['properties']))
        (incomplete if missing else complete).add(name)
        # Outer observation metadata may contain owners; the group/properties
        # have already been checked and must not be counted as orphan objects.
        return name

    def walk(value, depth=0):
        nonlocal nodes
        nodes += 1
        require(nodes <= MAX_NODES and depth <= MAX_DEPTH, 'Bounded complete tree traversal')
        if type(value) is list:
            for child in value: walk(child, depth+1)
            return
        if type(value) is not dict: return
        if type(value.get('group')) is str:
            name = empty_group(value); units.add(name); grouped.add(name)
            return
        if 'samples' in value:
            samples = value['samples']
            require(type(samples) is list and len(samples) == 2 and samples[0] == samples[1],
                    'Every original nested double observation remains complete and stable')
            expected = unit_name(value['unit']) if 'unit' in value else None
            require(expected is not None or type(value.get('stage')) is str,
                    'Original nested samples have unit or stage identity')
            require(all(type(sample) is dict and set(sample) in ({'properties', 'group'}, {'properties', 'groups'})
                        for sample in samples), 'Nested samples retain exact complete observation schema')
            names = [track_sample(sample, expected) for sample in samples]
            require(names[0] == names[1], 'Both samples observe the same exact service')
            for key, child in value.items():
                if key not in ('samples', 'unit', 'stage'): walk(child, depth+1)
            return
        if 'properties' in value:
            expected = unit_name(value['unit']) if 'unit' in value else None
            track_sample(value, expected)
            for key, child in value.items():
                if key not in ('properties', 'group', 'groups', 'unit', 'stage'): walk(child, depth+1)
            return
        if 'cgroup' in value or 'unit' in value:
            # An owner is only an identity, never terminal/group evidence.
            # Legacy sparse owners with no cgroup/unit remain verbatim and are
            # checked by their original producers and exact owner-list binding.
            require({'pid', 'start_ticks', 'argv', 'cgroup'} <= set(value)
                    and type(value['pid']) is int and value['pid'] > 0
                    and type(value['start_ticks']) in (str, int)
                    and re.fullmatch(r'[1-9][0-9]*', str(value['start_ticks']))
                    and type(value['argv']) is list and len(value['argv']) <= 4096
                    and all(type(arg) is str and arg and '\x00' not in arg for arg in value['argv'])
                    and type(value['cgroup']) is str and value['cgroup'].startswith('0::/'),
                    'Original owner identity and canonical unified cgroup')
            path = value['cgroup'][3:]
            require(str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
                    'Canonical owner cgroup has no traversal or alias')
            if path.startswith(GROUP_PREFIX) or 'unit' in value:
                name = group_unit(path)
                require(type(value.get('ppid')) is int and value['ppid'] > 0,
                        'Owned service process has its complete original parent identity')
                if 'unit' in value:
                    require(unit_name(value['unit']) == name
                            and type(value.get('invocation_id')) is str
                            and re.fullmatch(r'[a-f0-9]{32}', value['invocation_id']),
                            'Coordinator unit, cgroup and actual invocation are bound')
                owner_records.setdefault(name, []).append(deepcopy(value))
                units.add(name); owner_units.add(name)
        if 'stage' in value and type(value['stage']) is str:
            require('samples' in value or 'properties' in value,
                    'Historical stage cannot lose its property/group observation')
        for child in value.values(): walk(child, depth+1)

    walk(tree)
    for name in owner_units - grouped:
        # Existing historical service observations may retain pre-exec, exec
        # and zombie owner snapshots. Keep them all under original semantics.
        # Newly owner-only scope must have full consistent process identities.
        identities = {}
        for owner in owner_records[name]:
            require(owner['argv'], 'Owner-only service requires complete nonempty original argv')
            common = {key: owner[key] for key in ('pid','start_ticks','ppid','argv','cgroup')}
            key = (owner['pid'], str(owner['start_ticks']))
            require(key not in identities or identities[key] == common,
                    'Owner-only repeated coordinator/rank retains all common identity fields')
            identities[key] = common
    require(units and len(units) <= MAX_UNITS and units == grouped | owner_units,
            'Every named unit has original group or validated owner identity')
    if known_units is not None:
        known = {unit_name(x) for x in known_units}
        require(units == known, 'Exact complete authenticated service set without missing or extra units')
    return dict(units=units, needs_supplement=(units-complete) | incomplete,
                properties=observations, owner_only={name: owner_records[name] for name in owner_units-grouped}, nodes=nodes)


def _tree_header(tree):
    require(type(tree) is dict and tree.get('complete') is True
            and tree.get('physical_checks_performed') is True
            and tree.get('all_current_and_prior_services_preserved') is True
            and type(tree.get('all_additional_owners_checked')) is list
            and type(tree.get('lock_identity')) is dict, 'Complete actual full-tree owner/lock header')
    n = tree.get('nemotron')
    require(type(n) is dict and n.get('complete') is True and n.get('fresh_physical_drain_verified') is True
            and type(n.get('services')) is int and n['services'] == 1308, 'Original1308 historical service scope')


def validate_observation(value, known_units=None):
    require(type(value) is dict and set(value) == {'tree', 'supplemental_observations'},
            'Complete unchanged tree plus explicit exact supplemental observations')
    tree = value['tree']; _tree_header(tree)
    scan = inspect_tree(tree, known_units)
    supplements = value['supplemental_observations']
    require(type(supplements) is list and len(supplements) == len(scan['needs_supplement']),
            'Exactly every missing Job, group-only or owner-only unit receives real evidence')
    expected = sorted(scan['needs_supplement'])
    require([row.get('unit') for row in supplements if type(row) is dict] == expected,
            'Sorted exact supplemental unit set without omissions, substitutions or duplicates')
    for row in supplements:
        require(set(row) == {'unit', 'properties', 'group'}, 'Exact new full supplemental observation')
        require(LEGACY_PROPERTIES | {'Job'} <= set(row['properties'])
                and all(type(v) is str for v in row['properties'].values()),
                'Supplement retains full real terminal property set including explicit Job')
        name, missing = observation(row, row['unit'])
        require(not missing and name in scan['needs_supplement'], 'Supplement has actual explicit empty Job')
        if row['properties']['LoadState'] == 'loaded':
            for owner in scan['owner_only'].get(name, []):
                if 'unit' in owner:
                    require(row['properties']['InvocationID'] == owner['invocation_id']
                            and row['properties']['ExecMainPID'] == str(owner['pid']),
                            'Loaded owner-only service retains original coordinator invocation and PID')
        # Preserve all observable original properties, not merely its unit name.
        for prior in scan['properties'].get(name, []):
            for key in ('Id', 'MainPID', 'ControlPID', 'ActiveState', 'SubState', 'Job',
                        'LoadState', 'InvocationID', 'ExecMainPID'):
                if key in prior:
                    require(row['properties'].get(key) == prior[key], 'Supplement agrees with the entire prior service identity/state')
    return scan['units']


def validate_pair(pair, required_units=None, expected_owners=None, lock=None, *, known_units=None):
    """Public API: decoded {schema,first,second} -> exact set of unit basenames.

    first and second are {tree, supplemental_observations}. Caller recorder can
    access pair['first']['tree']['all_additional_owners_checked'] explicitly.
    required_units is the exact1308 original history set; supplemental units are
    retained in returned set for equality against the saved outer observation.
    """
    require(type(pair) is dict and set(pair) == {'schema', 'first', 'second'}
            and pair['schema'] == SCHEMA, 'Explicit paired complete-tree envelope')
    first, second = pair['first'], pair['second']
    units = validate_observation(first, known_units)
    other = validate_observation(second, known_units)
    require(first == second and units == other, 'Both entire original trees and supplements are stable and equal')
    if required_units is not None:
        wanted = {unit_name(x) for x in required_units}
        require(len(wanted) == 1308 and wanted <= units, 'Every exact original1308 unit retained')
    for value in (first, second):
        tree = value['tree']
        if expected_owners is not None:
            require(tree['all_additional_owners_checked'] == expected_owners, 'All exact additional owners retained')
        if lock is not None:
            require(tree['lock_identity'] == lock, 'Both observations on the same original lock')
    return units


def collect_pair(done, guard, physical, read_properties, read_empty_group, *, known_units=None):
    """Future admitted caller only; no invocation during source preparation.

    All callbacks remain caller-owned and source-pinned. Two complete full-tree
    reads plus exact missing-field supplements occur under caller's original
    lock and existing resource guard. Bounds are not extended if work overruns.
    """
    require(callable(guard) and callable(physical) and callable(read_properties)
            and callable(read_empty_group), 'Explicit original admitted callbacks')
    required = {unit_name(row['unit']) for row in done['nemotron']['history']['rows']}
    require(len(required) == 1308, 'Exact original full history before observations')
    known = required if known_units is None else {unit_name(x) for x in known_units}
    require(required <= known and len(known) <= MAX_UNITS, 'Authenticated original plus exact saved supplemental unit set')
    values = []
    for _ in range(2):
        guard()
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
        guard()
    pair = dict(schema=SCHEMA, first=values[0], second=values[1])
    validate_pair(pair, required, done['owners'], done['lock_identity'], known_units=known)
    return pair
