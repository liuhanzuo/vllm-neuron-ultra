"""Versioned, lossless references at fixed formal-import argv positions only.

No IO, native import, process action, authority or source substitution. The
literal is the unchanged, independently pinned v6 -c payload. In-memory owners
remain full owners. Wire hashes authenticate file bytes; expanded hashes bind
the complete original JSON object. This module never follows a file pointer.
"""
import json
from research.minicpm5_sdk_mapping_nocache_entry_v6 import digest

LITERAL = dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_sdk_mapping_nocache_entry_v6.py',
               sha256='bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0', bytes=37178)
WIRE = '_minicpm5_literal_wire_v1'
NODE = '_minicpm5_literal_source_v1'
SCHEMA = 'minicpm5_formal_literal_boundary_v1'
OWNER_KEYS = {'pid', 'ppid', 'start_ticks', 'argv', 'cgroup'}
ADMISSION_KEYS = {'schema', 'witness', 'lease', 'output_root', 'original_native_source_receipt',
                  'diagnostic_spec', 'diagnostic_review', 'source_entry_plan', 'source_entry_binding'}
EXIT_KEYS = {'schema', 'phase', 'owner', 'argv', 'child_pid', 'returncode', 'parent_observed_wait',
             'process_group_empty', 'exact_owner_gone', 'owner_acknowledged', 'self_exit_claimed',
             'cleanup_after_failure'}
FALLBACK_KEYS = {'event', 'actual_parent_observed_terminal', 'exit_receipt_saved',
                 'reporting_error_type', 'diagnostic_passed'}
RUNTIME_CAP = 79008
SUMMARY_CAP = 16 * 1024
FILE_CAP = 2 * 2**20
OBSERVER_RESERVE = SUMMARY_CAP


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def parsed(raw):
    require(type(raw) is bytes and 0 < len(raw) <= FILE_CAP, 'Whole original bounded JSON bytes')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'Duplicate JSON key rejected')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON rejected: ' + value)
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    require(encoded(result) == raw, 'Exact canonical complete JSON, including newline')
    return result


def literal_bytes(literal, check=lambda: None):
    require(type(literal) is str, 'Actual complete -c literal string')
    raw = literal.encode()
    require(len(raw) == LITERAL['bytes'] and digest(raw, check) == LITERAL['sha256'],
            'Only unchanged pinned v6 bytes may replace argv[7]')
    return raw


def reference(value):
    require(type(value) is dict and set(value) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
            'Exact content reference fields')
    path, sha = value['path'], value['sha256']
    require(type(path) is str and path.startswith('/') and all(x and x not in ('.', '..')
            for x in path.split('/')[1:]), 'Canonical absolute reference path')
    require(type(sha) is str and len(sha) == 64 and all(x in '0123456789abcdef' for x in sha),
            'Exact SHA256 reference')
    if 'bytes' in value:
        require(type(value['bytes']) is int and 0 <= value['bytes'] <= FILE_CAP, 'Original file byte bound')


def _at(value, path):
    for key in path:
        require(type(value) is dict and key in value, 'Missing fixed codec position')
        value = value[key]
    return value


def _owner(owner):
    require(type(owner) is dict and set(owner) == OWNER_KEYS, 'Every original owner field, no additions')
    require(type(owner['pid']) is int and 0 < owner['pid'] < 2**31
            and type(owner['ppid']) is int and 0 <= owner['ppid'] < 2**31
            and type(owner['start_ticks']) is str and owner['start_ticks'].isdigit()
            and 0 <= int(owner['start_ticks']) < 2**64
            and str(int(owner['start_ticks'])) == owner['start_ticks']
            and type(owner['cgroup']) is str and owner['cgroup'], 'Exact Linux process owner scalar domains')


def _exit(value):
    require(type(value) is dict and set(value) == EXIT_KEYS
            and value['schema'] == 'minicpm5_pv_sdk_import_diagnostic_actual_child_exit_v1'
            and type(value['phase']) is int and value['phase'] == 1,
            'Complete original phase1 terminal schema')
    require(type(value['child_pid']) is int and 0 < value['child_pid'] < 2**31
            and type(value['returncode']) is int and -127 <= value['returncode'] <= 255,
            'Actual wait return domain')
    require(value['parent_observed_wait'] is True and value['process_group_empty'] is True
            and value['self_exit_claimed'] is False and type(value['cleanup_after_failure']) is bool,
            'Preserve actual parent wait/drain facts')
    if value['owner'] is None:
        require(value['owner_acknowledged'] is False and value['exact_owner_gone'] is None,
                'Explicit unacknowledged owner remains absent')
    else:
        _owner(value['owner'])
        require(value['child_pid'] == value['owner']['pid'] and value['owner_acknowledged'] is True
                and value['exact_owner_gone'] is True and value['owner']['argv'] == value['argv'],
                'Same complete actual owner and argv in terminal')


def paths(kind, value):
    """Fixed semantic positions; no user supplied JSON pointer or wildcard."""
    if kind == 'admission':
        require(type(value) is dict and set(value) == ADMISSION_KEYS
                and value['schema'] == 'minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1',
                'Complete original admission fields only')
        _owner(value['lease']['outer_owner'])
        _owner(value['lease']['initial_descriptor_proof']['owner'])
        require(value['lease']['outer_owner'] == value['lease']['initial_descriptor_proof']['owner'],
                'Original descriptor proof retains identical whole outer owner')
        require(type(value['source_entry_plan']) is dict and value['source_entry_plan']['role'] == 'sdk310',
                'Unchanged inline SDK plan remains visible to frozen v6 stage0')
        return (('lease', 'outer_owner', 'argv'), ('lease', 'initial_descriptor_proof', 'owner', 'argv'))
    if kind == 'owner':
        _owner(value)
        return (('argv',),)
    if kind == 'exit':
        _exit(value)
        return (('argv',),) if value['owner'] is None else (('owner', 'argv'), ('argv',))
    if kind == 'exit_fallback':
        require(type(value) is dict and set(value) == FALLBACK_KEYS
                and value['event'] == 'sdk_import_diagnostic_exit_receipt_write_failed'
                and value['exit_receipt_saved'] is False and value['diagnostic_passed'] is False
                and type(value['reporting_error_type']) is str, 'Complete original failed-persistence event')
        terminal = value['actual_parent_observed_terminal']
        return tuple(('actual_parent_observed_terminal',) + path for path in paths('exit', terminal))
    raise ValueError('Unknown codec kind')


def _reserved(value, allowed, prefix=()):
    """Reject codec markers anywhere except the complete enumerated argv node."""
    if type(value) is dict:
        if NODE in value:
            require(prefix in allowed and value == {NODE: LITERAL}, 'No arbitrary/partial literal reference')
            return
        require(WIRE not in value, 'Nested wire envelope rejected')
        for key, item in value.items():
            _reserved(item, allowed, prefix + (key,))
    elif type(value) is list:
        for index, item in enumerate(value):
            _reserved(item, allowed, prefix + (index,))


def _argv(argv, literal, role, packed):
    require(type(argv) is list and len(argv) == 15
            and argv[1:7] == ['-I', '-S', '-B', '-X', 'pycache_prefix=/dev/null', '-c']
            and argv[8:10] == ['--mini-source-entry', role]
            and argv[13] == LITERAL['sha256'], 'Exact unchanged complete literal argv layout')
    python = '/home/ubuntu/venvs/model_reference_2026/bin/python' if role == 'outer312' else '/home/ubuntu/venvs/vllm_neuron/bin/python'
    require(argv[0] == python and all(type(x) is str and '\0' not in x for i, x in enumerate(argv) if i != 7),
            'Original role/interpreter and every actual nonliteral argument preserved')
    require(argv[7] == ({NODE: LITERAL} if packed else literal), 'Exact complete v6 literal at argv[7]')


def encode(kind, value, literal, check=lambda: None):
    literal_bytes(literal, check)
    raw = encoded(value)
    require(len(raw) <= (2**20 if kind == 'admission' else FILE_CAP), 'Original expanded input/file ceiling')
    result = parsed(raw)
    selected = paths(kind, result)
    _reserved(result, set())
    for path in selected:
        argv = _at(result, path)
        _argv(argv, literal, 'outer312' if kind == 'admission' else 'sdk310', False)
        argv[7] = {NODE: dict(LITERAL)}
    result[WIRE] = dict(schema=SCHEMA, kind=kind, expanded_sha256=digest(raw, check), expanded_bytes=len(raw))
    wire = encoded(result)
    require(len(wire) <= FILE_CAP, 'Original whole wire file cap')
    return wire


def decode(kind, wire, literal, *, wire_sha256, check=lambda: None):
    literal_bytes(literal, check)
    raw = wire if type(wire) is bytes else encoded(wire)
    require(digest(raw, check) == wire_sha256, 'Actual canonical wire/file SHA256 mismatch')
    result = parsed(raw)
    meta = result.pop(WIRE, None)
    require(type(meta) is dict and set(meta) == {'schema', 'kind', 'expanded_sha256', 'expanded_bytes'}
            and meta['schema'] == SCHEMA and meta['kind'] == kind, 'Exact versioned complete boundary envelope')
    selected = paths(kind, result)
    _reserved(result, {path + (7,) for path in selected})
    for path in selected:
        argv = _at(result, path)
        _argv(argv, literal, 'outer312' if kind == 'admission' else 'sdk310', True)
        argv[7] = literal
    expanded = encoded(result)
    require(type(meta['expanded_bytes']) is int and len(expanded) == meta['expanded_bytes']
            and len(expanded) <= (2**20 if kind == 'admission' else FILE_CAP)
            and digest(expanded, check) == meta['expanded_sha256'], 'Complete expanded object size/SHA256 mismatch')
    paths(kind, result)
    return result


def adopt_entry_admission(entry):
    """First SDK successor action, before guard construction or native history.

    Frozen v6 already authenticated the wire reference and selected the full
    inline SDK plan. Keep that actual file reference and replace only the
    in-memory envelope with its fully verified original representation.
    """
    entry.check()
    require(entry.role == 'sdk310' and entry.admission_envelope is not None
            and not hasattr(entry, 'literal_wire_admission'), 'Once-only original SDK entry decode')
    reference(entry.plan_reference)
    wire = entry.admission_envelope
    value = decode('admission', wire, entry.runtime.orig_argv[7],
                   wire_sha256=entry.plan_reference['sha256'], check=entry.check)
    require(value['source_entry_plan'] == entry.plan, 'Frozen stage0 plan equals complete decoded admission plan')
    entry.literal_wire_admission = wire
    entry.admission_envelope = value
    entry.check()
    return value


class AdmissionMetadata:
    """One-reference view after the original stable Metadata.read succeeds."""
    def __init__(self, original, admission_ref, literal, check):
        reference(admission_ref)
        self.original, self.admission_ref, self.literal, self.check = original, dict(admission_ref), literal, check

    def read(self, ref):
        value = self.original.read(ref)
        if ref == self.admission_ref:
            return decode('admission', value, self.literal, wire_sha256=ref['sha256'], check=self.check)
        return value

    def __getattr__(self, name):
        return getattr(self.original, name)


def prospective_terminal(admission_wire, argv, literal):
    """Exact proposed admission/argv, conservative scalar sizes, no fake owner receipt.

    These maximum-width process values exist only for byte reservation. No
    native claim, saved owner, or release packet may be made from them.
    """
    admission = decode('admission', admission_wire, literal, wire_sha256=digest(admission_wire))
    outer = admission['lease']['outer_owner']
    require(type(argv) is list, 'Actual proposed SDK argv')
    owner_bound = dict(pid=2**31-1, ppid=outer['pid'], start_ticks=str(2**64-1),
                       argv=argv, cgroup=outer['cgroup'])
    terminal_bound = dict(schema='minicpm5_pv_sdk_import_diagnostic_actual_child_exit_v1', phase=1,
        owner=owner_bound, argv=argv, child_pid=owner_bound['pid'], returncode=-127,
        parent_observed_wait=True, process_group_empty=True, exact_owner_gone=True,
        owner_acknowledged=True, self_exit_claimed=False, cleanup_after_failure=False)
    return terminal_bound


def future_runtime_budget(admission_wire, argv, literal):
    terminal_bound = prospective_terminal(admission_wire, argv, literal)
    owner_bytes = len(encode('owner', terminal_bound['owner'], literal))
    exit_bytes = len(encode('exit', terminal_bound, literal))
    fallback_bound = dict(event='sdk_import_diagnostic_exit_receipt_write_failed',
        actual_parent_observed_terminal=terminal_bound, exit_receipt_saved=False,
        reporting_error_type='', diagnostic_passed=False)
    fallback_bytes = len(encode('exit_fallback', fallback_bound, literal))
    total = len(admission_wire) + owner_bytes + exit_bytes + OBSERVER_RESERVE
    require(total <= RUNTIME_CAP and fallback_bytes <= SUMMARY_CAP,
            'Complete original runtime/failure envelope must fit before SDK spawn')
    return dict(schema='minicpm5_formal_literal_runtime_reservation_v1',
                admission_wire_bytes=len(admission_wire), owner_wire_bound=owner_bytes,
                exit_wire_bound=exit_bytes, observer_receipt_reserve=OBSERVER_RESERVE,
                fallback_wire_without_error_name_bytes=fallback_bytes,
                fallback_error_name_serialized_bytes_available=SUMMARY_CAP-fallback_bytes,
                total_runtime_bound=total,
                runtime_cap=RUNTIME_CAP, summary_cap=SUMMARY_CAP, actual_sdk_owner_observed=False)
