"""One EngineCore title transition; immutable spawn identity stays authoritative.

Import is stdlib-only and inert. This is not runtime admission. The allowed raw
representation is checked against each actual before/after observation; it is
not inferred as a universal guarantee from the earlier 168-byte native probe.
"""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import time

TITLE = b'VLLM::EngineCore'
POLICY = 'actual-before-length-exact-title-zero-padding-v1'
IDENTITY = ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')
STABLE = ('pid', 'start_ticks', 'ppid', 'cgroup')
EXE_FIELDS = ('dev', 'ino', 'mode', 'size', 'mtime_ns', 'ctime_ns')
DEADLINE_SECONDS = 15


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(value):
    return {key: copy.deepcopy(value[key]) for key in IDENTITY}


def raw_record(raw):
    require(type(raw) is bytes and 0 < len(raw) <= 65536, 'bounded nonempty raw cmdline')
    return dict(base64=base64.b64encode(raw).decode(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def raw_bytes(record):
    require(type(record) is dict and set(record) == {'base64', 'bytes', 'sha256'}, 'exact raw cmdline record')
    raw = base64.b64decode(record['base64'], validate=True)
    require(record == raw_record(raw), 'canonical exact raw cmdline bytes')
    return raw


def observation(value):
    require(type(value) is dict and set(value) == {'identity', 'exe', 'cmdline'}, 'exact process observation')
    row = value['identity']
    require(type(row) is dict and set(row) == set(IDENTITY), 'original five-field process identity')
    require(type(row['pid']) is int and row['pid'] > 0 and type(row['ppid']) is int and row['ppid'] > 0,
            'positive actual PID and parent')
    require(type(row['start_ticks']) is str and row['start_ticks'].isdigit(), 'exact start tick string')
    require(type(row['argv']) is list and row['argv'] and all(type(v) is str for v in row['argv']), 'actual argv strings')
    require(type(row['cgroup']) is str and row['cgroup'].startswith('0::'), 'exact unified cgroup')
    raw = raw_bytes(value['cmdline'])
    require(raw.endswith(b'\0') and [v.decode() for v in raw.split(b'\0') if v] == row['argv'], 'raw and parsed argv agree')
    exe = value['exe']
    require(type(exe) is dict and set(exe) == {'target', 'file'} and type(exe['target']) is str
            and exe['target'].startswith('/') and type(exe['file']) is dict and set(exe['file']) == set(EXE_FIELDS),
            'independent executable target and file identity')
    require(all(type(v) is int for v in exe['file'].values()) and stat.S_ISREG(exe['file']['mode']), 'ordinary executable identity')
    return value


def expected_after(before):
    """A fail-closed condition, not a promise about unseen native behavior."""
    raw = raw_bytes(observation(before)['cmdline'])
    require(len(raw) > len(TITLE), 'original argument region can hold title and NUL')
    return raw_record(TITLE + b'\0' * (len(raw) - len(TITLE)))


def observed_transition(before, after):
    observation(before); observation(after)
    require(all(before['identity'][key] == after['identity'][key] for key in STABLE), 'title mutation cannot change process identity')
    require(before['exe'] == after['exe'], 'title mutation cannot change executable')
    require(after['cmdline'] == expected_after(before) and after['identity']['argv'] == [TITLE.decode()],
            'only exact title and original-region zero padding allowed')
    require(before['identity']['argv'] != after['identity']['argv'], 'one actual title mutation required')
    return True


def permit_binding(permit):
    require(permit['kind'] == 'server' and permit['spawned'] is True, 'only admitted spawned server may change title')
    spawn = identity(permit)
    require(spawn['argv'][0] == permit['python'] and '--multiprocessing-fork' in spawn['argv']
            and '-c' in spawn['argv'], 'original authenticated CPython spawn argv')
    return dict(permit_sha256=digest(permit), spawn_identity=spawn, run=copy.deepcopy(permit['run']),
                attempt_id=permit['attempt_id'], unit=permit['unit'], invocation_id=permit['invocation_id'],
                runtime_source_sha256=digest(permit['root_runtime_source_sha256']),
                controller=identity(permit['title_controller']))


def source_witness(value):
    require(type(value) is dict and set(value) == {'core_source_sha256', 'utils_source_sha256', 'envs_source_sha256',
            'native_binding', 'call', 'original_callable_verified', 'single_original_call'}, 'exact title source witness')
    for key in ('core_source_sha256', 'utils_source_sha256', 'envs_source_sha256'):
        require(type(value[key]) is str and len(value[key]) == 64 and all(c in '0123456789abcdef' for c in value[key]), 'fixed source digest')
    require(value['call'] == dict(function='EngineCoreProc.run_engine_core', line=697, data_parallel_size=1,
            dp_rank=0, local_dp_rank=0, args=['EngineCore'], kwargs={}, defaults=['', 'VLLM']), 'single original DP/title call')
    require(value['original_callable_verified'] is True and value['single_original_call'] is True,
            'original callable and once-only invocation witness')
    native = value['native_binding']
    require(type(native) is dict and set(native) == {'package', 'version', 'extension', 'file_identity', 'setter_is_original_builtin'},
            'exact native setter binding')
    require(native['package'] == 'setproctitle' and native['version'] == '1.3.7'
            and native['setter_is_original_builtin'] is True, 'fixed original native package and setter')
    return value


def make_before(permit, actual, witness, now):
    binding = permit_binding(permit); observation(actual); source_witness(witness)
    require(actual['identity'] == binding['spawn_identity'], 'title intent starts at immutable spawn identity')
    require(type(now) in (int, float) and now > 0, 'observed monotonic time')
    deadline = min(now + DEADLINE_SECONDS, permit['shared_run_deadline_monotonic'], permit['matrix_deadline_monotonic'])
    require(now < deadline, 'title intent within original deadlines')
    return dict(schema='smollm3_enginecore_title_before_v1', sequence=1, policy=POLICY, binding=binding,
                before=copy.deepcopy(actual), expected_after=expected_after(actual), witness=copy.deepcopy(witness),
                created_monotonic=now, deadline_monotonic=deadline)


def validate_before(permit, before):
    require(type(before) is dict and set(before) == {'schema', 'sequence', 'policy', 'binding', 'before',
            'expected_after', 'witness', 'created_monotonic', 'deadline_monotonic'}, 'exact title-before fields')
    require(before == make_before(permit, before['before'], before['witness'], before['created_monotonic']), 'complete title-before binding')
    return before


def before_ack(permit, before, controller, actual, now):
    validate_before(permit, before); observation(actual)
    require(actual == before['before'], 'controller independently sees original pre-call observation')
    require(identity(controller) == identity(permit['title_controller']), 'original independently admitted controller')
    require(before['created_monotonic'] <= now < before['deadline_monotonic'], 'before ACK deadline')
    return dict(schema='smollm3_enginecore_title_before_ack_v1', before_sha256=digest(before),
                binding=permit_binding(permit), controller=identity(controller), observation=copy.deepcopy(actual),
                acknowledged_monotonic=now, attempt_authorized=True, transition_completed=False)


def validate_before_ack(permit, before, ack):
    require(type(ack) is dict and set(ack) == {'schema', 'before_sha256', 'binding', 'controller', 'observation',
            'acknowledged_monotonic', 'attempt_authorized', 'transition_completed'}, 'exact before ACK fields')
    require(ack == before_ack(permit, before, ack['controller'], ack['observation'], ack['acknowledged_monotonic']),
            'complete before ACK binding')
    return ack


def make_after(permit, before, ack, actual, now, returned_none):
    validate_before_ack(permit, before, ack); observed_transition(before['before'], actual)
    require(returned_none is True, 'exact original native-wrapper None return')
    require(ack['acknowledged_monotonic'] <= now < before['deadline_monotonic'], 'after within fixed title deadline')
    return dict(schema='smollm3_enginecore_title_after_v1', sequence=1, binding=permit_binding(permit),
                before_sha256=digest(before), before_ack_sha256=digest(ack), after=copy.deepcopy(actual),
                observed_monotonic=now, original_return_none=True, original_call_count=1)


def validate_after(permit, before, ack, after):
    require(type(after) is dict and set(after) == {'schema', 'sequence', 'binding', 'before_sha256', 'before_ack_sha256',
            'after', 'observed_monotonic', 'original_return_none', 'original_call_count'}, 'exact title-after fields')
    require(after == make_after(permit, before, ack, after['after'], after['observed_monotonic'], after['original_return_none']),
            'complete after binding and once-only original return')
    return after


def commit_ack(permit, before, ack, after, controller, actual, now):
    validate_after(permit, before, ack, after); observation(actual)
    require(identity(controller) == ack['controller'], 'same original controller identity')
    require(actual == after['after'], 'controller independently observes exact returned title')
    require(after['observed_monotonic'] <= now < before['deadline_monotonic'], 'commit within title deadline')
    return dict(schema='smollm3_enginecore_title_commit_v1', binding=permit_binding(permit), before_sha256=digest(before),
                before_ack_sha256=digest(ack), after_sha256=digest(after), controller=identity(controller),
                observation=copy.deepcopy(actual), committed_monotonic=now, transition_completed=True)


def validate_complete(permit, before, ack, after, committed):
    require(type(committed) is dict and set(committed) == {'schema', 'binding', 'before_sha256', 'before_ack_sha256',
            'after_sha256', 'controller', 'observation', 'committed_monotonic', 'transition_completed'}, 'exact committed fields')
    require(committed == commit_ack(permit, before, ack, after, committed['controller'],
            committed['observation'], committed['committed_monotonic']), 'complete immutable transition chain')
    return dict(before_sha256=digest(before), before_ack_sha256=digest(ack), after_sha256=digest(after),
                commit_sha256=digest(committed))


def check_live(permit, initial, actual, receipts=None, *, pending=False, now=None):
    """Return both identities; only a complete chain allows a projection.

    The controller's bounded pending result does not authorize work or child
    admission. It is an explicit waiting state, not a successful authentication.
    """
    observation(initial); observation(actual)
    require(initial['identity'] == identity(permit), 'independent spawn observation retained')
    require(initial['exe'] == actual['exe'] and all(initial['identity'][k] == actual['identity'][k] for k in STABLE),
            'unchanged executable/PID/start/parent/cgroup')
    if receipts is None:
        require(actual == initial, 'unwitnessed argv mutation')
        return dict(state='original', authenticated=True, spawn_identity=identity(permit), live_observation=copy.deepcopy(actual), transition=None)
    require(type(receipts) is dict and set(receipts) in ({'before', 'before_ack'}, {'before', 'before_ack', 'after', 'commit'}),
            'exact pending or committed receipt set')
    before, ack = receipts['before'], receipts['before_ack']
    validate_before_ack(permit, before, ack)
    require(before['before'] == initial, 'retained initial observation matches transition')
    if 'commit' in receipts:
        closure = validate_complete(permit, before, ack, receipts['after'], receipts['commit'])
        require(actual == receipts['after']['after'], 'committed title/executable must stay exact; no reversion')
        return dict(state='committed', authenticated=True, spawn_identity=identity(permit), live_observation=copy.deepcopy(actual), transition=closure)
    require(pending is True and type(now) in (int, float) and ack['acknowledged_monotonic'] <= now < before['deadline_monotonic'],
            'only controller bounded pending wait')
    if actual != initial:
        observed_transition(initial, actual)
    return dict(state='pending', authenticated=False, spawn_identity=identity(permit), live_observation=copy.deepcopy(actual),
                transition=dict(before_sha256=digest(before), before_ack_sha256=digest(ack)))


def observe_process(pid, expected_executable):
    """Explicit future runtime I/O, never called by source/inert checks."""
    require(type(pid) is int and pid > 0 and type(expected_executable) is str, 'exact observation target')
    root = Path('/proc') / str(pid)
    def small(name, limit):
        with (root / name).open('rb') as stream:
            value = stream.read(limit + 1)
        require(len(value) <= limit, 'bounded kernel identity read')
        return value
    first = small('stat', 8192).decode().rsplit(')', 1)[1].split()
    raw = small('cmdline', 65536)
    cgroup = small('cgroup', 8192).decode().strip()
    target = os.readlink(root / 'exe')
    require(target == expected_executable, 'executable from admitted interpreter, never argv0')
    file = (root / 'exe').stat()
    metadata = dict(dev=file.st_dev, ino=file.st_ino, mode=file.st_mode, size=file.st_size,
                    mtime_ns=file.st_mtime_ns, ctime_ns=file.st_ctime_ns)
    second = small('stat', 8192).decode().rsplit(')', 1)[1].split()
    require(first[19] == second[19] and first[1] == second[1], 'start/parent stable across full kernel observation')
    value = dict(identity=dict(pid=pid, start_ticks=first[19], ppid=int(first[1]),
                 argv=[v.decode() for v in raw.split(b'\0') if v], cgroup=cgroup),
                 exe=dict(target=target, file=metadata), cmdline=raw_record(raw))
    return observation(value)


def run_admitted(*args, **kwargs):
    raise RuntimeError('source component only: complete lifecycle/Ray/spawn/controller/audit integration and root review required')
