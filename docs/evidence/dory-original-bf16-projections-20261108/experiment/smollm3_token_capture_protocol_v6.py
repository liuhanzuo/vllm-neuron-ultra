"""Frozen first22 capture contract; import is stdlib/source only, no launch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'sc6'
PLAN = ROOT / 'research/smollm3-engine-token-capture-inputs-20261107-v6.json'
SCRIPT = ROOT / 'run_smollm3_token_capture_v6.py'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
TARGET = Path('/mnt/nvme/smollm3-cache-ablation-native-v3/target')
ROLES = ('reference', 'nki_write', 'nki_read_write')
SPAWN = 'TRN1_SMOL_TOKEN_CAPTURE_V6'
GROUP_PREFIX = '/user.slice/user-1000.slice/user@1000.service/app.slice/'
CAP = 512 * 2**20


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def performance_guard():
    for name in ('smollm3-cache-ablation-full-v8/status.json', 'lfm3-package-paired-full-v2/status.json'):
        path = ROOT / name
        require(path.is_file() and path.stat().st_size < 32768, 'Missing bounded predecessor status')
        require(json.loads(path.read_text()).get('performance_window') is False, 'Performance work active')


def read(path, cap=8 * 2**20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, 'Invalid metadata: ' + str(path))
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path); s = path.stat()
    return dict(realpath=str(path.resolve()), device=s.st_dev, inode=s.st_ino,
                size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def once(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    require(len(raw) <= 8 * 2**20, 'Metadata output cap')
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def groups():
    result = [dict(id='initial-warm8', prompt='initial', count=1, tokens=8, stream=False, timeout=240)]
    for round_index in range(3):
        for mode, count in (('single', 1), ('concurrent', 4)):
            result.append(dict(id=f'initial-round{round_index}-{mode}', prompt='initial', count=count,
                               tokens=120, stream=False, timeout=240))
    result += [dict(id='initial-stream', prompt='initial', count=4, tokens=120, stream=True, timeout=240),
               dict(id='short-warm8', prompt='short', count=1, tokens=8, stream=False, timeout=600),
               dict(id='short-round0-single', prompt='short', count=1, tokens=512, stream=False, timeout=600)]
    require(sum(x['count'] for x in result) == 22, 'Original request prefix changed')
    return result


def request_body(plan, group):
    ids = plan['prompts'][group['prompt']]
    body = dict(model='smollm3-3b', prompt=ids, max_tokens=group['tokens'], min_tokens=group['tokens'],
                temperature=0.0, ignore_eos=True, return_token_ids=True, echo=False, n=1)
    if group['stream']:
        body.update(stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=False))
    return body


def unit(role, kind):
    require(role in ROLES and kind in ('server', 'client'), 'Invalid owned role/kind')
    return 'trn1-latest-verify-smollm3-engine-token-capture-v6-' + role + '-' + kind


def temp_root(role):
    require(role in ROLES, 'Owned temporary role')
    return DEST / ('t' + str(ROLES.index(role)))


def ipc_layout():
    """Pinned vLLM UUID and Python3.10 AF_UNIX paths, in filesystem bytes."""
    rows = {}
    for role in ROLES:
        base = temp_root(role)
        zmq = base / '00000000-0000-0000-0000-000000000000'
        listener = base / 'pymp-xxxxxxxx' / 'listener-xxxxxxxx'
        lengths = dict(zmq_uuid=len(os.fsencode(zmq)), multiprocessing_listener=len(os.fsencode(listener)))
        require(all(n <= 107 for n in lengths.values()), 'Owned IPC path exceeds Linux107byte limit')
        rows[role] = dict(temporary_root=str(base), rpc_base=str(base), max_path_bytes=lengths)
    return rows


def socket_owned(path):
    path = Path(path)
    return len(os.fsencode(path)) <= 107 and any(temp_root(role) in path.parents for role in ROLES)


def temporary_identity(role):
    path = temp_root(role)
    require(path.parent == DEST and not any(x.is_symlink() for x in (path, *path.parents)), 'Owned temporary path/symlink')
    value = path.stat()
    require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid() and
            stat.S_IMODE(value.st_mode) == 0o700, 'Owned temporary directory/mode/uid')
    return dict(path=str(path), device=value.st_dev, inode=value.st_ino)


def environment(role, kind):
    """Future supervisor must launch exact env; this function launches nothing."""
    require(role in ROLES, 'Role')
    ipc_layout()
    case = read(PLAN)['cases'][role]
    return dict(HOME='/home/ubuntu',USER='ubuntu',LANG='C.UTF-8',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
        PYTHONOPTIMIZE='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPYCACHEPREFIX=str(DEST / role / ('pycache-' + kind)), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        NEURON_RT_VISIBLE_CORES='0-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1', TOKENIZERS_PARALLELISM='false', NEURON_SMOLLM3='1',
        NEURON_SMOLLM3_CACHE_VARIANT=role, NEURON_SMOLLM3_ALLOW_COMPILE='0', NEURON_CPU_SAMPLER_FAST='0',
        VLLM_PLUGINS='neuron,neuron_smollm3', VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONPATH=str(TARGET),
        TMPDIR=str(temp_root(role)), VLLM_RPC_BASE_PATH=str(temp_root(role)), HF_HOME='/mnt/nvme/hf',
        PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',
        NEURON_COMPILED_ARTIFACTS=case['artifact'], BASE_COMPILE_WORK_DIR=str(DEST / role / 'compile-forbidden'),
        NEURON_COMPILE_CACHE_URL=str(DEST / role / 'cache-forbidden'),
        XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')


def properties(name):
    keys = 'MainPID,ControlPID,ActiveState,SubState,InvocationID,ControlGroup,ExecStart,MemoryMax,MemoryHigh,MemorySwapMax,KillMode'
    out = subprocess.check_output(['systemctl', '--user', 'show', name, '--property=' + keys], text=True, timeout=10)
    return dict(x.split('=', 1) for x in out.splitlines() if '=' in x)


def proc(pid):
    d = Path('/proc') / str(pid)
    return dict(pid=pid, start_ticks=d.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
                argv=[x.decode() for x in d.joinpath('cmdline').read_bytes().split(b'\0') if x],
                cgroup=d.joinpath('cgroup').read_text().strip())


def own(role, kind, admission, spawned=False):
    owner = read(DEST / role / (kind + '-owner.json'))
    require(owner['attempt_id'] == admission['attempt_id'] and owner['unit'] == unit(role, kind), 'Owner binding')
    group = GROUP_PREFIX + unit(role, kind) + '.service'
    require(owner['cgroup'] == group, 'Owner cgroup')
    current = proc(os.getpid())
    require(current['cgroup'] == '0::' + group, 'Process left admitted group')
    value = properties(unit(role, kind))
    require(value['InvocationID'] == owner['invocation_id'] and value['MainPID'] == str(owner['pid']), 'Invocation/main PID')
    require(value['ActiveState'] == 'active' and value['SubState'] == 'running' and value['ControlPID'] == '0', 'Unit not running')
    require(value['ControlGroup'] == group and value['KillMode'] == 'control-group', 'Unit ownership policy')
    limits = dict(server=(26, 25, 8), client=(3, 2, 0))[kind]
    require(tuple(int(value[k]) for k in ('MemoryMax', 'MemoryHigh', 'MemorySwapMax')) == tuple(x * 2**30 for x in limits), 'Resource limits')
    require(proc(owner['pid']) == dict(pid=owner['pid'], start_ticks=str(owner['start_ticks']), argv=owner['argv'], cgroup='0::' + group), 'Live owner changed')
    if not spawned:
        require(current['pid'] == owner['pid'] and current['argv'] == owner['argv'], 'Wrong worker')
    return owner


def checked_plan():
    performance_guard(); plan = read(PLAN)
    require(plan['schema'] == 'smollm3_engine_token_capture_plan_v6' and plan['roles'] == list(ROLES), 'Plan scope')
    require(plan['groups'] == groups() and plan['work'] == str(DEST), 'Request sequence/namespace')
    require(plan['ipc_layout'] == ipc_layout(), 'Exact bounded owned IPC layout')
    for name, digest in plan['source_sha256'].items():
        performance_guard()
        path = Path(name)
        require(path.suffix in ('', '.py', '.json', '.md', '.txt', '.log', '.jsonl', '.yaml', '.toml', '.ini', '.typed', '.jinja')
                and path.stat().st_size <= 8 * 2**20, 'Source-only pin type/size')
        require(sha(path) == digest, 'Source changed: ' + name)
    return plan


def wait_admission(role,kind,deadline_seconds=60):
    """Root must publish both records; timeout does not silently retry a unit."""
    began=time.monotonic()
    paths=[DEST/role/'admission.json',DEST/role/(kind+'-owner.json')]
    while not all(path.is_file() and not path.is_symlink() for path in paths):
        performance_guard()
        require(time.monotonic()-began<deadline_seconds,'Root owner/admission publication deadline')
        time.sleep(.1)


def guard(role, kind, spawned=False):
    performance_guard()
    require(__debug__ and sys.flags.optimize == 0 and sys.dont_write_bytecode, 'Use -B, no -O')
    wait_admission(role,kind)
    plan = checked_plan()
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve(), 'Wrong native interpreter')
    for key, expected in environment(role, kind).items():
        require(os.environ.get(key) == expected, 'Environment mismatch: ' + key)
    expected_env=environment(role,kind)
    allowed_extra={SPAWN,'INVOCATION_ID','JOURNAL_STREAM','SYSTEMD_EXEC_PID','MEMORY_PRESSURE_WATCH','MEMORY_PRESSURE_WRITE',
                   'VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD'}
    require(not any(os.environ.get(k) for k in ('LD_PRELOAD', 'PYTHONHOME', 'NXD_CPU_MODE', 'NEURON_CC_FLAGS',
        'NEURON_RT_STOCHASTIC_ROUNDING_EN', 'NEURON_SMOLLM3_NKI_CACHE','XLA_USE_BF16','XLA_DOWNCAST_BF16',
        'NEURON_LOGICAL_NC_CONFIG','NEURON_LFM3','NEURON_LFM3_NKI_WRITER')), 'Ambient override')
    path = DEST / role / 'admission.json'; admission = read(path)
    require(admission['plan_sha256'] == sha(PLAN) and admission['role'] == role, 'Admission plan/role')
    require(admission['passed'] is True and admission['root_serial_lock_held_for_execution'] is True, 'Not admitted')
    require(admission['all_predecessors_drained'] is True and admission['payloads_admitted'] is True, 'Incomplete admission')
    require(len(admission['attempt_id']) == 32 and os.environ.get(SPAWN) == role + ':' + admission['attempt_id'], 'Attempt identity')
    review = Path(admission['root_review']['path'])
    require(review.parent == ROOT / 'research' and sha(review) == admission['root_review']['sha256'], 'Root review pin')
    accepted = read(review)
    require(accepted['reviewer'] == '/root' and accepted['independent_review'] is True and accepted['passed'] is True,
            'Independent root acceptance required')
    require(accepted['plan_sha256'] == sha(PLAN) and accepted['driver_sha256'] == sha(SCRIPT), 'Acceptance source binding')
    supervisor = admission['supervisor']
    require(proc(supervisor['pid']) == supervisor, 'Root supervisor process changed')
    require(str(ROOT / 'serial-enrollment.lock') == admission['serial_lock']['path'] and
            (ROOT / 'serial-enrollment.lock').stat().st_ino == admission['serial_lock']['inode'], 'Lock identity')
    payload_path = DEST / role / 'payload-admission.json'
    require(sha(payload_path) == admission['payload_admission_sha256'], 'Payload admission pin')
    payload = read(payload_path)
    require(payload['passed'] is True and payload['plan_sha256'] == sha(PLAN) and
            payload['role'] == role and payload['attempt_id'] == admission['attempt_id'], 'Payload admission binding')
    require(payload['payload_sha256'] == {name: row['sha256'] for name, row in plan['payloads'].items()},
            'Incomplete authenticated payload set')
    for name, row in plan['payloads'].items():
        require(identity(name) == row['identity'], 'Payload identity changed after admission: ' + name)
    require(admission['canonical_sha256'].keys() == {'continuation-state.json','roadmap.json','best-known-configurations.json'},
            'Canonical snapshot missing')
    for name, digest in admission['canonical_sha256'].items():
        require(sha(ROOT / name) == digest, 'Canonical changed during capture')
    require(temporary_identity(role) == admission['temporary_identity'], 'Temporary directory replaced')
    own(role, kind, admission, spawned)
    lock = ROOT / 'serial-enrollment.lock'
    with lock.open('r+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Exclusive root supervisor lock is not held')
    runtime=Path('/run/user/1000');bus=runtime/'bus'
    require(not runtime.is_symlink() and not bus.is_symlink() and
            runtime.stat().st_uid == bus.stat().st_uid == os.getuid() == 1000 and
            stat.S_ISDIR(runtime.stat().st_mode) and stat.S_IMODE(runtime.stat().st_mode)==0o700 and
            stat.S_ISSOCK(bus.stat().st_mode), 'User bus identity')
    storage()
    return plan, admission


def storage():
    for path, floor in ((ROOT, 28 * 2**30), (Path('/mnt/nvme'), 5 * 2**30)):
        value = os.statvfs(path)
        require(value.f_bavail * value.f_frsize >= floor, 'Storage floor')
    used = 0
    if DEST.exists():
        for path in DEST.rglob('*'):
            require(not path.is_symlink(), 'Owned symlink')
            s = path.stat()
            if stat.S_ISSOCK(s.st_mode):
                require(socket_owned(path), 'Foreign/overlong owned socket')
            elif path.is_file():
                used += max(s.st_size, s.st_blocks * 512)
    require(used <= CAP, 'Owned output cap')
    return used
