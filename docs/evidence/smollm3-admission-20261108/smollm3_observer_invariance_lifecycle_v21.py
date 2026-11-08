"""Source-only reviewed transport library; importing has no runtime effects.

All filesystem/service/proc effects are explicit calls by the future admitted
controller. No native modules. Frozen scientific callables are never modified.
"""
import copy
import smollm3_observer_metadata_store_v1 as metadata_store
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

import smollm3_observer_invariance_protocol_v17 as p
import smollm3_spawn_environment_witness_v13 as witness
import historical_b3_payload_partition_v1 as historical
import smollm3_enginecore_title_transition_v1 as title
import smollm3_enginecore_title_runtime_v10 as title_runtime

ROOT = p.ROOT
DEST = p.DEST
CONTROL = DEST / 'control'
SELF = Path(__file__).resolve()
WRAPPER = ROOT / 'smollm3_observer_invariance_wrapper_v21.py'
CONTROLLER = ROOT / 'smollm3_observer_invariance_controller_v23.py'
AUDIT = ROOT / 'smollm3_observer_invariance_audit_v22.py'
UNIT = 'trn1-latest-verify-smollm3-observer-invariance-v13-supervisor'
LOCK = ROOT / 'serial-enrollment.lock'
HANDOFF = ROOT / 'research/smollm3-sio12-metadata-source-handoff-20261108-v2.json'
TEMPLATE = ROOT / 'research/smollm3-observer-invariance-source-template-20261108-v18.json'
CALLABLE_ACCEPTANCE = ROOT / 'research/smollm3-sio12-metadata-root-source-review-20261108-v2.json'
PREDECESSOR_FLOOR = dict(path=str(ROOT/'research/root-post-smol-sio5-failed-complete-predecessors-20261107-v1.json'),
                        sha256='b4beec61b5f7b0ec34ca65e8a69172ca9023b2ca55c8e62e83b0a936f7395a3e')
GROUP_PREFIX = p.old.GROUP_PREFIX
MIB = 2**20
GIB = 2**30
RESERVE = 8*MIB
KEYS = ('Id,LoadState,ActiveState,SubState,MainPID,ControlPID,Job,InvocationID,'
        'ControlGroup,Result,ExecMainPID,ExecMainCode,ExecMainStatus,ExecStart,'
        'MemoryMax,MemoryHigh,MemorySwapMax,AllowedCPUs,CPUAffinity,Nice,KillMode,'
        'RemainAfterExit,LimitFSIZE,RuntimeMaxUSec,WorkingDirectory,TasksMax')
require = p.require
_SERVICE_OBSERVATIONS = {}


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def ordinary(path, directory=False):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(), 'Absolute ordinary path')
    require(not any(x.is_symlink() for x in (path, *path.parents)), 'No symlink components')
    value = path.stat()
    require(stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode), 'Ordinary file/directory')
    if not directory:
        require(value.st_nlink == 1, 'No shared hardlink evidence')
    return value


def file_stat(path):
    value = Path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                realpath=str(Path(path).resolve()))


def directory_identity(path):
    value=ordinary(path,directory=True)
    require(value.st_uid==1000 and stat.S_IMODE(value.st_mode)==0o700,'Private owned runtime directory')
    return dict(path=str(path),device=value.st_dev,inode=value.st_ino,uid=value.st_uid,mode=stat.S_IMODE(value.st_mode))


def read_bytes(path, cap=8*MIB):
    before = ordinary(path)
    require(before.st_size <= cap, 'Bounded evidence file')
    with Path(path).open('rb') as stream:
        raw = stream.read(cap+1)
    after = ordinary(path)
    require(len(raw) <= cap and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'Evidence changed during bounded read')
    return raw


def decode_bytes(path, raw):
    return metadata_store.decode(DEST,path,raw)


def read(path, cap=8*MIB):
    return decode_bytes(path,read_bytes(path,cap))


def reference(ref):
    path = Path(ref['path'])
    require(path.suffix == '.json' and path.parent == ROOT/'research', 'Root-reviewed metadata reference')
    raw = p.bounded_source(path, ref['sha256'])
    return json.loads(raw)


def template_reference(ref):
    """Authenticate optional size before projecting only path and digest.

    Preserve the literal frozen template path and SHA. Size is provenance,
    not part of the runtime two-field reference; never ignore a supplied size.
    """
    require(type(ref) is dict and set(ref) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
            'Exact template metadata reference')
    require(type(ref['path']) is str and ref['path'] == str(TEMPLATE)
            and type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
            'Literal template path and digest')
    if 'bytes' in ref:
        require(type(ref['bytes']) is int and 0 < ref['bytes'] <= 8*MIB, 'Strict bounded template byte count')
    raw = p.bounded_source(Path(ref['path']), ref['sha256'])
    require('bytes' not in ref or ref['bytes'] == len(raw), 'Authenticated template byte count')
    return dict(path=ref['path'], sha256=ref['sha256'])


def rename_no_replace(source, destination):
    """Linux atomic no-overwrite publication, only used by explicit runtime I/O.

    No hardlink interval and no fallback to replacing an existing evidence file.
    The stdlib foreign-function interface is imported only for this OS call.
    """
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    rename=libc.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1)!=0:
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error),str(destination))


def write_once(path, raw):
    """Durable complete bytes precede atomic visibility; failures stay retained.

    Hidden pending files count in the same run/control budget. No cleanup,
    hardlink or overwrite is performed, even on failed publication or fsync.
    """
    path=Path(path)
    require(path.is_relative_to(DEST)and path!=DEST,'Owned output only')
    ordinary(path.parent,directory=True)
    require(type(raw)is bytes and len(raw)<=8*MIB,'Bounded durable evidence')
    pending=path.parent/('.pending-'+path.name+'-'+str(os.getpid())+'-'+os.urandom(16).hex())
    fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False)as stream:
            require(stream.write(raw)==len(raw),'Complete pending write')
            stream.flush();os.fsync(fd)
    finally:os.close(fd)
    # The complete file already exists durably under its pending name. Readers
    # discover only final names, so no hello/permit/bundle can be half-written.
    parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(parent)
        rename_no_replace(pending,path)
        os.fsync(parent)
    finally:os.close(parent)
    return dict(path=str(path),sha256=digest_bytes(raw),bytes=len(raw),durable=True,exclusive=True)


def record(path, value):
    return metadata_store.record(DEST,path,value,write_once,json_bytes)


def command(argv, timeout=10, cap=128*1024):
    """Bounded subprocess transport, never a shell or unbounded communicate."""
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    select = selectors.DefaultSelector(); select.register(child.stdout, selectors.EVENT_READ)
    raw = bytearray(); deadline = time.monotonic()+timeout
    try:
        while select.get_map():
            require(time.monotonic() < deadline, 'Transport timeout')
            for key, _ in select.select(.1):
                block = os.read(key.fileobj.fileno(), min(8192, cap+1-len(raw)))
                if not block: select.unregister(key.fileobj); continue
                raw.extend(block); require(len(raw) <= cap, 'Transport output cap')
        code = child.wait(timeout=max(.1, deadline-time.monotonic()))
        require(code == 0, 'Transport failure: '+raw[-2048:].decode(errors='replace'))
        return raw.decode()
    finally:
        select.close()
        if child.poll() is None: child.kill()
        child.wait(timeout=5); child.stdout.close()


def props(unit):
    require(re.fullmatch(r'trn1-[A-Za-z0-9_.@:-]+', unit) and not unit.endswith('.service'), 'Safe exact unit')
    return dict(x.split('=', 1) for x in command(['systemctl', '--user', 'show', unit, '--property='+KEYS]).splitlines() if '=' in x)


def installed(value):
    return value['ExecStart'].split(' ; argv[]=', 1)[1].split(' ; ignore_errors=', 1)[0]


def terminal(value):
    return value['MainPID'] == value['ControlPID'] == '0' and not value['Job'] and (
        value['ActiveState'] in ('inactive', 'failed') or (value['ActiveState'], value['SubState']) == ('active', 'exited'))


def process(pid):
    require(type(pid) is int and pid > 0, 'Actual process PID')
    root = Path('/proc')/str(pid)
    first = (root/'stat').read_text().rsplit(')', 1)[1].split()
    argv = [x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x]
    cgroup = (root/'cgroup').read_text().strip()
    second = (root/'stat').read_text().rsplit(')', 1)[1].split()
    require(first[19] == second[19] and first[1] == second[1], 'Process identity raced')
    return dict(pid=pid, start_ticks=first[19], ppid=int(first[1]), argv=argv, cgroup=cgroup)


def process_environment(pid):
    raw = Path('/proc', str(pid), 'environ').read_bytes()
    require(len(raw) <= 128*1024, 'Environment bound')
    return dict(x.decode().split('=', 1) for x in raw.split(b'\0') if x)


def same_process(owner):
    actual = process(owner['pid'])
    require(all(actual[k] == owner[k] for k in ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')), 'Process identity changed')
    return actual


def gone(owner):
    try: actual = process(owner['pid'])
    except FileNotFoundError: return True
    require(owner.get('start_ticks') is not None and actual['start_ticks'] != str(owner['start_ticks']), 'Prior process remains or start unknown')
    return True


def owned_live_or_retained_exit(owner,initial,refresh,timeout=2):
    """Bounded exit transition: empty argv is never a replacement identity.

    Already authenticated PID/start/PPID/cgroup and invocation must still match.
    A nonempty different argv fails immediately. Only a matching empty-argv
    exit transition may wait for two identical retained terminal samples.
    """
    deadline=time.monotonic()+timeout;value=initial;previous_terminal=None
    while True:
        require(value['InvocationID']==owner['invocation_id'],'Exit transition invocation changed')
        if value['MainPID']!='0':
            require(value['MainPID']==str(owner['pid']),'Exit transition PID changed')
            try:actual=process(owner['pid'])
            except FileNotFoundError:actual=None
            if actual is not None:
                require(all(actual[k]==owner[k]for k in('pid','start_ticks','ppid','cgroup')),'Exit transition identity changed')
                if actual['argv']:
                    require(actual['argv']==owner['argv'],'Foreign nonempty argv')
                    return value
            previous_terminal=None
        else:
            require(terminal(value)and value['ExecMainPID']==str(owner['pid']),'Exact retained exit PID')
            try:gone(owner)
            except RuntimeError:previous_terminal=None
            else:
                if previous_terminal==value:return value
                previous_terminal=value
        require(time.monotonic()<deadline,'Bounded exiting process transition expired')
        time.sleep(.02);value=refresh()


def cgroup_inventory(group, require_empty=False):
    require(group.startswith(GROUP_PREFIX+'trn1-') and '..' not in Path(group).parts, 'Owned cgroup')
    root = Path('/sys/fs/cgroup')/group.lstrip('/')
    if not root.exists(): return dict(group=group, absent=True, pids=[], directories=[])
    todo = [root]; directories = []; pids = set()
    while todo:
        directory = todo.pop(); require(not directory.is_symlink(), 'Cgroup symlink')
        directories.append(str(directory)); require(len(directories) <= 256, 'Cgroup depth/count')
        pids.update(int(x) for x in (directory/'cgroup.procs').read_text().split())
        with os.scandir(directory) as entries:
            for entry in entries:
                require(not entry.is_symlink(), 'Cgroup child symlink')
                if entry.is_dir(follow_symlinks=False): todo.append(Path(entry.path))
    require(not require_empty or not pids, 'Owned descendants remain')
    return dict(group=group, absent=False, pids=sorted(pids), directories=sorted(directories))


def lock_proof(owner, lock_identity):
    """Read the kernel lock owner; a busy-lock flag is not ownership evidence."""
    same_process(owner)
    value = ordinary(LOCK)
    require(lock_identity == dict(path=str(LOCK), device=value.st_dev, inode=value.st_ino), 'Lock path/inode binding')
    rows = []
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']: continue
        major, minor, inode = fields[5].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == (os.major(value.st_dev), os.minor(value.st_dev), value.st_ino):
            rows.append(fields)
    require(len(rows) == 1 and rows[0][4] == str(owner['pid']), 'Serial flock held by exact external owner')
    return dict(owner_pid=owner['pid'], lock=lock_identity, kernel_record=rows[0])


def budget_inventory(entries, run_id, root_free, nvme_free, reserve=True):
    """Pure allocation accounting, including every file, directory and IPC root."""
    row = p.run_row(run_id); total = control = current = 0
    run_totals = {r['run_id']: 0 for r in p.matrix()}
    for item in entries:
        path = Path(item['path']); require(path.is_relative_to(DEST), 'Foreign output root')
        relative = path.relative_to(DEST); parts = relative.parts
        require(not item['symlink'], 'Output symlink')
        require(item['kind'] in ('file', 'dir', 'socket'), 'Output special file')
        owner_run = None
        if parts and re.fullmatch(r'[rt][0-9]{2}', parts[0]):
            owner_run = 'r'+parts[0][1:]; p.run_row(owner_run)
        else:
            require(not parts or parts[0] == 'control', 'Foreign root output')
        amount = max(item['size'], item['blocks']*512)
        require(amount >= 0, 'Negative allocation')
        if item['kind'] == 'file':
            require(item['nlink'] == 1 and item['size'] <= 64*MIB, 'File/shared-link cap')
            if path.suffix == '.log': require(item['size'] <= 16*MIB, 'Inclusive log cap')
        if item['kind'] == 'socket':
            require(owner_run is not None and parts[0].startswith('t') and len(os.fsencode(path)) <= 107, 'Owned short IPC socket')
        total += amount
        if owner_run is None: control += amount
        else: run_totals[owner_run] += amount
    for run in p.matrix(): require(run_totals[run['run_id']] <= run['run_cap_bytes'], 'Inclusive per-run cap')
    require(total <= p.OUTPUT_CAP-(RESERVE if reserve else 0), 'Inclusive whole2GiB cap')
    require(control <= p.CONTROL_CAP-(RESERVE if reserve else 0), 'Control128MiB included, terminal reserve')
    require(root_free >= 28*GIB and nvme_free >= 5*GIB, 'Disk floors')
    current = run_totals[run_id]
    return dict(recursive=True, no_symlinks=True, foreign_files=[], total_allocated_bytes=total,
                run_allocated_bytes=current, aggregate_control_allocated_bytes=control,
                root_free_bytes=root_free, nvme_free_bytes=nvme_free, root=str(DEST), run_id=run_id,
                run_scope_paths=[str(p.directory(run_id)), str(p.temporary(run_id))], run_totals=run_totals)


def pending_final(path):
    path=Path(path)
    match=re.fullmatch(r'\.pending-([A-Za-z0-9_-]+\.(?:json|raw))-([1-9][0-9]*)-([0-9a-f]{32})',path.name)
    return None if match is None else path.with_name(match.group(1))


def output_entries():
    """Full fresh walk; never skip a vanished entry or return a partial budget."""
    entries=[];pending=[DEST]
    while pending:
        path=pending.pop()
        try:value=path.lstat()
        except FileNotFoundError:
            target=pending_final(path)
            if target is None:raise
            # Only the exact pending-to-final atomic transition can be retried.
            actual=ordinary(target)
            require(actual.st_uid==1000 and actual.st_nlink==1,'Published pending target ownership')
            raise PendingPublicationRace(str(path))
        kind='dir'if stat.S_ISDIR(value.st_mode)else'file'if stat.S_ISREG(value.st_mode)else'socket'if stat.S_ISSOCK(value.st_mode)else'other'
        entries.append(dict(path=str(path),kind=kind,size=value.st_size,blocks=value.st_blocks,
                            nlink=value.st_nlink,symlink=stat.S_ISLNK(value.st_mode)))
        require(value.st_uid==1000 and len(entries)<=32768,'Output owner/count')
        if kind=='dir':
            with os.scandir(path)as children:pending.extend(Path(x.path)for x in children)
    return entries


class PendingPublicationRace(RuntimeError):pass


def monitor(run_id, reserve=True):
    p.performance_guard();deadline=time.monotonic()+2
    for attempt in range(8):
        try:entries=output_entries();break
        except PendingPublicationRace:
            require(attempt<7 and time.monotonic()<deadline,'Bounded atomic-publication inventory retry exhausted')
            time.sleep(.005)
    def free(path):
        value=os.statvfs(path);return value.f_bavail*value.f_frsize
    return budget_inventory(entries,run_id,free(ROOT),free('/mnt/nvme'),reserve)


class DeadlineExceeded(TimeoutError):
    """The independent audit limit expired, including a long parser/export call."""


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded('Independent audit deadline expired')


@contextmanager
def bounded_phase(deadline):
    """Linux main-thread wall timer; import and mocks never install a timer.

    Restore the previous handler before recovery. No nested/preexisting timer
    is silently replaced. The deadline exception must escape inner audit code.
    """
    check_deadline(deadline)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'No existing audit timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise DeadlineExceeded('Independent audit exceeded1800s')
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.000001, deadline-time.monotonic()))
        yield
        check_deadline(deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def source_closure(source_plan, runtime_sources, aliases, hash_bytes=True, deadline=None):
    """Only four pre-existing exact metadata links may resolve through aliases."""
    pins = dict(source_plan['source_sha256'])
    for path, digest in runtime_sources.items():
        require(path not in pins or pins[path] == digest, 'Conflicting source pin')
        pins[path] = digest
    stats = {}
    for name, digest in pins.items():
        check_deadline(deadline)
        path = Path(name)
        if name in aliases:
            row = aliases[name]
            require(path.is_symlink() and str(path.resolve()) == row['target'] and row['sha256'] == digest, 'Exact original metadata alias')
            target = Path(row['target']); ordinary(target)
        else:
            require(not path.is_symlink(), 'Unreviewed source alias'); target = path
        require(target.suffix not in ('.pt', '.raw', '.safetensors', '.neff', '.so', '.bin', '.zlib', '.whl', '.pb'), 'Source-only closure')
        if hash_bytes: p.bounded_source(target, digest)
        stats[name] = file_stat(path)
        check_deadline(deadline)
    return stats


def verify_stat_closure(expected):
    require(all(file_stat(path) == wanted for path, wanted in expected.items()), 'Admitted source/payload changed')


def validate_spec(value, plan, base):
    p.validate_template(plan, base)
    require(value['schema'] == 'smollm3_observer_invariance_execution_spec_v10', 'Exact lifecycle schema')
    require(type(value['source_template']) is dict and set(value['source_template']) == {'path','sha256'}, 'Exact runtime template reference')
    require(value['source_template']['path'] == str(TEMPLATE), 'Fresh scientific template path')
    require(value['callable_handoff']['path'] == str(HANDOFF), 'Fresh callable handoff path')
    require(value['callable_root_acceptance']['path'] == str(CALLABLE_ACCEPTANCE), 'Fresh root source acceptance path')
    require(value['work'] == str(DEST) and value['matrix'] == p.matrix(), 'Fresh sio12 fixed matrix')
    require(value['runtime_admitted'] is True and value['root_review_required'] is True, 'Root-bound admission only')
    require(value['supervisor_unit'] == UNIT and value['planned_units'] == [p.unit(r['run_id'], k) for r in p.matrix() for k in ('server', 'client')]+[UNIT], 'All24 plus supervisor')
    require(value['wall_time'] == plan['wall_time'] and value['storage'] == plan['storage'], 'Unchanged budgets')
    nonces = value['attempt_ids']
    require(set(nonces) == {r['run_id'] for r in p.matrix()} and len(set(nonces.values())) == 12 and all(re.fullmatch('[0-9a-f]{32}', x) and x != '0'*32 for x in nonces.values()), 'Twelve fresh nonces')
    require(set(value['canonical_sha256']) == {'continuation-state.json', 'roadmap.json', 'best-known-configurations.json'}, 'Complete canonical snapshot')
    historical.pin_shape(value['runtime_source_sha256']); historical.pin_shape(value['historical_payload_sha256'])
    require(value['historical_payload_sha256'] == {row['path']:row['sha256'] for row in historical.WHITELIST}, 'Exact separately retained historical payload pins')
    require(not any(Path(name).suffix in historical.FORBIDDEN for name in value['runtime_source_sha256']), 'Payloads never enter runtime source closure')
    for path in (SELF, WRAPPER, CONTROLLER, AUDIT, Path(historical.__file__)): require(str(path) in value['runtime_source_sha256'], 'Full runtime source closure')
    require(value['final_predecessor_catalog']['path'] != base.get('successor_predecessor_catalog', {}).get('path'), 'Future final catalog, never old sc7 binding')
    require(value['automatic_successor_launch'] is False and value['compile_allowed'] is False, 'No successor/compile')
    require(set(value) == {'schema','source_template','callable_handoff','callable_root_acceptance','work','matrix',
        'runtime_admitted','root_review_required','supervisor_unit','planned_units','wall_time','storage','attempt_ids',
        'canonical_sha256','runtime_source_sha256','historical_payload_sha256','final_predecessor_catalog','automatic_successor_launch','compile_allowed'}, 'Exact execution specification fields')
    return value


def load_spec(path, wanted):
    value = reference(dict(path=str(path), sha256=wanted))
    plan = reference(value['source_template']); base = p.load_base()
    validate_spec(value, plan, base)
    partition = historical.partition(reference(value['final_predecessor_catalog']))
    require(value['historical_payload_sha256'] == partition['historical_payload_sha256'], 'Complete authenticated historical payload provenance')
    require(all(value['runtime_source_sha256'].get(k) == v for key in ('source_sha256','provenance_sha256')
                for k,v in partition[key].items()), 'Every authenticated current metadata/source pin retained')
    handoff = reference(value['callable_handoff'])
    require(template_reference(handoff['source_template'])==value['source_template'] and handoff['source_frozen'] is True,'Frozen sio12 handoff/template binding')
    require(all(value['runtime_source_sha256'].get(k)==v
                for k,v in handoff['source_sha256'].items()),'Every frozen sio12 source pin')
    # Use the same exact reviewed metadata aliases as runtime admission. The
    # ordinary reader stays strict; alias targets and bytes are authenticated.
    source_closure({'source_sha256': handoff['source_sha256']}, {},
                   handoff['original_metadata_symlink_resolutions'], hash_bytes=True)
    accepted=reference(value['callable_root_acceptance'])
    require(accepted['reviewer']=='/root' and accepted['passed'] is True and accepted['independent_review'] is True
            and accepted['source_handoff']==value['callable_handoff'] and accepted['blockers']==[],'Exact independent sio12 source acceptance')
    return value, plan, base, handoff


def complete_predecessor_extension(final,prior):
    """409/39 is the preserved floor, never a ceiling for later work."""
    require(len(prior['rows'])==409 and len(prior['processes'])==39,'Exact completed sio5 history floor')
    require(final['reviewer']=='/root' and final['root_finalized']is True and final['all_planned_launches_accounted_for']is True,'Root complete current catalog')
    require(final['complete_count']==len(final['rows']) and len(final['rows'])>=409 and len(final['processes'])>=39,'Complete floor plus later count')
    require(final['rows'][:409]==prior['rows'] and final['processes'][:39]==prior['processes'],'No predecessor deletion/reordering')
    dynamic={'schema','rows','processes','complete_count','source_sha256','root_finalization_observed_at','appended_from','appended_terminal_audit'}
    require(all(final.get(k)==v for k,v in prior.items()if k not in dynamic),'All prior failure metadata preserved')
    require(all(final['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'Every prior evidence/source pin preserved')
    return final


def resource_check(value, kind):
    limits = p.resource(kind)
    require(tuple(int(value[k]) for k in ('MemoryMax', 'MemoryHigh', 'MemorySwapMax')) ==
            tuple(limits[k] for k in ('memory_max_bytes', 'memory_high_bytes', 'swap_max_bytes')), 'Memory limits')
    require(value['AllowedCPUs'] == value['CPUAffinity'] == '0-7' and value['Nice'] == '19' and
            value['KillMode'] == 'control-group' and value['RemainAfterExit'] == 'yes' and
            value['LimitFSIZE'] == str(64*MIB) and value['RuntimeMaxUSec'] == '2h' and
            value['WorkingDirectory'] == str(ROOT) and value['TasksMax'] == '2048', 'Service resource contract')


def worker_command(spec_path, spec_sha, review_path, review_sha, base, run_id, kind, nonce):
    env = p.environment(base, run_id, kind, nonce)
    argv = [p.PYTHON, '-B', '-u', str(WRAPPER), '--kind', kind, '--run-id', run_id,
            '--spec', str(spec_path), '--spec-sha', spec_sha, '--review', str(review_path), '--review-sha', review_sha]
    execargv = ['/usr/bin/env', '-i', *[k+'='+v for k,v in env.items()], *argv]
    policy = p.resource(kind)
    options = ['MemoryMax='+str(policy['memory_max_bytes']), 'MemoryHigh='+str(policy['memory_high_bytes']),
               'MemorySwapMax='+str(policy['swap_max_bytes']), 'AllowedCPUs=0-7', 'CPUAffinity=0-7', 'Nice=19',
               'TasksMax=2048', 'RuntimeMaxSec=7200', 'TimeoutStopSec=60', 'KillMode=control-group',
               'SendSIGKILL=yes', 'RemainAfterExit=yes', 'LimitFSIZE='+str(64*MIB), 'UMask=0077',
               'WorkingDirectory='+str(ROOT), 'StandardOutput=append:'+str(p.directory(run_id)/(kind+'.log')), 'StandardError=inherit']
    command_argv = ['systemd-run', '--user', '--quiet', '--unit='+p.unit(run_id, kind)]
    for option in options: command_argv += ['-p', option]
    return dict(argv=argv, execargv=execargv, command=command_argv+execargv, environment=env,
                unit=p.unit(run_id, kind), run_id=run_id, kind=kind, attempt_id=nonce)


def authenticate_service(intent, claim=None):
    value = props(intent['unit'])
    require(value['LoadState'] == 'loaded' and value['Id'] == intent['unit']+'.service', 'Loaded exact service')
    require(installed(value) == ' '.join(intent['execargv']), 'Installed argv differs')
    require(re.fullmatch('[0-9a-f]{32}', value['InvocationID']) and value['InvocationID'] != '0'*32, 'Actual invocation ID')
    require(claim is None or value['InvocationID'] == claim, 'Invocation replaced')
    require(not value['ControlGroup'] or value['ControlGroup'] == GROUP_PREFIX+intent['unit']+'.service', 'Unexpected cgroup')
    resource_check(value, intent['kind'])
    return value


def service_binding(intent,value):
    """Only actual captured properties, never fabricated PID or invocation."""
    require(value['Id']==intent['unit']+'.service'and value['LoadState']=='loaded','Claim exact loaded unit')
    invocation=value['InvocationID'];pid=value['ExecMainPID']
    require(re.fullmatch('[0-9a-f]{32}',invocation)and invocation!='0'*32,'Claim actual invocation')
    require(type(pid)is str and pid.isdigit()and int(pid)>0,'Claim actual ExecMainPID')
    require(value['MainPID']in('0',pid),'Claim actual main PID differs')
    require(installed(value)==' '.join(intent['execargv']),'Claim actual installed argv')
    return dict(schema='smollm3_observer_invariance_service_binding_v3',unit=intent['unit'],
        invocation_id=invocation,exec_main_pid=pid,installed_argv=installed(value),intent_sha256=p.digest(intent))


def verify_binding(intent,value,binding):
    require(service_binding(intent,value)==binding,'Durably claimed service replaced')
    return value


def bind_service(path,intent,value,owner=None,claim=None):
    """Persist first authenticated invocation before any stop/kill/ACK effect."""
    binding=service_binding(intent,value)
    require(claim is None or binding['invocation_id']==claim,'Previously observed invocation changed')
    if owner is not None:
        require(binding['invocation_id']==owner['invocation_id']and binding['exec_main_pid']==str(owner['pid']),
                'Claim disagrees with actual owner')
    key=str(path)
    require(key not in _SERVICE_OBSERVATIONS or _SERVICE_OBSERVATIONS[key]==binding,'Previously observed service replaced after write failure')
    _SERVICE_OBSERVATIONS[key]=copy.deepcopy(binding)
    if Path(path).exists():
        saved=read(path);require(saved==binding,'Existing durable invocation binding changed')
    else:
        require(not list(Path(path).parent.glob('.pending-'+Path(path).name+'-*')),'Incomplete prior binding requires explicit root recovery')
        record(path,binding)
    return binding


def verify_retained_binding(intent,owner,binding):
    expected=dict(schema='smollm3_observer_invariance_service_binding_v3',unit=intent['unit'],
        invocation_id=owner['invocation_id'],exec_main_pid=str(owner['pid']),
        installed_argv=' '.join(intent['execargv']),intent_sha256=p.digest(intent))
    require(binding==expected,'Retained actual owner/command/service binding')
    return binding


def parent_chain_checked(actual, parent, run_id, title_transaction=None):
    chain = []; checks = []; current = actual
    for _ in range(16):
        if current['pid'] == parent['pid']:
            require(current == {k:parent[k] for k in current}, 'Authenticated root parent changed')
            return dict(spawn_chain=chain, live_checks=checks)
        require(current['cgroup'] == parent['cgroup'], 'Spawn escaped service')
        path = p.directory(run_id)/('permit-server-'+str(current['pid'])+'.json')
        if path.exists():
            permit = read(path)['permit']
            transaction = title_transaction if title_transaction is not None and title_transaction.permit['pid']==current['pid'] else None
            checked = title_runtime.check_producer(permit, transaction=transaction)
            require(checked['authenticated'] and checked['live_observation']['identity']==current,
                    'Every exact actual ancestor and committed title, or own in-flight transaction')
            checks.append(checked); chain.append(checked['spawn_identity'])
        else:
            require(current['argv'] != [title.TITLE.decode()], 'Unknown titled process has no spawn admission')
            chain.append(current)
        current = process(current['ppid'])
    raise RuntimeError('Spawn parent chain bound')


def parent_chain(actual, parent, run_id):
    return parent_chain_checked(actual,parent,run_id)['spawn_chain']


class PendingSpawnWitness(RuntimeError):pass


def authenticate_ray_state(permit,state):
    """Retained source receipt binding; recursion follows strictly shorter lineage."""
    ray=p.environment_policy.ray;ray.validate_state(state)
    root=p.directory(permit['run']['run_id'])
    if state['kind']=='local_import_zero':
        before=read(root/('ray-before-'+str(permit['pid'])+'.json'))
        after=read(root/('ray-after-'+str(permit['pid'])+'.json'))
        require(ray.validate_receipts(before,after,permit)==state,'Actual local Ray source receipt')
    elif state['kind']=='inherited_zero':
        require(ray.inherited_state(permit['ray_inheritance'],permit,{ray.KEY:'0'})==state,'Retained inherited Ray state')
    else:
        require(not permit['spawned'],'Spawn cannot claim absent Ray state');return
    if permit['spawned']:
        proof=permit['ray_inheritance'];edge=permit['environment_lineage']['edges'][-1]
        parent=read(root/('permit-server-'+str(edge['parent']['pid'])+'.json'))['permit']
        before=read(root/('spawn-before-'+str(edge['parent']['pid'])+'-'+str(edge['sequence'])+'.json'))
        after=read(root/('spawn-after-'+str(edge['parent']['pid'])+'-'+str(edge['sequence'])+'.json'))
        require(witness.validate_dispatch(before,after,parent,permit)==edge and
                p.environment_policy.extend_lineage(parent['environment_lineage'],edge)==permit['environment_lineage'],
                'Ray inheritance exact authenticated shorter lineage')
        require(proof['parent_state']==before['ray_source_state'] and proof['parent_observation']==before['ray_observation'] and
                proof['child_observation']==ray.ZERO,'Ray inheritance actual saved parent/child values')
        authenticate_ray_state(parent,proof['parent_state'])


def spawn_lineage(hello,intent,parent,admitted):
    """Root checks one actual immediate-parent dispatch, never inferred depth."""
    actual=hello['process'];run_id=intent['run_id'];root=p.directory(run_id)
    parent_key=(run_id,'server',actual['ppid'])
    if parent_key not in admitted:raise PendingSpawnWitness('Immediate parent not yet admitted')
    ack=admitted[parent_key];parent_permit=ack['permit']
    require(title_runtime.check_producer(parent_permit)['authenticated'], 'Exact original or committed spawning parent')
    require(parent_permit['attempt_id']==intent['attempt_id'] and parent_permit['kind']=='server','Same admitted parent run')
    matching=[]
    for path in sorted(root.glob('spawn-after-'+str(actual['ppid'])+'-*.json')):
        row=read(path)
        if row.get('child',{}).get('pid')==actual['pid']:matching.append((path,row))
    require(len(matching)<=1,'Unique dispatched child')
    if not matching:raise PendingSpawnWitness('Actual spawn after witness pending')
    path,after=matching[0];sequence=after['sequence']
    require(type(sequence)is int and 1<=sequence<=7 and path.name=='spawn-after-'+str(actual['ppid'])+'-'+str(sequence)+'.json','Exact spawn after filename')
    before=read(root/('spawn-before-'+str(actual['ppid'])+'-'+str(sequence)+'.json'))
    edge=witness.validate_dispatch(before,after,parent_permit,actual)
    authenticate_ray_state(parent_permit,before['ray_source_state'])
    require(hello['ray_clean_observation']==p.environment_policy.ray.ZERO,'Child earliest actual Ray value equals dispatched zero')
    require(not (root/('spawn-failure-'+str(actual['ppid'])+'-'+str(sequence)+'.json')).exists(),'Failed dispatch never admitted')
    # Parent serve-phase receipt must precede the spawn witness logically and
    # carry the same permit/lineage identity; root verifies retained bytes.
    served=read(root/('environment-serve-'+str(actual['ppid'])+'.json'))
    require(served['pid']==parent_permit['pid'] and served['start_ticks']==parent_permit['start_ticks'] and
            served['attempt_id']==intent['attempt_id'] and served['run_id']==run_id and
            served['ld_source_state']==p.environment_policy.ld_state(parent_permit['environment_lineage'],'serve') and
            served['exact_values_checked']is True and served['values_recorded']is False,'Actual parent serve receipt')
    lineage=p.environment_policy.extend_lineage(parent_permit['environment_lineage'],edge)
    chain=parent_chain(actual,parent,run_id)
    expected=[x['child']for x in reversed(lineage['edges'])]
    require(chain==expected and lineage['root']==p.environment_policy.identity(parent),'Whole actual admitted ancestry chain')
    proof=dict(schema='smollm3_ray_inheritance_v1',edge=edge,run=hello['run'],attempt_id=intent['attempt_id'],
               child=actual,parent_state=before['ray_source_state'],parent_observation=before['ray_observation'],
               child_observation=hello['ray_clean_observation'])
    return lineage,proof


def authenticate_hello(hello, intent, parent, value, base, admitted=None):
    run = p.run_row(intent['run_id'])
    require(hello['schema'] == 'smollm3_observer_invariance_hello_v6' and hello['run'] == run and
            hello['kind'] == intent['kind'] and hello['attempt_id'] == intent['attempt_id'], 'Hello exact run/nonce/mode')
    require(type(hello['spawned'])is bool,'Exact hello spawn flag')
    require(hello['wrapper_source'] == str(WRAPPER) and hello['wrapper_sha256'] == digest_bytes(p.bounded_source(WRAPPER)), 'Hello wrapper source')
    actual = same_process(hello['process'])
    require(actual['cgroup'] == '0::'+GROUP_PREFIX+intent['unit']+'.service', 'Actual hello cgroup')
    require(value['InvocationID'] == parent['invocation_id'] and value['MainPID'] == str(parent['pid']), 'Actual systemd invocation/parent')
    if hello['spawned']:
        require(intent['kind'] == 'server' and actual['pid'] != parent['pid'], 'Server spawn only')
        require(actual['argv'][0] == p.PYTHON and '-c' in actual['argv'] and '--multiprocessing-fork' in actual['argv'] and
                'from multiprocessing.spawn import spawn_main' in actual['argv'][actual['argv'].index('-c')+1], 'Exact CPython spawn transport')
        require(admitted is not None,'Actual admitted parent registry required')
        lineage,ray_inheritance=spawn_lineage(hello,intent,parent,admitted);chain=parent_chain(actual,parent,intent['run_id'])
    else:
        require(actual == {k: parent[k] for k in actual} and actual['argv'] == intent['argv'], 'Exact parent/client argv')
        chain=[];lineage=p.environment_policy.root_lineage(actual)if intent['kind']=='server'else None;ray_inheritance=None
    env=process_environment(actual['pid']);expected=p.environment(base,run['run_id'],intent['kind'],intent['attempt_id'])
    require(all(env.get(k)==v for k,v in expected.items()),'Actual process launch environment')
    system=p.environment_policy.capture_system(env,expected,dict(actual,invocation_id=value['InvocationID']))
    require(hello['ray_clean_observation']==p.environment_policy.ray.observe(env),'Earliest self observation and actual initial process Ray value')
    ray_state=None if intent['kind']=='client'else p.environment_policy.ray.absent_state()
    if hello['spawned']:
        temp=dict(actual,kind='server',spawned=True,run=run,attempt_id=intent['attempt_id'],environment_lineage=lineage)
        ray_state=p.environment_policy.ray.inherited_state(ray_inheritance,temp,env)
    p.environment_policy.check(env,expected,intent['kind'],'clean',hello['spawned'],system,lineage,ray_state)
    spawn_observation = title.observe_process(actual['pid'], str(Path(p.PYTHON).resolve()))
    require(spawn_observation['identity']==actual, 'Full executable/raw identity bound to actual initial admission')
    return dict(actual,spawn_observation=spawn_observation,unit=intent['unit'],invocation_id=value['InvocationID'],spawned=hello['spawned'],
                run=run,mode=run['mode'],attempt_id=intent['attempt_id'],parent_chain=chain,environment_lineage=lineage,
                ray_inheritance=ray_inheritance,ray_clean_observation=hello['ray_clean_observation'])


def permit_for(plan, base, run_id, kind, producer, admission, runtime_sources):
    """Only called after actual hello authentication, never invent process IDs."""
    row = p.run_row(run_id)
    require(producer['run']==row and producer['mode']==row['mode'] and producer['unit']==p.unit(run_id,kind), 'Authenticated producer for this permit')
    value = dict(schema='smollm3_observer_invariance_runtime_permit_v6', run=row, kind=kind,
                 spawned=producer['spawned'], source_plan_sha256=p.digest(plan), base_plan_sha256=p.BASE_SHA,
                 source_plan_immutable=True, uid=1000, python=p.PYTHON,
                 resources=p.resource(kind), environment=p.environment(base,run_id,kind,producer['attempt_id']),
                 output_root=str(DEST), run_directory=str(p.directory(run_id)), temporary_root=str(p.temporary(run_id)),
                 wrapper_source=str(WRAPPER), root_runtime_source_sha256=runtime_sources, root_runtime_sources_verified=True,
                 run_started_monotonic=admission['run_started_monotonic'], shared_run_deadline_monotonic=admission['run_started_monotonic']+7200,
                 matrix_started_monotonic=admission['matrix_started_monotonic'], matrix_deadline_monotonic=admission['matrix_started_monotonic']+86400,
                 canonical_unchanged=True, automatic_successor_launch=False)
    value.update({k:producer[k] for k in ('pid','ppid','argv','cgroup','start_ticks','unit','invocation_id','attempt_id','parent_chain','environment_lineage','ray_inheritance','ray_clean_observation','spawn_observation')})
    value['title_controller']=title.identity(admission['supervisor'])
    for key in ('independent_root_source_review','all_source_pins_verified','all_payloads_verified','all_final_predecessors_drained',
                'serial_lock_proven','live_service_identity_verified','recursive_output_monitor_active','previous_run_fully_drained',
                'fresh_run_namespace','actual_tiny_serialization_verified'):
        require(admission['proofs'].get(key) is True, 'Missing actual external proof: '+key)
        value[key] = True
    if kind == 'client':
        require(admission.get('live_server') is not None, 'Paired live server receipt')
        value['actual_live_server_verified'] = True
    return value


def verify_live_permit(permit, intent, admission, spec, title_transaction=None):
    """A fixed lease is returned only after current kernel/service observations."""
    p.performance_guard()
    producer_check=title_runtime.check_producer(permit, transaction=title_transaction)
    require(producer_check['authenticated'], 'Every ordinary or witnessed own-title producer check')
    value = authenticate_service(intent, permit['invocation_id'])
    require(value['ActiveState'] == 'active' and value['SubState'] == 'running', 'Current service must be live')
    require(value['MainPID'] == str(admission['service_parent']['pid']), 'Actual service parent PID')
    same_process(admission['service_parent'])
    if permit['kind']=='server':p.environment_policy.validate_lineage(permit['environment_lineage'],permit,permit['spawned'])
    if permit['spawned']:
        ray=p.environment_policy.ray
        authenticate_ray_state(permit,ray.inherited_state(permit['ray_inheritance'],permit,{ray.KEY:'0'}))
        chain_check=parent_chain_checked(process(permit['pid']),admission['service_parent'],permit['run']['run_id'],title_transaction)
        require(chain_check['spawn_chain']==permit['parent_chain'],'Every exact live parent-chain edge')
        for edge in permit['environment_lineage']['edges']:
            parent_bundle=read(p.directory(permit['run']['run_id'])/('permit-server-'+str(edge['parent']['pid'])+'.json'))
            before=read(p.directory(permit['run']['run_id'])/('spawn-before-'+str(edge['parent']['pid'])+'-'+str(edge['sequence'])+'.json'))
            after=read(p.directory(permit['run']['run_id'])/('spawn-after-'+str(edge['parent']['pid'])+'-'+str(edge['sequence'])+'.json'))
            require(witness.validate_dispatch(before,after,parent_bundle['permit'],edge['child'])==edge,'Every immutable live lineage dispatch')
    lock_proof(admission['external_owner'], admission['lock'])
    same_process(admission['supervisor'])
    require(admission['directory_identities']=={path:directory_identity(path) for path in admission['directory_identities']},'Runtime roots replaced')
    for path,wanted in spec['runtime_source_sha256'].items():
        require(file_stat(path) == admission['source_stats'][path], 'Runtime wrapper/controller source changed')
    verify_stat_closure(admission['source_stats']); verify_stat_closure(admission['payload_stats'])
    for name, wanted in spec['canonical_sha256'].items():
        require(digest_bytes(p.bounded_source(ROOT/name)) == wanted, 'Canonical changed')
    if permit['kind'] == 'client':
        server = admission['live_server']; sv = authenticate_service(server['intent'], server['owner']['invocation_id'])
        require(sv['MainPID'] == str(server['owner']['pid']) and not terminal(sv), 'Exact paired server live')
        same_process(server['owner'])
    require(time.monotonic() < min(permit['shared_run_deadline_monotonic'], permit['matrix_deadline_monotonic']), 'Shared deadline expired')
    return copy.deepcopy(permit)
