"""Standalone descriptor evidence; not installed in any runtime admission path.

All IO is supplied explicitly. Pass the unchanged protocol_v4.proc as `proc`.
The caller must authenticate helpers and prove the retained parent-only fd's
lifetime. Stable snapshots and CLOEXEC do not prove continuous ownership,
prevent same-inode descriptor ABA, or prevent inheritance across fork.
"""
import copy
import os
import re
import stat as stat_module

LOCK_PATH = '/home/ubuntu/experiments/trn1-latest-models-20261031/serial-enrollment.lock'
FDINFO_CAP = 65536
GLOBAL_CAP = 2 * 2**20
MAX_READS = 2048
ROW = re.compile(
    r'([1-9][0-9]*):\s+(?:(->)\s+)?'
    r'(FLOCK|POSIX|OFDLCK|LEASE|DELEG|ACCESS)\s+'
    r'(ADVISORY|MANDATORY|ACTIVE|BREAKING|BREAKER)\s+'
    r'(READ|WRITE|UNLCK)\s+(-?[0-9]+)\s+'
    r'([0-9a-f]+):([0-9a-f]+):([0-9]+)\s+([0-9]+)\s+(EOF|[0-9]+)')


class DescriptorProofError(RuntimeError):
    """Caller may persist observations; this exception never authorizes work."""
    def __init__(self, message, observations):
        super().__init__(message)
        self.observations = copy.deepcopy(observations)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fresh_text(path, cap, open_binary, observation):
    """A new supplied binary stream per call, bounded complete read to EOF.

    open_binary(path) must open this exact proc path freshly, not return a cached
    descriptor/seek position. No retries, seeks or reopening occur here.
    """
    observation.update(path=path, byte_limit=cap, read_calls=0, eof_observed=False)
    chunks = []; size = 0
    try:
        with open_binary(path) as stream:
            while True:
                require(observation['read_calls'] < MAX_READS, 'Bounded proc read calls')
                chunk = stream.read(cap + 1 - size)
                observation['read_calls'] += 1
                require(type(chunk) is bytes, 'Binary proc text reader required')
                chunks.append(chunk); size += len(chunk)
                require(size <= cap, 'Proc text exceeds fixed byte cap')
                if not chunk:
                    observation['eof_observed'] = True
                    break
    finally:
        raw = b''.join(chunks)
        observation['bytes_read'] = len(raw)
        try:
            observation['text'] = raw.decode('ascii')
        except UnicodeDecodeError:
            observation['hex'] = raw.hex()
    require('text' in observation, 'ASCII proc format required')
    text = observation['text']
    require(not text or text.endswith('\n'), 'Complete newline-terminated proc text')
    return text


def parse_row(text):
    match = ROW.fullmatch(text)
    require(match is not None, 'Unknown or malformed global/fdinfo lock row')
    ordinal, blocked, kind, state, mode, pid, major, minor, inode, start, end = match.groups()
    require(all(len(x) <= 20 for x in (ordinal, pid, major, minor, inode, start, end)),
            'Bounded lock integer fields')
    require(int(pid) >= -1 and (pid != '-1' or kind == 'OFDLCK'), 'Known lock owner PID format')
    require(state in (('ACTIVE', 'BREAKING', 'BREAKER') if kind in ('LEASE', 'DELEG') else ('ADVISORY',)),
            'Known lock type/state combination')
    require(end == 'EOF' or int(end) >= int(start), 'Ordered lock range')
    device = os.makedev(int(major, 16), int(minor, 16))
    return dict(ordinal=ordinal, blocked=bool(blocked), kind=kind, device=device,
                inode=int(inode), semantic=text.split()[1:])


def target_fields(owner, lock):
    return ['FLOCK', 'ADVISORY', 'WRITE', str(owner['pid']),
            '%02x:%02x:%d' % (os.major(lock['device']), os.minor(lock['device']), lock['inode']), '0', 'EOF']


def check_fdinfo(text, owner, lock):
    headers = {}; rows = []
    for line in text.splitlines():
        if line.startswith('lock:'):
            rows.append(parse_row(line[5:].strip()))
            continue
        match = re.fullmatch(r'(pos|flags|mnt_id|ino):\s+([0-9]+)', line)
        require(match is not None and match[1] not in headers, 'Exact unique fdinfo headers')
        headers[match[1]] = match[2]
    require(set(headers) == {'pos', 'flags', 'mnt_id', 'ino'}, 'Every required fdinfo header')
    require(all(len(x) <= 20 for x in headers.values()), 'Bounded fdinfo integers')
    require(re.fullmatch('[0-7]+', headers['flags']) is not None, 'Octal fdinfo flags')
    flags = int(headers['flags'], 8)
    require(flags & os.O_CLOEXEC and int(headers['mnt_id']) > 0 and int(headers['ino']) == lock['inode'],
            'Exact fdinfo inode, mount and CLOEXEC')
    require(len(rows) == 1 and rows[0]['ordinal'] == '1' and not rows[0]['blocked']
            and rows[0]['semantic'] == target_fields(owner, lock), 'One exact retained-fd whole-file FLOCK')
    return dict(headers=headers, lock_row=rows[0], cloexec=True)


def check_global(text, owner, lock):
    rows = [parse_row(line) for line in text.splitlines()]
    matches = [row for row in rows if (row['device'], row['inode']) == (lock['device'], lock['inode'])]
    wanted = target_fields(owner, lock)
    require(matches and all(not row['blocked'] and row['semantic'] == wanted for row in matches),
            'Global target missing or same-inode conflicting/blocked lock')
    require(len({row['ordinal'] for row in matches}) == len(matches), 'Repeated printed ordinal rejected')
    return dict(total_rows=len(rows), target_rows=matches, target_row_count=len(matches),
                ordinal_only_repetitions=len(matches) > 1)


def stat_record(info):
    fields = dict(device='st_dev', inode='st_ino', mode='st_mode', nlink='st_nlink',
                  uid='st_uid', gid='st_gid', size='st_size', mtime_ns='st_mtime_ns', ctime_ns='st_ctime_ns')
    row = {key: getattr(info, field) for key, field in fields.items()}
    require(all(type(x) is int and x >= 0 for x in row.values()), 'Complete nonnegative file stat metadata')
    require(stat_module.S_ISREG(row['mode']) and row['nlink'] == 1, 'Regular singleton lock file')
    return row


def prove(owner, lock, retained_fd, *, proc, open_binary, lstat, stat, readlink):
    """Observe one exact held descriptor with conservative global agreement.

    `proc` must be the original authenticated protocol_v4.proc contract. Other
    callbacks must implement ordinary fresh proc opening/lstat/stat/readlink,
    without caches or substituted observations. No native/runtime admission,
    actual process exit or continuous lifetime is established by this module.
    On denial, DescriptorProofError.observations retains every completed read.
    """
    evidence = dict(schema='descriptor_serial_lock_observation_v1', stage='binding',
                    passed=False, runtime_admitted=False, continuous_lifetime_proven=False,
                    same_inode_fd_ABA_excluded=False, fork_inheritance_excluded=False,
                    original_guard_invoked=False, proof_retried=False)
    try:
        require(type(owner) is dict and set(owner) == {'pid', 'start_ticks', 'ppid', 'argv', 'cgroup'}
                and type(owner['pid']) is int and owner['pid'] > 0
                and type(owner['ppid']) is int and owner['ppid'] >= 0
                and type(owner['start_ticks']) is str and owner['start_ticks'].isdigit()
                and type(owner['argv']) is list and owner['argv'] and all(type(x) is str for x in owner['argv'])
                and type(owner['cgroup']) is str and owner['cgroup'], 'Exact original owner record')
        require(type(lock) is dict and set(lock) == {'device', 'inode'}
                and type(lock['device']) is int and lock['device'] >= 0
                and type(lock['inode']) is int and lock['inode'] > 0, 'Exact original lock device/inode')
        require(type(retained_fd) is int and 0 <= retained_fd < 2**31, 'Explicit retained parent fd')
        require(all(callable(x) for x in (proc, open_binary, lstat, stat, readlink)), 'Explicit authenticated IO providers')
        owner = copy.deepcopy(owner); lock = copy.deepcopy(lock)
        fd_path = '/proc/%d/fd/%d' % (owner['pid'], retained_fd)
        evidence.update(owner=owner, lock=lock, retained_fd=retained_fd, lock_path=LOCK_PATH)

        def snapshot():
            require(proc(owner['pid']) == owner, 'Original process identity before descriptor snapshot')
            path_info = stat_record(lstat(LOCK_PATH)); target = readlink(fd_path); fd_info = stat_record(stat(fd_path))
            require(target == LOCK_PATH, 'Exact canonical retained fd target, not deleted/replaced')
            require(path_info == fd_info and (fd_info['device'], fd_info['inode']) == (lock['device'], lock['inode']),
                    'Original path and retained descriptor identify same exact file')
            require(proc(owner['pid']) == owner, 'Original process identity after descriptor snapshot')
            return dict(path_stat=path_info, fd_stat=fd_info, fd_target=target)

        evidence['stage'] = 'before'; evidence['before'] = snapshot()
        evidence['stage'] = 'fdinfo'; evidence['fdinfo_read'] = {}
        text = fresh_text('/proc/%d/fdinfo/%d' % (owner['pid'], retained_fd), FDINFO_CAP, open_binary, evidence['fdinfo_read'])
        evidence['fdinfo'] = check_fdinfo(text, owner, lock)
        evidence['stage'] = 'global'; evidence['global_read'] = {}
        text = fresh_text('/proc/locks', GLOBAL_CAP, open_binary, evidence['global_read'])
        evidence['global'] = check_global(text, owner, lock)
        evidence['stage'] = 'after'; evidence['after'] = snapshot()
        require(evidence['before'] == evidence['after'], 'Descriptor/path metadata changed across observation')
        evidence.update(stage='complete', passed=True)
        return evidence
    except Exception as error:
        evidence.update(error_type=type(error).__name__, error=str(error))
        raise DescriptorProofError(str(error), evidence) from error
