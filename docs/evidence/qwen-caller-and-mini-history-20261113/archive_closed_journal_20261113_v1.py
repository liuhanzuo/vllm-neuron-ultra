"""Lossless retirement of one fixed, archived, unused OS journal.

No experiment inputs, History, serial lock, model, or queue are modified.
The complete archive is durably verified before removing the original copy.
"""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
SOURCE = Path('/var/log/journal/ec2139d93c43b899a9a1defcd3a54e08/user-1000@72cf4ba58fce415ab780db3f965e741b-000000000068ab3b-0006600268e6c05d.journal')
ARCHIVE = SOURCE.with_name(SOURCE.name + '.gz')
FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid', 'st_size', 'st_blocks', 'st_mtime_ns', 'st_ctime_ns')
EXPECTED = dict(zip(FIELDS, (66306, 258114, 33184, 1, 0, 101, 134217728, 262152, 1794462030891251864, 1794462030891251864)))


def identity(path):
    assert path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents))
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
    return {k: getattr(s, k) for k in FIELDS}


def archived():
    assert identity(SOURCE) == EXPECTED
    with SOURCE.open('rb') as stream:
        header = stream.read(17)
    assert header[:8] == b'LPKSHHRH' and header[16] == 2


def unused():
    key = (EXPECTED['st_dev'], EXPECTED['st_ino'])
    processes = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fds = list((proc / 'fd').iterdir())
            for fd in fds:
                try:
                    s = fd.stat()
                except FileNotFoundError:
                    continue
                assert (s.st_dev, s.st_ino) != key, 'Original journal has an open descriptor'
            for row in (proc / 'maps').read_text().splitlines():
                fields = row.split(None, 5)
                if len(fields) >= 5:
                    major, minor = (int(v, 16) for v in fields[3].split(':'))
                    assert (os.makedev(major, minor), int(fields[4])) != key, 'Original journal has a mapping'
            processes += 1
        except (FileNotFoundError, ProcessLookupError):
            continue
    return processes


def save(name, value):
    path = Q / name
    payload = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        os.fchown(stream.fileno(), 1000, 1000)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(Q)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def free():
    s = os.statvfs('/')
    return s.f_bavail * s.f_frsize


def main():
    assert os.getuid() == 0 and sys.dont_write_bytecode and __debug__
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (150 * 2**20,) * 2)
    assert free() >= 29 * 2**30
    assert not ARCHIVE.exists() and not ARCHIVE.is_symlink()
    spec_path = Q / 'qwen35-writer-geometry-r2-20261112-spec.json'
    payload = spec_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == '9208483c46d292a1f1fa7224e9033bf177c86dd231e3e64e0592bc823fed5a46'
    spec = json.loads(payload)
    for key in ('source_sha256', 'input_extension'):
        assert str(SOURCE) not in spec[key] and str(ARCHIVE) not in spec[key]
    archived()
    before_processes = unused()
    before_free = free()
    digest = hashlib.sha256()
    count = 0
    with ARCHIVE.open('xb') as target:
        os.fchmod(target.fileno(), 0o640)
        os.fchown(target.fileno(), EXPECTED['st_uid'], EXPECTED['st_gid'])
        with gzip.GzipFile(filename='', mode='wb', fileobj=target, compresslevel=6, mtime=0) as zipped:
            fd = os.open(SOURCE, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as original:
                assert {k: getattr(os.fstat(fd), k) for k in FIELDS} == EXPECTED
                while block := original.read(2**20):
                    count += len(block)
                    assert count <= EXPECTED['st_size']
                    digest.update(block)
                    zipped.write(block)
                assert {k: getattr(os.fstat(fd), k) for k in FIELDS} == EXPECTED
        target.flush()
        os.fsync(target.fileno())
    sync_dir(ARCHIVE.parent)
    assert count == EXPECTED['st_size'] and identity(SOURCE) == EXPECTED
    check = hashlib.sha256()
    restored = 0
    with gzip.open(ARCHIVE, 'rb') as stream:
        while block := stream.read(2**20):
            restored += len(block)
            assert restored <= count
            check.update(block)
    assert restored == count and check.digest() == digest.digest()
    archive_identity = identity(ARCHIVE)
    saved_bytes = (EXPECTED['st_blocks'] - archive_identity['st_blocks']) * 512
    assert saved_bytes > 0
    value = dict(schema='closed_journal_lossless_archive_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_path=str(SOURCE),
        archive_path=str(ARCHIVE), original_identity=EXPECTED, archive_identity=archive_identity,
        original_sha256=digest.hexdigest(), decompressed_sha256=check.hexdigest(),
        decompressed_bytes=restored, full_decompressed_content_verified=True,
        archived_header_state=2, processes_checked_before=before_processes,
        root_free_before=before_free, allocated_bytes_reclaimed=saved_bytes,
        frozen_experiment_source_or_input=False, model_or_evidence_deleted=False,
        serial_lock_or_handoff_changed=False, uncompressed_copy_removed=False)
    verified = save('closed-journal-lossless-archive-verified-20261113-v1.json', value)
    archived()
    value['processes_checked_before_unlink'] = unused()
    assert identity(SOURCE) == EXPECTED and identity(ARCHIVE) == archive_identity
    SOURCE.unlink()
    sync_dir(SOURCE.parent)
    assert not SOURCE.exists() and identity(ARCHIVE) == archive_identity
    value.update(verified_before_retirement=verified, uncompressed_copy_removed=True, root_free_after=free())
    receipt = save('closed-journal-lossless-archive-20261113-v1.json', value)
    print(json.dumps(dict(receipt=receipt, allocated_bytes_reclaimed=saved_bytes, root_free_after=value['root_free_after'])))


if __name__ == '__main__':
    main()
