"""Pinned, bounded, read-only IO for a future serial MiniCPM tap analysis.

No CLI, runtime admission, source import authentication or serial ownership is
provided here. The caller supplies the authenticated complete manifest and a
fresh resource/ownership guard. Snapshot creation stats files but never opens
their payloads. Generated fixture tests may use temporary directories.
"""
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import stat

from minicpm5_saved_tap_analysis_v1 import digest, require

STAT_FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid',
               'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_blocks')
DIRECTORY_FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid')
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
CHUNK_BYTES = 65536


def identity(value):
    return {k: getattr(value, k) for k in STAT_FIELDS}


def directory_identity(value):
    require(stat.S_ISDIR(value.st_mode), 'Ordinary parent directory')
    return {k: getattr(value, k) for k in DIRECTORY_FIELDS}


def path_parts(path):
    require(type(path) is str and 0 < len(path) < 4096 and '\0' not in path,
            'Literal bounded file path')
    p = Path(path)
    require(p.is_absolute() and p.anchor == '/' and str(p) == path and len(p.parts) >= 3
            and all(part not in ('', '.', '..') for part in p.parts[1:]),
            'Canonical absolute file path without traversal')
    return p.parts[1:]


@contextmanager
def parents(path):
    """Open each ancestor without following symlinks and retain directory FDs."""
    parts = path_parts(path)
    opened, links = [], []
    try:
        fd = os.open('/', DIRECTORY_FLAGS)
        opened.append(fd)
        root_identity = directory_identity(os.fstat(fd))
        snapshots = [root_identity]
        for name in parts[:-1]:
            parent = fd
            fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
            opened.append(fd)
            current = directory_identity(os.fstat(fd))
            require(directory_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == current,
                    'Parent unchanged during open')
            links.append((parent, name, fd, current))
            snapshots.append(current)
        yield fd, parts[-1], snapshots
        require(directory_identity(os.stat('/', follow_symlinks=False)) == root_identity,
                'Root directory unchanged during read')
        for parent, name, child, expected in links:
            require(directory_identity(os.fstat(child))
                    == directory_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == expected,
                    'Parent directory replaced during read')
    finally:
        for fd in reversed(opened):
            os.close(fd)


def ordinary(value, size, uid):
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1
            and value.st_uid == uid and value.st_size == size,
            'Exact single-link owned ordinary file and declared size')


def snapshot(ref, size, uid):
    """Capture current file/ancestor identity with stat only; payload unopened."""
    require(set(ref) == {'path', 'sha256'} and type(size) is int and 0 < size <= 8*2**20,
            'Literal bounded snapshot reference')
    digest(ref['sha256'])
    with parents(ref['path']) as (parent, name, directories):
        value = os.stat(name, dir_fd=parent, follow_symlinks=False)
        ordinary(value, size, uid)
        row = dict(**ref, bytes=size, identity=identity(value), directories=directories)
    return row


def validate_entry(row, uid, file_cap):
    require(set(row) == {'path', 'sha256', 'bytes', 'identity', 'directories'},
            'Exact pinned-file descriptor fields')
    parts = path_parts(row['path'])
    digest(row['sha256'])
    require(type(row['bytes']) is int and 0 < row['bytes'] <= file_cap,
            'Exact bounded declared file bytes')
    info = row['identity']
    require(set(info) == set(STAT_FIELDS) and all(type(v) is int for v in info.values()),
            'Full integer file identity')
    require(stat.S_ISREG(info['st_mode']) and info['st_nlink'] == 1
            and info['st_uid'] == uid and info['st_size'] == row['bytes'],
            'Pinned single-link owned file identity')
    require(type(row['directories']) is list and len(row['directories']) == len(parts),
            'Pinned complete ancestor chain')
    for item in row['directories']:
        require(set(item) == set(DIRECTORY_FIELDS) and all(type(v) is int for v in item.values())
                and stat.S_ISDIR(item['st_mode']), 'Pinned ordinary ancestor identity')


class PinnedReader:
    """Exactly-once, complete bounded reads; any error permanently poisons it."""
    def __init__(self, entries, guard, *, uid, file_cap, total_cap):
        require(type(uid) is int and uid >= 0 and type(file_cap) is int
                and 0 < file_cap <= 8*2**20 and type(total_cap) is int
                and 0 < total_cap <= 64*2**20, 'Explicit finite IO limits')
        require(type(entries) is list and 0 < len(entries) <= 1024, 'Bounded explicit input list')
        for row in entries:
            validate_entry(row, uid, file_cap)
        require(len({r['path'] for r in entries}) == len(entries)
                and sum(r['bytes'] for r in entries) <= total_cap,
                'Unique paths and complete declared read budget')
        self.entries = {r['path']: copy.deepcopy(r) for r in entries}
        self.guard, self.uid = guard, uid
        self.file_cap, self.total_cap = file_cap, total_cap
        self.seen, self.bytes_read = set(), 0
        self.poisoned = self.finished = False

    def __call__(self, ref, size=None):
        require(not self.poisoned and not self.finished, 'Failed or finalized reader cannot resume')
        try:
            require(type(ref) is dict and set(ref) == {'path', 'sha256'}
                    and ref['path'] in self.entries, 'Only exact admitted input paths')
            row = self.entries[ref['path']]
            require(ref == dict(path=row['path'], sha256=row['sha256'])
                    and ref['path'] not in self.seen, 'Exact once-only pinned reference')
            require(size is None or type(size) is int and size == row['bytes'],
                    'Caller requested exact admitted size')
            require(self.bytes_read + row['bytes'] <= self.total_cap, 'Total budget before open')
            self.guard()
            with parents(row['path']) as (parent, name, directories):
                require(directories == row['directories'], 'Ancestor chain matches pinned identities')
                fd = os.open(name, FILE_FLAGS, dir_fd=parent)
                try:
                    before = os.fstat(fd)
                    ordinary(before, row['bytes'], self.uid)
                    require(identity(before) == row['identity'], 'File matches pinned full identity')
                    pieces, count, hasher = [], 0, hashlib.sha256()
                    while count < row['bytes']:
                        self.guard()
                        data = os.read(fd, min(CHUNK_BYTES, row['bytes']-count))
                        self.bytes_read += len(data)
                        require(bool(data) and self.bytes_read <= self.total_cap, 'Complete bounded file read')
                        pieces.append(data)
                        count += len(data)
                        hasher.update(data)
                        self.guard()
                    extra = os.read(fd, 1)
                    self.bytes_read += len(extra)
                    require(extra == b'' and hasher.hexdigest() == row['sha256'],
                            'Complete exact SHA256 without trailing bytes')
                    self.guard()
                    require(identity(os.fstat(fd)) == identity(os.stat(
                        name, dir_fd=parent, follow_symlinks=False)) == row['identity'],
                        'File changed or was replaced during read')
                finally:
                    os.close(fd)
            self.guard()
            self.seen.add(row['path'])
            return b''.join(pieces)
        except BaseException:
            self.poisoned = True
            raise

    def finish(self):
        require(not self.poisoned and not self.finished, 'Failed or finalized reader cannot finish')
        try:
            self.guard()
            require(self.seen == set(self.entries)
                    and self.bytes_read == sum(r['bytes'] for r in self.entries.values()),
                    'Every admitted file read exactly once at full length')
            self.finished = True
            return dict(files=len(self.seen), bytes_read=self.bytes_read,
                        all_pinned_inputs_read_once=True, all_input_hashes_verified=True)
        except BaseException:
            self.poisoned = True
            raise


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires a separately reviewed serial executor')
