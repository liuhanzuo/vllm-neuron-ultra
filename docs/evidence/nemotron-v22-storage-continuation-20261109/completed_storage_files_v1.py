"""Bounded file operations for a separately admitted serial maintenance worker.

No executable entry point or target selection. Hashes, verified private gzip
archives and stable identities precede an exact-file retirement. Callers must
hold the original serial lock and authenticate the actual completed history.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat

FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_blocks', 'st_nlink',
          'st_mtime_ns', 'st_ctime_ns')
BLOCK = 4*2**20


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(value):
    return {name:getattr(value, name) for name in FIELDS}


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'Canonical file and ordinary parent chain')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            'Single-link regular file only')
    return identity(info)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def once(path, value):
    path = Path(path)
    require(path.parent.resolve() == path.parent and not path.exists(),
            'New exact metadata output')
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= 32*2**20, 'Bounded maintenance metadata')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def append(path, value):
    path = Path(path)
    require(path.parent.resolve() == path.parent, 'Canonical action journal parent')
    before = ordinary(path) if path.exists() else None
    raw = (json.dumps(value, allow_nan=False)+'\n').encode()
    require(len(raw) <= 2**20, 'Bounded action journal record')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        opened = identity(os.fstat(stream.fileno()))
        require(stat.S_ISREG(opened['st_mode']) and opened['st_nlink'] == 1,
                'Ordinary single-link action journal')
        require(before is None or opened == before, 'Action journal identity unchanged before append')
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        require(identity(os.fstat(stream.fileno())) == ordinary(path), 'Action journal still at exact path')
    sync_directory(path.parent)


def file_hash(path, expected, guard):
    path = Path(path)
    require(ordinary(path) == expected, 'Exact file identity before hash')
    digest = hashlib.sha256()
    count = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        require(identity(os.fstat(stream.fileno())) == expected, 'Exact opened file identity')
        while block := stream.read(BLOCK):
            guard()
            count += len(block)
            require(count <= expected['st_size'], 'No file overrun during hash')
            digest.update(block)
        require(identity(os.fstat(stream.fileno())) == expected == ordinary(path),
                'Source unchanged throughout hash')
    require(count == expected['st_size'], 'Complete file hashed')
    return dict(bytes=count, sha256=digest.hexdigest())


def verify_gzip(path, original_bytes, original_sha256, guard):
    path = Path(path)
    before = ordinary(path)
    digest = hashlib.sha256()
    count = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as src:
        require(identity(os.fstat(src.fileno())) == before, 'Exact opened gzip identity')
        with gzip.GzipFile(fileobj=src, mode='rb') as stream:
            while block := stream.read(BLOCK):
                guard()
                count += len(block)
                require(count <= original_bytes, 'No decompressed size overrun')
                digest.update(block)
        require(identity(os.fstat(src.fileno())) == before, 'Opened gzip unchanged')
    require(count == original_bytes and digest.hexdigest() == original_sha256,
            'All decompressed journal bytes match original')
    require(ordinary(path) == before, 'Verified gzip unchanged')
    return dict(bytes=count, sha256=digest.hexdigest())


def archive_journal(source, target, expected, guard, cap):
    source, target = Path(source), Path(target)
    require(ordinary(source) == expected, 'Exact journal before compression')
    require(target.parent.resolve() == target.parent and not target.exists(), 'Fresh gzip path')
    digest = hashlib.sha256()
    count = 0
    srcfd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(srcfd, 'rb') as src:
        require(identity(os.fstat(src.fileno())) == expected, 'Exact opened journal')
        dstfd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(dstfd, 'wb') as dst:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dst, compresslevel=6, mtime=0) as zipped:
                while block := src.read(BLOCK):
                    guard()
                    count += len(block)
                    require(count <= expected['st_size'], 'No journal overrun')
                    digest.update(block)
                    zipped.write(block)
                    require(dst.tell() <= cap, 'Bounded compressed journal')
            dst.flush()
            os.fsync(dst.fileno())
            require(dst.tell() <= cap, 'Complete gzip including footer fits cap')
        require(identity(os.fstat(src.fileno())) == expected == ordinary(source),
                'Journal unchanged throughout compression')
    sync_directory(target.parent)
    require(count == expected['st_size'], 'Complete original journal compressed')
    verify_gzip(target, count, digest.hexdigest(), guard)
    target_stat = ordinary(target)
    compressed = file_hash(target, target_stat, guard)
    return dict(source=str(source), source_stat=expected, original_bytes=count,
                original_sha256=digest.hexdigest(), archive=str(target), archive_stat=target_stat,
                gzip_bytes=compressed['bytes'], gzip_sha256=compressed['sha256'],
                independently_decompressed=True)


def unlink_exact(path, expected):
    """Operate relative to an opened parent, without following the final path."""
    path = Path(path)
    require(ordinary(path) == expected, 'Exact retirement target identity')
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        value = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require(identity(value) == expected and stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
                'Exact ordinary target immediately before unlink')
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    require(not path.exists() and not path.is_symlink(), 'Exact target retired')


def snapshot_tree(root, excluded=()):
    """Stat every retained entry; root directory timestamps may change on unlink."""
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Ordinary protected tree')
    excluded = {str(Path(p)) for p in excluded}
    rows = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            path = Path(parent)/name
            if str(path) in excluded:continue
            info = path.lstat()
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
                    or stat.S_ISLNK(info.st_mode), 'Ordinary protected tree entry')
            rows.append(dict(path=str(path.relative_to(root)), **identity(info),
                             link=os.readlink(path) if stat.S_ISLNK(info.st_mode) else None))
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return dict(entries=len(rows), sha256=hashlib.sha256(raw).hexdigest())
