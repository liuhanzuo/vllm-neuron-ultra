"""Stable whole-file bridge for an independently admitted Stage A reviewer.

No executable entrypoint, native imports or implicit admission. The caller must
first authenticate original cases, new archive metadata and actual producer
exit/source/weight identities. Its guard must enforce that serial review lease.
"""
import os
from pathlib import Path
import stat
from dory_stage_a_saved_core_v1 import verify_archive, require

STAT_FIELDS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def identity(info):
    return {name: getattr(info, name) for name in STAT_FIELDS}


def ordinary(info, expected, size):
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and info.st_nlink == 1 and info.st_size == size,
            'Complete owned regular single-link archive')
    require(identity(info) == expected, 'Exact producer file identity before and after read')


class BoundedReader:
    """No seek and no unbounded reads; independent verifier starts at byte zero."""
    def __init__(self, stream, size, guard):
        require(stream.tell() == 0, 'Whole-file verification starts at offset zero')
        self.stream, self.size, self.guard = stream, size, guard
        self.consumed = 0

    def read(self, requested):
        self.guard()
        require(type(requested) is int and 1 <= requested <= 65536, 'Bounded positive archive read')
        require(self.stream.tell() == self.consumed, 'Sequential complete file bytes only')
        remaining = self.size-self.consumed
        require(remaining >= 0 and (requested <= remaining or remaining == 0 and requested == 1),
                'Only declared bytes and a single EOF probe')
        block = self.stream.read(requested)
        require(type(block) is bytes and len(block) <= requested
                and self.consumed+len(block) <= self.size, 'No archive extent overrun')
        self.consumed += len(block)
        require(self.stream.tell() == self.consumed, 'No skipped file bytes')
        return block


def verify_file(root, name, producer_identity, archive, layout, *, max_bytes,
                guard, expected_tensors=None, selected_row_checks=None):
    require(callable(guard), 'Explicit independently admitted review guard required')
    guard()
    require(type(producer_identity) is dict and set(producer_identity) == set(STAT_FIELDS)
            and all(type(v) is int and v >= 0 for v in producer_identity.values()), 'Exact producer file stat fields')
    size = archive['bytes']
    require(type(size) is int and type(max_bytes) is int and 0 < size <= max_bytes,
            'Known full archive extent within admission before opening')
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Canonical admitted archive directory')
    require(isinstance(name, str) and name not in ('', '.', '..') and Path(name).name == name,
            'Single ordinary archive filename')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent = os.fstat(directory)
        require(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.getuid(), 'Owned actual archive directory')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            stream = os.fdopen(fd, 'rb')
        except BaseException:
            # fdopen may already close its input while failing construction.
            # Preserve the original failure whether or not ownership moved.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with stream:
            before = os.fstat(stream.fileno()); ordinary(before, producer_identity, size)
            require(identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) == producer_identity,
                    'Producer name still selects opened file')
            reader = BoundedReader(stream, size, guard)
            result = verify_archive(reader, archive, layout, max_bytes=max_bytes,
                expected_tensors=expected_tensors, selected_row_checks=selected_row_checks, guard=guard)
            require(reader.consumed == size and stream.tell() == size, 'Entire actual file read from zero through EOF')
            guard()
            after = os.fstat(stream.fileno()); current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            ordinary(after, producer_identity, size); ordinary(current, producer_identity, size)
            require(root.resolve() == root, 'Canonical archive path remains free of symlink ancestors')
            now = root.stat(follow_symlinks=False)
            require(stat.S_ISDIR(now.st_mode) and now.st_uid == parent.st_uid
                    and (now.st_dev, now.st_ino) == (parent.st_dev, parent.st_ino), 'Same original archive directory')
            return dict(complete=True, file=name, file_identity=identity(after), bytes_verified=size,
                whole_file_from_offset_zero=True, stable_producer_identity_verified=True, byte_verification=result,
                source_weight_and_actual_exit_provenance_verified_here=False,
                runtime_admitted_here=False, full_model_qualified=False, new_speedup_claim=False)
    finally:
        os.close(directory)
