"""Read explicitly pinned Python source while preserving installed hardlinks.

Only caller-recorded identities authorize multiple links. This reader never
changes a link or source file, accepts no native/model payload, and cannot
establish the provenance of the interpreter that calls it.
"""
import hashlib
import os
from pathlib import Path
import re
import stat

from nemotron4b_helper_dependency_catalog_v1 import FIELDS
from nemotron4b_process_handshake_v1 import require

CAP = 8*2**20


def identity(info):
    return {key:getattr(info,key) for key in FIELDS}


def canonical(path):
    require(path.is_absolute() and path.suffix == '.py'
            and path.resolve() == path
            and not any(item.is_symlink() for item in (path,*path.parents)),
            'Canonical original Python source file')


def source(path, reference):
    """Read stable exact bytes; alias callers must authenticate link separately."""
    path = Path(path)
    require(type(reference) is dict and type(reference.get('fullstat')) is dict
            and set(reference['fullstat']) == set(FIELDS),
            'Exact complete recorded source identity')
    expected = reference['fullstat']
    require(all(type(expected[key]) is int for key in FIELDS)
            and stat.S_ISREG(expected['st_mode']) and expected['st_nlink'] >= 1,
            'Recorded ordinary regular source with original link count')
    require(type(reference.get('bytes')) is int
            and 0 <= reference['bytes'] == expected['st_size'] <= CAP,
            'Exact bounded source byte count')
    require(type(reference.get('sha256')) is str
            and re.fullmatch('[0-9a-f]{64}',reference['sha256']) is not None,
            'Explicit source SHA256 pin')
    if expected['st_nlink'] != 1 or 'original_nlink' in reference:
        require(type(reference.get('original_nlink')) is int
                and reference['original_nlink'] == expected['st_nlink'],
                'Multiple links require exact explicit original link-count policy')
    canonical(path)
    require(identity(path.lstat()) == expected, 'Recorded source identity before open')
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        require(identity(os.fstat(fd)) == expected, 'Exact source identity after open')
        chunks=[];size=0
        while True:
            raw=os.read(fd,min(65536,CAP+1-size))
            if not raw:
                break
            chunks.append(raw);size+=len(raw)
            require(size <= CAP, 'Bounded source did not grow past read cap')
        raw=b''.join(chunks)
        canonical(path)
        require(identity(os.fstat(fd)) == expected == identity(path.lstat())
                and len(raw) == reference['bytes'], 'Complete source identity unchanged during read')
        require(hashlib.sha256(raw).hexdigest() == reference['sha256'],
                'Complete original source digest unchanged')
        # Python compile/decode_source handles original PEP263 encoding.
        return raw,dict(path=str(path),sha256=reference['sha256'],bytes=len(raw),
                        fullstat=dict(expected),original_nlink=expected['st_nlink'])
    finally:
        os.close(fd)
