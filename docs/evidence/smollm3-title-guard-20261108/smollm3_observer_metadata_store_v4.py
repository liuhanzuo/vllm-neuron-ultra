"""Explicit filesystem transport for source-map chunks; inert on import.

The admitted lifecycle supplies its existing durable exclusive byte writer.
Every data reference resolves only to a digest-named JSON file in this run's
private control directory. Raw byte hashing remains separate from expansion.
"""
import os
from pathlib import Path
import stat
import smollm3_observer_metadata_codec_v5 as codec

require=codec.require

def ordinary(path,directory=False):
    path=Path(path)
    require(path.is_absolute()and path==path.resolve()and not any(x.is_symlink()for x in (path,*path.parents)),'Owned canonical metadata path')
    value=path.lstat()
    require(value.st_uid==os.getuid(),'Owned metadata identity')
    require(stat.S_ISDIR(value.st_mode)if directory else stat.S_ISREG(value.st_mode),'Ordinary metadata type')
    require(stat.S_IMODE(value.st_mode)==(0o700 if directory else 0o600),'Private metadata permissions')
    if not directory:require(value.st_nlink==1,'No metadata hardlinks')
    return value

def identity(value):return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns

def directory_identity(value):return value.st_dev,value.st_ino

def chunk_directory(root,catalog=False):
    require(type(catalog)is bool,'Explicit map namespace')
    root=Path(root);ordinary(root,True);ordinary(root/'control',True)
    return root/('control/catalog-maps'if catalog else'control/source-maps')


def catalog_record(root,path):return Path(path)==Path(root)/'control/successor-catalog.json'

def provision(root):
    """Call once, after fresh private run/control roots and before admission."""
    path=chunk_directory(root);path.mkdir(mode=0o700,exist_ok=False)
    chunk_directory(root,True).mkdir(mode=0o700,exist_ok=False)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),device=ordinary(path,True).st_dev,inode=ordinary(path,True).st_ino)

def read_chunk(root,digest,expected_directory=None,catalog=False):
    require(type(digest)is str and codec.SHA.fullmatch(digest),'Literal source chunk digest only')
    directory=chunk_directory(root,catalog);before_dir=ordinary(directory,True)
    require(expected_directory is None or directory_identity(before_dir)==expected_directory,'Source chunk directory replaced')
    path=directory/(digest+'.json');before=ordinary(path)
    require(before.st_size<=codec.CHUNK_CAP,'Source chunk byte cap')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(identity(os.fstat(fd))==identity(before),'Source chunk changed on open')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(codec.CHUNK_CAP+1)
        after=os.fstat(fd)
        require(len(raw)==before.st_size and identity(before)==identity(after)==identity(ordinary(path)),'Source chunk identity/length changed')
    finally:os.close(fd)
    require(directory_identity(ordinary(directory,True))==directory_identity(before_dir),'Source chunk directory changed while reading')
    require(codec.sha(raw)==digest,'Source chunk content hash')
    return raw

def record(root,path,value,write_once,legacy_json_bytes):
    """Chunks are durable before the parent record becomes visible."""
    root=Path(root);path=Path(path);ordinary(root,True)
    require(path.is_relative_to(root)and path!=root and path.suffix=='.json','Owned JSON record only')
    ordinary(path.parent,True)
    original=legacy_json_bytes(value)
    # Preserve the existing JSON writer's accepted Python input domain, e.g.
    # tuple shapes become arrays before the strict JSON-only codec sees them.
    normalized=codec.decode_json(original)
    def has_maps(item):
        if type(item)is dict:
            return any((key in codec.FIELDS and type(child)is dict and bool(child))or has_maps(child)for key,child in item.items())
        return type(item)is list and any(has_maps(child)for child in item)
    if not has_maps(normalized):return write_once(path,original)
    wire,chunks=codec.pack(normalized)
    if not chunks:return write_once(path,original)
    catalog=catalog_record(root,path);directory=chunk_directory(root,catalog);expected=directory_identity(ordinary(directory,True))
    for digest,raw in sorted(chunks.items()):
        target=directory/(digest+'.json')
        require(directory_identity(ordinary(directory,True))==expected,'Source chunk directory changed before write')
        if target.exists():require(read_chunk(root,digest,expected,catalog)==raw,'Existing immutable chunk differs')
        else:
            try:write_once(target,raw)
            except FileExistsError:
                # Only identical completed content can settle a concurrent publish.
                # Any pending loser file remains charged by the full inventory.
                require(read_chunk(root,digest,expected,catalog)==raw,'Concurrent immutable chunk differs')
            require(read_chunk(root,digest,expected,catalog)==raw,'Published chunk must be complete before record')
    require(directory_identity(ordinary(directory,True))==expected,'Source chunk directory changed before parent record')
    return write_once(path,wire)

def decode(root,path,raw):
    """Expand semantics without substituting bytes used by provenance hashes."""
    require(type(raw)is bytes and len(raw)<=codec.RECORD_CAP,'Original evidence byte cap')
    value=codec.decode_json(raw)
    if type(value)is not dict or value.get('schema')!=codec.RECORD_SCHEMA:
        require(not(type(value)is dict and str(value.get('schema','')).startswith('smollm3_compact_source_record_')),'Unsupported compact evidence version')
        return value
    root=Path(root);path=Path(path);ordinary(root,True)
    require(path.is_relative_to(root)and path!=root and path.suffix=='.json','Compact record belongs to this run only')
    ordinary(path.parent,True)
    catalog=catalog_record(root,path);directory=chunk_directory(root,catalog);expected=directory_identity(ordinary(directory,True))
    return codec.unpack(raw,lambda digest:read_chunk(root,digest,expected,catalog))


def authenticated_chunks(root,path,raw):
    """Root finalizer uses this to retain the complete physical dependency closure."""
    value=codec.decode_json(raw)
    if type(value)is not dict or value.get('schema')!=codec.RECORD_SCHEMA:return []
    catalog=catalog_record(root,path);directory=chunk_directory(root,catalog);expected=directory_identity(ordinary(directory,True));refs={}
    def resolve(digest):
        data=read_chunk(root,digest,expected,catalog);refs[digest]=dict(path=str(directory/(digest+'.json')),sha256=digest,bytes=len(data));return data
    codec.unpack(raw,resolve)
    return [refs[digest]for digest in sorted(refs)]
