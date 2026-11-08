# Candidate v5: only inline constant-message require checks from frozen v4.
"""Pure lossless metadata codec. No filesystem, service, model or runtime admission.

The caller must publish each returned chunk durably in its owned control tree
before the enclosing record, and authenticate that directory on every read.
Only complete source-hash/stat maps move to chunks. Process identities, permits,
scientific results and failure/exit fields stay in the record unchanged.
"""
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
RECORD_SCHEMA = 'smollm3_compact_source_record_v2'
CHUNK_SCHEMA = 'smollm3_shared_source_chunk_v2'
MARKER = '$smollm3_source_map'
FIELDS = {'source_stats': 'stat', 'root_runtime_source_sha256': 'sha256', 'runtime_source_sha256': 'sha256', 'source_sha256': 'fingerprint', 'new_source_and_evidence_pins': 'fingerprint'}
CHUNK_ENTRIES = 1024
CHUNK_CAP = 1024 * 1024
RECORD_CAP = 8 * 1024 * 1024
EXPANSION_CAP = 32 * 1024 * 1024
ENTRY_CAP = 100000
DEPTH_CAP = 48
SHA = re.compile('[0-9a-f]{64}')
STAT_KEYS = {'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'realpath'}
FORBIDDEN_SUFFIXES = {'.raw', '.pt', '.bin', '.safetensors', '.neff', '.so', '.whl', '.zlib', '.pb'}

def require(value, message):
    if not value:
        raise ValueError(message)

def encode_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def pairs(items):
    result = {}
    for (key, value) in items:
        if not key not in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result

def decode_json(raw):

    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw.decode('utf8'), object_pairs_hook=pairs, parse_constant=bad_constant)

def absolute(name):
    if not (type(name) is str and 0 < len(name) <= 4096 and ('\x00' not in name)):
        raise ValueError('Bounded source path')
    p = PurePosixPath(name)
    if not (p.is_absolute() and str(p) == name and ('..' not in p.parts)):
        raise ValueError('Canonical absolute source path')
    return p

def absolute_suffix(name):
    """Same lexical acceptance and suffix as absolute(), without Path objects.

    POSIX exactly-two-leading-slash semantics stay intact. No filesystem or
    global cache; every path is checked on every decoded chunk.
    """
    if not (type(name) is str and 0 < len(name) <= 4096 and ('\x00' not in name)):
        raise ValueError('Bounded source path')
    if not name.startswith('/'):
        raise ValueError('Canonical absolute source path')
    if name in ('/', '//'):
        return ''
    root = 2 if name.startswith('//') else 1
    if not (not name.startswith('///') and (not name.endswith('/')) and ('//' not in name[root:]) and ('/./' not in name) and ('/../' not in name) and (not name.endswith(('/.', '/..')))):
        raise ValueError('Canonical absolute source path')
    tail = name.rsplit('/', 1)[-1]
    index = tail.rfind('.')
    return tail[index:] if 0 < index < len(tail) - 1 else ''

def validate_map(kind, entries):
    if not (kind in ('sha256', 'stat', 'fingerprint') and type(entries) is dict and (len(entries) <= ENTRY_CAP)):
        raise ValueError('Known bounded source map')
    for (name, value) in entries.items():
        if kind == 'fingerprint':
            if not (type(name) is str and 0 < len(name) <= 4096 and ('\x00' not in name)):
                raise ValueError('Bounded fingerprint name')
        elif not absolute_suffix(name) not in FORBIDDEN_SUFFIXES:
            raise ValueError('No model/native/raw payload in source map')
        if kind in ('sha256', 'fingerprint'):
            if not (type(value) is str and SHA.fullmatch(value)):
                raise ValueError('Exact source SHA256')
        else:
            if not (type(value) is dict and set(value) == STAT_KEYS):
                raise ValueError('Exact source stat fields')
            absolute_suffix(value['realpath'])
            for field in STAT_KEYS - {'realpath'}:
                if not (type(value[field]) is int and value[field] >= (0 if field == 'size' else 1)):
                    raise ValueError('Exact source stat integer')
    return entries

def pack_map(kind, entries, blobs):
    validate_map(kind, entries)
    names = sorted(entries)
    chunks = []
    for offset in range(0, len(names), CHUNK_ENTRIES):
        selected = names[offset:offset + CHUNK_ENTRIES]
        raw = encode_json(dict(schema=CHUNK_SCHEMA, kind=kind, entries={n: entries[n] for n in selected}))
        if not len(raw) <= CHUNK_CAP:
            raise ValueError('Bounded source chunk')
        digest = sha(raw)
        if not (digest not in blobs or blobs[digest] == raw):
            raise ValueError('Content hash collision')
        blobs[digest] = raw
        chunks.append(dict(sha256=digest, bytes=len(raw), count=len(selected)))
    manifest = dict(kind=kind, count=len(names), chunks=chunks)
    return {MARKER: dict(manifest, manifest_sha256=sha(encode_json(manifest)))}

def pack(value):
    """Return compact wire bytes and SHA-addressed chunks, without writing."""
    if not (type(value) is dict and value.get('schema') != RECORD_SCHEMA):
        raise ValueError('Original ordinary record required')
    blobs = {}

    def visit(item, field=None, depth=0):
        if not depth <= DEPTH_CAP:
            raise ValueError('Metadata nesting cap')
        if type(item) is dict:
            if not (MARKER not in item and all((type(k) is str for k in item))):
                raise ValueError('Reserved marker or non-string key')
            if field in FIELDS:
                return pack_map(FIELDS[field], item, blobs)
            return {k: visit(v, k, depth + 1) for (k, v) in item.items()}
        if type(item) is list:
            return [visit(v, None, depth + 1) for v in item]
        if not (item is None or type(item) in (str, bool, int, float)):
            raise ValueError('Ordinary JSON values only')
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Finite metadata number')
        return item
    record = visit(value)
    if not sum((len(x) for x in blobs.values())) <= EXPANSION_CAP:
        raise ValueError('Bounded distinct source maps')
    raw = encode_json(dict(schema=RECORD_SCHEMA, record=record))
    if not len(raw) <= RECORD_CAP:
        raise ValueError('Compact record cap')
    return (raw, blobs)

def unpack(raw, resolve):
    """Authenticate chunks supplied by caller; return the original JSON object.

    resolve(digest) must read only the matching owned control-tree chunk. No
    path supplied by a record is passed to the resolver. No global cache is kept.
    """
    if not (type(raw) is bytes and len(raw) <= RECORD_CAP):
        raise ValueError('Bounded compact record bytes')
    envelope = decode_json(raw)
    if not (type(envelope) is dict and set(envelope) == {'schema', 'record'} and (envelope['schema'] == RECORD_SCHEMA) and (type(envelope['record']) is dict)):
        raise ValueError('Exact compact record envelope')
    cache = {}
    loaded_bytes = 0
    expanded_bytes = 0

    def expand(value, field):
        nonlocal loaded_bytes, expanded_bytes
        if not (field in FIELDS and set(value) == {MARKER}):
            raise ValueError('Source reference only in its explicit map field')
        ref = value[MARKER]
        if not (type(ref) is dict and set(ref) == {'kind', 'count', 'chunks', 'manifest_sha256'}):
            raise ValueError('Exact map reference')
        if not (ref['kind'] == FIELDS[field] and type(ref['count']) is int and (0 <= ref['count'] <= ENTRY_CAP)):
            raise ValueError('Exact map kind/count')
        if not (type(ref['chunks']) is list and len(ref['chunks']) == (ref['count'] + CHUNK_ENTRIES - 1) // CHUNK_ENTRIES):
            raise ValueError('Exact bounded chunk count')
        manifest = {k: ref[k] for k in ('kind', 'count', 'chunks')}
        if not (type(ref['manifest_sha256']) is str and sha(encode_json(manifest)) == ref['manifest_sha256']):
            raise ValueError('Map manifest digest')
        result = {}
        last = None
        for (index, chunk) in enumerate(ref['chunks']):
            if not (type(chunk) is dict and set(chunk) == {'sha256', 'bytes', 'count'}):
                raise ValueError('Exact chunk reference')
            digest = chunk['sha256']
            if not (type(digest) is str and SHA.fullmatch(digest)):
                raise ValueError('Chunk digest')
            if not (type(chunk['bytes']) is int and 0 < chunk['bytes'] <= CHUNK_CAP):
                raise ValueError('Chunk byte bound')
            if not (type(chunk['count']) is int and chunk['count'] == min(CHUNK_ENTRIES, ref['count'] - index * CHUNK_ENTRIES)):
                raise ValueError('Exact chunk entries')
            if digest not in cache:
                payload = resolve(digest)
                if not (type(payload) is bytes and len(payload) == chunk['bytes'] and (sha(payload) == digest)):
                    raise ValueError('Exact source chunk bytes/hash')
                loaded_bytes += len(payload)
                if not loaded_bytes <= EXPANSION_CAP:
                    raise ValueError('Bounded loaded source maps')
                cache[digest] = payload
            payload = cache[digest]
            if not len(payload) == chunk['bytes']:
                raise ValueError('Cached chunk exact byte count')
            expanded_bytes += len(payload)
            if not expanded_bytes <= EXPANSION_CAP:
                raise ValueError('Bounded repeated map expansion')
            item = decode_json(payload)
            if not (type(item) is dict and set(item) == {'schema', 'kind', 'entries'} and (item['schema'] == CHUNK_SCHEMA) and (item['kind'] == ref['kind'])):
                raise ValueError('Exact source chunk schema/kind')
            entries = validate_map(ref['kind'], item['entries'])
            if not len(entries) == chunk['count']:
                raise ValueError('Exact source chunk count')
            for name in entries:
                if not (last is None or name > last):
                    raise ValueError('Strict global source order; no duplicate/overlapping chunks')
                last = name
                result[name] = entries[name]
        if not len(result) == ref['count']:
            raise ValueError('Complete original source map')
        return result

    def visit(item, field=None, depth=0):
        if not depth <= DEPTH_CAP:
            raise ValueError('Metadata nesting cap')
        if type(item) is dict:
            if MARKER in item:
                return expand(item, field)
            if not field not in FIELDS:
                raise ValueError('Explicit source map must use authenticated chunks')
            return {k: visit(v, k, depth + 1) for (k, v) in item.items()}
        if type(item) is list:
            return [visit(v, None, depth + 1) for v in item]
        if not (item is None or type(item) in (str, bool, int, float)):
            raise ValueError('Ordinary JSON values only')
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Finite metadata number')
        return item
    return visit(envelope['record'])
