"""Bounded streaming USTAR/member verification; no filesystem or process IO."""
import hashlib
import json
import re

BLOCK = 512
CHUNK = 65536
MAX_HEADERS = 10000
MAX_MEMBER = 64 * 2**20
MAX_STREAM = 256 * 2**20
MAX_HOPS = 32


def need(ok, message):
    if not ok:
        raise ValueError(message)


def text_field(raw):
    head, sep, tail = raw.partition(b'\0')
    need(not sep or not any(tail), 'Zero-filled USTAR text field')
    value = head.decode('utf-8', 'strict')
    need(all(ord(c) >= 32 and ord(c) != 127 for c in value), 'No control characters in USTAR text')
    return value


def octal(raw, *, empty=False):
    if empty and not raw.strip(b' \0'):
        return 0
    need(re.fullmatch(rb' *[0-7]+[ \0]*', raw) is not None, 'Finite octal USTAR number; no base-256')
    return int(raw.strip(b' \0'), 8)


def member_name(value, directory=False):
    need(type(value) is str and 0 < len(value) <= 4096 and not value.startswith('/')
         and '\\' not in value and all(ord(c) >= 32 and ord(c) != 127 for c in value),
         'Bounded relative archive member name')
    if value in ('.', './'):
        need(directory, 'Archive root is a directory')
        return ''
    if value.startswith('./'):
        value = value[2:]
    if directory and value.endswith('/'):
        value = value[:-1]
    parts = value.split('/')
    need(all(p not in ('', '.', '..') for p in parts), 'No empty, dot or traversal member components')
    return '/'.join(parts)


def link_destination(name, value, kind):
    need(type(value) is str and 0 < len(value) <= 4096 and '\\' not in value
         and all(ord(c) >= 32 and ord(c) != 127 for c in value), 'Bounded archive-only link text')
    if kind == 'hardlink':
        return member_name(value)
    absolute = value.startswith('/')
    need(not value.startswith('//'), 'No ambiguous absolute link root')
    parts = [] if absolute else name.split('/')[:-1]
    source = value[1:] if absolute else value
    if source == '' and absolute:
        return ''
    for part in source.split('/'):
        need(part != '', 'No repeated or trailing separators in links')
        if part == '.':
            continue
        if part == '..':
            need(parts, 'Link cannot escape package root')
            parts.pop()
        else:
            parts.append(part)
    result = '/'.join(parts)
    need(len(result) <= 4096, 'Bounded resolved archive link')
    return result


def header(raw):
    need(len(raw) == BLOCK and raw[257:265] in (b'ustar\0' + b'00', b'ustar  \0'),
         'Exact POSIX USTAR or reviewed minimal GNU magic and version')
    if raw[257:265] == b'ustar  \0':
        need(not any(raw[345:512]), 'Minimal GNU header has no prefix, sparse or time extensions')
    need(octal(raw[148:156]) == sum(raw[:148]) + 8 * 32 + sum(raw[156:]), 'USTAR header checksum')
    need(not any(raw[500:512]), 'Zero USTAR header padding')
    kind = {b'0': 'file', b'\0': 'file', b'5': 'directory', b'2': 'symlink', b'1': 'hardlink'}.get(raw[156:157])
    need(kind is not None, 'Only file, directory, symlink and hardlink members')
    for start, end in ((100,108), (108,116), (116,124), (136,148)):
        octal(raw[start:end])
    need(octal(raw[329:337], empty=True) == octal(raw[337:345], empty=True) == 0,
         'No device numbers in supported USTAR members')
    text_field(raw[265:297]); text_field(raw[297:329])
    name, prefix = text_field(raw[:100]), text_field(raw[345:500])
    need(name != '', 'Nonempty USTAR member name')
    name = member_name(prefix + '/' + name if prefix else name, kind == 'directory')
    size = octal(raw[124:136]); link = text_field(raw[157:257])
    need(size <= MAX_MEMBER and (kind == 'file' or size == 0), 'Bounded file size; non-files have no body')
    result = dict(name=name, type=kind, size=size)
    if kind in ('symlink', 'hardlink'):
        result.update(link=link, destination=link_destination(name, link, kind))
    else:
        need(link == '', 'No link text on regular files or directories')
    return result


class Stream:
    def __init__(self, read, guard):
        need(callable(read) and callable(guard), 'Explicit read(n) and guard callbacks')
        self.read, self.guard = read, guard
        self.bytes = 0
        self.hash = hashlib.sha256()

    def take(self, size):
        need(0 < size <= CHUNK, 'Bounded stream read request')
        size = min(size, MAX_STREAM - self.bytes + 1)
        self.guard()
        raw = self.read(size)
        need(type(raw) is bytes and len(raw) <= size, 'Reader returns at most requested exact bytes')
        self.bytes += len(raw)
        need(self.bytes <= MAX_STREAM, 'Complete decompressed stream byte cap')
        self.hash.update(raw)
        self.guard()
        return raw

    def exact(self, size):
        parts = []
        while size:
            raw = self.take(min(size, CHUNK))
            need(raw, 'Truncated USTAR header, member, padding or end marker')
            parts.append(raw); size -= len(raw)
        return b''.join(parts)


def resolve(name, table):
    parts, pending, seen, chain, hops, hard = [], name.split('/'), set(), [], 0, False
    while pending:
        need(len('/'.join(parts + pending)) <= 4096, 'Bounded archive link resolution')
        part = pending.pop(0)
        if part == '.':
            continue
        if part == '..':
            need(parts, 'Resolved link cannot escape package root')
            parts.pop(); continue
        path = '/'.join(parts + [part]); row = table.get(path)
        if row is None:
            need(pending and any(key.startswith(path + '/') for key in table),
                 'Required target or parent directory is missing from archive')
            parts.append(part); continue  # Explicit directory records are optional.
        if row['type'] in ('symlink', 'hardlink'):
            need(not hard or row['type'] == 'hardlink', 'Hardlinks must resolve directly through hardlinks to a file')
            state = (path, tuple(pending))
            need(state not in seen, 'No cyclic archive link resolution'); seen.add(state)
            hops += 1; need(hops <= MAX_HOPS, 'At most 32 archive link hops')
            chain.append(dict(row))
            if row['type'] == 'hardlink':
                need(not pending, 'Hardlink cannot be a parent directory')
                parts = []; pending = row['destination'].split('/'); hard = True
            else:
                link = row['link']
                if link.startswith('/'):
                    parts = []; link = link[1:]
                pending = (link.split('/') if link else []) + pending
            continue
        if pending:
            need(row['type'] == 'directory', 'Only directory members can contain archive paths')
            parts.append(part)
        else:
            need(row['type'] == 'file', 'Required target or alias must end at a regular file')
            chain.append(dict(row))
            return dict(requested=name, chain=chain, terminal=path,
                        size=row['size'], sha256=row['sha256'], link_hops=hops)
    raise ValueError('Required target or alias cannot end at archive root')


def verify(read, guard, targets, aliases=None):
    """Scan one full USTAR stream and resolve fixed targets/required aliases.

    targets: caller-fixed target identifier -> archive member name (1..12).
    aliases: required archive alias name -> target identifier (0..32).
    read(n) may short-read, returns b'' only at EOF, and must itself be bounded
    against blocking; the caller authenticates the archive and producer exit.
    """
    need(type(targets) is dict and 1 <= len(targets) <= 12
         and all(type(k) is str and 0 < len(k) <= 4096 for k in targets), 'Bounded fixed target mapping')
    aliases = {} if aliases is None else aliases
    need(type(aliases) is dict and len(aliases) <= 32, 'Bounded required alias mapping')
    wanted = {key: member_name(value) for key, value in targets.items()}
    required = {member_name(name): key for name, key in aliases.items()}
    need(len(set(wanted.values())) == len(wanted) and len(required) == len(aliases)
         and all(type(key) is str and key in wanted for key in required.values()), 'Unique fixed targets and aliases')
    stream = Stream(read, guard); table = {}; payload_bytes = 0
    counts = dict(file=0, directory=0, symlink=0, hardlink=0)
    while True:
        block = stream.exact(BLOCK)
        if block == bytes(BLOCK):
            need(stream.exact(BLOCK) == bytes(BLOCK), 'Two consecutive zero USTAR end blocks')
            while True:
                extra = stream.take(CHUNK)
                need(not any(extra), 'Only zero bytes after the two USTAR end blocks')
                if not extra:
                    break
            need(stream.bytes % BLOCK == 0, 'Complete block-aligned USTAR stream through EOF')
            break
        need(len(table) < MAX_HEADERS, 'At most 10000 USTAR member headers')
        row = header(block); name = row['name']
        need(name not in table, 'No duplicate normalized archive members')
        if row['type'] == 'file':
            h = hashlib.sha256(); left = row['size']
            while left:
                raw = stream.exact(min(CHUNK, left)); h.update(raw); left -= len(raw)
            row['sha256'] = h.hexdigest(); payload_bytes += row['size']
            padding = (-row['size']) % BLOCK
            need(not padding or not any(stream.exact(padding)), 'Zero regular-member block padding')
        table[name] = row; counts[row['type']] += 1
    for name in table:
        guard(); parts = name.split('/')
        for end in range(1, len(parts)):
            parent = table.get('/'.join(parts[:end]))
            need(parent is None or parent['type'] == 'directory', 'No archive member beneath a non-directory entry')
    resolved = {}
    for key, name in wanted.items():
        guard(); resolved[key] = resolve(name, table)
    alias_results = {}
    for name, key in required.items():
        guard(); result = resolve(name, table)
        need(result['terminal'] == resolved[key]['terminal'], 'Required alias resolves to its exact target member')
        alias_results[name] = dict(result, target=key)
    descriptor_hash = hashlib.sha256()
    for name in sorted(table):
        guard()
        descriptor_hash.update((json.dumps(table[name], sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    guard()
    return dict(schema='dory_ustar_member_verification_v1', complete=True,
                stream_bytes=stream.bytes, stream_sha256=stream.hash.hexdigest(), headers=len(table),
                member_counts=counts, regular_payload_bytes=payload_bytes,
                descriptor_table_sha256=descriptor_hash.hexdigest(), targets=resolved, aliases=alias_results,
                extracted_files=0, filesystem_resolution_used=False, archive_or_producer_authenticated_here=False,
                package_content_compared_to_host=False, native_ABI_qualified=False, full_model_qualified=False)
