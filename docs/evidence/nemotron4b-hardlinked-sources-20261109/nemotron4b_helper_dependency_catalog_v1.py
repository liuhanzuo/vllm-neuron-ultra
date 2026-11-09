"""Source-only installed distribution catalog and conservative import edges.

Never import inspected packages, execute startup hooks, read native payloads,
or admit a runtime. RECORD proves agreement with the pinned installed metadata,
not publisher authenticity. An inventory is not a complete runtime closure.
"""
import base64
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat

from nemotron4b_helper_import_graph_v1 import inspect_imports, require

FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical ordinary path')
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1, 'Ordinary single-link file')
    return {key: getattr(value, key) for key in FIELDS}


def source(path, wanted=None):
    """Only bounded Python/package metadata text; never native/model payloads."""
    path = Path(path)
    require(path.suffix in ('.py', '.pth', '.json') or path.name in ('RECORD', 'METADATA', 'pyvenv.cfg'),
            'Source and metadata allowlist')
    before = identity(path)
    require(0 <= before['st_size'] <= 8*2**20, 'Bounded source/metadata')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require({k: getattr(os.fstat(fd), k) for k in FIELDS} == before, 'Opened exact original file')
        parts, size = [], 0
        while True:
            raw = os.read(fd, min(65536, 8*2**20+1-size))
            if not raw: break
            parts.append(raw); size += len(raw)
            require(size <= 8*2**20, 'Source grew beyond cap')
        raw = b''.join(parts)
        require({k: getattr(os.fstat(fd), k) for k in FIELDS} == before == identity(path)
                and len(raw) == before['st_size'], 'Stable complete source read')
    finally:
        os.close(fd)
    raw.decode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    require(wanted is None or digest == wanted, 'Original source pin changed')
    return raw, dict(path=str(path), sha256=digest, bytes=len(raw), fullstat=before)


def parse_record(raw):
    result = {}
    for row in csv.reader(io.StringIO(raw.decode('utf-8'))):
        require(len(row) == 3 and row[0] and row[0] not in result, 'Unique complete RECORD rows')
        # RECORD may legitimately include ../../../bin scripts. Such rows are
        # retained as metadata and must never be joined to a selected file path.
        require('\x00' not in row[0] and '\\' not in row[0], 'Unambiguous RECORD path')
        result[row[0]] = tuple(row[1:])
    return result


def record_digest(records, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and str(name) == relative,
            'Selected source stays under installed site')
    require(relative in records, 'Selected source absent from distribution RECORD')
    encoded, size = records[relative]
    require(encoded.startswith('sha256=') and size.isdecimal(), 'SHA256 and size in RECORD')
    token = encoded.split('=', 1)[1]
    try:
        digest = base64.b64decode(token+'='*((-len(token)) % 4), altchars=b'-_', validate=True)
    except Exception as error:
        raise ValueError('Invalid RECORD digest') from error
    require(len(digest) == 32 and base64.urlsafe_b64encode(digest).decode().rstrip('=') == token,
            'Canonical SHA256 RECORD digest')
    return digest.hex(), int(size)


def verified_source(path, site, records, previous=None):
    relative = Path(path).relative_to(site).as_posix()
    digest, size = record_digest(records, relative)
    raw, ref = source(path, digest)
    require(size == len(raw), 'Original RECORD source size')
    require(previous is None or previous == digest, 'Prior source pin preserved')
    return raw, dict(**ref, installed_RECORD_verified=True)


def module_name(relative):
    p = PurePosixPath(relative)
    require(p.suffix == '.py' and '..' not in p.parts and not p.is_absolute(), 'Python module path')
    parts = p.parts[:-1] if p.name == '__init__.py' else (*p.parts[:-1], p.stem)
    require(parts and all(part.isidentifier() for part in parts), 'Ordinary module name')
    return '.'.join(parts), p.name == '__init__.py'


def dependency_graph(entry, modules, namespaces, native_modules):
    """Union of known package module-body edges, including implicit parents.

    Unknown from-import members can be attributes; report them only when the
    containing module itself is unknown. Native leaves are boundaries, not
    executed or hashed here. Deferred functions/dynamic calls remain explicit.
    """
    require(not (set(modules) & set(namespaces) or set(modules) & set(native_modules)
                 or set(namespaces) & set(native_modules)), 'Disjoint module kinds')
    known = set(modules) | set(namespaces) | set(native_modules)
    pending, visited, edges, boundaries, scans = ['<entry>'], set(), [], [], {}

    def add(importer, target, line, reason):
        if target not in known:
            boundaries.append(dict(importer=importer, target=target, line=line, reason=reason))
            return
        parts = target.split('.')
        for n in range(1, len(parts)+1):
            parent = '.'.join(parts[:n])
            require(parent in known, 'Explicit implicit-parent package: '+parent)
            edge = dict(importer=importer, target=parent, line=line,
                        reason=reason if parent == target else 'implicit_parent_package')
            if edge not in edges: edges.append(edge)
            if parent not in visited and parent not in pending: pending.append(parent)

    while pending:
        name = pending.pop(0)
        if name in visited: continue
        visited.add(name)
        if name in namespaces or name in native_modules: continue
        row = dict(source=entry, is_package=False) if name == '<entry>' else modules[name]
        scan = inspect_imports(row['source']); scans[name] = scan
        package = '' if name == '<entry>' else name if row['is_package'] else name.rpartition('.')[0]
        for item in scan['module_body_imports']:
            if item['kind'] == 'import':
                for target in item['names']: add(name, target, item['line'], 'explicit_import')
                continue
            if item['level']:
                parts = package.split('.') if package else []
                require(len(parts) >= item['level'], 'Relative import stays inside package')
                stem = '.'.join(parts[:len(parts)-item['level']+1])
                target = stem+('.'+item['module'] if item['module'] else '')
            else:
                target = item['module'] or ''
            require(target, 'Explicit import target')
            add(name, target, item['line'], 'from_module')
            for member in item['names']:
                candidate = target+'.'+member
                if member == '*':
                    boundaries.append(dict(importer=name, target=candidate, line=item['line'],
                                           reason='star_exports_unresolved'))
                elif candidate in known: add(name, candidate, item['line'], 'from_submodule_candidate')
    return dict(reachable_python_modules=sorted(visited & set(modules)),
        reachable_namespace_packages=sorted(visited & set(namespaces)),
        reachable_native_boundaries=sorted(visited & set(native_modules)),
        import_edges=edges, unresolved_import_boundaries=boundaries, module_scans=scans,
        complete_runtime_dependency_closure_proven=False, installed_code_executed=False)


if __name__ == '__main__':
    raise SystemExit('Source-only library; no runtime admission or helper execution')
