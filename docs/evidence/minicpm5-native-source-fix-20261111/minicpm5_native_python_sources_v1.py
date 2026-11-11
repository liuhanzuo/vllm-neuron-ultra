"""Selected editable Transformers Python inventory and exact loaded-path errors.

Stdlib only. This module does not import a native package or touch model data,
services, locks, execution admission, or terminal records. A complete package
inventory is intentional: Transformers uses lazy and conditional imports, so a
static walk of direct imports is not a complete Python dependency closure.
The original source map is retained; unrelated installed packages are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'research/transformers-reference/src/transformers'
MAX_PACKAGE_FILES = 8192
MAX_SOURCE_BYTES = 8 * 2**20


class SourceClosureError(RuntimeError):
    def __init__(self, reason, *, module=None, path=None, paths=None):
        report = dict(schema='minicpm5_native_source_closure_error_v1', reason=reason)
        if module is not None:
            report['module'] = str(module)
        if path is not None:
            report['path'] = str(path)
        if paths is not None:
            ordered = sorted(paths)
            report['paths_count'] = len(ordered)
            report['first_paths'] = ordered[:4]
        self.source_closure_error = report
        super().__init__('Native Python source authentication: ' + json.dumps(report, sort_keys=True))


def python_paths(package_root=PACKAGE):
    """Read only directory metadata; never import or open a package payload."""
    package_root = Path(package_root)
    if not package_root.is_absolute() or package_root.resolve() != package_root:
        raise SourceClosureError('noncanonical_package_root', path=package_root)
    for parent in (package_root, *package_root.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise SourceClosureError('non_directory_or_symlink_ancestry', path=parent)
    paths = []
    def fail_walk(error):
        raise error
    for folder, directories, files in os.walk(package_root, followlinks=False, onerror=fail_walk):
        directories.sort()
        for name in directories:
            path = Path(folder) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise SourceClosureError('non_directory_or_symlink_package_child', path=path)
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            path = Path(folder) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SourceClosureError('non_regular_or_linked_python_source', path=path)
            if info.st_size > MAX_SOURCE_BYTES:
                raise SourceClosureError('python_source_extent_exceeded', path=path)
            paths.append(str(path))
            if len(paths) > MAX_PACKAGE_FILES:
                raise SourceClosureError('package_python_file_bound_exceeded', path=package_root)
    if str(package_root / '__init__.py') not in paths:
        raise SourceClosureError('package_initializer_missing', path=package_root)
    return tuple(sorted(paths))


def collect_package_sources(read_source, guard, *, package_root=PACKAGE):
    """Prepare additive pins using the caller's original stable text transport.

    This separate package inventory does not inflate or replace the independently
    bounded local orchestration import graph. Existing pins must be merged with
    the original collision-rejecting merge, never overwritten.
    """
    guard()
    paths = python_paths(package_root)
    result = {}
    for path in paths:
        guard()
        raw = read_source(path, None)
        if type(raw) is not bytes or len(raw) > MAX_SOURCE_BYTES:
            raise SourceClosureError('invalid_complete_python_source', path=path)
        raw.decode('utf-8')
        result[path] = hashlib.sha256(raw).hexdigest()
    if python_paths(package_root) != paths:
        raise SourceClosureError('package_inventory_changed', path=package_root)
    guard()
    return result


def require_package_coverage(plan, guard, *, package_root=PACKAGE):
    """Complete filename coverage before the original full-source hash phase.

    This is not content authentication. RuntimeContext.verify_sources retains
    responsibility for hashing every original and additive pin before imports.
    """
    guard()
    paths = set(python_paths(package_root))
    prefix = str(package_root) + '/'
    pinned = {path for path in plan['source_sha256']
              if path.startswith(prefix) and path.endswith('.py')}
    if paths - pinned:
        raise SourceClosureError('package_python_sources_unpinned', paths=paths - pinned)
    if pinned - paths:
        raise SourceClosureError('pinned_package_python_sources_absent', paths=pinned - paths)
    for path in sorted(paths):
        wanted = plan['source_sha256'][path]
        if type(wanted) is not str or re.fullmatch('[a-f0-9]{64}', wanted) is None:
            raise SourceClosureError('invalid_python_source_digest', path=path)
    guard()
    return dict(package_root=str(package_root), python_files=len(paths),
                complete_filename_coverage=True, content_authentication_performed=False,
                native_import_performed=False)


def verify_loaded(plan, guard, read_source, *, modules=None, root=ROOT):
    """Keep the original all-loaded-local-source gate, adding exact diagnostics.

    The per-source guard and full hash check are retained. No broad suppression,
    exclusions, synthetic pinning, or native import takes place here.
    """
    snapshot = tuple((sys.modules if modules is None else modules).items())
    prefix = str(root) + '/'
    verified = 0
    for name, module in snapshot:
        # Avoid running lazy module __getattr__ while observing its source.
        namespace = getattr(module, '__dict__', {})
        declared = namespace.get('__file__')
        if not declared:
            continue
        declared = str(declared)
        resolved = str(Path(declared).resolve())
        if not (declared.startswith(prefix) or resolved.startswith(prefix)):
            continue
        guard()
        if not resolved.startswith(prefix):
            raise SourceClosureError('loaded_local_source_escaped_root', module=name, path=resolved)
        wanted = plan['source_sha256'].get(resolved)
        if wanted is None:
            raise SourceClosureError('loaded_local_source_unpinned', module=name, path=resolved)
        try:
            raw = read_source(resolved, wanted)
        except Exception as error:
            raise SourceClosureError('loaded_local_source_read_or_authentication_failed',
                                     module=name, path=resolved) from error
        if hashlib.sha256(raw).hexdigest() != wanted:
            raise SourceClosureError('loaded_local_source_digest_changed', module=name, path=resolved)
        verified += 1
    return dict(loaded_local_modules_verified=verified, native_import_performed=False)
