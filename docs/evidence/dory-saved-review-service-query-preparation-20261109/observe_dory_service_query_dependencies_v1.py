"""Read-only import/dependency observation for a future service-query transport.

No model, raw tensor, service mutation, runtime admission or current-reviewer
change. Post-import hashes describe observed files; they are not pre-import
runtime authentication or an exhaustive proof of every lazy/error import path.
"""
import datetime
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import stat
import sys

Q = Path(__file__).resolve().parent
PREFIX = Q / 'dory-service-query-dependencies-no-bytecode-cache-v1'
assert sys.dont_write_bytecode and sys.flags.optimize == 0
assert not PREFIX.exists() and not any(n == 'dbus' or n.startswith(('dbus.', '_dbus')) for n in sys.modules)
sys.pycache_prefix = str(PREFIX)
ENTRY = Q / 'dory_saved_review_service_query_v1.py'
assert hashlib.sha256(ENTRY.read_bytes()).hexdigest() == '9e02f2dc025b263628c08f3fda0ddb2ba705acb0c6de2e41984cc9f210470a62'
import dory_saved_review_service_query_v1 as candidate


def file_mappings():
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith('/'):
            continue
        assert not fields[5].endswith(' (deleted)')
        path = Path(fields[5])
        info = path.stat()
        major, minor = fields[3].split(':')
        assert info.st_dev == os.makedev(int(major, 16), int(minor, 16))
        assert info.st_ino == int(fields[4])
        assert stat.S_ISREG(info.st_mode) and path.resolve() == path
        identity = dict(device=info.st_dev, inode=info.st_ino)
        assert str(path) not in result or result[str(path)] == identity
        result[str(path)] = identity
    return result


def main():
    before_modules = set(sys.modules)
    before_maps = file_mappings()
    query = candidate.connect_fixed_user_bus()
    try:
        assert query.check() == []
        after_maps = file_mappings()
        new_modules = {}
        for name in sorted(set(sys.modules) - before_modules):
            module = sys.modules[name]
            filename = getattr(module, '__file__', None)
            if filename is None:
                new_modules[name] = dict(kind='builtin_or_frozen', origin=getattr(module.__spec__, 'origin', None))
                continue
            path = Path(filename)
            assert path.is_absolute() and path.resolve() == path
            loader = module.__loader__
            if isinstance(loader, importlib.machinery.SourceFileLoader):
                assert path.suffix == '.py'
                cached = Path(module.__cached__)
                assert cached.is_relative_to(PREFIX) and not cached.exists()
                kind = 'source'
            else:
                assert isinstance(loader, importlib.machinery.ExtensionFileLoader)
                assert str(path) in after_maps
                cached, kind = None, 'extension'
            new_modules[name] = dict(kind=kind, path=str(path),
                                     cached_path=None if cached is None else str(cached))
        new_maps = {p: info for p, info in after_maps.items() if p not in before_maps}
        import dbus
        package = Path(dbus.__file__).parent
        files = set(package.rglob('*.py')) | {Path(p) for p in new_maps}
        files |= {Path(row['path']) for row in new_modules.values() if 'path' in row}
        files.add(ENTRY)
        total, pins = 0, {}
        for path in sorted(files):
            before = path.stat()
            assert stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 8 * 2**20
            total += before.st_size
            assert total <= 64 * 2**20
            raw = path.read_bytes()
            after = path.stat()
            keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            assert len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in keys)
            pins[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                   device=before.st_dev, inode=before.st_ino)
        assert file_mappings() == after_maps
        assert query.check() == [] and query.requests == 2
    finally:
        query.close()
    assert query.closed and not PREFIX.exists()
    result = dict(schema='dory_service_query_dependency_observation_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        read_only=True, complete_observation=True, new_module_observations=new_modules,
        new_native_file_mappings=new_maps, before_native_file_mappings=before_maps,
        after_native_file_mappings=after_maps, source_and_binary_pins=pins,
        source_and_binary_bytes_hashed=total, bytecode_prefix=str(PREFIX),
        bytecode_prefix_absent=True, private_connection_closed=True,
        service_mutated=False, running_reviewer_modified=False,
        raw_tensor_bytes_read=0, runtime_admitted=False,
        pre_import_authentication_performed=False,
        exhaustive_lazy_or_error_import_paths_verified=False,
        full_runtime_environment_review_complete=False,
        full_model_qualified=False, new_speedup_claim=False)
    path = Q / 'dory-service-query-dependency-observation-20261109-v1.json'
    raw = (json.dumps(result, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(result=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()),
        new_modules=len(new_modules), new_native_files=len(new_maps),
        hashed_files=len(pins), bytes_hashed=total,
        native_files=[Path(p).name for p in new_maps])))


if __name__ == '__main__':
    main()
