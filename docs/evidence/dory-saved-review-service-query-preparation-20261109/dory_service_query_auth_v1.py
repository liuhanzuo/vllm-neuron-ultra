"""Source preparation: authenticate the proposed service-query dependencies.

This does not admit a saved reviewer. The future executor must authenticate this
module and the manifest, retain the original guard and budgets, and account for
the actual previous review's terminal history. No model or raw archive is read.
"""
import hashlib
import importlib.machinery as machinery
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

MAX_FILE = 8 * 2**20
MAX_TOTAL = 64 * 2**20
ENTRY = 'dory_saved_review_service_query_v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(info):
    return tuple(getattr(info, k) for k in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def authenticated_bytes(path, pin):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            'Canonical dependency path required')
    require(type(pin['bytes']) is int and 0 < pin['bytes'] <= MAX_FILE,
            'Bounded dependency size required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes']
                and (before.st_dev, before.st_ino) == (pin['device'], pin['inode']),
                'Exact dependency file identity required')
        raw = stream.read(pin['bytes'] + 1)
        after = os.fstat(stream.fileno())
    require(identity(before) == identity(after) == identity(path.lstat())
            and len(raw) == pin['bytes']
            and hashlib.sha256(raw).hexdigest() == pin['sha256'],
            'Exact stable dependency bytes required')
    return raw


def file_mappings():
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        row = line.split(maxsplit=5)
        if len(row) < 6 or not row[5].startswith('/'):
            continue
        require(not row[5].endswith(' (deleted)'), 'No deleted native mapping')
        path = Path(row[5])
        info = path.lstat()
        major, minor = row[3].split(':')
        expected = dict(device=os.makedev(int(major, 16), int(minor, 16)),
                        inode=int(row[4]))
        require(path.resolve() == path and stat.S_ISREG(info.st_mode)
                and expected == dict(device=info.st_dev, inode=info.st_ino),
                'Exact current native mapping identity required')
        require(str(path) not in result or result[str(path)] == expected,
                'Stable native mapping identity required')
        result[str(path)] = expected
    return result


def check_mapping_delta(before, after, allowed):
    require(all(after.get(p) == row for p, row in before.items()),
            'Original native mappings must be retained')
    delta = {p: row for p, row in after.items() if p not in before}
    require(all(allowed.get(p) == row for p, row in delta.items()),
            'Unreviewed native dependency mapping')
    require(all(after.get(p) == row for p, row in allowed.items()),
            'All expected query native dependencies must be mapped')
    return delta


class PinnedSourceLoader(machinery.SourceFileLoader):
    """Compile authenticated bytes directly; never consult a bytecode cache."""
    def __init__(self, name, path, pin, loaded):
        super().__init__(name, path)
        self.pin, self.loaded = pin, loaded

    def get_code(self, fullname):
        raw = authenticated_bytes(self.path, self.pin)
        code = compile(raw, self.path, 'exec', dont_inherit=True, optimize=0)
        self.loaded[fullname] = self.path
        return code


class PinnedFinder:
    def __init__(self, pins):
        self.pins, self.loaded = pins, {}

    def find_spec(self, fullname, path=None, target=None):
        for finder in (machinery.BuiltinImporter, machinery.FrozenImporter):
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                return spec
        spec = machinery.PathFinder.find_spec(fullname, path, target)
        require(spec is not None and spec.origin in self.pins,
                'Import outside reviewed dependency files: ' + fullname)
        pin = self.pins[spec.origin]
        authenticated_bytes(spec.origin, pin)
        if type(spec.loader) is machinery.SourceFileLoader:
            spec.loader = PinnedSourceLoader(fullname, spec.origin, pin, self.loaded)
        else:
            require(type(spec.loader) is machinery.ExtensionFileLoader,
                    'Only authenticated source and extension imports permitted')
            self.loaded[fullname] = spec.origin
        return spec


class AuthenticatedServiceQuery:
    def __init__(self, observation, *, before_check):
        require(callable(before_check), 'Original startup guard required')
        require(sys.dont_write_bytecode and sys.flags.optimize == 0,
                'No bytecode writes and original assertions enabled')
        require(len(list(Path('/proc/self/task').iterdir())) == 1,
                'Import authentication requires a single-threaded owner')
        require(not any(n == ENTRY or n == 'dbus' or n.startswith(('dbus.', '_dbus'))
                        for n in sys.modules), 'Query dependencies must be freshly imported')
        require(observation['schema'] == 'dory_service_query_dependency_observation_v1'
                and observation['complete_observation'] is True,
                'Exact reviewed dependency observation required')
        self.pins = observation['source_and_binary_pins']
        require(0 < len(self.pins) <= 128
                and sum(pin['bytes'] for pin in self.pins.values()) <= MAX_TOTAL,
                'Bounded complete query dependency closure')
        self.allowed_maps = observation['new_native_file_mappings']
        require(all(p in self.pins and row == {k: self.pins[p][k] for k in ('device', 'inode')}
                    for p, row in self.allowed_maps.items()),
                'Every expected native mapping has an exact byte pin')
        self.closed, self.poisoned, self.query = False, False, None
        self.restored = False
        self.old_finders = list(sys.meta_path)
        self.before_modules = set(sys.modules)
        self.before_maps = file_mappings()
        self.finder = PinnedFinder(self.pins)
        self.startup_file_count = 0
        self.startup_bytes = 0
        self.final_audit = None
        for path, pin in self.pins.items():
            before_check()
            self.startup_bytes += len(authenticated_bytes(path, pin))
            self.startup_file_count += 1
        try:
            sys.meta_path[:] = [self.finder]
            candidate = __import__(ENTRY)
            self.query = candidate.connect_fixed_user_bus()
            self.audit()
        except BaseException:
            self.poisoned = True
            self.close()
            raise

    def audit(self):
        require(sys.meta_path == [self.finder], 'Query import policy must remain installed')
        delta = check_mapping_delta(self.before_maps, file_mappings(), self.allowed_maps)
        for name in set(sys.modules) - self.before_modules:
            module = sys.modules[name]
            path = getattr(module, '__file__', None)
            if path is not None:
                require(path in self.pins and self.finder.loaded.get(name) == path,
                        'Every new file-backed module used an authenticated loader')
        return dict(new_native_files=len(delta), authenticated_imports=dict(self.finder.loaded))

    def check(self, excluded=()):
        require(not self.closed and not self.poisoned, 'Closed or failed query cannot resume')
        try:
            require(sys.meta_path == [self.finder], 'Query import policy must remain installed')
            return self.query.check(excluded)
        except BaseException:
            self.poisoned = True
            raise

    @property
    def requests(self):
        return 0 if self.query is None else self.query.requests

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.query is not None:
                self.query.close()
            self.final_audit = self.audit()
            for path, pin in self.pins.items():
                authenticated_bytes(path, pin)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            sys.meta_path[:] = self.old_finders
            self.restored = True


def read_observation(path, sha256):
    path = Path(path)
    info = path.lstat()
    raw = authenticated_bytes(path, dict(bytes=info.st_size, device=info.st_dev,
                                        inode=info.st_ino, sha256=sha256))
    return json.loads(raw)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; not an admitted saved reviewer')
