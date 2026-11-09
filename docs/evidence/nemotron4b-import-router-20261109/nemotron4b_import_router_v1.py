"""Reviewed import dispatch core for the future gated Nano4B helper context.

Keeps caller-approved finder order and absence semantics. It does not install
itself, authenticate the interpreter or preloaded modules, authorize arbitrary
finder changes, or establish the full real SDK runtime closure. A real outer
context must bind/register actual startup/six finders and audit native mappings.
"""
import copy
import importlib.machinery as machinery
import json
from pathlib import Path
import sys

from nemotron4b_helper_dependency_catalog_v1 import source
from nemotron4b_process_handshake_v1 import require


class ProvenanceError(RuntimeError):
    """Not ImportError: provenance failure must not select an optional fallback."""


class PinnedSourceLoader(machinery.SourceFileLoader):
    def __init__(self, name, path, router, reference):
        super().__init__(name, path)
        self.router, self.reference = router, reference

    def get_code(self, fullname):
        self.router.check()
        raw = self.router.read_source(self.path, self.reference)
        # Do not load even an apparently valid preexisting bytecode file.
        return compile(raw, self.path, 'exec', dont_inherit=True, optimize=0)

    def get_source(self, fullname):
        self.router.check()
        return self.router.read_source(self.path, self.reference).decode('utf-8')

    def exec_module(self, module):
        self.router.check()
        try:
            super().exec_module(module)
            self.router.check()
            self.router.record('source_executed', module.__name__, self.path)
        except BaseException as error:
            self.router.fail(error)
            raise


class CheckedDelegatingLoader:
    """Delegate native/custom module semantics under exact caller authorization."""
    def __init__(self, original, fullname, router, kind, before, after):
        self.original, self.fullname, self.router = original, fullname, router
        self.kind, self.before, self.after = kind, before, after

    def create_module(self, spec):
        self.router.check()
        try:
            self.before(self.fullname, self.original, spec)
            self.router.check()
            module = self.original.create_module(spec)
            self.after(self.fullname, self.original, module, 'create')
            self.router.check()
            return module
        except BaseException as error:
            self.router.fail(error)
            raise

    def exec_module(self, module):
        self.router.check()
        try:
            self.before(self.fullname, self.original, module.__spec__)
            self.router.check()
            self.original.exec_module(module)
            self.after(self.fullname, self.original, module, 'exec')
            self.router.check()
            self.router.record(self.kind+'_executed', self.fullname,
                               getattr(module, '__file__', None))
        except BaseException as error:
            self.router.fail(error)
            raise

    def __getattr__(self, name):
        # Needed by source/custom package consumers; the exact provider remains
        # the caller's responsibility, not a generic custom-loader allowance.
        return getattr(self.original, name)


class ImportRouter:
    def __init__(self, *, finders, modules, namespaces, guard, authorize_finder,
                 authorize_builtin, authorize_native, authorize_custom):
        require(type(finders) in (list, tuple) and finders, 'Explicit ordered finder objects')
        require(not set(modules) & set(namespaces), 'Disjoint file and namespace modules')
        require(all(callable(f) for f in (guard, authorize_finder, authorize_builtin,
                    authorize_native, authorize_custom)), 'Complete caller origin hooks')
        self.finders = tuple(finders)
        # Snapshot metadata; callbacks cannot silently rebind this catalog.
        self.modules = copy.deepcopy(modules)
        self.namespaces = copy.deepcopy(namespaces)
        self.guard, self.authorize_finder = guard, authorize_finder
        self.authorize_builtin = authorize_builtin
        self.authorize_native, self.authorize_custom = authorize_native, authorize_custom
        self.events = []
        self.poisoned = False
        self.failure = None
        self.closed = False

    def fail(self, error):
        self.poisoned = True
        if self.failure is None:
            self.failure = dict(type=type(error).__name__, message=str(error)[:2048])

    def check(self):
        if self.poisoned or self.closed:
            raise ProvenanceError('Failed or closed import router cannot continue')
        try:
            self.guard()
        except BaseException as error:
            self.fail(error)
            raise

    def record(self, kind, name, origin):
        require(len(self.events) < 4096, 'Bounded import provenance events')
        self.events.append(dict(kind=kind, name=name, origin=origin))

    def read_source(self, path, reference):
        try:
            self.check()
            require(reference['kind'] == 'source' and reference['path'] == path,
                    'Exact source origin record')
            raw, actual = source(path, reference['sha256'])
            require(actual['bytes'] == reference['bytes'] and actual['fullstat'] == reference['fullstat'],
                    'Exact source identity and size')
            self.check()
            return raw
        except BaseException as error:
            self.fail(error)
            raise

    def namespace(self, fullname, spec):
        expected = self.namespaces.get(fullname)
        require(expected is not None and spec.origin is None and spec.loader is None,
                'Exact explicit namespace origin')
        require(list(spec.submodule_search_locations or ()) == expected,
                'Exact namespace search locations and ordering')
        for name in expected:
            path = Path(name)
            require(path.is_dir() and path.resolve() == path
                    and not any(p.is_symlink() for p in (path, *path.parents))
                    and not (path/'__init__.py').exists(), 'Original namespace directories')
        self.record('namespace', fullname, list(expected))
        return spec

    def validate_spec(self, fullname, spec, finder):
        require(spec.name == fullname, 'Exact requested import name')
        if spec.origin in ('built-in', 'frozen'):
            require(spec.loader in (machinery.BuiltinImporter, machinery.FrozenImporter),
                    'Original interpreter loader')
            self.authorize_builtin(fullname, spec, finder)
            self.record('interpreter', fullname, spec.origin)
            return spec
        if spec.loader is None:
            return self.namespace(fullname, spec)
        row = self.modules.get(fullname)
        if type(spec.loader) is machinery.SourceFileLoader:
            require(row is not None and row['kind'] == 'source'
                    and spec.origin == row['path'] and spec.has_location,
                    'Source import outside exact reviewed origin')
            expected_locations = [str(Path(row['path']).parent)] if row['package'] else None
            actual_locations = (None if spec.submodule_search_locations is None
                                else list(spec.submodule_search_locations))
            require(actual_locations == expected_locations, 'Exact source package search path')
            self.read_source(row['path'], row)
            spec.loader = PinnedSourceLoader(fullname, row['path'], self, row)
            spec.cached = None
            self.record('source_selected', fullname, row['path'])
            return spec
        if type(spec.loader) is machinery.ExtensionFileLoader:
            require(row is not None and row['kind'] == 'native' and spec.origin == row['path'],
                    'Native import outside exact reviewed origin')
            before, after = self.authorize_native(fullname, spec, copy.deepcopy(row))
            kind = 'native'
        else:
            # A custom loader can resolve aliases without a filesystem origin.
            # The caller must bind this exact provider/name and validate results.
            before, after = self.authorize_custom(fullname, spec, finder)
            kind = 'custom'
        require(callable(before) and callable(after)
                and callable(getattr(spec.loader, 'create_module', None))
                and callable(getattr(spec.loader, 'exec_module', None)),
                'Explicit approved provider lifecycle callbacks')
        spec.loader = CheckedDelegatingLoader(spec.loader, fullname, self, kind, before, after)
        self.record(kind+'_selected', fullname, spec.origin)
        return spec

    def find_spec(self, fullname, path=None, target=None):
        self.check()
        try:
            for finder in self.finders:
                # Finder code can itself import or execute; authorize first.
                self.authorize_finder(finder)
                self.check()
                spec = finder.find_spec(fullname, path, target)
                self.check()
                if spec is not None:
                    return self.validate_spec(fullname, spec, finder)
            # Genuine absence remains absent. Import machinery raises its normal
            # ModuleNotFoundError so original optional imports can fall back.
            self.record('absent', fullname, None)
            return None
        except BaseException as error:
            self.fail(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit, ProvenanceError)):
                raise
            raise ProvenanceError('Import provenance rejected '+fullname+': '+str(error)) from error

    def finish(self):
        """Outer context calls after it verifies no finder/module/native bypass."""
        self.check()
        self.closed = True
        return dict(events=copy.deepcopy(self.events), poisoned=False,
            complete_runtime_dependency_closure_proven=False,
            router_dispatch_completed=True, runtime_admitted=False,
            full_model_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; needs authenticated outer import context and exact SDK bootstrap')
