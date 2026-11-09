"""Module-origin ledger for the dedicated Nano4B helper import context.

Actual preloaded/interpreter/native/custom-alias authentication remains an
explicit caller policy. This checks registered module objects and import
metadata against that policy and the selected source loader; it is not an
attestation of arbitrary Python globals or an adversarial Python sandbox.
"""
import copy
import importlib.machinery as machinery
from importlib._bootstrap_external import _NamespaceLoader
from pathlib import Path
import sys
import types

from nemotron4b_import_router_v3 import ImportRouter, PinnedSourceLoader, CheckedDelegatingLoader
from nemotron4b_process_handshake_v1 import require


def metadata(module):
    """Freeze meaningful import metadata without invoking module __getattr__."""
    if module is None:
        return None
    require(isinstance(module, types.ModuleType), 'sys.modules entry is a module or approved None')
    values = vars(module)
    spec = values.get('__spec__')
    if spec is None:
        frozen_spec = None
    else:
        require(type(spec) is machinery.ModuleSpec, 'Original module spec type')
        locations = spec.submodule_search_locations
        frozen_spec = (id(spec), spec.name, spec.origin, id(spec.loader),
                       None if locations is None else tuple(locations))
    path = values.get('__path__')
    return (values.get('__name__'), values.get('__file__'), values.get('__package__'),
            id(values.get('__loader__')), frozen_spec,
            None if path is None else tuple(path))


class ModuleOriginAudit:
    def __init__(self, router, *, preloaded, authenticate_preloaded,
                 interpreter_modules, authenticate_interpreter,
                 authenticate_native, authenticate_alias):
        require(type(router) is ImportRouter, 'Exact source-alias-aware router v3')
        require(type(preloaded) is dict and len(preloaded) <= 4096,
                'Explicit bounded preloaded module objects')
        require(all(callable(f) for f in (authenticate_preloaded,
            authenticate_interpreter, authenticate_native, authenticate_alias)),
            'Complete actual origin policy callbacks')
        self.router = router
        self.preloaded = dict(preloaded)
        self.authenticate_preloaded = authenticate_preloaded
        self.interpreter_modules = copy.deepcopy(interpreter_modules)
        self.authenticate_interpreter = authenticate_interpreter
        self.authenticate_native = authenticate_native
        # Return True only for an exact authenticated alias/synthetic provider
        # module. False means ordinary source/interpreter/native checks apply.
        self.authenticate_alias = authenticate_alias
        self.preloaded_metadata = {}
        self.bound = False
        self.in_audit = False
        self.seen = {}
        self.events = []

    def bind_preloaded(self):
        """Before installation, authenticate every expected original module."""
        require(not self.bound, 'Preloaded origin binding is one-use')
        require(set(sys.modules) == set(self.preloaded), 'Complete initial preloaded name set')
        for name, module in self.preloaded.items():
            require(sys.modules[name] is module, 'Exact initial preloaded module identity')
            self.authenticate_preloaded(name, module)
            self.preloaded_metadata[name] = metadata(module)
        self.bound = True

    def source_module(self, name, module, row):
        values = vars(module)
        spec = values.get('__spec__')
        require(type(spec) is machinery.ModuleSpec and spec.name == name
                and spec.origin == row['path'], 'Exact source module spec name and origin')
        loader = spec.loader
        require(type(loader) is PinnedSourceLoader and loader.router is self.router
                and loader.reference == row and loader.path == row['path']
                and values.get('__loader__') is loader, 'Selected authenticated source loader')
        require(values.get('__name__') == name and values.get('__file__') == row['path'],
                'Exact source module name and original file')
        locations = [str(Path(row['path']).parent)] if row['package'] else None
        require((None if spec.submodule_search_locations is None else list(spec.submodule_search_locations)) == locations,
                'Original source package spec path')
        require((None if '__path__' not in values else list(values['__path__'])) == locations,
                'Original source module package path')
        package = name if row['package'] else name.rpartition('.')[0]
        require(values.get('__package__') == package, 'Original source package name')

    def namespace_module(self, name, module, locations):
        values = vars(module)
        spec = values.get('__spec__')
        require(type(spec) is machinery.ModuleSpec and spec.name == name and spec.origin is None
                and type(spec.loader) is _NamespaceLoader and values.get('__loader__') is spec.loader,
                'Original interpreter namespace loader')
        require(values.get('__name__') == name and values.get('__file__') is None
                and values.get('__package__') == name
                and list(values.get('__path__', ())) == locations
                and list(spec.submodule_search_locations or ()) == locations,
                'Exact namespace module identity and locations')

    def interpreter_module(self, name, module, policy):
        values = vars(module)
        spec = values.get('__spec__')
        kind = policy['origin']
        require(kind in ('built-in', 'frozen'), 'Explicit interpreter module origin')
        loader = machinery.BuiltinImporter if kind == 'built-in' else machinery.FrozenImporter
        require(type(spec) is machinery.ModuleSpec and spec.name == name
                and spec.origin == kind and spec.loader is loader
                and values.get('__loader__') is loader and values.get('__name__') == name,
                'Exact original interpreter module metadata')
        self.authenticate_interpreter(name, module, copy.deepcopy(policy))

    def native_module(self, name, module, row):
        values = vars(module)
        spec = values.get('__spec__')
        require(type(spec) is machinery.ModuleSpec and spec.name == name and spec.origin == row['path'],
                'Exact original native module origin')
        loader = spec.loader
        require(type(loader) is CheckedDelegatingLoader and loader.router is self.router
                and loader.kind == 'native' and type(loader.original) is machinery.ExtensionFileLoader
                and loader.original.path == row['path'] and values.get('__loader__') is loader,
                'Explicit authenticated native loader lifecycle')
        require(values.get('__name__') == name and values.get('__file__') == row['path'],
                'Exact original native module file')
        self.authenticate_native(name, module, copy.deepcopy(row))

    def audit(self):
        """Run inside DispatchContext.guard; callbacks must not import."""
        require(self.bound and not self.in_audit, 'Bound nonrecursive module origin audit')
        self.in_audit = True
        try:
            items = tuple(sys.modules.items())
            require(len(items) <= 4096, 'Bounded actual module registry')
            names = {name for name, module in items}
            require(set(self.preloaded) <= names, 'Preloaded modules cannot disappear')
            for name, module in items:
                if name in self.preloaded:
                    require(module is self.preloaded[name], 'Preloaded module cannot be replaced: '+name)
                    require(metadata(module) == self.preloaded_metadata[name],
                            'Preloaded import metadata changed: '+name)
                    kind = 'preloaded'
                else:
                    require(isinstance(module, types.ModuleType), 'New registered entry must be a module: '+name)
                    # Alias provider must authenticate both owner and exact
                    # returned object, including partially initialized aliases.
                    alias = self.authenticate_alias(name, module)
                    require(type(alias) is bool, 'Explicit alias approval boolean')
                    if alias:
                        kind = 'alias'
                    elif name in self.interpreter_modules:
                        self.interpreter_module(name, module, self.interpreter_modules[name])
                        kind = 'interpreter'
                    elif name in self.router.namespaces:
                        self.namespace_module(name, module, self.router.namespaces[name])
                        kind = 'namespace'
                    else:
                        row = self.router.modules.get(name)
                        require(row is not None, 'Module outside reviewed origins: '+name)
                        if row['kind'] == 'source':
                            self.source_module(name, module, row)
                            kind = 'source'
                        elif row['kind'] == 'native':
                            self.native_module(name, module, row)
                            kind = 'native'
                        else:
                            raise ValueError('Unreviewed module origin kind')
                prior = self.seen.get(name)
                if prior is None or prior[0] is not module:
                    require(len(self.events) < 4096, 'Bounded module origin events')
                    self.events.append(dict(name=name, kind=kind))
                    self.seen[name] = (module, kind)
            # Preserve normal optional import failure and module reload/removal:
            # non-preloaded names may disappear, and a later reimport is audited.
            for name in set(self.seen)-names:
                self.seen.pop(name)
        finally:
            self.in_audit = False

    def result(self):
        self.audit()
        return dict(module_origin_events=copy.deepcopy(self.events),
                    origin_ledger_checked=True,
                    complete_runtime_dependency_closure_proven=False,
                    preloaded_code_proven_by_this_module=False,
                    actual_SDK_runtime_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Preparation only: exact authenticated preloaded/interpreter/native/alias policies required')
