"""Guard CPython spec dispatch while preserving real sys.meta_path objects.

Designed for the dedicated gated Nano4B helper. Real source/native/preloaded
policies are caller supplied; the context cannot establish those policies or
prove arbitrary code cannot bypass Python import machinery. It deliberately
does not wrap/replace sys.meta_path entries: editable startup and six rely on
their original identities, order and normal registration operations.
"""
from contextlib import contextmanager
import importlib._bootstrap as bootstrap
import importlib.machinery as machinery
import importlib.util as util
import sys

from nemotron4b_import_router_v4 import ImportRouter, ProvenanceError
from nemotron4b_process_handshake_v1 import require

ACTIVE = None


class SdkFinderBindings:
    """Bind exact observed SDK startup providers without executing their code.

    authenticate_module(name, object) must bind the actual loaded module to its
    reviewed source/interpreter provenance; matching a module name alone is not
    authentication. Registration/removal and finder order remain Python-owned.
    """
    def __init__(self, *, authenticate_module, authenticate_interpreter_finder):
        require(callable(authenticate_module) and callable(authenticate_interpreter_finder),
                'Actual module and interpreter finder policies required')
        self.authenticate_module = authenticate_module
        self.authenticate_interpreter_finder = authenticate_interpreter_finder
        self.standard = (machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder)
        self.bindings = (
            ('_distutils_hack', 'DISTUTILS_FINDER', 'DistutilsMetaFinder', False),
            ('__editable___vllm_neuron_0_2_2_lts_finder', '_EditableFinder', '_EditableFinder', True),
            ('six', '_importer', '_SixMetaPathImporter', False),
        )

    def authorize(self, finder):
        if any(finder is x for x in self.standard):
            self.authenticate_interpreter_finder(finder)
            return
        for name, attribute, cls_name, is_class in self.bindings:
            module = sys.modules.get(name)
            if module is None:
                continue
            values = vars(module)
            if values.get(attribute) is not finder:
                continue
            self.authenticate_module(name, module)
            cls = values.get(cls_name)
            require(cls is not None and (finder is cls if is_class else type(finder) is cls),
                    'Exact original startup finder class')
            if name == 'six':
                require(finder.name == 'six', 'Original six alias namespace')
            return
        raise ProvenanceError('Finder is not an authenticated SDK startup/interpreter provider')


class DispatchContext:
    def __init__(self, router, *, authenticate_bootstrap, audit_modules,
                 audit_environment, approve_existing_spec):
        require(type(router) is ImportRouter, 'Exact selected v4 router')
        require(all(callable(x) for x in (authenticate_bootstrap, audit_modules,
                audit_environment, approve_existing_spec)), 'Complete context policies')
        self.router = router
        self.authenticate_bootstrap = authenticate_bootstrap
        self.audit_modules = audit_modules
        self.audit_environment = audit_environment
        self.approve_existing_spec = approve_existing_spec
        self.original_find_spec = None
        self.original_util_spec = None
        self.hook = None
        self.used = False
        self.in_audit = False
        self.original_guard = router.guard
        self.provenance = dict(schema='nemotron4b_import_context_v3',
            dispatch_context_completed=False, full_runtime_dependency_closure_proven=False,
            actual_SDK_runtime_qualified=False, runtime_admitted=False, new_speedup_claim=False)

    def guard(self):
        # Policy callbacks must use only authenticated, already-loaded helpers.
        # Reentrant imports during an audit must fail rather than evade checks.
        require(not self.in_audit, 'Origin audit cannot perform recursive imports')
        self.in_audit = True
        try:
            self.original_guard()
            require(ACTIVE is self and bootstrap._find_spec is self.hook
                    and util._find_spec is self.hook, 'Exact active dispatch hooks')
            require(type(sys.meta_path) is list and sys.meta_path,
                    'Live original finder list required')
            self.audit_environment()
            self.audit_modules()
        finally:
            self.in_audit = False

    def find_spec(self, name, path=None, target=None):
        router = self.router
        router.check()
        # Keep CPython live-list iteration: a finder may append/remove another
        # finder during its call. Snapshotting this list changes that behavior.
        meta_path = sys.meta_path
        is_reload = name in sys.modules
        for finder in meta_path:
            # Authorization must precede even find_spec attribute lookup.
            router.authenticate(router.authorize_finder, finder)
            with bootstrap._ImportLockContext():
                router.check()
                try:
                    method = router.invoke(getattr, finder, 'find_spec')
                except AttributeError as error:
                    # All observed installed SDK finders implement PEP451.
                    # Legacy load_module would execute before source wrapping.
                    router.reject(ProvenanceError('Unreviewed legacy finder: '+type(finder).__name__))
                spec = router.invoke(method, name, path, target)
            if spec is None:
                continue
            # CPython prefers a module installed as a finder side effect unless
            # this was already a reload. Validate that effective spec separately.
            if not is_reload and name in sys.modules:
                module = sys.modules[name]
                existing = getattr(module, '__spec__', None)
                if existing is not None:
                    router.authenticate(self.approve_existing_spec, name, module, existing, finder)
                    router.check()
                    router.record('existing_spec', name, getattr(existing, 'origin', None))
                    return existing
            selected = router.authenticate(router.validate_spec, name, spec, finder)
            router.check()
            return selected
        router.record('absent', name, None)
        return None

    @contextmanager
    def installed(self):
        global ACTIVE
        require(not self.used and ACTIVE is None, 'One-use dedicated helper context; no nesting')
        self.used = True
        router = self.router
        router.check()
        self.original_find_spec = bootstrap._find_spec
        self.original_util_spec = util._find_spec
        require(self.original_util_spec is self.original_find_spec,
                'Unmodified standard importlib.util dispatch alias')
        # Caller authenticates the actual interpreter/stdlib and preloaded code
        # before any replacement. These callbacks must not load the SDK.
        router.authenticate(self.authenticate_bootstrap, self.original_find_spec)
        router.authenticate(self.audit_environment)
        router.authenticate(self.audit_modules)
        self.hook = self.find_spec
        ACTIVE = self
        bootstrap._find_spec = self.hook
        util._find_spec = self.hook
        router.guard = self.guard
        try:
            router.check()
            yield self.provenance
            router.check()
            result = router.finish()
            self.provenance.update(dispatch_context_completed=True, router=result)
        except BaseException as error:
            # An uncaught caller/helper error means this context did not finish.
            # Ordinary optional imports caught inside the helper remain valid.
            router.fail(error)
            self.provenance['failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
            raise
        finally:
            # Restore only our hooks for this context's caller; imported modules
            # and finder registration are not rolled back or claimed reset.
            bootstrap._find_spec = self.original_find_spec
            util._find_spec = self.original_util_spec
            router.guard = self.original_guard
            ACTIVE = None


if __name__ == '__main__':
    raise SystemExit('Preparation only: real module, finder, environment and native policies required')
