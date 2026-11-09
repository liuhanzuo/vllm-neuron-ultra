"""Import actual synthetic modules through router; no SDK/native loading."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery as machinery
import importlib.util
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron4b_import_router_v1 as routing
import nemotron4b_helper_dependency_catalog_v1 as catalog


class Imports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano4b-import-router-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.modules, self.namespaces = {}, {}
        self.names = set()
        self.auth_calls = []

    def source(self, name, text, package=False):
        path = self.root.joinpath(*name.split('.'))
        path = path/'__init__.py' if package else path.with_suffix('.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        _, row = catalog.source(path)
        self.modules[name] = dict(row, kind='source', package=package)
        self.names.add(name)
        return path

    def router(self, finders=None, **kwargs):
        chosen = list(finders or [machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder])
        def authorize(finder):
            self.assertTrue(any(finder is x for x in chosen))
            self.auth_calls.append(finder)
        def deny(*args):
            raise RuntimeError('Fixture provider not approved')
        args = dict(finders=chosen, modules=self.modules, namespaces=self.namespaces,
            guard=lambda:None, authorize_finder=authorize, authorize_builtin=lambda *a:None,
            authorize_native=deny, authorize_custom=deny)
        args.update(kwargs)
        return routing.ImportRouter(**args)

    @contextmanager
    def installed(self, router):
        old_finders, old_path = list(sys.meta_path), list(sys.path)
        before = set(sys.modules)
        sys.meta_path[:]=[router]
        sys.path.insert(0, str(self.root))
        try:
            yield
        finally:
            sys.meta_path[:]=old_finders
            sys.path[:]=old_path
            for name in set(sys.modules)-before:
                if name.startswith('_nano_import_') or name in self.names:
                    sys.modules.pop(name,None)
            importlib.invalidate_caches()

    def test_real_package_relative_import_and_original_origin(self):
        name='_nano_import_package'
        path=self.source(name,'from .leaf import VALUE\n',True)
        self.source(name+'.leaf','VALUE = 17\n')
        router=self.router()
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertEqual(module.VALUE,17)
            self.assertEqual(module.__file__,str(path))
            self.assertEqual(list(module.__path__),[str(path.parent)])
            self.assertFalse(Path(module.__cached__).exists())
            result=router.finish()
        self.assertTrue(result['router_dispatch_completed'])
        self.assertFalse(result['complete_runtime_dependency_closure_proven'])
        self.assertFalse(list(self.root.rglob('*.pyc')))

    def test_existing_valid_timestamp_bytecode_cannot_replace_source(self):
        name='_nano_import_bytecode'
        path=self.source(name,'VALUE = 10\n')
        py_compile.compile(str(path),doraise=True)
        old=path.stat()
        # Same size/time makes the existing timestamp-based .pyc appear valid.
        path.write_text('VALUE = 20\n')
        os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        _,row=catalog.source(path)
        self.modules[name]=dict(row,kind='source',package=False)
        router=self.router()
        with self.installed(router):
            self.assertEqual(importlib.import_module(name).VALUE,20)
            router.finish()

    def test_absent_optional_dependency_preserves_importerror_fallback(self):
        name='_nano_import_optional'
        self.source(name,'try:\n import _nano_import_absent_xyz\nexcept ImportError:\n VALUE = "fallback"\n')
        router=self.router()
        with self.installed(router):
            self.assertEqual(importlib.import_module(name).VALUE,'fallback')
            result=router.finish()
        self.assertTrue(any(x['kind']=='absent' for x in result['events']))
        self.assertFalse(router.poisoned)

    def test_found_unpinned_dependency_cannot_become_optional_absence(self):
        name='_nano_import_optional_bad'
        self.source(name,'try:\n import _nano_import_unpinned\nexcept ImportError:\n VALUE = "fallback"\n')
        (self.root/'_nano_import_unpinned.py').write_text('VALUE = "unexpected"\n')
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)
        self.assertTrue(router.poisoned)
        self.assertNotIn('_nano_import_unpinned',sys.modules)

    def test_source_change_prevents_any_module_body_execution(self):
        name='_nano_import_changed'
        marker=self.root/'should-not-execute'
        path=self.source(name,'VALUE = 1\n')
        path.write_text('open('+repr(str(marker))+',"w").write("ran")\n')
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)
        self.assertFalse(marker.exists())
        self.assertTrue(router.poisoned)

    def test_namespace_locations_preserved(self):
        name='_nano_import_namespace'
        directory=self.root/name
        directory.mkdir()
        self.namespaces[name]=[str(directory)]
        self.source(name+'.leaf','VALUE = 31\n')
        router=self.router()
        with self.installed(router):
            module=importlib.import_module(name+'.leaf')
            self.assertEqual(module.VALUE,31)
            self.assertEqual(list(sys.modules[name].__path__),[str(directory)])
            router.finish()

    def test_namespace_extra_search_root_rejected(self):
        name='_nano_import_space_wrong'
        directory=self.root/name;directory.mkdir()
        self.namespaces[name]=[str(directory),str(self.root/'not-original')]
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)

    def test_exact_finder_is_authorized_before_call(self):
        calls=[]
        class Finder:
            def find_spec(self,*args):
                calls.append('invoked')
                return None
        finder=Finder()
        def deny(f):
            raise RuntimeError('Finder source not approved')
        router=self.router(finders=[finder],authorize_finder=deny)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module('_nano_import_denied')
        self.assertEqual(calls,[])

    def test_finder_exception_is_not_reclassified_as_absence(self):
        class Finder:
            def find_spec(self,*args):
                raise ModuleNotFoundError('Finder-internal missing dependency')
        router=self.router(finders=[Finder()])
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module('_nano_import_broken_finder')
        self.assertTrue(router.poisoned)

    def test_custom_dynamic_alias_loader_semantics_preserved(self):
        name='_nano_import_dynamic_alias'
        self.names.add(name)
        alias=types.ModuleType('_nano_fixture_alias_target')
        alias.VALUE=41
        calls=[]
        class Loader:
            def create_module(self,spec):
                calls.append('create')
                return alias
            def exec_module(self,module):
                self_is_alias=(module is alias)
                calls.append(('exec',self_is_alias))
        loader=Loader()
        class Finder:
            def find_spec(self,fullname,path=None,target=None):
                return importlib.util.spec_from_loader(fullname,loader) if fullname==name else None
        finder=Finder()
        def authorize(fullname,spec,provider):
            self.assertEqual(fullname,name)
            self.assertIs(provider,finder);self.assertIs(spec.loader,loader)
            def before(n,original,spec):
                self.assertIs(original,loader);calls.append('before')
            def after(n,original,module,phase):
                self.assertIs(module,alias);calls.append('after-'+phase)
            return before,after
        router=self.router(finders=[finder],authorize_custom=authorize)
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertIs(module,alias);self.assertEqual(module.VALUE,41)
            router.finish()
        self.assertEqual(calls,['before','create','after-create','before',('exec',True),'after-exec'])

    def test_custom_loader_without_explicit_provider_rejected(self):
        name='_nano_import_custom_denied'
        calls=[]
        class Loader:
            def create_module(self,spec):
                calls.append('executed');return None
            def exec_module(self,module):
                calls.append('executed')
        class Finder:
            def find_spec(self,*args):
                return importlib.util.spec_from_loader(name,Loader())
        router=self.router(finders=[Finder()])
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)
        self.assertEqual(calls,[])

    def test_native_rejection_precedes_extension_initialization(self):
        name='_nano_import_fake_native'
        path=self.root/(name+'.so');path.write_bytes(b'not-an-actual-library')
        self.modules[name]=dict(kind='native',path=str(path))
        class Finder:
            def find_spec(self,*args):
                return importlib.util.spec_from_file_location(name,path,
                    loader=machinery.ExtensionFileLoader(name,str(path)))
        router=self.router(finders=[Finder()])
        with self.installed(router):
            with self.assertRaisesRegex(routing.ProvenanceError,'provider not approved'):
                importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def test_caught_provenance_failure_still_poisoned_at_finish(self):
        name='_nano_import_catches_everything'
        self.source(name,'try:\n import _nano_import_unpinned\nexcept Exception:\n VALUE = "caught"\n')
        (self.root/'_nano_import_unpinned.py').write_text('VALUE = 1\n')
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)
            with self.assertRaises(routing.ProvenanceError):
                router.finish()
        self.assertTrue(router.poisoned)

    def test_catalog_mutation_after_construction_does_not_rebind_source(self):
        name='_nano_import_snapshot'
        path=self.source(name,'VALUE = 1\n');router=self.router()
        path.write_text('VALUE = 2\n')
        _,row=catalog.source(path);self.modules[name]=dict(row,kind='source',package=False)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(name)

    def test_nonzero_module_exit_remains_original(self):
        name='_nano_import_exit'
        self.source(name,'raise SystemExit(29)\n');router=self.router()
        with self.installed(router):
            with self.assertRaises(SystemExit)as error:
                importlib.import_module(name)
            self.assertEqual(error.exception.code,29)
        self.assertTrue(router.poisoned)


if __name__ == '__main__':
    unittest.main(verbosity=2)
