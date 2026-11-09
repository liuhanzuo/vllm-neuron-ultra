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
import nemotron4b_import_router_v4 as routing
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

    def test_finder_exception_retains_identity_and_is_not_absence(self):
        original = ModuleNotFoundError('Finder-internal missing dependency')
        class Finder:
            def find_spec(self,*args):
                raise original
        router=self.router(finders=[Finder()])
        with self.installed(router):
            with self.assertRaises(ModuleNotFoundError)as caught:
                importlib.import_module('_nano_import_broken_finder')
            self.assertIs(caught.exception, original)
            result=router.finish()
        self.assertFalse(router.poisoned)
        self.assertFalse(any(row['kind']=='absent' for row in result['events']))

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

    def test_parent_can_catch_child_body_importerror_and_continue(self):
        parent='_nano_import_parent';child='_nano_import_child'
        self.source(child,'raise ImportError("unsupported optional platform")\n')
        self.source('_nano_import_followup','VALUE = 53\n')
        self.source(parent,'try:\n import '+child+'\nexcept ImportError:\n from _nano_import_followup import VALUE\n')
        router=self.router()
        with self.installed(router):
            self.assertEqual(importlib.import_module(parent).VALUE,53)
            self.assertNotIn(child,sys.modules)
            router.finish()
        self.assertFalse(router.poisoned)

    def test_parent_can_catch_child_missing_dependency(self):
        parent='_nano_import_parent';child='_nano_import_child'
        self.source(child,'import _nano_import_optional_nonexistent\n')
        self.source(parent,'try:\n import '+child+'\nexcept ImportError:\n VALUE = "fallback"\n')
        router=self.router()
        with self.installed(router):
            self.assertEqual(importlib.import_module(parent).VALUE,'fallback')
            self.assertNotIn(child,sys.modules)
            router.finish()
        self.assertFalse(router.poisoned)

    def test_ordinary_body_valueerror_is_not_provenance_failure(self):
        child='_nano_import_child'
        self.source(child,'raise ValueError("optional data unavailable")\n')
        self.source('_nano_import_parent','try:\n import '+child+'\nexcept ValueError:\n VALUE = "fallback"\n')
        router=self.router()
        with self.installed(router):
            self.assertEqual(importlib.import_module('_nano_import_parent').VALUE,'fallback')
            router.finish()
        self.assertFalse(router.poisoned)

    def test_importerror_from_guard_cannot_become_optional_fallback(self):
        child='_nano_import_child'
        self.source(child,'VALUE = 1\n')
        self.source('_nano_import_parent','try:\n import '+child+'\nexcept ImportError:\n VALUE = "fallback"\n')
        def guard():
            if child in sys.modules:
                raise ImportError('Guard failed, not optional absence')
        router=self.router(guard=guard)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module('_nano_import_parent')
            with self.assertRaises(routing.ProvenanceError):
                router.finish()
        self.assertTrue(router.poisoned)

    def test_importerror_from_source_reader_cannot_select_fallback(self):
        child='_nano_import_child'
        child_path=self.source(child,'VALUE = 1\n')
        self.source('_nano_import_parent','try:\n import '+child+'\nexcept ImportError:\n VALUE = "fallback"\n')
        original=routing.source
        def reader(path,digest):
            if path==str(child_path):
                raise ImportError('Authenticated source reader failed')
            return original(path,digest)
        routing.source=reader
        try:
            router=self.router()
            with self.installed(router):
                with self.assertRaises(routing.ProvenanceError):
                    importlib.import_module('_nano_import_parent')
                with self.assertRaises(routing.ProvenanceError):
                    router.finish()
        finally:
            routing.source=original

    def custom_failure_fixture(self,phase,*,audit_reject=False,terminal=False):
        parent='_nano_import_parent';child='_nano_import_custom_child'
        self.source(parent,'try:\n import '+child+'\nexcept ImportError:\n VALUE = "fallback"\n')
        self.names.add(child)
        original=SystemExit(37)if terminal else ImportError('Provider unavailable')
        calls=[]
        class Loader:
            def create_module(self,spec):
                calls.append('create')
                if phase=='create':raise original
                return None
            def exec_module(self,module):
                calls.append('exec')
                if phase=='exec':raise original
        loader=Loader()
        class Finder:
            def find_spec(self,fullname,path=None,target=None):
                return importlib.util.spec_from_loader(fullname,loader)if fullname==child else None
        finder=Finder()
        def authorize(fullname,spec,provider):
            self.assertIs(provider,finder)
            def before(n,original_loader,spec):
                self.assertIs(original_loader,loader)
                calls.append('before')
            def after(n,original_loader,module,stage):
                self.assertIs(original_loader,loader)
                calls.append(stage)
                if stage.endswith('_failed'):
                    self.assertIsNone(module)if stage=='create_failed'else self.assertIsInstance(module,types.ModuleType)
                    if audit_reject:raise ImportError('Postfailure origin audit rejected')
            return before,after
        router=self.router(finders=[finder,machinery.PathFinder],authorize_custom=authorize)
        return parent,child,original,calls,router

    def test_custom_create_importerror_preserves_parent_fallback_and_failure_audit(self):
        parent,child,original,calls,router=self.custom_failure_fixture('create')
        with self.installed(router):
            self.assertEqual(importlib.import_module(parent).VALUE,'fallback')
            self.assertNotIn(child,sys.modules)
            router.finish()
        self.assertEqual(calls,['before','create','create_failed'])
        self.assertFalse(router.poisoned)

    def test_custom_exec_importerror_preserves_parent_fallback_and_failure_audit(self):
        parent,child,original,calls,router=self.custom_failure_fixture('exec')
        with self.installed(router):
            self.assertEqual(importlib.import_module(parent).VALUE,'fallback')
            self.assertNotIn(child,sys.modules)
            router.finish()
        self.assertEqual(calls,['before','create','create','before','exec','exec_failed'])
        self.assertFalse(router.poisoned)

    def test_provider_exception_identity_retained(self):
        parent,child,original,calls,router=self.custom_failure_fixture('exec')
        with self.installed(router):
            with self.assertRaises(ImportError)as caught:
                importlib.import_module(child)
            self.assertIs(caught.exception,original)
            router.finish()

    def test_postfailure_audit_rejection_blocks_optional_fallback(self):
        parent,child,original,calls,router=self.custom_failure_fixture('exec',audit_reject=True)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module(parent)
            with self.assertRaises(routing.ProvenanceError):
                router.finish()
        self.assertIn('exec_failed',calls)
        self.assertTrue(router.poisoned)
        self.assertIn('Postfailure origin audit',router.failure['message'])

    def test_terminal_provider_exit_preserved_when_failure_audit_also_fails(self):
        parent,child,original,calls,router=self.custom_failure_fixture('create',audit_reject=True,terminal=True)
        with self.installed(router):
            with self.assertRaises(SystemExit)as caught:
                importlib.import_module(child)
            self.assertIs(caught.exception,original)
            self.assertEqual(caught.exception.code,37)
            with self.assertRaises(routing.ProvenanceError):
                router.finish()
        self.assertIn('create_failed',calls)
        self.assertTrue(router.poisoned)
        self.assertIn('Postfailure origin audit',router.failure['message'])

    def test_importerror_from_authorization_is_sticky(self):
        def deny(*args):
            raise ImportError('Finder authorization failed')
        router=self.router(authorize_finder=deny)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):
                importlib.import_module('_nano_import_denied')
            with self.assertRaises(routing.ProvenanceError):
                router.finish()

    def test_provider_failure_cannot_hide_failed_post_guard(self):
        state={'failed':False}
        original=ImportError('Natural provider failure')
        class Finder:
            def find_spec(self,*args):
                state['failed']=True
                raise original
        def guard():
            if state['failed']:raise RuntimeError('Post-provider resource check failed')
        router=self.router(finders=[Finder()],guard=guard)
        with self.installed(router):
            with self.assertRaisesRegex(routing.ProvenanceError,'resource check failed'):
                importlib.import_module('_nano_import_guarded_failure')
        self.assertTrue(router.poisoned)

    def alias_source(self,name,text):
        target=self.root/'physical'/(name+'.py');target.parent.mkdir(exist_ok=True);target.write_text(text)
        link=self.root/(name+'.py');link.symlink_to(target)
        _,row=catalog.source(target)
        alias=dict(target=str(target),link_text=os.readlink(link),
                   fullstat={k:getattr(link.lstat(),k)for k in catalog.FIELDS})
        self.modules[name]=dict(row,path=str(link),kind='source',package=False,alias=alias)
        self.names.add(name)
        return link,target

    def test_reviewed_source_alias_preserves_original_module_file(self):
        name='_nano_import_alias_source';link,target=self.alias_source(name,'VALUE=127\n')
        router=self.router()
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertEqual(module.VALUE,127)
            self.assertEqual(module.__file__,str(link))
            self.assertEqual(module.__spec__.origin,str(link))
            router.finish()

    def test_source_alias_retarget_is_rejected_before_body(self):
        name='_nano_import_alias_retarget';link,target=self.alias_source(name,'VALUE=127\n')
        router=self.router();other=target.with_name('other.py');other.write_bytes(target.read_bytes())
        link.unlink();link.symlink_to(other)
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def test_source_alias_target_change_is_rejected(self):
        name='_nano_import_alias_target_change';link,target=self.alias_source(name,'VALUE=127\n')
        router=self.router();target.write_text('VALUE=131\n')
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def test_source_alias_without_explicit_record_is_rejected(self):
        name='_nano_import_alias_unreviewed';link,target=self.alias_source(name,'VALUE=127\n')
        del self.modules[name]['alias']
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def test_source_alias_chain_requires_separate_explicit_policy(self):
        name='_nano_import_alias_chain';link,target=self.alias_source(name,'VALUE=127\n')
        intermediate=self.root/'middle.py';intermediate.symlink_to(target)
        link.unlink();link.symlink_to(intermediate)
        alias=self.modules[name]['alias']
        alias['link_text']=os.readlink(link)
        alias['fullstat']={k:getattr(link.lstat(),k)for k in catalog.FIELDS}
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def hardlinked_source(self,name,text,package=False):
        path=self.source(name,text,package)
        sibling=self.root/(name.replace('.','_')+'-installed-peer.py')
        os.link(path,sibling)
        raw=path.read_bytes()
        self.modules[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),fullstat={k:getattr(path.stat(),k)for k in catalog.FIELDS},
            original_nlink=2,kind='source',package=package)
        return path,sibling

    def test_actual_hardlinked_source_import_preserves_inode_and_peer(self):
        name='_nano_import_hardlink';path,peer=self.hardlinked_source(name,'VALUE=173\n')
        before=path.stat();router=self.router()
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertEqual(module.VALUE,173)
            self.assertEqual(module.__file__,str(path))
            router.finish()
        self.assertEqual(path.stat().st_ino,before.st_ino)
        self.assertEqual(peer.stat().st_ino,before.st_ino)
        self.assertEqual(path.stat().st_nlink,2)
        self.assertEqual(path.read_bytes(),peer.read_bytes())

    def test_hardlink_requires_explicit_original_link_count(self):
        name='_nano_import_hardlink_unapproved';path,peer=self.hardlinked_source(name,'VALUE=179\n')
        del self.modules[name]['original_nlink']
        router=self.router()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)
        self.assertTrue(router.poisoned)

    def test_hardlink_modified_through_peer_is_rejected(self):
        name='_nano_import_hardlink_changed';path,peer=self.hardlinked_source(name,'VALUE=181\n')
        router=self.router();peer.write_text('VALUE=191\n')
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)

    def test_hardlink_peer_removal_is_rejected(self):
        name='_nano_import_hardlink_removed';path,peer=self.hardlinked_source(name,'VALUE=193\n')
        router=self.router();peer.unlink()
        with self.installed(router):
            with self.assertRaises(routing.ProvenanceError):importlib.import_module(name)

    def test_hardlinked_package_relative_imports_keep_original_paths(self):
        name='_nano_import_hardlink_package'
        path,peer=self.hardlinked_source(name,'from .leaf import VALUE\n',True)
        self.hardlinked_source(name+'.leaf','VALUE=197\n')
        router=self.router()
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertEqual(module.VALUE,197)
            self.assertEqual(list(module.__path__),[str(path.parent)])
            router.finish()

    def test_original_declared_source_encoding_is_preserved(self):
        name='_nano_import_encoded'
        path=self.root/(name+'.py')
        raw=b'# coding: latin-1\nVALUE = "caf\xe9"\n'
        path.write_bytes(raw);self.names.add(name)
        self.modules[name]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),fullstat={k:getattr(path.stat(),k)for k in catalog.FIELDS},
            kind='source',package=False)
        router=self.router()
        with self.installed(router):
            module=importlib.import_module(name)
            self.assertEqual(module.VALUE,'café')
            self.assertIn('café',module.__loader__.get_source(name))
            router.finish()


if __name__ == '__main__':
    unittest.main(verbosity=2)
