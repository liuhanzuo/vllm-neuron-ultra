"""Actual synthetic imports through bootstrap hooks; installed SDK stays unloaded."""
from contextlib import contextmanager
import importlib
import importlib.machinery as machinery
import importlib.util
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_nemotron4b_import_router_v4_r2 as fixtures
import nemotron4b_import_context_v3 as context
import nemotron4b_import_router_v4 as routing


class ContextImports(unittest.TestCase):
    setUp=fixtures.Imports.setUp
    source=fixtures.Imports.source
    router=fixtures.Imports.router

    def make(self,router,**kwargs):
        args=dict(authenticate_bootstrap=lambda original: self.assertIs(original,self.original),
            audit_modules=lambda:None,audit_environment=lambda:None,
            approve_existing_spec=lambda *a:None)
        args.update(kwargs)
        return context.DispatchContext(router,**args)

    @contextmanager
    def runtime(self,finders=None):
        old_finders,old_path=sys.meta_path,list(sys.path)
        self.original=context.bootstrap._find_spec
        original_util=context.util._find_spec
        before=dict(sys.modules)
        chosen=list(finders or [machinery.BuiltinImporter,machinery.FrozenImporter,machinery.PathFinder])
        sys.meta_path=chosen
        sys.path.insert(0,str(self.root))
        try:
            yield chosen
        finally:
            sys.meta_path=old_finders
            sys.path[:]=old_path
            for name in set(sys.modules)-set(before):
                if name.startswith('_nano_import_')or name in self.names:sys.modules.pop(name,None)
            # Assertions below tests never silently repair leaking dispatch hooks.
            self.assertIs(context.bootstrap._find_spec,self.original)
            self.assertIs(context.util._find_spec,original_util)
            self.assertIsNone(context.ACTIVE)
            importlib.invalidate_caches()

    def test_source_import_keeps_real_finder_objects_and_restores_hooks(self):
        name='_nano_import_plain';self.source(name,'VALUE=101\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);scope=self.make(router)
            with scope.installed()as result:
                self.assertIs(sys.meta_path,finders)
                self.assertTrue(all(a is b for a,b in zip(sys.meta_path,router.finders)))
                self.assertEqual(importlib.import_module(name).VALUE,101)
            self.assertTrue(result['dispatch_context_completed'])
            self.assertFalse(result['full_runtime_dependency_closure_proven'])

    def test_parent_optional_child_body_failure_still_works(self):
        self.source('_nano_import_child','raise ImportError("optional unsupported platform")\n')
        self.source('_nano_import_parent','try:\n import _nano_import_child\nexcept ImportError:\n VALUE=103\n')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.make(router).installed():
                self.assertEqual(importlib.import_module('_nano_import_parent').VALUE,103)
                self.assertNotIn('_nano_import_child',sys.modules)
            self.assertFalse(router.poisoned)

    def test_registrar_adds_finder_then_immediately_imports_with_it(self):
        registrar='_nano_import_registrar';leaf='_nano_import_registered_leaf'
        self.source(leaf,'VALUE=107\n')
        self.source(registrar,
            'import sys\nfrom importlib.machinery import PathFinder\n'
            'class Finder:\n'
            ' def find_spec(self,name,path=None,target=None):\n'
            '  return PathFinder.find_spec(name,path,target)\n'
            'FINDER=Finder()\n'
            'if not any(item is FINDER for item in sys.meta_path):\n sys.meta_path.insert(0,FINDER)\n'
            'import '+leaf+'\nVALUE='+leaf+'.VALUE\n')
        with self.runtime()as finders:
            originals=tuple(finders);seen=[]
            def authorize(finder):
                if any(finder is x for x in originals):return
                module=sys.modules.get(registrar)
                self.assertIsNotNone(module)
                self.assertIs(finder,module.FINDER)
                seen.append(finder)
            router=self.router(finders=finders,authorize_finder=authorize)
            with self.make(router).installed():
                self.assertEqual(importlib.import_module(registrar).VALUE,107)
                self.assertIs(finders[0],sys.modules[registrar].FINDER)
            self.assertTrue(seen)

    def test_live_finder_list_append_affects_same_lookup(self):
        name='_nano_import_live_append';self.source(name,'VALUE=109\n')
        calls=[];allowed=[]
        class Second:
            def find_spec(self,n,path=None,target=None):
                calls.append('second');return machinery.PathFinder.find_spec(n,path,target)
        second=Second()
        class First:
            def find_spec(self,n,path=None,target=None):
                calls.append('first')
                if second not in sys.meta_path:sys.meta_path.append(second)
                return None
        first=First();allowed[:]=[first,second]
        with self.runtime([first])as finders:
            router=self.router(finders=[first],authorize_finder=lambda f:self.assertTrue(any(f is x for x in allowed)))
            with self.make(router).installed():
                self.assertEqual(importlib.import_module(name).VALUE,109)
            self.assertEqual(calls,['first','second'])

    def test_util_find_spec_uses_authenticated_source_dispatch(self):
        name='_nano_import_util';self.source(name,'VALUE=113\n')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.make(router).installed():
                spec=importlib.util.find_spec(name)
                self.assertIs(type(spec.loader),routing.PinnedSourceLoader)

    def test_unknown_finder_rejected_before_property_lookup(self):
        calls=[]
        class Rogue:
            @property
            def find_spec(self):
                calls.append('executed')
                return lambda *a:None
        with self.runtime()as finders:
            originals=tuple(finders)
            def authorize(f):
                if not any(f is x for x in originals):raise RuntimeError('Unknown provider')
            router=self.router(finders=finders,authorize_finder=authorize)
            with self.assertRaises(routing.ProvenanceError):
                with self.make(router).installed():
                    finders.insert(0,Rogue())
                    importlib.import_module('_nano_import_rejected')
            self.assertEqual(calls,[])

    def test_replaced_bootstrap_hook_rejected_at_exit_and_restored(self):
        with self.runtime()as finders:
            router=self.router(finders=finders);scope=self.make(router)
            with self.assertRaises(routing.ProvenanceError):
                with scope.installed():
                    context.bootstrap._find_spec=lambda *a:None
            self.assertTrue(router.poisoned)
            self.assertFalse(scope.provenance['dispatch_context_completed'])

    def test_replaced_util_alias_rejected_at_exit_and_restored(self):
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                with self.make(router).installed():
                    context.util._find_spec=lambda *a:None

    def test_preloaded_module_audit_refuses_entry(self):
        with self.runtime()as finders:
            router=self.router(finders=finders)
            def reject():raise ImportError('Unreviewed preloaded module')
            with self.assertRaises(routing.ProvenanceError):
                with self.make(router,audit_modules=reject).installed():self.fail('Must not enter')
            self.assertTrue(router.poisoned)

    def test_finder_side_effect_module_audited_before_return(self):
        name='_nano_import_injected';self.names.add(name)
        class Finder:
            def find_spec(self,*args):
                sys.modules[name]=types.ModuleType(name)
                return None
        finder=Finder()
        def audit():
            if name in sys.modules:raise RuntimeError('Unreviewed injected module')
        with self.runtime([finder])as finders:
            router=self.router(finders=finders)
            with self.assertRaisesRegex(routing.ProvenanceError,'injected module'):
                with self.make(router,audit_modules=audit).installed():
                    importlib.import_module('_nano_import_target')

    def test_existing_side_effect_spec_preserves_cpython_selection(self):
        name='_nano_import_side_spec';self.names.add(name)
        returned=machinery.ModuleSpec(name,loader=None)
        effective=machinery.ModuleSpec(name,loader=None)
        module=types.ModuleType(name);module.__spec__=effective
        class Finder:
            def find_spec(self,*args):
                sys.modules[name]=module
                return returned
        finder=Finder();approved=[]
        def approve(*args):
            self.assertEqual(args,(name,module,effective,finder));approved.append(True)
        with self.runtime([finder])as finders:
            router=self.router(finders=finders)
            with self.make(router,approve_existing_spec=approve).installed():
                self.assertIs(context.bootstrap._find_spec(name,None),effective)
            self.assertEqual(approved,[True])

    def test_existing_side_effect_spec_rejection_is_sticky(self):
        name='_nano_import_bad_side_spec';self.names.add(name)
        module=types.ModuleType(name);module.__spec__=machinery.ModuleSpec(name,loader=None)
        class Finder:
            def find_spec(self,*args):
                sys.modules[name]=module;return module.__spec__
        finder=Finder()
        def reject(*args):raise ImportError('Unreviewed effective module spec')
        with self.runtime([finder])as finders:
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                with self.make(router,approve_existing_spec=reject).installed():
                    context.bootstrap._find_spec(name,None)
            self.assertTrue(router.poisoned)

    def test_reload_uses_selected_source_and_original_module(self):
        name='_nano_import_reload';self.source(name,'COUNT=globals().get("COUNT",0)+1\n')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.make(router,approve_existing_spec=lambda *a:self.fail('Reload must not choose side-effect spec')).installed():
                module=importlib.import_module(name);self.assertEqual(module.COUNT,1)
                self.assertIs(importlib.reload(module),module)
                self.assertEqual(module.COUNT,2)

    def test_legacy_finder_is_denied_before_load_module(self):
        calls=[]
        class Legacy:
            def find_module(self,*args):calls.append('executed');return self
            def load_module(self,*args):calls.append('loaded')
        finder=Legacy()
        with self.runtime([finder])as finders:
            router=self.router(finders=finders)
            with self.assertRaisesRegex(routing.ProvenanceError,'legacy finder'):
                with self.make(router).installed():
                    importlib.import_module('_nano_import_legacy')
            self.assertEqual(calls,[])

    def test_nested_context_refusal_keeps_outer_hooks(self):
        with self.runtime()as finders:
            router=self.router(finders=finders);scope=self.make(router)
            other=self.make(self.router(finders=finders))
            with scope.installed():
                with self.assertRaises(ValueError):
                    with other.installed():self.fail('Nested context entered')
                self.assertIs(context.bootstrap._find_spec,scope.hook)

    def test_audit_recursive_import_is_rejected_without_unbounded_recursion(self):
        def audit():
            if context.ACTIVE is not None:importlib.import_module('_nano_import_recursive')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaisesRegex(routing.ProvenanceError,'recursive imports'):
                with self.make(router,audit_modules=audit).installed():self.fail('Must not enter')
            self.assertTrue(router.poisoned)

    def test_systemexit_keeps_original_code_and_restores_hooks(self):
        name='_nano_import_terminal';self.source(name,'raise SystemExit(43)\n')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaises(SystemExit)as error:
                with self.make(router).installed():importlib.import_module(name)
            self.assertEqual(error.exception.code,43)


class Bindings(unittest.TestCase):
    def setUp(self):
        self.saved={name:sys.modules.get(name)for name in
            ('six','_distutils_hack','__editable___vllm_neuron_0_2_2_lts_finder')}
        self.addCleanup(self.restore)
        self.authenticated=[]
        self.policy=context.SdkFinderBindings(authenticate_module=lambda n,m:self.authenticated.append((n,m)),
            authenticate_interpreter_finder=lambda f:self.authenticated.append(f))

    def restore(self):
        for name,module in self.saved.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module

    def fixture(self,name,attribute,classname,is_class=False):
        module=types.ModuleType(name);cls=type(classname,(),{})
        finder=cls if is_class else cls()
        if name=='six':finder.name='six'
        setattr(module,classname,cls);setattr(module,attribute,finder)
        sys.modules[name]=module
        return module,finder

    def test_standard_interpreter_objects_are_exact(self):
        for f in (machinery.BuiltinImporter,machinery.FrozenImporter,machinery.PathFinder):self.policy.authorize(f)
        self.assertEqual(len(self.authenticated),3)

    def test_three_real_startup_binding_shapes(self):
        for name,attribute,cls_name,is_class in self.policy.bindings:
            module,finder=self.fixture(name,attribute,cls_name,is_class)
            self.policy.authorize(finder)
            self.assertIn((name,module),self.authenticated)

    def test_same_type_unregistered_object_is_rejected(self):
        module,finder=self.fixture('six','_importer','_SixMetaPathImporter')
        imitation=type(finder)();imitation.name='six'
        with self.assertRaises(routing.ProvenanceError):self.policy.authorize(imitation)

    def test_module_authentication_cannot_be_replaced_by_name_match(self):
        module,finder=self.fixture('_distutils_hack','DISTUTILS_FINDER','DistutilsMetaFinder')
        def reject(*a):raise RuntimeError('Unpinned module body')
        self.policy.authenticate_module=reject
        with self.assertRaisesRegex(RuntimeError,'Unpinned'):self.policy.authorize(finder)

    def test_six_alias_namespace_must_match(self):
        module,finder=self.fixture('six','_importer','_SixMetaPathImporter');finder.name='other_six'
        with self.assertRaises(ValueError):self.policy.authorize(finder)


if __name__=='__main__':
    unittest.main(verbosity=2)
