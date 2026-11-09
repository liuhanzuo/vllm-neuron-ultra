"""Exercise origin ledger with real synthetic imports and explicit fixture policies."""
import importlib
import importlib.machinery as machinery
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_nemotron4b_import_router_v3 as fixtures
import check_nemotron4b_import_context_v2 as contexts
import nemotron4b_import_router_v3 as routing
import nemotron4b_module_origin_audit_v1 as origins


class OriginImports(unittest.TestCase):
    setUp=fixtures.Imports.setUp
    source=fixtures.Imports.source
    router=fixtures.Imports.router
    runtime=contexts.ContextImports.runtime
    make=contexts.ContextImports.make

    def ledger(self,router,**kwargs):
        def deny(*args):raise RuntimeError('Actual fixture provider not approved')
        args=dict(preloaded=dict(sys.modules),authenticate_preloaded=lambda *a:None,
            interpreter_modules={},authenticate_interpreter=deny,
            authenticate_native=deny,authenticate_alias=lambda *a:False)
        args.update(kwargs)
        ledger=origins.ModuleOriginAudit(router,**args)
        router.authenticate(ledger.bind_preloaded)
        return ledger

    def scope(self,router,ledger):
        return self.make(router,audit_modules=ledger.audit)

    def test_actual_source_import_records_original_loader_origin(self):
        name='_nano_import_origin';self.source(name,'VALUE=137\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.scope(router,ledger).installed():
                self.assertEqual(importlib.import_module(name).VALUE,137)
                result=ledger.result()
            self.assertIn(dict(name=name,kind='source'),result['module_origin_events'])
            self.assertFalse(result['complete_runtime_dependency_closure_proven'])

    def test_real_package_and_namespace_modules_are_audited(self):
        ns='_nano_import_origin_namespace';folder=self.root/ns;folder.mkdir()
        self.namespaces[ns]=[str(folder)]
        self.source(ns+'.leaf','VALUE=139\n')
        self.source('_nano_import_origin_package','from .leaf import VALUE\n',True)
        self.source('_nano_import_origin_package.leaf','VALUE=149\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.scope(router,ledger).installed():
                self.assertEqual(importlib.import_module(ns+'.leaf').VALUE,139)
                self.assertEqual(importlib.import_module('_nano_import_origin_package').VALUE,149)
                result=ledger.result()
            self.assertIn(dict(name=ns,kind='namespace'),result['module_origin_events'])

    def test_source_metadata_change_is_rejected_even_with_same_object(self):
        name='_nano_import_origin_tamper';self.source(name,'VALUE=151\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    module=importlib.import_module(name)
                    module.__file__='/unreviewed/source.py'
            self.assertTrue(router.poisoned)

    def test_source_loader_replacement_is_rejected(self):
        name='_nano_import_loader_tamper';self.source(name,'VALUE=157\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    module=importlib.import_module(name)
                    module.__spec__.loader=machinery.SourceFileLoader(name,module.__file__)

    def test_optional_failed_module_can_disappear_and_later_reimport(self):
        child='_nano_import_optional_origin_child'
        self.source(child,'raise ImportError("optional backend")\n')
        self.source('_nano_import_optional_origin_parent',
            'for i in range(2):\n try:\n  import '+child+'\n except ImportError:\n  pass\nVALUE=163\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.scope(router,ledger).installed():
                self.assertEqual(importlib.import_module('_nano_import_optional_origin_parent').VALUE,163)
                self.assertNotIn(child,sys.modules)
                ledger.audit()
            self.assertFalse(router.poisoned)

    def test_regular_reload_preserves_object_and_revalidates_metadata(self):
        name='_nano_import_origin_reload';self.source(name,'N=globals().get("N",0)+1\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.scope(router,ledger).installed():
                module=importlib.import_module(name)
                self.assertIs(importlib.reload(module),module)
                self.assertEqual(module.N,2)
                ledger.audit()

    def test_preloaded_origin_callback_denial_prevents_binding(self):
        with self.runtime()as finders:
            router=self.router(finders=finders)
            def deny(name,module):raise RuntimeError('Preloaded bytes not authenticated')
            with self.assertRaises(routing.ProvenanceError):
                self.ledger(router,authenticate_preloaded=deny)
            self.assertTrue(router.poisoned)

    def test_preloaded_name_set_must_be_complete(self):
        with self.runtime()as finders:
            router=self.router(finders=finders);pre=dict(sys.modules);pre.pop('sys')
            with self.assertRaises(routing.ProvenanceError):self.ledger(router,preloaded=pre)

    def test_preloaded_object_cannot_be_replaced(self):
        name='_nano_import_preloaded_identity';self.names.add(name)
        with self.runtime()as finders:
            original=types.ModuleType(name);sys.modules[name]=original
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():sys.modules[name]=types.ModuleType(name)

    def test_preloaded_spec_inplace_mutation_is_detected(self):
        name='_nano_import_preloaded_metadata';self.names.add(name)
        with self.runtime()as finders:
            module=types.ModuleType(name)
            module.__spec__=machinery.ModuleSpec(name,None,origin='fixture')
            sys.modules[name]=module
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    module.__spec__.origin='changed'

    def test_preloaded_none_preserved_but_new_none_denied(self):
        name='_nano_import_blocked_original';new='_nano_import_blocked_new'
        self.names.update([name,new])
        with self.runtime()as finders:
            sys.modules[name]=None
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    sys.modules[new]=None

    def test_unreviewed_sys_modules_injection_is_detected(self):
        name='_nano_import_unreviewed_registry';self.names.add(name)
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    sys.modules[name]=types.ModuleType(name)

    def test_approved_alias_requires_exact_module_object(self):
        alias='_nano_import_approved_alias';self.names.add(alias)
        original=types.ModuleType('_nano_import_alias_provider')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_alias=lambda n,m:n==alias and m is original)
            with self.scope(router,ledger).installed():
                sys.modules[alias]=original
                self.assertIn(dict(name=alias,kind='alias'),ledger.result()['module_origin_events'])

    def test_alias_permission_does_not_approve_same_named_replacement(self):
        alias='_nano_import_alias_spoof';self.names.add(alias)
        original=types.ModuleType(alias)
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_alias=lambda n,m:n==alias and m is original)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    sys.modules[alias]=types.ModuleType(alias)

    def test_interpreter_policy_is_explicit_even_for_plausible_builtin_metadata(self):
        name='_nano_import_synthetic_builtin';self.names.add(name)
        module=types.ModuleType(name)
        module.__spec__=machinery.ModuleSpec(name,machinery.BuiltinImporter,origin='built-in')
        module.__loader__=machinery.BuiltinImporter
        called=[]
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,interpreter_modules={name:dict(origin='built-in',fixture=True)},
                authenticate_interpreter=lambda *a:called.append(a))
            with self.scope(router,ledger).installed():
                sys.modules[name]=module
                ledger.audit()
            self.assertTrue(called)

    def test_native_looking_metadata_cannot_bypass_lifecycle_authorization(self):
        name='_nano_import_synthetic_native';self.names.add(name)
        path=str(self.root/(name+'.so'));self.modules[name]=dict(kind='native',path=path)
        module=types.ModuleType(name);module.__file__=path
        loader=machinery.ExtensionFileLoader(name,path)
        module.__spec__=machinery.ModuleSpec(name,loader,origin=path);module.__loader__=loader
        called=[]
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_native=lambda *a:called.append(a))
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():sys.modules[name]=module
            self.assertEqual(called,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
