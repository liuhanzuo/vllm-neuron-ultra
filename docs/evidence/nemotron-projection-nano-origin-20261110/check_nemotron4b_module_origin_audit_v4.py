"""Exercise origin ledger with real synthetic imports and explicit fixture policies."""
import importlib
import importlib.machinery as machinery
from pathlib import Path
import sys
import types
import typing
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import check_nemotron4b_import_router_v4_r2 as fixtures
import check_nemotron4b_import_context_v3 as contexts
import nemotron4b_import_router_v4 as routing
import nemotron4b_module_origin_audit_v4 as origins


class OriginImports(unittest.TestCase):
    setUp=fixtures.Imports.setUp
    source=fixtures.Imports.source
    router=fixtures.Imports.router
    runtime=contexts.ContextImports.runtime
    make=contexts.ContextImports.make

    def approve_typing_fixture(self,name,value,owner):
        # Fixture identity policy only; no claim of actual source authentication.
        self.assertIs(owner,typing)
        self.assertIs(value,vars(typing)[name.split('.')[1]])
        self.assertIs(value,sys.modules[name])
        return True

    def ledger(self,router,**kwargs):
        def deny(*args):raise RuntimeError('Actual fixture provider not approved')
        args=dict(preloaded=dict(sys.modules),authenticate_preloaded=lambda *a:None,
            interpreter_modules={},authenticate_interpreter=deny,
            authenticate_native=deny,authenticate_alias=lambda *a:False,
            authenticate_typing_namespace=self.approve_typing_fixture)
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


    def test_real_typing_class_entries_bind_and_survive_source_import(self):
        name='_nano_import_typing_origin';self.source(name,'VALUE=173\n')
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            with self.scope(router,ledger).installed():
                self.assertEqual(importlib.import_module(name).VALUE,173)
                result=ledger.result()
            for key in ('typing.io','typing.re'):
                self.assertIn(dict(name=key,kind='preloaded-typing-namespace'),
                              result['module_origin_events'])
            self.assertFalse(result['preloaded_code_proven_by_this_module'])
            self.assertFalse(result['actual_SDK_runtime_qualified'])

    def test_typing_owner_callback_is_required(self):
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                self.ledger(router,authenticate_typing_namespace=None)
            self.assertTrue(router.poisoned)

    def test_typing_approval_requires_true_boolean(self):
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                self.ledger(router,authenticate_typing_namespace=lambda *a:1)

    def test_typing_owner_authentication_failure_propagates(self):
        calls=[]
        def deny(*args):
            calls.append(args)
            raise RuntimeError('Original typing source not authenticated')
        with self.runtime()as finders:
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                self.ledger(router,authenticate_typing_namespace=deny)
            self.assertEqual(len(calls),1)

    def test_typing_owner_callback_rechecked_after_binding(self):
        permitted=[True]
        def check(*args):return permitted[0]
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_typing_namespace=check)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    permitted[0]=False

    def test_unknown_preloaded_class_is_rejected_even_with_alias_approval(self):
        name='_nano_import_unreviewed_class';self.names.add(name)
        with self.runtime()as finders:
            sys.modules[name]=type('Unexpected',(),{})
            router=self.router(finders=finders)
            with self.assertRaises(routing.ProvenanceError):
                self.ledger(router,authenticate_typing_namespace=lambda *a:True,
                            authenticate_alias=lambda *a:True)

    def test_new_class_entry_cannot_use_general_module_alias_policy(self):
        name='_nano_import_new_class';self.names.add(name)
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_alias=lambda *a:True)
            with self.assertRaises(routing.ProvenanceError):
                with self.scope(router,ledger).installed():
                    sys.modules[name]=vars(typing)['io']

    def test_typing_registry_replacement_rejected(self):
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            original=sys.modules['typing.io']
            try:
                with self.assertRaises(routing.ProvenanceError):
                    with self.scope(router,ledger).installed():
                        sys.modules['typing.io']=type('typing.io',(),{})
            finally:
                sys.modules['typing.io']=original

    def test_typing_export_mutation_rejected(self):
        value=vars(typing)['io'];old=vars(value)['IO']
        with self.runtime()as finders:
            router=self.router(finders=finders);ledger=self.ledger(router)
            try:
                with self.assertRaises(routing.ProvenanceError):
                    with self.scope(router,ledger).installed():
                        type.__setattr__(value,'IO',object())
            finally:
                type.__setattr__(value,'IO',old)

    def test_typing_metadata_cannot_use_module_transition_escape(self):
        value=vars(typing)['io'];exports=vars(value)['__all__'];old=list(exports)
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=self.ledger(router,authenticate_preloaded_transition=lambda *a:True)
            try:
                with self.assertRaises(routing.ProvenanceError):
                    with self.scope(router,ledger).installed():
                        exports.reverse()
            finally:
                exports[:]=old


if __name__=='__main__':
    unittest.main(verbosity=2)
