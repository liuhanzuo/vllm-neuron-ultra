"""Real typing source/code and synthetic external-origin callbacks; no SDK."""
import copy
import importlib
import importlib.machinery as machinery
import json
from pathlib import Path
import sys
import types
import typing
import unittest
from unittest.mock import patch

import nemotron4b_typing_source_policy_v1 as policy
from nemotron4b_empty_bytecode_cache_v1 import EmptyBytecodeCache,record_empty_directory
import nemotron4b_module_origin_audit_v4 as origins
import check_nemotron4b_import_router_v4_r2 as fixtures
import check_nemotron4b_import_context_v3 as contexts

Q=Path(__file__).resolve().parent


def guard():
    assert not {'site','torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules)


class TypingPolicy(unittest.TestCase):
    source=fixtures.Imports.source
    router=fixtures.Imports.router
    runtime=contexts.ContextImports.runtime
    context=contexts.ContextImports.make

    def setUp(self):
        fixtures.Imports.setUp(self)
        phase='neuron'if sys.version_info[:2]==(3,10)else'native'
        self.manifest=json.loads((Q/'nemotron4b-typing-code-observation-20261110-v1'/(phase+'-typing-source-manifest.json')).read_text())
        self.cache=EmptyBytecodeCache(record_empty_directory(sys.pycache_prefix,guard=guard),guard=guard)
        self.cache.audit_preloaded_source_caches()
        self.calls=[]

    def wrapper_fixture(self,name,value):
        # Fixture approval only; original native functools code is not hashed.
        self.calls.append(name)
        return name=='_lazy_load_getattr_static'and value is vars(typing)[name]

    def make(self,**kwargs):
        args=dict(cache=self.cache,guard=guard,authenticate_external=self.wrapper_fixture)
        args.update(kwargs)
        return policy.TypingSourcePolicy(self.manifest,**args)

    def test_actual_source_definitions_and_original_namespace_callbacks(self):
        p=self.make()
        for n in('typing.io','typing.re'):
            self.assertTrue(p.authenticate_namespace(n,sys.modules[n],typing))
        self.assertTrue(p.authenticate_owner('typing',typing))
        r=p.result()
        self.assertEqual(r['functions_checked'],185 if sys.version_info[:2]==(3,10)else 200)
        self.assertTrue(r['original_source_hash_and_code_definitions_checked'])
        self.assertFalse(r['arbitrary_globals_or_mutable_closure_contents_authenticated'])
        self.assertFalse(r['runtime_admitted'])

    def test_concrete_callback_connects_to_actual_origin_ledger(self):
        p=self.make();name='_nano_import_typing_source_policy';self.source(name,'VALUE=181\n')
        def deny(*args):raise ValueError('Not a fixture native/interpreter origin')
        def preloaded(n,m):
            if n=='typing':p.authenticate_owner(n,m)
        with self.runtime()as finders:
            router=self.router(finders=finders)
            ledger=origins.ModuleOriginAudit(router,preloaded=dict(sys.modules),
                authenticate_preloaded=preloaded,interpreter_modules={},
                authenticate_interpreter=deny,authenticate_native=deny,
                authenticate_alias=lambda *a:False,
                authenticate_typing_namespace=p.authenticate_namespace)
            router.authenticate(ledger.bind_preloaded)
            with self.context(router,audit_modules=ledger.audit).installed():
                self.assertEqual(importlib.import_module(name).VALUE,181)
                ledger.audit()
            self.assertGreater(p.checks,2)

    def test_wrong_pinned_source_digest_rejected(self):
        self.manifest['source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):self.make()

    def test_wrong_interpreter_version_rejected(self):
        self.manifest['version']=[3,9,0]
        with self.assertRaisesRegex(ValueError,'interpreter version'):self.make()

    def test_changed_reviewed_code_definition_rejected(self):
        next(iter(self.manifest['functions'].values()))['code_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'reviewed typing'):self.make()

    def test_incomplete_external_wrapper_manifest_rejected(self):
        self.manifest['external_wrappers'].append('not_original')
        with self.assertRaisesRegex(ValueError,'reviewed typing'):self.make()

    def test_missing_source_cache_observation_rejected(self):
        self.cache.checked_sources=None
        with self.assertRaisesRegex(ValueError,'cache exclusion'):self.make()

    def test_guard_denial_precedes_source_read(self):
        def deny():raise RuntimeError('Source-only guard denied')
        with patch.object(policy,'read_source')as reader:
            with self.assertRaisesRegex(RuntimeError,'guard denied'):self.make(guard=deny)
            reader.assert_not_called()

    def test_live_owner_replacement_rejected(self):
        p=self.make()
        with patch.dict(sys.modules,{'typing':types.ModuleType('typing')}):
            with self.assertRaisesRegex(ValueError,'source loader'):p.check()

    def test_equivalent_replacement_code_object_is_rejected(self):
        p=self.make();f=vars(typing)['_type_check'];old=f.__code__
        try:
            f.__code__=old.replace()
            self.assertEqual(old,f.__code__)
            with self.assertRaisesRegex(ValueError,'bindings retained'):p.check()
        finally:f.__code__=old

    def test_same_code_replacement_function_is_rejected(self):
        p=self.make();old=vars(typing)['_type_check']
        clone=types.FunctionType(old.__code__,old.__globals__,old.__name__,old.__defaults__,old.__closure__)
        with patch.object(typing,'_type_check',clone):
            with self.assertRaisesRegex(ValueError,'bindings retained'):p.check()

    def test_default_tuple_identity_change_rejected(self):
        p=self.make();f=vars(typing)['_type_check'];old=f.__defaults__
        self.assertTrue(old)
        try:
            f.__defaults__=tuple(list(old))
            self.assertIsNot(f.__defaults__,old)
            with self.assertRaisesRegex(ValueError,'bindings retained'):p.check()
        finally:f.__defaults__=old

    def test_namespace_export_change_rejected(self):
        p=self.make();value=vars(typing)['io'];old=vars(value)['IO']
        try:
            type.__setattr__(value,'IO',object())
            with self.assertRaisesRegex(ValueError,'namespace exports'):p.check()
        finally:type.__setattr__(value,'IO',old)

    def test_namespace_callback_cannot_approve_other_owner(self):
        p=self.make()
        with self.assertRaisesRegex(ValueError,'owning typing'):
            p.authenticate_namespace('typing.io',sys.modules['typing.io'],types.ModuleType('typing'))

    def test_loader_replacement_with_same_path_rejected(self):
        p=self.make();old=typing.__loader__
        new=machinery.SourceFileLoader('typing',typing.__file__)
        try:
            typing.__loader__=new;typing.__spec__.loader=new
            with self.assertRaisesRegex(ValueError,'object identity'):p.check()
        finally:
            typing.__loader__=old;typing.__spec__.loader=old

    def test_caller_manifest_rebinding_does_not_change_policy(self):
        p=self.make();self.manifest['source']['sha256']='0'*64
        self.assertTrue(p.check())

    @unittest.skipUnless(sys.version_info[:2]==(3,12),'Original 3.10 typing has no native cache wrapper')
    def test_external_origin_denial_rejected(self):
        with self.assertRaisesRegex(ValueError,'wrapper origin'):
            self.make(authenticate_external=lambda *a:False)

    @unittest.skipUnless(sys.version_info[:2]==(3,12),'Original 3.10 typing has no native cache wrapper')
    def test_external_origin_truthy_nonboolean_rejected(self):
        with self.assertRaisesRegex(ValueError,'wrapper origin'):
            self.make(authenticate_external=lambda *a:1)

    @unittest.skipUnless(sys.version_info[:2]==(3,12),'Original 3.10 typing has no native cache wrapper')
    def test_external_origin_rechecked_and_revocation_rejected(self):
        permitted=[True]
        p=self.make(authenticate_external=lambda *a:permitted[0])
        self.assertTrue(p.check());permitted[0]=False
        with self.assertRaisesRegex(ValueError,'wrapper origin'):p.check()


if __name__=='__main__':unittest.main(verbosity=2)
