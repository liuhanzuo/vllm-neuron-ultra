"""Synthetic entrypoint source only; no installed helper or SDK code executed."""
from contextlib import contextmanager
import hashlib
import types
import sys
import unittest
from unittest.mock import patch

import nemotron4b_guarded_helper_entry_v3 as entry

SOURCE=b'import sys\nprint("/synthetic/lib.so")\nsys.exit(None)\n'
EXPECTED=b'/synthetic/lib.so\n'
LIBRARY={'path':'/synthetic/lib.so','inode':1,'size':7}


class GuardedEntry(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.source=SOURCE
        self.identity=dict(LIBRARY)
        self.ref=dict(path='/synthetic/helper',sha256=hashlib.sha256(self.source).hexdigest())

    def guard(self):self.events.append('guard')
    def read(self,ref):
        self.assertEqual(ref,self.ref);self.events.append('read-source');return self.source
    def library(self):self.events.append('library');return dict(self.identity)
    def wait(self):self.events.append('release');return {'synthetic_parent_release':True}

    @contextmanager
    def imports(self):
        self.events.append('imports-enter')
        provenance={'synthetic_startup_context_only':True,'complete':False}
        try:yield provenance
        finally:self.events.append('imports-exit')
        provenance['complete']=True

    def invoke(self,**changes):
        args=dict(source_reference=self.ref,read_source=self.read,expected_stdout=EXPECTED,
                  library_identity=self.library,expected_library_identity=LIBRARY,
                  guard=self.guard,wait_for_parent=self.wait,startup_context=self.imports)
        args.update(changes)
        return entry.run(**args)

    def test_authenticated_source_waits_before_imports_and_restores_argv(self):
        before=sys.argv
        result=self.invoke()
        self.assertEqual(result['stdout'],EXPECTED)
        self.assertIs(sys.argv,before)
        self.assertLess(self.events.index('read-source'),self.events.index('release'))
        self.assertLess(self.events.index('release'),self.events.index('imports-enter'))
        self.assertEqual(result['source'],self.ref)
        self.assertEqual(result['library_before'],result['library_after'])
        self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['full_runtime_source_closure_proven_by_this_module'])

    def test_bad_source_stops_before_parent_release(self):
        self.source=SOURCE+b'#changed\n'
        with self.assertRaisesRegex(ValueError,'source authenticated'):self.invoke()
        self.assertNotIn('release',self.events)
        self.assertNotIn('imports-enter',self.events)

    def test_library_change_before_gate_prevents_release(self):
        self.identity['inode']=2
        with self.assertRaisesRegex(ValueError,'before gate'):self.invoke()
        self.assertNotIn('release',self.events)

    def test_library_change_after_gate_prevents_entry(self):
        def wait():self.identity['inode']=2;return self.wait()
        with self.assertRaisesRegex(ValueError,'after gate'):self.invoke(wait_for_parent=wait)
        self.assertNotIn('imports-enter',self.events)

    def test_library_change_after_execution_prevents_result(self):
        @contextmanager
        def imports():
            yield {'synthetic':True,'complete':True}
            self.identity['inode']=2
        with self.assertRaisesRegex(ValueError,'after startup exit'):self.invoke(startup_context=imports)

    def test_parent_denial_prevents_imports(self):
        def no_release():raise RuntimeError('parent denied')
        with self.assertRaisesRegex(RuntimeError,'parent denied'):self.invoke(wait_for_parent=no_release)
        self.assertNotIn('imports-enter',self.events)

    def test_preexisting_sdk_import_rejected(self):
        with patch.dict(sys.modules,{'libneuronxla.unexpected':object()}),self.assertRaisesRegex(ValueError,'before parent gate'):
            self.invoke()
        self.assertEqual(self.events,[])

    def test_original_nonzero_systemexit_value_is_preserved(self):
        for value in (17, -9, 'error', True):
            self.source=('raise SystemExit('+repr(value)+')').encode()
            self.ref['sha256']=hashlib.sha256(self.source).hexdigest()
            original=sys.argv
            with self.subTest(value=value),self.assertRaises(SystemExit) as caught:self.invoke()
            self.assertEqual(caught.exception.code,value)
            self.assertIs(sys.argv,original)

    def test_extra_or_wrong_stdout_cannot_qualify(self):
        for code in (b'print("/other/lib.so")',SOURCE.replace(b'sys.exit(None)',b'print("extra")')):
            self.source=code;self.ref['sha256']=hashlib.sha256(code).hexdigest()
            original=sys.argv
            with self.subTest(code=code),self.assertRaises(ValueError):self.invoke()
            self.assertIs(sys.argv,original)

    def test_source_namespace_and_argv_are_expected_but_os_argv_unchanged(self):
        self.source=b'import sys\nassert __name__ == "__main__"\nassert __file__ == "/synthetic/helper"\nassert sys.argv == [__file__]\nprint("/synthetic/lib.so")\n'
        self.ref['sha256']=hashlib.sha256(self.source).hexdigest()
        result=self.invoke()
        self.assertTrue(result['operating_system_argv_was_not_replaced'])

    def test_stdout_overflow_rejected_even_when_helper_catches_exception(self):
        self.source=b'try:\n print("x"*5000)\nexcept ValueError:\n pass\n'
        self.ref['sha256']=hashlib.sha256(self.source).hexdigest()
        with self.assertRaisesRegex(ValueError,'Failed helper stdout'):self.invoke()

    def test_import_context_failure_and_source_exception_restore_argv(self):
        @contextmanager
        def failing():
            raise RuntimeError('external import provenance missing')
            yield
        original=sys.argv
        with self.assertRaisesRegex(RuntimeError,'provenance'):self.invoke(startup_context=failing)
        self.assertIs(sys.argv,original)
        self.source=b'raise RuntimeError("entry failed")';self.ref['sha256']=hashlib.sha256(self.source).hexdigest()
        with self.assertRaisesRegex(RuntimeError,'entry failed'):self.invoke()
        self.assertIs(sys.argv,original)

    def test_parent_guard_failure_prevents_source_or_import_execution(self):
        for stop in (1,2):
            count=0
            def guard():
                nonlocal count
                count+=1
                if count==stop:raise RuntimeError('guard failed')
            self.events=[]
            with self.subTest(stop=stop),self.assertRaisesRegex(RuntimeError,'guard failed'):self.invoke(guard=guard)
            self.assertNotIn('imports-enter',self.events)


    def test_old_entry_order_reproduces_wrapper_argv_startup_rejection(self):
        import nemotron4b_guarded_helper_entry_v2 as previous
        original=sys.argv
        @contextmanager
        def startup_requires_wrapper():
            if sys.argv is not original:raise ValueError('fixture startup expected wrapper argv')
            yield {'complete':True}
        with self.assertRaisesRegex(ValueError,'expected wrapper argv'):
            previous.run(source_reference=self.ref,read_source=self.read,expected_stdout=EXPECTED,
                library_identity=self.library,expected_library_identity=LIBRARY,guard=self.guard,
                wait_for_parent=self.wait,import_context=startup_requires_wrapper)
        self.assertIs(sys.argv,original)

    def test_startup_and_its_exit_observe_wrapper_argv_while_source_observes_helper(self):
        original=sys.argv;events=[]
        self.source=b'import sys\nassert sys.argv == ["/synthetic/helper"]\nprint("/synthetic/lib.so")\n'
        self.ref['sha256']=hashlib.sha256(self.source).hexdigest()
        @contextmanager
        def startup():
            self.assertIs(sys.argv,original);self.assertIn('release',self.events)
            events.append('startup-enter');provenance={'complete':False}
            yield provenance
            self.assertIs(sys.argv,original);events.append('startup-exit');provenance['complete']=True
        result=self.invoke(startup_context=startup)
        self.assertEqual(events,['startup-enter','startup-exit'])
        self.assertTrue(result['startup']['complete'])
        self.assertIs(result['startup'],result['import_provenance'])
        self.assertTrue(result['original_wrapper_argv_restored'])

    def test_unfinished_startup_cannot_return_success(self):
        @contextmanager
        def unfinished():yield {'complete':False}
        with self.assertRaisesRegex(ValueError,'context completed'):self.invoke(startup_context=unfinished)

    def test_startup_argv_mutation_rejected_before_helper_body(self):
        original=sys.argv
        @contextmanager
        def changed():
            sys.argv=['changed-wrapper']
            try:yield {'complete':True}
            finally:sys.argv=original
        with self.assertRaisesRegex(ValueError,'wrapper argv retained'):self.invoke(startup_context=changed)
        self.assertIs(sys.argv,original)


class DeferredStartup(unittest.TestCase):
    def setUp(self):
        from nemotron4b_participant_invocation_v1 import ParticipantBinding
        from nemotron4b_phase_origin_context_v1 import PhaseOriginContext
        from nemotron4b_deferred_participant_startup_v1 import DeferredParticipantStartup
        self.events=[];self.complete=True;self.original_argv=sys.argv
        self.cache=types.SimpleNamespace(check_runtime=lambda:self.events.append('cache'))
        self.binding=object.__new__(ParticipantBinding)
        self.binding.role='helper-0';self.binding.guard=lambda:self.events.append('guard')
        self.binding.release=None;self.binding.argv=['python','-B','-S','-X','prefix',*sys.argv]
        self.binding.attempt=types.SimpleNamespace(caches={'helper-0':self.cache},bootstrap={'path':sys.argv[0]})
        self.origin=object.__new__(PhaseOriginContext)
        self.origin.cache=self.cache;self.origin.guard=self.binding.guard
        @contextmanager
        def restore(policy):
            self.assertEqual(policy['pre_argv'],sys.argv);self.events.append('startup-enter')
            value={'fixture_origin_only':True,'complete':False}
            try:yield value
            finally:self.events.append('startup-exit')
            value['complete']=self.complete
        self.origin.restored_startup=restore
        def build(binding):
            self.assertIs(binding,self.binding);self.assertIsNotNone(binding.release)
            self.events.append('build-origin');return self.origin
        self.start=DeferredParticipantStartup(self.binding,
            dict(script=sys.argv[0],pre_argv=[sys.argv[0]]),build_origin=build)

    def test_no_origin_builder_runs_before_gate(self):
        with self.assertRaisesRegex(ValueError,'parent grant'):
            with self.start():pass
        self.assertNotIn('build-origin',self.events)

    def test_origin_built_after_gate_once_with_same_cache_guard_and_bound_argv(self):
        self.binding.release={'explicit_generated_release_fixture':True}
        with self.start()as value:
            self.assertTrue(value['fixture_origin_only'])
            self.assertEqual(self.events.count('build-origin'),1)
            self.events.append('body')
        self.assertLess(self.events.index('build-origin'),self.events.index('startup-enter'))
        self.assertLess(self.events.index('body'),self.events.index('startup-exit'))
        self.assertTrue(self.start.provenance['complete'])
        with self.assertRaisesRegex(ValueError,'One deferred'):
            with self.start():pass

    def test_different_origin_cache_rejected(self):
        self.binding.release={'fixture':True};self.origin.cache=object()
        with self.assertRaisesRegex(ValueError,'Same actual participant'):
            with self.start():pass
        self.assertNotIn('startup-enter',self.events)

    def test_body_failure_keeps_unfinished_provenance(self):
        self.binding.release={'fixture':True}
        with self.assertRaisesRegex(RuntimeError,'generated body failure'):
            with self.start():raise RuntimeError('generated body failure')
        self.assertFalse(self.start.provenance['complete'])
        self.assertIn('startup-exit',self.events)

    def test_incomplete_startup_cannot_qualify(self):
        self.binding.release={'fixture':True};self.complete=False
        with self.assertRaisesRegex(ValueError,'startup completed'):
            with self.start():pass

    def test_changed_wrapper_argv_prevents_building_origins(self):
        self.binding.release={'fixture':True}
        try:
            sys.argv=['wrong-wrapper']
            with self.assertRaisesRegex(ValueError,'bound wrapper argv'):
                with self.start():pass
        finally:sys.argv=self.original_argv
        self.assertNotIn('build-origin',self.events)


if __name__=='__main__':unittest.main(verbosity=2)
