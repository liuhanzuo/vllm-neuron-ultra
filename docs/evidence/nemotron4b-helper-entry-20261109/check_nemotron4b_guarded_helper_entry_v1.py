"""Synthetic entrypoint source only; no installed helper or SDK code executed."""
from contextlib import contextmanager
import hashlib
import sys
import unittest
from unittest.mock import patch

import nemotron4b_guarded_helper_entry_v1 as entry

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
        try:yield {'synthetic_import_context_only':True}
        finally:self.events.append('imports-exit')

    def invoke(self,**changes):
        args=dict(source_reference=self.ref,read_source=self.read,expected_stdout=EXPECTED,
                  library_identity=self.library,expected_library_identity=LIBRARY,
                  guard=self.guard,wait_for_parent=self.wait,import_context=self.imports)
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
            yield {'synthetic':True}
            self.identity['inode']=2
        with self.assertRaisesRegex(ValueError,'after helper'):self.invoke(import_context=imports)

    def test_parent_denial_prevents_imports(self):
        def no_release():raise RuntimeError('parent denied')
        with self.assertRaisesRegex(RuntimeError,'parent denied'):self.invoke(wait_for_parent=no_release)
        self.assertNotIn('imports-enter',self.events)

    def test_preexisting_sdk_import_rejected(self):
        with patch.dict(sys.modules,{'libneuronxla.unexpected':object()}),self.assertRaisesRegex(ValueError,'before parent gate'):
            self.invoke()
        self.assertEqual(self.events,[])

    def test_nonzero_exit_or_extra_stdout_cannot_qualify(self):
        for code in (b'raise SystemExit(17)',b'raise SystemExit("error")',
                     b'print("/other/lib.so")',SOURCE.replace(b'sys.exit(None)',b'print("extra")')):
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
        with self.assertRaisesRegex(RuntimeError,'provenance'):self.invoke(import_context=failing)
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


if __name__=='__main__':unittest.main(verbosity=2)
