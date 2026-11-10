"""Timing hook transparency and exception cleanup; synthetic functions only."""
import json
import sys
import unittest
import run_nemotron9b_preexec_timing_v1 as r

def known(x):
    value=x+3
    value*=2
    return value

def broken():
    raise LookupError('original error retained')

class Timing(unittest.TestCase):
    def test_untraced_keeps_result_and_collects_no_line_samples(self):
        actual,t=r.measure(known,(4,))
        self.assertEqual(actual,14);self.assertEqual(t['lines'],{})
        self.assertGreaterEqual(t['wall_seconds'],0);self.assertIsNone(sys.gettrace())

    def test_trace_keeps_result_function_identity_and_finite_accounting(self):
        code=known.__code__
        actual,t=r.measure(known,(4,),lines=True)
        self.assertEqual(actual,14);self.assertIs(known.__code__,code)
        self.assertTrue(t['lines'])
        self.assertLessEqual(sum(x['seconds']for x in t['lines'].values()),t['wall_seconds'])
        self.assertTrue(all(x['visits']>0 and x['seconds']>=0 for x in t['lines'].values()))
        json.dumps(t,allow_nan=False);self.assertIsNone(sys.gettrace())

    def test_exception_is_not_hidden_and_trace_is_restored(self):
        for traced in (False,True):
            with self.subTest(traced=traced),self.assertRaisesRegex(LookupError,'original error retained'):
                r.measure(broken,(),lines=traced)
            self.assertIsNone(sys.gettrace())

    def test_existing_trace_is_preserved_and_rejected(self):
        original=lambda *args:None
        sys.settrace(original)
        try:
            with self.assertRaisesRegex(RuntimeError,'existing trace'):r.measure(known,(4,),lines=True)
            self.assertIs(sys.gettrace(),original)
        finally:sys.settrace(None)

if __name__=='__main__':unittest.main(verbosity=2)
