"""Source regression plus actual fresh-process predecessor metadata check."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research'
OLD=R/'minicpm5_frame1_mlp_runner_v2.py';NEW=R/'minicpm5_frame1_mlp_runner_v3.py'

def funcs(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
def dump(node):return ast.dump(node,include_attributes=False)

class Checks(unittest.TestCase):
    def test_old_frozen(self):
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),'e72b23403791feab49cf214ad4cf67c048e28523404301bf83e122a0aa8c624d')
    def test_worker_frozen(self):
        self.assertEqual(hashlib.sha256((R/'minicpm5_frame1_mlp_worker_v2.py').read_bytes()).hexdigest(),'8fdd1a477e8e5159a77f3a4c176eea2120f4c0f970d9deef57e0a50a269ceb85')
    def test_replay_frozen(self):
        self.assertEqual(hashlib.sha256((R/'minicpm5_frame1_mlp_replay_v2.py').read_bytes()).hexdigest(),'8c1a96c664eee0775416d2ce9a6b1cbdf9d44ed04b42d074df66a855198b109e')
    def test_only_helper_added(self):
        self.assertEqual(set(funcs(NEW))-set(funcs(OLD)),{'load_history_helpers'})
    def test_all_other_functions_unchanged(self):
        old,new=funcs(OLD),funcs(NEW)
        for name in old:
            if name not in ('load_spec','outer'):
                self.assertEqual(dump(old[name]),dump(new[name]),name)
    def test_load_spec_only_adds_pinned_imports(self):
        old,new=funcs(OLD)['load_spec'],funcs(NEW)['load_spec']
        first=new.body.pop(0)
        self.assertEqual(dump(first),dump(ast.parse('load_history_helpers()').body[0]))
        self.assertEqual(dump(old),dump(new))
    def test_outer_same_checks_reordered(self):
        old,new=funcs(OLD)['outer'],funcs(NEW)['outer']
        # Original four setup statements, then historical read/check, become
        # historical read/check followed by those original four statements.
        self.assertEqual([dump(n)for n in new.body], [dump(n)for n in old.body[4:6]+old.body[:4]+old.body[6:]])
    def test_builder_science_construction_unchanged(self):
        old,new=funcs(Q/'prepare_minicpm5_frame1_runtime_v2.py'),funcs(Q/'prepare_minicpm5_frame1_runtime_v3.py')
        self.assertEqual({k:dump(v)for k,v in old.items()},{k:dump(v)for k,v in new.items()})
    def test_actual_fresh_predecessor_check(self):
        code="""import sys,json
sys.path.insert(0,'/home/ubuntu/experiments/trn1-latest-models-20261031')
import minicpm5_frame1_mlp_runner_v3 as r
assert 'smollm3_observer_invariance_environment_v2'not in sys.modules
assert 'current_finalizer'not in r.__dict__
previous=r.p.read_ref(r.BASE_SPEC)
bound,owners,priors=r.old.check_prerequisites(previous)
assert 'torch'not in sys.modules
r.load_history_helpers()
assert 'smollm3_observer_invariance_environment_v2'in sys.modules
assert 'current_finalizer'in r.__dict__
assert 'torch'not in sys.modules
print(json.dumps(dict(passed=True,actual_old_prerequisites=True,original_owner_rows=len(owners),current_helpers_loaded_after=True,native_imported=False)))
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=R,text=True,capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        proof=json.loads(result.stdout)
        self.assertTrue(proof['actual_old_prerequisites'])
        print('ACTUAL_PREDECESSOR_PROOF '+result.stdout.strip(),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
