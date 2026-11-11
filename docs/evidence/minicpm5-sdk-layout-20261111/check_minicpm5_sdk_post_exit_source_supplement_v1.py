"""Bounded stdlib source/synthetic regressions; never executes an actual audit."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import minicpm5_sdk_post_exit_source_supplement_v1 as s


def parsed(name):return ast.parse((s.Q/name).read_text())
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)


class Sources(unittest.TestCase):
    def setUp(self):
        self.plan={'source_sha256':{str(s.ROOT/'frozen.py'):'a'*64},'scientific_sentinel':{'all_inputs':['unchanged']}}
        self.pins={path:hashlib.sha256(b'synthetic-source').hexdigest()for path in s.SOURCE_PATHS}
        self.review=dict(schema=s.SCHEMA,reviewer='/root',passed=True,blockers=[],model_exit=copy.deepcopy(s.MODEL_EXIT),
            model_source_map_sha256=s.digest(self.plan['source_sha256']),source_sha256=self.pins,
            original_model_sources_preserved=True,frozen_model_closure_unchanged=True)
        self.ref=dict(path=str(s.Q/'minicpm5-sdk-post-exit-source-supplement-synthetic.json'),sha256='b'*64)

    def test_exact_four_sources_original_model_binding_and_no_replace(self):
        self.assertEqual(s.validate_review(self.review,s.MODEL_EXIT,self.plan),self.pins)
        self.assertEqual(len(self.pins),4)
        for key,value in [('passed',False),('reviewer','/child'),('blockers',['open']),
                          ('model_exit',dict(s.MODEL_EXIT,sha256='c'*64)),('model_source_map_sha256','d'*64),
                          ('original_model_sources_preserved',False),('frozen_model_closure_unchanged',False)]:
            r=copy.deepcopy(self.review);r[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):s.validate_review(r,s.MODEL_EXIT,self.plan)
        for pins in [dict(self.pins,**{str(s.Q/'arbitrary.py'):'f'*64}),dict(list(self.pins.items())[:-1]),
                     {str(s.Q/Path(p).name.replace('v1','v9')):v for p,v in self.pins.items()}]:
            with self.assertRaises(RuntimeError):s.validate_review(dict(self.review,source_sha256=pins),s.MODEL_EXIT,self.plan)
        other=copy.deepcopy(self.plan);other['source_sha256'].update(self.pins)
        review=dict(self.review,model_source_map_sha256=s.digest(other['source_sha256']))
        with self.assertRaises(RuntimeError):s.validate_review(review,s.MODEL_EXIT,other)

    def test_original_verifier_runs_on_explicit_union_only_plan_unchanged(self):
        before=copy.deepcopy(self.plan);captured=[]
        def original(view,guard):
            captured.append(view)
            self.assertIsNot(view,self.plan)
            self.assertEqual(view['source_sha256'],dict(self.plan['source_sha256'],**self.pins))
            self.assertEqual(view['scientific_sentinel'],self.plan['scientific_sentinel'])
            return dict(complete=True,count=len(view['source_sha256']),source_sha256=dict(view['source_sha256']))
        runner=SimpleNamespace(verify_loaded_sources=Mock(side_effect=original));guard=Mock()
        with patch.object(s.sources,'load',return_value=self.review),patch.object(s.sources,'text',return_value=b'synthetic-source')as read,patch.object(s,'identity',return_value={'stable':True}):
            result=s.verify_loaded_sources(runner,self.plan,guard,self.ref,s.MODEL_EXIT)
        runner.verify_loaded_sources.assert_called_once();self.assertEqual(self.plan,before)
        self.assertEqual({c.args[0]for c in read.call_args_list},set(self.pins))
        self.assertEqual(result['supplement_source_sha256'],self.pins)
        self.assertEqual(result['post_exit_source_supplement'],self.ref)

    def test_changed_supplement_hash_or_metadata_rejected_before_original_verifier(self):
        runner=SimpleNamespace(verify_loaded_sources=Mock())
        with patch.object(s.sources,'load',return_value=self.review),patch.object(s.sources,'text',return_value=b'changed'),patch.object(s,'identity',return_value={'stable':True}):
            with self.assertRaises(RuntimeError):s.verify_loaded_sources(runner,self.plan,lambda:None,self.ref,s.MODEL_EXIT)
        runner.verify_loaded_sources.assert_not_called()
        with patch.object(s.sources,'load',return_value=self.review),patch.object(s,'identity',side_effect=[{'inode':1},{'inode':2}]):
            with self.assertRaises(RuntimeError):s.verify_loaded_sources(runner,self.plan,lambda:None,self.ref,s.MODEL_EXIT)
        runner.verify_loaded_sources.assert_not_called()

    def test_original_verifier_failure_not_skipped(self):
        runner=SimpleNamespace(verify_loaded_sources=Mock(side_effect=RuntimeError('Unknown loaded helper')))
        with patch.object(s.sources,'load',return_value=self.review),patch.object(s.sources,'text',return_value=b'synthetic-source'),patch.object(s,'identity',return_value={'stable':True}):
            with self.assertRaisesRegex(RuntimeError,'Unknown loaded helper'):s.verify_loaded_sources(runner,self.plan,lambda:None,self.ref,s.MODEL_EXIT)

    def test_ast_original_admission_bodies_exact_except_explicit_binding_and_source_call(self):
        old=ast.parse((s.ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v4.py').read_text())
        new=parsed('minicpm5_sdk_post_exit_source_supplement_v1.py')
        bindings={'p','source_auth','latest','same','SELF','ROOT','CAP','PROPS','OBSERVED_EXIT',
                  'RESULTS_SOURCE_PLAN','validate_current_history_extension'}
        class Expected(ast.NodeTransformer):
            def visit_Name(self,n):
                if isinstance(n.ctx,ast.Load)and n.id in bindings:
                    return ast.Attribute(value=ast.Name(id='runner',ctx=ast.Load()),attr=n.id,ctx=ast.Load())
                return n
            def visit_Call(self,n):
                if isinstance(n.func,ast.Name)and n.func.id=='verify_loaded_sources':
                    return ast.Call(func=n.func,args=[ast.Name(id='runner',ctx=ast.Load()),*n.args,
                        ast.Name(id='supplement_ref',ctx=ast.Load()),ast.Name(id='model_exit',ctx=ast.Load())],keywords=[])
                return self.generic_visit(n)
        for name in ('read_bundle','admitted_bundles'):
            original=function(old,name);expected=Expected().visit(copy.deepcopy(original))
            expected.args.args.insert(0,ast.arg(arg='runner'))
            expected.args.args.extend([ast.arg(arg='supplement_ref'),ast.arg(arg='model_exit')]);expected.args.defaults=[]
            self.assertEqual(dump(expected),dump(function(new,name)),name)
            self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='verify_loaded_sources'for n in ast.walk(original)),1)

    def test_frozen_raw_verifier_guards_codec_limits_legacy_extraction_unchanged(self):
        old=parsed('audit_minicpm5_layer1_sdk_batch_layout_result_v4.py')
        new=parsed('audit_minicpm5_layer1_sdk_batch_layout_result_v5.py')
        for name in ('output_path','packed_physical','verify_outer_drain','legacy_drain_units','verify_rank_guards'):
            self.assertEqual(dump(function(old,name)),dump(function(new,name)),name)
        for name in ('LIMITS','GUARD_BOUNDARIES'):
            def assignment(t):return next(n for n in t.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==name for x in n.targets))
            self.assertEqual(dump(assignment(old)),dump(assignment(new)))
        a=function(old,'audit');b=function(new,'audit')
        self.assertEqual(dump(function(a,'pair')),dump(function(b,'pair')))
        def calls(t,name):return [dump(n) for n in ast.walk(t) if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr==name]
        for name in ('recompute','verify_owned_drain','validate_native_receipt','check_native_identities','verify_text_sources','usage'):
            self.assertEqual(calls(a,name),calls(b,name),name)
        old=parsed('record_minicpm5_layer1_sdk_batch_layout_audit_exit_v4.py');new=parsed('record_minicpm5_layer1_sdk_batch_layout_audit_exit_v5.py')
        self.assertEqual(dump(function(old,'unpack')),dump(function(new,'unpack')))

    def test_no_native_imports(self):self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
