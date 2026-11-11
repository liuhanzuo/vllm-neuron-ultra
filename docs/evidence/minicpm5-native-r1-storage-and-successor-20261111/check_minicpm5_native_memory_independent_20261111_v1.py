"""Independent small-source review checks; no original plans or runtime work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from research import check_minicpm5_native_source_phase_seal_v1 as fixture
from research import minicpm5_native_source_phase_seal_v3 as candidate


class Independent(unittest.TestCase):
    def test_complete_history_and_permit_never_sealed_and_exit_makes_no_snapshots(self):
        h=fixture.Harness();leaf={'shared':[{'escape':'雪\ud800','v':-0.0}]}
        h.resolved.update(current_scope={'original':leaf,'again':leaf},native_bundle=leaf)
        forbidden={id(h.resolved),id(h.resolved['current_scope']),id(leaf),id(h.permit)}
        original=candidate.seal_json;visited=[]
        def observed(value,*args,**kwargs):
            if type(value)in(dict,list):
                self.assertNotIn(id(value),forbidden);visited.append(id(value))
            return original(value,*args,**kwargs)
        with h.patches():
            context=h.context();phase=candidate.SourcePhase(h.contract,h.resolved,context,h.partition,h.lease,h.ports,fixture.sources)
            with patch.object(candidate,'seal_json',side_effect=observed):phase.__enter__()
            self.assertEqual(set(phase.snapshots),{'context_plan','contract','partition','catalog'})
            phase.verify_text_sources()
            with patch.object(candidate,'seal_json',side_effect=AssertionError('Exit must not copy containers')):
                self.assertFalse(phase.__exit__(None,None,None))
            self.assertIs(type(context.plan),dict);self.assertFalse(context._sources_checked)
            self.assertEqual(set(phase.__dict__),{'started','closed','authenticated','failure'})
        self.assertTrue(visited)

    def test_shared_history_leaf_changes_and_invalid_exit_types_latch(self):
        for fault in ('leaf','nan','cycle','tuple'):
            h=fixture.Harness();leaf={'rows':[{'value':1}]}
            h.resolved['current_scope']={'a':leaf,'b':leaf}
            with h.patches(),self.subTest(fault=fault):
                context=h.context()
                with self.assertRaises(RuntimeError):
                    with candidate.SourcePhase(h.contract,h.resolved,context,h.partition,h.lease,h.ports,fixture.sources)as phase:
                        phase.verify_text_sources()
                        if fault=='leaf':leaf['rows'][0]['value']=2
                        elif fault=='nan':leaf['new']=float('nan')
                        elif fault=='cycle':leaf['back']=leaf
                        else:leaf['new']=(1,2)
                self.assertFalse(context._sources_checked)
                with self.assertRaises(RuntimeError):context.guard()

    def test_stream_bytes_match_original_encoder_for_shared_and_unusual_scalars(self):
        shared={'z':['\\','\x00','\ud800','é雪',-0.0,5e-324,1.7976931348623157e308],
            'a':{'empty':{},'none':None,'false':False,'large':2**100}}
        value={'z':[shared]*5,'a':shared}
        expected=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.assertEqual(candidate.stream_digest(value),hashlib.sha256(expected).hexdigest())
        reversed_value=dict(reversed(list(value.items())))
        self.assertEqual(candidate.stream_digest(reversed_value),candidate.stream_digest(value))

    def test_only_phase_boundaries_override_frozen_dynamic_and_hashing_methods(self):
        tree=ast.parse((ROOT/'research/minicpm5_native_source_phase_seal_v3.py').read_bytes())
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        self.assertEqual({n.name for n in klass.body if isinstance(n,ast.FunctionDef)},{'_enter','__exit__'})
        for method in ('guard','verify_text_sources','_latch','_reject','_writers','__enter__','__init__'):
            self.assertIs(getattr(candidate.SourcePhase,method),getattr(fixture.seal.SourcePhase,method))
        for old,new,a,b in (
            ('minicpm5_layer1_operator_native_runtime_context_v2.py','minicpm5_layer1_operator_native_runtime_context_v3.py',
             'minicpm5_native_source_phase_seal_v2','minicpm5_native_source_phase_seal_v3'),
            ('minicpm5_layer1_operator_native_runner_v8.py','minicpm5_layer1_operator_native_runner_v9.py',
             'minicpm5_layer1_operator_native_runtime_context_v2','minicpm5_layer1_operator_native_runtime_context_v3')):
            self.assertEqual((ROOT/old).read_text().replace(a,b),(ROOT/new).read_text())


if __name__=='__main__':unittest.main(verbosity=2)
