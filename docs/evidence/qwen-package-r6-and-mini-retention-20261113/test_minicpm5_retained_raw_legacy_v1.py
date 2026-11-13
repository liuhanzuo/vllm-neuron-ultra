"""AST/source and synthetic tests only; never import original model/replay code."""
import ast
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import minicpm5_retained_raw_transport_v1 as t
import minicpm5_retained_raw_legacy_v1 as legacy
from test_minicpm5_retained_raw_transport_v1 import fixture


class SourcePreservation(unittest.TestCase):
    def function(self,path,name):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)

    def test_full_scientific_loop_has_only_explicit_tensor_injection_and_final_identity_change(self):
        old=self.function(t.ROOT/'run_minicpm5_history_analysis_recovery_v3.py','main')
        new=self.function(Path(legacy.__file__),'replay_analysis')
        old.name=new.name;old.args=new.args
        old.body=[n for n in old.body if not(isinstance(n,ast.Import)and any(a.name=='minicpm5_history_tensors_v1'for a in n.names))]
        loop=next(n for n in old.body if isinstance(n,ast.For)and ast.unparse(n.iter)=="plan['analysis_payloads'].items()")
        i=old.body.index(loop);old.body[i:i+1]=ast.parse('p.verify_payload_identities(plan)').body
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_full_protocol_verifier_preserves_every_other_statement(self):
        old=self.function(t.ROOT/'minicpm5_history_recovery_protocol_v3.py','verify')
        new=self.function(Path(legacy.__file__),'verify_protocol_metadata')
        class Unqualify(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name)and node.value.id=='protocol':return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
                return self.generic_visit(node)
        new=Unqualify().visit(new);new.name=old.name;new.args=old.args
        loop=next(n for n in new.body if isinstance(n,ast.For)and ast.unparse(n.iter)=="plan['analysis_payloads'].items()")
        loop.body=ast.parse("assert identity(name)==row['identity'],name").body
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_no_frozen_module_patch_or_native_top_level_import(self):
        tree=ast.parse(Path(legacy.__file__).read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[n.name for n in node.names]if isinstance(node,ast.Import)else[node.module]
                self.assertFalse(any(n.startswith(('torch','numpy','minicpm5_history_'))for n in names))
        self.assertNotIn('setattr(',Path(legacy.__file__).read_text())


class Interfaces(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-legacy-test-',dir='/tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan=fixture(self.root)
        ref=t._verify_pairs(self.plan,self.root/'verified.json',lambda:None)
        payloads=legacy.Payloads.__new__(legacy.Payloads)
        payloads.plan={'analysis_payloads':{r['original']['realpath']:dict(identity=r['original'],sha256=r['sha256'])for r in self.plan['entries']}}
        payloads.replay=t.Replay(self.plan,ref,lambda:None,_synthetic=True)
        payloads.guard=lambda:None;payloads.resolutions={};self.payloads=payloads

    def test_capture_resolution_never_reports_historical_inode_as_live(self):
        row=self.plan['entries'][0];Path(row['original']['realpath']).unlink()
        proof=self.payloads.check_capture(row['original']['realpath'],self.payloads.plan['analysis_payloads'][row['original']['realpath']])
        self.assertEqual(proof['capture_identity'],row['original'])
        self.assertEqual(proof['storage_identity'],row['keeper'])
        self.assertFalse(proof['original_identity_is_current_stat']);self.assertTrue(proof['relocated'])

    def test_full_logical_admission_keeps_logical_keys_and_hashes_after_retirement(self):
        for row in self.plan['entries']:Path(row['original']['realpath']).unlink()
        result=self.payloads.admit_all(self.payloads.plan)
        self.assertEqual(result['sha256'],{r['original']['realpath']:r['sha256']for r in self.plan['entries']})
        self.assertEqual(set(result['physical_resolutions']),set(result['sha256']))

    def test_evidence_same_code_isolated_and_capture_and_live_maps_separate(self):
        module=types.ModuleType('synthetic_evidence')
        exec('TEXT={}\nSTATS={}\nMANIFESTS={}\ndef payload_stat(*args):\n raise AssertionError("old IO used")\ndef manifest(path,identity,digest):\n return payload_stat(path,identity,digest)\n',module.__dict__)
        module.checks=types.SimpleNamespace(check_hash=lambda d:None)
        view=legacy.evidence_view(module,self.payloads);row=self.plan['entries'][0]
        Path(row['original']['realpath']).unlink()
        captured=view.manifest(row['original']['realpath'],row['original'],row['sha256'])
        self.assertEqual(captured,row['original']);self.assertIs(view.manifest.__code__,module.manifest.__code__)
        self.assertEqual(module.STATS,{})
        self.assertEqual(view.STATS[row['original']['realpath']]['identity'],row['original'])
        self.assertEqual(view.LIVE_STORAGE[row['original']['realpath']]['storage_identity'],row['keeper'])

    def test_unmapped_payload_retains_original_identity_requirement(self):
        path=self.root/'unchanged.pt';path.write_bytes(b'unchanged')
        original=t.identity(path);row=dict(identity=original,sha256=t.digest(b'unchanged'))
        self.payloads.plan['analysis_payloads'][str(path)]=row
        proof=self.payloads.check_capture(path,row);self.assertTrue(proof['original_identity_is_current_stat'])
        replacement=self.root/'replacement.pt';replacement.write_bytes(b'unchanged');replacement.replace(path)
        with self.assertRaises(ValueError):self.payloads.check_capture(path,row)

    def test_earlier_producer_evidence_outside11859_keeps_original_live_stat(self):
        module=types.ModuleType('earlier_evidence');module.checks=types.SimpleNamespace(check_hash=lambda d:None)
        module.p=types.SimpleNamespace(identity=t.identity)
        path=self.root/'earlier-control.zlib';path.write_bytes(b'earlier');ident=t.identity(path)
        view=legacy.evidence_view(module,self.payloads)
        captured=view.payload_stat(path,ident,t.digest(b'earlier'))
        self.assertEqual(captured,ident);self.assertEqual(view.LIVE_STORAGE,{})
        self.assertNotIn(str(path),self.payloads.plan['analysis_payloads'])
        path.write_bytes(b'changed size')
        with self.assertRaises(ValueError):view.payload_stat(path,ident,t.digest(b'earlier'))

    def test_terminal_adapter_passes_verified_plan_to_original_queue(self):
        expected=self.payloads.plan;received=[]
        queue=types.SimpleNamespace(check_receipt=lambda plan:received.append(plan)or{'passed':True})
        with patch.object(legacy,'verify_protocol_metadata',return_value=expected):
            result=legacy.check_saved_terminal(object(),queue,self.payloads)
        self.assertEqual(received,[expected]);self.assertTrue(result['original_plan_unchanged'])


if __name__=='__main__':unittest.main(verbosity=2)
