"""Independent bounded SDK fixtures and preserved observer/source checks."""
import ast,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research');sys.path[:0]=[str(Q),str(Q.parent)]
from qwen35_package_serial_runtime_v9 import compiler_assets as ca,contract as c
import render_qwen35_package_r7_root_helpers_source_v1 as renderer

class IndependentReview(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='qwen35-independent-mini-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.dve=self.root/'dve';self.dve.mkdir();self.pwp=self.root/'pwp';self.pwp.mkdir()
        for attr,value in (('DVE_ROOT',self.dve),('ASSET_ROOT',self.pwp)):
            ctx=patch.object(ca,attr,value);ctx.start();self.addCleanup(ctx.stop)
    def make_asset(self):
        path=self.dve/'transformer_control_table.bin';data=b'a'*43998+b'z';path.write_bytes(data)
        row=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),identity=dict(zip(ca.ATTRIBUTES,ca.attrs(path.lstat()))))
        return path,data,row
    def test_twenty_basenames_have_exact_disjoint_targets(self):
        self.assertEqual(len(ca.NAMES),len(set(ca.NAMES)));self.assertEqual(len(ca.PWP_NAMES),16);self.assertEqual(len(ca.DVE_NAMES),4)
        for n in ca.PWP_NAMES:self.assertEqual(ca.target_path(n),self.pwp/n)
        for n in ca.DVE_NAMES:self.assertEqual(ca.target_path(n),self.dve/n)
        for n in ('transformer_control_table.bin.bak','../dve_info.json','other.bin'):
            with self.assertRaises(ValueError):ca.target_path(n)
    def test_full_last_byte_contributes_to_hash(self):
        path,data,row=self.make_asset();result=ca.asset(path,row);self.assertEqual(result['logical_bytes'],43999)
        row['sha256']=hashlib.sha256(data[:-1]+b'x').hexdigest()
        with self.assertRaisesRegex(ValueError,'content/hash'):ca.asset(path,row)
    def test_every_directory_and_leaf_open_is_nofollow(self):
        path,data,row=self.make_asset();original=os.open;seen=[]
        def observed(path,flags,*args,**kwargs):seen.append((str(path),flags));return original(path,flags,*args,**kwargs)
        with patch.object(ca.os,'open',side_effect=observed):ca.asset(path,row)
        self.assertGreater(len(seen),3)
        for _,flags in seen:self.assertTrue(flags&os.O_NOFOLLOW);self.assertTrue(flags&os.O_CLOEXEC)
        self.assertEqual(seen[-1][0],path.name)
        for _,flags in seen[:-1]:self.assertTrue(flags&os.O_DIRECTORY)
    def test_ancestor_replacement_during_read_rejected(self):
        path,data,row=self.make_asset();original=os.read;changed=[]
        def replacing(fd,count):
            value=original(fd,count)
            if not changed:
                moved=self.dve.with_name('original-dve');self.dve.rename(moved);self.dve.symlink_to(moved);changed.append(True)
            return value
        with patch.object(ca.os,'read',side_effect=replacing):
            with self.assertRaises(ValueError):ca.asset(path,row)
    def test_file_growth_during_full_read_rejected(self):
        path,data,row=self.make_asset();original=os.read;changed=[]
        def growing(fd,count):
            value=original(fd,count)
            if not changed:
                with path.open('ab')as stream:stream.write(b'x')
                changed.append(True)
            return value
        with patch.object(ca.os,'read',side_effect=growing):
            with self.assertRaises(ValueError):ca.asset(path,row)
    def test_retained_failed_observer_skips_stage25_and_cannot_claim_history_reexecution(self):
        old=Path('/home/ubuntu/qwen35_package_full_r6_post_exit_20261112_v1.py').read_text()
        tree=ast.parse(old);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit_outcome')
        def require(value,message):
            if not value:raise ValueError(message)
        ns=dict(require=require);exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated exact observer outcome>','exec'),ns)
        audit=Mock(side_effect=AssertionError('Failed outer must not claim scientific audit'))
        result=ns['audit_outcome'](dict(exit_code=1),dict(failed=True,result_present=False),{}, {}, {},Mock(),audit)
        self.assertFalse(result['all25_stages_audited']);self.assertIsNone(result['stage25']);audit.assert_not_called()
        self.assertFalse(result['full_model_qualified'])
    def test_runtime_and_audit_lifecycle_science_functions_preserved_byte_for_byte(self):
        old=Q/'qwen35_package_serial_runtime_v8';new=Q/'qwen35_package_serial_runtime_v9'
        for name in ('adapter.py','contract.py','service.py','history.py','wiring.py','readiness.py','final_audit.py'):
            self.assertEqual((old/name).read_bytes(),(new/name).read_bytes(),name)
        for name in ('qwen35_package_post_audit','qwen35_package_audit_physical'):
            before=(Q/(name+'_v1.py')).read_text().replace('qwen35_package_serial_runtime_v8','qwen35_package_serial_runtime_v9')
            before=before.replace('qwen35_package_audit_physical_v1','qwen35_package_audit_physical_v2')
            self.assertEqual(before,(Q/(name+'_v2.py')).read_text(),name)

if __name__=='__main__':unittest.main(verbosity=2)
