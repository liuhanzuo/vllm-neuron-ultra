"""Stdlib fixtures only: tests capture-file review, never actual native storage."""
import copy,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import dory_weight_storage_capture_v1 as c


def write(path,value):
    raw=(json.dumps(value,indent=2)+'\n').encode();path.write_bytes(raw)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def fixture(root):
    (root/'raw').mkdir();rows=[];records={}
    for kind in ('plain','tracked'):
        for shape in c.probe.SHAPES:
            raw=c.probe.fixture_bytes(shape);changed=b'\x80\x3f'+raw[2:];label=kind+'_'+str(len(raw)//2)
            row=dict(shape=list(shape),buffer_type=kind,bytes=len(raw),allocator_pressure_bytes=4*len(raw),tracked_buffer_released=True if kind=='tracked'else None,original_weights_loaded=False,full_model_memory_fit_proven=False)
            for flag in ('frombuffer_shares_exact_buffer','detach_shares_exact_buffer','bidirectional_alias_observed','detached_owner_survives_original_deletion','copy_negative_control_distinct','raw_word_probe_only'):row[flag]=True
            for stage,data in [('initial',raw),('mutation',changed),('detached',raw)]:
                row[stage+'_sha256']=hashlib.sha256(data).hexdigest();name=label+'_'+stage;path=root/'raw'/(name+'.raw');path.write_bytes(data)
                records[name]=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),shape=list(shape),dtype='bfloat16')
            rows.append(row)
    native=dict(schema='dory_weight_storage_native_probe_v1',complete=True,passed=True,plan=c.probe.plan(),cases=rows,torch_version='synthetic-source-test-only',plain_buffer_uses_production_frombuffer_detach_operations=True,tracked_buffer_only_supplements_lifetime_observation=True,all_raw_words_exact=True,source_sha256={str(Path(c.probe.__file__).resolve()):c.PINS[Path(c.probe.__file__).resolve()],str(Path(c.probe.bridge.__file__).resolve()):c.probe.BRIDGE_SHA,str(Path(c.probe.bridge.loader.__file__).resolve()):c.probe.LOADER_SHA},**{name:False for name in c.FLAGS})
    manifest=dict(schema='dory_weight_storage_saved_boundaries_v1',rows=records,tensors=18,raw_bytes=786864,all_files_rehashed=True)
    result=dict(schema='dory_weight_storage_capture_result_v1',complete=True,passed=True,probe=write(root/'probe.json',native),manifest=write(root/'raw-manifest.json',manifest),capture_source_sha256={str(c.SELF):hashlib.sha256(c.SELF.read_bytes()).hexdigest(),**{str(p):sha for p,sha in c.PINS.items()}},**{name:False for name in c.FLAGS})
    return native,manifest,result,write(root/'result.json',result)


class Files(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.native,self.manifest,self.result,self.ref=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def review(self):return c.review(self.ref,lambda:True)
    def sync(self):
        self.result['probe']=write(self.root/'probe.json',self.native);self.result['manifest']=write(self.root/'raw-manifest.json',self.manifest);self.ref=write(self.root/'result.json',self.result)
    def test_all18_full_saved_payloads_reproduced(self):
        result=self.review();self.assertEqual((result['saved_tensors'],result['saved_raw_bytes']),(18,786864));self.assertTrue(result['all_saved_words_exact']);self.assertFalse(result['native_lifetime_independently_reexecuted']);self.assertFalse(result['actual_process_exit_verified']);self.assertFalse(result['full_model_qualified'])
    def test_raw_corruption_rejected(self):
        path=Path(self.manifest['rows']['plain_4_initial']['path']);path.write_bytes(b'badbytes')
        with self.assertRaises(ValueError):self.review()
    def test_corruption_rehashed_metadata_still_rejected(self):
        record=self.manifest['rows']['plain_4_initial'];data=b'badbytes';Path(record['path']).write_bytes(data);record['sha256']=hashlib.sha256(data).hexdigest();self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_missing_raw_rejected(self):
        Path(self.manifest['rows']['plain_4_initial']['path']).unlink()
        with self.assertRaises(ValueError):self.review()
    def test_extra_raw_rejected(self):
        (self.root/'raw/extra.raw').write_bytes(b'x')
        with self.assertRaises(ValueError):self.review()
    def test_extra_metadata_rejected(self):
        (self.root/'unexpected.json').write_text('{}')
        with self.assertRaises(ValueError):self.review()
    def test_same_bytes_do_not_allow_label_path_alias(self):
        self.manifest['rows']['plain_4_initial']['path']=self.manifest['rows']['plain_4_detached']['path'];self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_raw_symlink_rejected(self):
        path=Path(self.manifest['rows']['plain_4_initial']['path']);path.unlink();path.symlink_to('plain_4_detached.raw')
        with self.assertRaises(ValueError):self.review()
    def test_raw_hardlink_rejected(self):
        path=Path(self.manifest['rows']['plain_4_initial']['path']);path.unlink();os.link(self.root/'raw/plain_4_detached.raw',path)
        with self.assertRaises(ValueError):self.review()
    def test_truncated_metadata_rejected(self):
        path=self.root/'result.json';path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):self.review()
    def test_wrong_optional_reference_size_rejected(self):
        self.ref['bytes']+=1
        with self.assertRaises(ValueError):self.review()
    def test_wrong_shape_rejected(self):
        self.manifest['rows']['plain_32_initial']['shape']=[32];self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_wrong_dtype_rejected(self):
        self.manifest['rows']['plain_4_initial']['dtype']='float16';self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_missing_case_rejected(self):
        self.native['cases'].pop();self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_duplicate_case_rejected(self):
        self.native['cases'][-1]=copy.deepcopy(self.native['cases'][0]);self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_lifetime_failure_not_promoted(self):
        self.native['cases'][-1]['tracked_buffer_released']=False;self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_false_plain_lifetime_claim_rejected(self):
        self.native['cases'][0]['tracked_buffer_released']=True;self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_modified_source_hash_rejected(self):
        self.native['source_sha256'][str(Path(c.probe.__file__).resolve())]='0'*64;self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_unjustified_model_qualification_rejected(self):
        self.result['native_reference_qualified']=True;self.sync()
        with self.assertRaises(ValueError):self.review()
    def test_expired_guard_precedes_file_read(self):
        with self.assertRaisesRegex(RuntimeError,'expired'):c.review(self.ref,lambda:(_ for _ in ()).throw(RuntimeError('expired')))
    def test_expired_runtime_guard_precedes_any_output(self):
        target=self.root/'not-created'
        with self.assertRaisesRegex(RuntimeError,'expired'):c.run(torch=object(),guard=lambda:(_ for _ in ()).throw(RuntimeError('expired')),output=target)
        self.assertFalse(target.exists())
    def test_reader_budget_rejects_oversize_file(self):
        with self.assertRaises(ValueError):c.stable_bytes(self.root/'result.json',1,lambda:True)
    def test_guard_rechecked_after_complete_read(self):
        calls=[]
        def guard():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('expired during read')
        with self.assertRaisesRegex(RuntimeError,'expired during read'):c.stable_bytes(self.root/'result.json',2**20,guard)
    def test_original_pins_and_no_framework_import(self):
        c.source_check();self.assertFalse({'torch','numpy','torch_neuronx','transformers'}&set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
