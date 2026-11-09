"""Generated input binding fixtures; no original model/tensor files opened."""
import copy
import json
import os
import stat
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_inputs_v1 as inputs
from minicpm5_saved_tap_analysis_v1 import ARMS
from check_minicpm5_saved_tap_report_v1_r2 import trajectory


def descriptor(path, fingerprint, size, inode):
    directory = dict(st_dev=1, st_ino=1, st_mode=stat.S_IFDIR | 0o700, st_uid=os.getuid(), st_gid=0)
    return dict(path=path, sha256=fingerprint, bytes=size,
        identity=dict(st_dev=1, st_ino=inode, st_mode=stat.S_IFREG | 0o600, st_nlink=1,
                      st_uid=os.getuid(), st_gid=0, st_size=size,
                      st_mtime_ns=1, st_ctime_ns=1, st_blocks=(size+511)//512),
        directories=[dict(directory) for _ in Path(path).parts[1:]])


def fixture():
    docs, raw = trajectory()
    def read(ref):
        expected, value = docs[ref['path']]
        if ref != expected: raise ValueError('Generated fixture reference differs')
        return copy.deepcopy(value)
    diagnosis = metadata.analyze(read)
    diagnosis['metadata_source_sha256'] = {p:r['sha256'] for p,(r,_) in docs.items()}
    diagnosis['metadata_bytes_read'] = 27711882
    entries = [descriptor(p,r['sha256'], 290000 if i < 94 else 451882, i+1)
               for i,(p,(r,_)) in enumerate(docs.items())]
    bundles = [descriptor(x['bundle']['path'], x['bundle']['sha256'], x['bundle']['raw_bytes'], 96+i)
               for i,x in enumerate(diagnosis['bundle_inventory'])]
    manifest = dict(schema='minicpm5_saved_tap_input_manifest_v1',reviewer='/root',
        diagnosis=inputs.DIAGNOSIS_REF, uid=os.getuid(), metadata=entries, bundles=bundles,
        raw_payloads_opened_during_snapshot=False, runtime_admitted=False,
        full_model_qualified=False, new_speedup_claim=False)
    return diagnosis, manifest, docs, raw


class CompleteBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()

    def setUp(self):
        self.diagnosis, self.manifest, self.docs, self.raw = copy.deepcopy(self.original)

    def validate(self):
        return inputs.validate_manifest(self.diagnosis, self.manifest, os.getuid())

    def test_complete_original_scope_accepted(self):
        meta, raw = self.validate()
        self.assertEqual((len(meta),len(raw)),(95,84))
        self.assertEqual((sum(x['bytes'] for x in meta),sum(x['bytes'] for x in raw)),(27711882,3145728))

    def test_missing_duplicate_or_extra_descriptor_rejected(self):
        for kind in ('metadata','bundles'):
            for action in ('missing','duplicate','extra'):
                with self.subTest(kind=kind,action=action):
                    self.manifest=copy.deepcopy(self.original[1])
                    rows=self.manifest[kind]
                    if action=='missing':rows.pop()
                    elif action=='duplicate':rows[-1]=copy.deepcopy(rows[0])
                    else:rows.append(copy.deepcopy(rows[0]))
                    with self.assertRaises(ValueError):self.validate()

    def test_wrong_bundle_path_hash_or_size_rejected(self):
        for field,value in (('path','/tmp/unadmitted.bf16'),('sha256','0'*64),('bytes',1)):
            self.manifest=copy.deepcopy(self.original[1])
            self.manifest['bundles'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate()

    def test_wrong_rank_arm_call_or_encoding_rejected(self):
        for field,value in (('rank',True),('rank',2),('arm','unrecognized'),('call',14)):
            self.diagnosis=copy.deepcopy(self.original[0])
            self.diagnosis['bundle_inventory'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.validate()
        self.diagnosis=copy.deepcopy(self.original[0])
        self.diagnosis['bundle_inventory'][0]['bundle']['encoding']='FP32'
        with self.assertRaises(ValueError):self.validate()

    def test_duplicate_logical_key_and_physical_inode_rejected(self):
        self.diagnosis['bundle_inventory'][-1]=copy.deepcopy(self.diagnosis['bundle_inventory'][0])
        with self.assertRaises(ValueError):self.validate()
        self.diagnosis=copy.deepcopy(self.original[0])
        self.manifest['bundles'][1]['identity']['st_ino']=self.manifest['bundles'][0]['identity']['st_ino']
        with self.assertRaisesRegex(ValueError,'physical identities'):self.validate()

    def test_metadata_pin_or_total_change_rejected(self):
        self.manifest['metadata'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'metadata pins'):self.validate()
        self.manifest=copy.deepcopy(self.original[1])
        row=self.manifest['metadata'][0]
        row['bytes']+=1;row['identity']['st_size']+=1
        with self.assertRaisesRegex(ValueError,'metadata pins'):self.validate()

    def test_preparation_cannot_claim_runtime_or_model_success(self):
        for key in ('raw_payloads_opened_during_snapshot','runtime_admitted','full_model_qualified','new_speedup_claim'):
            self.manifest=copy.deepcopy(self.original[1]);self.manifest[key]=True
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Preparation'):self.validate()

    def test_diagnosis_binding_and_owner_cannot_change(self):
        self.manifest['diagnosis']=dict(inputs.DIAGNOSIS_REF,sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'root-reviewed'):self.validate()
        self.manifest=copy.deepcopy(self.original[1]);self.manifest['uid']+=1
        with self.assertRaisesRegex(ValueError,'root-reviewed'):self.validate()

    def test_strict_json_rejects_duplicates_and_nonfinite_constants(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":{"b":0,"b":1}}',b'{"x":NaN}',b'[Infinity]',b'[-Infinity]',b'{"x":1e9999}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):inputs.strict_json(raw)
        self.assertEqual(inputs.strict_json(b'{"x":[0,1.25,true,null]}'),{'x':[0,1.25,True,None]})

    def test_invalid_manifest_stops_before_any_reader_is_constructed(self):
        self.manifest['bundles'].pop()
        with patch.object(inputs,'PinnedReader') as factory,self.assertRaises(ValueError):
            inputs.analyze(self.diagnosis,self.manifest,lambda:None,uid=os.getuid())
        factory.assert_not_called()

    def test_full_report_wiring_consumes_both_readers_and_finishes_once(self):
        readers=[]
        outer=self
        class MemoryReader:
            def __init__(self,entries,guard,**limits):
                self.entries={e['path']:e for e in entries}
                self.seen=set();self.finished=0
                self.is_raw=limits['file_cap']==65536
                readers.append(self)
            def __call__(self,ref,size=None):
                row=self.entries[ref['path']]
                outer.assertEqual(ref,dict(path=row['path'],sha256=row['sha256']))
                outer.assertNotIn(row['path'],self.seen)
                self.seen.add(row['path'])
                if self.is_raw:
                    outer.assertEqual(len(readers[0].seen),95)
                    outer.assertEqual(size,row['bytes'])
                    return outer.raw[row['path']]
                outer.assertIsNone(size)
                return json.dumps(outer.docs[row['path']][1]).encode()
            def finish(self):
                outer.assertEqual(self.seen,set(self.entries))
                self.finished+=1
                return dict(files=len(self.seen),bytes_read=sum(e['bytes'] for e in self.entries.values()),
                            all_pinned_inputs_read_once=True,all_input_hashes_verified=True)
        with patch.object(inputs,'PinnedReader',MemoryReader):
            result=inputs.analyze(self.diagnosis,self.manifest,lambda:None,uid=os.getuid())
        self.assertEqual(result['pairwise_vector_comparisons'],384)
        self.assertEqual([x.finished for x in readers],[1,1])
        self.assertEqual(result['input_io']['metadata']['files'],95)
        self.assertEqual(result['input_io']['raw']['bytes_read'],3145728)
        self.assertFalse(result['runtime_admission_proven_by_this_module'])
        self.assertFalse(result['full_model_qualified'])
        self.assertFalse(result['new_speedup_claim'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
