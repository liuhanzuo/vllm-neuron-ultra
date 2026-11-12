"""In-memory ELF and mocked fd tests; no native, process or real payload IO."""
import sys
class Blocker:
    names={'torch','numpy','ctypes','_ctypes','transformers','safetensors','torch_xla','torch_neuronx',
           'neuronx_distributed','neuronx_distributed_inference','vllm','vllm_neuron','packaging',
           'lfm3_nki05_serial_history_v1'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in self.names:raise AssertionError('Forbidden source-test import: '+fullname)
        return None
BLOCKER=Blocker();sys.meta_path.insert(0,BLOCKER)

import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import spark26_elf_preflight_core_v1 as core
import spark26_elf_preflight_worker_v1 as worker
import run_spark26_elf_preflight_outer_v1 as outer


def elf_fixture():
    data=bytearray(512);data[:16]=b'\x7fELF\x02\x01\x01'+bytes(9)
    struct.pack_into('<HHIQQQIHHHHHH',data,16,3,62,1,0,64,0,0,64,56,2,0,0,0)
    struct.pack_into('<IIQQQQQQ',data,64,1,4,0,0x1000,0x1000,512,512,4096)
    struct.pack_into('<IIQQQQQQ',data,120,2,4,192,0x10c0,0x10c0,96,96,8)
    strings=b'libdep.so\0libself.so\0$ORIGIN\0';data[320:320+len(strings)]=strings
    entries=[(5,0x1140),(10,len(strings)),(1,0),(14,10),(29,21),(0,0)]
    for index,(tag,value)in enumerate(entries):struct.pack_into('<qQ',data,192+16*index,tag,value)
    return data


def parse(data):return core.parse_elf(lambda offset,size:bytes(data[offset:offset+size]),len(data))


class ElfParsing(unittest.TestCase):
    def test_actual_needed_soname_and_runpath_from_synthetic_ELF(self):
        result=parse(elf_fixture());self.assertEqual(result['needed'],['libdep.so'])
        self.assertEqual(result['soname'],'libself.so');self.assertEqual(result['search_tags'],[dict(tag='runpath',value='$ORIGIN')])
        self.assertFalse(result['actual_loader_selection_proven']);self.assertFalse(result['dependency_closure_proven'])

    def test_wrong_machine_endianness_and_class_rejected(self):
        for offset,value in [(4,1),(5,2),(18,183)]:
            data=elf_fixture();data[offset]=value
            with self.subTest(offset=offset):
                with self.assertRaises(core.ObjectFailure):parse(data)

    def test_extended_or_excessive_program_header_count_rejected(self):
        data=elf_fixture();struct.pack_into('<H',data,56,65535)
        with self.assertRaises(core.ObjectFailure):parse(data)

    def test_out_of_file_segment_and_unbacked_string_table_rejected(self):
        data=elf_fixture();struct.pack_into('<Q',data,64+32,513)
        with self.assertRaises(core.ObjectFailure):parse(data)
        data=elf_fixture();struct.pack_into('<Q',data,192+8,0x100000)
        with self.assertRaises(core.ObjectFailure):parse(data)

    def test_missing_termination_and_duplicate_singleton_rejected(self):
        data=elf_fixture();struct.pack_into('<qQ',data,192+80,1,0)
        with self.assertRaises(core.ObjectFailure):parse(data)
        data=elf_fixture();struct.pack_into('<qQ',data,192+32,5,0x1140)
        with self.assertRaises(core.ObjectFailure):parse(data)

    def test_out_of_range_unterminated_and_invalid_utf8_strings_rejected(self):
        data=elf_fixture();struct.pack_into('<Q',data,192+32+8,1000)
        with self.assertRaises(core.ObjectFailure):parse(data)
        data=elf_fixture();data[320:349]=b'x'*29
        with self.assertRaises(core.ObjectFailure):parse(data)
        data=elf_fixture();data[320]=255
        with self.assertRaises(core.ObjectFailure):parse(data)

    def test_no_dynamic_segment_and_truncated_reads_rejected(self):
        data=elf_fixture();struct.pack_into('<I',data,120,1)
        with self.assertRaises(core.ObjectFailure):parse(data)
        with self.assertRaises(core.ObjectFailure):core.parse_elf(lambda o,n:b'',512)


class Authentication(unittest.TestCase):
    def fake_observe(self,data,expected_sha=None,identities=None):
        path='/fixture/lib.so';before=[1,2,len(data),3,4]
        expected=dict(bytes=len(data),sha256=expected_sha or hashlib.sha256(data).hexdigest())
        stream=io.BytesIO(data);context=SimpleNamespace(check=mock.Mock());budget=core.ReadBudget()
        stat=SimpleNamespace(st_dev=1,st_ino=2,st_size=len(data),st_mtime_ns=3,st_ctime_ns=4)
        with mock.patch.object(core.provider,'identity',side_effect=identities or [before,before]),\
             mock.patch.object(core.os,'open',return_value=123)as opened,\
             mock.patch.object(core.os,'fstat',return_value=stat),\
             mock.patch.object(core.os,'read',side_effect=lambda fd,n:stream.read(n)),\
             mock.patch.object(core.os,'pread',side_effect=lambda fd,n,o:bytes(data[o:o+n]))as pread,\
             mock.patch.object(core.os,'close')as closed:
            result=core.PayloadObserver(context,budget).observe(path,expected,True)
        closed.assert_called_once_with(123);opened.assert_called_once()
        self.assertGreaterEqual(context.check.call_count,3);self.assertGreater(pread.call_count,0)
        return result,budget

    def test_full_hash_and_ELF_use_same_mock_descriptor(self):
        result,budget=self.fake_observe(elf_fixture())
        self.assertTrue(result['selected_RECORD_match']);self.assertIsNone(result['ELF_error'])
        self.assertEqual(budget.hash_bytes,513);self.assertGreater(budget.elf_bytes,64)

    def test_mismatched_hash_remains_false_but_metadata_is_observed(self):
        result,budget=self.fake_observe(elf_fixture(),'f'*64)
        self.assertFalse(result['selected_RECORD_match']);self.assertIsNotNone(result['ELF'])

    def test_unsupported_ELF_remains_an_explicit_per_object_failure(self):
        data=elf_fixture();data[4]=1;result,budget=self.fake_observe(data)
        self.assertTrue(result['selected_RECORD_match']);self.assertIsNone(result['ELF'])
        self.assertIn('ELF64',result['ELF_error'])

    def test_same_path_replacement_rejected(self):
        with self.assertRaises(core.ObjectFailure):self.fake_observe(elf_fixture(),identities=[[1,2,512,3,4],[1,99,512,3,4]])

    def test_read_budget_charges_before_any_read(self):
        budget=core.ReadBudget();budget.reserve('hash',core.CAPS['total_payload_and_ELF_bytes'])
        with self.assertRaises(ValueError):budget.reserve('elf',1)
        budget=core.ReadBudget();budget.reserve('elf',core.CAPS['ELF_metadata_bytes'])
        with self.assertRaises(ValueError):budget.reserve('elf',1)
        self.assertEqual(budget.elf_bytes,core.CAPS['ELF_metadata_bytes'])

    def test_authority_dictionary_cannot_invoke_collector(self):
        with self.assertRaises(ValueError):core.collect({'metadata_preflight_admitted':True})

    def test_missing_file_becomes_object_failure(self):
        with mock.patch.object(core.provider,'identity',side_effect=FileNotFoundError('fixture')):
            with self.assertRaises(core.ObjectFailure):core.payload_identity('/fixture/missing.so')


class ScopeAndLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=worker.read(core.MANIFEST_REF);cls.preparation=worker.read(core.PREPARATION_REF)
        cls.sources=json.loads((core.CANDIDATE/'source-map.json').read_text())
        cls.payloads=core.provider.merge_manifest(cls.manifest,cls.sources,lambda ref:worker.metadata_bytes(ref).decode())
        cls.plan=core.plan_from_payloads(cls.manifest,cls.preparation,cls.payloads)

    def test_actual_plan_keeps_entire1271_candidates_and15conflicts(self):
        self.assertEqual(len(self.plan['ELF_paths']),1271);self.assertEqual(len(self.plan['conflicting_owners']),15)
        self.assertEqual(self.plan['payload_count'],41173);self.assertEqual(self.plan['payload_bytes'],13247146338)
        self.assertFalse(self.plan['source_candidate_runtime_admitted'])
        bad=copy.deepcopy(self.preparation);bad['issues']=[r for r in bad['issues']if not r.get('path','').endswith('/cv2/cv2.abi3.so')]
        with self.assertRaises(ValueError):core.plan_from_payloads(self.manifest,bad,self.payloads)

    def test_budget_and_scientific_sources_unchanged(self):
        import spark26_native_serial_worker_v2 as original
        self.assertEqual(worker.PROPS,original.PROPS);self.assertEqual(worker.BUDGET,original.BUDGET)
        self.assertEqual(worker.contract.COUNTS if hasattr(worker.contract,'COUNTS')else original.COUNTS,
                         dict(qkv_projection=216,cache_step=2136,cache_host_reorder=2108,cache_physical_order=384))
        self.assertEqual(core.CAPS['total_payload_and_ELF_bytes'],original.BUDGET['max_runtime_authentication_read_bytes'])

    def test_no_native_history_imports_and_no_weight_dependent_worker_path(self):
        self.assertFalse(BLOCKER.names&set(sys.modules))
        source=Path(worker.__file__).read_text();tree=ast.parse(source)
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
        self.assertNotIn('diagnostic',ast.unparse(execute));self.assertNotIn('native_providers',source)
        self.assertNotIn("spec['checkpoint_root']",Path(outer.__file__).read_text())
        self.assertNotIn('def authority(',source)

    def test_descriptor_gate_precedes_historical_import(self):
        source=Path(worker.__file__).read_text()
        self.assertLess(source.index('self.descriptor_check()'),source.index('import lfm3_nki05_serial_history_v1'))
        self.assertIn("metadata_preflight_admitted=True,runtime_admitted=False",Path(outer.__file__).read_text())

    def test_metadata_result_never_admits_runtime_and_preserves_failures(self):
        plan=self.plan;spec=dict(preflight_plan=plan);ref=dict(path='/fixture/admission.json',sha256='a'*64)
        result=dict(schema='spark26_elf_preflight_result_v1',metadata_collection_complete=True,
            plan_sha256=worker.contract.digest(plan),counts=worker.COUNTS,
            payload_count=plan['payload_count'],payload_bytes=plan['payload_bytes'],payload_sha256=plan['payload_sha256'],
            conflicting_owner_records_preserved=15,authentication_failures=1,ELF_metadata_failures=2,
            all_manifest_payloads_authenticated=False,all_candidate_ELF_metadata_collected=False,
            actual_hash_read_requests_bytes=plan['payload_bytes'],actual_ELF_read_requests_bytes=1000)
        result.update({key:False for key in ('conflicts_resolved','dependency_closure_proven',
            'source_candidate_runtime_admitted','native_imports_executed','model_checkpoint_reads','package_mutations',
            'full_model_qualified','new_speedup_claim','worker_exit_or_outer_drain_proven')})
        events=dict(counts=worker.COUNTS,bytes=100,complete=True)
        envelope=dict(admission=ref,worker_exit_or_outer_drain_proven=False,result=result,event_receipt=events)
        outer.validate_worker_result(envelope,spec,ref,events)
        for field in ('source_candidate_runtime_admitted','all_manifest_payloads_authenticated','dependency_closure_proven'):
            result[field]=True
            with self.subTest(field=field):
                with self.assertRaises(ValueError):outer.validate_worker_result(envelope,spec,ref,events)
            result[field]=False
        result['actual_ELF_read_requests_bytes']=worker.core.CAPS['ELF_metadata_bytes']+1
        with self.assertRaises(ValueError):outer.validate_worker_result(envelope,spec,ref,events)

    def test_all_metadata_events_roundtrip_and_tamper_rejected(self):
        ordered=sorted(self.payloads);indexes={p:i for i,p in enumerate(ordered)}
        with tempfile.TemporaryDirectory(prefix='spark-ELF-text-events-')as folder:
            sink=worker.EventSink(folder)
            for index,path in enumerate(ordered):
                expected=self.payloads[path]
                sink.emit('payload_authentication',dict(index=index,status='observed',identity=[1,2,expected['bytes'],3,4],
                    actual_sha256=expected['sha256'],selected_RECORD_match=True))
            metadata=parse(elf_fixture())
            for path in self.plan['ELF_paths']:
                sink.emit('ELF_metadata',dict(index=indexes[path],status='observed',selected_RECORD_match=True,ELF=metadata))
            for row in sorted(self.plan['conflicting_owners'],key=lambda r:r['path']):
                sink.emit('conflicting_owner',dict(index=indexes[row['path']],selected=row['selected_record'],
                    ambient=row['ambient_record'],unselected_distribution=row['unselected_distribution'],resolved_ownership=False))
            receipt=sink.close(True)
            spec=dict(native_manifest=core.MANIFEST_REF,source_sha256=self.sources,preflight_plan=self.plan)
            result=dict(authentication_failures=0,ELF_metadata_failures=0)
            references=outer.verify_event_files(folder,receipt,lambda:None,spec,result)
            self.assertEqual(len(references),13);self.assertLess(receipt['bytes'],worker.BUDGET['event_total_bytes'])
            path=Path(folder)/'payload_authentication-00.jsonl';lines=path.read_text().splitlines(keepends=True)
            value=json.loads(lines[0]);value['row']['index']=1;lines[0]=worker.encode(value).decode();path.write_text(''.join(lines))
            receipt['files'][path.name]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):outer.verify_event_files(folder,receipt,lambda:None,spec,result)


if __name__=='__main__':unittest.main(verbosity=2)
