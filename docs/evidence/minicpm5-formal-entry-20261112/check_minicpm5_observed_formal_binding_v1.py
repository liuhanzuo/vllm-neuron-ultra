"""Complete candidate assembly with saved JSON and synthetic original plan only."""
import sys


class NoNativeOrHistorical:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'ctypes','_ctypes','torch','numpy','transformers','vllm','neuronxcc',
                'torch_neuronx','neuronx_distributed','neuronx_distributed_inference'} or name in {
                'research.prepare_minicpm5_pv_sdk_import_diagnostic_v1',
                'research.minicpm5_pv_sdk_import_diagnostic_v1'}:
            raise AssertionError('No historical/native import in candidate tests: '+name)


sys.meta_path.insert(0,NoNativeOrHistorical())
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
from research import check_minicpm5_observed_formal_integration_v1 as saved
from research import prepare_minicpm5_observed_formal_binding_v1 as binding
from research import minicpm5_sdk_mapping_outer_source_entry_v5 as outer


def fixture():
    reader,pins,manifest_ref,ended,_=saved.fixture()
    manifest=reader.overrides[manifest_ref['path']]
    original=Path(binding.ORIGINAL)
    manifest['retained_source_sha256'][str(original)]=hashlib.sha256(original.read_bytes()).hexdigest()
    manifest['preparation_source_sha256']={binding.SELF:hashlib.sha256(Path(binding.SELF).read_bytes()).hexdigest()}
    pins.update(manifest['retained_source_sha256']);pins.update(manifest['preparation_source_sha256'])
    def add(name,value):
        ref=dict(path='/synthetic/'+name+'.json',sha256=hashlib.sha256(binding.launch.encoded(value)).hexdigest())
        reader.overrides[ref['path']]=value
        return ref
    predecessor=add('predecessor',dict(schema='synthetic_complete_terminal'))
    catalog=add('catalog',dict(runs=[dict(terminal=predecessor,source_sha256=copy.deepcopy(pins))]))
    old=dict(schema='minicpm5_sdk_import_future_binding_request_v1',
        current_serial_catalog=catalog,independent_audit_exit=add('audit',{}),
        recovery_binding={'synthetic':'original_saved_recovery'},diagnostic_source_review=add('review',{}),
        additional_source_refs=[],output_root=str(Q/'minicpm5-fixed-input-pv-sdk-import-source-fixture'))
    request=dict(schema=binding.REQUEST_SCHEMA,original_request=add('request',old),
        current_handoff=add('handoff',dict(complete_current_serial_catalog=catalog)),
        handoff_catalog_field='complete_current_serial_catalog',expected_predecessor_exit=predecessor,
        source_manifest=manifest_ref,artifact_prefix=str(Q/'minicpm5-formal-import-source-fixture'),
        attempt_id='a'*32,unit='trn1-mini-formal-source-fixture')
    # The original source contract is deliberately synthetic. No test claims
    # actual reconstruction; production prepare calls the unchanged original.
    full=dict(source_contract={'synthetic_original_contract':['retained','unaltered']},source_sha256=pins)
    pv=dict(schema='minicpm5_fixed_input_pv_serial_spec_v1',independent_audit_exit=old['independent_audit_exit'],
        current_serial_catalog=catalog,source_extension=copy.deepcopy(pins),output_root=old['output_root'],
        worker=str(Q/'minicpm5_pv_sdk_import_diagnostic_worker_v1.py'),
        limits=copy.deepcopy(binding.logic.LIMITS),artifact_limits=copy.deepcopy(binding.logic.CAPS))
    candidate=dict(schema='minicpm5_sdk_import_future_binding_candidate_v1',source_metadata_only=True,
        original_runtime_reconstruction_routes_agree=True,all_original_catalog_rows_retained=True,
        all_current_catalog_sources_retained=True,actual_freshness_or_physical_checks_performed=False,
        runtime_admitted=False,launch_authorized=False,pv_spec_candidate=pv,
        complete_source_map_sha256=binding.policies.history.digest(pins),complete_source_files=len(pins),
        complete_source_release_bytes=len(binding.launch.encoded(full)))
    def text(path,sha):
        assert Path(path).suffix=='.py' and Path(path).is_relative_to(ROOT)
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==sha
        return raw
    return reader,copy.deepcopy(request),copy.deepcopy(candidate),copy.deepcopy(full),text


class Checks(unittest.TestCase):
    def test_complete_artifact_graph_retains_contract_and_canonical_references(self):
        reader,request,candidate,full,text=fixture()
        result=binding.assemble(request,candidate,full,read=reader.json,text=text)
        artifacts=result['artifacts'];refs={row['reference']['path']:row for row in artifacts.values()}
        self.assertEqual(artifacts['complete-source-map']['value'],full['source_sha256'])
        self.assertEqual(artifacts['complete-source-contract']['value'],full['source_contract'])
        for name,row in artifacts.items():
            raw=binding.launch.encoded(row['value'])
            self.assertEqual(len(raw),row['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['reference']['sha256'])
        def walk(value):
            if type(value) is dict:
                if set(value)=={'path','sha256'} and value['path'] in refs:
                    self.assertEqual(value,refs[value['path']]['reference'])
                for part in value.values():walk(part)
            elif type(value) is list:
                for part in value:walk(part)
        walk(result)
        successor=json.loads(artifacts['outer-entry-plan']['value']['successor_input'])
        self.assertEqual(successor['review'],artifacts['outer-review-draft']['reference'])
        self.assertEqual(artifacts['sdk-entry-template']['value']['successor_input'],'admission:self')
        self.assertEqual(artifacts['diagnostic-spec']['value']['source_extension'],
                         artifacts['pv-spec']['value']['source_extension'])

    def test_no_draft_claims_root_admission_or_drops_original_pending_work(self):
        reader,request,candidate,full,text=fixture()
        result=binding.assemble(request,candidate,full,read=reader.json,text=text)
        for name in result['review_artifacts']:
            d=result['artifacts'][name]['value'];self.assertIsNone(d['reviewer'])
            self.assertFalse(d.get('passed',d.get('accepted')))
        for name in ('runtime_admitted','launch_authorized','accepted_native_authority_created',
                     'native_ABI_qualified','full_model_qualified','new_speedup_claim'):
            self.assertFalse(result[name])
        self.assertEqual(result['original_nine_native_checks_pending'],9)
        self.assertEqual(result['original_scientific_rows_still_pending'],1284)
        self.assertNotIn(saved.types_source.PIN['path'],full['source_sha256'])

    def test_changed_catalog_predecessor_or_original_map_rejected(self):
        for change in ('catalog','predecessor','map','contract','source'):
            reader,request,candidate,full,text=fixture()
            if change=='catalog':candidate['pv_spec_candidate']['current_serial_catalog']['sha256']='0'*64
            elif change=='predecessor':request['expected_predecessor_exit']['sha256']='0'*64
            elif change=='map':full['source_sha256'].pop(next(iter(full['source_sha256'])))
            elif change=='contract':full['source_contract']['extra']='changed'
            else:
                old_text=text
                text=lambda path,sha:b'changed' if path==binding.SELF else old_text(path,sha)
            with self.assertRaises(ValueError):binding.assemble(request,candidate,full,read=reader.json,text=text)

    def test_original_resource_limits_worker_and_packet_cap_are_not_bypassed(self):
        for change in ('wall','worker','release','unit'):
            reader,request,candidate,full,text=fixture()
            if change=='wall':candidate['pv_spec_candidate']['limits']['wall_seconds']+=1
            elif change=='worker':candidate['pv_spec_candidate']['worker']=str(Q/'foreign_worker.py')
            elif change=='release':candidate['complete_source_release_bytes']=16*2**20
            else:request['unit']='foreign-service'
            with self.assertRaises(ValueError):binding.assemble(request,candidate,full,read=reader.json,text=text)

    def test_root_injected_original_pure_functions_called_without_import_or_authenticate(self):
        reader,request,candidate,full,text=fixture();calls=[]
        def prepare(old,**kw):
            calls.append('prepare')
            required=reader.overrides[request['source_manifest']['path']]
            self.assertEqual(dict((r['path'],r['sha256']) for r in old['additional_source_refs']),
                {**required['retained_source_sha256'],**required['new_runtime_source_sha256'],
                 **required['preparation_source_sha256']})
            return copy.deepcopy(candidate)
        def reconstruct(*args):calls.append('reconstruct');return {'plan':copy.deepcopy(full)}
        def merge(plan,extension):calls.append('merge');self.assertEqual(plan,full);return plan
        original=NS(__file__=binding.ORIGINAL,prepare=prepare,reconstruct_audit_base=reconstruct,
                    audited=NS(merged_plan=merge))
        result=binding.prepare(request,read=reader.json,text=text,guard=lambda:calls.append('guard'),original=original)
        self.assertEqual(calls,['prepare','reconstruct','merge','guard','guard'])
        self.assertFalse(result['runtime_admitted'])
        self.assertNotIn('research.prepare_minicpm5_pv_sdk_import_diagnostic_v1',sys.modules)

    def test_only_complete_map_gets_original_16mib_input_ceiling(self):
        for change in ('none','race','rebound','cap'):
            calls=[];reference=dict(path='/synthetic/map.json',sha256='a'*64)
            identities=[('saved',),('changed',) if change=='race' else ('saved',)]
            original=NS(PIPE_CAP=8*2**20 if change=='cap' else 16*2**20,
                r=NS(identity=lambda path:identities.pop(0),sources=NS(reference=lambda ref:calls.append('ref'))))
            metadata=NS(identities={},guard=lambda:calls.append('guard'))
            if change=='rebound':metadata.identities[reference['path']]=('earlier',)
            def read(ref,cap):
                self.assertEqual(ref,reference);self.assertEqual(cap,16*2**20)
                calls.append('read');return {'full':'map'}
            reader=NS(json=read)
            if change=='none':
                self.assertEqual(outer._complete_source_map(reader,reference,original,metadata),{'full':'map'})
                self.assertEqual(metadata.identities,{reference['path']:('saved',)})
                self.assertEqual(calls,['guard','ref','read','guard'])
            else:
                with self.assertRaises(ValueError):outer._complete_source_map(reader,reference,original,metadata)
        source=(Q/'minicpm5_sdk_mapping_outer_source_entry_v5.py').read_text()
        self.assertIn('len(original.serial.encoded(source_subset))+65536<=original.PIPE_CAP',source)
        self.assertIn("reader.unchanged({spec['complete_source_map']['path']})",source)
        self.assertNotIn("metadata.read(spec['complete_source_map'])",source)


if __name__=='__main__':unittest.main()
