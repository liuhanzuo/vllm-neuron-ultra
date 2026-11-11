"""Small synthetic entry controls; no real metadata/raw/process/lock operations."""
import copy
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import run_minicpm5_native_sdk_saved_comparison_v1 as e


def ref(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)


def terminal_fixture():
    spec=dict(schema='minicpm5_native_post_exit_audit_spec_v1',model_exit=ref('model'),recorder_exit=ref('recorder'),
        current_serial_catalog=ref('catalog'),source_extension={e.a.SELF:'a'*64,str(Path(e.a.core.__file__).resolve()):'b'*64},
        output=str(e.Q/'minicpm5-layer1-operator-native-synthetic-post-exit-audit-v1.json'),limits=e.LIMITS)
    review=dict(schema='minicpm5_native_post_exit_audit_source_review_v1',reviewer='/root',passed=True,blockers=[],
        spec=ref('audit-spec'),source_extension=spec['source_extension'])
    argv=['python3','-B',e.a.SELF,'--spec','/synthetic/audit-spec.json','--spec-sha256','a'*64,
        '--review','/synthetic/audit-review.json','--review-sha256','a'*64]
    owner=dict(pid=12,start_ticks='13',argv=argv,ppid=1,cgroup='0::/synthetic')
    terminal=dict(schema='minicpm5_native_independent_audit_actual_exit_v1',reviewer='/root',audit_spec=ref('audit-spec'),
        audit_review=ref('audit-review'),result=dict(path=spec['output'],sha256='c'*64),owner=owner,
        actual_tool=dict(kind='tool_session',session_id=123,terminal_chunk='abcdef',exit_code=0),
        actual_tool_exit_observed=True,exact_owner_gone=True,lock_identity=e.r.ORIGINAL_LOCK,self_exit_claimed=False,
        full_model_qualified=False,new_speedup_claim=False)
    result=dict(schema='minicpm5_native_post_exit_independent_audit_v1',complete=True,reviewer='/root',
        audit_spec=terminal['audit_spec'],audit_review=terminal['audit_review'],model_exit=spec['model_exit'],recorder_exit=spec['recorder_exit'],
        auditor_owner=owner,lock_identity=e.r.ORIGINAL_LOCK,source=dict(path=e.a.SELF,sha256='a'*64),
        self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,actual_auditor_tool_exit_verified=False,
        actual_native_model_exit_verified=True,actual_native_recorder_exit_verified=True,
        every_raw_byte_reread_after_all_model_processes_exited=True,original_sdk_model_exit_code=1,
        result=dict(schema='minicpm5_layer1_operator_native_independent_verification_v2',complete=True,frames=24,
            direct_attention_calls=72,raw_files=936,journals=72,arm_result_files=72,full_model_qualified=False,new_speedup_claim=False))
    return terminal,result,spec,review


def anchor_fixture():
    """Independent all-metadata synthetic trajectory, with nontrivial slot order."""
    b=e.boundary;calls=[{}for _ in range(14)];graphs=[{}for _ in range(14)];frames=[];fixtures=[]
    for slot in range(4):fixtures.append(dict(id=slot,input_ids=[1]*(20+slot),continuation_ids=[70+slot,80+slot,90+slot]))
    for ci in b.DECODE_CALLS:
        fs=[dict(index=len(frames)+j,slot=slot,fixture=slot,step=(ci-4)%7+1)for j,slot in enumerate((3,1,0,2))]
        frames+=fs;calls[ci]=dict(index=ci,phase='decode',frames=fs);graphs[ci]=dict(bucket=2048 if ci<10 else 128)
    refs=[dict(ref('original-%d'%i),rank=i//42,arm=e.sdk_reader.inputs.stream.raw.ARMS[(i%42)//14],call=i%14)for i in range(84)]
    plan=dict(calls=calls,graph_calls=graphs,frames=frames,fixtures=fixtures,layer1_observation=dict(original_calls=refs))
    mapping={};results=[]
    for rank in (0,1):
        rows=[];expected=[dict(rank=rank,arm=arm,call=ci)for arm in e.sdk_reader.inputs.stream.raw.ARMS for ci in b.DECODE_CALLS]
        original=dict(schema='minicpm5_layer1_rank_decode_stream_v1',complete=True,rank=rank,all_history_calls=42,decode_calls=18,
            physical_decode_rows=72,full_cache_records=84,restored_full_cache_bytes=84*e.sdk_reader.inputs.stream.raw.CACHE_BYTES,
            read_payload_bytes=100,all_read_bytes_match_saved_audit=True,prefill_replayed=False,attention_reexecuted=False,
            full_model_qualified=False,new_speedup_claim=False)
        stream=dict(schema='minicpm5_layer1_sdk_batch_layout_anchor_stream_v1',complete=True,rank=rank,original_stream=original,
            selected_anchor_calls=expected[:6],verified_not_reexecuted_calls=expected[6:],selected_decode_calls=6,skipped_decode_calls=12,
            all_other_arms_verified_not_reexecuted=True,prefill_reexecuted=False,full_model_qualified=False,new_speedup_claim=False)
        for ci in b.DECODE_CALLS:
            call=calls[ci];bucket=graphs[ci]['bucket'];step=call['frames'][0]['step'];positions=[20+slot+step-1 for slot in range(4)]
            tokens=[fixtures[slot]['continuation_ids'][step-1]for slot in range(4)];mask=[int(col<pos)for pos in positions for col in range(bucket)]
            pack=lambda values:struct.pack('<'+'q'*len(values),*values)
            metas=[b.tensor_meta(pack(tokens),[4,1],[1,1],'torch.int64'),b.tensor_meta(pack(mask),[4,bucket],[bucket,1],'torch.int64'),
                b.tensor_meta(pack(positions),[4,1],[1,1],'torch.int64'),b.tensor_meta(pack(range(4)),[4],[1],'torch.int64'),{},
                b.tensor_meta(bytes(16),[4],[1],'torch.int32'),b.tensor_meta(bytes(16),[4],[1],'torch.int32')]
            route=dict(layer=1,mask=b.tensor_meta(bytes(mask),[4,1,1,bucket],[bucket,bucket,bucket,1],'torch.bool'),
                positions=metas[2],active_mask=None,physical_key_width=bucket,phase='decode')
            cache=[{},{}]+[dict(name=name,shape=b.CACHE_SHAPE,stride=b.CACHE_STRIDE,offset=0,dtype='torch.bfloat16',sha256=str(rank+1)*64)for name in b.CACHE_NAMES]
            record=dict(complete=True,rank=rank,arm='strict_anchor_s0',call=call,bucket=bucket,
                physical_rows=[dict(frame=f,physical_row=f['slot'],current_position=positions[f['slot']])for f in call['frames']],
                changed_mask_cells=[dict(row=f['slot'],column=positions[f['slot']],before=1,after=0)for f in call['frames']],
                execution_input_metadata=metas,route_records=[{},route]+[{}for _ in range(40)],
                stage_metadata={b.NORMALIZED:b.NORM_META,b.ATTENTION:b.OUTPUT_META},stage_sha256={b.NORMALIZED:'a'*64,b.ATTENTION:'b'*64},
                before_metadata=cache,input_after_metadata=copy.deepcopy(cache))
            reference={k:refs[rank*42+ci][k]for k in ('path','sha256')};mapping[reference['path']]=record
            case=b.boundary(plan,record,reference)
            rows.append(dict(result=dict(source_case_sha256=e.r.digest(case),original_call=reference,call=call)))
        results.append(dict(rank=rank,source_stream=stream,records=rows))
    return dict(done=dict(collector=dict(collector_plan=plan))),results,mapping


class Entry(unittest.TestCase):
    def test_external_auditor_exact_contract_and_pending_flag_remain(self):
        ended,result,spec,review=terminal_fixture();self.assertEqual(e.audit_terminal(ended,result,spec,review),ended['actual_tool'])
        self.assertFalse(result['actual_auditor_tool_exit_verified']);self.assertEqual(result['original_sdk_model_exit_code'],1)
    def test_failed_bool_tool_call_missing_or_wrong_owner_audit_rejected(self):
        for fault in ('exit1','bool','call','pending','wrong_owner','truncated_proof','missing_ref','self_exit'):
            ended,result,spec,review=copy.deepcopy(terminal_fixture())
            if fault=='exit1':ended['actual_tool']['exit_code']=1
            elif fault=='bool':ended['actual_tool']['exit_code']=False
            elif fault=='call':ended['actual_tool']=dict(kind='tool_call',terminal_chunk='abcdef',exit_code=0)
            elif fault=='pending':result['complete']=False
            elif fault=='wrong_owner':ended['owner']=dict(ended['owner'],start_ticks='999')
            elif fault=='truncated_proof':result['result']['raw_files']=935
            elif fault=='missing_ref':ended.pop('audit_spec')
            else:ended['self_exit_claimed']=True
            with self.subTest(fault=fault),self.assertRaises((RuntimeError,KeyError)):e.audit_terminal(ended,result,spec,review)
    def test_original_serial_catalog_accepts_exact_external_producer_terminal(self):
        ended,*_=terminal_fixture();terminal_ref=ref('external');pin={e.SELF:'a'*64}
        row=dict(terminal=terminal_ref,tool=ended['actual_tool'],owners=[ended['owner']],units=[],source_sha256=pin)
        catalog_ref=ref('catalog');handoff_ref=ref('handoff')
        value=dict(schema='minicpm5_operator_later_serial_metadata_catalog_v1',reviewer='/root',passed=True,blockers=[],
            completed_sdk=e.r.completion.latest.sdk.BINDING,current_handoff=handoff_ref,all_later_serial_runs_included=True,runs=[row])
        mapping={catalog_ref['path']:value,handoff_ref['path']:dict(live_tool_sessions=[],active_heavy_units=[]),terminal_ref['path']:ended}
        serial=e.r.completion.latest.serial_catalog(dict(serial_predecessors=catalog_ref,source_sha256=pin),lambda ref:mapping[ref['path']],lambda:None)
        e.catalog_binding(serial,dict(runs=[]),[(terminal_ref,ended['owner'],ended['actual_tool'])])
        for change in ('missing','owner','tool','history'):
            bad=copy.deepcopy(serial);prior=dict(runs=[])
            if change=='missing':bad['runs']=[]
            elif change=='owner':bad['runs'][0]['owners']=[]
            elif change=='tool':bad['runs'][0]['tool']['session_id']+=1
            else:prior['runs']=[dict(row,terminal=ref('omitted-prior'))]
            with self.subTest(change=change),self.assertRaises(RuntimeError):e.catalog_binding(bad,prior,[(terminal_ref,ended['owner'],ended['actual_tool'])])
    def test_twelve_pure_original_cases_real_boundary_and_exact_stream(self):
        recovered,results,mapping=anchor_fixture();reads=[]
        def read(ref):reads.append(ref['path']);return mapping[ref['path']]
        with patch.object(e.sdk_reader.inputs,'AnchorStream',side_effect=AssertionError('No raw stream')):
            cases,refs=e.anchor_cases(recovered,results,read,lambda:None)
        self.assertEqual(len(reads),12);self.assertEqual(len(set(reads)),12)
        self.assertEqual([x['call']['index']for x in cases[1]],list(e.boundary.DECODE_CALLS))
        self.assertEqual([x['slot']for x in cases[0][0]['physical_frame_order']],[0,1,2,3])
        self.assertEqual(cases[0][0]['positions'],[20,21,22,23]);self.assertEqual(len(refs),12)
    def test_case_hash_stream_and_changed_slot_fail_closed(self):
        for fault in ('hash','skip','history','slot'):
            recovered,results,mapping=anchor_fixture()
            if fault=='hash':results[0]['records'][0]['result']['source_case_sha256']='f'*64
            elif fault=='skip':results[1]['source_stream']['verified_not_reexecuted_calls'].pop()
            elif fault=='history':results[0]['source_stream']['original_stream']['all_history_calls']=41
            else:mapping['/synthetic/original-4.json']['physical_rows'][0]['physical_row']=0
            with self.subTest(fault=fault),self.assertRaises((RuntimeError,ValueError)):
                e.anchor_cases(recovered,results,lambda ref:mapping[ref['path']],lambda:None)
    def test_failed_external_audit_cannot_reach_native_result_or_payload(self):
        ended,result,aspec,areview=terminal_fixture();ended['actual_tool']['exit_code']=1
        spec_ref=ref('comparison-spec');review_ref=ref('comparison-review');spec=dict(native_audit_exit=ref('audit-exit'),source_extension={})
        mapping={spec_ref['path']:spec,review_ref['path']:{},spec['native_audit_exit']['path']:ended,
            ended['result']['path']:result,ended['audit_spec']['path']:aspec,ended['audit_review']['path']:areview}
        seen=[];metadata=Mock()
        def read(reference):seen.append(reference['path']);return mapping[reference['path']]
        metadata.read.side_effect=read
        with patch.object(e,'validate_spec'),patch.object(e.a,'AuditGuard'),patch.object(e.r,'Metadata',return_value=metadata),\
             patch.object(e,'ReadIO')as reader,patch.object(e.r.p,'gone')as gone,patch.object(e.comparison,'compare')as compare,\
             self.assertRaises(RuntimeError):e.compare_saved(spec_ref,review_ref,Mock())
        reader.assert_not_called();compare.assert_not_called();gone.assert_not_called();self.assertNotIn(aspec['model_exit']['path'],seen)
    def test_reader_routes_only_original_roots_and_detects_postread_identity_change(self):
        native=Mock();native._read=set(range(1008));native.unchanged.return_value={'complete':True};guard=Mock()
        with patch.object(e.a.audit_io,'NativeAuditIO',return_value=native),patch.object(e.sdk_reader,'payload',return_value=b'ab')as sdk:
            io=e.ReadIO(dict(output_root='/synthetic/native'),{},guard,'/synthetic/sdk')
            io.read_bytes({'path':'/synthetic/native/a'},'/synthetic/native/a',2);native.read_bytes.assert_called_once()
            ref=dict(path='/synthetic/sdk/a',identity={'same':True});self.assertEqual(io.read_bytes(ref,ref['path'],2),b'ab')
            sdk.assert_called_once()
            for path in (ref['path'],'/synthetic/outsider/a'):
                with self.assertRaises(RuntimeError):io.read_bytes(ref,path,2)
            io.sdk={str(i):{'same':True}for i in range(624)}
            with patch.object(e.r,'identity',return_value={'same':False}),self.assertRaisesRegex(RuntimeError,'identity'):io.finish()
        with patch.object(e.a.audit_io,'NativeAuditIO',return_value=native),self.assertRaises(RuntimeError):
            e.ReadIO(dict(output_root='/synthetic/sdk/nested'),{},guard,'/synthetic/sdk')
    def test_guard_failure_stops_case_reads_and_native_imports_absent(self):
        recovered,results,mapping=anchor_fixture();read=Mock()
        with self.assertRaisesRegex(RuntimeError,'denied'):e.anchor_cases(recovered,results,read,Mock(side_effect=RuntimeError('denied')))
        read.assert_not_called();self.assertFalse({'torch','numpy','transformers','torch_neuronx','neuronxcc'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
