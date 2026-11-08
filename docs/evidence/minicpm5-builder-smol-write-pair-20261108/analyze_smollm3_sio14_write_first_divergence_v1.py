"""Actual saved graph bytes through first token divergence; no native execution."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
import smollm3_observer_invariance_audit_v25 as audit
import smollm3_token_capture_export_v3 as exporter


def reference(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def once(path,value):
    raw=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode();assert len(raw)<8*2**20
    with path.open('xb')as f:f.write(raw)
    return reference(path)

def load_frames(run,export,request):
    selected={i for sample in request['samples']for i in sample['frames']};frames={};refs=[]
    for ref in export['frames_chunks']:
        if not any(ref['start']<=i<ref['start']+ref['count']for i in selected):continue
        path=Path(ref['path']);assert path.parent==ROOT/'sio14'/run and reference(path)['sha256']==ref['sha256']
        chunk=audit.l.read(path);assert chunk['kind']=='frames'and chunk['start']==ref['start']and len(chunk['rows'])==ref['count']
        frames.update({row['index']:row for row in chunk['rows']if row['index']in selected});refs.append(ref)
    assert set(frames)==selected and len(frames)==512
    return frames,refs


def raw_selected(run,export,frames):
    events={row['observer_event']:row for row in frames.values()};observed={};path=Path(export['observer_file'])
    assert path.parent==ROOT/'sio14'/run
    before=audit.l.ordinary(path);assert before.st_size<=64*2**20
    h=hashlib.sha256();count=0;last_event=0
    with path.open('rb')as stream:
        for line in stream:
            assert len(line)<=256*1024 and line.endswith(b'\n');h.update(line);count+=len(line);row=json.loads(line)
            assert row['event']==last_event+1;last_event=row['event']
            assert row['pid']==export['observer_producer']['pid']and row['role']==export['role']
            if row['event']not in events:continue
            expected=events[row['event']]
            assert row['kind']=='graph_inputs'and row['scope']=='request'
            assert all(row[k]==expected[k]for k in ('engine_step','tag','route','scope'))
            decoded=[exporter.decode_tensor(item)for item in row['inputs']]
            metadata=[{k:v for k,v in item.items()if k!='values'}for item in decoded]
            assert metadata==expected['inputs']and len(decoded)==7
            observed[expected['index']]=decoded
    after=audit.l.ordinary(path)
    assert count==before.st_size and h.hexdigest()==export['observer_sha256']
    assert all(getattr(before,k)==getattr(after,k)for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    assert set(observed)==set(frames)
    return observed,dict(path=str(path),sha256=h.hexdigest(),bytes_read=count,events=last_event,selected_graph_invocations=len(frames))


def run():
    previous=Q/'smollm3-sio14-first-write-pair-root-audit-20261108-v1.json'
    assert reference(previous)['sha256']=='550c536c8ce5a7c233c7ed8504eea61b0d8a52ef0ea6211258d5831188a549ea'
    # Independently reauthenticate the reference observer producer and wire data.
    base=audit.p.load_base();capture,proof,export=audit.audit_run(base,'r01',write_export=False)
    proof_ref=once(Q/'smollm3-sio14-r01-divergence-mode-audit-20261108-v1.json',proof)
    reports={'r01':export,'r05':audit.l.read(ROOT/'sio14/r05/replay-export.json')}
    earlier=json.loads(previous.read_text());assert reference(ROOT/'sio14/r05/replay-export.json')==earlier['runs']['r05']['export']
    requests={};frames={};decoded={};provenance={}
    for name,value in reports.items():
        request=value['requests'][-1]
        assert request['group']=='short-round0-single'and request['index']==0 and len(request['samples'])==len(request['token_ids'])==512
        assert value['no_cache_snapshot_included'] is True and value['teacher_forced_replay_executed'] is False
        requests[name]=request;frames[name],chunk_refs=load_frames(name,value,request)
        decoded[name],raw_proof=raw_selected(name,value,frames[name])
        provenance[name]=dict(export=reference(ROOT/'sio14'/name/'replay-export.json'),chunks=chunk_refs,observer=raw_proof,
                              engine_request_id=request['engine_request_id'],actual_graph_frames=value['frames_count'],actual_engine_steps=value['engine_steps_count'])
    assert requests['r01']['prompt_token_ids']==requests['r05']['prompt_token_ids']
    rows=[];different=[]
    for index,(a,b)in enumerate(zip(requests['r01']['samples'],requests['r05']['samples'])):
        assert len(a['frames'])==len(b['frames'])==1
        left=frames['r01'][a['frames'][0]];right=frames['r05'][b['frames'][0]]
        left_inputs=decoded['r01'][left['index']];right_inputs=decoded['r05'][right['index']]
        different_args=[i for i,(x,y)in enumerate(zip(left_inputs,right_inputs))if x!=y]
        token_equal=a['token']==b['token']
        if not token_equal:different.append(index)
        rows.append(dict(generated_index=index,token_equal=token_equal,reference_token=a['token'],write_token=b['token'],
                         physical_slot_equal=a['physical_slot']==b['physical_slot'],physical_slots=[a['physical_slot'],b['physical_slot']],
                         graph_route_equal=left['route']==right['route'],graph_tag_equal=left['tag']==right['tag'],
                         reference_frame=left['index'],write_frame=right['index'],different_external_input_arguments=different_args))
    assert different
    first=different[0];a=rows[first];left=frames['r01'][a['reference_frame']];right=frames['r05'][a['write_frame']]
    aligned_before=all(not row['different_external_input_arguments']and row['physical_slot_equal']and row['graph_route_equal']and row['graph_tag_equal']for row in rows[:first+1])
    report=dict(schema='smollm3_sio14_first_write_divergence_graph_input_analysis_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_write_pair_audit=reference(previous),independent_reference_producer_audit=proof_ref,provenance=provenance,
        request_group='short-round0-single',prompt_tokens=len(requests['r01']['prompt_token_ids']),output_tokens=512,
        first_different_generated_index=first,all_different_generated_indices=different,
        same_external_input_bytes_layout_dtype_shape_through_first_difference=aligned_before,
        identical_external_input_invocations_through_first_difference=first+1 if aligned_before else None,
        first_difference=dict(reference_frame=left,write_frame=right,reference_token=a['reference_token'],write_token=a['write_token']),
        per_generated_token=rows,all_actual_graph_inputs_decoded_and_raw_hash_verified=True,
        same_prior_global_schedule_proven=False,same_internal_cache_values_proven=False,same_internal_storage_aliases_proven=False,
        same_internal_model_load_effects_proven=False,no_cache_snapshots=True,causal_attribution_proven=False,
        full_graph_state_equivalence_proven=False,teacher_forced_replay_executed=False,model_execution_performed=False,
        interpretation='The first different sampled token follows identical recorded external graph arguments and same physical request slot; cache contents, storage aliases and all prior model execution state were not captured or proven equal.',
        next_required_evidence='After the full observer invariance matrix, review actual cache/alias/internal execution state or a controlled full-history replay before attributing the difference to a numerical kernel or cache update.',
        full_model_qualified=False,new_speedup_claim=False)
    out=Q/'smollm3-sio14-first-write-divergence-graph-input-analysis-20261108-v1.json'
    ref=once(out,report)
    print(json.dumps(dict(report=ref,first_difference=first,different_token_count=len(different),identical_graph_inputs_through_divergence=aligned_before,
                         invocations=report['identical_external_input_invocations_through_first_difference'],later_input_difference_counts={str(i):sum(i in row['different_external_input_arguments']for row in rows)for i in range(7)},
                         internal_cache_equivalence_proven=False)))

if __name__=='__main__':run()
