"""Stream authenticated complete histories into separately admitted decode replay.

This module supplies no runtime admission. The caller must authenticate actual
successful collector and independent saved-audit exits, bind this exact plan,
rank result and audit, and retain serial ownership in guard before every read.
All42 calls per rank are read and checked; only18 complete decode inputs are
returned. Prefill prediction taps never become complete prefill inputs.
"""
from pathlib import Path
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research import minicpm5_layer1_raw_verifier_v1 as raw
from research import minicpm5_layer1_payload_io_v1 as payload_io
require=raw.require


def audit_records(audit):
    require(audit['schema']=='minicpm5_layer1_reviewed_replay_inputs_v1'
        and audit['complete']is True and audit['attention_causality_proven']is False
        and audit['full_model_qualified']is audit['new_speedup_claim']is False,
        'Exact independent complete saved-data audit remains diagnostic')
    require(audit['original_fp32_metrics_exact_by_authenticated_equivalence']is True
        and audit['strict_float64_scalar_tolerance_passed']is False
        and audit['metric_equivalence']['metric_frames']==96
        and audit['metric_equivalence']['call_count']==84,
        'Explicit original-metric proof with the scalar tolerance failure retained')
    numeric=audit['numeric'];cache=audit['layer1_cache']
    require(numeric['schema']=='minicpm5_layer1_full_logits_metric_diagnostic_v1'
        and numeric['strict_numeric_audit_passed']is False
        and numeric['complete']is True and numeric['logits_payload_count']==84
        and numeric['official_reference_frames']==32 and numeric['candidate_metric_frames']==96
        and numeric['original_selected_tap_bytes']==3145728
        and numeric['replicated_ranks_verified_and_counted_once']is True
        and numeric['metrics_independently_recomputed']is True
        and numeric['exact_classification_and_argmax']is True,
        'All full-vocabulary outputs and original official metrics already audited')
    require(cache['schema']=='minicpm5_layer1_independent_byte_verification_v1'
        and cache['complete']is True and cache['ranks']==2 and cache['arms']==3
        and cache['calls']==84 and cache['frames_per_arm']==32
        and cache['full_cache_records']==168 and cache['complete_cache_bytes']==168*raw.CACHE_BYTES
        and cache['every_tap_and_full_cache_hash_exact']is True
        and cache['all_own_history_links_exact']is True,
        'Both ranks and every complete original input cache already audited')
    rows=cache['raw_records']
    require(type(rows)is list and len(rows)==84,'Exact84 audited call records')
    for index,row in enumerate(rows):
        require([row['rank'],row['arm'],row['call']]==[index//42,raw.ARMS[(index%42)//14],index%14]
            and row['complete_cache_bytes']==2*raw.CACHE_BYTES,'All audited calls in original rank/arm order')
    return rows


class RankStream:
    """One-use same-rank full trajectory; caller drives both TP ranks in order."""
    def __init__(self,plan,rank_result,audit,read_json,guard):
        require(callable(guard)and callable(read_json),'Independent actual admission and authenticated JSON reader')
        guard()
        self.plan=plan;self.result=rank_result;self.audit=audit
        self.read_json=read_json;self.guard=guard;self.rows=audit_records(audit)
        self.rank=rank_result['rank'];self.root=payload_io.safe_path(plan['output_root'])
        require(type(self.rank)is int and self.rank in (0,1)and rank_result['complete']is True
            and rank_result['schema']=='minicpm5_layer1_observation_cpu_rank_v1'
            and rank_result['arms']==list(raw.ARMS),'Actual complete collected rank')
        require(len(plan['layer1_observation']['original_calls'])==84
            and len(plan['calls'])==14 and len(plan['frames'])==32,'Original complete trajectory plan')
        self.entered=False;self.complete=False;self.seen=[];self.decoded=[];self.bytes=0

    def __iter__(self):
        self.guard();require(not self.entered,'Decode stream may be consumed only once')
        self.entered=True
        entries=self.result['arm_reports']
        require(len(entries)==3 and [v['arm']for v in entries]==list(raw.ARMS),'All3 original arms in order')
        for ai,entry in enumerate(entries):
            self.guard();arm=raw.ARMS[ai];folder=self.root/('cpu-rank'+str(self.rank))/arm
            require(entry['report']==str(folder/'result.json'),'Exact actual collected arm path')
            report=self.read_json(entry['report'],entry['sha256'])
            require(report['complete']is True and report['rank']==self.rank and report['arm']==arm
                and report['schema']=='minicpm5_layer1_observation_cpu_arm_v1'
                and [item['call']for item in report['reports']]==list(range(14)),
                'All14 actual calls of this rank and arm')
            previous=None;frames=[]
            for ci,item in enumerate(report['reports']):
                self.guard();callfolder=folder/('call-%02d'%ci)
                require(item['report']==str(callfolder/'result.json'),'Exact collected call path')
                call=self.read_json(item['report'],item['sha256'])
                index=self.rank*42+ai*14+ci;original_ref=self.plan['layer1_observation']['original_calls'][index]
                require([original_ref[k]for k in ('rank','arm','call')]==[self.rank,arm,ci],
                    'Exact original frozen call reference ordering')
                original_ref={k:original_ref[k]for k in ('path','sha256')}
                original=self.read_json(original_ref['path'],original_ref['sha256'])
                require(call['complete']is True and call['schema']=='minicpm5_layer1_observation_cpu_call_v1'
                    and [call['rank'],call['arm'],call['call']]==[self.rank,arm,self.plan['calls'][ci]],
                    'Actual complete original call identity')
                packed=payload_io.read_owned_payload(call['layer1_observation'],callfolder/'layer1-observation.bin',self.guard)
                proof,state=raw.verify_call(call,original,original_ref,packed,previous)
                require(proof==self.rows[index],'Every reread byte agrees with independently audited original call')
                previous=state;self.seen.append(dict(rank=self.rank,arm=arm,call=ci))
                frames+=proof['physical_frame_order'];self.bytes+=len(packed)
                require(len(frames)==len(set(frames)) and self.bytes<=128*2**20,'Original frame coverage and rank byte cap')
                if ci in boundary.DECODE_CALLS:
                    case=boundary.boundary(self.plan,original,original_ref)
                    tap_entries={v['name']:v for v in call['layer1_observation']['entries'][:3]}
                    def tap(name):
                        entry=tap_entries[name];data=packed[entry['offset']:entry['offset']+entry['bytes']]
                        require(len(data)==16384 and raw.sha(data)==original['stage_sha256'][name],
                            'Complete four-row decode tap bytes')
                        return data
                    normalized=tap(boundary.NORMALIZED);attention=tap(boundary.ATTENTION)
                    require(len(proof['physical_frame_order'])==4 and
                        proof['physical_frame_order']==[f['index']for f in case['physical_frame_order']],
                        'Original logical-to-physical decode row mapping')
                    self.guard();self.decoded.append(dict(rank=self.rank,arm=arm,call=ci))
                    yield dict(case=case,normalized_raw=normalized,cache_raw=dict(state['caches']),
                        expected_attention_raw=attention,audited_call=proof)
                    self.guard()
                if ci==13:require(sorted(frames)==list(range(32)),'Every32 frame in each original arm retained')
            previous=None
        require(len(self.seen)==42 and len(self.decoded)==18,'All42 history calls and18 complete decode calls')
        self.guard();self.complete=True

    def finish(self):
        self.guard();require(self.entered and self.complete,'Must exhaust all original histories and decode inputs')
        return dict(schema='minicpm5_layer1_rank_decode_stream_v1',complete=True,rank=self.rank,
            all_history_calls=42,decode_calls=18,physical_decode_rows=72,full_cache_records=84,
            restored_full_cache_bytes=84*raw.CACHE_BYTES,read_payload_bytes=self.bytes,
            all_read_bytes_match_saved_audit=True,prefill_replayed=False,attention_reexecuted=False,
            full_model_qualified=False,new_speedup_claim=False)
