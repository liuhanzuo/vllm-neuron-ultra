"""Original full stream verification with explicit anchor-only replay selection.

No import-time I/O. External caller authenticates completed predecessor bundles
and owns admission. Other arms are reread/verified, never reported reexecuted.
"""
from pathlib import Path
from research import minicpm5_layer1_decode_stream_v2 as stream
from research import minicpm5_layer1_sdk_batch_layout_v1 as core
from research import minicpm5_layer1_payload_io_v1 as payload
require=core.require


class AnchorStream:
    def __init__(self,plan,rank_result,audit,read_json,guard):
        guard();self.guard=guard;self.rank=rank_result['rank']
        self.source=stream.RankStream(plan,rank_result,audit,read_json,guard)
        self.checked=[];self.selected=[];self.skipped=[];self.complete=False
    def __iter__(self):
        for item in self.source:
            self.guard();case=item['case']
            descriptor=dict(rank=case['rank'],arm=case['arm'],call=case['call']['index'])
            self.checked.append(descriptor)
            if case['arm']==core.ANCHOR:
                self.selected.append(descriptor);yield item
            else:self.skipped.append(descriptor)
        expected=[dict(rank=self.rank,arm=arm,call=ci)for arm in stream.raw.ARMS for ci in core.boundary.DECODE_CALLS]
        require(self.checked==expected,'Every original18 decode input verified in original order')
        require(self.selected==expected[:6] and len(self.skipped)==12,'Only six original anchor calls selected')
        self.complete=True;self.guard()
    def finish(self):
        self.guard();require(self.complete,'Entire original trajectory exhausted including skipped arms')
        original=self.source.finish()
        require(original['all_history_calls']==42 and original['decode_calls']==18
            and original['attention_reexecuted']is original['prefill_replayed']is False,
            'Original stream authenticates inputs; no forward work claimed')
        return dict(schema='minicpm5_layer1_sdk_batch_layout_anchor_stream_v1',complete=True,rank=self.rank,
            original_stream=original,selected_anchor_calls=self.selected,verified_not_reexecuted_calls=self.skipped,
            selected_decode_calls=6,skipped_decode_calls=12,all_other_arms_verified_not_reexecuted=True,
            prefill_reexecuted=False,full_model_qualified=False,new_speedup_claim=False)


def previous_controls(plan,bundle,case,guard):
    guard()
    require(case['arm']==core.ANCHOR and bundle['binding']==plan['layer1_sdk_batch_layout'],
        'Exact externally authenticated completed current/history binding')
    rank=case['rank'];result=bundle['records']['rank%d_result'%rank]
    expected=[(arm,ci,mode)for arm in stream.raw.ARMS for ci in core.boundary.DECODE_CALLS for mode in core.factorial.MODES]
    require(result['schema']=='minicpm5_layer1_current_history_factorial_cpu_driver_v1'
        and result['complete']is True and result['rank']==rank and len(result['records'])==90
        and [(r['result']['arm'],r['result']['call']['index'],r['result']['mode'])for r in result['records']]==expected,
        'Full actual previous90 output record order preserved')
    controls={}
    for mode in ('own','official_both','restore'):
        row=result['records'][expected.index((core.ANCHOR,case['call']['index'],mode))]
        actual=row['result'];desc=row['output']
        path=Path(bundle['spec']['output_root'])/('cpu-rank%d'%rank)/core.ANCHOR/('call-%02d'%case['call']['index'])/mode/'attention.bf16'
        require(actual['rank']==rank and actual['call']==case['call'] and desc['path']==str(path)
            and desc['raw_bytes']==16384 and desc['sha256']==actual['attention_output_sha256']
            and plan['inputs'][str(path)]==dict(sha256=desc['sha256'],identity=desc['identity']),
            'Exact previous full output descriptor and frozen identity')
        raw=payload.read_owned_payload(dict(desc,payload_bytes=16384,payload_sha256=desc['sha256']),path,guard)
        require(type(raw)is bytes and len(raw)==16384 and core.boundary.sha(raw)==desc['sha256'],
            'Full authenticated previous output bytes')
        controls[mode]=dict(result=actual,output=desc,raw=raw)
    guard();return controls
