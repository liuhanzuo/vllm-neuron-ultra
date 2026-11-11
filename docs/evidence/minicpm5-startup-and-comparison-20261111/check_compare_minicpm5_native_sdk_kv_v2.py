"""Pure compare_frame controls with distinct KV heads, positions and padded bytes."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import compare_minicpm5_layer1_native_sdk_v3 as c
from research.check_compare_minicpm5_layer1_native_sdk_v2 import pack


def fixture():
    p=5;length=6;bucket=128;slot=2
    frames=[dict(index=10+i,fixture=7-i,step=1,slot=i)for i in range(4)]
    case=dict(frame=frames[slot],call=11,cache_length=length,logical_position=p)
    sc=dict(rank=1,call=dict(index=11,phase='decode'),physical_frame_order=frames,positions=[3,4,5,6],bucket=bucket)
    def value(wanted):return dict(raw=bytes(wanted['bytes']),shape=list(wanted['shape']),dtype=wanted['dtype'],evidence=[{'reference':{'synthetic':True}}])
    ne={};se={}
    for event in c.native_manifest.event_manifest(length):
        wanted=event['value']
        if type(wanted)is list:
            for i,t in enumerate(wanted):ne[event['label'],(i,)]=value(t)
        elif'shape'in wanted:ne[event['label'],()]=value(wanted)
    for event in c.sdk_manifest.events(bucket):
        for t in event['tensors']:se[event['label'],tuple(t['path'])]=value(t)
    for label in ('attention_mask','prior_actual_mask'):
        se[label,()]['raw']=b''.join(bytes([1]*n+[0]*(bucket-n))for n in sc['positions'])
    def word(head,pos,kind):return 16*head+pos+1+(64 if kind=='value'else 0)
    for kind,i in [('key',0),('value',1)]:
        def raw(heads,positions):return b''.join(pack(word(h,j,kind),'torch.bfloat16')*128 for h in heads for j in positions)
        ne['cache_before',(i,)]['raw']=raw((0,1),range(p))
        ne['cache_after',(i,)]['raw']=raw((0,1),range(length))
        ne['native_effective_'+kind,()]['raw']=raw((0,1),range(length))
        active='cache_update_active_key'if kind=='key'else'v_heads'
        ne[active,()]['raw']=raw((0,1),(p,))
        if kind=='key':ne['k_post_rope',()]['raw']=raw((0,1),(p,))
        # Every SDK physical slot has distinct padding; selected slot contains rank1 head.
        rows=[]
        for index in range(4):
            rows.append(raw((1,),range(p)) + pack(200+index,'torch.bfloat16')*128*(bucket-p)
                if index==slot else pack(100+index,'torch.bfloat16')*128*bucket)
        se['effective_prior_kv',(i,)]['raw']=b''.join(rows)
        prepared='k_after_prep'if kind=='key'else'v_after_prep'
        for label in [prepared,'k_post_rope'if kind=='key'else'v_heads']:
            se[label,()]['raw']=b''.join(raw((1,),(p,))if index==slot else pack(100+index,'torch.bfloat16')*128 for index in range(4))
    native=dict(case=case,events=ne,scale_operations={});sdk=dict(case=sc,events=se,scale_operations={})
    shards=dict(schema='minicpm5_layer1_official_shards_v1',complete=True,rank=1,all_four_projection_shards_raw_exact=True,
        precision_changed=False,q_head_ids=list(range(8,16)),kv_head_ids=[1])
    return native,sdk,shards


class KV(unittest.TestCase):
    def test_nonzero_rank1_head_all_prior_positions_active_and_padding(self):
        native,sdk,shards=fixture();result=c.compare_frame(native,sdk,shards,lambda:None)
        by={x['boundary']:x for x in result['boundaries']}
        for label in ('cache_before_key','cache_before_value','native_effective_key','native_effective_value',
            'cache_after_key','cache_after_value','cache_update_active_key','k_post_rope','k_after_prep','v_heads','v_after_prep'):
            self.assertEqual(by[label]['status'],'equal',label)
        selected=by['cache_before_key']['sdk_evidence'][0]['logical_selection_chain']
        self.assertEqual(selected[0]['ranges'][0],[2,3]);self.assertEqual(selected[1]['ranges'][2],[0,5])
        native_selection=by['cache_before_key']['native_evidence'][0]['logical_selection_chain'][0]
        self.assertEqual(native_selection['ranges'][1],[1,2])
        padding=next(x for x in result['sdk_padding']if x['label']=='effective_prior_kv'and x['tensor_path']==[0])
        self.assertEqual(padding['minimum'],202);self.assertEqual(padding['maximum'],202)
        self.assertEqual(padding['elements'],(128-5)*128)
    def test_shifted_prior_position_wrong_kv_head_and_active_value_report_exact_boundary(self):
        for fault in ('position','head','active'):
            native,sdk,shards=fixture();events=sdk['events']
            label,path=('v_after_prep',())if fault=='active'else('effective_prior_kv',(0,))
            value=events[label,path];raw=bytearray(value['raw'])
            start=2*128*128*2 if fault!='active'else 2*128*2
            if fault=='position':raw[start:start+256],raw[start+256:start+512]=raw[start+256:start+512],raw[start:start+256]
            elif fault=='head':raw[start:start+256]=pack(1,'torch.bfloat16')*128
            else:raw[start:start+2]=pack(0,'torch.bfloat16')
            events[label,path]=dict(value,raw=bytes(raw));result=c.compare_frame(native,sdk,shards,lambda:None)
            expected='native_effective_value'if fault=='active'else'cache_before_key'
            # v_after_prep is observed before complete effective cache, and correctly wins the first boundary.
            if fault=='active':expected='v_after_prep'
            self.assertEqual(result['first_observed_difference'],expected,fault)
            row=next(x for x in result['boundaries']if x['boundary']==expected)
            self.assertGreater(row['numeric_mismatch_elements'],0);self.assertGreater(row['maximum_absolute_error'],0)
    def test_bucket128_head_dimension_never_masquerades_as_prior_padding(self):
        native,sdk,shards=fixture();result=c.compare_frame(native,sdk,shards,lambda:None)
        actual={(x['label'],tuple(x['tensor_path']))for x in result['sdk_padding']}
        expected={(x,())for x in c.SDK_PRIOR_SCORE_OR_MASK_LABELS}|{
            ('effective_prior_kv',(0,)),('effective_prior_kv',(1,)),
            ('route_inputs_sdk_decode',('past_key_value',0)),('route_inputs_sdk_decode',('past_key_value',1)),
            ('route_inputs_sdk_decode',('attention_mask',))}
        self.assertEqual(actual,expected)
        for label in ('q_rope_input','k_rope_input','q_post_rope','k_post_rope','q_after_prep','k_after_prep','v_after_prep',
            'v_heads','pv_prior','pv_active','qk1_scaled','probabilities_active_fp32'):
            self.assertIsNone(c.padding_axis(label,()))
            self.assertNotIn((label,()),actual)
        self.assertIsNone(c.padding_axis('route_inputs_sdk_decode',('Q',)))
        self.assertIsNone(c.padding_axis('returned_current_kv',(0,)))
        self.assertNotIn('all_available_boundaries_compared',result)
        coverage={x['boundary']for x in result['comparison_coverage']}
        self.assertTrue({'q_after_prep','k_after_prep','v_after_prep'}<=coverage)
        only={(x['label'],tuple(x['tensor_path']))for x in result['authenticated_sdk_leaves_not_individually_cross_compared']}
        self.assertIn(('returned_current_kv',(0,)),only);self.assertIn(('route_inputs_sdk_decode',('Q',)),only)
    def test_unsupported_mask_stays_unknown_recognized_blocked_position_is_difference(self):
        for word,status in [(1,'unknown'),(-3.3895313892515355e38,'different')]:
            native,sdk,shards=fixture()
            for label in ('attention_mask','actual_eager_mask'):
                value=native['events'][label,()];native['events'][label,()]=dict(value,raw=pack(word,'torch.bfloat16')+value['raw'][2:])
            result=c.compare_frame(native,sdk,shards,lambda:None)
            row=next(x for x in result['boundaries']if x['boundary']=='mask')
            self.assertEqual(row['status'],status)
            if status=='unknown':self.assertIsNone(row['values_equal']);self.assertIn('mask',result['unknown_boundaries'])


if __name__=='__main__':unittest.main(verbosity=2)
