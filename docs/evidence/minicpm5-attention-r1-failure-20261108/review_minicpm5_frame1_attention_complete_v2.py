"""Whole saved attention diagnostic review, under future serial admission.

All source/metadata/byte dependencies are pinned. Native runtimes and projection,
RoPE, softmax, normalization math are not executed by this reviewer.
"""
import hashlib,sys
from pathlib import Path
import review_minicpm5_frame1_attention_metadata_v2 as meta
import review_minicpm5_frame1_attention_bytes_v1 as attention
import review_minicpm5_upstream_saved_bytes_v1 as upstream
require,same=meta.require,meta.same
BASE_PLAN=dict(path=str(meta.R/'research/minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json'),sha256='ce599a2beae22df0a05c13591854cfab8ba57519df11ee93a0d25ad4392306c8')
SOURCE_PINS={'/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_frame1_attention_metadata_v1.py': '3f2cdc52c8134b346060f0319582f488d12bd4ff043a4e8a79d3066693bb0088', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_frame1_attention_bytes_v1.py': '75573f2767fabff4343c8225115c7ba1b20d7398006b587c604e1611b75dbc53', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_upstream_saved_bytes_v1.py': '399e26d41e733bc45f88becbdc0ef4427fd19167588312730bdaef9489ec1df7', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_frame1_raw_boundaries_v1.py': '22a5e192c31ad6fe838e5ab9bb6e274e982b2a3d1239c1776d934bb2ba5d0895', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_frame1_metadata_v1.py': '8d5243d29480870b97b3821a2a54bceee3228d05eb72817c61ad872500e38ce3', '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5_frame1_attention_observer_v1.py': '87a262c1172d5cb545ec8f87c81d4950ed0f653777a800a73b6e1e0695284028', '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5_frame1_upstream_worker_v1.py': '038d998e3b04f80469a99e33d277f1c508ee33c9ca2a30b24535c1398f47a08f', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/review_minicpm5_frame1_attention_metadata_v2.py': '4b91ab5c4198ad2d18c570de2e4e224a847459516c8953cc060bf68c4c976c71', '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5_frame1_attention_observer_v2.py': '497c34a9c57ac8e402fdbe29b56b5df285e41849f410055c1f5e8f9f8827f6f1', '/home/ubuntu/experiments/trn1-latest-models-20261031/minicpm5_attention_k_consumer_layout_v1.py': 'b11685ee1b18d23681608196a751aa37bb6b83b5f87e016498d07d503ebd07d2'}
REORDERS=(('q_projection','q_rope_input'),('k_projection','k_rope_input'),('v_projection','v_heads'),('context_heads','context_before_o'))


def authentic_sources():
    for path,sha in SOURCE_PINS.items():same(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha,'Original saved-byte reviewer source remains exact')
    require(not {'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Independent stdlib review imports only')


def equal_rows(left,right,guard):
    count=0
    while True:
        a=next(left,None);b=next(right,None)
        if a is None or b is None:
            require(a is None and b is None,'No truncated reshape rows');break
        if count%64==0:guard()
        same(a,b,'Original reshape/transpose changes no logical token values');count+=1
    require(count>0,'Nonempty original layout transformation');return count


def reorders(root,reports,guard):
    out={}
    for role in meta.ROLES:
        native=role=='native';rows={x['name']:x for x in reports[role]['attention']['captures']};out[role]={}
        for before,after in REORDERS:
            pair=[]
            for name in (before,after):
                shape=attention.source.geometry(native)[name]['shape']
                data=attention.read_payload(rows[name],root/role/'attention',shape,'torch.bfloat16',True,guard)
                pair.append(iter(attention.token_rows(data,shape,name,native)))
            count=equal_rows(*pair,guard)
            same(count,attention.source.LENGTH if native else attention.source.BUCKET,'All original real and padded rows of reshape')
            out[role][before+'_to_'+after]=dict(all_rows_bitwise_equal=True,rows=count)
    return out


def consistency(metadata,internal,boundaries,norms):
    require(metadata['all21_prior_upstream_metadata_exact']is True and internal['all_payloads_authenticated']is True,'Actual complete observations and exact retained reference hashes')
    same(internal['controls'],metadata['controls'],'Independent own-input bytes match metadata conclusions')
    same(internal['all21_own_input_controls_exact'],metadata['all21_own_input_controls_claim_exact'],'Actual own-input pass verdict')
    same(len(boundaries['references']),21,'Whole original seven boundaries for all roles');same(len(norms),6,'All original norm files')
    count=internal['files']+len(boundaries['references'])+len(norms)
    size=internal['raw_bytes']+sum(x['bytes']for x in boundaries['references'])+sum(x['bytes']for x in norms)
    same((count,size),(metadata['files'],metadata['raw_bytes']),'Every complete saved payload accounted for')
    same((count,size),(96,353284922),'Original complete96-file byte budget')
    for internal_name,upstream_name in [('normalized_input','attention_input'),('projection_output','attention_output')]:
        same(internal['observed_boundary_metrics'][internal_name]['real_tokens'],boundaries['metrics'][upstream_name]['real_tokens'],'Same full comparison at overlapping internal/upstream boundary')
        same(internal['observed_boundary_metrics'][internal_name]['prediction'],boundaries['metrics'][upstream_name]['prediction'],'Same prediction comparison at overlapping boundary')
    return count,size


def review(*,root,capture_refs,guard):
    require(callable(guard),'Actual admitted serial reviewer guard');guard();authentic_sources();root=Path(root)
    require(root.is_absolute()and root.resolve()==root,'Exact actual output root');same(list(capture_refs),list(meta.ROLES),'All three ordered original roles')
    reports={}
    for role,reference in capture_refs.items():
        same(reference['path'],str(root/role/'capture.json'),'Exact completed capture metadata path');reports[role]=meta.load(reference)
    plan=meta.load(BASE_PLAN)
    metadata=meta.validate(root=root,reports=reports,plan=plan)
    # Reuse the independently reviewed whole-row reader. Empty replay means
    # recompute metrics without borrowing any earlier run's claimed result.
    boundaries=upstream.audit_boundaries(root,reports,{},guard)
    norms=upstream.verify_norm_weights(root,reports,guard)
    residuals=upstream.audit_residuals(root,reports,guard)
    internal=attention.audit_attention(root,reports,guard)
    transformations=reorders(root,reports,guard)
    files,size=consistency(metadata,internal,boundaries,norms)
    passed=internal['all21_own_input_controls_exact']
    guard();return dict(schema='minicpm5_frame1_attention_complete_review_v1',complete=True,captures=capture_refs,
                source_pins=SOURCE_PINS,base_plan=BASE_PLAN,metadata=metadata,upstream_boundary_files=boundaries['references'],
                upstream_boundary_metrics=boundaries['metrics'],norm_weight_files=norms,independent_residual_checks=residuals,
                attention=internal,reorder_checks=transformations,files=files,raw_bytes=size,
                all96_full_payloads_authenticated=True,all21_original_upstream_payloads_exact=True,
                all21_own_input_controls_exact=passed,passed=passed,
                classification='original_full_prefix_observation_and_own_input_controls_only'if passed else'own_input_control_failed_no_precision_intervention',
                trigonometry_or_attention_independently_reexecuted=False,projection_math_independently_reexecuted=False,
                norm_math_independently_reexecuted=False,residual_math_independently_reexecuted=True,
                process_lifecycle_verified=False,unique_internal_or_final_KL_cause_proven=False,
                full_model_qualified=False,new_speedup_claim=False)
