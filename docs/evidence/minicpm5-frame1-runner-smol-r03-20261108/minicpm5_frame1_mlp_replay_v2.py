"""One fixed-input BF16 down-projection contrast, only after exact controls."""
import hashlib
import json
from pathlib import Path

import minicpm5_frame1_mlp_worker_v2 as w
from minicpm5_saved_oproj_fullshape_intervention_worker_v1 import measure


def restore(torch,row,directory,guard):
    guard();path=Path(row['path'])
    w.require(path.parent==directory and path.name==row['name']+'.bf16','Finite captured tensor path')
    w.same(w.identity(path),row['identity'],'Fresh observed payload identity')
    w.require(row['observed_not_recomputed'] is True and row['encoding']=='packed_bf16_le','Actual observed BF16 capture')
    native=directory.name=='native'
    w.require(directory.name in ('native','cpu-rank0','cpu-rank1') and row['name'] in w.FIELDS,'Exact observation role/field')
    w.same(row['metadata']['shape'],w.expected_shapes(native)[row['name']],'Original full observed geometry')
    w.require(0<row['bytes']<=w.CAPTURE_FILE_CAP,'Bounded capture payload')
    data=path.read_bytes()
    w.require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']==row['metadata']['sha256'],
              'Actual captured tensor bytes')
    meta=row['metadata'];span=w.checked_layout(meta['shape'],meta['stride'],meta['offset'])
    w.same(meta['dtype'],'torch.bfloat16','BF16 original observation')
    w.require(2*span==len(data),'Exact dense storage extent')
    packed=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(meta['shape'])
    restored=torch.empty_strided(meta['shape'],meta['stride'],dtype=torch.bfloat16)
    restored.copy_(packed)
    w.same(w.metadata(torch,restored),meta,'Full original shape/stride/dtype/raw roundtrip')
    w.require(bool(torch.isfinite(restored).all()),'Finite restored capture')
    w.same(w.identity(path),row['identity'],'Capture unchanged during read')
    return restored


def native_math(torch,x,weights):
    gate=torch.nn.functional.linear(x,weights['gate'])
    up=torch.nn.functional.linear(x,weights['up'])
    product=torch.nn.functional.silu(gate)*up
    output=torch.nn.functional.linear(product,weights['down'])
    return dict(input=x,gate=gate,up=up,down_input=product,output=output)


def sdk_math(torch,x,weights):
    ranks=[]
    for rank in (0,1):
        lo,hi=rank*3072,(rank+1)*3072
        wg=weights['gate'][lo:hi].contiguous();wu=weights['up'][lo:hi].contiguous()
        wd=weights['down'][:,lo:hi].contiguous()
        gate=torch.einsum('...m,mn->...n',x,wg.t())
        up=torch.einsum('...m,mn->...n',x,wu.t())
        product=torch.nn.functional.silu(gate)*up
        partial=torch.einsum('...m,mn->...n',product,wd.t())
        ranks.append(dict(input=x,gate=gate,up=up,down_input=product,partial=partial))
    output=ranks[0]['partial']+ranks[1]['partial']
    for rank in ranks:rank['output']=output
    return ranks


def exact_controls(torch,computed,observed,label):
    rows={}
    for field in w.FIELDS:
        # Layout is a control too; a contiguous shortcut for Xc is forbidden.
        w.same(w.metadata(torch,computed[field]),w.metadata(torch,observed[field]),label+' exact full '+field)
        rows[field]=dict(raw_exact=True,source_layout_exact=True,sha256=w.metadata(torch,computed[field])['sha256'])
    return rows


def compare_scopes(torch,a,b):
    w.same(list(a.shape),list(b.shape),'Same compared full geometry')
    return dict(full=measure(torch,a,b),real_tokens=measure(torch,a[:,:w.LENGTH],b[:,:w.LENGTH]),
                prediction=measure(torch,a[:,w.LENGTH-1:w.LENGTH],b[:,w.LENGTH-1:w.LENGTH]))


def classify(front_exact,same_sdk_full,down_matches_full):
    if front_exact and not same_sdk_full and down_matches_full:
        return 'down_path_sufficient_for_this_fixed_input_not_final_KL_causality'
    if front_exact and not down_matches_full:
        raise RuntimeError('Identical complete down inputs with same layout produced contradictory output')
    if not front_exact:
        return 'front_half_differs_down_is_not_established_as_unique_cause'
    return 'no_output_difference_for_this_fixed_input'


def run(plan,root,capture_refs,guard):
    root=Path(root);guard();w.validate_science(plan)
    w.require(w.sys.byteorder=='little','Explicit captured BF16 little-endian host')
    w.require(set(capture_refs)=={'native','cpu-rank0','cpu-rank1'},'All three actual captures required')
    reports={}
    for role,reference in capture_refs.items():
        path=Path(reference['path']);w.require(path==root/role/'capture.json','Exact capture metadata path')
        data=path.read_bytes();w.require(len(data)<2*w.MIB and hashlib.sha256(data).hexdigest()==reference['sha256'],'Bounded actual capture metadata')
        report=json.loads(data);native=role=='native';rank=-1 if native else int(role[-1])
        w.require(report['complete']is True and report['native']is native and report['rank']==rank,'Complete actual role')
        w.same(report['frame'],w.FRAME,'One original frame1')
        w.require(report['anchor']['all_six_exact']is True and report['history_consumed']is False and
                  report['downstream_layers_executed']==0,'Anchored no-history prefix only')
        w.same([x['name']for x in report['captures']],list(w.FIELDS),'Five finite actual observations')
        w.same(report['schema'],'minicpm5_frame1_mlp_capture_v2','Actual successor capture')
        w.same(report['precision_interventions'],[],'Capture used strict unmodified baseline')
        w.same(report['oproj_witnesses'],[],'No o_proj precision intervention')
        if not native:
            w.require(report['cpu_full_anchor']['all_four_fullshape_exact'] is True,'Four full CPU anchors exact')
            w.same(report['cpu_full_anchor']['reference'],w.FULL_CPU_ANCHORS[rank],'Original full CPU anchor')
        reports[role]=report
    guard()
    import torch
    previous=torch.get_num_threads()
    try:
        with torch.inference_mode():
            weights,proof=w.weights(torch,plan,guard)
            for report in reports.values():
                w.same(report['weights']['matrices'],proof['matrices'],'Same official matrix intervals')
            observed={role:{row['name']:restore(torch,row,root/role,guard)for row in report['captures']}
                      for role,report in reports.items()}
            for field in ('input','output'):
                w.same(w.metadata(torch,observed['cpu-rank0'][field]),w.metadata(torch,observed['cpu-rank1'][field]),
                       'Actual complete replicated CPU '+field)
            native=observed['native'];cpu=[observed['cpu-rank0'],observed['cpu-rank1']]
            torch.set_num_threads(2);guard()
            actual_native=native_math(torch,native['input'],weights)
            native_control=exact_controls(torch,actual_native,native,'Native1231')
            torch.set_num_threads(1);guard()
            actual_cpu=sdk_math(torch,cpu[0]['input'],weights)
            cpu_controls=[exact_controls(torch,actual_cpu[i],cpu[i],'SDK2048/rank'+str(i))for i in (0,1)]
            # No mechanism contrast is evaluated until all 15 complete controls pass.
            guard();x=cpu[0]['input'];full=native_math(torch,x,weights)
            product=torch.cat([cpu[0]['down_input'],cpu[1]['down_input']],dim=-1).contiguous()
            w.same_bytes(w.raw(torch,product),w.raw(torch,torch.cat([a['down_input']for a in actual_cpu],dim=-1)),
                   'Intervention reuses actual observed SDK front-half product')
            w.same(list(product.shape),[1,w.BUCKET,6144],'Whole official down input, no row-only replay')
            w.same(list(product.stride()),list(full['down_input'].stride()),'Common down input physical layout')
            changed_down=torch.nn.functional.linear(product,weights['down'])
            front={field:compare_scopes(torch,torch.cat([a[field]for a in actual_cpu],dim=-1).contiguous(),full[field])
                   for field in ('gate','up','down_input')}
            output=actual_cpu[0]['output'];relations=dict(S_vs_F=compare_scopes(torch,output,full['output']),
                D_vs_F=compare_scopes(torch,changed_down,full['output']),D_vs_S=compare_scopes(torch,changed_down,output))
            boundary_differences={}
            for field in w.FIELDS:
                # Compare observed tensors only; the native prompt has no padding.
                complete=cpu[0][field] if field in ('input','output') else torch.cat([a[field]for a in cpu],dim=-1)
                boundary_differences[field]=dict(
                    real_tokens=measure(torch,native[field],complete[:,:w.LENGTH]),
                    prediction=measure(torch,native[field][:,w.LENGTH-1:w.LENGTH],complete[:,w.LENGTH-1:w.LENGTH]))
            result=dict(schema='minicpm5_frame1_mlp_replay_v2',complete=True,frame=w.FRAME,captures=capture_refs,
                        controls=dict(native=native_control,cpu=cpu_controls,all15_fullshape_exact=True),
                        fixed_input=w.metadata(torch,x),weight_proof=proof,front_half=front,contrasts=relations,
                        observed_native_vs_cpu_input=boundary_differences['input']['real_tokens'],
                        observed_native_vs_cpu_boundaries=boundary_differences,
                        classification=classify(front['down_input']['full']['raw_exact'],relations['S_vs_F']['full']['raw_exact'],
                                                relations['D_vs_F']['full']['raw_exact']),
                        original_remaining_gates=plan['remaining_gates'],
                        actual_gloo_reexecuted_in_replay=False,native_interpreter_reused_in_replay=False,
                        native_own1231_control_threads=2,fixed_input2048_contrast_threads=1,
                        whole_native_input_equality_previously_known=False,unique_final_KL_cause_proven=False,
                        full_model_qualified=False,new_speedup_claim=False)
            guard();w.once(root/'replay.json',result);return result
    finally:torch.set_num_threads(previous)
