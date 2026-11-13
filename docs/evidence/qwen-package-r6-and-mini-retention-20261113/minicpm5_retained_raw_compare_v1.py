"""Independent full old/new saved replay comparison. Metadata only, inert import.

Only output namespace references and two explicit honest admission fields may
change. All14 scientific reports and all98 layout proofs must be byte-identical.
The full comparison JSON must match after replacing its exact14 output refs.
"""
import copy
from pathlib import Path
import minicpm5_retained_raw_transport_v1 as t

ORIGINAL_RESULT=dict(path=str(t.ROOT/'minicpm5-history-analysis-recovery-v2/compare-result.json'),
    sha256='83d3329b4c0cce71cb30dfe95b6942a750fad65081cc4d6d93dd7f6aebeb1c11',bytes=775)


def _metadata(ref, guard):
    guard();value=t.read_receipt(ref);guard();return value


def _with_size(path, sha, guard):
    guard();ident=t.identity(path)
    t.require(0<ident['size']<=16<<20 and Path(path).suffix=='.json','Bounded complete JSON metadata')
    return dict(path=str(path),sha256=sha,bytes=ident['size'])


def compare_original_replay(storage_plan, verification_ref, replay_result_ref, output_ref_path, guard):
    """Call after actual replay worker exit and complete physical drain.

    guard owns those externally authenticated conditions; no service or History
    code executes here. Output receipt is separate from source/replay admission.
    """
    t.validate_plan(storage_plan);t._verification(storage_plan,verification_ref)
    new_root=Path(replay_result_ref['path']).parent
    t.require(new_root.parent==t.ROOT and new_root.name.startswith('minicpm5-history-retained-replay-')
        and Path(replay_result_ref['path']).name=='compare-result.json','Exact independent replay namespace')
    result=_compare(storage_plan,verification_ref,ORIGINAL_RESULT,replay_result_ref,guard)
    guard();return t._write_once(output_ref_path,result)


def _compare(storage_plan, verification_ref, original_result_ref, replay_result_ref, guard):
    old=_metadata(original_result_ref,guard);new=_metadata(replay_result_ref,guard)
    old_root,new_root=(Path(r['path']).parent for r in (original_result_ref,replay_result_ref))
    t.require(old_root!=new_root,'Independent original/replay namespaces')
    t.require(set(old)==set(new) and {k:v for k,v in old.items()if k!='outputs'}==
        {k:v for k,v in new.items()if k!='outputs'},'All terminal fields and original plan hash preserved')
    names=('payload-admission.json','saved-layout-admission.json','comparison.json')
    t.require(set(old['outputs'])=={str(old_root/n)for n in names}
        and set(new['outputs'])=={str(new_root/n)for n in names},'Exact complete original/replay output list')
    old_refs={n:_with_size(old_root/n,old['outputs'][str(old_root/n)],guard)for n in names}
    new_refs={n:_with_size(new_root/n,new['outputs'][str(new_root/n)],guard)for n in names}
    before={n:_metadata(old_refs[n],guard)for n in names};after={n:_metadata(new_refs[n],guard)for n in names}
    guard();old_plan=t.REFERENCES['recovery_plan']['sha256']
    t.require(old['plan_sha256']==new['plan_sha256']==old_plan,'Original plan digest is unchanged')
    new_plan_ref=_with_size(new_root/'plan.json',old_plan,guard);_metadata(new_plan_ref,guard)
    old_admit,new_admit=before['payload-admission.json'],after['payload-admission.json']
    allowed={'original_capture_identities_preserved','physical_resolutions'}
    t.require(set(new_admit)==set(old_admit)|allowed and new_admit['original_capture_identities_preserved']is True
        and {k:v for k,v in new_admit.items()if k not in allowed}==old_admit,
        'Full11859 original admission map and every original admission field preserved')
    t.require(old_admit['payloads']==11859 and len(old_admit['sha256'])==11859,'No reduced logical admission')
    expected={row['original']['realpath']:dict(logical_path=row['original']['realpath'],
        capture_identity=row['original'],storage_identity=row['keeper'],sha256=row['sha256'],
        relocated=True,original_identity_is_current_stat=False)for row in storage_plan['entries']}
    t.require(new_admit['physical_resolutions']==expected,'Exactly the complete finite honest storage translation')
    layout=before['saved-layout-admission.json']
    t.require(layout==after['saved-layout-admission.json'] and layout['outputs']==len(layout['records'])==98
        and old_refs['saved-layout-admission.json']['sha256']==new_refs['saved-layout-admission.json']['sha256'],
        'All98 complete producer/serialized layout proofs byte-identical')
    old_report,new_report=before['comparison.json'],after['comparison.json']
    t.require(len(old_report['reports'])==len(new_report['reports'])==14,'All14 call reports required')
    translations=[];frames=caches=hooks=predictions=effects=0
    for index,(a,b)in enumerate(zip(old_report['reports'],new_report['reports'])):
        relative=Path('analysis')/('call-%02d.json'%index)
        t.require(a['path']==str(old_root/relative)and b['path']==str(new_root/relative), 'Exact same call in new namespace')
        ar=_with_size(a['path'],a['sha256'],guard);br=_with_size(b['path'],b['sha256'],guard)
        av,bv=_metadata(ar,guard),_metadata(br,guard)
        t.require(a['sha256']==b['sha256'] and ar['bytes']==br['bytes'] and av==bv,
            'Every complete scientific call report must be byte-identical')
        t.require(av['complete']is True and len(av['caches'])==672 and len(av['cpu_rank_hooks'])==426,
            'Complete cache and rank-hook scope for each original call')
        frames+=len(av['frames']);caches+=len(av['caches']);hooks+=len(av['cpu_rank_hooks'])
        effects+=sum(len(frame['effects'])for frame in av['frames'])
        predictions+=sum(row['prediction_vectors']for row in av['cpu_rank_hooks'])
        translations.append(dict(original=ar,replay=br))
    normalized=copy.deepcopy(new_report);normalized['reports']=copy.deepcopy(old_report['reports'])
    t.require(t.canonical(normalized)==t.canonical(old_report),'Every full32 summary, metric, failure and scientific field preserved')
    t.require((frames,caches,hooks,predictions,effects)==(32,9408,5964,13632,256), 'Entire original scientific scope')
    t.require(len(old_report['hf_summaries'])==7 and all(len(row['rows'])==32 for row in old_report['hf_summaries']),
        'All seven original HF32 arm/rank summaries')
    t.require(old_report['full_model_qualified']is old_report['new_speedup_claim']is False
        and old_report['original_failure_preserved']is True,'Original failures and nonqualification remain')
    return dict(schema='minicpm5_retained_raw_complete_replay_equality_v1',
        storage_plan_sha256=t.digest(t.canonical(storage_plan)),verification=verification_ref,
        original_result=original_result_ref,replay_result=replay_result_ref,
        old_outputs=old_refs,new_outputs=new_refs,exact_namespace_translations=translations,
        complete=True,full11859_logical_inputs_preserved=True,
        full32_replay_and_all_scientific_outputs_equal=True,all_original_failures_preserved=True,
        layouts98_byte_identical=True,scientific_reports14_byte_identical=True,
        full_frames32=frames,cache_pairs9408=caches,cpu_hook_pairs5964=hooks,stage_vectors13632=predictions,
        allowed_changes=['exact14 report output namespaces','two explicit admission storage-provenance fields','new execution metadata outside original scientific outputs'],
        full_model_qualified=False,new_speedup_claim=False)
