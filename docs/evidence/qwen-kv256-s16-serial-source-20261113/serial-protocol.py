"""Exact extra two-graph assembly pair; no model or runtime actions on import."""
import json
from pathlib import Path

import qwen35_kv256_s16_pair_leaf_v1 as leaf
from qwen35_predicated_slot_protocol_v1 import RESOURCES, digest_file
from qwen35_kv256_alias_diagnostic_bytes_v1 import require
from qwen35_kv256_alias_diagnostic_serial_v1.protocol import REFERENCE_EVIDENCE

LAYOUTS=(('kv256_s16_pair',),)
FLAGS=leaf.FLAGS
GRAPHS=leaf.GRAPHS
PROFILES=leaf.PROFILES
RECEIPT_FIELDS={'name','artifact','origin','bucket','alias','hlo','hlo_proof'}


def workload():return leaf.workload()


def counts():
    per=dict(graphs=2,correctness_calls=6,total_runtime_calls=6)
    return dict(per_layout=per,total=dict(per))


def expected_calls():
    for graph in GRAPHS:
        for profile in PROFILES:yield dict(graph=graph['name'],profile=profile,slots=[-1]*4)


def verify_sources():
    manifest=json.loads((Path(__file__).parent/'primitive_manifest.json').read_text())
    pins={row['path']:row['sha256']for row in manifest['source_files']}
    return leaf.sources(pins)


def receipt(graph):
    return {key:graph[key]for key in RECEIPT_FIELDS}


def validate_graph_files(info,work):
    """Bounded full refs also apply before an HLO proof/receipt exists."""
    g=next((g for g in GRAPHS if g['name']==info['name']),None)
    require(g is not None and info['bucket']==256 and info['alias']==1 and info['origin']=='fresh_control',
            'Exact fresh matched graph identity')
    require(Path(work).is_absolute() and '..'not in Path(work).parts,'Absolute canonical pair namespace')
    directory=Path(work)/'artifacts'/'graphs'/info['name']
    for key,filename,cap in (('artifact','model.pt',leaf.FRESH_ARTIFACT_CAP),('hlo','model/graph.hlo',16*2**20)):
        ref=info[key]
        require(type(ref)is dict and set(ref)=={'path','size_bytes','sha256'}
                and ref['path']==str(directory/filename) and type(ref['size_bytes'])is int and 0<ref['size_bytes']<=cap
                and type(ref['sha256'])is str and len(ref['sha256'])==64 and all(c in '0123456789abcdef'for c in ref['sha256']),
                'Exact complete bounded '+key+' reference')


def validate_receipt(info,name,work):
    """Exact full references only; stage separately rereads stable artifact/HLO."""
    require(name==LAYOUTS[0][0] and type(info)is dict and set(info)==RECEIPT_FIELDS,'Exact pair artifact and HLO receipt')
    validate_graph_files(info,work)
    proof=info['hlo_proof']
    require(type(proof)is dict and proof.get('schema')=='qwen35_kv256_s16_pair_hlo_proof_v1'
            and proof.get('arm')==info['name'] and proof.get('hlo_sha256')==info['hlo']['sha256']
            and proof.get('hlo_bytes')==info['hlo']['size_bytes'] and proof.get('state_alias_output')==1
            and proof.get('concat_dtype')==('bf16'if info['name']=='bf16_cat'else's16')
            and proof.get('numeric_converts_in_assembly')==0,'Exact complete HLO proof binding')
    return info


def validate_report(report,name,artifacts):
    require(name==LAYOUTS[0][0],'Only the extra matched pair layout')
    result=leaf.validate_report(report)
    require(report['source_sha256']==verify_sources(),'Exact separately pinned leaf sources')
    require(len(artifacts)==2 and artifacts==[receipt(g)for g in report['graphs']],
            'Both complete model/HLO/proof receipts required')
    for info in artifacts:validate_receipt(info,name,Path(report['output_root']).parent)
    return dict(result,layout=name,passed=True)


def validate_prefix(report,name,artifacts):
    """Retain a graph saved before its HLO gate/callback, without promoting it.

    This leaf saves g after obtaining HLO, before computing its proof or writing
    the receipt. Exactly the final graph may therefore lack a receipt, with no
    calls for that graph. A failed HLO check remains interrupted evidence.
    """
    require(name==LAYOUTS[0][0] and report['schema']=='qwen35_kv256_s16_pair_result_v1'
            and report['workload']==workload() and report['source_sha256']==verify_sources(),
            'Exact interrupted pair envelope')
    require(type(report['complete'])is bool and all(report[k]is False for k in
            ('full_geometry_qualified','full_model_qualified','new_speedup_claim')),'No partial promotion')
    order=[g['name']for g in GRAPHS];graphs=report['graphs']
    require([g['name']for g in graphs]==order[:len(graphs)] and [a['name']for a in artifacts]==order[:len(artifacts)]
            and 0<=len(graphs)-len(artifacts)<=1 and len(graphs)<=2,'Exact observed graph/receipt prefix')
    work=Path(report['output_root']).parent
    require(report['output_root']==str(work/'artifacts'),'Exact interrupted capture namespace')
    for graph,expected in zip(graphs,GRAPHS):
        require(all(graph.get(k)==v for k,v in expected.items())
                and graph['physical_output_order']==['old','new'] and type(graph['calls'])is int
                and 0<=graph['calls']<=3,'Exact partial graph identity and call count')
        validate_graph_files(graph,work)
    for graph,info in zip(graphs,artifacts):
        require(receipt(graph)==info,'Reported graph matches its complete receipt')
        validate_receipt(info,name,Path(report['output_root']).parent)
    if len(graphs)>len(artifacts):
        pending=graphs[-1]
        require(pending['calls']==0 and not any(r['graph']==pending['name']for r in report['calls']),
                'No device call before final graph HLO gate and receipt')
    calls=report['calls'];wanted=list(expected_calls())
    require(len(calls)<=6 and all(all(row[k]==v for k,v in expected.items())for row,expected in zip(calls,wanted)),
            'Exact six-call ordered prefix')
    started,returned=report['started_calls'],report['returned_calls']
    require(type(started)is int and type(returned)is int and 0<=returned<=started<=len(calls)<=6
            and started-returned<=1 and len(calls)-returned<=1,'Exact interrupted invocation boundary')
    complete=[row for row in calls if 'all_expected_bytes_equal'in row]
    require(len(complete)<=returned and complete==calls[:len(complete)] and len(calls)-len(complete)<=1,
            'Only final call may lack complete proofs')
    require(sum(g['calls']for g in graphs)==returned,'Exact graph returned-call accounting')
    require(all(row['graph']in {a['name']for a in artifacts} for row in calls),
            'Every saved call follows a complete artifact receipt')
    require(all(type(row.get('call_returned'))is bool and row['call_returned']is(index<returned)
                for index,row in enumerate(calls)), 'Saved per-call return flags match counters')
    require(all(g['calls']==sum(row['graph']==g['name']and row['call_returned']for row in calls)for g in graphs),
            'Each graph count matches its own returned calls')
    if report['complete']:validate_report(report,name,artifacts)
    return dict(calls=len(complete),graphs=len(graphs),returned_calls=returned,
        retained_mismatch_calls=sum(not row['all_expected_bytes_equal']for row in complete),
        incomplete_proofs=len(calls)-len(complete),graphs_without_receipt=len(graphs)-len(artifacts),
        last_saved_stage=('graph_hlo_captured_before_receipt'if len(graphs)>len(artifacts)else
                          'call_started_or_proof_capture_pending'if len(calls)>len(complete)else'complete_call_prefix'),
        pending_hlo_proof=bool(len(graphs)>len(artifacts)and 'hlo_proof'not in graphs[-1]),
        saved_failure=report.get('error'),
        full_geometry_qualified=False)
