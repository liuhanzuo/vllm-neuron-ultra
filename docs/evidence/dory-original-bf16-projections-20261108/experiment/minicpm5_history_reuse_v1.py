"""Read-only authentication of complete frozen v2 device and v3 CPU evidence."""
from copy import deepcopy
from pathlib import Path
import minicpm5_matched32_protocol_v3 as v3
import minicpm5_matched32_reuse_v3 as v2
import minicpm5_history_contract_v1 as c

PRIOR=v3.DEST

def inputs():
    plan=v3.read(PRIOR/'plan.json');sources=deepcopy(plan['source_sha256']);payloads=deepcopy(plan['payloads'])
    def metadata(path,digest=None):
        path=Path(path);actual=v3.text_sha(path)
        assert digest is None or actual==digest
        assert str(path) not in sources or sources[str(path)]==actual
        sources[str(path)]=actual;return v3.read(path)
    def payload(path,digest,identity):
        assert v3.identity(path)==identity
        row=dict(sha256=digest,identity=identity)
        assert str(path) not in payloads or payloads[str(path)]==row
        payloads[str(path)]=row
    frozen=v2.inputs()
    for target,adds in ((sources,frozen['source_sha256']),(payloads,frozen['payloads'])):
        for name,value in adds.items():assert name not in target or target[name]==value;target[name]=value
    metadata(PRIOR/'plan.json');status=metadata(PRIOR/'status.json')
    assert status['stage']=='complete_diagnostic_only' and not status['full_model_qualified']
    for phase in v3.PHASES:
        receipt=metadata(PRIOR/(phase+'-result.json'));outcome=metadata(PRIOR/(phase+'-outcome.json'))
        assert receipt['passed'] and receipt['plan_sha256']==v3.text_sha(PRIOR/'plan.json') and receipt['phase']==phase
        assert outcome['returncode']==0 and not outcome['oom_kill_confirmed'] and not outcome['timeout_confirmed']
        assert all(outcome[k] is None for k in ('error','cleanup_error','journal_error'))
        for name,digest in receipt['outputs'].items():metadata(name,digest)
        monitor=metadata(PRIOR/(phase+'-storage-monitor.json'))
        assert monitor['passed'] is True
    control=metadata(PRIOR/'control-result.json');v3.check_control(control)
    assert v3.text_sha(PRIOR/'control-result.json')==frozen['receipt']['control_sha256']
    cpu=[];states=0
    for rank in (0,1):
        report=metadata(PRIOR/f'cpu-rank{rank}/result.json');v3.check_cpu(report);cpu.append(report)
        environment=metadata(PRIOR/f'cpu-rank{rank}/sdk-environment-transition.json')
        assert report['sdk_environment_transition']==environment
        for spec,ref in zip(c.calls(),report['reports']):
            row=metadata(ref['path'],ref['sha256']);assert row['rank']==rank and row['call']==spec
            assert row['stage_names']==c.stages() and list(row['stage_raw_sha256'])==c.stages()
            assert row['all84_imported_states_raw_exact'] and row['all84_input_states_unchanged']
            assert row['all_input_values_roundtrip_exact'] and row['all_sdk_placeholder_values_preserved'] and row['graph_input_count']==7
            payload(row['payload'],row['payload_sha256'],row['payload_identity'])
            manifest=metadata(row['after']['path'],row['after']['sha256']);c.check_manifest(manifest,rank)
            for item in manifest['rows']:
                payload(item['file'],item['payload_sha256'],item['identity']);states+=1
    assert states==2352
    metadata(c.DESIGN)
    return dict(source_sha256=sources,payloads=payloads,legacy_source_aliases=deepcopy(plan['legacy_source_aliases']),
      reused_control=frozen['receipt'],reused_cpu=dict(directory=str(PRIOR),plan_sha256=sources[str(PRIOR/'plan.json')],
      rank_results=[dict(path=str(PRIOR/f'cpu-rank{rank}/result.json'),sha256=sources[str(PRIOR/f'cpu-rank{rank}/result.json')]) for rank in (0,1)],
      calls=14,frames=32,stage_count=213,poststate_tensors=2352))
