"""Source-only bundle/plan interface. Root owns queue and canonical enrollment."""
from copy import deepcopy
from pathlib import Path
import ast
import sys
import minicpm5_history_protocol_v1 as p
import minicpm5_history_contract_v1 as c


def merge(destination,additions):
    for name,value in additions.items():
        assert name not in destination or destination[name]==value,name
        destination[name]=value


def source_paths():
    pending=[p.ROOT/name for name in (*p.source_names(),'prepare_minicpm5_history_v1.py')];found=set()
    if p.CONTROLLER_SCRIPT.exists():pending.append(p.CONTROLLER_SCRIPT)
    while pending:
        path=p.no_symlink(pending.pop())
        if path in found:continue
        found.add(path);tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=p.ROOT/(name.split('.')[0]+'.py')
                if local.exists():pending.append(local)
    return sorted(found)


def inputs():
    from minicpm5_history_reuse_v1 import inputs as reused
    value=reused();sources=value['source_sha256']
    for path in source_paths():merge(sources,{str(path):p.text_sha(path)})
    merge(sources,{str(c.DESIGN):p.text_sha(c.DESIGN)})
    for name,digest in sources.items():assert p.text_sha(name)==digest,name
    for name,row in value['payloads'].items():assert p.identity(name)==row['identity']
    assert not any(x in sys.modules for x in ('torch','numpy','transformers','neuronx_distributed'))
    return value


def build_plan(tail,review_path):
    assert not p.DEST.exists() and not p.DATA.exists()
    assert p.CONTROLLER_SCRIPT.is_file()
    for name in ('continuation-state.json','roadmap.json'):assert p.read(p.ROOT/name)['current_authoritative_queue_tail']==tail
    old=p.read(p.ROOT/tail['plan']);owner=p.read(p.ROOT/tail['directory']/'queue-identity.json');assert owner==tail['queue']
    predecessors=deepcopy(old['predecessors'])+[owner]
    assert len(predecessors)==len({(x['pid'],str(x['start_ticks'])) for x in predecessors})
    review_path=p.no_symlink(review_path);review=p.read(review_path)
    assert all(review.get(k) is True for k in ('passed','source_only','independent_review','suitable_for_enrollment_as_independent_review','original_gates_preserved'))
    assert not review.get('blockers',[])
    frozen=inputs()
    assert all(review['source_sha256'].get(str(path))==frozen['source_sha256'][str(path)] for path in source_paths())
    for path in (review_path,p.ROOT/tail['plan'],p.ROOT/tail['directory']/'queue-identity.json'):
        merge(frozen['source_sha256'],{str(path):p.text_sha(path)})
    return dict(**p.contract(),**frozen,prior_tail=deepcopy(tail),predecessors=predecessors,
      review=dict(path=str(review_path),sha256=p.text_sha(review_path),passed=True,source_only=True),
      original_report_sha256=p.text_sha(p.OLD/'real-logits-validation.json'),
      source_only_preparation=True,large_payloads_read_during_preparation=False,native_imports_during_preparation=False,
      serial_controller_required=True,queue_startup_ack_required=True,enrollment_barrier_required=True,
      serial_admission_contract='Root owns exact PID/start/script/cgroup queue/children; serial publication lock, fresh namespaces, startup ACK and consumed barrier; authenticate all payloads at exclusive admission; supervise storage every5s through owned drain and launcher reap, final sample before success.',
      scope_description='Both CPU arms14calls perrank; device Hd/Hc/Hd restoration each14calls,70logicalcalls160frames,original7inputs/all168states/213CPUhooks. Hc seedHd0 then allpreviousCPUposts,including resetcall7 withoutzero. Anytriplet failure invalidates allintervention conclusions.',
      preserved_followup_gates=p.read(c.DESIGN)['preserved_followup_gates'])
