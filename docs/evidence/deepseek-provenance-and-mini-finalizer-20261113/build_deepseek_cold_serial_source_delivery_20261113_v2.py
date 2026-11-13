"""Bind the new source delta and test text; no runtime/current/native inputs."""
import ast
import hashlib
import json
from pathlib import Path
Q=Path(__file__).resolve().parent
OLD_SHA='e6b2bf70a438094a7d2e37df234510d558c84778c19a4a7c11bb78f123501dcc'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def publish(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')

def main():
    old=Q/'deepseek-cold-serial-source-delivery-20261113-v1.json'
    assert digest(old)==OLD_SHA;previous=json.loads(old.read_text())
    files=[*sorted((Q/'deepseek_cold_serial_v2').glob('*.py')),Q/'render_deepseek_cold_serial_v2.py',
        Q/'test_deepseek_cold_serial_v2.py',Q/'run_deepseek_cold_serial_tests_audited_20261113_v2.py',Path(__file__).resolve()]
    source={}
    for path in files:
        raw=path.read_bytes();ast.parse(raw,filename=str(path))
        source[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    delta_path=Q/'deepseek-cold-serial-source-delta-20261113-v2.json'
    publish(delta_path,dict(schema='deepseek_cold_serial_source_delta_v2',source_only=True,new_sources=source,
        inherited_delivery=dict(path=str(old),sha256=OLD_SHA),inherited_static_closure=previous['static_source_closure'],
        inherited_source_acceptance=False,full_inherited_closure_rehashed=False,actual_runtime_source_closure=False))
    runs=[]
    for suffix,chunk,code,count in [('first','b66cb0',1,18),('second','e61370',0,18),('final','377d99',0,20),('fourth','f24cf7',0,21)]:
        path=Q/('deepseek-cold-serial-source-tests-20261113-v2-'+suffix+'.txt')
        runs.append(dict(path=str(path),sha256=digest(path),actual_tool_chunk=chunk,exit_code=code,test_count=count,session_id=None))
    receipt_path=Q/'deepseek-cold-serial-tests-actual-exit-20261113-v2.json'
    publish(receipt_path,dict(schema='deepseek_cold_serial_successor_tests_actual_tool_exit_v2',source_only=True,
        runs=runs,final_passed=True,v1_test_suite_rerun=False,actual_native_execution=False,
        actual_weight_or_runtime_json_contents_read=False,
        first_failure='Fixture direct latest import attempted historical JSON open; audit blocked before content read. Entire research/latest boundary then mocked without weakening audit.'))
    doc=Q/'deepseek-cold-serial-source-delivery-20261113-v2.md'
    target=Q/'deepseek-cold-serial-source-delivery-20261113-v2.json'
    value=dict(schema='deepseek_cold_serial_source_delivery_v2',source_only=True,runtime_admitted=False,
        full_seven_phase_execution_available=False,fail_before_bindings_lock_history_services=True,
        full_model_qualified=False,new_speedup_claim=False,
        v1_root_rejection=dict(path=str(Q/'deepseek-cold-serial-root-source-review-20261113-v1.json'),
            sha256='3884fec12072d31da7ab64e5491270d0d5ec3a6878028cda3398daf9d144bb62',source_accepted=False),
        addressed_source_gaps=['Actual complete DeepSeek running=True plus concrete active_serial; full inherited history and real tool/owner/catalog bindings.',
            'Explicit complete original checkpoint/config/index/tokenizer manifest; all unique physical bytes authenticated under original retained lease; exact immutable symlink/hardlink sharing and complete parent proof.'],
        source_delta=dict(path=str(delta_path),sha256=digest(delta_path),new_python_files=len(source)),
        tests=dict(path=str(receipt_path),sha256=digest(receipt_path),final_count=21,actual_tool_chunk='f24cf7',exit_code=0),
        document=dict(path=str(doc),sha256=digest(doc)),
        raw_two_arm_bytes=47150344192,original_logits=126,original_KV_tensors=9072,original_states=144,
        frozen_service_phase_bytes_retained=True,original_sources_modified=False,
        running_idle_publication_implemented=False,root_successor_review_pending=True,
        unresolved=['No actual original manifest/source authority/capacity/current runtime binding selected or verified.',
            'Actual loaded-weight / TP-layout / graph-weight-ABI semantic proof producer remains missing; pre-cost gate always refuses.',
            'No native/systemd/compiler/scientific raw comparison or model qualification/performance validation was performed.',
            'Actual external outer exit and complete current catalog/full History reconciliation remain independent external duties.'])
    publish(target,value)
    print(json.dumps(dict(path=str(target),sha256=digest(target),new_python_files=len(source),
        final_tests=21,actual_tool_chunk='f24cf7',exit_code=0,runtime_admitted=False),sort_keys=True))

if __name__=='__main__':main()
