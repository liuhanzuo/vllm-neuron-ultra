"""Full original32 two-history crossover contract; stdlib/source metadata only."""
from copy import deepcopy
from pathlib import Path
import minicpm5_matched32_protocol_v3 as old

ROOT=old.ROOT
NAMESPACE='minicpm5-history-crossover-v1'
ARMS=('C_Hd_control','C_Hc_history_generation','D_Hd_control','D_Hc_intervention','D_Hd_restore_control')
CPU_ARMS=ARMS[:2]
DEVICE_ARMS=ARMS[2:]
PHASES=('admit','reuse_control','cpu_control','cpu_history','control_admission','device_crossover','compare')
CPU_PHASES=('cpu_control','cpu_history')
CRITERIA=deepcopy(old.CRITERIA)
CACHE_BYTES=old.CACHE_BYTES
SNAPSHOT_BYTES=168*CACHE_BYTES
RAW_POSTSTATE_BYTES=70*SNAPSHOT_BYTES
# Raw state files are uncompressed. 2 GiB covers output/tap serialization, all
# metadata and bounded transient files. Histories refer to seed/prior posts.
EXTRA_BYTES=2*1024**3
NVME_FLOOR=5*1024**3
ROOT_FLOOR=28*1024**3
START_NVME=RAW_POSTSTATE_BYTES+EXTRA_BYTES+NVME_FLOOR
DESIGN=ROOT/'research/minicpm5-matched32-next-action-design-20261106-v1.json'

def calls():return old.calls()
def layout():return old.layout()
def stages():return old.stages()

def scope():
    return dict(arms=list(ARMS),calls_per_arm=14,frames_per_arm=32,total_logical_calls=70,total_logical_frames=160,
      cpu_calls_per_rank=28,cpu_prediction_vectors_per_rank=13632,cpu_hooks_per_call=213,
      all_rank_cache_tensors=168,cache_shape=[4,1,4224,128],graph_input_count=7,
      total_postcache_tensors=11760,original_criteria=CRITERIA,original_teacher_forced_ids=True,
      original_groups=[[0,7,2,5],[4,1,6,3]],slot_order=[3,1,0,2],new_compilation=False,
      raw_poststate_upper_bound_bytes=RAW_POSTSTATE_BYTES,extra_budget_bytes=EXTRA_BYTES,
      nvme_floor_bytes=NVME_FLOOR,root_floor_bytes=ROOT_FLOOR,start_nvme_free_bytes=START_NVME,
      compression_assumed=False,duplicate_prestate_payloads=False,full_model_qualified=False,new_speedup_claim=False)

def logical_state(index):
    assert type(index) is int and 0<=index<14
    return dict(group=index//7,reset_before=index in (0,7),kv_cache_populated_before=index not in (0,7),
        kv_cache_populated_after=True,reset_zeroes_kv=False,reconstruct_stage_before=index in (0,4,7,11))

def history_origin(arm,index):
    assert arm in ARMS and type(index) is int and 0<=index<14
    if arm in ('C_Hd_control','D_Hd_control','D_Hd_restore_control'):
        return dict(kind='frozen_v2_device_before',call=index)
    if index==0:return dict(kind='frozen_v2_device_before',call=0)
    return dict(kind='cpu_history_previous_post',arm='C_Hc_history_generation',call=index-1)

def check_manifest(manifest,rank=None):
    expected=[x for x in layout() if rank is None or x['rank']==rank]
    assert manifest['complete'] is True and len(manifest['rows'])==len(expected)
    for row,want in zip(manifest['rows'],expected):
        assert {k:row[k] for k in want}==want
        assert row['raw_bytes']==CACHE_BYTES and row['nonfinite_elements']==0
        assert all(isinstance(row[k],str) and len(row[k])==64 and set(row[k])<=set('0123456789abcdef') for k in ('raw_sha256','payload_sha256'))
    assert manifest['raw_bytes']==len(expected)*CACHE_BYTES
    return True

def check_history_reference(ref,arm,index,seed,previous=None):
    assert ref['origin']==history_origin(arm,index)
    assert ref['logical_state']==logical_state(index)
    if ref['origin']['kind']=='frozen_v2_device_before':expected=seed
    else:
        assert previous is not None and previous['arm']=='C_Hc_history_generation'
        assert previous['call']==calls()[index-1] and previous['complete'] is True
        expected=previous['after']
    assert ref['manifest']==expected
    assert ref['duplicate_payload_written'] is False and ref['zero_history_substituted'] is False
    return True

def check_triplet(rows,index):
    assert len(rows)==3 and [r['arm'] for r in rows]==list(DEVICE_ARMS)
    before,candidate,restored=rows;spec=calls()[index]
    assert all(r['complete'] is True and r['call']==spec for r in rows)
    assert all(r['all168_imports_reexport_exact'] is True and r['bindings_unchanged'] is True for r in rows)
    assert all(r['input_metadata']==before['input_metadata'] and r['graph']==before['graph'] and r['bucket']==before['bucket'] for r in rows)
    assert all(r['logical_state']==logical_state(index) for r in rows)
    for control in (before,restored):
        assert control['original_output_raw_exact'] is True and control['original_postcache_raw_exact'] is True
    assert before['output_metadata']==restored['output_metadata']
    assert before['postcache_raw_sha256']==restored['postcache_raw_sha256']
    assert all(r['numeric_scoring_permitted'] is False for r in rows)
    return True

def check_terminal_device(value):
    assert value['complete'] is True and value['calls']==42 and value['logical_frames']==96
    assert len(value['triplets'])==14 and value['all14_baseline_and_restore_passed'] is True
    assert value['failed_triplet_invalidates_all_intervention_conclusions'] is True
    assert value['new_compilation'] is False and value['full_model_qualified'] is False
    return True
