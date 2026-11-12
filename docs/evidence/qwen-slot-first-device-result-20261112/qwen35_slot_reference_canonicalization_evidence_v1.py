"""Pure stdlib reconstruction of the saved r1 reference failure; no native IO."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/mnt/nvme/neuron-work/qwen35-predicated-slot-r1-v3-20261112')
WIDTH = 12288
SHAPE = (4,3072,4)
SPECIALS = (0,0x8000,0x7f80,0xff80,0x7fc1,0x7fa1,1,0x8001)

def digest(raw): return hashlib.sha256(raw).hexdigest()
def reference(path):
    data=Path(path).read_bytes()
    return dict(path=str(path),sha256=digest(data),bytes=len(data))

def input_bytes(salt, *, negative_nan=False):
    """Exact raw BF16 array_for formula, little endian host-storage words.

    negative_nan is a new diagnostic profile only. It leaves the historical
    profile unchanged and replaces positions 8/9 in each row with negative
    quiet/signaling NaN words, for a separately labeled future experiment.
    """
    values=[]
    for index in range(4*WIDTH):
        slot,offset=divmod(index,WIDTH)
        value=SPECIALS[offset] if offset<8 else (index+slot*101+salt*37)%65536
        if negative_nan and offset in (8,9): value=0xffc1 if offset==8 else 0xffa1
        values.append(value)
    return struct.pack('<%dH'%len(values),*values)

def normalized_bytes(raw):
    values=struct.unpack('<%dH'%(len(raw)//2),raw)
    result=[0 if value==0x8000 else 0x7fc0 if value&0x7f80==0x7f80 and value&0x7f else value
            for value in values]
    return struct.pack('<%dH'%len(result),*result)

def evidence():
    report_path=RUN/'conv/report.json';report=json.loads(report_path.read_text())
    assert reference(report_path)['sha256']=='74645c6ee057d423654bbb88a22cbc6c097c612aa0bd4149a2e1901f85967351'
    row,=report['checks'];assert row['variant']=='reference' and row['mode']=='standalone'
    assert row['phase']=='correctness' and row['detail']==dict(profile='raw',round=0,step=0)
    assert row['slots']==[-1]*4 and report['runtime_counts']==dict(correctness=1,warmup=0,measured=0,reset=0)
    cache,updates=input_bytes(0),input_bytes(17)
    check=row['checks']['updated_output'];transformed=normalized_bytes(cache)
    assert digest(cache)==check['expected_sha256']==row['checks']['cache_input_unchanged']['actual_sha256']
    assert digest(updates)==row['checks']['updates_unchanged']['actual_sha256']
    assert digest(struct.pack('<4q',-1,-1,-1,-1))==row['checks']['slots_unchanged']['actual_sha256']
    assert all(row['checks'][name]['passed'] for name in ('cache_input_unchanged','updates_unchanged','slots_unchanged'))
    assert digest(transformed)==check['actual_sha256'] and len(transformed)==check['bytes_compared']==98304
    changed=[i for i,(left,right) in enumerate(zip(cache,transformed)) if left!=right]
    assert changed[:16]==check['first_different_bytes']
    words=struct.unpack('<49152H',cache)
    negative_nan_count=sum(value&0xff80==0xff80 and bool(value&0x7f) for value in words)
    assert negative_nan_count==0
    names=('qwen35_predicated_slot_device_leaf_v1.py','qwen35_predicated_slot_protocol_v1.py',
           'qwen35_text_port.py','probe_slot_cache_nki_read.py','slot_cache_nki_predicated.py',
           'slot_cache_nki_predicated_torch.py')
    spec_path=ROOT/'research/qwen35-predicated-slot-r1-v3-20261112-spec.json'
    spec=json.loads(spec_path.read_text());sources={name:reference(ROOT/name) for name in names}
    assert all(spec['source_sha256'][row['path']]==row['sha256'] for row in sources.values())
    ended=ROOT/'research/qwen35-predicated-slot-r1-root-actual-exit-20261112-v2.json'
    assert reference(ended)['sha256']=='6e07d96c6da69cfeb141d020e251cac8376ae007113a240d6d204db387318769'
    return dict(schema='qwen35_slot_reference_canonicalization_saved_evidence_v1',
        evidence_kind='saved hashes plus deterministic whole-byte reconstruction; no saved actual raw output',
        report=reference(report_path),log=reference(RUN/'conv/native.log'),source_spec=reference(spec_path),
        actual_root_exit=reference(ended),sources=sources,shape=list(SHAPE),dtype='bfloat16',
        slots=[-1]*4,profile='raw',initial_salt=0,update_salt=17,bytes_per_tensor=len(cache),
        expected_cache_sha256=digest(cache),updates_sha256=digest(updates),
        transformed_cache_sha256=digest(transformed),actual_reported_output_sha256=check['actual_sha256'],
        whole_sha256_match=True,first16_difference_positions_match=True,
        transformation='0x8000 -> 0x0000; every NaN -> 0x7fc0; every other word unchanged',
        derived_different_bytes=len(changed),derived_different_words=len(changed),
        negative_zero_words=sum(value==0x8000 for value in words),
        nan_words=sum(value&0x7f80==0x7f80 and bool(value&0x7f) for value in words),
        already_canonical_nan_words=sum(value==0x7fc0 for value in words),negative_nan_words=negative_nan_count,
        negative_nan_sign_behavior_determined=False,layer_attribution_determined=False,
        candidate_runtime_correctness_calls=0,reference_runtime_correctness_calls=1,
        native_code_run=False,current_processes_services_locks_inspected=False,
        qualification_matrix_completed=False,full_model_qualified=False,new_speedup_claim=False)

if __name__=='__main__':
    print(json.dumps(evidence(),sort_keys=True,separators=(',',':')))
