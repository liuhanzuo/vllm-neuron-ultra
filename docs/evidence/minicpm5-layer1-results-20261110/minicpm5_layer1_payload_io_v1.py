"""Bounded payload reads for a separately admitted layer1 audit.

No launcher or admission is provided here. The caller must own the serial lock,
authenticate all source/JSON inputs and prove collection processes drained. Its
guard must recheck those conditions at each read boundary. Tests use synthetic
temporary files only.
"""
import hashlib
import os
from pathlib import Path
import stat
from research import minicpm5_layer1_raw_verifier_v1 as raw

def require(ok,message):
    if not ok:raise ValueError(message)

def file_identity(path,info):
    return dict(realpath=str(path),device=info.st_dev,inode=info.st_ino,size=info.st_size,
                mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)

def safe_path(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path
        and not any(p.is_symlink()for p in (path,*path.parents)),'Canonical payload path')
    return path

def checked_info(path,info,wanted):
    require(stat.S_ISREG(info.st_mode) and info.st_uid==1000 and info.st_nlink==1
        and 0<info.st_size<=raw.MAX_PAYLOAD<16*2**20,'Owned bounded regular payload')
    require(file_identity(path,info)==wanted,'Exact captured payload identity')

def read_owned_payload(descriptor,expected_path,guard):
    """Never bypass admission; guard exceptions propagate before any byte read."""
    require(callable(guard),'Explicit external audit admission guard')
    guard()
    expected=safe_path(expected_path)
    require(descriptor['path']==str(expected),'Only admitted call observation path')
    wanted=descriptor['identity'];before=expected.lstat()
    checked_info(expected,before,wanted)
    require(type(descriptor['payload_bytes'])is int and descriptor['payload_bytes']==before.st_size
        and raw.hash_string(descriptor['payload_sha256']),'Exact payload extent and digest')
    fd=os.open(expected,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        checked_info(expected,os.fstat(fd),wanted)
        parts=[];count=0;digest=hashlib.sha256()
        while True:
            guard()
            block=os.read(fd,min(65536,before.st_size-count+1))
            if not block:break
            count+=len(block)
            require(count<=before.st_size,'Payload grew during read')
            parts.append(block);digest.update(block)
        checked_info(expected,os.fstat(fd),wanted)
        require(count==before.st_size and digest.hexdigest()==descriptor['payload_sha256'],
                'Complete original observation bytes')
    finally:os.close(fd)
    guard();checked_info(expected,expected.lstat(),wanted)
    payload=b''.join(parts)
    require(len(payload)==before.st_size,'Complete bounded payload returned')
    return payload

def audit_trajectory(plan,rank_results,read_json,guard):
    """Use only after separate admission; retain pending claims until return.

    read_json(path, sha256) must authenticate the exact JSON reference. Scientific
    rank validation uses the unchanged original gates plus observation metadata.
    No model, native SDK, tensor loader, compression writer or attention runs.
    """
    require(callable(guard) and callable(read_json),'External admission and authenticated JSON reader')
    guard()
    from research import minicpm5_layer1_runtime_results_v1 as results
    # These validators authenticate exact JSON hashes and preserve every old
    # scientific gate; the outer admission must bind their full input closure.
    controls=results.validate_rank_pair(rank_results,plan)
    guard()
    refs=plan['layer1_observation']['original_calls']
    require(len(refs)==84,'Complete original call catalog')
    trajectory=raw.Trajectory()
    root=safe_path(plan['output_root'])
    for rank,result in enumerate(rank_results):
        for ai,entry in enumerate(result['arm_reports']):
            guard();arm=raw.ARMS[ai]
            expected=root/('cpu-rank'+str(rank))/arm/'result.json'
            require(entry['report']==str(expected),'Exact collected arm')
            report=read_json(entry['report'],entry['sha256'])
            for ci,item in enumerate(report['reports']):
                guard();folder=expected.parent/('call-%02d'%ci)
                require(item['report']==str(folder/'result.json'),'Exact collected call')
                call=read_json(item['report'],item['sha256'])
                original_ref=refs[rank*42+ai*14+ci]
                require([original_ref[k]for k in ('rank','arm','call')]==[rank,arm,ci],
                        'Exact frozen original call ordering')
                original_ref={k:original_ref[k]for k in ('path','sha256')}
                original=read_json(original_ref['path'],original_ref['sha256'])
                payload=read_owned_payload(call['layer1_observation'],folder/'layer1-observation.bin',guard)
                trajectory.accept(call,original,original_ref,payload)
    guard()
    proof=trajectory.finish()
    proof['metadata_controls']=controls
    proof['metadata_controls']['layer1_raw_reconstruction_pending']=False
    proof['raw_payloads_read']=True
    proof['requires_separate_runtime_admission']=True
    return proof
