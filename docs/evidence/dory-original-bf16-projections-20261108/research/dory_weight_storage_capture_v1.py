"""Dory storage capture and independent saved-byte review; no runtime admission.

Caller supplies actual Torch, inference context and a serial/resource guard.
No framework imports, checkpoint access, model construction or automatic CLI.
"""
import hashlib,json,os,resource,stat,time
from pathlib import Path
import dory_weight_storage_probe_v1 as probe
from dory_rotary_capture_runtime_v1 import Capture,once
require=probe.require
SELF=Path(__file__).resolve()
PINS={Path(probe.__file__).resolve():'c75a1bcbb532fb5e0c473df0dfcdcb0ad9b8b09e98a9925df5d36ed427e68699',
      SELF.parent/'dory_rotary_capture_runtime_v1.py':'d4a4f4d6e952f987bf47b15250c564f3d855bf801f0da7595efbec1d2f72597e'}
FLAGS=('checkpoint_downloaded','original_weights_loaded','model_constructed','model_forward_executed','peak_full_model_memory_fit_proven','native_reference_qualified','neuron_baseline_qualified','new_speedup_claim')


def stable_bytes(path,limit,guard,expected=None):
    require(callable(guard),'Actual resource guard required');guard();path=Path(path)
    require(path.is_absolute()and path.resolve()==path,'Canonical original saved file path')
    require(type(limit)is int and 0<limit<=2**20,'Bounded individual metadata or payload')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and 0<before.st_size<=limit,'Ordinary bounded single-link file')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(limit+1)
        after=os.fstat(fd);current=path.lstat()
        identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(identity(before)==identity(after)==identity(current)and len(raw)==before.st_size,'Stable complete saved file')
        if expected is not None:require(hashlib.sha256(raw).hexdigest()==expected,'Exact saved file SHA256')
        guard();return raw
    finally:os.close(fd)


def read_ref(reference,guard):
    require(set(reference)in ({'path','sha256'},{'path','sha256','bytes'}),'Explicit complete metadata reference')
    raw=stable_bytes(reference['path'],2**20,guard,reference['sha256'])
    require('bytes'not in reference or type(reference['bytes'])is int and reference['bytes']==len(raw),'Optional exact metadata byte count')
    return json.loads(raw)


def source_check():
    for path,wanted in PINS.items():require(hashlib.sha256(path.read_bytes()).hexdigest()==wanted,'Frozen original probe/capture code')
    probe.source_check()


def validate_probe(value):
    require(value['schema']=='dory_weight_storage_native_probe_v1'and value['complete']is value['passed']is True,'Actual complete native probe report')
    require(value['plan']==probe.plan()and value['all_raw_words_exact']is True,'Original full storage fixture plan')
    require(value['plain_buffer_uses_production_frombuffer_detach_operations']is value['tracked_buffer_only_supplements_lifetime_observation']is True,'Both original buffer implementations')
    require(all(value[key]is False for key in FLAGS),'No full checkpoint, model or performance claims')
    wanted={str(Path(probe.__file__).resolve()):PINS[Path(probe.__file__).resolve()],str(Path(probe.bridge.__file__).resolve()):probe.BRIDGE_SHA,str(Path(probe.bridge.loader.__file__).resolve()):probe.LOADER_SHA}
    require(value['source_sha256']==wanted,'Exact original executed source bindings')
    require(type(value['torch_version'])is str and bool(value['torch_version']),'Actual Torch version recorded')
    rows=value['cases'];require(len(rows)==6,'All actual six native cases')
    for row in rows:
        require(row['buffer_type']in ('plain','tracked'),'Original buffer kind')
        raw=probe.fixture_bytes(row['shape']);changed=b'\x80\x3f'+raw[2:]
        require(row['bytes']==len(raw)and row['allocator_pressure_bytes']==4*len(raw),'Original bounded native storage fixture')
        for stage,data in [('initial',raw),('mutation',changed),('detached',raw)]:require(row[stage+'_sha256']==hashlib.sha256(data).hexdigest(),'Complete per-case native byte proof')
        for name in ('frombuffer_shares_exact_buffer','detach_shares_exact_buffer','bidirectional_alias_observed','detached_owner_survives_original_deletion','copy_negative_control_distinct','raw_word_probe_only'):require(row[name]is True,'Native storage observation must succeed')
        require(row['original_weights_loaded']is row['full_model_memory_fit_proven']is False,'Small fixture does not verify full model')
        require(row['tracked_buffer_released']is (True if row['buffer_type']=='tracked'else None),'Tracked lifetime outcome; plain case has no weakref')
    return rows


def review(result_ref,guard):
    """Re-read all18 complete saved payloads without importing native libraries."""
    require(callable(guard),'Actual independent serial/resource guard');guard();source_check()
    result=read_ref(result_ref,guard);root=Path(result_ref['path']).parent
    require(Path(result_ref['path'])==root/'result.json','Original complete result entrypoint')
    require(result['schema']=='dory_weight_storage_capture_result_v1'and result['complete']is result['passed']is True,'Actual successful capture result')
    require(all(result[name]is False for name in FLAGS),'No full-model claims in capture result')
    require(Path(result['probe']['path'])==root/'probe.json'and Path(result['manifest']['path'])==root/'raw-manifest.json','Exact original metadata paths')
    require(result['capture_source_sha256']=={str(SELF):hashlib.sha256(SELF.read_bytes()).hexdigest(),**{str(p):sha for p,sha in PINS.items()}},'Frozen complete capture source identity')
    native=read_ref(result['probe'],guard);rows=validate_probe(native);manifest=read_ref(result['manifest'],guard)
    require(manifest['schema']=='dory_weight_storage_saved_boundaries_v1'and manifest['tensors']==18 and manifest['raw_bytes']==786864 and manifest['all_files_rehashed']is True,'All complete saved tensor observations')
    records=manifest['rows'];rawroot=root/'raw';require(rawroot.resolve()==rawroot and rawroot.is_dir()and not rawroot.is_symlink(),'Original saved directory')
    require(set(root.iterdir())=={root/name for name in ('probe.json','raw-manifest.json','result.json','raw')},'No unexpected capture outputs')
    expected_paths={rawroot/(name+'.raw')for name in records};require(set(rawroot.iterdir())==expected_paths,'No missing or extra raw outputs')
    def read(record):
        path=Path(record['path']);require(path in expected_paths and path.parent==rawroot,'Original declared raw file')
        # A byte record cannot redirect one valid label to another equal fixture.
        labels=[name for name,row in records.items()if row is record];require(len(labels)==1 and path==rawroot/(labels[0]+'.raw'),'Saved label binds exact original path')
        require(type(record['bytes'])is int and 0<record['bytes']<=probe.MAX_BUFFER_BYTES,'Bounded whole tensor')
        raw=stable_bytes(path,record['bytes'],guard,record['sha256']);require(len(raw)==record['bytes'],'Exact complete raw size');return raw
    checked=probe.validate_saved(records,rows,read);guard()
    return dict(schema='dory_weight_storage_saved_review_v1',complete=True,result=result_ref,probe=result['probe'],manifest=result['manifest'],**checked,native_lifetime_report_authenticated=True,native_lifetime_independently_reexecuted=False,actual_process_exit_verified=False,full_model_qualified=False)


def run(*,torch,guard,output):
    require(callable(guard),'Actual admitted serial/resource guard');guard();source_check()
    output=Path(output);require(output.is_absolute()and output.resolve()==output and not output.exists(),'Fresh canonical output root')
    output.mkdir(mode=0o700);started=time.monotonic();writer=None
    try:
        writer=Capture(output/'raw',torch,guard)
        native=probe.run(torch=torch,guard=guard,emit=writer);validate_probe(native)
        native_ref=once(output/'probe.json',native);proof=writer.verify()
        require(proof==dict(tensors=18,raw_bytes=786864,all_files_rehashed=True),'Exact original complete native capture')
        manifest=once(output/'raw-manifest.json',dict(schema='dory_weight_storage_saved_boundaries_v1',rows=writer.rows,**proof))
        usage=resource.getrusage(resource.RUSAGE_SELF)
        value=dict(schema='dory_weight_storage_capture_result_v1',complete=True,passed=True,probe=native_ref,manifest=manifest,
                   capture_source_sha256={str(SELF):hashlib.sha256(SELF.read_bytes()).hexdigest(),**{str(p):sha for p,sha in PINS.items()}},
                   measured_resources=dict(elapsed_seconds=time.monotonic()-started,process_peak_rss_bytes=usage.ru_maxrss*1024),
                   measured_memory_scope='Process peak includes framework import and earlier process allocations; not full model memory.',**{name:False for name in FLAGS})
        guard();result_ref=once(output/'result.json',value)
        return dict(result=result_ref,**value)
    except BaseException as error:
        if not(output/'failure.json').exists():once(output/'failure.json',dict(schema='dory_weight_storage_capture_failure_v1',complete=False,error_type=type(error).__name__,error=str(error),partial_rows={}if writer is None else writer.rows,partial_raw_bytes=0 if writer is None else writer.total,native_lifetime_qualified=False,full_model_qualified=False,new_speedup_claim=False))
        raise
