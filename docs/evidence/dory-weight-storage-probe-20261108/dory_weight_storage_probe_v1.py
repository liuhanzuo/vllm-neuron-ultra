"""Deferred native Dory storage probe; never loads or qualifies official weights.

Import/source checks use stdlib only. An independently admitted serial native
runner must supply its actual torch module, live resource guard and byte sink.
"""
import ctypes,gc,hashlib,math,sys,weakref
from pathlib import Path
import dory_reference_loader_bridge_v1 as bridge
require=bridge.require
SELF=Path(__file__).resolve()
SHAPES=((4,),(4,8),(256,256))
STAGES=('initial','mutation','detached')
MAX_BUFFER_BYTES=128*1024
MAX_LIVE_SYNTHETIC_BYTES=2*2**20
BRIDGE_SHA='02574979fd2b2abaed6999f64f03297d94c1e4cf4c35ec02ccb280b0c1022a6d'
LOADER_SHA='61e20085087318d79b94e98fac6f82ada245818a2f61723de4b4afa888659cf8'


class TrackedBuffer(bytearray):
    """Supplemental weak-reference probe; plain bytearray is tested separately."""


def fixture_bytes(shape):
    require(tuple(shape)in SHAPES,'Only fixed bounded synthetic shapes')
    count=math.prod(shape);require(0<2*count<=MAX_BUFFER_BYTES,'Small exact BF16 storage fixture')
    # Largest fixture visits every 16-bit word, including +/-0, subnormals,
    # infinities and NaN payloads. No float arithmetic or model uses these words.
    return b''.join(((i*257+13)%65536).to_bytes(2,'little')for i in range(count))


def plan():
    sizes=[len(fixture_bytes(shape))for shape in SHAPES]
    return dict(schema='dory_weight_storage_probe_plan_v1',shapes=[list(s)for s in SHAPES],buffer_types=['plain','tracked'],stages=list(STAGES),cases=6,saved_tensors=18,saved_raw_bytes=2*len(STAGES)*sum(sizes),maximum_buffer_bytes=MAX_BUFFER_BYTES,live_synthetic_byte_budget=MAX_LIVE_SYNTHETIC_BYTES,live_synthetic_budget_scope='Logical tensor/byte buffers only; excludes Python lists, allocator and framework. Actual runner resident-memory budget remains required.',original_checkpoint_bytes=bridge.loader.FILE_BYTES,official_weight_values_read=False,model_constructed=False,model_forward_executed=False,native_reference_qualified=False,new_speedup_claim=False)


def source_check():
    for path,wanted in [(Path(bridge.__file__).resolve(),BRIDGE_SHA),(Path(bridge.loader.__file__).resolve(),LOADER_SHA)]:
        require(hashlib.sha256(path.read_bytes()).hexdigest()==wanted,'Original loader/bridge source unchanged')
    # Authenticates actual507-tensor header/source only; never accesses weights.
    _,_,contract=bridge.source_bundle()
    require(contract['tensor_count']==507 and contract['data_bytes']==7950458880,'Original full physical checkpoint contract')
    return contract


def packed(torch,tensor):
    require(tensor.device.type=='cpu'and tensor.dtype==torch.bfloat16,'CPU BF16 storage only')
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def case(torch,guard,emit,shape,kind):
    require(kind in ('plain','tracked'),'Explicit backing-buffer implementation');guard()
    expected=fixture_bytes(shape);buffer=bytearray(expected)if kind=='plain'else TrackedBuffer(expected)
    weak=weakref.ref(buffer)if kind=='tracked'else None
    address=ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    refs_before=sys.getrefcount(buffer)
    tensor=torch.frombuffer(buffer,dtype=torch.bfloat16).reshape(shape)
    require(sys.getrefcount(buffer)>refs_before,'Actual frombuffer retains Python owner')
    describe=bridge.tensor_storage(tensor)
    require(describe['data_pointer']==describe['storage_pointer']==address and describe['storage_bytes']==len(expected),'Actual frombuffer shares exact buffer storage')
    require(describe['shape']==list(shape)and describe['offset']==0 and describe['contiguous']is True and describe['requires_grad']is False,'Original complete layout')
    require(packed(torch,tensor)==expected,'All original BF16 words retained without conversion')
    label=kind+'_'+str(math.prod(shape));emit(label+'_initial',tensor);guard()
    adopted=tensor.detach();require(bridge.tensor_storage(adopted)==describe,'Actual detach preserves full physical storage descriptor')
    changed=b'\x80\x3f'+expected[2:];buffer[:2]=changed[:2]
    require(packed(torch,tensor)==changed and packed(torch,adopted)==changed,'Both original and detach observe buffer mutation')
    emit(label+'_mutation',adopted)
    byte_view=adopted.view(torch.uint8).reshape(-1)
    byte_view[0]=expected[0];byte_view[1]=expected[1]
    require(bytes(buffer)==expected and packed(torch,tensor)==expected and packed(torch,adopted)==expected,'Tensor-side mutation restores the same original buffer')
    del byte_view
    del tensor;gc.collect();guard()
    require(sys.getrefcount(buffer)>refs_before,'Detached tensor retains original owner after source tensor deleted')
    del buffer;gc.collect();guard()
    if weak is not None:require(weak()is not None,'Backing buffer lives after all caller-owned references deleted')
    pressure=[bytearray(len(expected))for _ in range(4)]
    require(sum(map(len,pressure))+5*len(expected)<=MAX_LIVE_SYNTHETIC_BYTES,'Bounded allocator-pressure fixture')
    require(bridge.tensor_storage(adopted)==describe and packed(torch,adopted)==expected,'Detached storage remains unchanged through GC and allocator pressure')
    emit(label+'_detached',adopted);guard()
    cloned=adopted.clone();clone_description=bridge.tensor_storage(cloned)
    require(clone_description['storage_pointer']!=address and packed(torch,cloned)==expected,'Copy negative control has distinct storage and same values')
    del adopted;gc.collect();guard()
    released=None
    if weak is not None:
        released=weak()is None;require(released,'No original buffer leak after last tensor alias released')
    # Clone is intentionally still alive when tracked original must disappear.
    require(packed(torch,cloned)==expected,'Independent copy survives original storage release')
    del cloned,pressure
    return dict(shape=list(shape),buffer_type=kind,bytes=len(expected),initial_sha256=hashlib.sha256(expected).hexdigest(),mutation_sha256=hashlib.sha256(changed).hexdigest(),detached_sha256=hashlib.sha256(expected).hexdigest(),frombuffer_shares_exact_buffer=True,detach_shares_exact_buffer=True,bidirectional_alias_observed=True,detached_owner_survives_original_deletion=True,allocator_pressure_bytes=4*len(expected),copy_negative_control_distinct=True,tracked_buffer_released=released,raw_word_probe_only=True,original_weights_loaded=False,full_model_memory_fit_proven=False)


def validate_saved(records,rows,read):
    """Independent stdlib verification of every complete tiny saved payload."""
    require(callable(read),'Bounded saved-byte reader required')
    require(len(rows)==6 and {(r['buffer_type'],tuple(r['shape']))for r in rows}=={(kind,shape)for kind in ('plain','tracked')for shape in SHAPES},'All six original buffer/shape cases')
    expected={};count=total=0
    for row in rows:
        shape=row['shape'];kind=row['buffer_type'];label=kind+'_'+str(math.prod(shape));raw=fixture_bytes(shape)
        for stage in STAGES:
            name=label+'_'+stage;require(name not in expected,'Unique complete capture name')
            expected[name]=((b'\x80\x3f'+raw[2:])if stage=='mutation'else raw,list(shape))
    require(set(records)==set(expected)and len(records)==18,'Exactly all18 saved whole-tensor observations')
    for name,(wanted,shape)in expected.items():
        record=records[name];require(record['shape']==shape and record['dtype']=='bfloat16','Complete actual saved shape and raw BF16 dtype')
        require(record['bytes']==len(wanted)and record['sha256']==hashlib.sha256(wanted).hexdigest(),'Actual complete saved metadata')
        actual=read(record);require(type(actual)is bytes and actual==wanted,'Independent complete saved raw words match deterministic fixture')
        count+=1;total+=len(actual)
    require(total==plan()['saved_raw_bytes'],'All saved payload bytes accounted')
    return dict(saved_tensors=count,saved_raw_bytes=total,all_saved_words_exact=True,torch_storage_lifetime_reexecuted=False,official_weights_verified=False,new_speedup_claim=False)


def run(*,torch,guard,emit):
    require(callable(guard)and callable(emit),'Actual independent serial lease and bounded byte sink');guard()
    require(sys.modules.get('torch')is torch and sys.byteorder=='little'and sys.dont_write_bytecode and __debug__,'Actual original little-endian native torch runtime')
    require(torch.is_inference_mode_enabled()and torch.get_num_threads()==1,'Original single-thread inference context')
    source_check();rows=[]
    for kind in ('plain','tracked'):
        for shape in SHAPES:rows.append(case(torch,guard,emit,shape,kind))
    guard();return dict(schema='dory_weight_storage_native_probe_v1',complete=True,passed=True,plan=plan(),cases=rows,torch_version=str(torch.__version__),source_sha256={str(SELF):hashlib.sha256(SELF.read_bytes()).hexdigest(),str(Path(bridge.__file__).resolve()):BRIDGE_SHA,str(Path(bridge.loader.__file__).resolve()):LOADER_SHA},plain_buffer_uses_production_frombuffer_detach_operations=True,tracked_buffer_only_supplements_lifetime_observation=True,all_raw_words_exact=True,checkpoint_downloaded=False,original_weights_loaded=False,model_constructed=False,model_forward_executed=False,peak_full_model_memory_fit_proven=False,native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False)
