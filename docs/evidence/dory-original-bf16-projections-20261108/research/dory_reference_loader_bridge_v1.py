"""Connect the exact Dory checkpoint loader to the original full reference candidate.

Source-only import: no torch, download, checkpoint access or model construction.
Execution requires the caller's admitted native runtime and live serial guard.
This bridge does not qualify the candidate math, model output or Neuron baseline.
"""
import hashlib,sys,types
from pathlib import Path
import dory_checkpoint_loader_v1 as loader
require=loader.require
Q=Path(__file__).resolve().parent
CANDIDATE=Q/'dory-native-reference-source-20261107-v1/dory_reference_candidate.py'
CANDIDATE_SHA='366dc0ab95bbeebb3bd2311f205e23afc63ce3fbbcfca321345185dcbc191716'
LOADER_SHA='61e20085087318d79b94e98fac6f82ada245818a2f61723de4b4afa888659cf8'
MODULE_NAME='_dory_original_reference_candidate_loader_v1'


def source_bundle():
    """Authenticate unchanged original sources and all507weight metadata only."""
    raw=loader.read_pinned(CANDIDATE,CANDIDATE_SHA,65536)
    loader.read_pinned(Path(loader.__file__).resolve(),LOADER_SHA,65536)
    contract_raw=loader.read_pinned(loader.CONTRACT_PATH,loader.CONTRACT_SHA,65536)
    specification=loader.source_contract()
    return raw,contract_raw,specification


def import_original(torch,guard):
    """Execute byte-for-byte candidate module only inside an admitted torch process."""
    require(callable(guard),'Actual live serial guard');guard()
    require(sys.modules.get('torch')is torch,'Caller supplies actual imported torch module')
    require('dory_contract'not in sys.modules and MODULE_NAME not in sys.modules,'Fresh isolated candidate module names')
    raw,contract_raw,specification=source_bundle()
    contract=types.ModuleType('dory_contract');contract.__file__=str(loader.CONTRACT_PATH)
    module=types.ModuleType(MODULE_NAME);module.__file__=str(CANDIDATE)
    sys.modules['dory_contract']=contract;sys.modules[MODULE_NAME]=module
    try:
        exec(compile(contract_raw,str(loader.CONTRACT_PATH),'exec',dont_inherit=True),contract.__dict__)
        exec(compile(raw,str(CANDIDATE),'exec',dont_inherit=True),module.__dict__)
        require(module.torch is torch and module.DoryContract is contract.DoryContract,'Actual original candidate imports')
        guard()
        return module,specification
    except BaseException:
        if sys.modules.get(MODULE_NAME)is module:sys.modules.pop(MODULE_NAME)
        if sys.modules.get('dory_contract')is contract:sys.modules.pop('dory_contract')
        raise


def tensor_storage(tensor):
    storage=tensor.untyped_storage()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=tensor.device.type,
                stride=list(tensor.stride()),offset=tensor.storage_offset(),elements=tensor.numel(),
                element_bytes=tensor.element_size(),contiguous=tensor.is_contiguous(),
                data_pointer=tensor.data_ptr(),storage_pointer=storage.data_ptr(),storage_bytes=storage.nbytes(),
                requires_grad=tensor.requires_grad)


def verify_bridge(weights,adopted,contract,proof,describe,guard):
    """Check every physical tensor and detached alias, retaining loader byte hashes.

    Shape/storage identity is not independently recomputed numerical parity.
    This pure interface accepts tiny explicit fixtures in source tests.
    """
    require(callable(guard)and callable(describe),'Actual guard and tensor inspector');guard()
    expected={row['name']:row for row in contract['rows']}
    require(len(expected)==contract['tensor_count']and set(weights)==set(adopted)==set(expected),'Complete original physical tensor set')
    require(proof['full_file_hash_verified']is True and proof['full_sha256']==contract['full_sha256']and
            proof['file_bytes_read']==contract['file_bytes']and proof['weight_bytes']==contract['data_bytes'],
            'Complete checkpoint hash verified before constructing reference')
    hashes={row['name']:row for row in proof['tensors']}
    require(len(hashes)==len(proof['tensors'])==len(expected)and set(hashes)==set(expected),'One loader byte proof per tensor')
    pointers=set();rows=[]
    for name,row in expected.items():
        guard();a,b=describe(weights[name]),describe(adopted[name]);size=row['end']-row['start']
        require(a==b,'Candidate detach retains exact physical storage/layout/dtype: '+name)
        require(a['shape']==row['shape']and a['dtype']=='torch.bfloat16'and a['device']=='cpu'and
                a['contiguous']is True and a['offset']==0 and a['requires_grad']is False and
                a['element_bytes']==2 and a['elements']*2==size==a['storage_bytes'],'Original complete BF16 tensor: '+name)
        require(type(a['storage_pointer'])is int and a['storage_pointer']>0 and
                a['data_pointer']==a['storage_pointer']and a['storage_pointer']not in pointers,'Distinct full physical backing buffer: '+name)
        pointers.add(a['storage_pointer'])
        h=hashes[name]
        require(h['shape']==row['shape']and h['dtype']=='BF16'and h['bytes']==size and
                isinstance(h['sha256'],str)and len(h['sha256'])==64 and all(c in '0123456789abcdef'for c in h['sha256']),
                'Original full tensor loader proof: '+name)
        rows.append(dict(name=name,shape=row['shape'],bytes=size,loader_sha256=h['sha256'],same_storage=True))
    require(sum(row['bytes']for row in rows)==contract['data_bytes'],'Complete checkpoint storage retained')
    guard();return dict(complete=True,tensors=rows,tensor_count=len(rows),weight_bytes=contract['data_bytes'],
                       all_detached_weights_share_original_storage=True,all_physical_weights_have_distinct_buffers=True,
                       tensor_payload_copied_by_bridge=False,learned_scales_included=True,
                       independent_numeric_equality_executed=False,model_forward_executed=False,new_speedup_claim=False)


def build_reference(checkpoint_path,expected_identity,*,torch,guard):
    """Future serial native execution; preserves original BF16 TP1 two-loop model."""
    require(callable(guard),'Independent admitted serial guard required');guard()
    require(sys.byteorder=='little'and sys.dont_write_bytecode and __debug__,'Original little-endian runtime, no bytecode')
    require(torch.is_inference_mode_enabled(),'Caller enters original inference mode')
    module,contract=import_original(torch,guard)
    weights,load_proof=loader.load_checkpoint(checkpoint_path,expected_identity,torch=torch,guard=guard)
    guard()
    model=module.DoryReferenceCandidate(contract['config'],weights,tensor_parallel_size=1,float32_fixture=False)
    require(model.dtype==torch.bfloat16 and model.config.n_recurrent_loops==2 and len(model.config.cache_plan())==56,
            'Original BF16 two loops and56independent cache histories')
    require(not model.slots and not model.epochs,'No request or forward during construction')
    storage_proof=verify_bridge(weights,model.weights,contract,load_proof,tensor_storage,guard)
    del weights
    guard()
    return model,dict(schema='dory_original_reference_loader_bridge_v1',complete=True,
                      loader=load_proof,storage=storage_proof,physical_tensors=507,parameters=3975229440,
                      n_recurrent_loops=2,per_loop_cache_histories=56,
                      source_sha256={str(CANDIDATE):CANDIDATE_SHA,str(loader.CONTRACT_PATH):loader.CONTRACT_SHA,
                                     str(Path(loader.__file__).resolve()):LOADER_SHA,str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                      model_forward_executed=False,native_reference_qualified=False,neuron_baseline_qualified=False,
                      original_dispatch_and_projection_sdpa_parity_proven=False,peak_memory_fit_proven=False,new_speedup_claim=False)
