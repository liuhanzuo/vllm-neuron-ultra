"""Official-only Stage A successor worker library, source preparation only.

No CLI, native imports, dynamic import, admission, binding, launch, checkpoint
open or model work at import. Future coordinator must authenticate all injected
native modules/callables, current checkpoint identity and this source closure,
then enforce a fresh serial/resource guard before calling run_worker.

Original loader/507 weights/399 parameters, upstream CPU ReferenceAttention,
182 oneDNN linear modules, BF16 math and RoPE remain. No candidate is built and
no F.linear route is installed. A successful return is NOT actual process exit;
only the external parent can observe that and admit an independent reviewer.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import dory_checkpoint_loader_v1 as loader
from dory_stage_a_archive_core_v1 import require
from dory_stage_a_capture_core_v1 import StageAObserver, CACHE_IDS, WIDE
from dory_stage_a_tensor_adapter_v1 import NativeTensorAdapter
from dory_stage_a_artifact_transaction_v2 import ArchiveFileStore, CaptureArtifactBatch
from dory_stage_a_durable_metadata_v1 import write_json_exclusive, MAX_METADATA_BYTES
from prepare_dory_stage_a_source_inputs_v1 import validate_source_inputs, LOADER_SHA

NATIVE_FIELDS = ('torch','monkeypatch_factory','upstream','official_source','ops','parallel_state',
    'VllmConfig','DeviceConfig','CompilationConfig','set_current_vllm_config','DoryConfig','extract_layer_index')


def mapped_parameter(name, shape):
    require(isinstance(name,str) and isinstance(shape,(list,tuple)) and len(shape) in (1,2)
            and all(type(x) is int and x > 0 for x in shape), 'Original rank1/rank2 physical weight')
    target, begin = name, 0
    for old,new,offset in (('q_proj','qkv_proj',0),('k_proj','qkv_proj',4096),('v_proj','qkv_proj',5120),
                           ('gate_proj','gate_up_proj',0),('up_proj','gate_up_proj',7168)):
        if '.'+old+'.' in name:
            target,begin = name.replace('.'+old+'.','.'+new+'.'),offset
            break
    return target,begin,begin+shape[0]


def tensor_sha256(tensor, adapter):
    value = hashlib.sha256()
    for block in adapter.iter_bytes(tensor, None):
        value.update(block)
    return value.hexdigest()


def capture_cases(torch, official, caches, counts, prepared, archive_root, *, guard):
    """Consume the real v2 transaction; no native construction or serialization alternative.

    Caller must own the entire official model exclusively with no live requests
    or cache banks. Stage A consists of six independent empty-history start0
    captures, including original event6179; this is not production slot reuse.
    """
    cases = prepared['cases']
    require(set(caches) == set(CACHE_IDS) and len({id(x) for x in caches.values()}) == 56
            and all(c.keys is None and c.values is None for c in caches.values()),
            'Exclusive original56 caches initially empty')
    adapter = NativeTensorAdapter(torch, guard=guard)
    observer = StageAObserver(official,caches,adapter,lambda:counts[0],guard)
    store = ArchiveFileStore(archive_root, guard=guard)
    isolation, initial = [], counts[0]
    try:
        batch = CaptureArtifactBatch(cases,observer,adapter,store,guard=guard)
        observer.install()
        for case in cases:
            guard()
            for cache in caches.values():
                cache.keys = cache.values = None
                cache.start_pos = 0
            require(all(c.keys is None and c.values is None and c.start_pos == 0 for c in caches.values()),
                    'All56 empty original histories at every independent start0')
            tokens = torch.tensor(case['input_ids'],dtype=torch.int64,device='cpu')
            positions = torch.arange(case['identity']['end'],dtype=torch.int64,device='cpu')
            try:
                batch.collect(tokens,positions)
            finally:
                # No aliases/banks may hold these caches; the admitted worker owns them.
                # Clear even when model/stream/guard fails, without completing that case.
                for cache in caches.values():
                    cache.keys = cache.values = None
                    cache.start_pos = 0
            isolation.append(dict(identity=deepcopy(case['identity']),rows=case['identity']['end'],
                original_tokens_and_positions_checked_by_observer=True,empty_histories_before=56,
                empty_histories_after=56,production_slot_reuse_qualified=False))
        require(counts[0]-initial == 6*281 and len(isolation) == 6, 'Original complete six-forward oneDNN count')
        artifacts = batch.finish()
    finally:
        try:
            observer.close()
        finally:
            store.abort()
    require(not observer.handles and all(c.keys is None and c.values is None and c.start_pos == 0
                                         for c in caches.values()), 'All passive hooks removed and histories released')
    return artifacts,isolation


def _native_capture(api, prepared, checkpoint_identity, archive_root, guard):
    require(all(hasattr(api,name) for name in NATIVE_FIELDS), 'Complete explicit original native API bundle')
    torch = api.torch
    guard()
    require(str(torch.__version__) == '2.14.0+cpu', 'Original CPU torch version')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    contract = loader.source_contract()
    expected = prepared['expected_model']
    require(contract['config'] == expected['config'] and contract['tensor_count'] == 507
            and contract['parameters'] == 3975229440, 'Original pinned full config/physical weights')
    counts = [0]
    original_op, original_linear = api.ops.onednn_mm, torch.nn.functional.linear
    original_attention, original_tp = api.official_source.Attention, api.parallel_state._TP
    initial_dtype, initial_inference = torch.get_default_dtype(), torch.is_inference_mode_enabled()
    def counted_op(*args,**kwargs):
        counts[0] += 1
        return original_op(*args,**kwargs)
    patches = api.monkeypatch_factory()
    try:
        patches.setattr(api.ops,'onednn_mm',counted_op)
        patches.setattr(api.parallel_state,'_TP',SimpleNamespace(world_size=1,rank_in_group=0))
        patches.setattr(api.official_source,'Attention',api.upstream.ReferenceAttention)
        vc = api.VllmConfig(device_config=api.DeviceConfig(device='cpu'),
            compilation_config=api.CompilationConfig(mode=0,custom_ops=['none']))
        vc.model_config = SimpleNamespace(hf_config=api.DoryConfig(**contract['config']),head_dtype=None,dtype=torch.bfloat16)
        with torch.inference_mode(),api.set_current_vllm_config(vc):
            guard()
            weights,proof = loader.load_checkpoint(Path(checkpoint_identity['path']),checkpoint_identity,
                                                   torch=torch,guard=guard)
            proof_keys = {'full_sha256','full_file_hash_verified','file_bytes_read','weight_bytes','tensors',
                'chunk_bytes','separately_qualified_model','checkpoint_identity','config_sha256','header_sha256',
                'parameters','physical_weight_sharing','loader_source_sha256','model_forward_executed',
                'peak_memory_fit_proven','new_speedup_claim'}
            require(type(proof) is dict and set(proof) == proof_keys
                    and proof['tensors'] == prepared['previous_loader_tensors'] and len(weights) == 507
                    and proof['full_file_hash_verified'] is True
                    and proof['full_sha256'] == loader.FULL_SHA
                    and proof['file_bytes_read'] == loader.FILE_BYTES
                    and proof['weight_bytes'] == loader.DATA_BYTES
                    and proof['parameters'] == 3975229440
                    and proof['checkpoint_identity'] == checkpoint_identity
                    and proof['loader_source_sha256'] == LOADER_SHA
                    and proof['config_sha256'] == loader.CONFIG_SHA and proof['header_sha256'] == loader.HEADER_SHA
                    and type(proof['chunk_bytes']) is int and proof['chunk_bytes'] == loader.CHUNK_BYTES
                    and proof['physical_weight_sharing'] == 'same 507 tensors for both original loops'
                    and all(proof[k] is False for k in ('separately_qualified_model','model_forward_executed',
                        'peak_memory_fit_proven','new_speedup_claim')),
                    'Every original507 full checkpoint tensor and exact frozen loader proof')
            prior_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                official = api.official_source.DoryForCausalLM(vllm_config=vc)
            finally:
                torch.set_default_dtype(prior_dtype)
            require(len(official.backbone.layers) == 72 and official.backbone.num_loops == 2
                    and all(p.dtype == torch.bfloat16 and p.device.type == 'cpu' for p in official.parameters()),
                    'Complete72-layer BF16 CPU official model with2 loops')
            params = dict(official.named_parameters())
            require(len(params) == 399 and sum(p.numel() for p in params.values()) == 3975229440,
                    'Original399 physical parameters and3975229440 elements')
            require(official.load_weights(list(weights.items())) == set(params), 'Every official parameter loaded')
            originals = {r['name']:r for r in proof['tensors']}
            require(set(originals) == set(weights), 'Original exact507 named weights')
            adapter, mapped = NativeTensorAdapter(torch,guard=guard), []
            for name,tensor in weights.items():
                guard()
                row = originals[name]
                require(list(tensor.shape) == row['shape'] and tensor.dtype == torch.bfloat16,
                        'Original BF16 weight geometry')
                target,begin,end = mapped_parameter(name,tensor.shape)
                require(target in params, 'Original merged official destination')
                actual = params[target][begin:end]
                require(list(actual.shape) == row['shape'] and actual.dtype == torch.bfloat16,
                        'Unchanged complete official weight slice')
                fingerprint = tensor_sha256(actual,adapter)
                require(fingerprint == row['sha256'], 'Exact bytes for every loaded official weight slice')
                mapped.append(dict(original=name,target=target,begin=begin,end=end,sha256=fingerprint))
            loaded = dict(all507_exact=True,physical_parameters=399,parameters=3975229440,mapping=mapped)
            del params,weights,tensor,actual
            original_gather = official.logits_processor._gather_logits
            patches.setattr(official.logits_processor,'_gather_logits',lambda logits:logits)
            api.upstream._prepare_for_inference(official)
            backends = {name:list(module.cpu_linear.__code__.co_names)
                        for name,module in official.named_modules() if hasattr(module,'cpu_linear')}
            require(backends == expected['backends'] and len(backends) == 182
                    and all('onednn_mm' in names for names in backends.values()), 'Exact original182 oneDNN modules')
            cache_list = [m for m in official.modules() if isinstance(m,api.upstream.ReferenceAttention)]
            caches = {api.extract_layer_index(m.prefix):m for m in cache_list}
            require(len(cache_list) == len(caches) == 56 and sorted(caches) == expected['cache_ids']
                    and set(caches) == set(CACHE_IDS), 'Original56 distinct per-loop ReferenceAttention caches')
            ropes = {id(m.rotary_emb):m.rotary_emb for m in (x.mixer for x in official.backbone.layers)
                     if hasattr(m,'rotary_emb')}
            require(len(ropes) == 2, 'Original two RoPE tables')
            rotary = []
            for rope in ropes.values():
                require(list(rope.cos_sin_cache.shape) == [262144,256]
                        and rope.cos_sin_cache.dtype == torch.bfloat16, 'Original full BF16 RoPE table')
                rotary.append(dict(shape=list(rope.cos_sin_cache.shape),dtype=str(rope.cos_sin_cache.dtype),base=rope.base))
            require(rotary == expected['rotary_tables'] and torch.nn.functional.linear is original_linear
                    and api.ops.onednn_mm is counted_op, 'Original RoPE and untouched F.linear with passive oneDNN counter')
            model = dict(config=deepcopy(contract['config']),official_parameters_before_postload=399,
                parameters=3975229440,all_original_weights_exact=True,backends=backends,
                cache_ids=sorted(caches),rotary_tables=rotary,physical_layers=72,physical_visits=112,
                attention='Original upstream CPU ReferenceAttention',official_only=True,
                candidate_constructed=False,functional_linear_route_installed=False,
                native_source_provenance_authenticated_here=False,
                native_api_authentication_required_from_caller=True,
                tokenizer_template_environment_authentication_required_from_caller=True,
                full_model_qualified=False,new_speedup_claim=False)
            artifacts,isolation = capture_cases(torch,official,caches,counts,prepared,archive_root,guard=guard)
    finally:
        patches.undo()
    require(api.ops.onednn_mm is original_op and torch.nn.functional.linear is original_linear
            and api.official_source.Attention is original_attention and api.parallel_state._TP is original_tp
            and official.logits_processor._gather_logits == original_gather
            and torch.get_default_dtype() == initial_dtype
            and torch.is_inference_mode_enabled() == initial_inference,
            'Original dispatch/default dtype/inference state restored before producer metadata completion')
    return dict(loader_proof=proof,loaded_parameters=loaded,model=model,artifacts=artifacts,isolation=isolation)


def run_worker(api, prepared, checkpoint_identity, archive_root, *, guard):
    """Future admitted fresh worker only; current task never calls this natively.

    checkpoint_identity must come from future actual admission, never from the
    historical source-input preparer. Caller authenticates api and all pinned
    originals; guard owns actual lock, resource caps and exclusive model lifetime.
    No reviewer is called here. Parent must observe actual exit and physical
    release, authenticate these durable metadata files, then admit a separately
    sourced reviewer calling review_saved_batch on stage-a-artifacts.json.
    """
    require(callable(guard), 'Fresh admitted producer guard')
    guard()
    prepared = deepcopy(validate_source_inputs(prepared))
    require(type(checkpoint_identity) is dict
            and set(checkpoint_identity) == {'path','device','inode','size','mtime_ns','ctime_ns'}
            and type(checkpoint_identity['path']) is str and Path(checkpoint_identity['path']).is_absolute()
            and all(type(checkpoint_identity[k]) is int and checkpoint_identity[k] >= 0
                    for k in ('device','inode','size','mtime_ns','ctime_ns'))
            and checkpoint_identity['size'] == loader.FILE_BYTES,
            'Explicit future actual checkpoint identity; not supplied by source preparation')
    result = _native_capture(api,prepared,deepcopy(checkpoint_identity),archive_root,guard)
    guard()
    refs = {}
    for key,name,bound in (('loader_proof','stage-a-loader-proof.json',2*1024**2),
                           ('loaded_parameters','stage-a-loaded-parameters.json',2*1024**2),
                           ('model','stage-a-model.json',2*1024**2),
                           ('artifacts','stage-a-artifacts.json',MAX_METADATA_BYTES)):
        refs[key] = write_json_exclusive(archive_root,name,result[key],max_bytes=bound,guard=guard)
    manifest = dict(schema='dory_stage_a_official_producer_v1',producer_function_completed=True,
        source_input_sha256=prepared['input_sha256'],evidence=refs,cache_isolation=result['isolation'],
        original_cases=[deepcopy(x['identity']) for x in prepared['cases']],
        original_quality_gates=deepcopy(prepared['quality_gates']),original_failures=list(prepared['original_failures']),
        producer_process_exit_observed=False,independent_saved_byte_review_complete=False,
        requires_actual_exit_and_separate_review=True,production_slot_reuse_qualified=False,
        original_FP32_acceptance_passed=False,all87_marked_calls_qualified=False,
        full_model_qualified=False,new_speedup_claim=False)
    return write_json_exclusive(archive_root,'stage-a-producer.json',manifest,max_bytes=2*1024**2,guard=guard)


if __name__ == '__main__':
    raise RuntimeError('Source preparation only; native API authentication, fresh binding/limits/admission and actual exit coordinator pending')
