"""Fresh observer-off DEVICE load/driver leaf; no CLI or import-time IO.

The external serial owner authenticates original weights/environment/closure,
supplies independent original artifact/native-reference hashes and a live guard.
This leaf loads only the original uncaptured model and calls the frozen full
control driver. It never compiles, launches, cleans outputs or grants admission.
"""
import hashlib
import json
from pathlib import Path
import sys

from deepseek_cold_boundary_device_entry_v1 import require,sha256_shape,STRICT_FAILURE,NATIVE_MODULES


def paths(artifact,destination,reference_directory,data_root):
    artifact,destination,reference_directory,data_root=map(Path,(artifact,destination,reference_directory,data_root))
    for path in (artifact,destination,reference_directory,data_root):
        require(path.is_absolute()and path.resolve()==path and path.is_dir()
            and not any(p.is_symlink()for p in (path,*path.parents))
            and 'deepseek-cold-kv-diagnostic-v2'not in path.parts,'Existing ordinary independent control roots required')
    require(reference_directory==destination/'cold-reference','Exact completed observer-off native reference required')
    independent=(artifact,destination,data_root)
    require(len(set(independent))==3 and all(a not in b.parents and b not in a.parents
        for i,a in enumerate(independent)for b in independent[i+1:]),'Disjoint original artifact and off outputs required')
    targets=[destination/'cache-captures/device',data_root/'device']
    targets += [destination/name for name in ('state-copy.json','state-lifecycle.json','lifecycle-states',
        'cold-buckets.json','cold-buckets.json.tmp','cold-buckets.pt')]
    require(all(not p.exists()and not p.is_symlink()and p.resolve()==p for p in targets),'Fresh complete observer-off DEVICE outputs required')
    return artifact,destination,reference_directory,data_root


def _native_model_class():
    import torch
    import server_with_bias_fix
    from deepseek_qwen3_neuron_adapter import NeuronDeepSeekQwen3ForCausalLM
    torch.set_num_threads(1)
    return NeuronDeepSeekQwen3ForCausalLM


def _config(artifact,expected_sha256):
    path=artifact/'neuron_config.json'
    require(path.stat().st_size==8378,'Original serialized config size required')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,'Externally authenticated original config changed')
    return json.loads(raw)


def run(artifact,expected_artifact_sha256,case,destination,reference_directory,data_root,
        original_cold_metadata,expected_reference_sha256,execution_guard):
    require(callable(execution_guard),'Owning authenticated serial execution guard required')
    revoked=False
    def guard():
        nonlocal revoked
        try:execution_guard()
        except BaseException:
            revoked=True;raise
    guard()
    require(__debug__ and not any(n.split('.')[0]in NATIVE_MODULES for n in sys.modules)
        and sys.gettrace()is None,'Fresh original DEVICE process without native imports or trace required')
    selected=paths(artifact,destination,reference_directory,data_root)
    artifact,destination,reference_directory,data_root=selected
    import deepseek_capture_graph_entry_v1 as graph
    import deepseek_cpu_capture_bridge_v1 as bridge
    from deepseek_cold_kv_protocol_v2 import expected_case,check_device_metadata,check_copy,check_lifecycle,read
    from deepseek_cold_kv_contract_v2 import check_manifest
    require(case==expected_case()and str(artifact)==case['artifact']==str(graph.ORIGINAL_ARTIFACT)
        and case['model_path']==str(graph.MODEL_PATH),'Exact original BF16 TP2 B4 context4096 case and artifact required')
    artifact_paths={str(artifact/name)for name in ('model.pt','neuron_config.json')}
    reference_paths={str(reference_directory/name)for name in ('cold-reference.pt','cold-reference.json')}
    for pins,expected in ((expected_artifact_sha256,artifact_paths),(expected_reference_sha256,reference_paths)):
        require(type(pins)is dict and set(pins)==expected,'Complete exact externally authenticated artifact/reference hashes required')
        for digest in pins.values():sha256_shape(digest)
    require(expected_artifact_sha256[str(artifact/'neuron_config.json')]==graph.ORIGINAL_CONFIG_SHA256,
        'Original uncaptured config digest required')
    require(not (artifact/'weights').exists()and not (artifact/'weights').is_symlink(),
        'Original unsharded artifact; no substitute saved weight shards')
    check_device_metadata(original_cold_metadata,original_cold_metadata)
    bridge.check_neutrality_metadata(original_cold_metadata,original_cold_metadata)
    def authenticated(pins):
        for path,digest in pins.items():
            guard();require(graph.file_record(path)['sha256']==digest,'Externally authenticated input changed: '+path)
    authenticated(expected_artifact_sha256);authenticated(expected_reference_sha256);guard()
    expected=_config(artifact,graph.ORIGINAL_CONFIG_SHA256);n=expected['neuron_config']
    require(expected['_name_or_path']==case['model_path']and n['save_sharded_checkpoint']is False
        and n['skip_sharding']is False and n['skip_warmup']is False
        and n.get('tensor_capture_config')is None and n.get('on_device_sampling_config')is None,
        'Complete original uncaptured CPU-sampler and sharding configuration required')
    cls=_native_model_class();guard()
    cfg=cls.get_config_cls().load(str(artifact))
    graph.check_config(json.loads(cfg.to_json_string()),expected)
    require(cfg.neuron_config.batch_size==4 and cfg.neuron_config.tp_degree==2
        and not cfg.neuron_config.async_mode and cfg.neuron_config.cast_type=='config',
        'Original BF16 TP2 B4 synchronous config required')
    model=cls(case['model_path'],config=cfg)
    authenticated(expected_artifact_sha256);guard()
    model.load(str(artifact),skip_warmup=True);guard()
    require(type(model)is cls and model.is_loaded_to_neuron is True
        and model.neuron_config.tensor_capture_config is None
        and model.neuron_config.on_device_sampling_config is None,'Loaded exact uncaptured original model required')
    for wrapper in (model.context_encoding_model,model.token_generation_model):
        require(tuple(wrapper.neuron_config.buckets)==graph.BUCKETS,'All six original CTE and six TKG buckets required')
    from qwen35_split_state_checks_v4 import graph_routes
    from deepseek_prior_mask_state_v1 import bindings
    routes,states=graph_routes(model),bindings(model)
    require(routes['complete']and len(routes['graphs'])==12 and len(states)==144,
        'All original loaded graph routes and144 state objects required')
    paths(*selected);authenticated(expected_reference_sha256);guard()
    from validate_deepseek_cold_control_v1 import main as original_loop
    saved=[]
    for wrapper in (model.context_encoding_model,model.token_generation_model):
        saved.append((wrapper,'_forward'in vars(wrapper),vars(wrapper).get('_forward'),wrapper._forward))
    try:
        for wrapper,own,stored,original in saved:
            def guarded(*args,_original=original,**kwargs):
                guard();result=_original(*args,**kwargs);guard();return result
            wrapper._forward=guarded
        guard()
        try:original_loop(model,case,destination,reference_directory,data_root,guard)
        except SystemExit as error:
            if revoked:raise
            require(type(error)is SystemExit and type(error.code)is str and error.code==STRICT_FAILURE,
                'Unexpected observer-off DEVICE termination')
        else:raise ValueError('Original failed cold gate unexpectedly changed')
        guard()
        manifest=read(destination/'cache-captures/device/manifest.json');check_manifest(manifest,'device')
        report=read(destination/'cold-buckets.json')
        require('error'not in report,'Partial/error cold report cannot classify original gate')
        metadata=check_device_metadata(report,original_cold_metadata)
        classification=bridge.check_neutrality_metadata(original_cold_metadata,report)
        check_copy(read(destination/'state-copy.json'))
        lifecycle=read(destination/'state-lifecycle.json');check_lifecycle(lifecycle)
        require(lifecycle['state_tensors_changed_by_warmup']==144 and lifecycle['same144_bindings_preserved'],
            'Complete original144 state lifecycle required')
        guard()
    finally:
        errors=[]
        for wrapper,own,stored,original in reversed(saved):
            try:
                if own:wrapper._forward=stored
                elif '_forward'in vars(wrapper):delattr(wrapper,'_forward')
            except BaseException as error:errors.append(error)
        if errors:raise errors[0]
    require(sys.gettrace()is None,'No residual observer trace permitted')
    for wrapper,own,stored,original in saved:
        require(('_forward'in vars(wrapper))==own and(not own or vars(wrapper)['_forward']is stored),
            'Original wrapper attribute ownership restored')
    guard()
    return dict(observer='off',device_frames=84,original_forward_calls=42,full_device_KV_tensors=6048,
        state_objects=144,original_failed_gate_preserved=True,metadata_reproduction=metadata,
        full_frame_classification=classification,artifact_sha256=dict(expected_artifact_sha256),
        native_reference_sha256=dict(expected_reference_sha256),route_hooks_restored=True,
        original_precision='BF16',tp=2,batch=4,context_length=4096,
        original_raw_reproduction_pending=True,neutrality_proven=False,full_model_qualified=False,new_speedup_claim=False)
