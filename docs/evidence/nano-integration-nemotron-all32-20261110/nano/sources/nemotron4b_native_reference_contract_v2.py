"""Runtime helpers for a future serially admitted pinned native reference.

Functions accept an already constructed model/checkpoint. Nothing is imported
from torch/Transformers and nothing is loaded or allocated by importing this file.
"""
from nemotron4b_baseline_contract_v1 import PATTERN, original_inventory, validate_config, require


def native_state_dict(original):
    inventory=original_inventory()
    require(set(original)==set(inventory),'Expected all263 original checkpoint tensors')
    result={}
    for name,value in original.items():
        require(list(value.shape)==inventory[name] and str(value.dtype)=='torch.bfloat16','Wrong original tensor: '+name)
        target=name.replace('backbone.','model.',1) if name.startswith('backbone.') else name
        require(target not in result,'Duplicate native tensor')
        result[target]=value
    return result


def apply_native_runtime_contract(model):
    """Explicitly override the pinned CPU mixer's initialization-bound clamp."""
    validate_config(model.config.to_dict() | dict(hybrid_override_pattern=model.config.hybrid_override_pattern,
                                                num_hidden_layers=model.config.num_hidden_layers))
    require(model.config._attn_implementation=='eager','Native reference must use eager attention')
    require(len(model.model.layers)==42,'Wrong native layer count')
    changed=[]
    for index,(layer,kind) in enumerate(zip(model.model.layers,PATTERN)):
        require(layer.block_type=={'M':'linear_attention','-':'mlp','*':'full_attention'}[kind],'Native layer pattern mismatch')
        if kind=='M':
            m=layer.mixer
            require((m.num_heads,m.head_dim,m.n_groups,m.ssm_state_size,m.chunk_size)==(96,80,8,128,256),'Native Mamba geometry mismatch')
            before=m.time_step_limit
            require(before in ((.001,float('inf')),(0.,float('inf'))),'Unexpected pinned native clamp')
            m.time_step_limit=(0.,float('inf'))
            changed.append(dict(layer=index,before=[before[0],'inf'],after=[0.,'inf']))
        elif kind=='*':
            require(layer.mixer.head_dim==128,'Native attention head_dim fallback is invalid')
    require(len(changed)==21,'Expected21 runtime dt overrides')
    return changed


def verify_loaded_original_values(model, original):
    """Call only after admitted loading; compares every learned value in blocks."""
    import torch
    expected=native_state_dict(original);actual=model.state_dict()
    require(set(actual)==set(expected),'Native model contains missing/extra checkpoint tensors')
    checked=[]
    for name,source in expected.items():
        value=actual[name]
        require(value.dtype==torch.bfloat16 and value.shape==source.shape,'Loaded native shape/dtype changed')
        rows=max(1,524288//max(1,source[0].numel()))
        for start in range(0,source.shape[0],rows):
            require(torch.equal(value[start:start+rows].cpu(),source[start:start+rows].cpu()),'Loaded original values changed: '+name)
        checked.append(name)
    return dict(checked=checked,original_tensors=263,all_values_equal=True)


def verify_native_cache(cache, batch, length):
    """Run after each cached forward; do not infer state dtype from config."""
    require(len(cache.layers)==42,'Native cache must represent all42 layers')
    rows=[]
    for index,kind in enumerate(PATTERN):
        layer=cache.layers[index]
        if kind=='M':
            conv=cache.get_conv_state(index);state=cache.get_recurrent_state(index)
            require(list(conv.shape)==[batch,9728,4] and str(conv.dtype)=='torch.bfloat16','Native conv cache mismatch')
            require(list(state.shape)==[batch,96,80,128] and str(state.dtype)=='torch.float32','Native recurrent cache must stay FP32')
            rows.append(dict(layer=index,kind='M',conv_shape=list(conv.shape),ssm_shape=list(state.shape),ssm_dtype=str(state.dtype)))
        elif kind=='*':
            for value in (layer.keys,layer.values):
                require(list(value.shape)==[batch,8,length,128] and str(value.dtype)=='torch.bfloat16','Native attention cache mismatch')
            rows.append(dict(layer=index,kind='*',shape=list(layer.keys.shape)))
    require(len(rows)==25,'Native cached layer count changed')
    return rows
