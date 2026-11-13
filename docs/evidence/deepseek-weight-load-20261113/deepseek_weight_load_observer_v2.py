"""Observe the actual original SDK load, without replacing sharding or loading.

Only the owning serial worker may install this process-local observer. Its
callback receives partial evidence before load_weights returns to load/driver;
the callback must raise while the separate layout/inference gate is unmet.
No report from this module grants layout equivalence or inference admission.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 'deepseek_actual_weight_load_observation_v1'
CHUNK_BYTES = 4 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _dependencies():
    import torch
    from neuronx_distributed.trace.model_builder import ModelBuilder
    from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
    return torch, ModelBuilder, NeuronApplicationBase


def _descriptor(tensor):
    storage = tensor.untyped_storage()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                device=str(tensor.device), layout=str(tensor.layout),
                storage_offset=int(tensor.storage_offset()), storage_bytes=int(storage.nbytes()),
                storage_address=int(storage.data_ptr()), tensor_version=int(tensor._version),
                numel=int(tensor.numel()), element_bytes=int(tensor.element_size()))


def tensor_record(tensor, torch, guard):
    """Hash every byte of the CPU tensor representation, with no numeric cast.

This is NOT a read of a device allocator's physical backing buffer. .cpu() and
contiguous() may present a logical view; that limitation is part of the report.
One full tensor may be copied to host, in addition to SDK loading peaks.
"""
    guard()
    require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
            'Complete dense tensor required; no opaque/sparse placeholder')
    before = _descriptor(tensor)
    require(before['numel'] > 0 and before['element_bytes'] > 0,
            'Nonempty real tensor required')
    cpu = tensor.detach().cpu().contiguous()
    require(str(cpu.dtype) == before['dtype'] and list(cpu.shape) == before['shape'],
            'CPU observation must preserve exact tensor dtype and shape')
    # Viewing bytes preserves NaN payloads and signed zero; dtype conversion is forbidden.
    # Flatten before the byte view so scalar rank/buffer tensors are supported.
    byte_view = cpu.reshape(-1).view(torch.uint8)
    view = memoryview(byte_view.numpy()).cast('B')
    total = before['numel'] * before['element_bytes']
    require(len(view) == total, 'Complete tensor byte denominator required')
    digest = hashlib.sha256()
    for offset in range(0, total, CHUNK_BYTES):
        guard()
        digest.update(view[offset:offset + CHUNK_BYTES])
    guard()
    require(_descriptor(tensor) == before, 'Tensor storage or version changed while observed')
    return dict(descriptor=before, cpu_bytes=total, cpu_sha256=digest.hexdigest(),
                observation='detach.cpu.contiguous.byte_view', physical_device_layout_proven=False)


def tensor_map_record(ranks, torch, guard):
    require(type(ranks) is list and len(ranks) == 2, 'Both actual TP ranks required')
    records = []
    for rank, weights in enumerate(ranks):
        require(type(weights) is dict and weights and '__neuronprivatetensor__' not in weights,
                'Complete initialized rank dictionary required')
        require(all(type(key) is str and key for key in weights), 'Exact complete weight keys')
        rows = []
        for key in sorted(weights):
            rows.append(dict(key=key, **tensor_record(weights[key], torch, guard)))
        records.append(dict(rank=rank, original_key_order=list(weights), entries=rows))
    require(set(ranks[0]) == set(ranks[1]), 'Both ranks must have the same complete key set')
    return records


def _semantic_rows(records):
    """Compare values/TP rank and represented geometry, excluding process addresses."""
    require(type(records) is list and [r['rank'] for r in records] == [0, 1], 'Both ordered ranks')
    result = []
    for record in records:
        rows = record['entries']
        require(rows and [x['key'] for x in rows] == sorted(set(x['key'] for x in rows)),
                'Unique sorted complete entries')
        require(set(record['original_key_order']) == {x['key'] for x in rows}
                and len(record['original_key_order']) == len(rows), 'Complete original key order')
        selected = []
        for row in rows:
            d = row['descriptor']
            require(type(d['shape']) is list and all(type(x) is int and x > 0 for x in d['shape'])
                    and type(d['numel']) is int and d['numel'] == math.prod(d['shape'])
                    and type(d['element_bytes']) is int and d['element_bytes'] > 0,
                    'Exact scalar/dense tensor geometry and byte size')
            require(row['physical_device_layout_proven'] is False
                    and row['observation'] == 'detach.cpu.contiguous.byte_view', 'Exact observation scope')
            require(type(row['cpu_sha256']) is str and len(row['cpu_sha256']) == 64
                    and all(c in '0123456789abcdef' for c in row['cpu_sha256']), 'Complete tensor digest')
            require(row['cpu_bytes'] == d['numel'] * d['element_bytes'] > 0, 'Full tensor bytes')
            selected.append(dict(key=row['key'], shape=d['shape'], dtype=d['dtype'],
                                 numel=d['numel'], element_bytes=d['element_bytes'],
                                 cpu_bytes=row['cpu_bytes'], cpu_sha256=row['cpu_sha256']))
        result.append(dict(rank=record['rank'], entries=selected))
    return result


def compare_observations(before, after):
    """Pure partial comparison. Authenticating producer exit/drain is external.

Even exact values here do not identify how each inference graph interprets a
transformed buffer. The caller must preserve False qualification fields.
"""
    for value, arm in ((before, 'off'), (after, 'on')):
        require(value['schema'] == SCHEMA and value['arm'] == arm
                and value['original_loader_calls'] == value['original_shard_calls'] == 1,
                'One actual original loader/sharder observation per arm')
        require(all(value[k] is False for k in ('layout_semantics_proven', 'graph_weight_ABI_proven',
                'full_model_qualified', 'new_speedup_claim', 'inference_admitted')),
                'Load observation cannot grant graph or inference qualification')
    require(before['checkpoint_provenance'] == after['checkpoint_provenance'],
            'Same externally authenticated original checkpoint/configuration lineage')
    pre_equal = _semantic_rows(before['pre_layout_shards']) == _semantic_rows(after['pre_layout_shards'])
    loaded_equal = _semantic_rows(before['loaded_cpu_views']) == _semantic_rows(after['loaded_cpu_views'])
    return dict(schema='deepseek_weight_load_partial_comparison_v1',
                pre_layout_full_rank_key_values_equal=pre_equal,
                loaded_full_rank_key_cpu_views_equal=loaded_equal,
                weight_loader_presence_equal=before['weight_loader_present'] == after['weight_loader_present'],
                observation_sha256=[hashlib.sha256(canonical(v)).hexdigest() for v in (before, after)],
                graph_weight_ABI_proven=False, layout_semantics_proven=False,
                full_model_qualified=False, new_speedup_claim=False, inference_admitted=False)


@contextmanager
def observe_load(model_class, arm, checkpoint_provenance, execution_guard, observation_callback):
    """Wrap the SDK's real load once; preserve returned objects and exceptions.

A future load-boundary successor must first perform the original fresh-process
gate and authenticate/import native dependencies, then install this observer
immediately around its existing model.load call. It CANNOT wrap the current
off/on run functions: _dependencies here imports native modules, which those
functions correctly reject at entry. No complete successor is provided here.
The callback runs after actual load_weights/initialize but BEFORE load_weights
returns. It must retain/validate the partial report and raise if the independent
graph-layout gate is unresolved. No second load or hidden build occurs here.
The caller authenticates SDK/source/checkpoint/env, reserves host/device space,
and binds reports only after actual producer exit and service drainage.
"""
    require(arm in ('off', 'on') and type(checkpoint_provenance) is dict and checkpoint_provenance,
            'Explicit arm and external checkpoint provenance required')
    require(callable(execution_guard) and callable(observation_callback), 'Owning guard/callback required')
    execution_guard()
    torch, builder_class, base_class = _dependencies()
    original_load = base_class.load_weights
    original_shard = builder_class.shard_checkpoint
    require(model_class.load_weights is original_load, 'Exact inherited original SDK load method required')
    state = dict(loader_calls=0, shard_calls=0, active=False, report=None)

    def shard(builder, *args, **kwargs):
        require(state['active'] and state['shard_calls'] == 0, 'Exactly one shard call inside actual load')
        execution_guard()
        require(not args and not kwargs and builder.tp_degree == 2
                and builder.start_rank_id == 0 and builder.local_ranks_size == 2,
                'Original in-memory TP2 sharding call required')
        state['shard_calls'] += 1
        result = original_shard(builder, *args, **kwargs)
        state['pre'] = tensor_map_record(result, torch, execution_guard)
        execution_guard()
        return result

    def load(model, *args, **kwargs):
        require(type(model) is model_class and state['loader_calls'] == 0 and not state['active'],
                'Exactly one actual expected model load')
        state['loader_calls'] += 1
        state['active'] = True
        try:
            execution_guard()
            result = original_load(model, *args, **kwargs)
            require(state['shard_calls'] == 1, 'Actual original sharding must have occurred')
            execution_guard()
            loaded = model.traced_model.nxd_model
            report = dict(schema=SCHEMA, arm=arm, checkpoint_provenance=deepcopy(checkpoint_provenance),
                          original_loader_calls=state['loader_calls'], original_shard_calls=state['shard_calls'],
                          pre_layout_shards=state['pre'],
                          loaded_cpu_views=tensor_map_record(loaded.weights, torch, execution_guard),
                          weight_loader_present=loaded.weight_loader is not None,
                          loaded_model_class=model_class.__module__ + '.' + model_class.__name__,
                          graph_weight_ABI_proven=False, layout_semantics_proven=False,
                          full_model_qualified=False, new_speedup_claim=False, inference_admitted=False)
            state['report'] = deepcopy(report)
            execution_guard()
            observation_callback(deepcopy(report))
            execution_guard()
            return result
        finally:
            state['active'] = False

    base_class.load_weights = load
    builder_class.shard_checkpoint = shard
    try:
        yield state
        require(state['loader_calls'] == state['shard_calls'] == 1 and state['report'] is not None,
                'Context completed without one full actual load observation')
    finally:
        # Restore both hooks even when sharding, hashing, callback or driver fails.
        base_class.load_weights = original_load
        builder_class.shard_checkpoint = original_shard
