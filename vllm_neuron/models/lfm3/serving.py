# SPDX-License-Identifier: Apache-2.0
"""Pure admission for the LFM3 contiguous scheduler proxy and real slots.

The 24 attention-shaped scheduler specs budget B request blocks plus one null
block. They do not describe the 35 flattened hybrid states per TP rank.
"""
import os
from .identity import EXPECTED_MODEL, require, selected_scope, selected_variant


def is_lfm3(model_config):
    hf = getattr(model_config, "hf_config", None)
    return (getattr(hf, "model_type", None) == "lfm3" or
            "Lfm3ForCausalLM" in (getattr(hf, "architectures", None) or ()))


def validate_serving_config(config, *, before_platform=False):
    selected_variant()
    scope = selected_scope()
    batch, context = scope["decode_batch"], scope["context"]
    model, cache = config.model_config, config.cache_config
    scheduler, parallel = config.scheduler_config, config.parallel_config
    require(model.max_model_len == context and scheduler.max_num_seqs == batch,
            "LFM3 vLLM capacity/context must match the explicit candidate scope")
    require(str(model.dtype) in ("bfloat16", "torch.bfloat16"), "LFM3 requires BF16")
    require(cache.block_size == context, "LFM3 requires block_size == max_model_len")
    require(not cache.enable_prefix_caching and scheduler.enable_chunked_prefill is False,
            "LFM3 requires prefix caching and chunked prefill disabled explicitly")
    require(getattr(scheduler, "async_scheduling", None) is False,
            "LFM3 requires async_scheduling=False explicitly")
    require(not getattr(model, "quantization", None), "LFM3 quantization is unsupported")
    require(getattr(cache, "cache_dtype", "auto") in ("auto", "bfloat16"),
            "LFM3 cache dtype must preserve BF16")
    require(parallel.tensor_parallel_size == 2, "LFM3 requires TP2")
    for name in ("pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size",
                 "prefill_context_parallel_size"):
        require(getattr(parallel, name, 1) == 1, "LFM3 unsupported parallel mode: " + name)
    require(not getattr(parallel, "enable_expert_parallel", False), "LFM3 EP unsupported")
    for name in ("kv_transfer_config", "ec_transfer_config", "speculative_config", "lora_config"):
        require(getattr(config, name, None) is None, "LFM3 unsupported route: " + name)
    require(os.environ.get("DISABLE_NEURON_CUSTOM_SCHEDULER", "0") == "0",
            "LFM3 requires the contiguous Neuron scheduler")
    require(os.environ.get("NEURON_COMBINE_KV_ALLOCATIONS", "0") == "0",
            "LFM3 requires all 35 uncombined states per rank")
    blocks = cache.num_gpu_blocks_override
    if before_platform:
        expected = batch + int(getattr(cache, "_neuron_null_block_adjusted", False))
        require(blocks is None or blocks == expected, "LFM3 proxy block override differs from B")
        if blocks is None:
            cache.num_gpu_blocks_override = batch
    else:
        require(blocks == batch + 1, "LFM3 scheduler proxy requires B+1 blocks including null")
    return scope


def configure_neuron_defaults(defaults, model, cache, parallel, scheduler):
    scope = selected_scope()
    batch, context = scope["decode_batch"], scope["context"]
    require(model.max_model_len == context and scheduler.max_num_seqs == batch
            and parallel.tensor_parallel_size == 2 and cache.block_size == context,
            "LFM3 loader scope mismatch")
    # Only defaults are filled. Explicit overrides are applied afterwards and
    # validated by Lfm3InferenceConfig; unsupported overrides never disappear.
    defaults.update(max_batch_size=batch, tkg_batch_size=batch, ctx_batch_size=1,
                    max_length=context, max_context_length=context, seq_len=context,
                    apply_seq_ids_mask=True, allow_input_truncation=False,
                    on_device_sampling_config=None, async_mode=False, buckets=[context],
                    logical_nc_config=1, start_rank_id=0, local_ranks_size=2, world_size=2)


def state_shapes(capacity, context):
    shapes = []
    for kind in EXPECTED_MODEL["layer_types"]:
        shapes.extend([(capacity, 1024, 7)] if kind == "conv" else
                      [(capacity, 2, context, 128)] * 2)
    require(len(shapes) == 35, "LFM3 state ledger changed")
    return shapes


def validate_loaded_states(states, capacity, context):
    """Check all 70 runtime entries without reading device tensor payloads."""
    expected = state_shapes(capacity, context)
    require(len(states) == 2, "LFM3 loaded state must contain TP2 ranks")
    for rank in states:
        require(len(rank) == 35, "LFM3 requires exactly 35 uncombined states per rank")
        indexed = {}
        for name, tensor in rank.items():
            prefix, separator, index = name.rpartition("kv_mgr.past_key_values.")
            require(separator and index.isdigit(), "Unexpected LFM3 flattened state name: " + name)
            require(int(index) not in indexed, "Duplicate LFM3 state index")
            indexed[int(index)] = tensor
        require(set(indexed) == set(range(35)), "Incomplete LFM3 flattened state order")
        for index, shape in enumerate(expected):
            tensor = indexed[index]
            require(tuple(tensor.shape) == shape and str(tensor.dtype) in
                    ("bfloat16", "torch.bfloat16"), "LFM3 state shape/dtype differs")


def validate_proxy_cache(runner, config):
    scope = selected_scope()
    batch, context = scope["decode_batch"], scope["context"]
    require(config.num_blocks == batch + 1 and len(config.kv_cache_groups) == 1,
            "LFM3 requires one scheduler group and B+1 proxy blocks")
    group = config.kv_cache_groups[0]
    require(set(group.layer_names) == {f"layers.{i}.self_attn" for i in range(24)},
            "LFM3 scheduler proxy must cover all 24 layers")
    spec = group.kv_cache_spec
    require(spec.block_size == context and spec.num_kv_heads == 2 and
            spec.head_size == 128 and getattr(spec, "sliding_window", None) is None,
            "LFM3 contiguous proxy shape differs")
    application = runner.model.model
    for stage in (application.context_encoding_model, application.token_generation_model):
        validate_loaded_states(stage.model.nxd_model.state, batch, context)


def proxy_memory_bytes(config):
    scope = validate_serving_config(config)
    # Scheduler budget, deliberately not a measurement of HBM usage.
    return (scope["decode_batch"] + 1) * 24 * 2 * scope["context"] * 2 * 128 * 2


def _blocks(block_ids, capacity):
    require(len(block_ids) == 1 and len(block_ids[0]) == 1,
            "LFM3 requests require exactly one block in the single proxy group")
    block = block_ids[0][0]
    require(type(block) is int and 1 <= block <= capacity,
            "LFM3 request cannot use the null/out-of-range proxy block")
    return block


def validate_new_request(request, context, capacity):
    tokens = request.prompt_token_ids
    require(isinstance(tokens, (list, tuple)) and 1 <= len(tokens) <= context,
            "LFM3 prompt is empty, oversized or not token IDs; no truncation is allowed")
    require(all(type(token) is int and 0 <= token < EXPECTED_MODEL["vocab_size"] for token in tokens),
            "Invalid LFM3 prompt token ID")
    require(request.num_computed_tokens == 0, "LFM3 new request cannot resume a prefix")
    require(not getattr(request, "mm_features", None) and
            getattr(request, "lora_request", None) is None and
            getattr(request, "pooling_params", None) is None and
            getattr(request, "sampling_params", None) is not None,
            "LFM3 only supports text generation without LoRA")
    return _blocks(request.block_ids, capacity)


def preflight_step(runner, output):
    """Validate the entire step before *any* request, slot or batch mutation.

    The existing runner owns allocation. Intermittently unscheduled requests
    retain their slot; finished/aborted IDs free it before same-ID admission.
    """
    capacity, context = runner.max_num_reqs, runner.max_model_len
    mapping, free = runner.vllm_req_to_neuron_seq_id_mapping, runner.free_seq_ids
    require(len(set(mapping.values())) == len(mapping) and
            set(mapping.values()).isdisjoint(free) and
            set(mapping.values()) | set(free) == set(range(capacity)), "Corrupt LFM3 slot ledger")
    cached = output.scheduled_cached_reqs
    require(not getattr(output, "preempted_req_ids", None) and not cached.resumed_req_ids,
            "LFM3 preemption/resume has no hybrid state restoration implementation")
    for key in ("scheduled_spec_decode_tokens", "scheduled_encoder_inputs", "kv_connector_metadata",
                "ec_connector_metadata", "pending_structured_output_tokens"):
        require(not getattr(output, key, None), "LFM3 unsupported scheduler payload: " + key)
    require(not any(getattr(output, "num_common_prefix_blocks", ())), "LFM3 prefix reuse unsupported")
    finished = set(output.finished_req_ids)
    surviving = set(mapping) - finished
    new_ids = [req.req_id for req in output.scheduled_new_reqs]
    cached_ids = list(cached.req_ids)
    all_ids = new_ids + cached_ids
    require(len(set(all_ids)) == len(all_ids) and not (set(new_ids) & surviving),
            "Duplicate or already-live LFM3 request ID")
    require(set(cached_ids) <= surviving, "LFM3 cached request has no live slot")
    require(set(output.num_scheduled_tokens) == set(all_ids) and
            output.total_num_scheduled_tokens == sum(output.num_scheduled_tokens.values()),
            "Inconsistent LFM3 scheduler token accounting")
    require(len(surviving) + len(new_ids) <= capacity, "LFM3 request slots exhausted")
    used_blocks = set()
    for req_id in surviving:
        require(req_id in runner.requests, "LFM3 slot has no cached request state")
        block = _blocks(runner.requests[req_id].block_ids, capacity)
        require(block not in used_blocks, "Duplicate LFM3 request proxy block")
        used_blocks.add(block)
    for req in output.scheduled_new_reqs:
        block = validate_new_request(req, min(context, runner.max_prompt_length), capacity)
        require(block not in used_blocks, "LFM3 new request reuses a live proxy block")
        used_blocks.add(block)
        require(output.num_scheduled_tokens[req.req_id] == len(req.prompt_token_ids),
                "LFM3 chunked/partial prefill unsupported")
    require(len(cached.new_block_ids) == len(cached_ids) and
            len(cached.num_computed_tokens) == len(cached_ids), "Malformed LFM3 cached request arrays")
    for index, req_id in enumerate(cached_ids):
        state = runner.requests[req_id]
        position = len(state.prompt_token_ids) + len(state.output_token_ids) - 1
        require(bool(state.output_token_ids) and 0 <= position < context,
                "LFM3 cached request lacks output or exceeds its context")
        require(output.num_scheduled_tokens[req_id] == 1 and
                cached.num_computed_tokens[index] == position,
                "LFM3 only supports contiguous one-token decode")
        require(cached.new_block_ids[index] is None or
                cached.new_block_ids[index] == ([],) or
                cached.new_block_ids[index] == [[]], "LFM3 cached request cannot grow/replace proxy blocks")
