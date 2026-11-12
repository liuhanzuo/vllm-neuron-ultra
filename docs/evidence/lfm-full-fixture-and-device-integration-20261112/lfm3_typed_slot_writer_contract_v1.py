"""Source-only candidate contract. No runtime admission is available in v1."""
CAPACITIES = (4, 8, 16)
DTYPES = ('bfloat16', 'float32')
WIDTHS = (128, 256, 384, 7168, 16256, 16384, 16512, 32768, 65536,
          131072, 262144, 524288, 2097152)
SOURCE_ONLY_LIMITS = dict(cpu=7, nice=19, address_space_bytes=256*2**20,
    cpu_seconds=60, wall_seconds=120, total_output_bytes=2*2**20, per_file_bytes=512*2**10,
    native_imports=False,simulator=False,compile=False,model=False,device=False)
RUNTIME_ADMITTED = False
COMPILE_PEAK_BYTES = None
COMPILE_ARTIFACT_CAP_BYTES = None
PREDICTED_SPEEDUP = None
CRITERIA = {'cosine_min':.995,'kl_max_nats':.02,'max_abs_error_max':2.,
    'argmax_match_fraction_min':.9,'all_reference_margins_gt_0.5_must_match':True}


def forbid_runtime_launch():
    raise RuntimeError('Source-only candidate: compiler peak/artifact budgets and independent runtime admission are unknown; launch prohibited')


def selected_rows(capacity, slots):
    """Integer control contract; last valid writer wins, OOB stays inactive."""
    assert capacity in CAPACITIES and 1 <= len(slots) <= capacity
    assert all(type(slot) is int for slot in slots)
    selected = [None] * capacity
    for row, slot in enumerate(slots):
        if 0 <= slot < capacity:
            selected[slot] = row
    return selected


def bit_patterns(dtype):
    if dtype == 'bfloat16':
        return (0x0000,0x8000,0x0001,0x007f,0x0080,0x3f80,0xbf80,0x7f7f,0xff7f,
            0x7f80,0xff80,0x7fc1,0x7fff,0xffc3,0x7f81,0xff81)
    assert dtype == 'float32'
    return (0x00000000,0x80000000,0x00000001,0x007fffff,0x00800000,0x3f800000,
        0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,
        0x7fffffff,0xffc00003,0x7f800001,0xff800001)


def logical_state_budget(capacity):
    assert capacity in CAPACITIES
    # Pinned prior full-model contract; one logical cache pool, one TP rank.
    return dict(per_rank_single_cache_pool_bytes=23255040*capacity,
        tp2_single_pool_bytes=46510080*capacity,total_runtime_hbm_unknown=True,
        graph_scratch_unknown=True,compiler_peak_unknown=True,artifact_growth_unknown=True)


def required_gates():
    return dict(primitive=dict(regression_cpu_cases=1648,extended_cpu_cases=2068,
        raw_device_graphs=4,raw_device_checks=36,raw_bf16_all_uint16_words_required=True,
        regression_device_graphs=9,regression_device_checks=43,
        regression_alias_graphs=12,regression_alias_checks=60,
        extended_device_graphs=42,extended_device_checks=2520,
        extended_alias_graphs=24,extended_alias_checks=192),
        added=dict(typed_raw_bit_patterns_both_dtypes=True,production_dispatch_one_row_original=True,
            slot_int64_oob_normalized_before_int32=True,fp32_alias_companion_required=True,
            every_store_exactly_once=True,input_buffers_unchanged=True,
            no_dtype_views_or_payload_arithmetic=True),
        b4=dict(numeric_frames=32,cold_frames=96,long_steps=1529,long_frames=84,quality_cases=96),
        b8=dict(numeric_frames=32,cold_frames=200,long_steps=1529,long_frames=168,quality_cases=96),
        b16=dict(numeric_frames=64,cold_frames=400,long_steps=1529,long_frames=336,quality_cases=96),
        lifecycle=dict(fresh_cold_processes=7,states=70,fill_roundtrips=140,zero_roundtrips=140,snapshots=5,
            all_slots_and_original_buckets=True,request_slot_reuse=True,actual_must_alias_state=True),
        comparisons=dict(controls=['original_b4','selected_writer_v4_b4'],candidates=['typed_writer_b4','typed_writer_b8','typed_writer_b16'],
            workloads=5,rounds=3,same_load_concurrency=[1,4],separate_capacity_concurrency=[8,16],
            original_output_budgets=True,original_sampler=True,original_criteria=True,
            measured_requests_per_variant=704,measured_output_tokens_per_variant=189952,
            warmup_requests_per_variant=16,greedy_text_pairs_per_candidate=384))
