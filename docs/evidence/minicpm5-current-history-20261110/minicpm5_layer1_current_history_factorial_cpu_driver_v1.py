"""Original TP2 attention with official current normalized input and/or valid prior K/V."""
import hashlib
import os
from pathlib import Path
from research import minicpm5_layer1_decode_stream_v2 as stream
from research import minicpm5_layer1_official_shards_v2 as shards
from research import minicpm5_layer1_current_history_factorial_callable_v1 as native
from research import minicpm5_layer1_current_history_factorial_v1 as contract
from research import minicpm5_layer1_official_history_inputs_v1 as official_inputs
from research import minicpm5_layer1_current_history_factorial_controls_v1 as controls
import minicpm5_layer0_cpu_constructor_v1 as constructor
require = stream.require


def save_output(context, arm, ci, mode, raw, guard):
    guard(); context.guard('cpu')
    require(arm in stream.raw.ARMS and ci in stream.boundary.DECODE_CALLS
            and mode in contract.MODES and type(raw) is bytes and len(raw) == 16384, 'Complete bounded output')
    rank = context.permit['rank']; root = stream.payload_io.safe_path(context.plan['output_root'])
    require(rank in (0, 1) and root.is_dir() and root.stat().st_uid == os.getuid() == 1000, 'Owned output root')
    folder = root / ('cpu-rank%d' % rank) / arm / ('call-%02d' % ci) / mode
    folder.mkdir(parents=True, exist_ok=False); path = stream.payload_io.safe_path(folder/'attention.bf16')
    with path.open('xb') as f:
        require(f.write(raw) == len(raw), 'Full raw output write'); f.flush(); os.fsync(f.fileno())
    desc = dict(path=str(path), raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                identity=stream.payload_io.file_identity(path, path.lstat()))
    require(stream.payload_io.read_owned_payload(dict(desc, payload_bytes=len(raw),
        payload_sha256=desc['sha256']), path, guard) == raw, 'Independent saved-byte reread')
    return desc


def run_rank(context, torch, collector_plan, rank_result, audit, read_json, guard, native_bundle, previous_bundle):
    guard(); context.guard('cpu'); require(context._sources_checked, 'Frozen sources authenticated')
    rank = context.permit['rank']
    require(rank in (0, 1) and rank_result['rank'] == rank and torch.is_inference_mode_enabled()
            and torch.get_num_threads() == 1 and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() == 2 and torch.distributed.get_rank() == rank
            and torch.distributed.get_backend() == 'gloo', 'Original TP2 CPU inference geometry')
    data = official_inputs.load(context.plan, native_bundle, guard)
    inputs = stream.RankStream(collector_plan, rank_result, audit, read_json, guard)
    model, loading = constructor.build(rank, 'decode'); constructor.check_load_metadata(loading, rank)
    weights = shards.verify_model(context, torch, model, rank)
    require(weights['all_four_projection_shards_raw_exact'] is True, 'Actual original official weight shards')
    rows = []
    for item in inputs:
        case = item['case']; arm = case['arm']; ci = case['call']['index']
        donors, current, target, target_refs = contract.for_call(context.plan, data, case)
        previous = controls.for_call(context.plan, previous_bundle, case, guard)
        for mode in contract.MODES:
            guard(); bucket = constructor.select_bucket(model, case['bucket'])
            result, raw = native.replay(torch, model, case, mode, item['normalized_raw'], item['cache_raw'],
                donors, current, item['expected_attention_raw'], weights['hashes'], guard)
            require(result['complete'] is True and result['rank'] == rank and result['arm'] == arm
                    and result['call'] == case['call'] and result['mode'] == mode
                    and type(raw) is bytes and len(raw) == 16384
                    and hashlib.sha256(raw).hexdigest() == result['attention_output_sha256'], 'Actual complete output identity')
            if mode in ('own', 'restore'): require(raw == item['expected_attention_raw'], 'Own/restored full output exact')
            previous_control = controls.verify(previous, mode, raw)
            saved = save_output(context, arm, ci, mode, raw, guard)
            rows.append(dict(result=result, output=saved, bucket=bucket, previous_control=previous_control,
                independent_audit_payload_sha256=item['audited_call']['payload_sha256'],
                official_target_sha256=hashlib.sha256(target).hexdigest(), official_target_refs=target_refs))
    proof = inputs.finish(); guard()
    require(len(rows) == 90, 'All18 original calls x5 history modes')
    return dict(schema='minicpm5_layer1_current_history_factorial_cpu_driver_v1', complete=True, rank=rank,
        load_proof=loading, official_layer1_shards=weights, source_stream=proof, replay_calls=90,
        physical_rows=360, raw_output_bytes=1474560, records=rows, complete_original_model_constructed=True,
        full_model_forward_reexecuted=False, layer1_attention_reexecuted=True,
        all_own_history_outputs_exact=True, official_history_intervention_performed=True,
        official_current_intervention_performed=True, current_qkv_change_together=True,
        previous_real_controls_full_bytes_exact=True,
        attention_causality_proven=False, full_model_qualified=False, new_speedup_claim=False, self_exit_claimed=False)
