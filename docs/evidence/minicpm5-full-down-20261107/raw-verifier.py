"""Independent saved-logits audit; ROOT MUST LAUNCH UNDER ITS SERIAL OWNER.

This is a worker, not an execution spec, admission controller or launcher.
Imports are stdlib-only until main authenticates the root exit and references.
Only 56 finite output tensors and the 16.7 MB original HF tensor file are read.
No model, weight checkpoint, device, cache tensor or production validator loads.

V2 recreates only the two complete original output layouts admitted by JSON.
Contiguous and original-layout metric results are both retained for all frames.
A mismatch never relaxes the original metric equality or numerical thresholds.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
PLAN = Q / 'minicpm5-full-down-r1-20261107-input-plan.json'
PLAN_SHA = '8d7b6b76cc9af0c638e1cb97e827e2817279ed5953f93d52c0f136f9c5548a8a'
REVIEW = Q / 'minicpm5-full-down-independent-metadata-review-20261107-v1.json'
REVIEW_SHA = '44596ec541ca67e0f3b7a9d925ddfe538e62dd7825abcd153af4bcd575710ce9'
ARMS = ('strict_anchor_s0', 'strict_layer0_full_bf16_down')
OUT = ROOT / 'minicpm5-original32-full-down-cpu-r1-20261107'


def check(value, message):
    if not value:
        raise ValueError(message)


def read_json(path, sha):
    path = Path(path)
    check(path.is_absolute() and path.suffix == '.json', 'JSON path')
    with path.open('rb') as stream:
        raw = stream.read(8 * 2**20 + 1)
    check(len(raw) <= 8 * 2**20 and hashlib.sha256(raw).hexdigest() == sha, 'Bounded exact JSON')
    return json.loads(raw)


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def load_authenticated(torch, path, sha, limit, expected_size=None, expected_identity=None):
    """Hash exact opened bytes before weights_only deserialization from memory."""
    path = Path(path)
    check(path.suffix == '.pt', 'Only saved PT logits')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        check(0 < before.st_size <= limit, 'Finite raw file size')
        if expected_size is not None:
            check(before.st_size == expected_size, 'Declared output file size')
        if expected_identity is not None:
            check(str(path) == expected_identity['realpath'], 'Exact HF path')
            check(fingerprint(before) == tuple(expected_identity[k] for k in
                ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')), 'Frozen HF identity')
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    check(fingerprint(before) == fingerprint(after), 'Opened raw identity stable')
    check(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == sha, 'Exact raw bytes')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return value, dict(path=str(path), sha256=sha, bytes=len(raw), device=before.st_dev, inode=before.st_ino)


def tensor_sha(torch, value):
    check(value.device.type == 'cpu', 'CPU tensor only')
    return hashlib.sha256(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def metrics(torch, got, reference):
    """Independent formulas, full original 130560 values with one CPU thread."""
    check(got.shape == reference.shape == (130560,), 'Full vocabulary, no slicing')
    check(got.dtype == reference.dtype == torch.float32, 'Original FP32 logits')
    check(bool(torch.isfinite(got).all()) and bool(torch.isfinite(reference).all()), 'Finite logits')
    delta = got - reference
    p_log = torch.log_softmax(reference, dim=0)
    q_log = torch.log_softmax(got, dim=0)
    top_values = reference.topk(2).values
    reference_id, actual_id = int(reference.argmax()), int(got.argmax())
    result = dict(max_abs_error=float(delta.abs().max()), mean_abs_error=float(delta.abs().mean()),
        cosine=float(torch.nn.functional.cosine_similarity(got, reference, dim=0)),
        kl_reference_to_neuron=float((p_log.exp() * (p_log - q_log)).sum()),
        argmax_match=actual_id == reference_id,
        reference_margin=float(top_values[0] - top_values[1]), reference_top1=reference_id,
        neuron_top1=actual_id)
    check(all(math.isfinite(v) for v in result.values()), 'Finite calculated metrics')
    return result


def criteria(values):
    conditions = [('cosine', values['cosine'] >= .995),
        ('KL', values['kl_reference_to_neuron'] <= .02),
        ('max_abs', values['max_abs_error'] <= 2),
        ('high_margin_argmax', values['reference_margin'] <= .5 or values['argmax_match'])]
    return [name for name, passed in conditions if not passed]


def layout_contract(metadata):
    """Pure finite shape/stride proof; no native operations or payload access."""
    check(set(metadata) == {'shape', 'stride', 'offset', 'dtype', 'sha256'}, 'Complete original output metadata')
    shape, stride = metadata['shape'], metadata['stride']
    check(all(type(v) is int for v in [*shape, *stride, metadata['offset']]), 'Integer original layout')
    check((shape, stride) in (([1, 1, 130560], [1, 1, 1]),
                             ([4, 1, 130560], [1, 4, 4])), 'Only two frozen original layouts')
    check(metadata['offset'] == 0 and metadata['dtype'] == 'torch.float32', 'Zero-offset original FP32 layout')
    check(re.fullmatch('[0-9a-f]{64}', metadata['sha256']) is not None, 'Original logical value digest')
    # Prefill addresses are token. Decode addresses are row + 4*token,
    # so the four row residues modulo4 partition the complete storage span.
    elements = shape[0] * shape[1] * shape[2]
    storage_elements = 1 + sum((size-1)*step for size, step in zip(shape, stride))
    check(elements == storage_elements and storage_elements * 4 <= 2 * 2**20,
          'Complete nonoverlapping finite layout span')
    return dict(shape=shape, stride=stride, offset=0, dtype='torch.float32',
                storage_elements=storage_elements, storage_bytes=storage_elements*4,
                nonoverlap_proof='token' if shape[0] == 1 else 'row+4*token; distinct row residues modulo4')


def differences(actual, expected):
    check(set(actual) == set(expected), 'Same metric fields')
    return [dict(field=key, actual=actual[key], expected=expected[key],
                 delta=None if type(actual[key]) is bool else actual[key]-expected[key])
            for key in actual if actual[key] != expected[key]]


def main(output):
    check(sys.dont_write_bytecode and __debug__, 'No bytecode; checks enabled')
    check(output.parent == Q and output.name.startswith('minicpm5-full-down-independent-raw-') and
          output.suffix == '.json' and not output.exists(), 'Fresh bounded report path')
    first_mismatch_path = output.with_name(output.stem + '-first-mismatch.json')
    check(not first_mismatch_path.exists(), 'Fresh first-mismatch evidence path')
    plan = read_json(PLAN, PLAN_SHA)
    review = read_json(REVIEW, REVIEW_SHA)
    check(review['metadata_checks_passed'] is True and review['raw_payload_read_hash_stat'] is False,
          'Accepted independent metadata review')
    root_exit = read_json(review['actual_exit']['path'], review['actual_exit']['sha256'])
    check(root_exit['reviewer'] == '/root' and root_exit['actual_exit_observed'] is True and
          root_exit['actual_exit_code'] == 0 and root_exit['tool_chunk_id'] == 'd9e62d' and
          root_exit['actual_execution_handle'] == {'kind': 'tool_session', 'session_id': 32830},
          'Real root-observed original completion')
    check(root_exit['all_four_processes_gone'] is root_exit['cgroup_empty'] is True,
          'Prior root physical drain evidence')
    check(plan['output_root'] == str(OUT), 'Exact experiment output')
    refs = {r['path']: r['sha256'] for r in review['source_references']}
    records = []
    for rank in (0, 1):
        for arm in ARMS:
            for ci in range(14):
                path = OUT / f'cpu-rank{rank}' / arm / f'call-{ci:02d}' / 'result.json'
                call = read_json(path, refs[str(path)])
                check(call['complete'] is True and call['rank'] == rank and call['arm'] == arm and
                      call['call'] == plan['calls'][ci], 'Original call identity')
                payload = call['payload']
                check(payload['path'] == str(path.parent / 'output.pt'), 'Exact output payload path')
                check(0 < payload['bytes'] <= 3 * 2**20 and re.fullmatch('[0-9a-f]{64}', payload['sha256']),
                      'Bounded hashed output')
                layout_contract(call['output_metadata'])
                records.append((rank, arm, ci, call))
    check(len(records) == 56 and len({c['payload']['path'] for _, _, _, c in records}) == 56,
          'Exactly 56 current logits files')
    hf_ref = plan['hf_reference']
    check(hf_ref['path'] == str(ROOT / 'minicpm5-baseline-full-v4/hf-reference.pt') and
          hf_ref['sha256'] == '6cd455e1d9f37c935981f056ba6c33733acf3079faeb8a2cf91b0c030cfe36dd',
          'Original HF endpoint only')
    # Native import occurs only here. Root is responsible for the real current
    # serial lease, service resource limits and observed owner/actual exit.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw_references = []
    hf, raw_ref = load_authenticated(torch, hf_ref['path'], hf_ref['sha256'], 32 * 2**20,
        expected_identity=hf_ref['identity'])
    raw_references.append(raw_ref)
    check(len(hf) == 8 and [f['id'] for f in hf] == list(range(8)), 'All eight original HF fixtures')
    for frozen, original in zip(plan['fixtures'], hf):
        check(all(original[k] == frozen[k] for k in ('id', 'input_ids', 'continuation_ids')), 'Original fixtures')
        check(len(original['logits']) == 4, 'Four steps per fixture')
        for value in original['logits']:
            check(value.shape == (130560,) and value.dtype == torch.float32 and value.device.type == 'cpu' and
                  bool(torch.isfinite(value).all()), 'All 32 original HF vectors')
    rows, layouts, rank_results = [], [], {}
    all_metric_comparisons, metric_mismatches = [], []
    first_mismatch_ref = None
    for rank, arm, ci, call in records:
        payload = call['payload']
        saved, raw_ref = load_authenticated(torch, payload['path'], payload['sha256'], 3 * 2**20,
            expected_size=payload['bytes'])
        raw_references.append(raw_ref)
        check(type(saved) is dict and set(saved) == {'output'}, 'Exact output envelope')
        value = saved['output']
        check(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == 'cpu', 'Exact CPU FP32 output')
        check(list(value.shape) == call['output_metadata']['shape'] == [len(call['physical_rows']), 1, 130560], 'Complete output shape')
        check(value.is_contiguous() and value.storage_offset() == 0 and bool(torch.isfinite(value).all()), 'Saved clone is contiguous and finite')
        digest = tensor_sha(torch, value)
        check(digest == call['output_metadata']['sha256'], 'Saved full tensor matches original logical byte digest')
        contract = layout_contract(call['output_metadata'])
        restored = torch.empty_strided(tuple(contract['shape']), tuple(contract['stride']),
                                       dtype=torch.float32, device='cpu')
        restored.copy_(value)
        check(list(restored.shape) == contract['shape'] and list(restored.stride()) == contract['stride'] and
              restored.storage_offset() == 0, 'Exact reconstructed original output layout')
        check(restored.untyped_storage().nbytes() == contract['storage_bytes'], 'Finite complete original storage span')
        check(bool(torch.equal(restored, value)) and tensor_sha(torch, restored) == digest,
              'Reconstructed layout retains every logical output value exactly')
        layouts.append(dict(rank=rank, arm=arm, call=ci, shape=list(value.shape), saved_stride=list(value.stride()),
            saved_offset=value.storage_offset(), original_stride=call['output_metadata']['stride'],
            original_offset=call['output_metadata']['offset'], reconstructed_stride=list(restored.stride()),
            logical_sha256=digest, saved_contiguous=True, original_layout_reconstructed=True,
            reconstruction_contract=contract, original_storage_object_or_pointer_recreated=False))
        derived = []
        for mapping, expected in zip(call['physical_rows'], call['HF_metrics']):
            check(expected['frame'] == mapping['frame'] and expected['physical_row'] == mapping['physical_row'], 'Exact scoring row map')
            frame = mapping['frame']
            original = hf[frame['fixture']]['logits'][frame['step']]
            contiguous_scores = metrics(torch, value[mapping['physical_row'], 0], original)
            scores = metrics(torch, restored[mapping['physical_row'], 0], original)
            failed = criteria(scores)
            comparison = dict(rank=rank, arm=arm, call=ci, frame=frame,
                physical_row=mapping['physical_row'], payload=payload,
                original_metadata=call['output_metadata'], expected_metrics=expected['metrics'],
                contiguous_metrics=contiguous_scores, original_layout_metrics=scores,
                contiguous_vs_expected=differences(contiguous_scores, expected['metrics']),
                original_layout_vs_expected=differences(scores, expected['metrics']),
                contiguous_vs_original_layout=differences(contiguous_scores, scores),
                expected_failed_criteria=expected['failed_criteria'], original_layout_failed_criteria=failed)
            all_metric_comparisons.append(comparison)
            if comparison['original_layout_vs_expected'] or failed != expected['failed_criteria']:
                if first_mismatch_ref is None:
                    with first_mismatch_path.open('x') as stream:
                        json.dump(dict(schema='minicpm5_full_down_first_original_layout_metric_mismatch_v2',
                            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            metadata_review={'path': str(REVIEW), 'sha256': REVIEW_SHA},
                            original_experiment_exit=review['actual_exit'], comparison=comparison,
                            exact_metric_requirement_unchanged=True, tolerance_added=False,
                            full_model_qualified=False, new_speedup_claim=False), stream, indent=2)
                        stream.write('\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    first_mismatch_ref = dict(path=str(first_mismatch_path),
                        sha256=hashlib.sha256(first_mismatch_path.read_bytes()).hexdigest())
                    print(json.dumps(dict(first_original_layout_mismatch=first_mismatch_ref,
                        rank=rank, arm=arm, call=ci, frame=frame)), flush=True)
                metric_mismatches.append(comparison)
            row = dict(rank=rank, arm=arm, call=ci, frame=frame, metrics=scores, failed_criteria=failed,
                hf_tensor_sha256=tensor_sha(torch, original))
            rows.append(row)
            derived.append(row)
        check(len(derived) == len(call['physical_rows']) == len(call['HF_metrics']), 'No omitted score rows')
        rank_results[(rank, arm, ci)] = (digest, [dict(r, rank=None) for r in derived])
        del saved, value, restored
    check(len(rows) == 128 and len(raw_references) == 57, 'Both ranks, both arms, all 32 frames each')
    for arm in ARMS:
        for ci in range(14):
            check(rank_results[(0, arm, ci)] == rank_results[(1, arm, ci)], 'Raw logits and all derived metrics match both ranks')
    summaries = {}
    for arm in ARMS:
        selected = [r for r in rows if r['rank'] == 0 and r['arm'] == arm]
        check([r['frame'] for r in selected] == plan['frames'], 'All 32 ordered original frames')
        failed = [r['frame']['index'] for r in selected if r['failed_criteria']]
        top1 = sum(r['metrics']['argmax_match'] for r in selected)
        summary_matches_metadata = (failed == review['summaries'][arm]['failed_frames'] and
                                    top1 / 32 == review['summaries'][arm]['argmax_match_fraction'])
        summaries[arm] = dict(frames=32, passed_frames=32-len(failed), failed_frames=failed,
            top1_matches=top1, numerical_gates_passed=not failed and top1 / 32 >= .9,
            original_metadata_summary_exact=summary_matches_metadata)
    exact = not metric_mismatches and all(s['original_metadata_summary_exact'] for s in summaries.values())
    report = dict(schema='minicpm5_full_down_independent_raw_logits_review_v2',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete=True, exact_raw_metric_reproduction=exact,
        exact_metric_requirement_unchanged=True, tolerance_added=False,
        original_layout_metric_mismatch_count=len(metric_mismatches),
        first_original_layout_metric_mismatch=metric_mismatches[0] if metric_mismatches else None,
        first_original_layout_metric_mismatch_file=first_mismatch_ref,
        contiguous_metric_mismatch_count=sum(bool(r['contiguous_vs_expected']) for r in all_metric_comparisons),
        contiguous_vs_original_layout_changed_rows=sum(bool(r['contiguous_vs_original_layout']) for r in all_metric_comparisons),
        all_metric_comparisons=all_metric_comparisons,
        prior_contiguous_worker_sha256='0554567e333f1216f6594461c22d650e8f294ef35f8609b4d7c79a4fffce8139',
        prior_attempt_root_cause_inferred=False,
        actual_original_experiment_exit=review['actual_exit'],
        metadata_review={'path': str(REVIEW), 'sha256': REVIEW_SHA},
        source={'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        torch_version=str(torch.__version__), threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        original_shape_stride_offset_reconstructed=True,
        original_storage_object_or_pointer_recreated=False, model_or_weights_loaded=False,
        raw_payload_files=57, output_logits_files=56, scored_rows=128, unique_arm_frames=64,
        raw_references=raw_references, layouts=layouts, rows=rows, summaries=summaries,
        remaining_gates=plan['remaining_gates'], full_model_qualified=False, new_speedup_claim=False,
        current_worker_self_exit_claimed=False)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'summaries': summaries, 'raw_files': 57, 'rows': 128,
        'exact_raw_metric_reproduction': exact,
        'original_layout_metric_mismatch_count': len(metric_mismatches),
        'contiguous_metric_mismatch_count': report['contiguous_metric_mismatch_count'],
        'first_original_layout_metric_mismatch': report['first_original_layout_metric_mismatch']}))
    check(exact, 'Original-layout exact metric reproduction failed; all comparisons and first mismatch retained in report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
