"""Explicit original/official current input x original/official history contract.

Pure bytes and metadata only. The caller owns admission and authentication of
the complete previous experiment, original SDK cases, and official native rows.
Changing the current normalized input changes the current Q, K and V together;
this is not a query-only intervention or a model qualification.
"""
import copy
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research import minicpm5_layer1_official_history_v1 as history
from research import minicpm5_layer1_official_history_inputs_v1 as native_inputs

require = boundary.require
MODES = ('own', 'official_current', 'official_history', 'official_both', 'restore')


def for_call(plan, native_data, case):
    """Read no data; select previous K/V donors and current normalized rows."""
    prior, target, target_refs = native_inputs.for_call(plan, native_data, case)
    current = []
    for frame in case['physical_frame_order']:
        row = native_data[frame['fixture'], frame['step']]
        require(row['record']['frame'] == frame, 'Current normalized source belongs to the current original frame')
        current.append(dict(record=row['record'], normalized_raw=row['targets'][boundary.NORMALIZED]))
    return prior, current, target, target_refs


def select(case, own_normalized_raw, own_cache_raw, prior_donors, current_donors, mode):
    """Return selected logical input, physical storage, full caches and proof.

    The original case is never changed or relabelled. Official per-row sources
    are retained separately, then packed into the actual original [1,4,4] stride.
    Current native records must come from the current frame; K/V come from its
    preceding frame. All five modes validate both sets of official source bytes.
    """
    require(mode in MODES, 'Explicit current/history factorial mode')
    use_current = mode in ('official_current', 'official_both')
    use_history = mode in ('official_history', 'official_both')
    history_mode = 'official_both' if use_history else ('restore' if mode == 'restore' else 'own')
    cache, history_proof = history.overlay(case, own_cache_raw, prior_donors, history_mode)
    normalized = case['normalized_input']
    require(normalized['name'] == boundary.NORMALIZED and normalized['metadata'] == boundary.NORM_META
            and normalized['logical_bytes'] == 16384 and normalized['restore_source_stride'] is True,
            'Exact complete original current-input contract')
    require(type(own_normalized_raw) is bytes and len(own_normalized_raw) == 16384
            and boundary.sha(own_normalized_raw) == normalized['logical_sha256']
            and history.finite_bf16(own_normalized_raw), 'Complete original finite BF16 current input')
    require(type(current_donors) is list and len(current_donors) == 4, 'All four current normalized native rows')
    rows = []; sources = []
    for slot, (frame, donor, previous) in enumerate(zip(case['physical_frame_order'], current_donors, prior_donors)):
        require(type(donor) is dict and set(donor) == {'record', 'normalized_raw'}, 'Exact current-donor fields')
        record = donor['record']; raw = donor['normalized_raw']
        require(record['schema'] == 'minicpm5_layer1_native_cache_frame_v1' and record['complete'] is True
                and record['frame'] == frame and frame['slot'] == slot
                and record['native_fixture_order'] == frame['fixture'] and record['step'] == frame['step']
                and record['cache_length'] == case['positions'][slot] + 1,
                'Current frame, fixture, slot and logical position match normalized source')
        require(record['original_reference'] == previous['record']['original_reference'],
                'Current normalized input and preceding history share official reference provenance')
        for field in ('full_logits_raw_exact', 'all213_selected_taps_raw_exact', 'official_bf16', 'history_unmodified'):
            require(record[field] is True, 'Exact official source guarantee ' + field)
        require(record['full_model_qualified'] is record['new_speedup_claim'] is False, 'Original diagnostic boundary')
        meta = record['tap_metadata'][boundary.NORMALIZED]
        require(meta == dict(source_shape=[1, 1, 2048], source_stride=[2048, 2048, 1],
            local_prediction_position=0, logical_prediction_position=case['positions'][slot], dtype='torch.bfloat16'),
            'Original official decode input-layernorm output and current position')
        entries = [e for e in record['tap_entries'] if e['name'] == boundary.NORMALIZED]
        require(len(entries) == 1, 'One explicit official current normalized tap')
        entry = entries[0]; descriptor = record['outputs']['taps']
        require(type(entry['offset']) is int and entry['offset'] >= 0 and entry['bytes'] == 4096
                and entry['offset'] + entry['bytes'] <= descriptor['raw_bytes']
                and entry['shape'] == [1, 1, 2048] and type(raw) is bytes and len(raw) == 4096
                and boundary.sha(raw) == entry['sha256'] and history.finite_bf16(raw),
                'Complete authenticated finite BF16 normalized source row')
        rows.append(raw)
        sources.append(dict(frame=copy.deepcopy(frame), stage=boundary.NORMALIZED,
            original_frame=copy.deepcopy(record['original_frame']), original_reference=copy.deepcopy(record['original_reference']),
            native_payload=copy.deepcopy(descriptor), tap_entry=copy.deepcopy(entry), source_metadata=copy.deepcopy(meta)))
    official_current = b''.join(rows)
    selected = official_current if use_current else own_normalized_raw
    packed = boundary.normalized_storage(selected, normalized['metadata'], boundary.sha(selected))
    require(len(selected) == len(packed) == 16384, 'Full four-row logical input and original physical storage')
    proof = dict(schema='minicpm5_layer1_current_history_factorial_contract_v1', rank=case['rank'], arm=case['arm'],
        call=copy.deepcopy(case['call']), mode=mode, original_call=copy.deepcopy(case['original_call']),
        own_control=mode in ('own', 'restore'), current_source='official' if use_current else 'own',
        history_source='official' if use_history else 'own', original_normalized_input=copy.deepcopy(normalized),
        original_normalized_sha256=boundary.sha(own_normalized_raw), official_normalized_sha256=boundary.sha(official_current),
        selected_normalized_sha256=boundary.sha(selected), selected_storage_sha256=boundary.sha(packed),
        selected_layout=copy.deepcopy(boundary.NORM_META), official_current_sources=sources,
        history_intervention=history_proof, current_qkv_change_together=use_current,
        full_model_qualified=False, new_speedup_claim=False)
    return selected, packed, cache, proof
