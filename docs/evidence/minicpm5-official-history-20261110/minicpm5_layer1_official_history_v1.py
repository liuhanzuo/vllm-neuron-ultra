"""Pure-byte official K/V overlays on authenticated complete SDK histories.

No I/O, model import or runtime admission. The caller must independently bind
the original decode case and all native records to their successful collection
and audit exits before supplying bytes. No source metadata is relabelled.
"""
import copy
import struct
from research import minicpm5_layer1_decode_boundary_v1 as boundary

require = boundary.require
MODES = ('own', 'official_key', 'official_value', 'official_both', 'restore')
CACHE_BYTES = 4 * 4224 * 128 * 2
SLOT_BYTES = 4224 * 128 * 2


def finite_bf16(data):
    return type(data) is bytes and len(data) % 2 == 0 and all(
        (word & 0x7f80) != 0x7f80 for (word,) in struct.iter_unpack('<H', data))


def overlay(case, own_cache_raw, donors, mode):
    """Return full immutable stores and an explicit official-source contract.

    donors is in physical slot order. Every entry holds its authenticated native
    frame record and both complete two-head raw payloads as key_raw/value_raw.
    Only the selected head's valid prior prefix can replace original SDK bytes.
    """
    require(mode in MODES, 'Explicit official history mode')
    require(case['schema'] == 'minicpm5_layer1_decode_boundary_v1'
            and type(case['rank']) is int and case['rank'] in (0, 1)
            and case['arm'] in boundary.ARMS and case['call']['index'] in boundary.DECODE_CALLS
            and case['call']['phase'] == 'decode' and case['bucket'] in (128, 2048)
            and case['original_layer_index'] == 1 and case['seq_ids'] == [0, 1, 2, 3]
            and case['strict_mask_kind'] == 'prior_positions_only', 'Original TP2 layer1 decode case')
    require(case['full_model_qualified'] is case['new_speedup_claim'] is False,
            'Original case retains qualification boundaries')
    frames = case['physical_frame_order']; positions = case['positions']
    require(type(frames) is list and len(frames) == 4 and type(positions) is list and len(positions) == 4
            and [f['slot'] for f in frames] == list(range(4))
            and len({f['fixture'] for f in frames}) == len({f['index'] for f in frames}) == 4,
            'Four distinct original fixtures in physical slot order')
    require(type(donors) is list and len(donors) == 4 and type(own_cache_raw) is dict
            and set(own_cache_raw) == set(boundary.CACHE_NAMES), 'All four donors and full original K/V')
    require([v['name'] for v in case['caches']] == list(boundary.CACHE_NAMES), 'Original K/V source descriptors')
    input_hashes = {}
    for name, descriptor in zip(boundary.CACHE_NAMES, case['caches']):
        raw = own_cache_raw[name]; metadata = descriptor['complete_source_metadata']
        require(type(raw) is bytes and len(raw) == CACHE_BYTES == descriptor['complete_original_bytes']
                and metadata == dict(name=name, shape=boundary.CACHE_SHAPE,
                    stride=boundary.CACHE_STRIDE, offset=0, dtype='torch.bfloat16', sha256=boundary.sha(raw)),
                'Exact full original input store, layout and digest')
        require(descriptor['view_must_retain_complete_storage'] is True
                and descriptor['sdk_view']['shape'] == [4, 1, case['bucket'], 128]
                and descriptor['sdk_view']['stride'] == boundary.CACHE_STRIDE
                and descriptor['sdk_view']['offset'] == 0
                and descriptor['sdk_view']['dtype'] == 'torch.bfloat16', 'Original physical cache view')
        require(finite_bf16(raw), 'Every original SDK cache word finite')
        input_hashes[name] = metadata['sha256']
    replace = {'own': (), 'restore': (), 'official_key': ('key',),
               'official_value': ('value',), 'official_both': ('key', 'value')}[mode]
    buffers = {name: bytearray(data) for name, data in own_cache_raw.items()}
    sources = []
    for slot, (frame, donor, length) in enumerate(zip(frames, donors, positions)):
        require(type(length) is int and 0 < length < case['bucket'] and frame['step'] in (1, 2, 3)
                and type(frame['index']) is int and 4 <= frame['index'] < 32, 'Original valid prior extent')
        require(type(donor) is dict and set(donor) == {'record', 'key_raw', 'value_raw'}, 'Exact donor argument fields')
        record = donor['record']; previous = record['frame']
        require(record['schema'] == 'minicpm5_layer1_native_cache_frame_v1' and record['complete'] is True
                and previous == dict(index=frame['index'] - 4, fixture=frame['fixture'], slot=slot, step=frame['step'] - 1)
                and record['native_fixture_order'] == frame['fixture'] and record['step'] == previous['step']
                and record['cache_length'] == length, 'Same fixture and physical slot at the preceding native frame')
        for field in ('full_logits_raw_exact', 'all213_selected_taps_raw_exact', 'official_bf16', 'history_unmodified'):
            require(record[field] is True, 'Original official native guarantee ' + field)
        require(record['full_model_qualified'] is record['new_speedup_claim'] is False, 'Native source stays diagnostic')
        source = dict(current_frame=copy.deepcopy(frame), previous_native_frame=copy.deepcopy(previous),
            slot=slot, head=case['rank'], valid_positions=length,
            native_original_frame=copy.deepcopy(record['original_frame']),
            original_reference=copy.deepcopy(record['original_reference']), caches={})
        width = length * 128 * 2; start = slot * SLOT_BYTES
        for kind, name in zip(('key', 'value'), boundary.CACHE_NAMES):
            raw = donor[kind + '_raw']; metadata = record['cache_metadata'][kind]; desc = record['outputs'][kind]
            require(type(raw) is bytes and len(raw) == 2 * width == desc['raw_bytes']
                    and boundary.sha(raw) == desc['sha256'], 'Every byte of both official native heads')
            require(metadata['shape'] == [1, 2, length, 128] and metadata['dtype'] == 'torch.bfloat16'
                    and len(metadata['stride']) == 4 and all(type(v) is int and v >= 0 for v in metadata['stride'])
                    and type(metadata['offset']) is int and metadata['offset'] >= 0, 'Full original BF16 native geometry')
            require(finite_bf16(raw), 'Every official native cache word finite')
            selected = raw[case['rank'] * width:(case['rank'] + 1) * width]
            require(len(selected) == width, 'Complete original rank-local head prefix')
            if kind in replace: buffers[name][start:start + width] = selected
            source['caches'][kind] = dict(native_output=copy.deepcopy(desc),
                selected_head_sha256=boundary.sha(selected), destination_cache=name,
                destination_offset=start, bytes=width, replaced=(kind in replace))
        sources.append(source)
    result = {name: bytes(value) for name, value in buffers.items()}
    for kind, name in zip(('key', 'value'), boundary.CACHE_NAMES):
        require(len(result[name]) == CACHE_BYTES, 'Complete unchanged storage extent')
        for slot, length in enumerate(positions):
            start = slot * SLOT_BYTES; stop = start + length * 256
            require(result[name][stop:start + SLOT_BYTES] == own_cache_raw[name][stop:start + SLOT_BYTES],
                    'All unused current, future and padded history bytes unchanged')
            if kind not in replace:
                require(result[name][start:stop] == own_cache_raw[name][start:stop], 'Nonselected valid prefix unchanged')
    if not replace: require(result == own_cache_raw, 'Own and restore preserve every original cache byte')
    proof = dict(schema='minicpm5_layer1_official_history_contract_v1', rank=case['rank'], arm=case['arm'],
        call=copy.deepcopy(case['call']), mode=mode, original_call=copy.deepcopy(case['original_call']),
        own_control=not bool(replace), replaced_kinds=list(replace),
        input_cache_sha256=input_hashes, output_cache_sha256={k: boundary.sha(v) for k, v in result.items()},
        donors=sources, all_unselected_bytes_preserved=True, official_previous_frame_only=True,
        full_model_qualified=False, new_speedup_claim=False)
    return result, proof
