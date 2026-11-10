"""Authenticated native cache donors and full four-row official target outputs."""
from pathlib import Path
import hashlib
from research import minicpm5_layer1_payload_io_v1 as payload
from research import minicpm5_layer1_decode_boundary_v1 as boundary
require = boundary.require


def load(plan, bundle, guard):
    require(callable(guard), 'Separate serial admission guard'); guard()
    require(bundle['binding'] == plan['layer1_official_history'], 'Exact admitted native completion')
    result = {}
    for row in bundle['native']['records']:
        guard(); frame = row['frame']; arrays = {}
        for name in ('key', 'value', 'taps'):
            desc = row['outputs'][name]
            require(plan['inputs'][desc['path']] == dict(sha256=desc['sha256'], identity=desc['identity']),
                    'Native raw data included in frozen input identities')
            arrays[name] = payload.read_owned_payload(dict(desc, payload_bytes=desc['raw_bytes'],
                payload_sha256=desc['sha256']), Path(desc['path']), guard)
        target = {}
        for name in ('layers.1.input_layernorm', 'layers.1.self_attn'):
            matches = [e for e in row['tap_entries'] if e['name'] == name]
            require(len(matches) == 1, 'One original native tap')
            e = matches[0]; raw = arrays['taps'][e['offset']:e['offset']+e['bytes']]
            require(e['shape'] == [1, 1, 2048] and len(raw) == 4096
                    and hashlib.sha256(raw).hexdigest() == e['sha256'], 'Complete official current selected row')
            target[name] = raw
        result[frame['fixture'], frame['step']] = dict(record=row, key_raw=arrays['key'], value_raw=arrays['value'], targets=target)
    require(len(result) == 32, 'All official native frames loaded'); guard()
    return result


def for_call(plan, native_data, case):
    require(case['call'] == plan['calls'][case['call']['index']]
            and case['call']['index'] in boundary.DECODE_CALLS, 'Original decode call only')
    donors = []; targets = []; refs = []
    for slot, frame in enumerate(case['physical_frame_order']):
        require(frame['slot'] == slot and frame in plan['frames'], 'Every original physical slot')
        previous = native_data[frame['fixture'], frame['step']-1]; current = native_data[frame['fixture'], frame['step']]
        require(current['record']['frame'] == frame, 'Current official target frame')
        donors.append({key: previous[key] for key in ('record', 'key_raw', 'value_raw')})
        targets.append(current['targets'][boundary.ATTENTION])
        refs.append(dict(frame=frame, native_frame=current['record']['original_frame'],
            native_payload=current['record']['outputs']['taps'],
            attention_sha256=hashlib.sha256(targets[-1]).hexdigest()))
    target = b''.join(targets)
    require(len(donors) == 4 and len(target) == 16384, 'Complete original four-row target')
    return donors, target, refs
