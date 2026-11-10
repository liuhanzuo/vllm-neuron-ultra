"""Independent bounded byte verification for Stage A; source component only.

No writer implementation, tensors, paths, native libraries or runtime admission
is imported. Caller must authenticate both original metadata and fresh producer
metadata, open stable bounded files after actual producer exit, and separately
verify runtime/source/weight provenance. This verifier never supplies admission.
"""
from copy import deepcopy
import hashlib
import math


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fingerprint(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Exact SHA256')


def verify_archive(stream, archive, layout, *, max_bytes, expected_tensors=None,
                   selected_row_checks=None, guard=lambda: None):
    """Read every actual byte; compare full controls and target selected rows.

    expected_tensors maps labels to independently authenticated old descriptors.
    selected_row_checks maps labels to {rows, sha256}; rows are sorted original
    row indices, not reordered/sparse invented full inputs. Only selected bytes
    feed the row hash; every unselected byte still feeds the whole/tensor hash.
    max_bytes must be supplied before any payload is opened by the caller.
    """
    expected_tensors = {} if expected_tensors is None else expected_tensors
    selected_row_checks = {} if selected_row_checks is None else selected_row_checks
    require(type(max_bytes) is int and max_bytes > 0
            and set(archive) == {'complete', 'bytes', 'sha256', 'tensors'}
            and archive['complete'] is True and list(archive['tensors']) == list(layout)
            and set(expected_tensors) <= set(layout) and set(selected_row_checks) <= set(layout),
            'Complete archive and exact scope with explicit bound')
    fingerprint(archive['sha256'])
    offset = 0
    for label, shape_dtype in layout.items():
        shape, dtype = shape_dtype['shape'], shape_dtype['dtype']
        require(set(shape_dtype) == {'shape', 'dtype'} and isinstance(shape, list) and bool(shape)
                and all(type(n) is int and n > 0 for n in shape) and dtype in ('BF16', 'F32'),
                'Positive original tensor layout')
        size = math.prod(shape) * (2 if dtype == 'BF16' else 4)
        desc = archive['tensors'][label]
        require(set(desc) == {'shape', 'dtype', 'finite', 'offset', 'bytes', 'sha256'}
                and desc['shape'] == shape and desc['dtype'] == dtype and desc['finite'] is True
                and type(desc['offset']) is int and desc['offset'] == offset
                and type(desc['bytes']) is int and desc['bytes'] == size, 'Exact dense tensor extents')
        fingerprint(desc['sha256'])
        offset += size
        if label in expected_tensors:
            old = expected_tensors[label]
            require(old['shape'] == shape and old['dtype'] == dtype and old['finite'] is True
                    and old['bytes'] == size and old['sha256'] == desc['sha256'], 'Original full control descriptor')
        if label in selected_row_checks:
            check = selected_row_checks[label]
            require(set(check) == {'rows', 'sha256'} and len(shape) == 2
                    and isinstance(check['rows'], list) and bool(check['rows'])
                    and check['rows'] == sorted(set(check['rows']))
                    and all(type(row) is int and 0 <= row < shape[0] for row in check['rows']),
                    'Real selected original rows within full matrix')
            fingerprint(check['sha256'])
    require(type(archive['bytes']) is int and archive['bytes'] == offset <= max_bytes,
            'Complete known archive size within supplied limit')
    whole, read_bytes, row_verified = hashlib.sha256(), 0, 0
    for label, desc in archive['tensors'].items():
        part, selected_hash, consumed = hashlib.sha256(), hashlib.sha256(), 0
        word = 2 if desc['dtype'] == 'BF16' else 4
        check = selected_row_checks.get(label)
        intervals = [] if check is None else [
            (row*desc['shape'][1]*word, (row+1)*desc['shape'][1]*word) for row in check['rows']]
        while consumed < desc['bytes']:
            guard()
            request = min(65536, desc['bytes']-consumed)
            block = stream.read(request)
            require(type(block) is bytes and 0 < len(block) <= request and len(block) % word == 0,
                    'Complete aligned actual payload without oversized reads')
            # Independent exponent-bit check, including NaNs and signed infinities.
            for i in range(word-1, len(block), word):
                require(not ((block[i] & 0x7f) == 0x7f and (block[i-1] & 0x80)),
                        'Finite actual saved words')
            for begin, end in intervals:
                left, right = max(begin, consumed), min(end, consumed+len(block))
                if left < right:
                    selected_hash.update(block[left-consumed:right-consumed])
            part.update(block)
            whole.update(block)
            consumed += len(block)
            read_bytes += len(block)
        require(part.hexdigest() == desc['sha256'], 'Actual full tensor SHA256')
        if check is not None:
            require(selected_hash.hexdigest() == check['sha256'], 'Actual full target contains exact old selected rows')
            row_verified += len(check['rows'])
    require(stream.read(1) == b'' and read_bytes == archive['bytes']
            and whole.hexdigest() == archive['sha256'], 'No trailing bytes and complete archive SHA256')
    return dict(complete=True, bytes_verified=read_bytes, tensor_count=len(layout),
                full_original_controls=len(expected_tensors), selected_target_rows_verified=row_verified,
                original_source_weight_runtime_identity_verified_here=False,
                production_lifecycle_qualified=False, full_model_qualified=False, new_speedup_claim=False)


def stage_a_requirements(case):
    """Independent geometry for the three streams; no producer layout import."""
    rows = case['identity']['end']
    identity = case['identity']
    accepted = [(1, 0, 10, 1, 2060, 'full_causal'), (4, 1, 11, 1, 2060, 'full_causal'),
                (7, 0, 3, 1, 23, 'prefill'), (9, 1, 1, 1, 127, 'prefill'),
                (11, 0, 0, 1, 511, 'prefill'), (6179, 1, 3, 2, 31, 'slot_reuse_prefill')]
    require(identity['start'] == 0 and tuple(identity[k] for k in ('event', 'fixture', 'slot', 'epoch', 'end', 'phase')) in accepted,
            'One exact original complete wide call')
    chosen = case['selected_rows']
    require(chosen == sorted(set(chosen)) and bool(chosen) and all(type(i) is int and 0 <= i < rows for i in chosen)
            and (len(chosen) == 33 if identity['phase'] == 'full_causal' else chosen == [rows-1]),
            'Original selected control rows')
    visits = ([(i, 0) for i in range(16)] + [(i, loop) for loop in range(2) for i in range(16, 56)]
              + [(i, 0) for i in range(56, 72)])
    selected_layout = {f'layer-{p}-loop-{l}-{k}': dict(shape=[len(chosen), 2560], dtype='BF16')
                       for p, l in visits for k in ('input', 'residual_output')}
    target_layout = {f'layer-{p}-loop-{l}-{k}': dict(shape=[rows, 2560], dtype='BF16')
                     for p, l in ((32, 0), (32, 1), (56, 0)) for k in ('input', 'mixer_output', 'residual_output')}
    cache_ids = sorted([*range(0, 16, 2), *range(16, 56, 2), *range(88, 128, 2), *range(56, 72, 2)])
    original_layout = dict(hidden=dict(shape=[rows, 2560], dtype='BF16'),
                           logits=dict(shape=[len(chosen), 131072], dtype='F32'))
    original_layout.update({f'cache-{i}-{k}': dict(shape=[4, rows, 256], dtype='BF16')
                            for i in cache_ids for k in ('key', 'value')})
    require(list(case['selected_archive']['tensors']) == list(selected_layout)
            and list(case['original_archive']['tensors']) == list(original_layout),
            'All224 residual and114 original controls present')
    return dict(target=dict(layout=target_layout, selected_row_checks={
        label: dict(rows=list(chosen), sha256=case['selected_archive']['tensors'][label]['sha256'])
        for label in target_layout if not label.endswith('-mixer_output')}),
        selected=dict(layout=selected_layout, expected_tensors=deepcopy(case['selected_archive']['tensors'])),
        original=dict(layout=original_layout, expected_tensors=deepcopy(case['original_archive']['tensors'])))
