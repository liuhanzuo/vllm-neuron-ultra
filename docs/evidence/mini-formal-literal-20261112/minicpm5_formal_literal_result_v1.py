"""Bounded stdout references to original complete durable result artifacts.

No extra on-disk artifact and no partial successful result. Failure reports
explicitly distinguish an unavailable complete observer payload from a saved
one; they never manufacture a reference to a missing receipt.
"""
from research import minicpm5_formal_literal_codec_v1 as codec

SCHEMA = 'minicpm5_formal_literal_result_references_v1'
FAILED = 'minicpm5_formal_literal_result_unavailable_v1'
FILES = {'admission.json', 'phase-1-owner.json', 'phase-1-summary.json', 'phase-1-exit.json',
         'phase-1-memory-syscalls.jsonl', 'phase-1-observer-exit.json'}
RESULT_KEYS = {'terminal', 'observer', 'diagnostic_passed', 'full_model_qualified', 'new_speedup_claim', 'exit_code'}


def _result(value):
    codec.require(type(value) is dict and set(value) in (RESULT_KEYS, RESULT_KEYS | {'summary'})
                  and type(value['diagnostic_passed']) is bool and type(value['exit_code']) is int
                  and 0 <= value['exit_code'] <= 255
                  and value['full_model_qualified'] is False and value['new_speedup_claim'] is False,
                  'Complete original observed result fields')


def _unavailable(result, terminal_wire, observer_raw, refs):
    return dict(schema=FAILED, diagnostic_passed=False, exit_code=result['exit_code'],
        full_model_qualified=False, new_speedup_claim=False,
        terminal_wire=codec.parsed(terminal_wire), terminal_wire_sha256=codec.digest(terminal_wire),
        observer_complete_payload_available=False, observer_payload_sha256=codec.digest(observer_raw),
        observer_payload_bytes=len(observer_raw), saved_files=refs,
        reason='No durable complete observer/terminal pair; original in-memory observer is not promoted to saved evidence')


def future_stdout_budget(admission_wire, argv, literal, output_root):
    """Size both exact output schemas before spawn; these are not artifact refs.

    Max file bytes use the original 2MiB limit. Failed observer byte length uses
    the 64-bit Python length domain solely as a numeric-width bound; no observer
    bytes are fabricated or promoted to saved evidence.
    """
    refs = {name: dict(path=output_root + '/' + name, sha256='f' * 64, bytes=codec.FILE_CAP) for name in FILES}
    normal = dict(schema=SCHEMA, output_root=output_root, files=refs, has_summary=False,
        expanded_result_sha256='f' * 64, expanded_result_bytes=codec.FILE_CAP + codec.SUMMARY_CAP + 4096)
    terminal = codec.prospective_terminal(admission_wire, argv, literal)
    failed = _unavailable({'exit_code': 255}, codec.encode('exit', terminal, literal), b'', refs)
    failed['observer_payload_bytes'] = 2**63 - 1
    sizes = dict(stdout_reference_bound=len(codec.encoded(normal)),
                 stdout_unavailable_bound=len(codec.encoded(failed)))
    codec.require(max(sizes.values()) <= codec.SUMMARY_CAP, 'Both whole stdout schemas must fit before SDK spawn')
    return sizes


def reconstruct(report, read_raw, literal):
    """Read-only audit interface; every provided reference must verify in full."""
    codec.require(type(report) is dict and set(report) == {'schema', 'output_root', 'files', 'has_summary',
                  'expanded_result_sha256', 'expanded_result_bytes'} and report['schema'] == SCHEMA,
                  'Exact complete stdout reference schema; failed/unavailable is not decodable')
    refs = report['files']
    codec.require(type(refs) is dict and set(refs) <= FILES
                  and {'phase-1-exit.json', 'phase-1-observer-exit.json'} <= set(refs), 'Required durable terminal/observer')
    values = {}
    for name, ref in refs.items():
        codec.reference(ref)
        codec.require(ref['path'] == report['output_root'] + '/' + name and 'bytes' in ref,
                      'Only exact existing output artifact names, no arbitrary result pointer')
        raw = read_raw(ref)
        codec.require(type(raw) is bytes and len(raw) == ref['bytes'] and codec.digest(raw) == ref['sha256'],
                      'Whole durable artifact size and wire SHA256')
        if name in ('admission.json', 'phase-1-owner.json', 'phase-1-exit.json'):
            kind = {'admission.json': 'admission', 'phase-1-owner.json': 'owner', 'phase-1-exit.json': 'exit'}[name]
            values[name] = codec.decode(kind, raw, literal, wire_sha256=ref['sha256'])
        elif name != 'phase-1-memory-syscalls.jsonl':
            values[name] = codec.parsed(raw)
    observer = values['phase-1-observer-exit.json']
    codec.require(observer.pop('self_persistence_claimed', None) is False and 'exit_receipt_saved' not in observer,
                  'Original observer receipt does not claim its own persistence')
    observer['exit_receipt_saved'] = True
    terminal = values['phase-1-exit.json']
    passed = (terminal['returncode'] == 0 and observer['observation_complete'] is True
              and observer['trace_complete'] is True and observer['trace_saved'] is True
              and observer['failure'] is None)
    result = dict(terminal=terminal, observer=observer, diagnostic_passed=passed,
                  full_model_qualified=False, new_speedup_claim=False,
                  exit_code=0 if passed else terminal['returncode'] if 0 < terminal['returncode'] <= 255 else 1)
    codec.require(type(report['has_summary']) is bool and report['has_summary'] == passed,
                  'Summary reference only for complete successful original result')
    if passed:
        codec.require(set(refs) == FILES, 'Exactly the original six accounted successful artifacts')
        summary = values['phase-1-summary.json']
        codec.require(summary['diagnostic_passed'] is True and summary['stage'] == 'complete'
                      and summary['observation_and_error_payload_complete'] is True, 'Original complete successful import summary')
        result['summary'] = {key: refs['phase-1-summary.json'][key] for key in ('path', 'sha256')}
    raw = codec.encoded(result)
    codec.require(len(raw) == report['expanded_result_bytes'] and codec.digest(raw) == report['expanded_result_sha256'],
                  'Whole original result reconstructed exactly; no hidden omission')
    return result


def render(result, store, literal):
    _result(result)
    inventory = store.inventory()
    codec.require(set(inventory['names']) <= FILES, 'Original six-name output domain')
    refs = {name: store.reference(name) for name in inventory['names']}
    if {'phase-1-exit.json', 'phase-1-observer-exit.json'} <= set(refs):
        raw = codec.encoded(result)
        report = dict(schema=SCHEMA, output_root=str(store.root), files=refs,
                      has_summary='summary' in result, expanded_result_sha256=codec.digest(raw), expanded_result_bytes=len(raw))
        codec.require(reconstruct(report, store.read, literal) == result, 'Actual complete producer/consumer roundtrip')
    else:
        codec.require(result['diagnostic_passed'] is False and result['exit_code'] != 0,
                      'Missing durable receipt can never qualify a result')
        # The original object remains in adapter.observed/result. Explicitly
        # mark whole observer evidence unavailable; never call this a partial
        # reconstruction or create a fake complete receipt from a subset.
        observer_raw = codec.encoded(result['observer'])
        report = _unavailable(result, codec.encode('exit', result['terminal'], literal), observer_raw, refs)
    raw = codec.encoded(report)
    codec.require(len(raw) <= codec.SUMMARY_CAP, 'Original whole stdout16KiB cap; no chunking or truncation')
    return raw
