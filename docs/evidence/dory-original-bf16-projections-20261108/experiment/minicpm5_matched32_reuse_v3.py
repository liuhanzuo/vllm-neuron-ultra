"""Authenticate completed v2 control metadata; never recapture or import a model."""
from copy import deepcopy
from pathlib import Path
import minicpm5_matched32_protocol_v3 as p

PRIOR = p.ROOT / 'minicpm5-matched32-v2'
PRIOR_DATA = Path('/mnt/nvme/minicpm5-matched32-v2')


def inputs():
    """Read small JSON and stat existing payloads. Full hashes belong to admission."""
    sources = {}; payloads = {}
    def metadata(path, digest=None):
        path = p.no_symlink(path)
        actual = p.text_sha(path)
        assert digest is None or actual == digest, str(path)
        assert str(path) not in sources or sources[str(path)] == actual
        sources[str(path)] = actual
        return p.read(path)
    def payload(path, digest, identity):
        path = p.no_symlink(path)
        assert p.identity(path) == identity
        assert isinstance(digest, str) and len(digest) == 64
        row = dict(sha256=digest, identity=identity)
        assert str(path) not in payloads or payloads[str(path)] == row
        payloads[str(path)] = row

    prior_plan = metadata(PRIOR / 'plan.json')
    prior_plan_sha = sources[str(PRIOR / 'plan.json')]
    assert prior_plan['schema'] == 'minicpm5-matched32-v2'
    assert prior_plan['criteria'] == p.CRITERIA and prior_plan['calls'] == p.calls()
    assert prior_plan['state_layout'] == p.layout() and prior_plan['stages'] == p.stages()
    assert metadata(PRIOR / 'status.json')['stage'] == 'failed'
    for phase in ('admit', 'control_device'):
        outcome = metadata(PRIOR / (phase + '-outcome.json'))
        assert outcome['returncode'] == 0
        assert all(outcome[key] is None for key in ('error', 'cleanup_error', 'journal_error'))
        assert not outcome['oom_kill_confirmed'] and not outcome['timeout_confirmed']
        receipt = metadata(PRIOR / (phase + '-result.json'))
        assert receipt['passed'] and receipt['phase'] == phase and receipt['plan_sha256'] == prior_plan_sha
        for name, digest in receipt['outputs'].items(): metadata(name, digest)
    admission = metadata(PRIOR / 'admission.json')
    assert admission['passed'] and admission['plan_sha256'] == prior_plan_sha
    assert admission['sha256'] == {name: row['sha256'] for name, row in prior_plan['payloads'].items()}
    failure = metadata(PRIOR / 'matched_cpu-outcome.json')
    assert failure['returncode'] != 0 and not failure['oom_kill_confirmed'] and not failure['timeout_confirmed']
    assert not (PRIOR / 'matched_cpu-result.json').exists()
    control = metadata(PRIOR / 'control-result.json')
    p.check_control(control)
    assert control['plan_sha256'] == prior_plan_sha
    assert control['original_warmup_and_resets_preserved']
    assert control['expected_original_failure'] == 'Original 32-frame numerical gate failed'
    assert len(control['bindings']) == 168
    count_states = count_inputs = 0
    for spec, reference in zip(p.calls(), control['reports']):
        folder = PRIOR / 'control' / ('call-%02d' % spec['index'])
        assert Path(reference['path']) == folder / 'result.json'
        row = metadata(reference['path'], reference['sha256'])
        assert row['complete'] and row['call'] == spec and row['inputs_unchanged'] and row['same168_bindings']
        assert row['bucket'] == p.check_input_metadata(row['input_metadata'], spec['phase'])
        payload(row['payload'], row['payload_sha256'], row['payload_identity'])
        assert Path(row['payload']) == folder / 'inputs-output.pt'
        assert len(row['compressed_inputs']) == 7
        for name, value, meta in zip(p.INPUT_NAMES, row['compressed_inputs'], row['input_metadata']):
            assert value['name'] == name and value['metadata'] == meta
            assert value['raw_sha256'] == meta['sha256']
            assert Path(value['path']) == folder / 'graph-inputs' / (name + '.zlib')
            payload(value['path'], value['payload_sha256'], value['identity']); count_inputs += 1
        for when in ('before', 'after'):
            ref = row[when]; assert ref['tensors'] == 168 and Path(ref['path']) == folder / when / 'manifest.json'
            state = metadata(ref['path'], ref['sha256'])
            assert state['complete'] and len(state['rows']) == 168
            assert [{key: entry[key] for key in p.layout()[0]} for entry in state['rows']] == p.layout()
            for index, entry in enumerate(state['rows']):
                assert entry['raw_bytes'] == p.CACHE_BYTES and len(entry['raw_sha256']) == 64
                expected = PRIOR_DATA / 'control' / folder.name / when / ('%02d.zlib' % index)
                assert Path(entry['file']) == expected
                payload(entry['file'], entry['payload_sha256'], entry['identity']); count_states += 1
    numeric = metadata(PRIOR / 'control/real-logits-validation.json')
    original = p.original(); variable = {'tensor', 'tensor_sha256', 'tensor_identity'}
    assert {key: value for key, value in numeric.items() if key not in variable} == {
        key: value for key, value in original.items() if key not in variable}
    payload(numeric['tensor'], numeric['tensor_sha256'], numeric['tensor_identity'])
    assert count_states == 4704 and count_inputs == 98 and len(payloads) == 4817
    return dict(source_sha256=sources, payloads=payloads,
        receipt=dict(directory=str(PRIOR), plan_sha256=prior_plan_sha,
            control_sha256=sources[str(PRIOR / 'control-result.json')],
            calls=14, frames=32, snapshots=28, state_tensors=4704, graph_inputs=98,
            payload_files=4817, original32_raw_comparison_receipt_preserved=True,
            device_recapture_permitted=False))


def run():
    plan = p.guard('reuse_control'); p.admitted(plan)
    saved = inputs()
    assert saved['receipt'] == plan['reused_control']
    assert all(plan['source_sha256'][name] == value for name, value in saved['source_sha256'].items())
    assert all(plan['payloads'][name] == value for name, value in saved['payloads'].items())
    # Preserve the authentic old receipt verbatim, including its original plan
    # identity. The new phase receipt binds the copy to this new admission.
    source = PRIOR / 'control-result.json'; destination = p.DEST / 'control-result.json'
    with destination.open('xb') as stream: stream.write(source.read_bytes())
    assert p.text_sha(destination) == saved['receipt']['control_sha256']
    p.once(p.DEST / 'reused-control.json', dict(passed=True, **saved['receipt'],
        source_result=str(source), complete_payload_authentication_at_admission=True,
        plan_sha256_current=p.text_sha(p.DEST / 'plan.json')))
