"""Thin source-only wiring successor to the unchanged formal assembler.

No IO, source discovery, historical import, approval, catalog choice or launch.
The caller supplies original saved inputs, frozen literal delivery, this source
reference, and the ordinary original preparer under its existing context.
"""
import copy
import hashlib
import json
from pathlib import Path

from research import prepare_minicpm5_observed_formal_binding_v1 as previous
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import minicpm5_formal_literal_codec_v1 as codec

Q, require = previous.Q, previous.require
SELF = str(Q / 'prepare_minicpm5_literal_formal_binding_v1.py')
RUNTIME = (
    'minicpm5_formal_literal_codec_v1.py',
    'minicpm5_formal_literal_result_v1.py',
    'minicpm5_sdk_mapping_observer_backend_v3.py',
    'minicpm5_sdk_mapping_observed_adapter_v4.py',
    'minicpm5_sdk_mapping_source_adapter_v2.py',
    'minicpm5_sdk_mapping_nocache_launch_v3.py',
    'minicpm5_sdk_mapping_outer_source_entry_v6.py',
    'minicpm5_sdk_mapping_worker_source_entry_v6.py',
)


def source_additions(literal_source_manifest, preparation_source, *, read, text):
    launch.reference(literal_source_manifest)
    launch.reference(preparation_source)
    require(preparation_source['path'] == SELF, 'Explicit exact new preparation source')
    require(hashlib.sha256(text(SELF, preparation_source['sha256'])).hexdigest() == preparation_source['sha256'],
            'Current entire new preparer equals caller source pin')
    delivery = read(literal_source_manifest)
    require(delivery['schema'] == 'minicpm5_formal_literal_source_delivery_v1', 'Exact frozen literal source delivery')
    rows = delivery['runtime_source_files']
    require(type(rows) is list and len(rows) == len(RUNTIME)
            and {row['path'] for row in rows} == {str(Q / name) for name in RUNTIME},
            'Exactly eight explicitly delivered runtime sources')
    runtime = {}
    for row in rows:
        codec.reference(row)
        require(row in delivery['source_files'], 'Runtime file is also in complete delivered source list')
        raw = text(row['path'], row['sha256'])
        require(type(raw) is bytes and len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'Each complete frozen runtime source remains unchanged')
        runtime[row['path']] = row['sha256']
    return runtime, previous.merge(runtime, {SELF: preparation_source['sha256']})


def prepare(request, *, literal_source_manifest, preparation_source, read, text, guard, original):
    """Reuse original prepare/reconstruct/merged_plan unchanged, then wire v6.

    No accepted inputs are generated here. The caller retains the original
    responsibility for importing the ordinary historical preparer correctly.
    """
    old, manifest, additions, _ = previous.validate_request(request, read)
    _, extra = source_additions(literal_source_manifest, preparation_source, read=read, text=text)
    require(str(Path(original.__file__).resolve()) == previous.ORIGINAL,
            'Only the ordinary original historical preparation module')
    expected = manifest['retained_source_sha256'].get(previous.ORIGINAL)
    require(expected is not None and hashlib.sha256(text(previous.ORIGINAL, expected)).hexdigest() == expected,
            'Unchanged original historical preparation source')
    old = copy.deepcopy(old)
    supplied = {row['path']: row['sha256'] for row in old['additional_source_refs']}
    require(len(supplied) == len(old['additional_source_refs']), 'No duplicate caller source additions')
    supplied = previous.merge(previous.merge(supplied, additions), extra)
    old['additional_source_refs'] = [dict(path=path, sha256=sha) for path, sha in sorted(supplied.items())]
    candidate = original.prepare(old, read=read, text=text, guard=guard)
    base = original.reconstruct_audit_base(old['independent_audit_exit'], old['recovery_binding'], read, text, guard)
    full = original.audited.merged_plan(base['plan'], candidate['pv_spec_candidate']['source_extension'])
    require(previous.policies.history.digest(full['source_sha256']) == candidate['complete_source_map_sha256']
            and len(full['source_sha256']) == candidate['complete_source_files'],
            'Complete unchanged original reconstruction agrees')
    guard()
    result = assemble(request, candidate, full, literal_source_manifest=literal_source_manifest,
                      preparation_source=preparation_source, read=read, text=text)
    guard()
    return result


def assemble(request, candidate, full, *, literal_source_manifest, preparation_source, read, text):
    """Rewire four existing draft artifacts; original scientific graph is kept."""
    runtime, extra = source_additions(literal_source_manifest, preparation_source, read=read, text=text)
    pins = full['source_sha256']
    require(all(pins.get(path) == sha and candidate['pv_spec_candidate']['source_extension'].get(path) == sha
                for path, sha in extra.items()), 'Original preparation must already include all nine source additions')
    require(pins.get(codec.LITERAL['path']) == codec.LITERAL['sha256'], 'Same unchanged complete v6 literal source pin')
    literal = text(codec.LITERAL['path'], codec.LITERAL['sha256'])
    codec.literal_bytes(literal.decode('utf-8'))
    result = previous.assemble(request, candidate, full, read=read, text=text)
    artifacts = result['artifacts']
    old_bootstrap = artifacts['outer-entry-plan']['value']['bootstrap_sources']
    require(old_bootstrap == artifacts['sdk-entry-template']['value']['bootstrap_sources']
            == artifacts['outer-spec']['value']['source_extension'], 'Original paired complete bootstrap equality')
    bootstrap = previous.merge(old_bootstrap, runtime)
    require(0 < len(bootstrap) <= 2048 and all(pins.get(path) == sha for path, sha in bootstrap.items()),
            'Both full bootstrap closures fit original bound and whole original map')

    def replace(name, value):
        raw = launch.encoded(value)
        require(0 < len(raw) <= 2**20, 'Original complete draft metadata ceiling')
        row = artifacts[name]
        row.update(value=copy.deepcopy(value), bytes=len(raw),
                   reference=dict(path=row['reference']['path'], sha256=hashlib.sha256(raw).hexdigest()))
        return copy.deepcopy(row['reference'])

    sdk = copy.deepcopy(artifacts['sdk-entry-template']['value'])
    sdk.update(bootstrap_sources=copy.deepcopy(bootstrap),
        entry_module='research.minicpm5_sdk_mapping_worker_source_entry_v6',
        entry_path=str(Q / 'minicpm5_sdk_mapping_worker_source_entry_v6.py'))
    sdk_ref = replace('sdk-entry-template', sdk)
    spec = copy.deepcopy(artifacts['outer-spec']['value'])
    spec.update(sdk_entry_template=sdk_ref, source_extension=copy.deepcopy(bootstrap))
    spec_ref = replace('outer-spec', spec)
    review = copy.deepcopy(artifacts['outer-review-draft']['value'])
    review.update(spec=spec_ref, source_extension=copy.deepcopy(bootstrap))
    review_ref = replace('outer-review-draft', review)
    outer = copy.deepcopy(artifacts['outer-entry-plan']['value'])
    outer.update(bootstrap_sources=copy.deepcopy(bootstrap),
        entry_module='research.minicpm5_sdk_mapping_outer_source_entry_v6',
        entry_path=str(Q / 'minicpm5_sdk_mapping_outer_source_entry_v6.py'),
        successor_input=json.dumps(dict(spec=spec_ref, review=review_ref), sort_keys=True, separators=(',', ':')))
    result['outer_entry_plan'] = replace('outer-entry-plan', outer)
    launch.separate_authorities(outer, sdk)
    argv = launch.launch(literal, result['entry_source'], outer, result['outer_entry_plan'])
    require(argv[7] == literal.decode('utf-8') and argv[13] == codec.LITERAL['sha256'], 'Actual complete original literal argv')
    result.update(literal_boundary_source_manifest=copy.deepcopy(literal_source_manifest),
        literal_boundary_preparation_source=copy.deepcopy(preparation_source),
        literal_boundary_runtime_source_sha256=runtime)
    return result
