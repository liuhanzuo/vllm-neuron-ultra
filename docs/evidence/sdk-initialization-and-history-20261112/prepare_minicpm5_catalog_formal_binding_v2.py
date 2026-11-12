"""Thin original binding plus exact completed-Qwen catalog compatibility.

No IO, current input discovery, approval, lock or native execution. Keeps the
literal delivery, ordinary original preparer and complete original graph.
"""
import copy
import hashlib
import json
from pathlib import Path

from research import prepare_minicpm5_literal_formal_binding_v1 as literal
from research import prepare_minicpm5_pv_sdk_import_catalog_v2 as catalog_prepare

previous, launch, Q, require = literal.previous, literal.launch, literal.Q, literal.require
RUNTIME = ('minicpm5_completed_qwen_package_text_v1.py',
           'minicpm5_fixed_input_pv_catalog_auth_v1.py',
           'minicpm5_sdk_mapping_outer_source_entry_v8.py',
           'minicpm5_sdk_mapping_observer_logic_v2.py',
           'minicpm5_sdk_mapping_observer_backend_v4.py',
           'minicpm5_sdk_mapping_observed_adapter_v5.py',
           'minicpm5_sdk_mapping_source_adapter_v3.py')
SOURCES = RUNTIME + ('prepare_minicpm5_pv_sdk_import_catalog_v2.py',
                     'prepare_minicpm5_catalog_formal_binding_v2.py')


def checked_sources(catalog_source_refs, text):
    require(type(catalog_source_refs) is list and len(catalog_source_refs) == len(SOURCES)
            and {row['path'] for row in catalog_source_refs} == {str(Q / name) for name in SOURCES},
            'Exactly nine explicit catalog/literal-boundary successor source refs')
    result = {}
    for row in catalog_source_refs:
        launch.reference(row)
        require(hashlib.sha256(text(row['path'], row['sha256'])).hexdigest() == row['sha256'],
                'Entire exact caller-selected successor text')
        result[row['path']] = row['sha256']
    return result


def prepare(request, *, literal_source_manifest, preparation_source, catalog_source_refs,
            read, text, guard, original):
    old, manifest, additions, _ = previous.validate_request(request, read)
    _, extra = literal.source_additions(literal_source_manifest, preparation_source, read=read, text=text)
    extra = previous.merge(extra, checked_sources(catalog_source_refs, text))
    require(str(Path(original.__file__).resolve()) == previous.ORIGINAL,
            'Only ordinary original historical preparer supplied')
    expected = manifest['retained_source_sha256'].get(previous.ORIGINAL)
    require(expected is not None and hashlib.sha256(text(previous.ORIGINAL, expected)).hexdigest() == expected,
            'Unchanged ordinary original preparer source')
    old = copy.deepcopy(old)
    supplied = {row['path']: row['sha256'] for row in old['additional_source_refs']}
    require(len(supplied) == len(old['additional_source_refs']), 'No duplicate caller additions')
    supplied = previous.merge(previous.merge(supplied, additions), extra)
    old['additional_source_refs'] = [dict(path=path, sha256=sha) for path, sha in sorted(supplied.items())]
    candidate = catalog_prepare.prepare(original, old, read=read, text=text, guard=guard)
    base = original.reconstruct_audit_base(old['independent_audit_exit'], old['recovery_binding'], read, text, guard)
    full = original.audited.merged_plan(base['plan'], candidate['pv_spec_candidate']['source_extension'])
    require(previous.policies.history.digest(full['source_sha256']) == candidate['complete_source_map_sha256']
            and len(full['source_sha256']) == candidate['complete_source_files'], 'Same complete original reconstruction')
    guard()
    result = assemble(request, candidate, full, literal_source_manifest=literal_source_manifest,
        preparation_source=preparation_source, catalog_source_refs=catalog_source_refs, read=read, text=text)
    guard()
    return result


def assemble(request, candidate, full, *, literal_source_manifest, preparation_source,
             catalog_source_refs, read, text):
    sources = checked_sources(catalog_source_refs, text)
    require(all(full['source_sha256'].get(path) == candidate['pv_spec_candidate']['source_extension'].get(path) == sha
                for path, sha in sources.items()), 'Complete original prepared map and PV extension include every successor')
    result = literal.assemble(request, candidate, full, literal_source_manifest=literal_source_manifest,
                              preparation_source=preparation_source, read=read, text=text)
    artifacts = result['artifacts']
    bootstrap = previous.merge(artifacts['outer-entry-plan']['value']['bootstrap_sources'],
                               {str(Q / name): sources[str(Q / name)] for name in RUNTIME})
    require(len(bootstrap) <= 2048, 'Original bootstrap source-count cap')
    def replace(name, value):
        raw = launch.encoded(value)
        require(0 < len(raw) <= 2**20, 'Original complete artifact ceiling')
        row = artifacts[name]
        row.update(value=copy.deepcopy(value), bytes=len(raw),
                   reference=dict(path=row['reference']['path'], sha256=hashlib.sha256(raw).hexdigest()))
        return copy.deepcopy(row['reference'])
    sdk = copy.deepcopy(artifacts['sdk-entry-template']['value'])
    sdk['bootstrap_sources'] = copy.deepcopy(bootstrap)
    sdk_ref = replace('sdk-entry-template', sdk)
    spec = copy.deepcopy(artifacts['outer-spec']['value'])
    spec.update(sdk_entry_template=sdk_ref, source_extension=copy.deepcopy(bootstrap))
    spec_ref = replace('outer-spec', spec)
    review = copy.deepcopy(artifacts['outer-review-draft']['value'])
    review.update(spec=spec_ref, source_extension=copy.deepcopy(bootstrap))
    review_ref = replace('outer-review-draft', review)
    outer = copy.deepcopy(artifacts['outer-entry-plan']['value'])
    outer.update(bootstrap_sources=copy.deepcopy(bootstrap),
        entry_module='research.minicpm5_sdk_mapping_outer_source_entry_v8',
        entry_path=str(Q / 'minicpm5_sdk_mapping_outer_source_entry_v8.py'),
        successor_input=json.dumps(dict(spec=spec_ref, review=review_ref), sort_keys=True, separators=(',', ':')))
    result['outer_entry_plan'] = replace('outer-entry-plan', outer)
    launch.separate_authorities(outer, sdk)
    launch.launch(text(result['entry_source']['path'], result['entry_source']['sha256']),
                  result['entry_source'], outer, result['outer_entry_plan'])
    result['catalog_compatibility_source_sha256'] = sources
    return result
