"""Pure exact r2+r3 writer-domain renderer; inputs and outputs remain in memory.

This adds only four reviewed r3 root helpers to the accepted r2 source domain.
It neither selects completion files nor proves that caller bytes are authentic.
"""
import ast
import copy
import hashlib
import json
import re
from pathlib import PurePosixPath as Path
import render_minicpm5_completed_writer_domain_sources_v1 as r2

Q, require, PREDECESSORS = r2.Q, r2.require, r2.PREDECESSORS
R3_HELPERS = {
 '/home/ubuntu/accept_qwen35_writer_geometry_root_r3_20261113_v2.py': '427091c4d88d12d18a9e623bf8740ff159472016ff1e2cb6150178e286dc8d12',
 '/home/ubuntu/ack_qwen35_writer_geometry_root_r3_20261113_v1.py': '2f4d459b9f1fbff7cb3e0a2a0362c6e16163b66053497265b55035ce7e0ad31e',
 '/home/ubuntu/finalize_qwen35_writer_geometry_r3_20261113_v3.py': 'd6268f30721a0f28b8a38c26cf0bc5fc60fcffd3b4ea37d5c629de5ad43a180e',
 '/home/ubuntu/prepare_qwen35_writer_geometry_r3_request_20261113_v2.py': 'baf1b416b8c317bdab049d96bde809aba134a2c58b7c6e74105f82f6e728e9c7'}
AUDITOR = dict(path=str(Q/'qwen35_writer_geometry_audit_v5.py'),
 sha256='562ba7b655c331d59948812d68dca8fb0e29f22a4bd04234f95db82f5d2913ac')
R3_RUNTIME_PINS = {'/home/ubuntu/experiments/trn1-latest-models-20261031/research/check_qwen35_geometry_audit_freshness_v5.py': '1e2de46a74371afe307ccffd177407bd610a8feb065e38b172e0bf0eb6db8b11', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/check_qwen35_geometry_storage_race_v3.py': '7719551063abd2a941207c5e02c66b95641be6e427b365e926f78e1c3f9ab5f0', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_geometry_audit_physical_v5.py': '908d1e478a4cc2792bb97a49f21c79a29cde65d5af9948a60c8b60ba0a50e5bd', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_audit_v5.py': '562ba7b655c331d59948812d68dca8fb0e29f22a4bd04234f95db82f5d2913ac', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_serial_v3/__init__.py': 'f57bc6f248065275725344845b1dbd58b1e645aa872276613dd438a171ba4a98', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_serial_v3/audit.py': 'cc32e877f16f419673f0cfac6b52cc605332021d5387c077582a525fc54379a2', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_serial_v3/contract.py': 'a9d76e1cc316c8f3b2a6329a003c0e0c8d6043924657f1825f63f957795bad27', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_serial_v3/run.py': '8950b9c7a69617fd7b0734c9126c29cda57d4870bef339327b4da434fd30d4bb', '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_writer_geometry_serial_v3/storage.py': '53f41ffee0a31b539dcaf58d3a4bb1273c9e0db73c505785aef913720240b321'}
RECOVERED_MAPPING_SHA256 = '4a740e86760e46bbd19bc9874c8b1e765a10fe2122380262b27197f9c2a2d07a'
R2_KEYS = {'base', 'catalog', 'ended', 'finalization'}
R3_KEYS = {'r3_catalog', 'r3_ended', 'r3_finalization', 'r3_spec', 'r3_overlay', 'r3_qualification'}


def tool(value):
    require(type(value) is dict and set(value) == {'kind', 'session_id', 'terminal_chunk', 'exit_code'}
        and value['kind'] == 'tool_session' and type(value['session_id']) is int and value['session_id'] > 0
        and type(value['exit_code']) is int and type(value['terminal_chunk']) is str
        and re.fullmatch('[a-f0-9]{3,64}', value['terminal_chunk']), 'Actual complete r3 external tool terminal')
    return value


def source_map(value):
    require(type(value) is dict and len(value) <= 65536, 'Bounded complete source map')
    for path, sha in value.items():
        require(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path
            and '..' not in Path(path).parts and type(sha) is str and re.fullmatch('[a-f0-9]{64}', sha),
            'Canonical complete source pair')
    return value


def merge(*maps):
    result = {}
    for value in maps:
        for path, sha in source_map(value).items():
            require(path not in result or result[path] == sha, 'Full source composition conflict')
            result[path] = sha
    return result


def geometry_spec_metadata(row):
    # Geometry's inherited load_spec reads complete source specs with 64 MiB.
    # This pure input reader does not alter Mini's 8 MiB readers or runtime.
    require(type(row) is dict and set(row) == {'reference', 'payload'}, 'Explicit r3 spec byte envelope')
    r2.reference(row['reference']); raw = row['payload']
    require(type(raw) is bytes and 0 < len(raw) <= 64 * 2**20
        and hashlib.sha256(raw).hexdigest() == row['reference']['sha256'],
        'Full r3 spec digest and inherited geometry 64 MiB source-spec cap')
    value = json.loads(raw)
    require(value['schema'] == 'qwen35_writer_geometry_serial_spec_v1'
        and value['run_name'] == 'writer-r3-20261113', 'Exact completed geometry r3 source spec')
    return value


def completed_inputs(inputs):
    require(type(inputs) is dict and set(inputs) == R2_KEYS | R3_KEYS,
            'Exact r2 and r3 completed metadata envelopes, including qualification or explicit None')
    old, oldrefs = r2.completed_inputs({key: inputs[key] for key in R2_KEYS})
    values = {key: (geometry_spec_metadata(inputs[key]) if key == 'r3_spec' else r2.metadata(inputs[key]))
        for key in R3_KEYS if key != 'r3_qualification'}
    refs = {key: copy.deepcopy(inputs[key]['reference']) for key in R3_KEYS if key != 'r3_qualification'}
    catalog, ended, final, spec, overlay = (values[k] for k in (
        'r3_catalog', 'r3_ended', 'r3_finalization', 'r3_spec', 'r3_overlay'))
    require(catalog['schema'] == old['catalog']['schema'] and catalog['reviewer'] == '/root'
        and catalog['passed'] is True and catalog['blockers'] == [] and len(catalog['runs']) == 165
        and catalog['runs'][:164] == old['catalog']['runs']
        and catalog['completed_sdk'] == old['catalog']['completed_sdk'], 'Entire exact164 prefix and SDK history retained')
    row = catalog['runs'][164]
    actual = tool(ended['actual_tool']); audit = tool(final['actual_independent_auditor_tool'])
    require(actual['session_id'] not in (72363, 9791) and audit['session_id'] not in (72363, 9791, actual['session_id'])
        and audit['exit_code'] == 0, 'Separate actually ended r3 outer and independent auditor')
    require(set(row) == {'terminal', 'tool', 'owners', 'units', 'source_sha256'}
        and row['terminal'] == refs['r3_ended'] and row['tool'] == actual
        and row['owners'] == ended['owners'] and row['units'] == ended['units']
        and type(row['owners']) is list and type(row['units']) is list and bool(row['units'])
        and len(set(row['units'])) == len(row['units']), 'Complete exact r3 catalog row')
    qualified = ended['geometry_qualified']
    counts = tuple(ended[k] for k in ('completed_layouts', 'completed_calls', 'compiled_graphs'))
    require(type(qualified) is bool and all(type(n) is int and 0 <= n <= cap for n, cap in zip(counts, (8, 40000, 32)))
        and ((qualified and actual['exit_code'] == 0 and counts == (8, 40000, 32))
            or (not qualified and actual['exit_code'] != 0)), 'Actual geometry status, counts and outer exit agree')
    require(ended['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1' and ended['reviewer'] == '/root'
        and ended['actual_tool_exit_observed'] is ended['actual_outer_exit_observed'] is True
        and ended['owners_gone'] is ended['physically_drained'] is ended['original_lock_held_during_audit'] is True
        and ended['lock_identity'] == r2.LOCK_IDENTITY and ended['reconciliation']['passed'] is qualified
        and ended['full_model_qualified'] is ended['new_speedup_claim'] is False
        and ended['auditor_source'] == AUDITOR and ended['spec'] == refs['r3_spec'],
        'Full independent r3 terminal, original lock and exact v5 auditor')
    require(spec['run_name'] == 'writer-r3-20261113' and spec['current_serial_catalog'] == oldrefs['catalog']
        and spec['history_overlay'] == refs['r3_overlay']
        and overlay['schema'] == 'minicpm5_layer1_operator_native_small_overlay_v2'
        and overlay['current_serial_catalog'] == oldrefs['catalog'], 'Actual r3 spec and full source overlay bind r2 history')
    runtime = source_map(spec['source_sha256']); extension = source_map(overlay['source_extension'])
    require(all(runtime.get(p) == sha for p, sha in R3_RUNTIME_PINS.items()),
        'Exact nine accepted r3 runtime/auditor additions retained')
    recovered = {p: sha for p, sha in runtime.items() if p not in extension}
    require(merge(recovered, extension) == runtime, 'Exact recovered/overlay source composition')
    require(hashlib.sha256(json.dumps(recovered, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        == RECOVERED_MAPPING_SHA256, 'Every exact original recovered source pair retained')
    prior = merge(recovered, *(saved['source_sha256'] for saved in old['catalog']['runs']))
    require(all(runtime.get(p) == sha for p, sha in prior.items()), 'Every historical source retained in actual runtime')
    executed = merge(runtime, R3_HELPERS)
    delta = {p: sha for p, sha in executed.items() if p not in prior}
    require(all(Path(p).is_relative_to(r2.ROOT) or p in R3_HELPERS for p in delta),
        'No unreviewed new external r3 source domain')
    require(merge(prior, delta) == executed and row['source_sha256'] == delta
        and all(delta.get(p) == sha for p, sha in dict(R3_HELPERS, **R3_RUNTIME_PINS).items()), 'Exact entire r3 source delta and four reviewed helpers')
    composition = dict(recovered_sources=len(recovered), retained_prior_sources=len(prior),
        runtime_sources=len(runtime), post_audit_sources=len(executed), new_row_sources=len(delta),
        all_previous_rows_retained=True, exact_full_mapping_reconstructed=True)
    require(len(recovered) == r2.COMPOSITION['recovered_sources']
        and len(prior) == r2.COMPOSITION['post_audit_sources'], 'Same original recovered sources and complete r2 source count')
    external = final['external_sources']
    require(type(external) is list and len(external) == 4
        and all(type(ref) is dict and set(ref) == {'path', 'sha256'} for ref in external)
        and {ref['path']: ref['sha256'] for ref in external} == R3_HELPERS,
        'Exactly four authenticated r3 helper pairs; no future or draft helpers')
    require(final['schema'] == 'qwen35_writer_geometry_root_finalization_v1' and final['reviewer'] == '/root'
        and final['actual_exit'] == refs['r3_ended'] and final['actual_outer_tool'] == actual
        and final['complete_old_catalog'] == oldrefs['catalog'] and final['catalog'] == refs['r3_catalog']
        and final['idle_handoff'] == catalog['current_handoff'] and final['source_composition'] == composition
        and final['all_prior164_rows_unchanged'] is final['original_8MiB_cap_preserved'] is True
        and final['total_rows'] == 165 and final['original_lock_identity'] == r2.LOCK_IDENTITY
        and final['geometry_qualified'] is qualified and final['full_model_qualified'] is final['new_speedup_claim'] is False,
        'Complete authentic r3 finalization and recomputed source composition')
    drains = final['two_current_drain_observations']
    require(type(drains) is list and len(drains) == 2 and drains[0] == drains[1]
        and set(drains[0]) == set(row['units']), 'Two complete stable r3 finalizer drains')
    if qualified:
        proof = r2.metadata(inputs['r3_qualification']); proofref = copy.deepcopy(inputs['r3_qualification']['reference'])
        require(final['qualification'] == proofref and proof['schema'] == 'qwen35_writer_geometry_independent_qualification_v1'
            and proof['reviewer'] == '/root' and proof['passed'] is True and proof['blockers'] == []
            and proof['actual_exit'] == refs['r3_ended'] and proof['reports'] == ended['reports']
            and proof['source_sha256'] == ended['source_sha256'] and proof['full_model_qualified'] is False,
            'Separate exact successful geometry qualification')
        values['r3_qualification'] = proof; refs['r3_qualification'] = proofref
    else:
        require(inputs['r3_qualification'] is None and final['qualification'] is None,
            'Failed r3 has no qualification; real filesystem absence remains caller duty')
        values['r3_qualification'] = None; refs['r3_qualification'] = None
    return old, oldrefs, values, refs


R3_AUTHENTICATOR_BODY = '''

_authenticate_completed_writer_r2 = authenticate_completed_writer


def authenticate_completed_writer(catalog_ref, read, guard):
    # Retain every r2 check before granting the four additional r3 pairs.
    original = _authenticate_completed_writer_r2(catalog_ref, read, guard)
    guard()
    historical = read(R3_HISTORICAL_CATALOG)
    old = read(HISTORICAL_CATALOG)
    current = historical if catalog_ref == R3_HISTORICAL_CATALOG else read(catalog_ref)
    require(historical['schema'] == old['schema'] and historical['reviewer'] == '/root'
        and historical['passed'] is True and historical['blockers'] == []
        and len(historical['runs']) == 165 and historical['runs'][:164] == old['runs']
        and historical['completed_sdk'] == old['completed_sdk'] and historical['runs'][164] == R3_ENDED_ROW
        and type(current['runs']) is list and len(current['runs']) >= 165
        and current['runs'][:165] == historical['runs']
        and current['completed_sdk'] == historical['completed_sdk'], 'Entire exact165 completed historical prefix retained')
    ended = read(R3_ACTUAL_EXIT)
    require(all(ended.get(k) == v for k, v in R3_ENDED_STATUS.items()), 'Exact selected r3 actual audit status retained')
    require(read(R3_FINALIZATION) == R3_FINALIZATION_STATUS, 'Entire exact r3 finalization and source composition retained')
    require(R3_FINALIZATION_STATUS['catalog'] == R3_HISTORICAL_CATALOG
        and R3_FINALIZATION_STATUS['actual_exit'] == R3_ACTUAL_EXIT
        and R3_FINALIZATION_STATUS['idle_handoff'] == historical['current_handoff'], 'Exact r3 completion references')
    if R3_QUALIFICATION is not None:
        require(read(R3_QUALIFICATION) == R3_QUALIFICATION_STATUS, 'Entire separate r3 qualification retained')
    require(all(R3_ENDED_ROW['source_sha256'].get(p) == sha for p, sha in COMPLETED_R3_TEXT.items()),
            'Four exact r3 root helpers retained in complete appended row')
    guard(); return dict(original, **COMPLETED_R3_TEXT)


def extension_shape(extension, *, completed_lfm_text, catalog_ref, read, guard=lambda: None):
    require(type(extension) is dict and 0 < len(extension) <= 2048, 'Original bounded additive extension')
    all_text = dict(COMPLETED_WRITER_TEXT, **COMPLETED_R3_TEXT)
    selected = {p: sha for p, sha in extension.items() if p in all_text}
    ordinary = {p: sha for p, sha in extension.items() if p not in all_text}
    previous.extension_shape(ordinary, completed_lfm_text=completed_lfm_text,
        catalog_ref=catalog_ref, read=read, guard=guard)
    if selected:
        require(selected == all_text, 'All42 exact completed writer pairs; no subset or changed SHA')
        require(authenticate_completed_writer(catalog_ref, read, guard) == selected,
            'Same exact r2+r3 domain in prepare and actual authentication')
    return extension
'''


def render(predecessors, inputs):
    """Render the still-unpublished seven successors; no actual IO or admission."""
    old, oldrefs, values, refs = completed_inputs(inputs)
    result = r2.render(predecessors, {key: inputs[key] for key in R2_KEYS})
    constants = dict(R3_HISTORICAL_CATALOG=refs['r3_catalog'], R3_ACTUAL_EXIT=refs['r3_ended'],
        R3_FINALIZATION=refs['r3_finalization'], R3_QUALIFICATION=refs['r3_qualification'],
        COMPLETED_R3_TEXT=R3_HELPERS, R3_ENDED_ROW=values['r3_catalog']['runs'][164],
        R3_ENDED_STATUS={key: values['r3_ended'][key] for key in (
            'schema', 'reviewer', 'actual_tool', 'actual_tool_exit_observed', 'actual_outer_exit_observed',
            'owners', 'units', 'owners_gone', 'physically_drained', 'original_lock_held_during_audit', 'lock_identity',
            'geometry_qualified', 'full_model_qualified', 'new_speedup_claim', 'reconciliation',
            'completed_layouts', 'completed_calls', 'compiled_graphs', 'auditor_source', 'spec', 'reports', 'source_sha256')},
        R3_FINALIZATION_STATUS=values['r3_finalization'],
        R3_QUALIFICATION_STATUS=values['r3_qualification'])
    path = str(Q/'minicpm5_completed_qwen_package_text_v4.py')
    addition = '\n' + ''.join(key+' = '+repr(value)+'\n' for key, value in constants.items()) + R3_AUTHENTICATOR_BODY
    result['files'][path] += addition.encode()
    require(len(result['files'][path]) < 4 * 2**20, 'Original source text cap')
    ast.parse(result['files'][path])
    result.update(schema='minicpm5_completed_writer_source_candidate_v2',
        completed_refs=dict(oldrefs, **refs), accepted_new_text_domain=dict(r2.EXTERNAL_TEXT, **R3_HELPERS),
        completed_prefix_rows=165, r3_source_composition_reconstructed=True,
        generated_names_must_be_fresh=True, failed_qualification_absence_independently_verified=False)
    return result
