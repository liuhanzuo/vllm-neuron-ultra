"""PV-only authentication adapter over the frozen original audit machinery.

Inert on import. authenticate() is an actual-admission operation and MUST NOT
be called by source-only preparation/tests. No raw capture reads or native
imports occur here. Reuses complete reconstruction rather than a second
history/catalog implementation; no subset source map can authorize workers.
"""
import copy
from pathlib import Path
import re

from research import audit_minicpm5_layer1_saved_operators_v1 as prior
from research import minicpm5_layer1_fixed_input_pv_serial_v1 as serial

r, require = prior.r, prior.require
previous, audited = prior.previous, prior.audited
ROOT = previous.ROOT


def validate(spec_ref, review_ref, spec, review):
    for ref in (spec_ref, review_ref): r.life.ref(ref)
    require(type(spec) is dict and set(spec) == {'schema', 'independent_audit_exit', 'current_serial_catalog',
        'source_extension', 'output_root', 'worker', 'limits', 'artifact_limits'}
        and spec['schema'] == 'minicpm5_fixed_input_pv_serial_spec_v1'
        and serial.exact(spec['limits'], prior.LIMITS) and serial.exact(spec['artifact_limits'], serial.CAPS),
        'Exact finite PV spec preserves original resource and artifact limits')
    for name in ('independent_audit_exit', 'current_serial_catalog'): r.life.ref(spec[name])
    require(type(review) is dict and set(review) == {'schema', 'reviewer', 'passed', 'blockers', 'spec', 'source_extension'}
        and review['schema'] == 'minicpm5_fixed_input_pv_serial_source_review_v1' and review['reviewer'] == '/root'
        and review['passed'] is True and review['blockers'] == [] and review['spec'] == spec_ref
        and serial.exact(review['source_extension'], spec['source_extension']), 'Exact independent concrete source/spec review')
    pins = spec['source_extension']
    require(type(pins) is dict and 0 < len(pins) <= 2048, 'Bounded additive text-source extension')
    for name, sha in pins.items():
        require(type(name) is str and str(Path(name)) == name and Path(name).is_absolute() and '..' not in Path(name).parts
            and (name.startswith(str(ROOT)+'/') or prior.COMPLETED_LFM_TEXT.get(name) == sha)
            and Path(name).suffix in ('.py', '.json', '.md', '.txt', '.sh')
            and type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha), 'Exact local/completed original text pin')
    out = Path(spec['output_root'])
    require(out.parent == ROOT/'research' and str(out) == spec['output_root']
        and re.fullmatch(r'minicpm5-fixed-input-pv-[a-z0-9][a-z0-9-]{0,39}', out.name), 'Fresh bounded PV-only output namespace')
    for path in (str(Path(__file__).resolve()), str(Path(serial.__file__).resolve()),
                 str(Path(serial.leaf.__file__).resolve()), str(Path(serial.core.__file__).resolve()), spec['worker']):
        require(path in pins, 'All new executed PV helpers/worker explicitly source reviewed')
    return out


def audit_terminal(ended, live, result, spec, review):
    """Exact completed audit terminal plus external live/owner/interpreter chain."""
    require(type(ended) is dict and set(ended) == {'schema', 'reviewer', 'audit_spec', 'audit_review', 'live', 'result',
        'owner', 'actual_tool', 'actual_tool_exit_observed', 'exact_owner_gone', 'lock_identity', 'self_exit_claimed',
        'full_model_qualified', 'new_speedup_claim'}
        and ended['schema'] == 'minicpm5_saved_operator_independent_audit_actual_exit_v1' and ended['reviewer'] == '/root'
        and ended['actual_tool_exit_observed'] is ended['exact_owner_gone'] is True and ended['lock_identity'] == r.ORIGINAL_LOCK
        and ended['self_exit_claimed'] is ended['full_model_qualified'] is ended['new_speedup_claim'] is False,
        'Independent root-observed actual successful comparison auditor terminal required')
    tool = prior.core.actual_tool(ended['actual_tool'])
    for name in ('audit_spec', 'audit_review', 'live', 'result'): r.life.ref(ended[name])
    prior.validate_spec(ended['audit_spec'], ended['audit_review'], spec, review)
    argv = ['python3', '-B', prior.SELF, '--spec', ended['audit_spec']['path'], '--spec-sha256', ended['audit_spec']['sha256'],
            '--review', ended['audit_review']['path'], '--review-sha256', ended['audit_review']['sha256']]
    r.life.process(ended['owner'], argv)
    require(type(live) is dict and set(live) == {'schema', 'reviewer', 'observed_at', 'spec', 'review', 'actual_tool', 'owner',
        'actual_executable', 'actual_owner_observed', 'actual_tool_exit_observed', 'full_model_qualified', 'new_speedup_claim'}
        and live['schema'] == 'root_minicpm5_saved_operator_independent_audit_live_v1' and live['reviewer'] == '/root'
        and type(live['observed_at']) is str and live['observed_at'] and live['spec'] == ended['audit_spec']
        and live['review'] == ended['audit_review'] and serial.exact(live['owner'], ended['owner'])
        and live['actual_tool'] == dict(kind='tool_session', session_id=tool['session_id'])
        and live['actual_executable'] == str(Path(serial.python_for(0)).resolve())
        and live['actual_owner_observed'] is True and live['actual_tool_exit_observed'] is False
        and live['full_model_qualified'] is live['new_speedup_claim'] is False,
        'Original auditor exact external live/argv/interpreter/owner/session binding')
    require(result['schema'] == prior.RESULT_SCHEMA and result['complete'] is True and result['reviewer'] == '/root'
        and result['audit_spec'] == ended['audit_spec'] and result['audit_review'] == ended['audit_review']
        and serial.exact(result['audit_owner'], ended['owner']) and ended['result']['path'] == spec['output'],
        'Exact original complete independent audit result bound to real external terminal')
    return tool


def merge_extensions(older, additions):
    result = copy.deepcopy(older)
    for path, sha in additions.items():
        require(path not in result or result[path] == sha, 'Never replace an inherited source pin')
        result[path] = sha
    return result


def authenticate(spec_ref, review_ref, *, lease, metadata, guard):
    """Run only under already acquired original outer descriptor, before children.

    Uses original comparison reconstruction on the full current catalog. Then
    validates completed independent auditor's exact old catalog/physical pair,
    complete source map, and every result field. All failed/old rows remain.
    """
    guard(force=True); read = metadata.read
    spec, review = read(spec_ref), read(review_ref); out = validate(spec_ref, review_ref, spec, review)
    ended = read(spec['independent_audit_exit']); audit_spec = read(ended['audit_spec']); audit_review = read(ended['audit_review'])
    audit_result = read(ended['result']); live = read(ended['live'])
    tool = audit_terminal(ended, live, audit_result, audit_spec, audit_review); r.p.gone(ended['owner'])
    comparison_ended = read(audit_spec['comparison_exit']); summary = read(comparison_ended['result'])
    require(tool['session_id'] != comparison_ended['actual_tool']['session_id'], 'Independent auditor has its own real completed tool')
    combined = merge_extensions(audit_spec['source_extension'], spec['source_extension'])
    context = prior.reconstruct(comparison_ended, summary,
        dict(comparison_exit=audit_spec['comparison_exit'], current_serial_catalog=spec['current_serial_catalog'], source_extension=combined),
        metadata, guard, lease)
    original_spec = context['original_spec']; model = read(original_spec['model_exit']); model_spec = read(model['spec'])
    native_auditor = read(original_spec['auditor_exit'])
    old_plan, old_source = previous.source_plan(context['resolved'], model, model_spec,
        read(native_auditor['audit_spec']), read(native_auditor['result']), original_spec)
    inherited = audited.merged_plan(old_plan, audit_spec['source_extension'])
    expected_plan = audited.merged_plan(inherited, spec['source_extension'])
    require(serial.exact(expected_plan, context['pins_plan']), 'Every original model/comparison/auditor source pin preserved exactly')
    view = dict(latest_completion=context['resolved']['plan']['latest_completion'], source_sha256=expected_plan['source_sha256'])
    historical = r.completion.latest.serial_catalog(dict(view, serial_predecessors=audit_spec['current_serial_catalog']), read, guard)
    current = context['scope']['later_serial']
    previous.catalog_rows(current, historical, [(spec['independent_audit_exit'], ended['owner'], tool)])
    prior.historical_freshness(audit_result['serial_freshness'], historical, read)
    recorder = read(original_spec['recorder_exit'])
    old_scope = previous.serial_scope(context['resolved'], historical,
        dict(model_owners=[model['owner'], model['coordinator'], model['native']],
             contract=context['owned'].contract), recorder)
    pair = r.completion.audit.unpack(audit_result['complete_current_physical_pair'])
    known = set(old_scope['complete_known_units']); required = {row['unit'] for row in old_scope['nemotron']['history']['rows']}
    require(r.completion.physical.validate_pair(pair, required, old_scope['owners'], r.ORIGINAL_LOCK, known_units=known) == known,
        'Complete original independent auditor physical scope, never a subset')
    storage = prior.saved_storage(audit_result['storage'], guard.storage['usage'])
    reports = previous.output_paths(original_spec['output_prefix'])
    original_reports = prior.report_references(original_spec, summary)
    for ref, row in zip(original_reports, context['plan']['calls']):
        saved = read(ref)
        require(ref['path'] == str(reports[str(row['call'])])
            and serial.exact(saved['complete_call_capture_descriptors'], row)
            and saved['descriptor_plan_sha256'] == r.digest(context['plan']), 'Every original report retains complete authenticated descriptors')
    expected = dict(schema=prior.RESULT_SCHEMA, complete=True, reviewer='/root', audit_spec=ended['audit_spec'], audit_review=ended['audit_review'],
        comparison_exit=audit_spec['comparison_exit'], original_summary=comparison_ended['result'], original_reports=summary['reports'],
        audit_owner=ended['owner'], lock_identity=r.ORIGINAL_LOCK, source=dict(path=prior.SELF, sha256=inherited['source_sha256'][prior.SELF]),
        complete_source_binding=dict(original_comparison_source=old_source, additional_sources_sha256=r.digest(audit_spec['source_extension']),
            additional_source_files=len(audit_spec['source_extension']), complete_source_map_sha256=r.digest(inherited['source_sha256']),
            complete_source_files=len(inherited['source_sha256']), complete_source_reconstruction_required=True,
            inherited_sources_replaced=False, inherited_sources_omitted=False),
        current_serial_catalog=audit_spec['current_serial_catalog'], serial_freshness=audit_result['serial_freshness'],
        complete_current_physical_pair=audit_result['complete_current_physical_pair'], physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
        capture_files_independently_reread=1284, capture_bytes_independently_reread=77790564, frames=24, calls=6, ranks=2,
        all_six_complete_reports_raw_recomputed_exact=True, complete_summary_reconstructed_exact=True,
        source_exit_and_full_control_lineage_verified=True, historical_storage_telemetry_bound_validated_not_remeasured=True,
        descriptor_plan_sha256=r.digest(context['plan']), recomputed_scientific_summary_sha256=r.digest(summary['comparison']),
        actual_comparison_tool_exit_verified=True, actual_auditor_tool_exit_verified=False, self_exit_claimed=False,
        model_reexecuted=False, causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False, storage=storage)
    require(serial.exact(audit_result, expected), 'Every complete original independently completed audit field reconstructed')
    authority = r.current_authority.Authority(r, lease); physical = audited.pair(context['scope'], guard, authority)
    r.verify_loaded(expected_plan, guard); metadata.unchanged(); guard(force=True)
    return dict(spec=spec, review=review, output_root=str(out), context=context, before_physical=physical,
        witness=dict(schema='minicpm5_fixed_input_pv_prerequisite_witness_v1', spec=spec_ref, review=review_ref,
            independent_audit_exit=spec['independent_audit_exit'], independent_audit_result=ended['result'],
            comparison_exit=audit_spec['comparison_exit'], comparison_summary=comparison_ended['result'],
            comparison_reports=original_reports, descriptor_plan_sha256=r.digest(context['plan']),
            complete_source_map_sha256=r.digest(expected_plan['source_sha256']), complete_source_files=len(expected_plan['source_sha256']),
            current_serial_catalog=spec['current_serial_catalog'], original_sdk_model_exit_code=1,
            source_history_and_catalog_reconstructed=True, raw_capture_bytes_read=0,
            full_model_qualified=False, new_speedup_claim=False))
