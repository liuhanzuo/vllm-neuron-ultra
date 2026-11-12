"""Source-only preparation successor for exact ended Qwen historical text.

Original complete preparation body is preserved except shared extension-shape
validation. Original ordinary module and readers are supplied by the caller;
this module never imports historical/native modules or chooses a current run.
"""
import copy
import hashlib
from pathlib import Path
import re
from research import minicpm5_completed_qwen_package_text_v2 as completed
ROOT = completed.ROOT
ORIGINAL = str(ROOT / 'research/prepare_minicpm5_pv_sdk_import_diagnostic_v1.py')
require = completed.require


def prepare(original_preparer, request, *, read, text, guard):
    """Return in-memory candidates only; never create a spec/review/catalog."""
    require(str(Path(original_preparer.__file__).resolve()) == ORIGINAL,
            'Only ordinary immutable original preparation module')
    diagnostic = original_preparer.diagnostic
    auth, serial, prior, r = diagnostic.auth, diagnostic.serial, diagnostic.prior, diagnostic.r
    original = diagnostic.original
    previous, audited = prior.previous, prior.audited
    merge, disjoint_extension = original_preparer.merge, original_preparer.disjoint_extension
    validated_review_sources = original_preparer.validated_review_sources
    reconstruct_audit_base = original_preparer.reconstruct_audit_base
    MANDATORY = original_preparer.MANDATORY
    ORIGINAL_ADAPTER_SHA = original_preparer.ORIGINAL_ADAPTER_SHA
    REQUEST_SCHEMA, RESULT_SCHEMA = original_preparer.REQUEST_SCHEMA, original_preparer.RESULT_SCHEMA
    require(type(request) is dict and set(request) == {'schema','independent_audit_exit','current_serial_catalog',
        'diagnostic_source_review','recovery_binding','additional_source_refs','output_root'}
        and request['schema'] == REQUEST_SCHEMA, 'Every history/source/output input explicitly supplied')
    for key in ('independent_audit_exit','current_serial_catalog','diagnostic_source_review'): r.life.ref(request[key])
    out = Path(request['output_root'])
    require(out.parent == ROOT/'research' and str(out) == request['output_root']
        and re.fullmatch('minicpm5-fixed-input-pv-sdk-import-[a-z0-9][a-z0-9-]{0,28}', out.name),
        'Future explicitly diagnostic namespace; not selected implicitly')
    reviewed = validated_review_sources(request['diagnostic_source_review'], read, text)
    base = reconstruct_audit_base(request['independent_audit_exit'], request['recovery_binding'], read, text, guard)
    inherited = base['plan']['source_sha256']; catalog = read(request['current_serial_catalog'])
    required = {}
    for row in catalog['runs']: required = merge(required, row['source_sha256'])
    required = merge(required, reviewed)
    # Original helpers are authenticated by immutable source bytes, not copied
    # from a previous attempt's potentially redundant extension.
    for path in MANDATORY:
        if path not in required:
            raw = text(path, None); required[path] = hashlib.sha256(raw).hexdigest()
    require(required[str(Path(original.__file__).resolve())] == ORIGINAL_ADAPTER_SHA, 'Frozen original adapter-v3')
    extra = request['additional_source_refs']
    require(type(extra) is list and len(extra) <= 2048, 'Explicit bounded additional reviewed text refs')
    for reference in extra:
        r.life.ref(reference); raw = text(reference['path'], reference['sha256'])
        require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Whole additional source bytes')
        required = merge(required, {reference['path']:reference['sha256']})
    complete, extension = disjoint_extension(inherited, required)
    completed.extension_shape(extension, completed_lfm_text=prior.COMPLETED_LFM_TEXT,
        catalog_ref=request['current_serial_catalog'], read=read, guard=guard)
    # Exercise the EXACT strict merge that runtime prerequisites will execute.
    actual = audited.merged_plan(base['plan'], extension)
    combined = auth.merge_extensions(base['audit_spec']['source_extension'], extension)
    require(audited.merged_plan(base['comparison_plan'], combined) == actual
        and actual['source_sha256'] == complete, 'Both original runtime reconstruction routes agree exactly')
    release_bytes = len(serial.encoded({key:actual[key] for key in ('source_contract','source_sha256')}))
    require(release_bytes+65536 <= original.PIPE_CAP, 'Original complete source release pipe reservation remains16MiB')
    view = dict(latest_completion=base['resolved']['plan']['latest_completion'], source_sha256=complete,
        serial_predecessors=request['current_serial_catalog'])
    serial_catalog = r.completion.latest.serial_catalog(view, read, guard)
    previous.catalog_rows(serial_catalog, base['old_catalog'], [(request['independent_audit_exit'], base['ended']['owner'], base['tool'])])
    witnessed = {}
    for row in serial_catalog['runs']:
        terminal = read(row['terminal'])
        if terminal.get('schema') in ('minicpm5_native_model_exit_recorder_actual_exit_v1','minicpm5_native_independent_audit_actual_exit_v1'):
            additions = previous.preparation.executed_extension(terminal, row, read, guard, base['witness_cache'])
            witnessed = merge(witnessed, additions)
    require(all(complete.get(path) == sha for path,sha in witnessed.items()), 'Complete actual recorder/auditor sources covered')
    pv = dict(schema='minicpm5_fixed_input_pv_serial_spec_v1', independent_audit_exit=request['independent_audit_exit'],
        current_serial_catalog=request['current_serial_catalog'], source_extension=extension, output_root=str(out),
        worker=diagnostic.RUNTIME_SOURCES[2], limits=copy.deepcopy(prior.LIMITS), artifact_limits=copy.deepcopy(serial.CAPS))
    guard()
    return dict(schema=RESULT_SCHEMA, source_metadata_only=True, pv_spec_candidate=pv,
        diagnostic_source_review=request['diagnostic_source_review'], source_binding=base['source_binding'],
        descriptor_plan_sha256=base['descriptor_plan_sha256'], complete_source_map_sha256=r.digest(complete),
        inherited_source_files=len(inherited), extension_source_files=len(extension), complete_source_files=len(complete),
        complete_source_release_bytes=release_bytes, original_pipe_reservation_bytes=65536,
        mandatory_sources={path:extension[path] for path in MANDATORY}, catalog_rows=len(serial_catalog['runs']),
        all_original_catalog_rows_retained=True, all_current_catalog_sources_retained=True,
        original_runtime_reconstruction_routes_agree=True, current_handoff_is_saved_metadata_only=True,
        actual_freshness_or_physical_checks_performed=False, original_runtime_full_authentication_still_required=True,
        current_runtime_spec_written=False, runtime_admitted=False, launch_authorized=False,
        full_model_qualified=False,new_speedup_claim=False)
