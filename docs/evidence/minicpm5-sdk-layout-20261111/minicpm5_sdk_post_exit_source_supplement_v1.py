"""Explicit post-exit source supplement, separate from frozen model closure.

The root writes an independently reviewed immutable JSON only after these four
sources are final. No original plan/spec/review is changed. The two admission
functions below retain the original runner bodies with explicit runner binding
and only their loaded-source call replaced. The original loaded-source validator
itself runs unchanged on an ephemeral union used ONLY for source verification.
No import-time model, physical, process or native work.
"""
import hashlib
from pathlib import Path
import re
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as sources
from minicpm5_layer0_runtime_auth_v1 import digest, identity
ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'research'
require = sources.require
SCHEMA = 'minicpm5_sdk_post_exit_source_supplement_v1'
MODEL_EXIT = dict(path=str(Q / 'minicpm5-layer1-sdk-batch-layout-r2-actual-exit-20261111-v1.json'),
    sha256='aa4c60e6eff2b0800eb470f1d3dbdb8180abfc1727da9c3393b861c6f0b26a43')
SOURCE_NAMES = ('minicpm5_sdk_physical_evidence_v3.py',
    'audit_minicpm5_layer1_sdk_batch_layout_result_v5.py',
    'record_minicpm5_layer1_sdk_batch_layout_audit_exit_v5.py',
    'minicpm5_sdk_post_exit_source_supplement_v1.py')
SOURCE_PATHS = frozenset(str(Q / name) for name in SOURCE_NAMES)


def validate_review(review, model_exit, plan):
    require(type(review) is dict and set(review) == {'schema','reviewer','passed','blockers',
            'model_exit','model_source_map_sha256','source_sha256','original_model_sources_preserved',
            'frozen_model_closure_unchanged'}, 'Exact independent post-exit supplement schema')
    require(review['schema'] == SCHEMA and review['reviewer'] == '/root'
            and review['passed'] is True and review['blockers'] == [],
            'Independent root-approved complete post-exit source supplement')
    require(model_exit == MODEL_EXIT == review['model_exit']
            and review['model_source_map_sha256'] == digest(plan['source_sha256'])
            and review['original_model_sources_preserved'] is True
            and review['frozen_model_closure_unchanged'] is True,
            'Exact actual model exit and complete unmodified original model source map')
    pins = review['source_sha256']
    sources.metadata.pin_shape(pins)
    require(set(pins) == SOURCE_PATHS and not set(pins) & set(plan['source_sha256']),
            'Only four explicitly reviewed new post-exit sources; no model pin replaced')
    return pins


def verify_loaded_sources(runner, plan, guard, supplement_ref, model_exit):
    sources.reference(supplement_ref)
    path = Path(supplement_ref['path'])
    require(path.parent == Q and re.fullmatch(
            r'minicpm5-sdk-post-exit-source-supplement-[a-z0-9-]{1,80}\.json', path.name),
            'Explicit independent supplement review namespace')
    guard(); before = identity(path)
    review = sources.load(supplement_ref)
    require(identity(path) == before, 'Stable independent source supplement metadata')
    pins = validate_review(review, model_exit, plan)
    for source, sha in pins.items():
        guard(); old = identity(source)
        require(hashlib.sha256(sources.text(source, sha)).hexdigest() == sha
                and identity(source) == old, 'Every exact supplement source authenticated')
    original = digest(plan)
    # Only this explicit verification view includes extra sources. It is never
    # passed to completion, raw-byte verification, model or scientific checks.
    view = dict(plan, source_sha256=dict(plan['source_sha256'], **pins))
    loaded = runner.verify_loaded_sources(view, guard)
    require(digest(plan) == original, 'Frozen scientific plan unchanged by source verification')
    guard(); require(identity(path) == before, 'Supplement unchanged throughout source verification')
    return dict(loaded, post_exit_source_supplement=supplement_ref,
        model_source_map_sha256=review['model_source_map_sha256'],
        supplement_source_sha256=dict(pins), frozen_model_closure_unchanged=True)


def read_bundle(runner, spec_ref, review_ref, guard, supplement_ref, model_exit):
    guard()
    spec = runner.source_auth.load(spec_ref)
    runner.latest.require_complete_binding(spec['latest_completion'])
    review = runner.source_auth.load(review_ref)
    runner.p.require(spec['schema'] == 'minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2', 'Exact new observation runtime')
    runner.p.require(review['schema'] == 'minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2' and review['reviewer'] == '/root' and (review['passed'] is True) and (review['blockers'] == []), 'Final concrete runtime source review')
    runner.same(review['spec'], spec_ref, 'Exact reviewed runtime spec')
    runner.same(review['source_sha256'], spec['source_sha256'], 'Complete reviewed source closure')
    runner.same([spec['runner'], spec['worker']], [runner.SELF, str(runner.ROOT / 'minicpm5_layer1_sdk_batch_layout_cpu_worker_v4.py')], 'Only new collector runner and worker')
    partition = runner.source_auth.classify(spec['source_sha256'], spec['source_contract'])
    runner.source_auth.verify_text_sources(partition, guard)
    for path in (runner.SELF, spec['worker']):
        wanted = spec['source_sha256'].get(path)
        runner.same(hashlib.sha256(runner.source_auth.text(path, wanted)).hexdigest(), wanted, 'Both actual entrypoints pinned')
    plan = runner.source_auth.load(spec['plan'])
    runner.same(plan['source_sha256'], spec['source_sha256'], 'Scientific runtime source closure')
    runner.same(plan['source_contract'], spec['source_contract'], 'One exact reviewed source contract')
    runner.same(plan['output_root'], spec['output_root'], 'Exact output root')
    runner.p.require(Path(spec['output_root']).parent == Path('/mnt/nvme') and Path(spec['output_root']).name.startswith('minicpm5-layer1-sdk-batch-layout-'), 'Fresh layer1-only output namespace')
    runner.same([spec['output_cap_bytes'], spec['root_floor_bytes'], spec['output_floor_bytes']], [runner.CAP, 28 * 2 ** 30, 30 * 2 ** 30], 'Original output cap and current serial storage floors')
    runner.same(spec['resources'], runner.PROPS, 'Original16GiB/TP2 CPU resources')
    runner.p.require(type(spec['master_port']) is int and 29000 < spec['master_port'] < 30000, 'Finite rendezvous port')
    runner.p.require(re.fullmatch('trn1-mini-layer1-sdk-batch-layout-[a-z0-9-]{1,80}', spec['unit']) is not None, 'Fresh layer1 service')
    runner.p.require(spec['model_qualification_claim'] is spec['automatic_full_gates_launch'] is False, 'Diagnostic only')
    runner.same(spec['original_completed_collection'], runner.OBSERVED_EXIT, 'Actual completed original trajectory')
    runner.same(spec['prepared_source_plan'], runner.RESULTS_SOURCE_PLAN, 'Reviewed complete observation/validator source plan')
    runner.same(spec['latest_completion'], plan['latest_completion'], 'Latest complete external r12 and MiniCPM lineage')
    runner.same(spec['completed_current_history'], plan['layer1_sdk_batch_layout'], 'Completed full180-call current/history prerequisite')
    runner.same(spec['completed_native_cache'], plan['layer1_official_history'], 'Completed native capture/audit/comparison')
    runner.same(spec['completed_official_history'], plan['layer1_current_history_factorial'], 'Completed original official history')
    runner.same(spec['completed_storage'], plan['completed_storage_maintenance'], 'Completed storage maintenance and audit')
    runner.same(spec['completed_factorial'], plan['layer1_native_cache'], 'Completed independent K/V matrix')
    runner.same(spec['completed_cross_history'], plan['layer1_kv_factorial'], 'Actual completed paired-history and audit')
    runner.same(spec['completed_own_history'], plan['layer1_cross_history'], 'Actual completed own-history plus post-exit audit')
    runner.same(spec['completed_layer1_data'], plan['layer1_own_history_replay'], 'Actual collection and audit binding')
    verify_loaded_sources(runner, plan, guard, supplement_ref, model_exit)
    return (spec, plan)


def admitted_bundles(runner, plan, guard, supplement_ref, model_exit):
    guard()
    runner.latest.require_complete_binding(plan['latest_completion'])

    def read(ref):
        guard()
        row = plan['inputs'][ref['path']]
        runner.same(row['sha256'], ref['sha256'], 'Admitted completion metadata digest')
        before = identity(ref['path'])
        runner.same(before, row['identity'], 'Admitted completion metadata identity')
        value = runner.source_auth.load(ref)
        runner.same(identity(ref['path']), before, 'Stable completion metadata read')
        return value
    done = runner.latest.completed(plan['latest_completion'], read, guard)
    runner.validate_current_history_extension(plan, done)
    runner.same(done['current_history']['binding'], plan['layer1_sdk_batch_layout'], 'Latest authenticated current/history')
    runner.same(done['native']['binding'], plan['layer1_official_history'], 'Latest authenticated native cache')
    for (path, sha) in done['source_sha256'].items():
        runner.same(plan['source_sha256'].get(path), sha, 'Completed sources retained in admitted plan')
    verify_loaded_sources(runner, plan, guard, supplement_ref, model_exit)

    def read_json(path, sha):
        return read(dict(path=path, sha256=sha))
    return (done, read_json)
