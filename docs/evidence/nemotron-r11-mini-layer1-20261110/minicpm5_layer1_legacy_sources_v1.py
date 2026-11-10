"""Check historical provenance against the current reviewed import closure.

Historical specs are immutable. A newly loaded helper can be present in an older
archive without belonging to that older run. Such helpers must match the current
review AND every applicable archive pin. No model or SDK imports occur here.
"""
from pathlib import Path
import sys
import minicpm5_original32_oproj_cpu_runner_v1 as oproj
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as layer0


def validate_source_maps(historical, current, loaded, archives, digest):
    require = oproj.p.require
    require(all(current.get(path) == sha for path, sha in historical.items()),
            'Every historical source pin retained in current review')
    for path in loaded:
        require(path in current, 'Loaded helper missing from current review')
        require(digest(path) == current[path], 'Loaded helper differs from current review')
        for archive in archives:
            if path in archive:
                require(current[path] == archive[path], 'Loaded helper differs from archive pin')


def validate_current_sources(spec, current_sources):
    loaded = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(oproj.ROOT) + '/'):
            loaded.add(str(Path(path).resolve()))
    validate_source_maps(spec['source_sha256'], current_sources, loaded, (), oproj.p.sha)


def check_oproj(spec, current_sources):
    previous_spec = oproj.p.read_ref(oproj.fullshape.SPEC)
    (bound, owners, prior) = oproj.fullshape.previous.check_prerequisites(previous_spec)
    (more, r2) = oproj.fullshape.prefill.r2_predecessors()
    owners += more
    owners.append(oproj.fullshape.prefill_predecessor(bound))
    owners.append(oproj.fullshape.projection_predecessor(bound))
    owners.append(oproj.fullshape_predecessor(bound, owners))
    for key in ('finalization', 'catalog', 'finalizer_exit'):
        oproj.same(spec[key], previous_spec[key], 'Same original finalized ' + key)
    oproj.same(spec['canonical_sha256'], previous_spec['canonical_sha256'], 'Canonical files retained')
    inherited = {}
    for mapping in (bound['plan']['source_sha256'], previous_spec['source_sha256']):
        for (path, wanted) in mapping.items():
            oproj.p.require(path not in inherited or inherited[path] == wanted, 'Consistent inherited source')
            inherited[path] = wanted
    inherited.update({oproj.fullshape.__file__: oproj.FULLSHAPE_SHA, oproj.fullshape.analysis.__file__: oproj.FULLSHAPE_WORKER_SHA, oproj.fullshape.previous.__file__: oproj.fullshape.PREVIOUS_SHA, oproj.fullshape.prefill.__file__: oproj.fullshape.PREFILL_RUNNER_SHA, oproj.fullshape.prefill.analysis.__file__: oproj.fullshape.PREFILL_ANALYSIS_SHA, oproj.fullshape.original_projection.__file__: oproj.fullshape.OPROJ_RUNNER_SHA, oproj.fullshape.original_projection.analysis.__file__: oproj.fullshape.OPROJ_WORKER_SHA})
    oproj.p.require(all((spec['source_sha256'].get(path) == wanted for (path, wanted) in inherited.items())), 'All predecessor sources retained')
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(oproj.ROOT) + '/'):
            path = str(Path(path).resolve())
            for archive in (bound['spec'], bound['catalog']):
                if path in archive['source_sha256']:
                    oproj.same(current_sources[path], archive['source_sha256'][path], 'Loaded helper retains archive pin')
    oproj.check_plan_extension(oproj.p.read_ref(spec['plan']), bound['plan'], spec['added_inputs'])
    return (bound, owners, dict(call4=prior, failed_r1=prior['failed_r1'], r2=r2))


def check_layer0(spec, current_sources):
    validate_current_sources(spec, current_sources)
    previous_spec = layer0.p.read_ref(layer0.PREVIOUS_SPEC)
    (bound, owners, priors) = check_oproj(previous_spec, current_sources)
    (more, ended) = layer0.fourarm_predecessor(owners)
    owners += more
    for key in ('finalization', 'catalog', 'finalizer_exit', 'canonical_sha256'):
        layer0.same(spec[key], previous_spec[key], 'Same original finalized ' + key)
    layer0.p.require(all((spec['source_sha256'].get(k) == v for (k, v) in previous_spec['source_sha256'].items())), 'Complete predecessor source closure retained')
    layer0.same(spec['source_sha256'].get(layer0.predecessor.__file__), layer0.PREVIOUS_RUNNER_SHA, 'Pinned predecessor runner')
    layer0.check_plan_extension(layer0.p.read_ref(spec['plan']), layer0.p.read_ref(layer0.PREVIOUS_PLAN), spec['added_inputs'])
    terminal = layer0.p.read_ref(layer0.GATHER_TERMINAL)
    completed = layer0.p.read_ref(layer0.GATHER_EXIT)
    layer0.p.require(terminal['passed'] and terminal['all12_new_payloads_independently_compared'] and (len(terminal['cases']) == 4), 'Complete actual local gather prerequisite')
    layer0.same(completed['terminal'], layer0.GATHER_TERMINAL, 'Latest actual terminal')
    layer0.same(completed['owner'], terminal['verifier'], 'Latest actual verifier')
    layer0.p.require(completed['actual_tool_exit_observed'] and completed['exact_owner_gone'] and (completed['actual_exit'] == dict(session_id=93505, exit_code=0, tool_chunk='24dfe8')), 'Latest actual successful external exit')
    for previous in [terminal['native_owner'], terminal['verifier'], *terminal['worker_owners']]:
        layer0.p.gone(previous)
    return (bound, owners, dict(inherited=priors, fourarm=ended, gather=terminal))
