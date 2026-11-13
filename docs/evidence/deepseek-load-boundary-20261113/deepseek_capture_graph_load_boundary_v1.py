"""New capture-on load boundary; original fresh gates and complete load body.

No compile entry or CLI. The original driver is not released by partial layout
observations. All original authentication/configuration/runtime calls remain.
"""
import hashlib
import json
from deepseek_capture_graph_entry_v1 import (require,_begin,ordinary,REPORT_NAME,SCHEMA,MODEL_PATH,
    ORIGINAL_CONFIG_SHA256,json_bytes,TAGS,file_record,_native_dependencies,check_config,_inspect,
    check_environment,check_wrapper_configs)

def load(artifact, cache, workdir, expected_report_sha256, execution_guard, *, weight_load_binding, retain_weight_partial):
    """Authenticate completed build evidence, load without warmup, return model.

expected_report_sha256 comes from the owning worker's independent actual-exit
audit, never from hashing an untrusted report immediately before this call.
Full observer-off/on raw neutrality, lifecycle144 and the original failed cold
gate remain mandatory downstream. Loading is not inference qualification.
"""
    (artifact, cache, workdir), environment, expected = _begin(
        artifact, cache, workdir, execution_guard, fresh=False)
    require(isinstance(expected_report_sha256, str) and len(expected_report_sha256) == 64 and
            all(c in '0123456789abcdef' for c in expected_report_sha256), 'externally authenticated build-report digest required')
    report_path = ordinary(artifact / REPORT_NAME)
    require(report_path.stat().st_size <= 4 * 1024**2, 'bounded build report required')
    raw = report_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_report_sha256, 'authenticated build-report SHA256 mismatch')
    report = json.loads(raw)
    require(report['schema'] == SCHEMA and report['complete'] is True and
            (report['artifact'], report['cache'], report['workdir']) == tuple(map(str, (artifact, cache, workdir))) and
            report['environment'] == environment and report['model_path'] == str(MODEL_PATH) and
            report['original_config_sha256'] == ORIGINAL_CONFIG_SHA256 and
            report['config_sha256'] == hashlib.sha256(json_bytes(expected)).hexdigest(), 'exact build identity required')
    require(all(report[key] is False for key in ('full_model_qualified', 'new_speedup_claim',
            'output_semantics_neutrality_proven', 'baseline_weight_layout_equality_proven')),
            'build evidence cannot grant model/neutrality/weight-layout qualification')
    # Require every file, not an arbitrary subset from a report. Reject any
    # untrusted path before hashing and before SDK or torch.jit deserialization.
    expected_files = {str(p) for p in (artifact / 'model.pt', artifact / 'neuron_config.json', workdir / 'neuron_config.json')}
    expected_files.update(str(workdir / tag / f'_tp0_bk{i}' / name)
                          for tag in TAGS for i in range(6) for name in ('neuron_config.json', 'metaneff.pb', 'graph.neff'))
    require(set(report['files']) == expected_files, 'complete exact 39-file build identity required')
    for path, identity in report['files'].items():
        execution_guard()
        require(file_record(path) == identity, 'compiled artifact/config/graph changed: ' + path)
    execution_guard()
    config_cls, model_cls, meta_cls = _native_dependencies()
    cfg = config_cls.from_json_string(json.dumps(expected))
    check_config(json.loads(cfg.to_json_string()), expected)
    model = model_cls(str(MODEL_PATH), config=cfg)
    files, graphs = _inspect(model, artifact, workdir, expected, meta_cls, execution_guard)
    require(files == report['files'] and graphs == report['graphs'], 'actual graph metadata must match authenticated build')
    execution_guard()
    check_environment(cache, workdir)
    from deepseek_weight_load_boundary_v1 import observe_bound_load
    with observe_bound_load(model_cls,'on',artifact,report['files'][str(artifact/'model.pt')]['sha256'],
            weight_load_binding,execution_guard,retain_weight_partial):
        model.load(str(artifact), skip_warmup=True)
    execution_guard()
    check_wrapper_configs(model, expected)
    require(model.is_loaded_to_neuron is True, 'completed SDK load required')
    from qwen35_split_state_checks_v4 import graph_routes
    from deepseek_prior_mask_state_v1 import bindings
    routes, states = graph_routes(model), bindings(model)
    require(routes['complete'] and len(routes['graphs']) == 12 and len(states) == 144,
            'all loaded graph routes and TP state objects required')
    execution_guard()
    return model
