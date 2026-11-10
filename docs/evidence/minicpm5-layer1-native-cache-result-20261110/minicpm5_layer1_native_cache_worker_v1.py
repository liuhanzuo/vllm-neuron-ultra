"""Single official native worker; parent supplies serial RuntimeContext."""
from research import minicpm5_layer1_after_saved_audit_v3 as collected
from research import minicpm5_layer1_after_own_history_v1 as own
from research import minicpm5_layer1_after_cross_history_v1 as cross
from research import minicpm5_layer1_after_kv_factorial_v1 as factorial
from research import minicpm5_layer1_native_cache_driver_v1 as driver
from minicpm5_layer0_runtime_auth_v1 import require,identity


def run(context):
    context.guard('native');require(context._sources_checked,'Authenticate all sources before native imports')
    observed={}
    def read(ref):
        value=context.read_json(ref['path'],ref['sha256']);observed[ref['path']]=context.plan['inputs'][ref['path']]['identity'];return value
    bundles=[(collected,collected.completed(context.plan['layer1_own_history_replay'],read)),
        (own,own.completed(context.plan['layer1_cross_history'],read)),
        (cross,cross.completed(context.plan['layer1_kv_factorial'],read)),
        (factorial,factorial.completed(context.plan['layer1_native_cache'],read))]
    original=bundles[-1][1]['plan'];allowed={'source_sha256','inputs','output_root','layer1_native_cache'}
    require({k:v for k,v in context.plan.items()if k not in allowed}=={k:v for k,v in original.items()if k not in allowed},
        'All original scientific inputs and acceptance gates')
    for key in ('source_sha256','inputs'):
        require(all(context.plan[key].get(path)==value for path,value in original[key].items()),'Every original source and input retained')
    def guard():
        context.guard('native')
        require(all(identity(path)==expected for path,expected in observed.items()),'Actual prerequisite metadata identities retained')
    for module,bundle in bundles:module.physical(bundle,guard)
    return driver.run(context,guard)
