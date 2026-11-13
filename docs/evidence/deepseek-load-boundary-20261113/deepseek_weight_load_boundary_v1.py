"""Partial archive/load evidence at the original post-native load boundary.

No CLI, launcher, admission or inference release. Original fresh-process checks
must already have run in the dedicated off/on successor. Every normal observed
load is stopped inside load_weights after retaining partial evidence, because
the accepted descriptor/CPU-view tools do not prove loaded layout/graph ABI.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import deepseek_weight_load_observer_v2 as observer
import deepseek_weight_serialized_descriptor_v1 as serialized

require=observer.require


class LayoutSemanticsUnresolved(RuntimeError):
    """Expected partial-evidence stop; never a successful model/retirement exit."""


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_binding(value,arm,artifact,expected_sha256):
    require(type(value)is dict and set(value)=={'schema','arm','archive','descriptor_limits','checkpoint_provenance'}
        and value['schema']=='deepseek_weight_load_boundary_binding_v1'and value['arm']==arm and arm in ('off','on'),
        'Exact externally authenticated arm-specific load-boundary binding')
    archive=value['archive'];require(type(archive)is dict and set(archive)=={'path','sha256','identity'}
        and archive['path']==str(Path(artifact)/'model.pt')and archive['sha256']==expected_sha256
        and type(expected_sha256)is str and re.fullmatch('[0-9a-f]{64}',expected_sha256),
        'Descriptor and original SDK load must bind the same exact original authenticated model.pt')
    identity=archive['identity'];require(type(identity)is dict and set(identity)==
        {'device','inode','size','mtime_ns','ctime_ns','mode','nlink'}and all(type(v)is int for v in identity.values())
        and identity['size']>0 and identity['nlink']==1,'Complete independently observed archive identity')
    limits=value['descriptor_limits'];require(type(limits)is dict and set(limits)=={'archive_bytes','pickle_bytes','members','code_bytes'}
        and all(type(v)is int and v>0 for v in limits.values())and identity['size']<=limits['archive_bytes'],
        'Explicit original-reservation-compatible finite descriptor caps')
    require(type(value['checkpoint_provenance'])is dict and value['checkpoint_provenance'],
        'Externally authenticated complete original checkpoint provenance required')
    return deepcopy(value)


def partial_evidence(bound,descriptor,observation):
    """Pure exact input/key linkage only; do not infer physical ABI semantics."""
    require(descriptor['schema']=='deepseek_serialized_weight_descriptor_v1'
        and descriptor['artifact_binding']==bound['archive']and descriptor['whole_archive_sha256_verified']is True
        and descriptor['all12_graphs_present']is True and len(descriptor['graphs'])==12,
        'Complete actual bound archive descriptor')
    for key in ('loaded_values_observed','graph_weight_ABI_semantics_proven','layout_semantics_proven',
                'inference_admitted','full_model_qualified','new_speedup_claim'):
        require(descriptor[key]is False,'Saved archive descriptor is partial evidence only')
    require(observation['schema']==observer.SCHEMA and observation['arm']==bound['arm']
        and observation['checkpoint_provenance']==bound['checkpoint_provenance']
        and observation['original_loader_calls']==observation['original_shard_calls']==1,
        'Exactly one actual original loader and sharder with bound checkpoint provenance')
    for key in ('graph_weight_ABI_proven','layout_semantics_proven','inference_admitted','full_model_qualified','new_speedup_claim'):
        require(observation[key]is False,'CPU observations cannot release graph/layout semantics')
    require(observation['weight_loader_present']==(descriptor['wlt']is not None),'Saved and actual WLT presence must agree')
    graph_keys={row['tensor']['checkpoint_key']for row in descriptor['graphs'][0]['weight_inputs']}
    require(graph_keys and all({row['tensor']['checkpoint_key']for row in graph['weight_inputs']}==graph_keys
        for graph in descriptor['graphs']),'All12 complete named graph weight interfaces')
    extras={}
    for name in ('pre_layout_shards','loaded_cpu_views'):
        rows=observation[name];observer._semantic_rows(rows)
        extras[name]=[]
        for row in rows:
            keys={entry['key']for entry in row['entries']}
            require(graph_keys<=keys,'Every declared graph weight key observed in each original TP rank')
            extras[name].append(dict(rank=row['rank'],additional_observed_keys=sorted(keys-graph_keys)))
    return dict(schema='deepseek_bound_load_partial_evidence_v1',arm=bound['arm'],load_binding=deepcopy(bound),
        serialized_descriptor=deepcopy(descriptor),serialized_descriptor_sha256=digest(descriptor),
        actual_load_observation=deepcopy(observation),actual_load_observation_sha256=digest(observation),
        same_authenticated_archive_selected=True,archive_identity_unchanged_across_load=True,
        all12_declared_weight_keys_observed_in_both_ranks=True,named_key_coverage_only=True,
        additional_observed_keys=extras,release_gate='loaded-layout-and-graph-weight-ABI-semantics-unresolved',
        graph_weight_ABI_proven=False,layout_semantics_proven=False,inference_admitted=False,
        full_model_qualified=False,new_speedup_claim=False,actual_producer_exit_observed=False)


def compare_bound_evidence(off,on):
    """Compare actual bound partial reports without turning them into release."""
    for arm,value in (('off',off),('on',on)):
        require(value['arm']==arm,'Independent original off/on reports in exact arm order')
        require(value==partial_evidence(value['load_binding'],value['serialized_descriptor'],value['actual_load_observation']),
            'Complete unchanged partial evidence and exact descriptor/observation digests')
    loaded=observer.compare_observations(off['actual_load_observation'],on['actual_load_observation'])
    saved=serialized.compare_descriptors(off['serialized_descriptor'],on['serialized_descriptor'])
    return dict(schema='deepseek_bound_load_partial_comparison_v1',off_sha256=digest(off),on_sha256=digest(on),
        actual_load_comparison=loaded,serialized_comparison=saved,
        next_required_observation=dict(
            original12_and_capture12_compiler_input_hlo='Complete actual per-bucket model/graph.hlo tied to archived graph.neff',
            actual_priority_layout_stub='Actual wrapped_neff.hlo and complete original weight_name_to_idx',
            actual_wlt_compiler_input='layout_opt/model/graph.hlo, metaneff and graph.neff tied to archived WLT',
            compile_binding='Fixed compiler/SDK/runtime binaries, flags and actual input/returned-NEFF producer witness',
            semantic_checks='All weight parameter positions/layouts and original logits+72KV dependence DAGs; only exact468 capture outputs may be additional'),
        graph_weight_ABI_proven=False,layout_semantics_proven=False,inference_admitted=False,
        full_model_qualified=False,new_speedup_claim=False)


@contextmanager
def observe_bound_load(model_class,arm,artifact,expected_sha256,binding,execution_guard,retain_partial):
    """Install only after original native imports; stop before load returns.

retain_partial is the authenticated owning worker's evidence sink, not a release
callback. Its return value cannot authorize inference. Parent authentication,
full reservations, actual producer exit and drainage remain independent duties.
"""
    require(callable(execution_guard)and callable(retain_partial),'Owning guard and partial-evidence sink required')
    bound=validate_binding(binding,arm,artifact,expected_sha256)
    execution_guard()
    descriptor=serialized.read_archive(bound['archive'],execution_guard,bound['descriptor_limits'])
    execution_guard()
    def observed(value):
        execution_guard()
        require(serialized.file_identity(Path(bound['archive']['path']).stat())==bound['archive']['identity'],
            'Authenticated archive identity changed during the actual original SDK load')
        report=partial_evidence(bound,descriptor,value)
        execution_guard();retain_partial(deepcopy(report));execution_guard()
        raise LayoutSemanticsUnresolved('Partial archive/load evidence retained; loaded-layout and graph-weight-ABI semantics remain unproven')
    with observer.observe_load(model_class,arm,bound['checkpoint_provenance'],execution_guard,observed):
        yield
    # A callback return (including True) is never a release mechanism.
    raise LayoutSemanticsUnresolved('Original load ended without the required unresolved-layout stop')
