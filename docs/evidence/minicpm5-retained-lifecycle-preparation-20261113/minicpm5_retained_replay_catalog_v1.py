"""Pure durable lifecycle composition using injected authenticated metadata only.

An actual outer and its independent observer are two tool runs. The pair/replay
services belong to that one outer; they never receive invented tool sessions.
All old catalog rows are retained verbatim. Scientific success is not inferred
from a catalog append, and no deletion or next experiment is admitted here.
"""
import copy
from pathlib import PurePosixPath as Path
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_replay_observer_v2 as observer


def process(value):
    c.require(type(value)is dict and type(value.get('pid'))is int and value['pid']>0
        and str(value.get('start_ticks','')).isdigit(),'Complete observed process identity')
    return copy.deepcopy(value)


def compose(spec,resolved,previous,observation_ref,observation,live_ref,live,actual_observer_tool,
            updated_at,finalization,admission,terminal,read,guard):
    """Return four exact objects/refs, with the idle handoff published last.

    read and guard are explicit callbacks. No filesystem access, History
    resolution, process inspection, or output publication takes place here.
    """
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    c.validate_spec(spec);guard()
    c.require(admission['spec']==observation['spec']and admission['review']==observation['review']
        and admission['current_serial_catalog']==spec['current_serial_catalog']
        and observation['lock_identity']==c.LOCK,'Exact same admitted lifecycle binding')
    checked=observer.reconcile(terminal,previous,observation['actual_outer_exit']['actual_tool'])
    c.require(checked==observation['actual_outer_exit'],'Every actual outer/phase exit retained')
    c.require(actual_observer_tool['session_id']!=checked['actual_tool']['session_id'],
        'Independent actual observer uses its own externally observed tool session')
    c.require(observation['running']['path']==spec['running_paths'][previous['minicpm5_retained_replay_current_execution']['phase']]
        and observation['outer_terminal']['path']==str(Path(spec['output_root'])/'outer-terminal.json')
        and observation_ref['path']==str(Path(spec['observer_root'])/'observation.json')
        and live_ref==observation['independent_live']and live_ref['path']==str(Path(spec['output_root'])/'observer-live.json'),
        'All exact prebound runtime evidence namespaces')
    c.require(finalization['final_owned']['all_declared_units']==sorted(spec['units'].values())
        and observation['final_owned']['all_declared_units']==sorted(spec['units'].values()),
        'Both observers retain both declared units including any unlaunched phase')
    if checked['passed']:
        equality_ref=observation['full_replay_equality'];c.metadata_reference(equality_ref)
        c.require(equality_ref['path']==str(Path(spec['observer_root'])/'full-replay-equality.json'),
            'Exact independent complete-science comparison')
        equality=read(equality_ref)
        c.require(equality['schema']=='minicpm5_retained_raw_complete_replay_equality_v1'
            and all(equality[k]is True for k in ('complete','full11859_logical_inputs_preserved',
                'full32_replay_and_all_scientific_outputs_equal','all_original_failures_preserved',
                'layouts98_byte_identical','scientific_reports14_byte_identical'))
            and tuple(equality[k]for k in ('full_frames32','cache_pairs9408','cpu_hook_pairs5964','stage_vectors13632'))
                ==(32,9408,5964,13632)
            and equality['full_model_qualified']is equality['new_speedup_claim']is False,
            'Complete original replay evidence; never metadata-only scientific success')
    else:c.require(observation['full_replay_equality']is None,'Failed outer has no successful equality claim')

    catalog=read(spec['current_serial_catalog']);overlay=read(spec['history_overlay'])
    c.require(overlay['current_serial_catalog']==spec['current_serial_catalog']
        and resolved['plan']['serial_predecessors']==spec['current_serial_catalog']
        and resolved['plan']['source_sha256']==spec['source_sha256'],
        'Complete actually resolved source map equals the reviewed spec')
    # The full recovered partition was authenticated by History.resolve(). It
    # contains extra recovered sources beyond base_plan; never subtract only base.
    delta=copy.deepcopy(overlay['source_extension']);pins=spec['source_sha256']
    c.require(all(pins.get(p)==h for p,h in delta.items()),'Exact authenticated overlay source delta')
    recovered={p:h for p,h in resolved['plan']['source_sha256'].items()if p not in delta}
    base=read({k:overlay['base_plan'][k]for k in ('path','sha256')})
    c.require(all(recovered.get(p)==h for p,h in base['source_sha256'].items())
        and dict(recovered,**delta)==pins,'Reconstruct full recovered plus extension map without replacements')
    original=latest.serial_catalog(dict(serial_predecessors=spec['current_serial_catalog'],source_sha256=pins),read,guard)
    c.require(original['runs']==catalog['runs']and len(catalog['runs'])+2<=latest.MAX_SERIAL_RUNS,
        'Preserve all old rows and reserve both actual tool rows within original limit')
    c.require(set(c.SUCCESSOR_FILES)<=set(delta),'Every new lifecycle source retained in the next catalog')
    # Old catalog rows already retain the bulk of this overlay extension. Append
    # each new source exactly once, without copying thousands of historical pins
    # into both new rows and exceeding the unchanged 8 MiB catalog cap.
    prior_pins=original['source_sha256']
    c.require(all(delta.get(p)==h for p,h in prior_pins.items()),'All earlier catalog source pins remain in the extension')
    added={p:h for p,h in delta.items()if p not in prior_pins}

    owners=[process(terminal['outer_owner'])];phases=terminal['phase_states']
    c.require([r['phase']for r in phases]in (['pair'],['pair','replay']),
        'Complete ordered actual phase prefix, without omitted or duplicate phases')
    # A failed terminal may omit an owner only for a phase never acknowledged.
    # Every acknowledged worker is independently retained in the running chain.
    execution=previous['minicpm5_retained_replay_current_execution']
    acknowledged={execution['phase']:execution['worker']}
    if execution['phase']=='replay':
        pair_ref=previous['previous'];c.reference(pair_ref)
        c.require(pair_ref['path']==spec['running_paths']['pair'],'Exact retained pair running predecessor')
        pair=read(pair_ref);pair_execution=pair['minicpm5_retained_replay_current_execution']
        c.require(pair['schema']=='minicpm5_retained_replay_running_v2'and pair['running']is True
            and pair['active_serial']==c.active_descriptor(pair_execution)
            and pair_execution['phase']=='pair'and pair_execution['tool']==execution['tool']
            and pair_execution['admission']==observation['admission']
            and pair['previous']==admission['previous'],'Full same-outer pair running chain')
        acknowledged['pair']=pair_execution['worker']
    else:c.require(previous['previous']==admission['previous'],'Full pair-to-idle admission chain')
    by_phase={row['phase']:row for row in phases}
    c.require(all(phase in by_phase and by_phase[phase]['owner']==owner for phase,owner in acknowledged.items()),
        'Every previously acknowledged worker owner remains in the actual terminal')
    for row in phases:
        phase=row['phase'];owner=row['owner']
        c.require(row['unit']==spec['units'][phase]and row['nonce']==admission['attempts'][phase],
            'Every actual phase retains its declared unit and nonce')
        if owner is None:
            c.require(not checked['passed']and row['exit']is None,'No invented owner/exit for an unobserved failed install')
            continue
        c.require(owner['unit']==row['unit']and owner['nonce']==row['nonce']
            and owner['process']['argv']==c.phase_argv(phase,observation['admission'])
            and owner['process']['pid']==owner['owner']['pid']
            and str(owner['process']['start_ticks'])==str(owner['owner']['start_ticks'])
            and (row['exit']is None or row['exit']['owner']==owner),'Complete actual phase process and retained exit')
        owners.append(process(owner['process']))
    all_owners=owners+[process(observation['observer_owner'])]
    c.require(len({(x['pid'],str(x['start_ticks']))for x in all_owners})==len(all_owners),
        'No substituted or repeated actual lifecycle owner')
    units=sorted(spec['units'].values())
    shared=dict(reviewer='/root',actual_tool_exit_observed=True,owners_gone=True,physically_drained=True,
        lock_identity=copy.deepcopy(c.LOCK),spec=copy.deepcopy(observation['spec']),review=copy.deepcopy(observation['review']),
        observation=copy.deepcopy(observation_ref),final_publication=copy.deepcopy(finalization),
        complete_source_map_sha256=c.sha(c.canonical(pins)),source_delta_sha256=c.sha(c.canonical(delta)),
        recovered_sources=len(recovered),source_delta_sources=len(delta),full_sources=len(pins),
        newly_recorded_sources=len(added),
        full_model_qualified=False,new_speedup_claim=False,deletion_authorized=False,automatic_successor=False)
    outer=dict(shared,schema='minicpm5_retained_replay_root_actual_outer_exit_v1',
        actual_tool=copy.deepcopy(checked['actual_tool']),actual_outer_exit_observed=True,
        owners=owners,units=units,all_declared_units=units,outer_terminal=copy.deepcopy(observation['outer_terminal']),
        complete_actual_phase_states=copy.deepcopy(phases),actual_outer_result=copy.deepcopy(checked),
        full_replay_equality=copy.deepcopy(observation['full_replay_equality']),passed=checked['passed'])
    paths=spec['publication_paths'];outer_ref=c.publication_ref(paths['outer_actual_exit'],outer)
    independent=dict(shared,schema='minicpm5_retained_replay_root_actual_observer_exit_v1',
        actual_tool=copy.deepcopy(actual_observer_tool),owners=[process(observation['observer_owner'])],units=[],
        independent_live=copy.deepcopy(live_ref),actual_outer_exit=outer_ref,passed=True,
        original_outer_failure_preserved=not checked['passed'])
    observer_ref=c.publication_ref(paths['observer_actual_exit'],independent)
    idle=observer.render_completed(previous,observation_ref,observation,live_ref,live,actual_observer_tool,updated_at,finalization)
    idle['minicpm5_retained_replay_completed'].update(actual_outer_terminal=outer_ref,actual_observer_terminal=observer_ref,
        successor_catalog_path=paths['successor_catalog'],catalog_predecessor=copy.deepcopy(spec['current_serial_catalog']))
    idle.update(complete_current_serial_catalog=copy.deepcopy(spec['current_serial_catalog']),
        complete_current_serial_catalog_next=paths['successor_catalog'])
    idle_ref=c.publication_ref(spec['terminal_handoff_path'],idle)
    new=copy.deepcopy(catalog);new['current_handoff']=idle_ref
    new['runs'] += [dict(terminal=ref,tool=copy.deepcopy(value['actual_tool']),owners=copy.deepcopy(value['owners']),
        units=copy.deepcopy(value['units']),source_sha256=copy.deepcopy(row_delta))
        for ref,value,row_delta in ((outer_ref,outer,added),(observer_ref,independent,{}))]
    catalog_ref=c.publication_ref(paths['successor_catalog'],new)
    c.require(new['runs'][:-2]==catalog['runs'],'Every previous catalog row retained verbatim and in order')
    objects=[(outer_ref,outer),(observer_ref,independent),(catalog_ref,new),(idle_ref,idle)]
    pending={ref['path']:(ref,value)for ref,value in objects}
    c.require(len(pending)==4 and not set(pending)&({r['terminal']['path']for r in catalog['runs']}|
        set(pins)|set(spec['input_extension'])|{spec['current_serial_catalog']['path'],catalog['current_handoff']['path']}),
        'Fresh disjoint append-only publication paths')
    def virtual_read(ref):
        if ref['path']not in pending:return read(ref)
        expected,value=pending[ref['path']];c.require(ref==expected,'Exact composed publication reference');return copy.deepcopy(value)
    # Exercise the original downstream catalog consumer, without replacing any
    # of its schema, tool, owner, unit, source or cardinality checks.
    checked_catalog=latest.serial_catalog(dict(serial_predecessors=catalog_ref,source_sha256=pins),virtual_read,guard)
    c.require(checked_catalog['runs']==new['runs']and checked_catalog['current_handoff']==idle_ref,
        'Original catalog consumer accepts the complete successor')
    guard();return dict(objects=objects,outer_actual_exit=outer_ref,observer_actual_exit=observer_ref,
        catalog=catalog_ref,idle_handoff=idle_ref,source_delta=delta,new_source_delta=added,full_sources=len(pins),
        catalog_rows=len(new['runs']),retained_replay_passed=checked['passed'],deletion_authorized=False,new_speedup_claim=False)
