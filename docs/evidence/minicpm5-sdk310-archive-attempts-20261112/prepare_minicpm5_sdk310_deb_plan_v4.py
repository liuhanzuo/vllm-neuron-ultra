"""Metadata-only plan composition after actual successful LFM r2 termination.

No launch, process/service probe, lock, transport, native or payload operation.
The optional CLI writes one unreviewed plan; it never creates a root review,
handoff, catalog, admission or an actual-exit claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q),str(Q.parent)]
import dory_serial_stage_contract_v2 as c
import minicpm5_sdk310_deb_input_binding_v1 as binding
import lfm3_full_cpu_successor_history_v3 as history
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement

MANIFEST = dict(path=str(Q/'minicpm5-sdk310-deb-source-manifest-20261112-v1.json'),sha256='483244adb6cc70d2ed84a9de3c9985627ea3c3d9054c5caa4f6f4d8575060f72')
ACCEPTANCE = dict(path=str(Q/'minicpm5-sdk310-deb-root-source-review-20261112-v1.json'),sha256='d74a1a80f536546304c7053e3a24a96a632a4f72781c26aaa17200f2d45b4a46')
BASE = dict(path=str(Q/'dory-six-deb-r3-20261111-plan.json'),sha256='7bff3e0748de0de975d27edc519c97239c981fb5d2878e150c5c3fe3593c574f')
HISTORY_SOURCE = dict(path=str(Q/'lfm3_full_cpu_successor_history_v3.py'),sha256='b5d957a6aaae30efefaf1a266897824b34986c61ff65579ef4ef6c7375fb821c')
AUDITOR_SOURCE = dict(path=str(Q/'audit_lfm3_full_cpu_r2_terminal_root_v2.py'),sha256='fc4749e5a8508b2e8b59ab6703f0c595c60c24d2613032782c1aaea4cf730903')
RETIREMENT_MANIFEST = dict(path=str(Q/'minicpm5-sdk310-deb-source-manifest-20261112-v3.json'),sha256='2fe3541566bcd56932df09b808f2f5fcee6ee2d8a28235c308f92b2f02a184de')
PREVIOUS_RETIREMENT_MANIFEST = dict(path=str(Q/'minicpm5-sdk310-deb-source-manifest-20261112-v2.json'),sha256='9b831d485949dc097b646de516a4d644909d7c14740f356203ef814185823163')
PREVIOUS_RETIREMENT_ACCEPTANCE = dict(path=str(Q/'minicpm5-sdk310-deb-retirement-root-source-review-20261112-v1.json'),sha256='66e562e81b87641cb5c769a746a724ab669310c53e5ac1f0851926280f57574a')


def same_ref(left,right):
    return all(left.get(k)==right.get(k) for k in ('path','sha256'))


def reference(value):
    history.reference(value)
    return value


def owner_key(owner):
    c.require(type(owner.get('pid')) is int and owner['pid']>0 and type(owner.get('start_ticks')) is str and owner['start_ticks'].isdigit(), 'Original PID/start identity')
    return owner['pid'],owner['start_ticks']


def complete_history(base,catalog,scope):
    """Keep every old owner row, including differing historical argv observations."""
    owners=copy.deepcopy(base['predecessor_owners'])
    services=copy.deepcopy(base['predecessor_services']); known={row['unit']:row.get('invocation_id') for row in services}
    c.require(len(known)==len(services), 'Unique base service names')
    def service(unit, invocation):
        c.require(type(unit) is str and unit and not unit.endswith('.service'), 'Original service names without suffix')
        if unit not in known:
            services.append(dict(unit=unit,invocation_id=invocation));known[unit]=invocation
        elif invocation:
            c.require(known[unit] in (None,'',invocation), 'No conflicting historical invocation IDs')
            if known[unit] in (None,''):
                # Preserve the original unknown-invocation row; add the stronger check.
                services.append(dict(unit=unit,invocation_id=invocation));known[unit]=invocation
    for run in catalog['runs']:
        owners.extend(copy.deepcopy(run['owners']))
        for unit in run['units']:
            invocations={o['invocation_id'] for o in run['owners'] if o.get('unit')==unit and o.get('invocation_id')}
            c.require(len(invocations)<=1, 'One recorded invocation for a catalog unit')
            service(unit,next(iter(invocations)) if invocations else None)
    # Catalog row has only the two main owners. Keep all 874 supplemental children.
    owners.extend(copy.deepcopy(scope['children']))
    for row in scope['predecessor_services']:service(row['unit'],row['invocation_id'])
    keys={owner_key(o) for o in owners}
    c.require(all(owner_key(o) in keys for o in scope['predecessor_owners']), 'All 876 r2 owners are included')
    c.require(owners[:len(base['predecessor_owners'])]==base['predecessor_owners'], 'Every previous owner observation retained unchanged')
    c.require(services[:len(base['predecessor_services'])]==base['predecessor_services'], 'Every previous service observation retained unchanged')
    return owners,services


def build(read, catalog_ref, scope_ref, predecessor_ref, mode, output_root,
          replay_from=None, producer_exit_ref=None, producer_recorder_exit_ref=None, retirement_acceptance_ref=None):
    """Return one plan only after authenticated real terminal metadata exists.

    read(ref)->dict must enforce full hash, stable identity and bounded JSON IO.
    The caller/root still reviews this exact plan and performs fresh physical checks.
    """
    for ref in (catalog_ref,scope_ref,predecessor_ref,retirement_acceptance_ref):reference(ref)
    c.require(mode in ('download','replay'), 'Exact archive mode')
    root=Path(output_root)
    c.require(root.parent==Path('/mnt/nvme') and root.name.startswith('dory-serial-inventory-minicpm5-sdk310-') and root.name not in ('.','..'), 'Fresh Mini archive inventory namespace')
    c.require(history.VERIFIERS.get(AUDITOR_SOURCE['path']) == AUDITOR_SOURCE['sha256'], 'Latest full r2 auditor and model_result_reads_saved_JSON_only contract')
    manifest,accepted,base=read(MANIFEST),read(ACCEPTANCE),read(BASE)
    c.require(accepted['reviewer']=='/root' and accepted['passed'] is True and accepted['blockers']==[] and same_ref(accepted['manifest'],MANIFEST), 'Accepted frozen Mini source review')
    c.require(len(manifest['source_sha256'])==88 and manifest['complete_retained_source_count']==88 and manifest['scientific_constraints']==binding.CONSTRAINTS and manifest['archive_stage_budget']==c.INVENTORY, 'Exact 88-pin source and original separate envelopes')
    c.require(base['schema']=='dory_serial_stage_plan_v1' and base['stage']=='inventory' and base['budget']==c.INVENTORY, 'Frozen prior archive plan as history/lifecycle base only')
    successor=read(RETIREMENT_MANIFEST);successor_accepted=read(retirement_acceptance_ref)
    c.require(successor_accepted['reviewer']=='/root' and successor_accepted['passed'] is True
              and successor_accepted['blockers']==[] and same_ref(successor_accepted['manifest'],RETIREMENT_MANIFEST),
              'Separate exact root acceptance of the 98-pin narrow retirement successor')
    c.require(successor['complete_retained_source_count']==len(successor['source_sha256'])==98
              and all(successor['source_sha256'].get(path)==sha for path,sha in manifest['source_sha256'].items())
              and successor['scientific_constraints']==binding.CONSTRAINTS and successor['archive_stage_budget']==c.INVENTORY,
              'Original 88 sources retained with exact 98-source closure and original envelopes')
    previous=read(PREVIOUS_RETIREMENT_MANIFEST);previous_accepted=read(PREVIOUS_RETIREMENT_ACCEPTANCE)
    c.require(len(previous['source_sha256'])==93 and all(successor['source_sha256'].get(path)==sha for path,sha in previous['source_sha256'].items())
              and same_ref(successor['previous_manifest'],PREVIOUS_RETIREMENT_MANIFEST)
              and previous_accepted['reviewer']=='/root' and previous_accepted['passed'] is True and previous_accepted['blockers']==[]
              and same_ref(previous_accepted['manifest'],PREVIOUS_RETIREMENT_MANIFEST), 'Complete 93 previously tested sources and their exact root acceptance retained')
    catalog,scope=read(catalog_ref),read(scope_ref)
    c.require(scope['schema']=='lfm3_full_cpu_successor_scope_v1', 'Exact complete r2 supplement')
    rebuilt=history.build(read,scope['actual_exit'],scope['full_audit'],catalog['current_handoff'])
    c.require(scope==rebuilt['r2_scope'], 'Exact recomputed successful full876-owner r2 scope')
    c.require(catalog['schema']==rebuilt['catalog']['schema'] and catalog['reviewer']=='/root' and catalog['passed'] is True and catalog['blockers']==[] and catalog['all_later_serial_runs_included'] is True, 'Root current complete catalog')
    c.require(len(catalog['runs'])>=122 and catalog['runs'][:122]==rebuilt['catalog']['runs'] and catalog['completed_sdk']==rebuilt['catalog']['completed_sdk'], 'Original121 rows and actual r2 row preserved in order')
    c.require(len({r['terminal']['path'] for r in catalog['runs'][121:]})==len(catalog['runs'][121:]), 'No repeated appended actual run')
    last=catalog['runs'][-1];predecessor=read(predecessor_ref)
    c.require(same_ref(last['terminal'],predecessor_ref) and last['tool']==predecessor['actual_tool'] and type(last['tool']['exit_code']) is int, 'Latest exact actual predecessor tool receipt')
    for row in catalog['runs'][121:]:
        ended=read(row['terminal']);c.require(ended['actual_tool']==row['tool'], 'Every later catalog tool matches its actual terminal record')
    if mode=='download':
        c.require(replay_from is None and producer_exit_ref is None and producer_recorder_exit_ref is None, 'Download has no invented producer/replay source')
    else:
        for ref in (replay_from,producer_exit_ref,producer_recorder_exit_ref):reference(ref)
        producer=read(producer_exit_ref);recorded=read(producer_recorder_exit_ref);saved=read(replay_from)
        c.require(producer['schema']=='minicpm5_sdk310_deb_stage_actual_exit_v1' and producer['stage_success_observed'] is True and producer['mode']=='download' and producer['actual_tool']['exit_code']==0 and producer['owners_gone'] is True and producer['child_group_empty'] is True and same_ref(producer['result'],replay_from), 'Actual successful drained Mini download source')
        c.require(recorded['actual_tool']['exit_code']==0 and recorded['owner_gone'] is True and same_ref(recorded['result'],producer_exit_ref), 'Actual external recorder termination separately observed')
        c.require(any(same_ref(row['terminal'],producer_exit_ref) for row in catalog['runs']) and any(same_ref(row['terminal'],producer_recorder_exit_ref) for row in catalog['runs']), 'Producer and recorder both in complete current catalog')
        binding.validate_result(saved,read(binding.NEXT),'download',None)
    retired_binding=retirement.bind(read,scope_ref)
    c.require(successor_accepted['retired_lfm_r2']==retired_binding==successor['actual_retirement_binding'],
              'Root source review includes this exact genuine retired r2 evidence binding')
    extra_retirements=retirement.bind_extra(read)
    c.require(extra_retirements==successor['retired_mini_history']==successor_accepted['retired_mini_history'], 'Both exact historical Mini retirement proofs independently reviewed')
    owners,services=complete_history(base,catalog,scope)
    value=dict(schema='dory_serial_stage_plan_v1',stage='inventory',runtime_bound=True,
        automatic_successor_launch=False,output_root=str(root),source_sha256=copy.deepcopy(successor['source_sha256']),
        budget=copy.deepcopy(c.INVENTORY),lock_policy=copy.deepcopy(base['lock_policy']),
        predecessor_exit=copy.deepcopy(predecessor_ref),predecessor_expected_exit_code=last['tool']['exit_code'],
        predecessor_owners=owners,predecessor_services=services,external_authorities=None,
        work_kind=binding.WORK_KIND,archive_inputs=copy.deepcopy(binding.TRUST),archive_mode=mode,replay_from=copy.deepcopy(replay_from),
        mini_scientific_constraints=copy.deepcopy(binding.CONSTRAINTS),complete_current_serial_catalog=copy.deepcopy(catalog_ref),
        physical_history_source=copy.deepcopy(base['physical_history_source']),root_handoff=copy.deepcopy(catalog['current_handoff']),
        r2_predecessor_scope=copy.deepcopy(scope_ref),r2_actual_exit=copy.deepcopy(scope['actual_exit']),r2_full_audit=copy.deepcopy(scope['full_audit']),
        producer_actual_exit=copy.deepcopy(producer_exit_ref),producer_recorder_actual_exit=copy.deepcopy(producer_recorder_exit_ref),
        source_only_acceptance=copy.deepcopy(retirement_acceptance_ref),source_manifest=copy.deepcopy(RETIREMENT_MANIFEST),
        prior_source_only_acceptance=copy.deepcopy(ACCEPTANCE),prior_source_manifest=copy.deepcopy(MANIFEST),retired_lfm_r2=retired_binding,
        retired_mini_history=extra_retirements,previous_retirement_source_manifest=copy.deepcopy(PREVIOUS_RETIREMENT_MANIFEST),
        previous_retirement_source_acceptance=copy.deepcopy(PREVIOUS_RETIREMENT_ACCEPTANCE),
        preparation_history_source=copy.deepcopy(HISTORY_SOURCE),required_lfm_full_auditor_source=copy.deepcopy(AUDITOR_SOURCE),
        fresh_physical_checks_required=True,root_plan_review_required=True,runtime_admitted=False)
    retirement.validate(value,read=read)
    return value


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','scope','predecessor-exit','retirement-source-review'):
        cli.add_argument('--'+name,required=True);cli.add_argument('--'+name+'-sha256',required=True)
    for name in ('replay-from','producer-exit','producer-recorder-exit'):
        cli.add_argument('--'+name);cli.add_argument('--'+name+'-sha256')
    cli.add_argument('--mode',required=True,choices=('download','replay'))
    cli.add_argument('--output-root',required=True);cli.add_argument('--destination',required=True)
    a=cli.parse_args();c.require(sys.dont_write_bytecode and __debug__, 'Source preparation uses -B with assertions')
    def argref(name):
        path=getattr(a,name);sha=getattr(a,name+'_sha256')
        c.require((path is None)==(sha is None), 'Reference path/hash supplied together')
        return dict(path=path,sha256=sha) if path else None
    for pin in (HISTORY_SOURCE,AUDITOR_SOURCE):
        actual,_=c.reference(pin['path'],2*c.MIB);c.require(actual['sha256']==pin['sha256'], 'Frozen source-only history helper and full auditor')
    value=build(c.load,argref('catalog'),argref('scope'),argref('predecessor_exit'),a.mode,a.output_root,
                argref('replay_from'),argref('producer_exit'),argref('producer_recorder_exit'),argref('retirement_source_review'))
    c.verify_sources(value['source_sha256'])
    destination=Path(a.destination);c.require(destination.parent==Q and destination.suffix=='.json' and not destination.name.startswith('continuation-handoff-'), 'New plan JSON only; never create a handoff')
    value['preparation_builder']=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(c.save(destination,value),sort_keys=True))


if __name__=='__main__':main()
