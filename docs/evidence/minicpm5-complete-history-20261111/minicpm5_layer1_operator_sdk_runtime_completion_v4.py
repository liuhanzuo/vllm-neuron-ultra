"""Complete old lineage + actual r2 + bounded root-observed later serial work.

Only injected readers/guards perform IO. No catalog is created here, and no
current physical proof is inferred from the completed historical audit pair.
"""
import copy
import hashlib
from pathlib import Path
from minicpm5_layer0_runtime_auth_v1 import identity
import re
from research import minicpm5_layer1_sdk_batch_layout_completion_v3 as prior
from research import minicpm5_layer1_operator_sdk_completion_v2 as sdk
from research import minicpm5_layer1_operator_sdk_output_catalog_v2 as outputs
from research import minicpm5_sdk_physical_evidence_v3 as physical_evidence
from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
from minicpm5_original88_decode_mask_analysis_v1 import p
require=sdk.require
require_complete_binding=prior.require_complete_binding
MAX_SERIAL_RUNS=256


def serial_catalog(plan,read,guard):
    ref=plan['serial_predecessors'];sources.reference(ref);guard();value=read(ref)
    require(type(value)is dict and set(value)=={'schema','reviewer','passed','blockers','completed_sdk',
        'current_handoff','all_later_serial_runs_included','runs'},'Exact bounded current serial catalog')
    require(value['schema']=='minicpm5_operator_later_serial_metadata_catalog_v1' and value['reviewer']=='/root'
        and value['passed']is True and value['blockers']==[] and value['completed_sdk']==sdk.BINDING
        and value['all_later_serial_runs_included']is True and type(value['runs'])is list and len(value['runs'])<=MAX_SERIAL_RUNS,
        'Root-observed complete current post-r2 serial scope')
    sources.reference(value['current_handoff']);handoff=read(value['current_handoff'])
    require(handoff['live_tool_sessions']==[] and handoff['active_heavy_units']==[],
        'Actual bounded current catalog follows root-observed idle handoff')
    owners=[];units=set();sources_seen={};terminals=[]
    for row in value['runs']:
        guard()
        require(type(row)is dict and set(row)=={'terminal','tool','owners','units','source_sha256'},
                'Exact retained serial run terminal/owners/units/source record')
        sources.reference(row['terminal']);terminal=read(row['terminal'])
        require(type(terminal)is dict and terminal,'Retain full actual producer terminal metadata')
        tool=row['tool']
        require(type(tool)is dict and (
                tool.get('kind')=='tool_session' and set(tool)=={'kind','session_id','exit_code','terminal_chunk'}
                and type(tool['session_id'])is int and tool['session_id']>0
                or tool.get('kind')=='tool_call' and set(tool)=={'kind','exit_code','terminal_chunk'})
            and type(tool['exit_code'])is int and 0<=tool['exit_code']<=255
            and type(tool['terminal_chunk'])is str and re.fullmatch(r'[a-f0-9]{3,64}',tool['terminal_chunk']),
            'Root catalog retains actual success OR failure terminal; never promotes failed Dory')
        require(type(row['owners'])is list and len(row['owners'])<=256
            and type(row['units'])is list and len(row['units'])<=32,'Bounded complete recorded serial identities')
        for owner in row['owners']:
            require(type(owner)is dict and type(owner.get('pid'))is int and owner['pid']>0
                and str(owner.get('start_ticks','')).isdigit(),'Actual recorded PID/start identity retained')
            owners.append(copy.deepcopy(owner))
        for unit in row['units']:units.add(physical_evidence.unit_name(unit))
        sources.metadata.pin_shape(row['source_sha256'])
        for path,sha in row['source_sha256'].items():
            require(plan['source_sha256'].get(path)==sha and (path not in sources_seen or sources_seen[path]==sha),
                    'All later executed serial sources retained without conflict')
            sources_seen[path]=sha
        terminals.append(copy.deepcopy(row))
    require(len({x['terminal']['path']for x in terminals})==len(terminals),'No repeated terminal substitutes for another run')
    return dict(reference=ref,current_handoff=value['current_handoff'],owners=owners,units=sorted(units),
        runs=terminals,source_sha256=sources_seen)


def handoff_candidates():
    """Metadata/stat only; preserve original latest-by-mtime discovery."""
    return tuple(prior.Q.glob('continuation-handoff-*.json'))


def latest_handoff(paths=handoff_candidates,stat=identity):
    rows=[]
    for path in paths():
        path=Path(path)
        require(path.parent==prior.Q and re.fullmatch(r'continuation-handoff-[a-z0-9-]+\.json',path.name),
            'Exact original bounded handoff namespace')
        row=stat(str(path));rows.append((row['mtime_ns'],str(path),row))
    require(0<len(rows)<=4096,'Finite nonempty handoff catalog')
    rows.sort(key=lambda row:(row[0],row[1]));latest=rows[-1]
    require(len(rows)==1 or rows[-2][0]!=latest[0],'Ambiguous latest handoff timestamp rejected')
    return dict(path=latest[1],identity=latest[2])


def publication_view(value,catalog_ref,bound,base):
    """Only presentation fields and exact redundant catalog references may vary."""
    result=copy.deepcopy(value)
    for key in ('schema','updated_at','previous','publication','publication_current_status_stale',
                'disk_available_bytes','publication_only_since_bound_terminal'):
        result.pop(key,None)
    operator=result.get('mini_operator_current')
    require(type(operator)is dict,'Complete original operator serial status retained')
    original=base['mini_operator_current']
    if 'serial_catalog'in operator:
        require(operator.pop('serial_catalog')in (catalog_ref,original.get('serial_catalog')),
            'Publication retains original historical catalog note or exact current serial catalog')
    if 'catalog_bound_terminal_handoff'in operator:
        require(operator.pop('catalog_bound_terminal_handoff')in (bound['path'],original.get('catalog_bound_terminal_handoff')),
            'Publication retains original historical terminal note or exact current serial terminal')
    return result


def require_latest_terminal(plan,read,guard,*,paths=handoff_candidates,stat=identity,read_text=sources.text):
    """Fresh preparation/outer only; completed() remains historical and pure.

    Authenticate the current later-serial catalog, then require latest handoff
    either IS its terminal or has a lossless publication-only previous chain.
    Unknown/new serial activity fails before actual preparation or launch.
    """
    guard();require_complete_binding(plan['latest_completion'])
    serial=serial_catalog(plan,read,guard);bound=serial['current_handoff'];base=read(bound)
    require(base['serial_execution']is True,'Original serial execution retained')
    first=latest_handoff(paths,stat);path=first['path'];before=first['identity']
    # Discover exactly one bounded root-owned metadata object, then authenticate
    # it through the same Reader so preparation pins its actual file identity.
    raw=read_text(path);require(type(raw)is bytes and len(raw)<=2*2**20,'Bounded current handoff JSON')
    tip=dict(path=path,sha256=hashlib.sha256(raw).hexdigest());value=read(tip)
    require(stat(path)==before,'Discovered handoff stable across authenticated read')
    ref=tip;visited=set();publications=[];baseline=publication_view(base,serial['reference'],bound,base)
    while ref!=bound:
        guard();require(ref['path']not in visited and len(visited)<16,'Bounded acyclic publication-only handoff chain')
        visited.add(ref['path'])
        require(value.get('publication_only_since_bound_terminal')is True
            and value.get('live_tool_sessions')==[] and value.get('active_heavy_units')==[]
            and value.get('serial_execution')is True
            and publication_view(value,serial['reference'],bound,base)==baseline,
            'Every later handoff is explicitly publication-only with unchanged complete serial state')
        publications.append(copy.deepcopy(ref));parent=value['previous'];sources.reference(parent)
        require(Path(parent['path']).parent==prior.Q and re.fullmatch(r'continuation-handoff-[a-z0-9-]+\.json',Path(parent['path']).name),
            'Publication previous stays inside original handoff namespace')
        ref=parent;value=read(ref)
    require(value==base,'Complete exact authenticated catalog terminal reached')
    require(latest_handoff(paths,stat)==first,'No newer/replaced handoff appeared during freshness check')
    guard();return dict(complete=True,serial_catalog=serial['reference'],bound_terminal=bound,
        latest_handoff=tip,publication_only_successors=publications,historical_binding_unchanged=True)


def completed(plan,read,guard):
    guard();require_complete_binding(plan['latest_completion'])
    require(plan['runtime_output_manifest']==outputs.catalog(),'Exact bounded current runtime file catalog')
    original=prior.completed(plan['latest_completion'],read,guard)
    current=sdk.completed(plan,read,guard);later=serial_catalog(plan,read,guard)
    require(current['lock_identity']==original['lock_identity'],'Same original serial lock in completed lineage')
    require(current['audit']['actual_predecessor_pair_schema']==physical_evidence.PAIR_SCHEMA,
            'Actual completed r2 used complete physical v3')
    # sdk.completed already proves this exact preserved original owner tree.
    expected=current['physical_pair']['first']['tree']['all_additional_owners_checked']
    require(expected==original['owners'],'All original1642 owners match independent r2 observation')
    owners=copy.deepcopy(current['owners'])+copy.deepcopy(later['owners'])
    units=set(current['complete_known_units'])|set(later['units'])
    units.add(physical_evidence.unit_name(current['model_exit']['coordinator']['unit']))
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+physical_evidence.GROUP_PREFIX):
            units.add(physical_evidence.group_unit(group[3:]))
    done=dict(original,owners=owners,source_sha256=dict(original['source_sha256'],**later['source_sha256']),
        operator_sdk=current,later_serial=later,complete_known_units=sorted(units),_original_done=original)
    guard();return done


def physical(done,guard):
    """One real full-tree observation; collect_pair below performs two passes."""
    guard();old=prior.physical(done['_original_done'],guard)
    for owner in done['owners']:guard();p.gone(owner)
    old_units=physical_evidence.inspect_tree(old)['units']
    added=[]
    for name in sorted(set(done['complete_known_units'])-old_units):
        guard();props=p.properties(name);group=p.empty_group(physical_evidence.GROUP_PREFIX+name+'.service')
        sample=dict(unit=name,properties=props,group=group)
        physical_evidence.observation(sample,name);added.append(sample)
    tree=dict(complete=True,physical_checks_performed=True,all_current_and_prior_services_preserved=True,
        all_additional_owners_checked=copy.deepcopy(done['owners']),lock_identity=copy.deepcopy(done['lock_identity']),
        nemotron=dict(complete=True,fresh_physical_drain_verified=True,services=1308),
        entire_original_physical_tree=old,later_services=added,
        complete_sdk_binding=copy.deepcopy(sdk.BINDING),later_serial_catalog=done['later_serial']['reference'])
    physical_evidence.inspect_tree(tree,done['complete_known_units']);guard();return tree


def collect_pair(done,guard):
    return physical_evidence.collect_pair(done,guard,physical,p.properties,p.empty_group,
        known_units=done['complete_known_units'])
