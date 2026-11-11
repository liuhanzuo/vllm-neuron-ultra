"""Pure saved-metadata/event checks; no file/process/package operations."""
import stat
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_system_package_input_binding_v1 as binding
import dory_system_package_metadata_v1 as collector
STAT_FIELDS={'st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns'}


def terminal_bindings(refs,loaded):
    plan=loaded['plan'];terminal=loaded['terminal'];admitted=loaded['admitted']
    binding.validate(plan,loaded['review'],refs['plan'])
    c.require(terminal['schema']=='dory_system_package_metadata_root_tool_terminal_v1'
        and terminal['plan']==refs['plan'] and terminal['review']==refs['review']
        and terminal['output_root']==plan['output_root'] and terminal['live_owner_observed_by_root'] is False,
        'Exact post-terminal observation; no fabricated live observation')
    tool=terminal['actual_tool'];c.require(tool['kind']=='tool_session' and type(tool['session_id'])is int
        and tool['session_id']>0 and type(tool['exit_code'])is int and tool['exit_code'] in (0,1)
        and type(tool['terminal_chunk'])is str and tool['terminal_chunk'],'Real supplied external terminal')
    c.require(admitted['schema']=='dory_system_package_metadata_root_admission_observation_v1'
        and admitted['observed_after_tool_terminal'] is True and admitted['live_owner_observed_by_root'] is False
        and admitted['terminal']==refs['terminal'] and admitted['plan']==refs['plan'] and admitted['review']==refs['review']
        and admitted['parent_owner']==terminal['owner'] and admitted['child_owner']==terminal['child_owner'],
        'Authenticated saved-owner observation after exact terminal')
    parent=terminal['owner'];child=terminal['child_owner']
    c.require(type(parent['pid'])is int and type(child['pid'])is int and parent['pid']!=child['pid']
        and child['ppid']==parent['pid'] and int(child['start_ticks'])>=int(parent['start_ticks']), 'Original parent/child identities')
    return tool,parent,child


def successful_outputs(refs,v,code,producer_refs,authenticated):
    c.require(type(code)is int and code in (0,1),'Only actual supplied zero/one producer exit')
    if code==1:
        c.require('failure.json' in refs and v['failure.json']['owner']==v['owner.json']['owner']
            and v['failure.json']['actual_process_exit_claimed'] is False,'Preserve actual original failed producer')
        return False
    c.require(not any(n in v for n in ('failure.json','worker-failure.json','cleanup-pending.json','descriptor-lock-failure-parent.json','descriptor-lock-failure-worker.json')),'No failure evidence on success')
    parent=v['result.json'];worker=v['worker-result.json'];receipt=v['metadata-result.json'];admission=v['admission.json'];fd=admission['retained_fd']
    c.require(parent['schema']=='dory_system_package_metadata_parent_result_v1' and parent['complete'] is True
        and parent['work_kind']==binding.WORK_KIND and parent['metadata_input_binding']==authenticated
        and parent['plan']==producer_refs['plan'] and parent['owner']==v['owner.json']['owner']
        and parent['lock']==runtime.ORIGINAL_LOCK and parent['retained_fd']==fd
        and parent['lock_policy']==runtime.POLICY and parent['first_lock_proof']==refs['initial-lock-proof.json']
        and parent['worker_result']==refs['worker-result.json'] and parent['child_actual_exit_code']==v['child-exit.json']['returncode']==0
        and parent['child_started'] is True and parent['child_group_empty'] is True,'Successful metadata parent schema and exact receipts')
    c.require(worker['schema']=='dory_system_package_metadata_worker_result_v1' and worker['complete'] is True
        and worker['stage']=='inventory' and worker['work_kind']==binding.WORK_KIND and worker['input_binding']==authenticated
        and worker['lock_policy']==runtime.POLICY and worker['retained_fd']==fd and worker['metadata_result']==refs['metadata-result.json'],
        'Successful metadata worker, not inventory model result')
    c.require(receipt['schema']=='dory_system_package_metadata_receipt_v1' and receipt['work_kind']==binding.WORK_KIND
        and receipt['input_binding']==authenticated and receipt['plan']==producer_refs['plan']
        and receipt['admission']['path']==refs['admission.json']['path'] and receipt['admission']['sha256']==refs['admission.json']['sha256'],
        'Full metadata receipt input/plan/admission binding')
    metadata=receipt['metadata'];c.require(metadata['schema']=='dory_system_package_metadata_v1'
        and metadata['saved_external_evidence']==binding.INPUTS['external'] and metadata['library_payload_bytes_read']==0,
        'Fixed metadata-only collector result')
    for value in (parent,worker,receipt,metadata):
        for k in ('native_ABI_qualified','full_model_qualified','new_speedup_claim','actual_process_exit_claimed'):c.require(value[k] is False,'No content/native/model/exit qualification')
    c.require(metadata['package_archive_authenticated'] is False and metadata['loader_selection_established'] is False,'Package metadata is not independent bytes or selection')
    for role,value in (('parent',parent),('worker',worker)):
        g=value['descriptor_guard'];c.require(g['failed'] is False and g['pending_duplicate'] is False,'Final guard succeeded without pending evidence')
        ref=g['first_duplicate_reference']
        if ref is not None:
            name='initial-lock-proof.json' if g['first_duplicate_was_acquire'] else 'first-descriptor-duplicate-'+role+'.json'
            c.require(ref==refs[name] and (role=='parent' or not g['first_duplicate_was_acquire']),'Exact first duplicate reference')
        c.require(g['duplicate_observations']==0 or ref is not None,'Every accepted duplicate has complete saved evidence')
    return True


class Events:
    """Independent finite journal crosswalk; retains summaries, not stdout copies."""
    def __init__(self):
        self.index=0;self.bytes=0;self.owners={};self.lists={};self.versions=None;self.last=None
    def accept(self,row):
        c.require(self.last is None,'No events after terminal completion')
        if row['event']=='system_package_metadata_complete':
            c.require(self.index==32 and set(row)=={'event','system_targets','fixed_commands','unresolved'},'Final event only after all 32 exact commands')
            self.last=row;return
        c.require(row['event']=='system_package_metadata_command' and set(row)=={'event','index','argv','returncode','stdout'}
            and type(row['index'])is int and row['index']==self.index<32 and row['argv']==list(collector.COMMANDS[self.index])
            and type(row['returncode'])is int and type(row['stdout'])is str,'Exact ordered fixed command event')
        size=len(row['stdout'].encode('utf8','surrogateescape'));self.bytes+=size
        c.require(size<=8*c.MIB and self.bytes<=32*c.MIB,'Original full command byte limits')
        argv=row['argv'];code=row['returncode'];text=row['stdout']
        if argv[1]=='-S':
            c.require(code in (0,1),'Fixed ownership returncode');owners=set();issues=[]
            if code==1:issues=['not_found_by_fixed_name']
            else:
                for line in text.splitlines():
                    head,sep,path=line.rpartition(': ')
                    if not sep or path!=argv[2]:issues.append('unparsed_or_nonexact_owner_row');continue
                    for item in head.split(', '):
                        package,sep,arch=item.partition(':')
                        if package not in collector.PACKAGES or sep and arch!='amd64':issues.append('owner_outside_reviewed_six_packages')
                        else:owners.add(package)
                c.require(len(owners)<=6 and len(issues)<=32,'Bounded ownership output')
            self.owners[argv[2]]=dict(owners=sorted(owners),unresolved=issues)
        elif argv[1]=='-W':
            c.require(code==0,'Successful complete fixed version query');versions={}
            for line in text.splitlines():
                package,version,architecture,status=line.split('\t')
                c.require(package in collector.PACKAGES and package not in versions and architecture=='amd64'
                    and status=='install ok installed' and 0<len(version)<=128,'Exact six package versions')
                versions[package]=dict(version=version,architecture=architecture,status=status)
            c.require(set(versions)==set(collector.PACKAGES),'All six packages returned')
            for package,version in collector.KNOWN_VERSIONS.items():c.require(versions[package]['version']==version,'Previous four package versions unchanged')
            self.versions=versions
        else:
            c.require(code==0,'Complete successful package file list');names=text.splitlines();c.require(len(names)<=200000,'Finite file list records')
            self.lists[argv[2].removesuffix(':amd64')]=dict(entries=len(names),fixed_names=sorted(set(names)&set(collector.PATHS)))
        self.index+=1
    def finish(self,metadata):
        c.require(self.index==32 and self.last is not None and self.versions==metadata['versions']
            and self.lists==metadata['package_file_lists'],'All command-derived versions and full-list summaries equal saved result')
        c.require(metadata['fixed_paths']==25 and metadata['system_targets']==12 and metadata['fixed_commands']==32 and metadata['fixed_packages']==6
            and metadata['budget_used']==dict(commands=32,command_total_bytes=self.bytes),'Exact collector count and byte budget recomputation')
        paths=metadata['paths'];c.require(set(paths)==set(collector.PATHS),'Every fixed path observation present')
        for name,row in paths.items():
            c.require(row['requested']==name and row['status'] in ('observed','missing','unreviewed_link_target')
                and type(row['steps'])is list and len(row['steps'])<=32,'Bounded saved metadata observation')
            for step in row['steps']:
                info=step['identity'];c.require(step['path'] in collector.KNOWN_NODES and set(info)==STAT_FIELDS
                    and all(type(x)is int and x>=0 for x in info.values()),'Complete saved finite-node stat metadata')
                if 'link_target' in step:c.require(stat.S_ISLNK(info['st_mode']) and len(step['link_target'].encode())<=4096,'Saved bounded link observation')
            if row['status']=='observed':c.require(row['steps'] and row['resolved']==row['steps'][-1]['path'] and row['identity']==row['steps'][-1]['identity'] and stat.S_ISREG(row['identity']['st_mode']),'Saved final ordinary metadata identity')
        targets=[];unresolved=[]
        for target,names in collector.TARGETS.items():
            matched=set();issues=[]
            for name in names:
                observation=paths[name]
                if observation['status']!='observed' or observation.get('resolved')!=target:
                    issues.append(dict(path=name,reason='fixed_alias_not_observed_at_expected_target',observation=observation['status']));continue
                issues.extend(dict(path=name,reason=reason) for reason in self.owners[name]['unresolved'])
                for package in self.owners[name]['owners']:
                    if name in self.lists[package]['fixed_names']:matched.add(package)
                    else:issues.append(dict(path=name,reason='owner_name_absent_from_same_package_file_list',package=package))
            if len(matched)!=1:issues.append(dict(path=target,reason='not_one_cross_checked_package',packages=sorted(matched)))
            targets.append(dict(target=target,observed_names=list(names),cross_checked_packages=sorted(matched),issues=issues));unresolved.extend(issues)
        c.require(targets==metadata['targets'] and unresolved==metadata['unresolved'],'All twelve saved ownership crosswalks and unresolved rows recomputed')
        c.require(self.last==dict(event='system_package_metadata_complete',system_targets=12,fixed_commands=32,unresolved=len(unresolved)),'Exact final completion event')
        return dict(command_events=32,completion_events=1,command_output_bytes=self.bytes,package_count=6,
            target_count=12,query_names=25,unresolved=len(unresolved),all_command_summaries_recomputed=True,
            filesystem_metadata_reobserved=False,package_content_authenticated=False,loader_selection_established=False)
