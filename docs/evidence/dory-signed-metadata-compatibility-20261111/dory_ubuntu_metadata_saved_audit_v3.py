"""Pure saved-metadata/event checks; no file/process/package operations."""
import stat
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_ubuntu_metadata_input_binding_v3 as binding
import dory_ubuntu_package_sources_v2 as discovery
import dory_ubuntu_metadata_adapter_v4 as adapter
import datetime as dt
from pathlib import Path


def terminal_bindings(refs,loaded):
    plan=loaded['plan'];terminal=loaded['terminal'];admitted=loaded['admitted']
    binding.validate(plan,loaded['review'],refs['plan'])
    c.require(terminal['schema']=='dory_ubuntu_metadata_root_tool_terminal_v1'
        and terminal['plan']==refs['plan'] and terminal['review']==refs['review']
        and terminal['output_root']==plan['output_root'] and terminal['live_owner_observed_by_root'] is False,
        'Exact post-terminal observation; no fabricated live observation')
    tool=terminal['actual_tool'];c.require(tool['kind']=='tool_session' and type(tool['session_id'])is int
        and tool['session_id']>0 and type(tool['exit_code'])is int and tool['exit_code'] in (0,1)
        and type(tool['terminal_chunk'])is str and tool['terminal_chunk'],'Real supplied external terminal')
    c.require(admitted['schema']=='dory_ubuntu_metadata_root_admission_observation_v1'
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
    c.require(parent['schema']=='dory_ubuntu_metadata_parent_result_v1' and parent['complete'] is True
        and parent['work_kind']==binding.WORK_KIND and parent['metadata_input_binding']==authenticated
        and parent['plan']==producer_refs['plan'] and parent['owner']==v['owner.json']['owner']
        and parent['lock']==runtime.ORIGINAL_LOCK and parent['retained_fd']==fd
        and parent['lock_policy']==runtime.POLICY and parent['first_lock_proof']==refs['initial-lock-proof.json']
        and parent['worker_result']==refs['worker-result.json'] and parent['child_actual_exit_code']==v['child-exit.json']['returncode']==0
        and parent['child_started'] is True and parent['child_group_empty'] is True,'Successful metadata parent schema and exact receipts')
    c.require(worker['schema']=='dory_ubuntu_metadata_worker_result_v1' and worker['complete'] is True
        and worker['stage']=='inventory' and worker['work_kind']==binding.WORK_KIND and worker['input_binding']==authenticated
        and worker['lock_policy']==runtime.POLICY and worker['retained_fd']==fd and worker['metadata_result']==refs['metadata-result.json'],
        'Successful metadata worker, not inventory model result')
    c.require(receipt['schema']=='dory_ubuntu_signed_metadata_receipt_v1' and receipt['work_kind']==binding.WORK_KIND
        and receipt['metadata_input_binding']==authenticated and receipt['plan']==producer_refs['plan']
        and receipt['admission']['path']==refs['admission.json']['path'] and receipt['admission']['sha256']==refs['admission.json']['sha256'],
        'Full metadata receipt input/plan/admission binding')
    metadata=v['signed-metadata/sources-result.json']
    c.require(receipt['sources_result']==refs['signed-metadata/sources-result.json']
        and metadata['schema']=='dory_ubuntu_exact_package_sources_v1'
        and metadata['trust_anchor']==binding.TRUST
        and receipt['status']==metadata['status'] and receipt['packages']==len(metadata['packages'])
        and metadata['status'] in ('exact_addresses_ready','exact_versions_missing'), 'Exact saved signed metadata result')
    for value in (parent,worker):
        for k in ('native_ABI_qualified','full_model_qualified','new_speedup_claim','actual_process_exit_claimed'):
            c.require(value[k] is False,'No native/model/exit qualification')
    for value in (receipt,metadata):
        for k in ('native_ABI_qualified','new_speedup_claim','package_content_authenticated'):
            c.require(value[k] is False,'No package/native/model qualification')
        c.require(value['unresolved_closed']==0,'No unresolved row closed by metadata')
    c.require(metadata['actual_archive_download'] is False and receipt['loader_selection_established'] is False,
        'No archive download or loader selection')
    for role,value in (('parent',parent),('worker',worker)):
        g=value['descriptor_guard'];c.require(g['failed'] is False and g['pending_duplicate'] is False,'Final guard succeeded without pending evidence')
        ref=g['first_duplicate_reference']
        if ref is not None:
            name='initial-lock-proof.json' if g['first_duplicate_was_acquire'] else 'first-descriptor-duplicate-'+role+'.json'
            c.require(ref==refs[name] and (role=='parent' or not g['first_duplicate_was_acquire']),'Exact first duplicate reference')
        c.require(g['duplicate_observations']==0 or ref is not None,'Every accepted duplicate has complete saved evidence')
    return True


NESTED = {'signed-metadata/ubuntu-archive-keyring.gpg': c.MIB,
          'signed-metadata/key-authority.json': c.MIB,
          'signed-metadata/sources-result.json': 8*c.MIB}
for _url, _name in adapter.URL_FILES.items():
    NESTED['signed-metadata/'+_name] = c.MIB if _url.endswith('.gpg') else 4*c.MIB if _url.endswith('/Release') else 32*c.MIB
for _n in range(1,4):
    for _suffix, _cap in (('.Release',4*c.MIB),('.gpg',c.MIB),('-status.txt',c.MIB),('-stderr.txt',c.MIB)):
        NESTED['signed-metadata/verify-'+str(_n)+_suffix] = _cap


def schedule():
    rows = []
    for n, (base, suite) in enumerate(discovery.SOURCES,1):
        release = base+'dists/'+suite+'/Release'
        rows.extend([('download',release),('download',release+'.gpg'),('start',n),('exit',n)])
        rows.extend(('download',base+'dists/'+suite+'/'+component+'/binary-amd64/Packages.xz') for component in ('main','universe'))
    return rows+[('complete',None)]


class Events:
    """Independently validate ordered actual download and verifier events."""
    def __init__(self,out,refs):
        self.out,self.refs=Path(out),refs
        self.index,self.transferred,self.failed_verifier=0,0,False
        self.downloads,self.exits={},{}
        self.last=None

    def ref(self,name):
        c.require(name in self.refs and self.refs[name]['path']==str(self.out/name),'Actual saved file reference')
        return self.refs[name]

    def argv(self,n):
        out=self.out/'signed-metadata'
        return [adapter.GPGV,'--homedir',str(out),'--keyring',str(out/'ubuntu-archive-keyring.gpg'),
                '--status-fd','1','--',str(out/('verify-'+str(n)+'.gpg')),str(out/('verify-'+str(n)+'.Release'))]

    def accept(self,row):
        c.require(not self.failed_verifier and self.index<19,'No later event after failed verifier or completion')
        kind,value=schedule()[self.index]
        if kind=='download':
            c.require(set(row)=={'event','url','saved','total_transfer_bytes'}
                and row['event']=='ubuntu_metadata_download' and row['url']==value,'Exact fixed download order')
            name='signed-metadata/'+adapter.URL_FILES[value]
            c.require(row['saved']==self.ref(name) and 0<row['saved']['bytes']<=NESTED[name],'Saved download bytes/reference')
            self.transferred+=row['saved']['bytes']
            c.require(self.transferred<=128*c.MIB and row['total_transfer_bytes']==self.transferred,'Cumulative complete transfer count')
            self.downloads[value]=row['saved']
        elif kind=='start':
            c.require(set(row)=={'event','invocation','argv'} and row['event']=='ubuntu_gpgv_start'
                and row['invocation']==value and row['argv']==self.argv(value),'Exact actual verifier command')
            base,suite=discovery.SOURCES[value-1];release=base+'dists/'+suite+'/Release'
            for suffix,url in (('.Release',release),('.gpg',release+'.gpg')):
                ref=self.ref('signed-metadata/verify-'+str(value)+suffix)
                c.require(all(ref[k]==self.downloads[url][k] for k in ('bytes','sha256')),'Verifier inputs equal downloaded full bytes')
        elif kind=='exit':
            c.require(set(row)=={'event','invocation','returncode','status','stderr'} and row['event']=='ubuntu_gpgv_exit'
                and row['invocation']==value and type(row['returncode'])is int,'Exact verifier exit event')
            for key,suffix in (('status','-status.txt'),('stderr','-stderr.txt')):
                c.require(row[key]==self.ref('signed-metadata/verify-'+str(value)+suffix),'Actual full verifier stream reference')
            self.exits[value]=row
            self.failed_verifier=row['returncode']!=0
        else:
            c.require(set(row)=={'event','result','status','packages','missing','package_content_authenticated'}
                and row['event']=='ubuntu_metadata_discovery_complete'
                and row['result']==self.ref('signed-metadata/sources-result.json')
                and row['status'] in ('exact_addresses_ready','exact_versions_missing')
                and type(row['packages'])is int and 0<=row['packages']<=6
                and row['package_content_authenticated']is False,'Final metadata completion only')
            self.last=row
        self.index+=1


def replay_saved(events,values,refs,raw_inputs,read_saved,guard):
    """Pure chain replay from fully rehashed saved data, no cryptographic process."""
    c.require(events.index==19 and not events.failed_verifier and events.last is not None,'Complete19event workflow')
    c.require(set(n for n in refs if n.startswith('signed-metadata/'))==set(NESTED),'Exact complete nested output set')
    authority=values['signed-metadata/key-authority.json']
    c.require(refs['signed-metadata/key-authority.json']['sha256']==binding.AUTHORITY['sha256']
        and authority['keyring']['sha256']==binding.TRUST['keyring_sha256']
        and authority['primary_fingerprints']==binding.TRUST['primary_fingerprints'],'Same actual authority and fixed Ubuntu keys')
    key=read_saved('signed-metadata/ubuntu-archive-keyring.gpg',c.MIB)
    c.require(discovery.sha(key)==authority['keyring']['sha256'] and len(key)==authority['keyring']['bytes'],'Full copied keyring bound to authority')
    result=values['signed-metadata/sources-result.json'];saved_adapter=result['adapter']
    expected_files={r['path']:r for n,r in refs.items() if n.startswith('signed-metadata/') and n!='signed-metadata/sources-result.json'}
    actual_files={r['path']:r for r in saved_adapter['metadata_files']}
    c.require(len(actual_files)==len(saved_adapter['metadata_files']) and actual_files==expected_files,'Every saved metadata file in adapter receipt')
    calculated=4096+sum(((ref['bytes']+4095)//4096)*4096 for ref in actual_files.values())
    c.require(saved_adapter['adapter_output_allocated_bound']==calculated
        and saved_adapter['gpgv_reference']==authority['gpgv'] and saved_adapter['requests']==12
        and saved_adapter['signature_invocations']==3 and saved_adapter['actual_cryptographic_verifier']=='gpgv'
        and saved_adapter['runtime_admission_checked_here']is False and saved_adapter['caller_preconditions_required']is True
        and saved_adapter['process_exit_claimed']is False,'Complete adapter bookkeeping and verifier identity')
    calls=[0]
    now=dt.datetime.now(dt.timezone.utc)
    def fetch(url,cap):
        c.require(url in events.downloads,'Saved exact download event required')
        name='signed-metadata/'+adapter.URL_FILES[url]
        raw=read_saved(name,min(cap,NESTED[name]))
        c.require(discovery.sha(raw)==events.downloads[url]['sha256'],'Replay download hash')
        return dict(body=raw,final_url=url)
    def verified_record(release,signature,trust):
        calls[0]+=1;n=calls[0]
        c.require(n in events.exits and events.exits[n]['returncode']==0 and trust==binding.TRUST,'Actual successful verifier event authority')
        for suffix,raw in (('.Release',release),('.gpg',signature)):
            name='signed-metadata/verify-'+str(n)+suffix
            c.require(read_saved(name,NESTED[name])==raw,'Verifier full input bytes identical')
        status=read_saved('signed-metadata/verify-'+str(n)+'-status.txt',c.MIB)
        signatures=adapter.parse_status(status,0,trust,now)
        return dict(verified=True,**signatures[0],verified_signatures=signatures,
                    release_sha256=discovery.sha(release),signature_sha256=discovery.sha(signature),
                    keyring_sha256=trust['keyring_sha256'],process_returncode=0,
                    status_reference=events.exits[n]['status'],stderr_reference=events.exits[n]['stderr'])
    replay=discovery.discover(raw_inputs,fetch,verified_record,binding.TRUST,now,guard)
    c.require(replay=={k:v for k,v in result.items() if k!='adapter'},'Entire signed metadata result independently reproduced')
    c.require(events.last['status']==replay['status'] and events.last['packages']==len(replay['packages'])
        and events.last['missing']==replay['missing'],'Final event equals independently replayed result')
    return dict(download_events=12,verifier_start_events=3,verifier_exit_events=3,completion_events=1,
        all_saved_metadata_rehashed=True,signature_status_reparsed=True,actual_signature_invocations_bound=True,
        signature_verification_rerun=False,entire_discovery_reproduced=True,status=replay['status'],
        packages=len(replay['packages']),missing=replay['missing'],package_content_authenticated=False,
        loader_selection_established=False,unresolved_closed=0)
