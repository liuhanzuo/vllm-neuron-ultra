"""Guarded metadata census; no native import, payload hashing, lock or writes.

collect(guard=...) returns complete named records plus a summary. The caller
owns serial admission, persistence and real exit observation. Only distribution
metadata and fixed historical JSON/lock/cfg files are opened. RECORD and current
hashes are observations, never independent trust anchors. No scientific gate is
introduced. Git status is deferred because refreshing it can read file bodies.
"""
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
ENV = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
SITE = ENV/'lib/python3.12/site-packages'
OFFICIAL = BASE/'vllm'
META_NAMES = {'METADATA', 'RECORD', 'direct_url.json', 'INSTALLER', 'WHEEL'}
LIMITS = dict(files=100000, inventory_bytes=8*2**30, single_inventory_file=2*2**30,
    directory_entries=200000, directory_name_bytes=32*2**20,
    metadata_file_bytes=2*2**20, metadata_total_bytes=32*2**20,
    command_bytes=8*2**20, command_total_bytes=16*2**20,
    command_seconds=30, wall_seconds=300)
FIELDS = ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
ANCHORS = (ENV/'pyvenv.cfg', BASE/'runtime-requirements.lock', BASE/'test-tools.lock',
    BASE/'install-r1/installation.json', BASE/'setup-tests-r1/wheel-installation.json',
    BASE/'setup-tests-r1/test-tools-installation.json', BASE/'qualify-r2/qualification/imports.json',
    ROOT/'research/dory-official-cpu-build-actual-exit-20261108-v1.json',
    ROOT/'research/dory-official-cpu-maintenance-preparation-20261108-v1.json')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def identity(info):
    return {name:getattr(info,name) for name in FIELDS}


class Budget:
    def __init__(self, guard):
        require(callable(guard), 'Explicit caller serial/resource guard')
        self.guard=guard;self.started=time.monotonic();self.used=collections.Counter()
    def check(self):
        self.guard()
        require(time.monotonic()-self.started <= LIMITS['wall_seconds'], 'Metadata census wall bound')
    def add(self, key, amount):
        self.check();self.used[key]+=amount
        require(self.used[key] <= LIMITS[key], 'Metadata census budget: '+key)


def commands():
    git=['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(OFFICIAL)]
    return [git+['rev-parse','HEAD'],git+['ls-files','--stage','-z'],
        *[['/usr/bin/readlink','-e',str(p)] for p in
          (ENV/'bin/python',ENV/'lib/libiomp5.so',Path('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4'))],
        ['/usr/bin/dpkg-query','-W','-f=${Package}\t${Version}\t${Architecture}\n',
         'libtcmalloc-minimal4','libc6','libstdc++6','libgcc-s1']]


def run_command(argv, budget):
    require(argv in commands(), 'Only fixed read-only metadata commands')
    budget.check();start=time.monotonic();raw=bytearray()
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',HOME='/home/ubuntu',
             GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_OPTIONAL_LOCKS='0')
    process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    try:
        while selector.get_map():
            budget.check();require(time.monotonic()-start <= LIMITS['command_seconds'], 'Metadata command timeout')
            for key,_ in selector.select(.1):
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                budget.add('command_total_bytes',len(block))
                require(len(raw)+len(block)<=LIMITS['command_bytes'], 'Metadata command output cap')
                raw.extend(block)
        code=process.wait(timeout=max(.01,LIMITS['command_seconds']-(time.monotonic()-start)))
        return dict(argv=argv,returncode=code,stdout_stderr=raw.decode('utf8','surrogateescape'),
                    encoding='utf8+surrogateescape',bytes=len(raw))
    finally:
        selector.close()
        if process.poll() is None:process.kill()
        process.wait(timeout=5);process.stdout.close()


def metadata_file(path, budget):
    """Return whole small metadata with stable identity; never follow symlinks."""
    path=Path(path);budget.check()
    allowed=path in ANCHORS or (path.parent.parent==SITE and path.parent.name.endswith('.dist-info')
                              and path.name in META_NAMES)
    require(allowed, 'Exact metadata allowlist, never a binary/model payload')
    try:before=path.lstat()
    except FileNotFoundError:return dict(path=str(path),status='missing')
    if path.resolve()!=path or not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:
        return dict(path=str(path),status='not_canonical_single_link_regular',identity=identity(before))
    require(before.st_size<=LIMITS['metadata_file_bytes'], 'Single metadata file cap')
    budget.add('metadata_total_bytes',before.st_size)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        require(identity(os.fstat(fd))==identity(before), 'Metadata opened identity')
        data=bytearray()
        while len(data)<before.st_size:
            budget.check();block=os.read(fd,min(65536,before.st_size-len(data)))
            require(bool(block), 'Complete metadata read');data.extend(block)
        require(os.read(fd,1)==b'' and identity(os.fstat(fd))==identity(before)
            and identity(path.lstat())==identity(before) and path.resolve()==path, 'Stable complete metadata')
        return dict(path=str(path),status='read',identity=identity(before),bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),text=data.decode('utf8'),
                    independently_authenticated=False)
    finally:os.close(fd)


def scan(root, scope, budget, index=None):
    """Every name retained; .git is an explicit opaque boundary; links not followed."""
    root=Path(root);require(root.is_absolute() and root.resolve()==root and root.is_dir(), 'Canonical census root')
    require(scope in ('site','official'), 'Known census scope')
    if scope=='official':require(isinstance(index,dict), 'Explicit official index, possibly unavailable/empty')
    rows=[];pending=[root]
    while pending:
        parent=pending.pop();budget.check();before=parent.lstat()
        require(stat.S_ISDIR(before.st_mode) and not parent.is_symlink(), 'No followed directory link')
        children=[]
        with os.scandir(parent) as directory:
            for entry in directory:
                budget.add('directory_entries',1)
                budget.add('directory_name_bytes',len(os.fsencode(entry.path)))
                children.append(entry.name)
        for name in sorted(children):
            budget.check();path=parent/name;info=path.lstat()
            kind=('dir' if stat.S_ISDIR(info.st_mode) else 'file' if stat.S_ISREG(info.st_mode)
                  else 'link' if stat.S_ISLNK(info.st_mode) else 'other')
            excluded=('git_administrative_tree' if scope=='official' and path==root/'.git'
                      else 'cached_bytecode' if kind!='dir' and path.suffix in ('.pyc','.pyo') else None)
            tracked=index.get(str(path.relative_to(root))) if scope=='official' else None
            eligible=(kind=='file' and excluded is None and (scope=='site' or
                (tracked is not None and tracked['stage']=='0' and tracked['mode'] in ('100644','100755'))))
            row=dict(path=str(path),scope=scope,kind=kind,identity=identity(info),excluded=excluded,
                     inventory_eligible=eligible)
            if kind=='link':row['link_target']=os.readlink(path)
            rows.append(row)
            if kind=='dir' and excluded is None:pending.append(path)
            elif eligible:
                budget.add('files',1);budget.add('inventory_bytes',info.st_size)
                require(info.st_size<=LIMITS['single_inventory_file'], 'Existing single inventory file bound')
        require(identity(parent.lstat())==identity(before), 'Directory changed during named census')
    return sorted(rows,key=lambda row:row['path'])


def git_index(command):
    if command['returncode']!=0:return {},['git index unavailable']
    result={};issues=[]
    for value in command['stdout_stderr'].split('\0'):
        if not value:continue
        head,path=value.split('\t',1);mode,blob,stage=head.split()
        require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'Git path stays in official root')
        require(path not in result, 'No duplicate/conflicted index name')
        result[path]=dict(mode=mode,blob=blob,stage=stage)
        if stage!='0' or mode=='160000':issues.append('unmerged_or_submodule:'+path)
    return result,issues


def summarize(entries, metadata, results):
    index,issues=git_index(results[1]);owners=collections.defaultdict(list);packages=[]
    issues.extend('metadata_command_failed:'+str(i) for i,row in enumerate(results) if row['returncode']!=0)
    external_record_claims=0
    for row in metadata:
        path=Path(row['path'])
        if row['status']!='read':issues.append(row['status']+':'+str(path));continue
        if path.name=='METADATA':
            fields={}
            for line in row['text'].splitlines():
                if not line:break
                if line.startswith(('Name: ','Version: ')):
                    key,value=line.split(': ',1);fields[key.lower()]=value
            packages.append(dict(directory=str(path.parent),**fields))
        if path.name=='RECORD':
            for record in csv.reader(io.StringIO(row['text'])):
                require(len(record)==3, 'Complete RECORD CSV row')
                candidate=os.path.normpath(str(SITE/record[0]))
                if not Path(candidate).is_relative_to(SITE):external_record_claims+=1
                owners[candidate].append(dict(distribution=path.parent.name,record=record))
    for row in entries:
        if row['scope']=='official':row['git_index']=index.get(str(Path(row['path']).relative_to(OFFICIAL)))
        if row['scope']=='site' and row['kind']!='dir':row['record_claims']=owners.get(row['path'],[])
    unknown=[row['path'] for row in entries if row['kind'] not in ('dir',) and row['excluded'] is None
             and ((row['scope']=='site' and not row['record_claims']) or
                  (row['scope']=='official' and row['git_index'] is None))]
    abnormalities=[row['path'] for row in entries if row['excluded'] is None and row['kind']!='dir'
        and (row['kind']!='file' or row['identity']['st_nlink']!=1 or row['identity']['st_uid']!=os.getuid())]
    official_names={x['path'] for x in entries if x['scope']=='official'}
    missing_git=[str(OFFICIAL/name) for name in index if str(OFFICIAL/name) not in official_names]
    torch=[p for p in packages if p.get('name','').lower()=='torch']
    candidates=[]
    for row in metadata:
        if row['status']=='read' and row['path'].endswith('installation.json'):
            report=json.loads(row['text'])
            for item in report.get('install',[]):
                if item.get('metadata',{}).get('name','').lower()=='torch':
                    candidates.append(dict(report=row['path'],download_info=item.get('download_info')))
    counted=[x for x in entries if x['inventory_eligible']]
    observed=[x for x in entries if x['kind']=='file']
    return dict(entry_kinds=dict(collections.Counter(x['kind'] for x in entries)),
        observed_regular_files=len(observed),observed_regular_bytes=sum(x['identity']['st_size'] for x in observed),
        by_scope={scope:dict(observed_regular_files=sum(x['scope']==scope for x in observed),
            observed_regular_bytes=sum(x['identity']['st_size'] for x in observed if x['scope']==scope),
            eligible_inventory_files=sum(x['scope']==scope for x in counted),
            eligible_inventory_bytes=sum(x['identity']['st_size'] for x in counted if x['scope']==scope))
            for scope in ('site','official')},
        official_index_available=results[1]['returncode']==0,
        inventory_files=len(counted),inventory_bytes=sum(x['identity']['st_size'] for x in counted),
        largest_inventory_file_bytes=max((x['identity']['st_size'] for x in counted),default=0),
        symlinks=sum(x['kind']=='link' for x in entries),
        hardlinked_regular_files=sum(x['kind']=='file' and x['identity']['st_nlink']!=1 for x in entries),
        empty_files=sum(x['kind']=='file' and x['identity']['st_size']==0 for x in entries),
        exclusions=[dict(path=x['path'],reason=x['excluded']) for x in entries if x['excluded']],
        unclaimed_names=unknown,abnormal_file_names=abnormalities,missing_git_names=missing_git,
        record_claims_outside_site=external_record_claims,record_paths_followed=False,
        issues=issues,packages=packages,torch=dict(packages=torch,installation_report_candidates=candidates,
            direct_url_metadata=[dict(path=x['path'],status=x['status']) for x in metadata if Path(x['path']).parent.name.startswith('torch-') and Path(x['path']).name=='direct_url.json'],
            independently_authenticated_wheel=False,unknown_reason='No wheel bytes or trusted archive provenance verified by this census'),
        cpython=dict(pyvenv_cfg=str(ENV/'pyvenv.cfg'),interpreter_bytes_read=False,
            trusted_distribution_source_verified=False,unknown_reason='pyvenv.cfg and readlink are location clues only'),
        record_is_independent_trust=False,working_tree_content_verified=False,
        all_payload_provenance='unknown: claims and stat identities are not authenticated content')


def collect(*, guard):
    budget=Budget(guard);budget.check()
    results=[run_command(argv,budget) for argv in commands()]
    index,_=git_index(results[1])
    entries=scan(SITE,'site',budget)+scan(OFFICIAL,'official',budget,index=index)
    paths=[Path(x['path']) for x in entries if x['scope']=='site'
           and Path(x['path']).parent.parent==SITE and Path(x['path']).parent.name.endswith('.dist-info')
           and Path(x['path']).name in META_NAMES]
    # Every discovered distribution gets explicit missing/unsafe metadata rows.
    for row in entries:
        path=Path(row['path'])
        if row['scope']=='site' and row['kind']=='dir' and path.parent==SITE and path.name.endswith('.dist-info'):
            paths.extend(path/name for name in META_NAMES)
    metadata=[metadata_file(path,budget) for path in sorted(set([*paths,*ANCHORS]))]
    summary=summarize(entries,metadata,results);budget.check()
    return dict(schema='dory_stage_a_native_metadata_census_v2',complete=True,
        entries=entries,metadata=metadata,commands=results,summary=summary,
        limits=dict(LIMITS),usage=dict(budget.used),elapsed_seconds=time.monotonic()-budget.started,
        skipped_commands=[dict(argv=['git','-C',str(OFFICIAL),'status','--porcelain=v1','--untracked-files=no'],
            reason='Index refresh may read working-tree bodies; deferred until content-verification admission')],
        completeness_scope='All observed names below SITE/OFFICIAL except contents of OFFICIAL/.git',
        inventory_eligibility_rule='SITE nonexcluded regular files; OFFICIAL nonexcluded stage-0 tracked regular files only',
        observed_raw_bytes_include_untracked_build_outputs=True,
        inventory_eligibility_is_authentication=False,
        git_administrative_boundary=str(OFFICIAL/'.git'),
        globally_atomic_snapshot=False,per_directory_and_metadata_identity_checks=True,
        file_content_hashes_computed_only_for_allowlisted_metadata=True,
        native_or_model_payload_bytes_read=0,complete_authenticated_inventory=False,
        lock_acquired_or_released=False,output_written=False,runtime_admitted_here=False,
        actual_process_exit_observed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)


if __name__=='__main__':
    raise RuntimeError('Library only: root supplies current serial/resource guard to collect(); no automatic admission')
