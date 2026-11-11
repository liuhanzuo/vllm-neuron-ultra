"""Finite provenance finishing API for a root-admitted serial metadata worker.

No lock acquisition, native imports, installation, network or automatic launch.
Root supplies original admission/guard and a durable bounded event sink. Actual
package-source authority may remain unresolved; observations are never promoted
to independent source evidence. No model qualification is produced.
"""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import struct
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT/'research'
ENV = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
SITE = ENV/'lib/python3.12/site-packages'
OFFICIAL = Path('/mnt/nvme/dory-official-cpu-20261108-v1/vllm')
PYROOT = Path('/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu')
COMMIT = '75fc05a53a366068f534a61ab3ee6b50485419f2'
LIMITS = dict(address_space_bytes=2*2**30, cpu_seconds=600, wall_seconds=900,
              output_bytes=64*2**20, per_output_bytes=32*2**20,
              metadata_file_bytes=8*2**20, event_compressed_bytes=32*2**20,
              event_raw_bytes=256*2**20, event_line_bytes=64*2**20,
              census_raw_bytes=128*2**20, event_count=200000,
              inventory_files=100000, inventory_bytes=8*2**30,
              official_bytes=256*2**20, system_hash_bytes=256*2**20,
              external_files=64, commands=192, command_bytes=8*2**20,
              command_total_bytes=32*2**20, command_seconds=30,
              root_floor_bytes=28*2**30, nvme_floor_bytes=30*2**30)
# Read-policy exception only for the two observed external system library paths.
# This is not package/source authentication and never applies to SITE/OFFICIAL.
SYSTEM_HARDLINK_PATHS = frozenset(('/usr/lib/x86_64-linux-gnu/libc.so.6',
                                   '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30'))
FIELDS = ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
PINS = {
 'metadata_exit': dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-full-wheel-metadata-r1-actual-exit-20261111-v1.json',sha256='34a32a602ba38fef1c885474aa217e6764b7830415f6aec033b4f33f6ca8afb2'),
 'relocations_exit': dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-full-wheel-relocations-r1-actual-exit-20261111-v1.json',sha256='7f1209dc781d69cfaaaec94abed7d8d3233ca65ba9101f7066ebf8dc9f8739a0'),
 'uv_confirmation': dict(path='/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-uv-sysconfig-root-confirmation-20261111-v1.json',sha256='0868d594a99a83330fe7719561e0d869f36a5d5f2ae0ccad8f6d1f07bf8475c8'),
 'wheel_review': dict(path=str(Q/'dory-full-wheel-comparison-r1-post-exit-root-review-20261111-v1.json'),sha256='96c6dcd6e4d16a05c7865c7db0feaa4670880c7e0191f63f49b10fb1e338d523'),
 'manifest': dict(path=str(Q/'dory-full-wheel-source-manifest-20261111-v1.json'),sha256='f4bf58eea14dd3d73856f6a5e4b14be1423560f059aedeef2afd5ef6f8656cea'),
 'python_review': dict(path=str(Q/'dory-torch-cpython-archive-r2-post-exit-root-review-20261111-v1.json'),sha256='26456bac52441bd98c201e43ac8481076286cfd43f43e0ac0f47bd645b4a7e04'),
 'metadata_result': dict(path='/mnt/nvme/dory-full-wheel-differences-r1-20261111/result.json',sha256='eb5d1c0a8525cae6245d7dccd3d89492c648f85adadacb5ab2c9d5c9839d5872'),
 'relocations': dict(path='/mnt/nvme/dory-full-wheel-differences-r1-20261111/record-relocations.json',sha256='0325f7028866d256895c1c0760ad3f1cf9cc24c2f940e1706330177ab142df5c'),
}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def ident(info):
    return {key:getattr(info,key) for key in FIELDS}


def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Unique JSON keys');result[key]=value
    return result


def decode(raw):
    return json.loads(raw,object_pairs_hook=no_duplicates,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON only')))


class Budget:
    def __init__(self,guard):
        require(callable(guard),'Original actual runtime guard required')
        self.guard=guard;self.started=time.monotonic();self.used=collections.Counter()
    def check(self):
        self.guard();require(time.monotonic()-self.started<LIMITS['wall_seconds'],'Finishing wall bound')
    def add(self,key,amount):
        self.check();require(type(amount)is int and amount>=0,'Nonnegative budget')
        self.used[key]+=amount;require(self.used[key]<=LIMITS[key],key+' cap')


def file_bytes(ref,budget,cap,*,retain=False,budget_key=None,git_blob=None):
    """Regular canonical stable file; source SHA optional only for observations."""
    path=Path(ref['path']);budget.check()
    require(path.is_absolute()and path.resolve()==path,'Canonical actual file')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and 0<=before.st_size<=cap,'Bounded single-link ordinary file: '+json.dumps(dict(path=str(path),regular=stat.S_ISREG(before.st_mode),nlink=before.st_nlink,size=before.st_size,cap=cap),sort_keys=True))
        require(ref.get('bytes',before.st_size)==before.st_size,'Expected file size')
        if budget_key:budget.add(budget_key,before.st_size)
        h=hashlib.sha256();g=hashlib.sha1(('blob '+str(before.st_size)+'\0').encode()) if git_blob else None
        chunks=[];count=0
        while count<before.st_size:
            budget.check();part=os.read(fd,min(65536,before.st_size-count));require(part,'Complete file read')
            count+=len(part);h.update(part)
            if g:g.update(part)
            if retain:chunks.append(part)
        require(os.read(fd,1)==b'','Exact file EOF')
        require(ident(before)==ident(os.fstat(fd))==ident(path.lstat())and path.resolve()==path,'Stable complete file')
        actual=h.hexdigest();require(ref.get('sha256',actual)==actual,'Expected complete SHA256')
        if g:require(g.hexdigest()==git_blob,'Original commit blob bytes')
        return dict(path=str(path),bytes=count,sha256=actual,identity=ident(before),uid=before.st_uid),b''.join(chunks)if retain else None
    finally:os.close(fd)


def external_system_file_bytes(ref,budget):
    """Only two canonical root-owned system paths may have exactly two links.

    Full bytes/size/hash and before/fd/path identity checks remain mandatory.
    This returns an observation; it supplies no independent source authority.
    Every other path still uses the unchanged single-link policy.
    """
    path=Path(ref['path']);cap=64*2**20
    if str(path)not in SYSTEM_HARDLINK_PATHS:
        return file_bytes(ref,budget,cap,budget_key='system_hash_bytes')
    budget.check();require(path.is_absolute()and path.resolve()==path,'Canonical actual system library')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode)and before.st_nlink==2 and before.st_uid==before.st_gid==0
                and not (before.st_mode & 0o022)and 0<=before.st_size<=cap,
                'Explicit two-link root system library: '+json.dumps(dict(path=str(path),regular=stat.S_ISREG(before.st_mode),nlink=before.st_nlink,uid=before.st_uid,gid=before.st_gid,mode=before.st_mode,size=before.st_size,cap=cap),sort_keys=True))
        require(ref.get('bytes',before.st_size)==before.st_size,'Expected system library size')
        budget.add('system_hash_bytes',before.st_size);h=hashlib.sha256();count=0
        while count<before.st_size:
            budget.check();part=os.read(fd,min(65536,before.st_size-count));require(part,'Complete system library read')
            count+=len(part);h.update(part)
        require(os.read(fd,1)==b'','Exact system library EOF')
        after=os.fstat(fd);named=path.lstat()
        require(ident(before)==ident(after)==ident(named)and path.resolve()==path
                and after.st_uid==after.st_gid==named.st_uid==named.st_gid==0,
                'Stable complete two-link system library: '+str(path))
        actual=h.hexdigest();require(ref.get('sha256',actual)==actual,'Expected complete system library SHA256')
        return dict(path=str(path),bytes=count,sha256=actual,identity=ident(before),uid=before.st_uid,
                    system_hardlink_read_policy='two_explicit_root_owned_paths_exactly_two_links'),None
    finally:os.close(fd)


def load_json(ref,budget,cap=None):
    _,raw=file_bytes(ref,budget,cap or LIMITS['metadata_file_bytes'],retain=True)
    return decode(raw)


def gzip_records(ref,budget,*,lines):
    """Two bounded reads of one stable fd: authenticate compressed, then raw."""
    path=Path(ref['path']);require(path.resolve()==path,'Canonical saved gzip')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb')as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size==ref['bytes']<=LIMITS['event_compressed_bytes'],'Saved gzip size')
        h=hashlib.sha256();count=0
        while True:
            budget.check();block=stream.read(65536)
            if not block:break
            count+=len(block);h.update(block);require(count<=ref['bytes'],'Bounded compressed input')
        require(count==ref['bytes']and h.hexdigest()==ref['sha256'],'Complete compressed evidence SHA')
        stream.seek(0);h=hashlib.sha256();count=0;rows=0
        require(ref['uncompressed_bytes']<=(LIMITS['event_raw_bytes']if lines else LIMITS['census_raw_bytes']),'Declared raw evidence cap')
        with gzip.GzipFile(fileobj=stream,mode='rb')as zipped:
            if lines:
                while True:
                    budget.check();raw=zipped.readline(LIMITS['event_line_bytes']+1)
                    if not raw:break
                    require(len(raw)<=LIMITS['event_line_bytes']and raw.endswith(b'\n'),'Bounded complete JSONL line')
                    count+=len(raw);rows+=1
                    require(count<=ref['uncompressed_bytes']and rows<=LIMITS['event_count'],'Raw event bounds')
                    h.update(raw);yield decode(raw)
            else:
                raw=bytearray()
                while True:
                    budget.check();part=zipped.read(65536)
                    if not part:break
                    count+=len(part);require(count<=ref['uncompressed_bytes'],'Raw census bound');h.update(part);raw.extend(part)
                yield decode(raw)
        require(count==ref['uncompressed_bytes']and h.hexdigest()==ref['uncompressed_sha256'],'Complete raw evidence SHA')
        if lines:require(rows==ref['events'],'Complete named event count')
        require(ident(before)==ident(os.fstat(stream.fileno()))==ident(path.lstat()),'Stable saved evidence')


def result_ref(row):
    require(type(row.get('bytes'))is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row.get('sha256','')),'Observed complete content ref')
    return {key:row[key]for key in ('path','sha256','bytes')}


def explanation_maps(metadata,relocations):
    require(len(metadata['records'])==163 and len(metadata['scripts'])==5,'All original differences explained')
    require(relocations['total']==len(relocations['records'])==65 and relocations['no_unexplained_original_record_row_changes'] is True,'All65 relocations reviewed')
    explained=set()
    for row in relocations['records']:
        require(row['verified']is True and row['installed_fields']==row['expected_installed_fields'],'Exact relocated RECORD fields')
        key=(row['package'],row['upstream_path']);require(key not in explained,'Unique relocation explanation');explained.add(key)
    record_map={}
    for row in metadata['records']:
        require(not row['changed'],'No unexplained common RECORD change')
        require(all((row['package'],name)in explained for name in row['missing']),'Every missing RECORD path explained')
        require(row['path']not in record_map,'Unique explained installed RECORD');record_map[row['path']]=row
    require(explained=={(r['package'],n)for r in metadata['records']for n in r['missing']},'No surplus relocation records')
    scripts={}
    for row in metadata['scripts']:
        require(row['body_identical']is True and row['expected_interpreter_rewrite_exact']is True,'Exact interpreter-only script change')
        require(row['path']not in scripts,'Unique script explanation');scripts[row['path']]=row
    return record_map,scripts


def collect_site(events,census,metadata,relocations,emit):
    records,scripts=explanation_maps(metadata,relocations)
    inventory={};observed={};provenance={};external={};excluded=[];unresolved=[];counts=collections.Counter();current=None
    def add(path,reference,kind,evidence):
        require(path not in inventory,'Unique SITE content authority')
        require(Path(path).is_relative_to(SITE),'Original SITE only')
        inventory[path]=reference;provenance[path]=dict(kind=kind,evidence=evidence)
    for event in events:
        kind=event['event'];counts[kind]+=1
        if kind=='current_environment_metadata':
            require(current is None,'One complete current environment');current={r['path']:r for r in event['entries']}
            require(len(current)==len(event['entries']),'Unique current names')
        elif kind=='wheel_member':
            path=event.get('path');require(path is not None,'No unmapped wheel member remains')
            if event['status']=='exact':
                require(event['sha256']==event['archive_sha256']and event['bytes']==event['archive_bytes'],'Exact full wheel member')
                origin=dict(package=event['package'],member=event['member'],archive_sha256=event['archive_sha256'])
                if event['scope']=='site':add(path,result_ref(event),'upstream_wheel_member',origin)
                else:external[path]=dict(reference=result_ref(event),identity=event['identity'],source='upstream_wheel_member',evidence=origin)
            elif path in records:
                require(event['sha256']==records[path]['installed_sha256']and event['archive_sha256']==records[path]['upstream_sha256'],'Exact reviewed RECORD identities')
                add(path,result_ref(event),'reviewed_installer_RECORD',dict(metadata_result=PINS['metadata_result'],relocations=PINS['relocations']))
            elif path in scripts:
                require(event['sha256']==scripts[path]['installed_sha256']and event['archive_sha256']==scripts[path]['upstream_sha256'],'Exact reviewed script identities')
                external[path]=dict(reference=result_ref(event),identity=event['identity'],source='exact_interpreter_line_rewrite',evidence=PINS['metadata_result'])
            else:unresolved.append(dict(path=path,reason='unexplained_wheel_difference',row=event))
        elif kind=='installed_additional':
            path=event['path']
            if event['scope']=='site':
                if event['excluded']=='cached_bytecode':excluded.append(dict(path=path,identity=event['identity'],reason='resolver_rejects_cached_bytecode_after_activation'))
                elif (event['classification']=='installer_generated_metadata_unverified'and Path(path).parent.name.endswith('.dist-info')
                      and Path(path).name in ('INSTALLER','REQUESTED','direct_url.json','uv_cache.json')):
                    add(path,result_ref(event['current_bytes']),'bounded_generated_metadata_observation',dict(event_source=PINS['wheel_review'],upstream_authenticated=False))
                else:unresolved.append(dict(path=path,reason='unexplained_SITE_addition',row=event))
            else:
                external[path]=dict(reference=result_ref(event['current_bytes'])if event.get('current_bytes')else None,
                                    identity=event['identity'],source='saved_external_observation',kind=event['kind'],link_target=event.get('link_target'))
        elif kind=='eligibility_crosswalk' and event.get('current',{}):
            if event['current']['scope']=='site'and event['current']['inventory_eligible']:
                observed[event['path']]=event['current']
                if event['original_census'] and event['status']!='census_identity_unchanged':
                    unresolved.append(dict(path=event['path'],reason='SITE_census_identity_changed',row=event))
    require(current is not None,'Complete environment event required')
    original={r['path']:r for r in census['entries']if r['scope']=='site'and r['inventory_eligible']}
    expected={p:r for p,r in current.items()if r['scope']=='site'and r['inventory_eligible']}
    require(set(expected)==set(observed),'Every current eligible SITE row crosswalked')
    require(set(original)==set(expected),'Original eligible SITE named set unchanged')
    missing=sorted(set(expected)-set(inventory));surplus=sorted(set(inventory)-set(expected))
    for path in missing:unresolved.append(dict(path=path,reason='eligible_SITE_without_content_reference'))
    require(not surplus,'No surplus SITE inventory member')
    for path,reference in inventory.items():
        require(reference['bytes']==expected[path]['identity']['st_size'],'Current SITE size bound')
        emit(dict(event='SITE_inventory',reference=reference,identity=expected[path]['identity'],provenance=provenance[path]))
    for row in excluded:emit(dict(event='SITE_excluded',**row))
    for row in unresolved:emit(dict(event='unresolved',**row))
    return dict(files=sorted(inventory.values(),key=lambda x:x['path']),identities={p:expected[p]['identity']for p in inventory},
                provenance=provenance,external=external,excluded=excluded,unresolved=unresolved,event_counts=dict(counts))


def command(argv,budget):
    """No shell; bounded stdout/stderr and always reap child on failure."""
    require(argv[0]in ('/usr/bin/git','/usr/bin/readelf','/usr/bin/dpkg-query','/sbin/ldconfig'),'Finite read-only command family')
    budget.add('commands',1);budget.check();started=time.monotonic()
    env=dict(PATH='/usr/bin:/bin:/sbin',LANG='C.UTF-8',HOME='/home/ubuntu',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_OPTIONAL_LOCKS='0')
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    selector=None;raw=bytearray()
    try:
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        while selector.get_map():
            budget.check();require(time.monotonic()-started<LIMITS['command_seconds'],'Command wall bound')
            for key,_ in selector.select(.1):
                block=os.read(key.fileobj.fileno(),65536)
                if not block:selector.unregister(key.fileobj);continue
                budget.add('command_total_bytes',len(block));raw.extend(block)
                require(len(raw)<=LIMITS['command_bytes'],'Command output cap')
        code=proc.wait(timeout=max(.01,LIMITS['command_seconds']-(time.monotonic()-started)))
        return dict(argv=argv,returncode=code,stdout=raw.decode('utf8','surrogateescape'))
    finally:
        # Always attempt every cleanup step, including failed selector setup.
        # An active primary error wins; cleanup details remain attached to it.
        primary=sys.exc_info()[1];cleanup_errors=[]
        def attempt(label,action):
            try:return action()
            except BaseException as error:
                cleanup_errors.append(dict(step=label,error_type=type(error).__name__,error=str(error)))
        if selector is not None:attempt('selector_close',selector.close)
        exited=attempt('process_poll',proc.poll)
        if exited is None:attempt('process_kill',proc.kill)
        attempt('process_wait',lambda:proc.wait(timeout=5))
        if proc.stdout is not None:attempt('stdout_close',proc.stdout.close)
        if cleanup_errors:
            if primary is not None:
                primary.dory_command_cleanup_errors=cleanup_errors
                if hasattr(primary,'add_note'):
                    primary.add_note('Dory command cleanup errors: '+json.dumps(cleanup_errors))
            else:
                error=RuntimeError('Dory command cleanup failed: '+json.dumps(cleanup_errors))
                error.dory_command_cleanup_errors=cleanup_errors
                raise error


def parse_tree(text):
    found={}
    for row in text.split('\0'):
        if not row:continue
        head,name=row.split('\t',1);mode,kind,blob=head.split()
        require(name not in found and not name.startswith('/')and all(x not in ('','.','..')for x in name.split('/')),'Canonical unique Git tree name')
        require(re.fullmatch('[0-9a-f]{40}',blob),'Original Git blob identifier')
        found[name]=dict(mode=mode,kind=kind,blob=blob)
    return found


def official_inventory(census,budget,emit):
    base=['/usr/bin/git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(OFFICIAL)]
    head=command(base+['rev-parse','HEAD'],budget);require(head['returncode']==0 and head['stdout'].strip()==COMMIT,'Original official HEAD')
    tree=command(base+['ls-tree','-rz','--full-tree',COMMIT],budget);require(tree['returncode']==0,'Complete fixed commit tree')
    emit(dict(event='official_git_commands',head=head,tree=tree))
    entries=parse_tree(tree['stdout'])
    selected={str(Path(r['path']).relative_to(OFFICIAL)):r for r in census['entries']if r['scope']=='official'and r['inventory_eligible']}
    tracked={name:r for name,r in entries.items()if r['kind']=='blob'and r['mode']in ('100644','100755')}
    require(set(selected)==set(tracked),'Original complete tracked regular-source set')
    result=[]
    for name,row in sorted(selected.items()):
        budget.check();require(row['git_index']['blob']==tracked[name]['blob'],'Saved census index matches original commit')
        record,_=file_bytes(dict(path=str(OFFICIAL/name)),budget,LIMITS['official_bytes'],budget_key='official_bytes',git_blob=tracked[name]['blob'])
        require(record['uid']==os.getuid(),'Original project-owned source')
        result.append(result_ref(record));emit(dict(event='official_inventory',reference=result_ref(record),identity=record['identity'],git_blob=tracked[name]['blob'],commit=COMMIT))
    return result


def parse_elf(text):
    needed=re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',text)
    runpath=re.findall(r'\((?:RUNPATH|RPATH)\).*?\[([^\]]*)\]',text)
    interpreter=re.findall(r'Requesting program interpreter:\s*([^\]]+)\]',text)
    require(len(interpreter)<=1 and len(needed)<=64 and all('/'not in n for n in needed),'Bounded simple ELF dependencies')
    return dict(needed=needed,search_paths=runpath,interpreter=interpreter[0]if interpreter else None)


def parse_ldcache(text):
    result={}
    for line in text.splitlines():
        match=re.match(r'\s*(\S+)\s+\(([^)]+)\)\s+=>\s+(/\S+)\s*$',line)
        if match and 'x86-64'in match[2]:result.setdefault(match[1],[]).append(match[3])
    return result


def dependency_candidates(needed,path,elf,cache,exists):
    direct=[];unknown=[]
    for group in elf['search_paths']:
        for directory in group.split(':'):
            directory=directory.replace('${ORIGIN}',str(Path(path).parent)).replace('$ORIGIN',str(Path(path).parent))
            if not directory or '$'in directory or not directory.startswith('/'):
                unknown.append(directory);continue
            candidate=str(Path(directory)/needed)
            if exists(candidate):direct.append(candidate)
    # RPATH/RUNPATH and CPU-hwcaps precedence is not guessed if multiple matches.
    return sorted(set(direct+cache.get(needed,[]))),unknown


def pyc_decision(header,source_size,source_mtime,magic=b'\xcb\r\r\n'):
    if len(header)<16:return dict(selection='source_fallback',reason='truncated_header')
    if header[:4]!=magic:return dict(selection='source_fallback',reason='wrong_python312_magic')
    flags=struct.unpack('<I',header[4:8])[0]
    if flags&~3:return dict(selection='source_fallback',reason='invalid_flags',flags=flags)
    if flags&1:return dict(selection='unresolved_hash_based_mode',flags=flags,source_hash_hex=header[8:16].hex())
    mtime,size=struct.unpack('<II',header[8:16])
    matches=mtime==(int(source_mtime)&0xffffffff)and size==(source_size&0xffffffff)
    return dict(selection='timestamp_cache_selected_content_unverified'if matches else 'source_fallback',
                reason='timestamp_size_match_is_not_content_authentication'if matches else 'timestamp_or_size_mismatch',
                flags=flags,stored_mtime=mtime,stored_source_size=size)


def pyc_evidence(source_ref,budget):
    observed,_=file_bytes(source_ref,budget,128*2**10)
    source=Path(source_ref['path']);path=source.parent/'__pycache__'/(source.stem+'.cpython-312.pyc')
    result=dict(source=result_ref(observed),source_identity=observed['identity'],cache_path=str(path))
    try:before=path.lstat()
    except FileNotFoundError:return dict(result,selection='source_fallback',reason='cache_absent')
    require(path.resolve()==path and stat.S_ISREG(before.st_mode),'Ordinary no-link cache header')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        require(ident(os.fstat(fd))==ident(before),'Stable cache header open');header=os.read(fd,16)
        require(ident(os.fstat(fd))==ident(before)==ident(path.lstat()),'Stable cache header')
    finally:os.close(fd)
    result.update(cache_identity=ident(before),header_hex=header.hex(),header_payload_bytes_read=len(header),
                  **pyc_decision(header,observed['bytes'],observed['identity']['st_mtime_ns']//10**9))
    return result


def _exact_external_reference(reference,*,expected_path=None,cap,metadata=False):
    """Strict reference syntax only; never turns observed bytes into authority."""
    require(type(reference)is dict and set(reference)=={'path','bytes','sha256'},
            'Exact external reference fields')
    path=reference['path']
    require(type(path)is str and '\\'not in path and '\0'not in path
            and Path(path).is_absolute() and str(Path(path))==path
            and all(part not in ('','.','..')for part in path.split('/')[1:]),
            'Canonical absolute external reference path')
    require(expected_path is None or path==expected_path,'Exact authority destination')
    require(type(reference['bytes'])is int and (1 if metadata else 0)<=reference['bytes']<=cap,
            'Explicit bounded external reference byte count')
    require(type(reference['sha256'])is str and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
            'Explicit external expected SHA256')
    if metadata:require(Path(path).suffix=='.json','Source evidence is bounded JSON metadata reference')
    return dict(reference)


def validate_external_authorities(authorities):
    """Copy and validate all supplied records before external commands or bytes.

    source_evidence is one exact metadata reference. This function does not read
    or authenticate that metadata or establish its provenance. Root must review
    and authenticate its bytes and meaning independently before supplying it.
    """
    if authorities is None:return {}
    require(type(authorities)is dict and len(authorities)<=LIMITS['external_files'],
            'Bounded external authority mapping')
    checked={}
    for path,row in authorities.items():
        require(type(path)is str and type(row)is dict and set(row)=={'reference','source_evidence'},
                'Explicit external authority and evidence fields')
        reference=_exact_external_reference(row['reference'],expected_path=path,cap=64*2**20)
        evidence=_exact_external_reference(row['source_evidence'],cap=LIMITS['metadata_file_bytes'],metadata=True)
        require(evidence['path']!=reference['path'],'Separate independent source evidence metadata')
        checked[path]=dict(reference=reference,source_evidence=evidence)
    return checked


def external_closure(site,census,python_review,python_comparison,budget,emit,authorities=None):
    """Finite startup/native seeds. Caller evidence gaps remain explicit."""
    authorities=validate_external_authorities(authorities)
    known={p:dict(reference=r,source='authenticated_SITE',identity=site['identities'][p])for p,r in ((r['path'],r)for r in site['files'])}
    known.update(site['external'])
    pyfiles=python_comparison['findings']['cpython']['files']
    pyrefs={r['path']:dict(path=r['path'],sha256=r['archive_sha256'],bytes=r['archive_bytes'])for r in pyfiles if r['status']=='exact'}
    interpreter_sha='23bc148d2b48964b621c5561ff2b81c1d9c7b717967a7491efbf12f4304882b5'
    interpreter=str(ENV/'bin/python');require(known[interpreter]['reference']['sha256']==interpreter_sha,'Saved interpreter matches verified CPython binary')
    known[interpreter]['source']='verified_CPython_copy'
    preload=str(ENV/'lib/libiomp5.so');require(known.get(preload,{}).get('source')=='upstream_wheel_member','Preload supplied by full163 wheel member')
    seeds=[interpreter,preload,'/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4']
    seeds.extend(r['path']for r in site['files']if Path(r['path']).parent==SITE/'torch'and Path(r['path']).name.startswith('_C.')and r['path'].endswith('.so'))
    seeds.extend(str(SITE/'vllm'/name)for name in ('_C.abi3.so','_C_AVX2.abi3.so','_C_AVX512.abi3.so','fs_io_C.abi3.so','spinloop.abi3.so')if str(SITE/'vllm'/name)in known)
    cache_result=command(['/sbin/ldconfig','-p'],budget);require(cache_result['returncode']==0,'System loader cache metadata')
    cache=parse_ldcache(cache_result['stdout']);emit(dict(event='loader_cache',**cache_result))
    pending=list(seeds);seen=set();rows=[];issues=[]
    while pending:
        budget.check();requested=pending.pop(0);path=Path(requested)
        resolved=str(path.resolve());require(Path(resolved).is_absolute(),'Absolute dependency destination')
        if resolved in seen:continue
        require(len(seen)<LIMITS['external_files'],'Finite ELF closure cap');seen.add(resolved)
        require(any(Path(resolved).is_relative_to(p)for p in (ENV,PYROOT,Path('/usr/lib'),Path('/lib'))),'Finite original dependency roots')
        link=dict(requested=requested,resolved=resolved,is_symlink=path.is_symlink(),link_target=os.readlink(path)if path.is_symlink()else None)
        emit(dict(event='external_dependency_attempt',link=link))
        if resolved in known and known[resolved].get('reference'):
            evidence=known[resolved];before=Path(resolved).lstat()
            require(ident(before)==evidence['identity']and Path(resolved).resolve()==Path(resolved),'Existing authenticated dependency identity unchanged')
            observed=dict(**evidence['reference'],identity=ident(before));authority=evidence['source']
        elif resolved in pyrefs:
            observed,_=file_bytes(pyrefs[resolved],budget,64*2**20,budget_key='system_hash_bytes');authority='verified_CPython_archive'
        else:
            expected=authorities.get(resolved)
            observed,_=external_system_file_bytes(expected['reference']if expected else dict(path=resolved),budget)
            authority='caller_independent_source_verified'if expected else 'observed_hash_only'
            if not expected:issues.append(dict(path=resolved,reason='independent_system_package_bytes_not_yet_bound'))
        elf_result=command(['/usr/bin/readelf','-lW','-dW',resolved],budget)
        require(elf_result['returncode']==0,'Original ELF header analysis');elf=parse_elf(elf_result['stdout'])
        packages=command(['/usr/bin/dpkg-query','-S',resolved],budget)if not Path(resolved).is_relative_to(ENV)and not Path(resolved).is_relative_to(PYROOT)else None
        dependencies=[]
        if elf['interpreter']:pending.append(elf['interpreter'])
        for name in elf['needed']:
            candidates,unknown=dependency_candidates(name,resolved,elf,cache,lambda p:Path(p).is_file())
            targets=sorted({str(Path(p).resolve())for p in candidates})
            dependencies.append(dict(name=name,candidates=candidates,resolved_candidates=targets,unresolved_search_paths=unknown))
            if len(targets)==1 and not unknown:pending.append(targets[0])
            else:issues.append(dict(path=resolved,needed=name,reason='ambiguous_or_unresolved_loader_search',candidates=targets,unknown=unknown))
        row=dict(link=link,reference=result_ref(observed),identity=observed['identity'],authority=authority,
                 supplied_authority=authorities.get(resolved),elf=elf,elf_command=elf_result,dpkg_owner=packages,dependencies=dependencies,
                 system_hardlink_read_policy=observed.get('system_hardlink_read_policy'))
        rows.append(row);emit(dict(event='external_dependency',**row))
    targets=[str(PYROOT/'lib/python3.12/encodings'/name)for name in ('__init__.py','aliases.py','utf_8.py')]
    sysconfig=str(PYROOT/'lib/python3.12/_sysconfigdata__linux_x86_64-linux-gnu.py')
    pyrefs[sysconfig]=dict(path=sysconfig,bytes=51256,sha256='cc3e58175aff8e0b034e55acbbf277613cfab33913cf904bb6c2daff2d985d5f')
    bytecode=[]
    for path in [*targets,sysconfig]:
        require(path in pyrefs,'Previously verified source for targeted cache')
        row=pyc_evidence(pyrefs[path],budget);bytecode.append(row);emit(dict(event='targeted_pyc',**row))
        if row['selection']!='source_fallback':issues.append(dict(path=row['cache_path'],reason=row['selection']))
    return dict(seeds=seeds,files=rows,pyc=bytecode,unresolved=issues,
                static_selected_dependencies_only=True,actual_loaded_maps_pending=True,
                supplied_source_evidence_read_or_authenticated_here=False,
                supplied_source_evidence_authority_is_root_review_precondition=True,
                system_package_versions_from_original_census=[r for r in census['commands']if r['argv'][0]=='/usr/bin/dpkg-query'])


def json_blocks(value):
    """Same compact ASCII JSON stream, buffered before guarded budget/write IO."""
    buffered=bytearray()
    # The unchanged encoder defaults to ensure_ascii=True. Slice its ASCII text
    # before encoding so even a large string fragment adds at most64KiB at once.
    for part in json.JSONEncoder(separators=(',',':'),allow_nan=False).iterencode(value):
        offset=0
        while offset<len(part):
            take=min(65536-len(buffered),len(part)-offset)
            buffered.extend(part[offset:offset+take].encode());offset+=take
            if len(buffered)==65536:
                yield buffered;buffered=bytearray()
    if buffered:yield buffered


def write_json(path,value,budget):
    """Exclusive bounded output; caller preserves partial files on any failure."""
    path=Path(path);h=hashlib.sha256();count=0
    with path.open('xb')as stream:
        budget.check()
        for block in json_blocks(value):
            count+=len(block);require(count<=LIMITS['per_output_bytes'],'Single output bound')
            budget.add('output_bytes',len(block));require(stream.write(block)==len(block),'Complete output write');h.update(block)
        stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    observed,_=file_bytes(dict(path=str(path),bytes=count,sha256=h.hexdigest()),budget,LIMITS['per_output_bytes'])
    return result_ref(observed)


def run(output_root,guard,emit,*,external_authorities=None):
    """Root binds exact sources, latest actual exit, resources and original lock.

    Authoritative input hashes are fixed below; caller-supplied external authority
    must be independently root reviewed. This API neither grants admission nor
    claims that package observations constitute trust. No automatic ABI launch.
    """
    require(callable(emit),'Durable bounded root event sink required');budget=Budget(guard)
    output_root=Path(output_root)
    require(output_root.is_absolute()and output_root.resolve()==output_root and output_root.is_dir()and not any(output_root.iterdir()),'Fresh root-admitted output directory')
    inputs={name:load_json(ref,budget)for name,ref in PINS.items()}
    review=inputs['wheel_review'];manifest=inputs['manifest']
    for key,exit_key in [('metadata_result','metadata_exit'),('relocations','relocations_exit')]:
        ended=inputs[exit_key]
        require(ended['actual_tool']['exit_code']==0 and ended['owner_gone'] and ended['original_lock_reacquired'] and ended['result']==PINS[key],'Actual completed explanation binding')
    require(inputs['uv_confirmation']['complete_bytes_match'] is True and inputs['uv_confirmation']['predicted_installed_sha256']=='cc3e58175aff8e0b034e55acbbf277613cfab33913cf904bb6c2daff2d985d5f','Completed exact uv source transformation')
    require(review['actual_tool']==dict(session=1751,terminal_chunk='f2621a',exit_code=0)and review['complete_saved_events_independently_reread']and review['owner_gone'],'Actual full163 and independent review')
    census_stream=gzip_records(manifest['original_census_archive'],budget,lines=False)
    census=next(census_stream)
    require(next(census_stream,None)is None,'Complete census authenticated')
    events=gzip_records(review['complete_events'],budget,lines=True)
    site=collect_site(events,census,inputs['metadata_result'],inputs['relocations'],emit)
    require(site['event_counts']==review['event_counts'],'Exact reviewed event-kind counts')
    for path,identity in site['identities'].items():
        budget.check();actual=Path(path).lstat()
        require(Path(path).resolve()==Path(path)and stat.S_ISREG(actual.st_mode)and actual.st_nlink==1 and actual.st_uid==os.getuid()and ident(actual)==identity,'Saved SITE identity unchanged before export')
    official=official_inventory(census,budget,emit)
    full_python=gzip_records(inputs['python_review']['full_comparison'],budget,lines=False)
    python_comparison=next(full_python);require(next(full_python,None)is None,'Complete original Python comparison authenticated')
    external=external_closure(site,census,inputs['python_review'],python_comparison,budget,emit,external_authorities)
    files=site['files']+official
    require(len(files)<=LIMITS['inventory_files']and sum(r['bytes']for r in files)<=LIMITS['inventory_bytes'],'Exact resolver inventory bounds')
    refs=dict(files=write_json(output_root/'native-files.json',files,budget),
              provenance=write_json(output_root/'site-provenance.json',site['provenance'],budget),
              external=write_json(output_root/'external-closure.json',external,budget),
              site_excluded=write_json(output_root/'site-excluded.json',site['excluded'],budget))
    unresolved=site['unresolved']+external['unresolved']
    result=dict(schema='dory_native_inventory_finishing_v2',source_successor='dory_native_inventory_finish_v4.py',inputs=PINS,outputs=refs,
                file_count=len(files),file_bytes=sum(r['bytes']for r in files),site_files=len(site['files']),official_files=len(official),
                unresolved=unresolved,site_named_coverage_complete=not site['unresolved'],budget_used=dict(budget.used),
                generated_metadata_is_observational_not_upstream=True,actual_loaded_maps_pending=True,
                complete_authenticated_inventory=False,native_ABI_qualified=False,native_imported=False,
                original_environment_modified=False,full_model_qualified=False,new_speedup_claim=False,
                actual_process_exit_claimed=False,next='Root reviews remaining external authority/cache rows, binds latest actual exit, then fresh resolver/probe ABI child')
    refs['result']=write_json(output_root/'result.json',result,budget)
    emit(dict(event='finishing_complete',result=refs['result'],unresolved_count=len(unresolved),native_ABI_qualified=False))
    return refs['result']


if __name__=='__main__':
    raise RuntimeError('Root-admitted API only: no automatic lock, process probe, ABI or model launch')
