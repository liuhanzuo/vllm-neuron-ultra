"""Bounded raw-byte manifests. Real payload operations are explicit runtime only."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import nemotron9b_package_protocol_v27 as p

RAW={'numeric':'numeric.json.pt','cold':'cold-buckets.pt','long':'long-decode-validation.pt'}
REPORT={'numeric':'numeric.json','cold':'cold-buckets.json','long':'long-decode-validation.json'}
COUNTS={'numeric':32,'cold':84,'long':93}


def source_model_files():
    """Read only the reviewed small Python source; do not import package/native."""
    source=p.PACKAGE/'vllm_neuron/models/nemotron9b/identity.py'
    tree=ast.parse(p.low.text(source).decode())
    node,=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE_FILES'for t in n.targets)]
    return ast.literal_eval(node.value)


def validate_payload_closure(spec):
    model={str(Path(spec['original_model'])/name):row for name,row in source_model_files().items()}
    p.require(set(spec['payloads'])==set(model)|set(spec['references'].values()),'Exact original model and six oracle payload closure')
    for path,row in model.items():
        item=spec['payloads'][path]
        p.require(row['algorithm']=='sha256'and item['sha256']==row['digest']and item['stat']['size']==row['size'],
            'Pinned original model/tokenizer payload identity')


def identity(path):
    path=Path(path)
    p.require(path.is_absolute()and path.resolve()==path and not any(x.is_symlink()for x in (path,*path.parents)),
        'Ordinary canonical payload path')
    info=path.lstat()
    p.require(stat.S_ISREG(info.st_mode)and info.st_nlink==1,'Single-link ordinary evidence file')
    p.require(info.st_uid==os.getuid()and info.st_size<=16*p.GIB,'Owned bounded evidence file')
    return p.file_stat(path)


def digest(path,guard):
    """Hash the actual no-follow FD and reject path swaps or concurrent writes."""
    guard();before=identity(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    def fd_identity():
        info=os.fstat(fd)
        return dict(realpath=before['realpath'],device=info.st_dev,inode=info.st_ino,size=info.st_size,
            mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)
    try:
        p.require(fd_identity()==before,'Payload changed before open')
        value=hashlib.sha256();size=0
        while True:
            guard();block=os.read(fd,p.MIB)
            if not block:break
            size+=len(block);p.require(size<=before['size'],'Payload grew while hashing');value.update(block)
        p.require(size==before['size']and fd_identity()==before and identity(path)==before,'Payload changed while hashing')
        return dict(path=str(path),stat=before,sha256=value.hexdigest())
    finally:os.close(fd)


def ipc_metadata(path,root,spec,guard):
    """Record a detached temporary socket identity without opening its body."""
    guard();path=Path(path);root=Path(root)
    p.require(type(spec)is dict and root==Path(spec['work_root']) and root.is_absolute()
        and root.resolve()==root and path.resolve()==path,'Socket evidence only in exact work root')
    p.require(path.parent==root/'temporary/tmp' and re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',path.name)is not None,
        'Only declared temporary UUID IPC socket evidence')
    def observed():
        info=path.lstat()
        p.require(stat.S_ISSOCK(info.st_mode) and info.st_uid==os.getuid()
            and info.st_nlink==1 and info.st_size==0,'Owned single-link zero-byte socket evidence')
        return dict(realpath=str(path),device=info.st_dev,inode=info.st_ino,size=info.st_size,
            mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=info.st_mode,uid=info.st_uid,nlink=info.st_nlink,blocks=info.st_blocks)
    before=observed()
    with Path('/proc/net/unix').open('rb')as stream:table=stream.read(8*p.MIB+1)
    p.require(len(table)<=8*p.MIB,'Bounded Unix endpoint table')
    for line in table.splitlines()[1:]:
        fields=line.split(None,7)
        p.require(len(fields)>=7,'Complete Unix endpoint table entry')
        p.require(len(fields)==7 or fields[7]!=os.fsencode(path),'Live Unix endpoint cannot be archived as detached')
    guard();p.require(observed()==before,'Socket identity changed during metadata capture')
    return dict(path=str(path),unix_socket=True,detached=True,stat=before)


def tree(root,guard,allow_view=False,spec=None,exclude=()):
    """Bound all files/links; recurse bounded ordinary dirs, never follow links."""
    root=Path(root);p.require(root.is_dir()and root.resolve()==root,'Ordinary output root')
    if allow_view:
        view=root/'view';expected_links={str(view/name):str(Path(spec['original_model'])/name)for name in source_model_files()if name not in ('config.json','tokenizer_config.json')}
    else:expected_links={}
    compiler_root=None;compiler_assets=None
    if type(spec)is dict and root in (Path(spec['work_root']),Path(spec['work_root'])/'compile-work'):
        compiler_root=Path(spec['work_root'])/'compile-work';compiler_assets=p.metadata.load(spec['compiler_asset_policy'])
        guard();p.compiler_links.scope(root,compiler_root,compiler_assets);p.compiler_links.verify_assets(compiler_assets,hash_content=True)
    rows={};count=0;sockets=0
    for base,dirs,files in os.walk(root,followlinks=False):
        guard();dirs.sort();files.sort()
        for name in dirs:p.require(not(Path(base)/name).is_symlink(),'No output directory symlink')
        for name in files:
            path=Path(base)/name
            if str(path)in exclude:continue
            count+=1;p.require(count<100000,'Bounded manifest file count')
            if path.is_symlink():
                if compiler_root is not None:
                    rows[str(path)]=p.compiler_links.link_record(path,compiler_root,compiler_assets,hash_content=True,guard=guard);continue
                p.require(allow_view and path.parent==root/'view','Only prepared model view symlinks')
                target=str(Path(spec['original_model'])/path.name)
                p.require(target in spec['payloads']and os.readlink(path)==target,'Exact original model view target')
                s=path.lstat();rows[str(path)]=dict(path=str(path),symlink=target,
                    stat=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
            elif stat.S_ISSOCK(path.lstat().st_mode):
                sockets+=1;p.require(sockets<=64,'Bounded temporary IPC socket evidence count')
                rows[str(path)]=ipc_metadata(path,root,spec,guard)
            else:rows[str(path)]=digest(path,guard)
    if allow_view:p.require({name:row['symlink']for name,row in rows.items()if 'symlink'in row}==expected_links,'Complete exact model view link closure')
    return rows


def phase_paths(spec,stage):
    e=Path(spec['evidence_root']);w=Path(spec['work_root']);k=p.kind(stage)
    if k=='prepare':return [e/'prepare.json',e/'workload-inputs.json',e/'view']
    if k=='alias':return [w/'primitives/device'/(stage+'.json'),w/'primitives/work'/stage]
    if k=='split':return [w/'primitives/jobs'/stage]
    if k=='compile':return [e/'compile-smoke.json',w/'artifact']
    if k=='state-copy':return [e/'state-copy.json']
    if k in RAW:
        names=[REPORT[k],RAW[k],k+'-migration.json']
        if k in ('numeric','long'):names += [k+'-partial.json',k+'-partial.pt']
        return [e/name for name in names]
    if k=='quality':return [e/'quality96.json',e/'quality-comparison.json']
    if k=='workloads':return [e/(stage+'.json')]
    raise RuntimeError('No scientific output closure for '+stage)


def phase_manifest(spec,stage,owner,lease,guard):
    files={}
    for path in phase_paths(spec,stage):
        guard()
        if path.is_dir():
            # View root is scanned from evidence root to preserve exact link rules.
            if path.name=='view':
                all_rows=tree(Path(spec['evidence_root']),guard,True,spec)
                rows={name:row for name,row in all_rows.items()if Path(name).parent==path}
            else:rows=tree(path,guard)
            p.require(not set(files)&set(rows),'Unique phase output paths');files.update(rows)
        else:files[str(path)]=digest(path,guard)
    result=dict(schema='nemotron_phase_raw_manifest_v3',phase=stage,producer=p.producer(owner),lease=lease,
        execution_spec=p.read(Path(spec['control_root'])/(stage+'-lease.json'))['execution_spec'],
        source_sha256=spec['source_sha256'],files=files,complete=True)
    path=Path(spec['control_root'])/(stage+'-raw-manifest.json');p.once(path,result);return p.ref(path)


def verify_rows(rows,guard,spec=None):
    for path,expected in rows.items():
        guard();p.require(expected['path']==path,'Manifest path binding')
        if expected.get('compiler_asset')is True:
            p.require(type(spec)is dict,'Compiler manifest requires exact work-root context')
            assets=p.metadata.load(spec['compiler_asset_policy']);p.compiler_links.verify_assets(assets,hash_content=True)
            actual=p.compiler_links.link_record(Path(path),Path(spec['work_root'])/'compile-work',assets,hash_content=True,guard=guard)
        elif 'symlink'in expected:
            q=Path(path);s=q.lstat();actual=dict(path=path,symlink=os.readlink(q),
                stat=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
        elif expected.get('unix_socket')is True:
            p.require(type(spec)is dict,'Socket manifest requires exact work-root context')
            actual=ipc_metadata(path,Path(spec['work_root']),spec,guard)
        else:actual=digest(path,guard)
        p.require(actual==expected,'Raw bytes/identity changed: '+path)


def verify_phase(spec,stage,result,owner,lease,guard,hash_files=True):
    manifest=p.read(result['raw_manifest']['path'],result['raw_manifest']['sha256'])
    p.require(result['raw_manifest']['path']==str(Path(spec['control_root'])/(stage+'-raw-manifest.json')),'Exact phase manifest path')
    p.require(manifest['complete']is True and manifest['phase']==stage and manifest['producer']==p.producer(owner)
        and manifest['lease']==lease and manifest['source_sha256']==spec['source_sha256'],'Actual producing worker/source/lease')
    p.require(manifest['execution_spec']==p.read(lease['path'],lease['sha256'])['execution_spec'],'Manifest execution-spec binding')
    expected=set()
    for path in phase_paths(spec,stage):
        if path.is_dir():
            for base,dirs,files in os.walk(path,followlinks=False):
                guard();p.require(not any((Path(base)/d).is_symlink()for d in dirs),'No phase directory symlink')
                expected.update(str(Path(base)/f)for f in files)
        else:expected.add(str(path))
    p.require(set(manifest['files'])==expected,'Exact required phase raw file set')
    crosscheck_report_hashes(spec,stage,result,manifest)
    if hash_files:verify_rows(manifest['files'],guard,spec)
    return manifest


def snapshot_inputs(spec,guard):
    rows={}
    for path,row in spec['payloads'].items():
        actual=digest(path,guard);p.require(actual['stat']==row['stat']and actual['sha256']==row['sha256'],'Original payload changed')
        rows[path]=actual
    return rows


def crosscheck_report_hashes(spec,stage,result,manifest):
    """Manifest SHA must agree with bytes already consumed by science/package."""
    files=manifest['files'];k=p.kind(stage)
    def bound(path,digest):
        p.require(path in files and files[path].get('sha256')==digest,'Manifest matches scientific consumed artifact bytes')
    if k in ('alias','split'):
        if result.get('artifact'):bound(result['artifact'],result['artifact_sha256'])
        for row in result.get('raw_failure_payloads',[]):bound(row['path'],row['sha256'])
        if result.get('raw_failure'):bound(result['raw_failure']['path'],result['raw_failure']['sha256'])
    elif k=='compile':
        ref=result['artifact_manifest'];bound(ref['path'],ref['sha256'])
        artifact=p.read(ref['path'],ref['sha256']);root=Path(spec['work_root'])/'artifact'
        expected={str(root/name):digest for name,digest in artifact['files'].items()}
        p.require(set(expected)|{ref['path']}==set(files)-{str(Path(spec['evidence_root'])/'compile-smoke.json')},
            'Exact package compiled artifact manifest closure')
        for path,digest in expected.items():bound(path,digest)
    return True


def final_audit(spec,admission,guard):
    """Only external post-drain audit calls this; no tensor/native import."""
    c=Path(spec['control_root']);e=Path(spec['evidence_root']);stage='verify-raw'
    record=p.read(c/(stage+'-worker-result.json'));report=p.read(record['result']['report']['path'],record['result']['report']['sha256'])
    p.require(report['complete']is report['passed']is True,'Independent raw replay completed')
    manifest=p.read(report['final_manifest']['path'],report['final_manifest']['sha256'])
    p.require(manifest['execution_spec']==admission['execution_spec']and manifest['source_review']==admission['source_review']
        and manifest['source_sha256']==spec['source_sha256'],'Final manifest source/spec/review closure')
    p.require(manifest['producer']==record['producer']and manifest['lease']==record['lease'],'Final manifest actual verifier')
    for old in p.SCIENTIFIC_STAGES:
        worker=p.read(c/(old+'-worker-result.json'));owner=p.read(c/(old+'-owner.json'))
        verify_phase(spec,old,worker['result'],owner,worker['lease'],guard)
    excluded=[str(e/'raw-verification.json'),str(e/'final-raw-manifest.json')]
    actual=dict(evidence=tree(e,guard,True,spec,exclude=excluded),work=tree(Path(spec['work_root']),guard,spec=spec),inputs=snapshot_inputs(spec,guard))
    p.require(actual==manifest['closure'],'Final raw outputs/oracles/artifacts changed after independent verifier')
    p.require(record['result']['report']==p.ref(e/'raw-verification.json')and report['final_manifest']==p.ref(e/'final-raw-manifest.json'),
        'Raw verifier report/manifest stayed immutable throughout final byte audit')
    p.require(set(report['phases'])==set(p.SCIENTIFIC_STAGES)and report['phases']==manifest['scientific_results'],
        'Final manifest binds all exact scientific worker results')
    for name,ref in report['phases'].items():p.require(ref==p.ref(c/(name+'-worker-result.json')),'Scientific result changed')
    return dict(complete=True,passed=True,raw_manifest=p.ref(e/'final-raw-manifest.json'),raw_verification=p.ref(e/'raw-verification.json'),
        payloads=len(actual['inputs']),output_files=len(actual['evidence'])+len(actual['work']),native_imported=False,
        verified_after_physical_drain=True)
