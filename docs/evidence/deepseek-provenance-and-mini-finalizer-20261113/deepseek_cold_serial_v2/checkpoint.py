"""Complete original file provenance with exact declared sharing; inert at import.

Only an admitted owning parent calls authenticate_checkpoint under its original
serial lock. Full payload hashes are computed once per physical inode. Fresh
workers authenticate that actual parent proof and recheck every immutable file,
logical alias and original/view file inventory before model imports. This proves
checkpoint/config identity, never Neuron weight-layout or graph-ABI semantics.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from .contract import digest,reference,require
from .io import read

OFFICIAL=Path('/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B')
VIEW=Path('/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B-neuron-yarn-exp20261031')
REPO='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
REVISION='6e8885a6ff5c1dc5201574c8fd700323f23c25fa'
PARAMETERS=8190735360
ATTRS=('device','inode','mode','nlink','size','mtime_ns','ctime_ns','uid','gid')
STAT_ATTRS=('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_uid','st_gid')
SMALL=8<<20

def canonical_path(value):
    require(type(value)is str and Path(value).is_absolute()and str(Path(value))==value and '..'not in Path(value).parts,
        'Exact absolute manifest path')
    return value

def stat_shape(value,kind):
    require(type(value)is dict and set(value)==set(ATTRS)and all(type(v)is int and v>=0 for v in value.values())
        and value['device']>0 and value['inode']>0 and value['nlink']>0,'Complete exact physical identity')
    require((stat.S_ISREG(value['mode'])if kind=='file'else stat.S_ISLNK(value['mode'])if kind=='symlink'else stat.S_ISDIR(value['mode'])),
        'Exact manifest entry file type')

def unique(pairs):
    value={}
    for name,item in pairs:
        require(name not in value,'No duplicate original metadata key');value[name]=item
    return value

def decode(data):return json.loads(data,object_pairs_hook=unique)

def admitted_metadata(spec,binding):
    reference(binding)
    require(spec['source_sha256'].get(binding['path'])==binding['sha256']or
        spec['input_extension'].get(binding['path'],{}).get('sha256')==binding['sha256'],
        'Manifest and original source authority must be explicit complete source/input pins')

def validate_manifest(spec,manifest,authority):
    require(type(manifest)is dict and set(manifest)=={'schema','repo','revision','official_root','view_root','authority',
        'logical_files','physical_files','directory_identities'}and manifest['schema']=='deepseek_original_checkpoint_manifest_v1',
        'Complete original checkpoint/config/tokenizer manifest')
    require(manifest['repo']==REPO and manifest['revision']==REVISION and manifest['official_root']==str(OFFICIAL)
        and manifest['view_root']==str(VIEW)==spec['original_case']['model_path'],'Exact original pinned model and loader view')
    admitted_metadata(spec,spec['original_checkpoint_manifest']);admitted_metadata(spec,manifest['authority'])
    require(type(authority)is dict and set(authority)=={'schema','repo','revision','files'}
        and authority['schema']=='deepseek_original_source_file_hashes_v1'and authority['repo']==REPO
        and authority['revision']==REVISION and type(authority['files'])is dict and authority['files'],
        'Hash-bound complete original snapshot file authority; never a boolean review')
    for name,row in authority['files'].items():
        require(type(name)is str and Path(name).name==name and name not in('.','..')
            and type(row)is dict and set(row)=={'sha256','bytes'}and type(row['bytes'])is int and row['bytes']>0,
            'Every original file has an exact name, byte length and original digest')
        reference(dict(path=str(OFFICIAL/name),sha256=row['sha256']))
    require({'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'}<=set(authority['files']),
        'Original configuration, complete index and both tokenizer files required')
    logical,physical=manifest['logical_files'],manifest['physical_files']
    require(type(logical)is dict and type(physical)is dict and logical and physical,'All logical and physical files required')
    require({Path(path).name for path in logical if Path(path).parent==OFFICIAL}==set(authority['files'])
        and all(Path(canonical_path(path)).parent in(OFFICIAL,VIEW)for path in logical),
        'Complete official files and only exact original/view logical paths')
    require(set(manifest['directory_identities'])=={str(OFFICIAL),str(VIEW)},'Both exact immutable model directory identities')
    for row in manifest['directory_identities'].values():stat_shape(row,'directory')
    owners=set();declared_links=set()
    for path,row in physical.items():
        canonical_path(path)
        require(type(row)is dict and set(row)=={'sha256','bytes','identity','hardlink_paths'},'Every unique full physical file')
        reference(dict(path=path,sha256=row['sha256']));stat_shape(row['identity'],'file')
        require(type(row['bytes'])is int and row['bytes']>0 and row['bytes']==row['identity']['size'],'Complete physical extent')
        inode=(row['identity']['device'],row['identity']['inode']);require(inode not in owners,'Hash each unique physical inode once');owners.add(inode)
        links=row['hardlink_paths']
        require(type(links)is dict and path in links and len(links)==row['identity']['nlink'],'Every actual hardlink is explicitly accounted')
        for alias,expected in links.items():
            canonical_path(alias)
            require(alias not in declared_links and expected==row['identity'],'Exact unique hardlink alias shares the complete inode identity')
            declared_links.add(alias)
    used=set()
    for path,row in logical.items():
        require(type(row)is dict and set(row)=={'kind','physical','identity','link_target'}
            and row['kind']in('regular','hardlink','symlink')and row['physical']in physical,'Exact logical-to-physical relation')
        target=physical[row['physical']];used.add(row['physical'])
        if row['kind']=='symlink':
            stat_shape(row['identity'],'symlink')
            require(row['identity']['nlink']==1 and type(row['link_target'])is str and row['link_target']
                and '\x00'not in row['link_target'],'Exact singleton declared symlink')
            resolved=os.path.normpath(os.path.join(str(Path(path).parent),row['link_target']))
            require(resolved in target['hardlink_paths'],'One exact declared symlink edge to an authenticated original physical alias')
        else:
            require(row['link_target']is None and path in target['hardlink_paths']and row['identity']==target['identity']
                and (row['kind']=='regular')==(path==row['physical']),'Exact ordinary file or declared same-inode hardlink')
        if Path(path).parent==OFFICIAL:
            expected=authority['files'][Path(path).name]
            require((target['sha256'],target['bytes'])==(expected['sha256'],expected['bytes']),
                'Every physical original file binds its original complete content digest')
    require(used==set(physical),'No unrelated physical input admitted')
    return manifest

def validate_metadata(manifest,contents):
    """Pure exact index coverage and original-to-view configuration check."""
    logical,physical=manifest['logical_files'],manifest['physical_files']
    def value(root,name):
        path=str(root/name);row=physical[logical[path]['physical']];data=contents[path]
        require(type(data)is bytes and len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],
            'Exact authenticated complete original metadata bytes')
        return decode(data)
    original=value(OFFICIAL,'config.json');view=value(VIEW,'config.json')
    require(view==dict(original,model_type='deepseek_qwen3_neuron_yarn',architectures=['DeepSeekQwen3NeuronYarnForCausalLM']),
        'Only the original frozen model registration identity changes in the Neuron view')
    index=value(OFFICIAL,'model.safetensors.index.json')
    require(index==value(VIEW,'model.safetensors.index.json')and type(index['weight_map'])is dict and index['weight_map']
        and index['metadata']['total_size']==PARAMETERS*2,'Complete unchanged original BF16 index and all parameter bytes')
    shards=set(index['weight_map'].values())
    require(all(type(key)is str and key and type(shard)is str and Path(shard).name==shard and shard.endswith('.safetensors')
        for key,shard in index['weight_map'].items()),'Every full original tensor maps to a declared shard')
    require({Path(path).name for path in logical if Path(path).parent==OFFICIAL and path.endswith('.safetensors')}==shards,
        'Every original checkpoint shard is covered, with no extra weight file')
    view_names={Path(path).name for path in logical if Path(path).parent==VIEW}
    require(view_names==shards|{'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'},
        'Exact frozen view file set including all weights and tokenizer inputs')
    for name in shards|{'model.safetensors.index.json','tokenizer.json','tokenizer_config.json'}:
        first=physical[logical[str(OFFICIAL/name)]['physical']];second=physical[logical[str(VIEW/name)]['physical']]
        require((first['sha256'],first['bytes'])==(second['sha256'],second['bytes']),'Original/view shared content cannot change')
    return dict(shards=sorted(shards),original_parameter_bytes=PARAMETERS*2,registration_only_view=True)

def stream_hash(stream,expected_bytes,guard):
    digest=hashlib.sha256();count=0;guard()
    while True:
        guard();block=stream.read(min(1<<20,expected_bytes+1-count))
        if not block:break
        count+=len(block);require(count<=expected_bytes,'Physical file exceeded admitted complete extent');digest.update(block)
    require(count==expected_bytes,'Complete original file bytes required; no prefix/truncation')
    guard();return dict(bytes=count,sha256=digest.hexdigest())

class Files:
    def identity(self,path):
        path=Path(path)
        require(path.is_absolute()and path.parent.resolve()==path.parent and not any(p.is_symlink()for p in path.parents),
            'No undeclared link in original file ancestry')
        info=path.lstat()
        return dict(zip(ATTRS,(getattr(info,key)for key in STAT_ATTRS)))
    def link(self,path):return os.readlink(path)
    def inventory(self,root):
        result=[]
        for path in Path(root).iterdir():
            info=path.lstat()
            if path.name=='.cache'and Path(root)==OFFICIAL:
                require(stat.S_ISDIR(info.st_mode),'Only the ordinary original HF metadata-cache directory is outside inference files');continue
            require(stat.S_ISREG(info.st_mode)or stat.S_ISLNK(info.st_mode),'Unexpected original model subdirectory/special file')
            result.append(str(path))
        return sorted(result)
    def hash(self,path,expected,guard):
        require(self.identity(path)==expected['identity'],'Physical identity before full read')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            info=os.fstat(fd)
            require(dict(zip(ATTRS,(getattr(info,key)for key in STAT_ATTRS)))==expected['identity'],'Exact nofollow opened physical inode')
            with os.fdopen(fd,'rb',closefd=False)as stream:result=stream_hash(stream,expected['bytes'],guard)
            require(self.identity(path)==expected['identity'],'Physical input unchanged after entire byte read')
            return result
        finally:os.close(fd)
    def small(self,path,expected):
        require(expected['bytes']<=SMALL and self.identity(path)==expected['identity'],'Bounded already-authenticated metadata input')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            info=os.fstat(fd);require(dict(zip(ATTRS,(getattr(info,key)for key in STAT_ATTRS)))==expected['identity'],'Exact opened metadata identity')
            with os.fdopen(fd,'rb',closefd=False)as stream:data=stream.read(expected['bytes']+1)
            require(len(data)==expected['bytes']and self.identity(path)==expected['identity'],'Stable complete metadata bytes');return data
        finally:os.close(fd)

class Checkpoint:
    def __init__(self,manifest,proof,files=None):self.manifest,self.proof,self.files=manifest,proof,files or Files()
    def check(self):
        m=self.manifest
        for root,wanted in m['directory_identities'].items():
            require(self.files.identity(root)==wanted,'Immutable original model directory changed')
            require(self.files.inventory(root)==sorted(path for path in m['logical_files']if str(Path(path).parent)==root),
                'Complete current original/view file inventory changed')
        for row in m['physical_files'].values():
            for path,wanted in row['hardlink_paths'].items():
                require(self.files.identity(path)==wanted,'An original physical file or explicitly shared hardlink changed')
        for path,row in m['logical_files'].items():
            require(self.files.identity(path)==row['identity'],'Original logical file or declared symlink changed')
            if row['kind']=='symlink':require(self.files.link(path)==row['link_target'],'Original symlink target changed')
        return True

def load_manifest(spec,guard):
    guard();admitted_metadata(spec,spec['original_checkpoint_manifest']);manifest=read(spec['original_checkpoint_manifest']);guard()
    admitted_metadata(spec,manifest['authority']);authority=read(manifest['authority']);guard()
    return validate_manifest(spec,manifest,authority)

def proof_value(spec,spec_ref,owner,manifest,hashes,metadata):
    return dict(schema='deepseek_original_checkpoint_actual_bytes_v1',spec=copy.deepcopy(spec_ref),owner=copy.deepcopy(owner),
        manifest=copy.deepcopy(spec['original_checkpoint_manifest']),authority=copy.deepcopy(manifest['authority']),
        unique_physical_files=copy.deepcopy(hashes),total_unique_bytes=sum(row['bytes']for row in hashes.values()),
        logical_manifest_sha256=digest(manifest['logical_files']),metadata=metadata,
        all_file_bytes_authenticated=True,loaded_weight_layout_proven=False,graph_weight_ABI_proven=False,
        full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)

def authenticate_checkpoint(spec,spec_ref,owner,guard,*,files=None):
    require(callable(guard),'Actual owning retained-lease guard required');manifest=load_manifest(spec,guard)
    check=Checkpoint(manifest,None,files);check.check();hashes={}
    for path,row in manifest['physical_files'].items():
        guard();actual=check.files.hash(path,row,guard)
        require(actual==dict(bytes=row['bytes'],sha256=row['sha256']),'Complete original physical bytes disagree with source authority')
        hashes[path]=dict(actual,identity=copy.deepcopy(row['identity']));check.check()
    contents={}
    for root in(OFFICIAL,VIEW):
        for name in('config.json','model.safetensors.index.json'):
            path=str(root/name);physical=manifest['logical_files'][path]['physical'];row=manifest['physical_files'][physical]
            guard();contents[path]=check.files.small(physical,row)
    metadata=validate_metadata(manifest,contents);check.check();guard()
    check.proof=proof_value(spec,spec_ref,owner,manifest,hashes,metadata);return check

def load_bound_proof(spec,spec_ref,owner,proof_ref,guard):
    require(proof_ref['path']==str(Path(spec['paths']['control'])/'original-checkpoint-provenance.json'),
        'Exact owning-parent full byte proof output')
    manifest=load_manifest(spec,guard);guard();proof=read(proof_ref);guard()
    hashes={path:dict(bytes=row['bytes'],sha256=row['sha256'],identity=copy.deepcopy(row['identity']))for path,row in manifest['physical_files'].items()}
    check=Checkpoint(manifest,proof);contents={}
    for root in(OFFICIAL,VIEW):
        for name in('config.json','model.safetensors.index.json'):
            path=str(root/name);physical=manifest['logical_files'][path]['physical'];guard()
            contents[path]=check.files.small(physical,manifest['physical_files'][physical])
    metadata=validate_metadata(manifest,contents)
    require(proof==proof_value(spec,spec_ref,owner,manifest,hashes,metadata),
        'Complete authenticated actual-parent byte proof; boolean or partial weight proof rejected')
    check.check();guard();return check
