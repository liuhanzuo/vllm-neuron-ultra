"""Concrete original native API/tokenizer resolver, inert on import.

Future admitted worker only: resolution imports native libraries and hashes an
explicit independently authenticated environment inventory supplied by the
coordinator. This module neither creates that inventory nor authorizes native
work. Versions/paths and old selected source pins are NOT a full environment
trust anchor. No model/forward/checkpoint load, spec, admission or launch here.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace

from dory_stage_a_archive_core_v1 import require
from dory_stage_a_saved_files_v1 import identity
from dory_stage_a_capture_core_v1 import sha
from dory_stage_a_artifact_transaction_v2 import digest
from prepare_dory_stage_a_source_inputs_v1 import (Q, DESIGN, read_pinned_json,
    prepare_source_inputs, validate_source_inputs)

BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
ENV = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
SITE = ENV/'lib/python3.12/site-packages'
OFFICIAL = BASE/'vllm'
META = Q/'dory-checkpoint-metadata-20261108-v1'
IMPORTS = dict(path=str(BASE/'qualify-r2/qualification/imports.json'),
    sha256='8daae2c286aeabd38dbcfc8c614c4d4039a124b2cb5ad573ae85f8be26ec50ea',bytes=2470)
UPSTREAM_NAME = '_dory_stage_a_pinned_upstream_v1'
SOURCE_RELATIVE = ('model_executor/models/dory.py','transformers_utils/configs/dory.py',
    'model_executor/layers/linear.py','model_executor/layers/vocab_parallel_embedding.py',
    'model_executor/layers/utils.py','model_executor/layers/rotary_embedding/base.py')
TOKEN_NAMES = ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
               'chat_template.jinja','generation_config.json')
MODULES = ('torch','pytest','transformers','tokenizers','vllm','vllm.config',
    'vllm.distributed.parallel_state','vllm.model_executor.models.dory',
    'vllm.model_executor.models.utils','vllm.transformers_utils.configs.dory','vllm._custom_ops')
ENVIRONMENT = dict(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    XDG_CACHE_HOME=str(BASE/'cache'),HF_HOME=str(BASE/'cache/huggingface'),TMPDIR=str(BASE/'tmp'),
    TRITON_CACHE_DIR=str(BASE/'cache/triton'),TORCH_EXTENSIONS_DIR=str(BASE/'cache/torch'),
    LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:'+str(ENV/'lib/libiomp5.so'),
    VLLM_TARGET_DEVICE='cpu',VLLM_PLUGINS='',VLLM_LOGGING_LEVEL='INFO',VLLM_CPU_KVCACHE_SPACE='1',
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def stable_file(ref, *, cap, guard, retain=False):
    """Future native inventory bytes or bounded original source/metadata only.

    Hashes sequential64KiB blocks; owned canonical single-link regular file.
    Explicitly pinned empty package sources are valid: the complete empty SHA256,
    EOF and stable ordinary-file identity are still checked.
    No allocation of an entire binary. retain is reserved for small source/JSON.
    Not called on native files by this source-only task or its fixtures.
    """
    require(type(ref) is dict and set(ref) == {'path','sha256','bytes'}
            and type(cap) is int and cap > 0 and type(ref['bytes']) is int
            and 0 <= ref['bytes'] <= cap and callable(guard), 'Explicit bounded file pin')
    sha(ref['sha256']);path=Path(ref['path']);guard()
    require(path.is_absolute() and path.resolve() == path, 'Canonical pinned file without symlink ancestors')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
                and before.st_size == ref['bytes'], 'Exact owned complete pinned file')
        value=hashlib.sha256();count=0;pieces=[]
        while count < before.st_size:
            guard();part=os.read(fd,min(65536,before.st_size-count))
            require(bool(part), 'No truncated authenticated file');count+=len(part);value.update(part)
            if retain:pieces.append(part)
        require(os.read(fd,1) == b'' and value.hexdigest() == ref['sha256'], 'Actual complete pinned SHA256')
        guard()
        require(identity(os.fstat(fd)) == identity(before)
                and identity(path.stat(follow_symlinks=False)) == identity(before) and path.resolve() == path,
                'Stable authenticated bytes and filename')
        return dict(path=str(path),sha256=value.hexdigest(),bytes=count,identity=identity(before)), b''.join(pieces) if retain else None
    finally:os.close(fd)


def original_context(*, guard):
    """Source/JSON evidence only, no native imports/inventory hashes or writes."""
    design=read_pinned_json(DESIGN,guard=guard)
    prior_ref=design['prior_evidence']['previous_runtime_spec']
    prior=read_pinned_json(prior_ref,guard=guard)
    require((prior['environment'],prior['official_source'],prior['tokenizer_directory']) ==
            (str(ENV),str(OFFICIAL),str(META)), 'Original CPU environment/source/tokenizer roots')
    pins=prior['source_sha256']
    imports=read_pinned_json(IMPORTS,guard=guard)
    token_native_path=str(BASE/'residual-replay-r1/diagnostic/tokenizer.json')
    token_native_ref=dict(path=token_native_path,sha256=pins[token_native_path])
    token_native=read_pinned_json(token_native_ref,guard=guard)
    versions={name:imports['versions'][name] for name in ('torch','vllm','transformers','pytest')}
    versions['tokenizers']=token_native['versions']['tokenizers']
    require(versions == dict(torch='2.14.0+cpu',vllm='0.1.dev1+g75fc05a53.cpu',
        transformers='5.20.2',pytest='9.0.2',tokenizers='0.23.3')
        and imports['torch_file'] == str(SITE/'torch/__init__.py')
        and imports['vllm_file'] == str(SITE/'vllm/__init__.py')
        and imports['ld_preload'] == ENVIRONMENT['LD_PRELOAD'], 'Original actual imported versions and environment')
    review_path=str(META/'metadata-review.json')
    review=read_pinned_json(dict(path=review_path,sha256=pins[review_path]),guard=guard)
    token_refs={name:deepcopy(review['input_files'][name]) for name in TOKEN_NAMES}
    require(all(ref['sha256'] == pins[ref['path']] and Path(ref['path']) == META/name
                for name,ref in token_refs.items()), 'Every original tokenizer/template file pin')
    captures_ref=dict(path=prior['previous_token_captures'],sha256=pins[prior['previous_token_captures']])
    captures=read_pinned_json(captures_ref,guard=guard)
    selection=read_pinned_json(design['prior_evidence']['selection_plan'],guard=guard)
    fixture_ref=selection['original_prefix_plan']['fixtures']
    full_fixtures=read_pinned_json(fixture_ref,guard=guard)
    require(full_fixtures['fixtures'] == selection['original_prefix_plan']['token_fixtures'],
            'Original two full2060-token input fixtures')
    fixture_source=str(Q/'run_dory_tokenizer_template_v1.py')
    fixture_pin=dict(path=fixture_source,sha256=pins[fixture_source],bytes=Path(fixture_source).stat().st_size)
    _,source=stable_file(fixture_pin,cap=65536,guard=guard,retain=True)
    tree=ast.parse(source,filename=fixture_source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixtures']
    require(len(nodes)==1 and not nodes[0].decorator_list and not nodes[0].args.args,
            'Only original pure fixture function; never import old runner')
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),fixture_source,'exec',dont_inherit=True),namespace)
    fixtures=namespace['fixtures']()
    require(len(fixtures)==len(captures['cases'])==13 and len(captures['specials'])==1000
            and token_native['special_ids_checked']==1000
            and [x['name'] for x in fixtures] == [x['name'] for x in captures['cases']],
            'Original13 template fixtures and1000 special symbols')
    source_pins={str(SITE/'vllm'/rel):pins[str(SITE/'vllm'/rel)] for rel in SOURCE_RELATIVE}
    upstream=str(OFFICIAL/'tests/models/test_dory.py')
    source_pins[upstream]=pins[upstream]
    return dict(schema='dory_stage_a_native_source_context_v1',prior_runtime=deepcopy(prior_ref),
        original_import_evidence=deepcopy(IMPORTS),original_native_tokenizer_evidence=token_native_ref,
        versions=versions,environment=dict(ENVIRONMENT),source_pins=source_pins,token_refs=token_refs,
        captures=captures,fixtures=fixtures,full_fixtures=full_fixtures,
        fixture_source=fixture_pin,captures_ref=captures_ref,full_fixtures_ref=fixture_ref,
        native_inventory_anchor_present_in_prior_runtime=False,runtime_admitted=False,
        singleton_native_adapter_ABI_verified=False,new_speedup_claim=False)


class NativeFileAuthority:
    """Actual future file validation, not a trust anchor or environment builder.

    files must be an independently reviewed, authenticated inventory covering
    installed SITE files and original OFFICIAL sources, supplied by the coordinator.
    Interpreter, ENV/lib preloads and root-owned/system shared libraries are
    deliberately outside this owned-file scope (including symlinked tcmalloc).
    Caller separately owns interpreter/loader/system-lib
    trust and a fresh import process. This class checks actual bytes and imported
    origins; it cannot authenticate a caller's invented/rehashed inventory.
    Limits are explicit inputs, not reserved memory/storage or runtime admission.
    """
    def __init__(self,files,*,byte_limit,guard):
        require(isinstance(files,list) and 1<=len(files)<=100000 and callable(guard)
                and type(byte_limit)is int and 0<byte_limit<=8*1024**3, 'Explicit reviewed inventory and byte limit')
        self.files={};self.verified={};self.guard=guard;self.active=False
        total=0
        for ref in files:
            require(type(ref)is dict and set(ref)=={'path','sha256','bytes'}
                    and type(ref['bytes'])is int and 0<=ref['bytes']<=2*1024**3
                    and Path(ref['path']).is_absolute() and ref['path'] not in self.files
                    and any(Path(ref['path']).is_relative_to(root) for root in (SITE,OFFICIAL)),
                    'Unique bounded original runtime file pin')
            sha(ref['sha256']);self.files[ref['path']]=deepcopy(ref);total+=ref['bytes']
        require(total<=byte_limit,'Complete native inventory inside explicit caller limit')
        self.bytes=total

    def verify_all(self,context):
        require(not self.verified,'Fresh single-use inventory validation')
        for name,wanted in context['source_pins'].items():
            require(name in self.files and self.files[name]['sha256']==wanted,'Old original source pins cannot be rebound')
        for name,ref in self.files.items():
            result,_=stable_file(ref,cap=2*1024**3,guard=self.guard)
            self.verified[name]=result
        self.verify_stable()

    def verify_stable(self):
        self.guard()
        require(set(self.verified)==set(self.files),'All explicit native files actually verified')
        for name,row in self.verified.items():
            p=Path(name)
            require(p.resolve()==p and identity(p.stat(follow_symlinks=False))==row['identity'],
                    'All verified native files retain original identity')

    def origin(self,path):
        require(type(path)is str and Path(path).is_absolute() and str(Path(path))==path
                and path in self.verified and Path(path).resolve()==Path(path),
                'Imported module/callable origin was authenticated before import')
        require(identity(Path(path).stat(follow_symlinks=False))==self.verified[path]['identity'],
                'Imported source retains verified identity')
        return dict(path=path,sha256=self.files[path]['sha256'])

    def audit(self,event,args):
        if not self.active:return
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            # Never accept stale/unpinned bytecode instead of verified source.
            if name.endswith('.pyc'):raise OSError('Stage A requires verified source instead of cached bytecode')
            p=Path(name)
            if p.is_absolute() and p.is_relative_to(SITE) and (p.suffix=='.py' or '.so' in p.name):
                require(name in self.verified,'Installed code read outside authenticated inventory')
        if event=='import' and len(args)>1 and args[1]:
            name=str(args[1])
            if str(SITE) in name:require(name in self.verified,'Native import outside authenticated inventory')


def _check_process(context):
    require(sys.executable==str(ENV/'bin/python') and sys.prefix==str(ENV)
            and sys.flags.optimize==0 and sys.dont_write_bytecode and sys.flags.no_user_site,
            'Fresh original CPU interpreter with assertions/no bytecode/no user site')
    require(all(os.environ.get(k)==v for k,v in context['environment'].items())
            and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME'),
            'Exact original CPU/offline/thread/preload environment without path overrides')
    require(not any(name.split('.')[0] in ('torch','numpy','vllm','transformers','tokenizers','pytest','_pytest')
                    for name in sys.modules), 'Authenticate before any native/test package import')
    require(UPSTREAM_NAME not in sys.modules,'No prior upstream module in this fresh worker')


def normalize_ids(ids):
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):
        require(len(ids)==1,'One original conversation');ids=ids[0]
    require(type(ids)is list and 0<len(ids)<=8192 and all(type(x)is int and 0<=x<131072 for x in ids),
            'Complete original bounded vocabulary IDs')
    return ids


def check_tokenizer(tokenizer,raw,context,prepared,template,policy,*,guard):
    """Actual tokenizer operations in future runtime, shared-backend controls.

    Bounded fake fixtures test rejection logic only; no tokenizer parity or
    native readiness follows from those tests. No model/tensor operations.
    """
    guard()
    require(tokenizer.chat_template==template and (len(tokenizer),tokenizer.bos_token_id,
            tokenizer.eos_token_id,tokenizer.pad_token_id)==(131072,1,11,11),'Exact original template/vocabulary/special IDs')
    rows=[]
    for fixture,old in zip(context['fixtures'],context['captures']['cases']):
        guard();rendered=tokenizer.apply_chat_template(fixture['messages'],tokenize=False,**fixture['kwargs'])
        require(type(rendered)is str and len(rendered)<=65536 and rendered==old['rendered'],
                'Every original template byte sequence')
        if fixture['expected'] is not None:require(rendered==fixture['expected'],'Original manually specified prompt')
        else:
            require(all(x in rendered for x in fixture['required_fragments']) and rendered.endswith(fixture['required_suffix']),
                    'Original tool schema fragments and suffix')
        ids=normalize_ids(tokenizer.apply_chat_template(fixture['messages'],tokenize=True,**fixture['kwargs']))
        require(ids==old['input_ids']==normalize_ids(raw.encode(rendered,add_special_tokens=False).ids)
                and normalize_ids(tokenizer.encode(rendered,add_special_tokens=True))==ids
                and tokenizer.decode(ids,skip_special_tokens=False)==rendered,'Exact original template IDs and text roundtrip')
        rows.append(dict(name=fixture['name'],tokens=len(ids),rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
    for token in context['captures']['specials']:
        guard()
        require(tokenizer.encode(token['content'],add_special_tokens=False)==[token['id']]
                and tokenizer.decode([token['id']],skip_special_tokens=False)==token['content'],
                'Every original1000 special token ID and decode')
    require(policy==context['captures']['generation_policy']
            and (policy['eos_token_id'],policy['do_sample'],policy['temperature'],policy['top_p'],policy['top_k'])
                ==([2,11],True,1.0,.95,0)
            and [tokenizer.convert_ids_to_tokens(x) for x in policy['eos_token_id']]==['</s>','<|im_end|>'],
            'Original complete generation policy and both stop symbols')
    full=context['full_fixtures'];protocol=full['protocol'];regenerated=[]
    require(protocol['LENGTH']==2060 and len(full['fixtures'])==len(protocol['prefixes'])==len(protocol['paragraphs'])==2,
            'Original two2060-token fixtures')
    for i,(prompt,paragraph,old) in enumerate(zip(protocol['prefixes'],protocol['paragraphs'],full['fixtures'])):
        guard();prefix=normalize_ids(tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True))
        body=normalize_ids(tokenizer.encode(paragraph,add_special_tokens=False))
        sequence=(prefix+body*((2060//len(body))+1))[:2060]
        require(old==dict(id=i,input_ids=sequence,prefix_ids=prefix,body_ids=body,
                    template_kwargs=dict(tokenize=True,add_generation_prompt=True),default_thinking_retained=True),
                'Regenerate original prompt/body composition without rewriting input')
        regenerated.append(sequence)
    for case in prepared['cases']:
        require(case['input_ids']==regenerated[case['identity']['fixture']][:case['identity']['end']],
                'Every Stage A original token belongs to authenticated actual tokenizer fixture')
    return dict(complete=True,template_cases=rows,special_symbols=1000,full_token_fixtures=2,
        stage_a_cases_checked=6,generation_policy_preserved=True,shared_backend_wrapper_consistency=True,
        independent_tokenizer_implementation_parity=False,model_forward_executed=False,
        full_model_qualified=False,new_speedup_claim=False)


def callable_origin(name,value,authority):
    """Bind the real contextmanager wrapper without generic unchecked unwrap.

    vLLM set_current_vllm_config is decorated with stdlib contextmanager. Its
    helper code lives in contextlib.py, while the actual generator source is
    the authenticated vllm/config/vllm.py. The stdlib trust remains external.
    """
    if name != 'set_current_vllm_config':
        return authority.origin(inspect.getsourcefile(value))
    reference=contextmanager(lambda:None)
    wrapped=getattr(value,'__wrapped__',None)
    require(inspect.isfunction(value) and value.__code__ is reference.__code__
            and value.__closure__ is not None and len(value.__closure__)==1
            and value.__closure__[0].cell_contents is wrapped
            and inspect.isgeneratorfunction(wrapped) and wrapped.__name__=='set_current_vllm_config',
            'Exact original stdlib contextmanager code and generator closure')
    origin=authority.origin(inspect.getsourcefile(wrapped))
    require(origin['path']==str(SITE/'vllm/config/vllm.py'), 'Original vLLM context generator source')
    return dict(origin,wrapper='stdlib.contextlib.contextmanager',stdlib_identity_authenticated_here=False)


def _module_origins(authority):
    result=[]
    for name,module in tuple(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not path:continue
        p=Path(path)
        # Reject foreign third-party shadows for every actual module, including
        # transitive imports. stdlib/system interpreter trust is external.
        if p.is_absolute() and (p.is_relative_to(SITE) or name.split('.')[0] in
                ('torch','numpy','vllm','transformers','tokenizers','pytest','_pytest','huggingface_hub')):
            result.append(dict(module=name,**authority.origin(str(p))))
    return result


@contextmanager
def resolve_native_api(prepared,files,*,byte_limit,guard):
    """Yield concrete worker API + evidence inside an authenticated import scope.

    Future caller must authenticate files independently BEFORE invoking this
    function, own original interpreter/system loader trust and fresh admission.
    Missing inventory rejects before imports; an arbitrary rehashed file list
    is not authentication. This function does not construct the model or run
    the worker; adapter ABI/singleton probes and actual-exit coordinator remain
    separate mandatory gates. Use with resolve_native_api(...) as (api,evidence)
    around run_worker so lazy imports remain covered. Normal context exit checks
    newly imported origins and original file identities; it is NOT process exit.
    Failure requires this fresh worker to terminate;
    imported native libraries cannot be unloaded and retried safely here.
    """
    require(callable(guard),'Explicit currently admitted native guard');guard()
    prepared=deepcopy(validate_source_inputs(prepared))
    require(prepared==prepare_source_inputs(guard=guard),'Reauthenticate original source inputs from fixed design')
    context=original_context(guard=guard)
    _check_process(context)
    authority=NativeFileAuthority(files,byte_limit=byte_limit,guard=guard)
    authority.verify_all(context)
    token_bytes={};token_evidence={}
    for name,ref in context['token_refs'].items():
        # tokenizer.json is original17,077,484B metadata, streamed with no retain.
        cap=20*1024**2 if name=='tokenizer.json' else 256*1024
        retain=name in ('chat_template.jinja','generation_config.json')
        token_evidence[name],data=stable_file(ref,cap=cap,guard=guard,retain=retain)
        if retain:token_bytes[name]=data
    upstream_path=str(OFFICIAL/'tests/models/test_dory.py')
    upstream_ref=authority.files[upstream_path]
    _,upstream_bytes=stable_file(upstream_ref,cap=65536,guard=guard,retain=True)
    sys.addaudithook(authority.audit);authority.active=True
    try:
        guard();modules={name:importlib.import_module(name) for name in MODULES}
        for name,version in context['versions'].items():
            require(str(modules[name].__version__)==version,'Original actual native package version: '+name)
        for name in ('torch','pytest','transformers','tokenizers','vllm'):
            require(Path(modules[name].__file__)==SITE/name/'__init__.py','Exact original installed package root: '+name)
        require(Path(modules['vllm.model_executor.models.dory'].__file__)==SITE/'vllm/model_executor/models/dory.py',
                'Actual original installed official Dory module')
        upstream=ModuleType(UPSTREAM_NAME);upstream.__file__=upstream_path;sys.modules[UPSTREAM_NAME]=upstream
        exec(compile(upstream_bytes,upstream_path,'exec',dont_inherit=True),upstream.__dict__)
        cfg=modules['vllm.config']
        api=SimpleNamespace(torch=modules['torch'],monkeypatch_factory=modules['pytest'].MonkeyPatch,
            upstream=upstream,official_source=modules['vllm.model_executor.models.dory'],ops=modules['vllm._custom_ops'],
            parallel_state=modules['vllm.distributed.parallel_state'],VllmConfig=cfg.VllmConfig,
            DeviceConfig=cfg.DeviceConfig,CompilationConfig=cfg.CompilationConfig,
            set_current_vllm_config=cfg.set_current_vllm_config,
            DoryConfig=modules['vllm.transformers_utils.configs.dory'].DoryConfig,
            extract_layer_index=modules['vllm.model_executor.models.utils'].extract_layer_index)
        callables={'monkeypatch_factory':api.monkeypatch_factory,'ReferenceAttention':upstream.ReferenceAttention,
            '_prepare_for_inference':upstream._prepare_for_inference,'DoryForCausalLM':api.official_source.DoryForCausalLM,
            'onednn_mm':api.ops.onednn_mm,'VllmConfig':api.VllmConfig,'DeviceConfig':api.DeviceConfig,
            'CompilationConfig':api.CompilationConfig,'set_current_vllm_config':api.set_current_vllm_config,
            'DoryConfig':api.DoryConfig,'extract_layer_index':api.extract_layer_index,
            'PreTrainedTokenizerFast':modules['transformers'].PreTrainedTokenizerFast}
        origins={}
        for name,value in callables.items():
            require(callable(value),'Concrete original native API callable: '+name)
            origins[name]=callable_origin(name,value,authority)
        require(origins['ReferenceAttention']['path']==origins['_prepare_for_inference']['path']==upstream_path,
                'Original upstream reference implementation bytes')
        tokenizer=modules['transformers'].PreTrainedTokenizerFast.from_pretrained(str(META),local_files_only=True)
        raw=modules['tokenizers'].Tokenizer.from_file(str(META/'tokenizer.json'))
        tokenizer_review=check_tokenizer(tokenizer,raw,context,prepared,token_bytes['chat_template.jinja'].decode(),
            json.loads(token_bytes['generation_config.json']),guard=guard)
        imported=_module_origins(authority)
        authority.verify_stable();guard()
        for name,row in token_evidence.items():
            require(identity(Path(row['path']).stat(follow_symlinks=False))==row['identity'],
                    'Tokenizer/template files unchanged after actual use')
        evidence=dict(schema='dory_stage_a_native_resolution_v4',resolved=True,
            source_input_sha256=prepared['input_sha256'],versions=context['versions'],callable_origins=origins,
            imported_modules=imported,tokenizer_review=tokenizer_review,tokenizer_files=token_evidence,
            inventory_sha256=digest(files),inventory_files=len(files),inventory_bytes=authority.bytes,
            inventory_bytes_verified_here=True,inventory_trust_anchor_authenticated_here=False,
            interpreter_system_libraries_authenticated_here=False,admission_created_here=False,
            native_adapter_ABI_verified=False,singleton_native_layout_verified=False,
            model_forward_executed_by_resolver=False,actual_process_exit_verified=False,
            full_model_qualified=False,new_speedup_claim=False,
            worker_context_source_stability_verified=False)
        yield api,evidence
        # Only normal caller completion reaches these checks. Preserve a caller
        # exception rather than replacing it with a cleanup-time provenance error.
        guard()
        later_imports=_module_origins(authority)
        authority.verify_stable()
        for row in token_evidence.values():
            path=Path(row['path'])
            require(path.resolve()==path and identity(path.stat(follow_symlinks=False))==row['identity'],
                    'Tokenizer assets unchanged through worker context')
        evidence['imported_modules_after_worker_context']=later_imports
        evidence['worker_context_source_stability_verified']=True
    finally:
        authority.active=False


if __name__=='__main__':
    raise RuntimeError('Source-only resolver API; requires authenticated native inventory and fresh coordinator admission')
