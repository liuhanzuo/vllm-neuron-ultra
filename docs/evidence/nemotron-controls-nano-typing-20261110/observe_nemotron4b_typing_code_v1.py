"""Observe actual typing code definitions against its installed Python source.

Source-only: no site hooks, SDK, model or explicit native payload reads. This
inventory is not complete source/runtime authentication or worker admission.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import types
import typing

from nemotron4b_empty_bytecode_cache_v1 import EmptyBytecodeCache,record_empty_directory,typing_registry_aliases
from nemotron4b_interpreter_origin_policy_v1 import code_sha256
from nemotron4b_pinned_source_reader_v1 import source as read_source,identity
from nemotron4b_process_handshake_v1 import require

Q=Path(__file__).resolve().parent
PLAN=Q/'nemotron4b-native-prerequisite-plan-20261106-v2.json'
PLAN_SHA='7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'
FORBIDDEN={'site','torch','transformers','vllm','neuronx_distributed','libneuronxla'}


def guard():
    require(not FORBIDDEN&set(sys.modules),'Source-only observation without SDK/model/site')


def observe(phase):
    guard()
    raw=PLAN.read_bytes();require(hashlib.sha256(raw).hexdigest()==PLAN_SHA,'Original fixed phase plan')
    expected=json.loads(raw)['environments'][phase]
    require(sys.executable==expected['python']and list(sys.version_info[:2])==expected['python_version'],
            'Original actual phase interpreter')
    cache=EmptyBytecodeCache(record_empty_directory(sys.pycache_prefix,guard=guard),guard=guard)
    caches=cache.audit_preloaded_source_caches()
    owner=sys.modules['typing'];path=Path(vars(owner)['__file__'])
    # Candidate source identity observation, not independent publisher evidence.
    before=identity(path.lstat());raw=path.read_bytes()
    reference=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                   fullstat=before,original_nlink=before['st_nlink'])
    raw,reference=read_source(path,reference)
    root=compile(raw,str(path),'exec',dont_inherit=True,optimize=0)
    definitions={};stack=[root]
    while stack:
        code=stack.pop();digest=code_sha256(code)
        definitions[digest]=dict(name=code.co_name,line=code.co_firstlineno)
        stack.extend(x for x in code.co_consts if type(x)is types.CodeType)
    functions={};classes={};external=[];visited=set();todo=[]
    for key,value in vars(owner).items():todo.append((key,value,0))
    while todo:
        name,value,depth=todo.pop()
        require(depth<=16 and len(visited)<=8192,'Bounded static typing object scan')
        if id(value)in visited:continue
        visited.add(id(value))
        if type(value)is types.FunctionType:
            if value.__module__!='typing'and value.__code__.co_filename!=str(path):continue
            digest=code_sha256(value.__code__)
            functions[name]=dict(name=value.__name__,qualname=value.__qualname__,
                module=value.__module__,filename=value.__code__.co_filename,
                code_sha256=digest,definition_in_source=digest in definitions,
                globals_are_typing=value.__globals__ is vars(owner))
            if value.__closure__:
                for index,cell in enumerate(value.__closure__):
                    try:child=cell.cell_contents
                    except ValueError:continue
                    if type(child)is types.FunctionType:todo.append((name+'.<closure'+str(index)+'>',child,depth+1))
            child=vars(value).get('__wrapped__')
            if child is not None:todo.append((name+'.__wrapped__',child,depth+1))
        elif isinstance(value,type) and type.__getattribute__(value,'__module__')=='typing':
            classes[name]=dict(name=type.__getattribute__(value,'__name__'),
                qualname=type.__getattribute__(value,'__qualname__'),metaclass=type(value).__name__)
            for key,child in type.__getattribute__(value,'__dict__').items():
                if type(child)in(classmethod,staticmethod):child=child.__func__
                if type(child)is property:
                    for part in('fget','fset','fdel'):
                        f=getattr(child,part)
                        if f is not None:todo.append((name+'.'+key+'.'+part,f,depth+1))
                elif type(child)is types.FunctionType or isinstance(child,type):
                    todo.append((name+'.'+key,child,depth+1))
        elif type(value).__module__=='functools'and type(value).__name__=='_lru_cache_wrapper':
            wrapped=vars(value).get('__wrapped__')
            if type(wrapped)is types.FunctionType and wrapped.__module__=='typing':
                external.append(dict(name=name,kind='functools._lru_cache_wrapper'))
                todo.append((name+'.__wrapped__',wrapped,depth+1))
    aliases=typing_registry_aliases()
    current,_=read_source(path,reference);require(current==raw,'Original source remains unchanged')
    cache.check_runtime();guard()
    mismatches={k:v for k,v in functions.items()if not v['definition_in_source']or not v['globals_are_typing']}
    return dict(schema='nemotron4b_typing_code_source_observation_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,
        python=sys.executable,version=list(sys.version_info[:3]),original_plan_sha256=PLAN_SHA,
        observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        typing_source=reference,compiled_definitions=len(definitions),function_count=len(functions),
        functions=functions,classes=classes,external_wrappers=external,mismatches=mismatches,
        typing_namespace_aliases=sorted(aliases),cache=cache.result(),
        source_snapshot_is_candidate=True,original_site_hooks_executed=False,
        actual_SDK_or_model_executed=False,explicit_native_payload_reads=0,
        full_module_globals_authenticated=False,complete_runtime_dependency_closure_proven=False,
        runtime_admitted=False,new_speedup_claim=False)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in('native-meta','nxd-meta'),'Original phase argument')
    print(json.dumps(observe(sys.argv[1]),sort_keys=True))
