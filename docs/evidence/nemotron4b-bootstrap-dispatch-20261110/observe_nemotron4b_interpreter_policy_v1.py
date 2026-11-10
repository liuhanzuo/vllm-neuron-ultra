"""Describe actual phase interpreters without site startup or native-file reads.

Frozen Python code definitions are inspected in memory; native/interpreter
files are stat-only. The result is a manifest candidate, not authentication or
runtime admission. NativeFiles must independently authenticate its file closure
in the serial slot before InterpreterPolicy can use this candidate.
"""
import datetime
import hashlib
import importlib.machinery as machinery
import _imp
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

import nemotron4b_interpreter_origin_policy_v1 as policy
from nemotron4b_native_origin_audit_v1 import parse_maps,fullstat
from nemotron4b_process_handshake_v1 import require

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
PLAN=ROOT/'research/nemotron4b-native-prerequisite-plan-20261106-v2.json'
PLAN_SHA='7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'


def frozen_names():
    enumerator=getattr(_imp,'_frozen_module_names',None)
    if enumerator is not None:
        require(type(enumerator)is types.BuiltinFunctionType and enumerator.__module__=='_imp',
                'Actual interpreter frozen-name enumerator')
        names=list(enumerator());require(len(names)<=1024 and all(type(n)is str for n in names),'Bounded frozen names')
        return sorted(set(names)),True,'interpreter _imp._frozen_module_names'
    # Python3.10 lacks that enumerator. Preserve a disclosed candidate scan.
    base=Path(sys.base_prefix)/'lib'/('python%d.%d'%sys.version_info[:2])
    names={'_frozen_importlib','_frozen_importlib_external','zipimport','__hello__','__phello__','__phello__.spam'}
    for p in base.rglob('*.py'):
        parts=p.relative_to(base).parts
        parts=parts[:-1]if parts[-1]=='__init__.py'else(*parts[:-1],p.stem)
        if parts and all(re.fullmatch('[A-Za-z_][A-Za-z0-9_-]*',x)for x in parts):names.add('.'.join(parts))
    names.update(sys.modules)
    return sorted(name for name in names if _imp.is_frozen(name)),False,'stdlib source names, preloaded names and explicit CPython frozen names'


def describe(phase):
    raw=PLAN.read_bytes();require(hashlib.sha256(raw).hexdigest()==PLAN_SHA,'Original plan unchanged')
    plan=json.loads(raw);expected=plan['environments'][phase]
    require(sys.executable==expected['python']and list(sys.version_info[:2])==expected['python_version']
            and sys.flags.no_site==sys.flags.no_user_site==1 and sys.dont_write_bytecode,
            'Original phase Python -B -S without user site')
    require('site'not in sys.modules,'No site module body or main executed by observation')
    frozen,complete,method=frozen_names();names=sorted(set(sys.builtin_module_names)|set(frozen))
    modules={};preloaded=[];unavailable=[];aliases=[]
    for name in names:
        builtin=name in sys.builtin_module_names
        finder=machinery.BuiltinImporter if builtin else machinery.FrozenImporter
        spec=finder.find_spec(name)
        if spec is None:
            unavailable.append(dict(name=name,reason='Interpreter enumerated name has no importable spec'));continue
        require(spec.loader is finder and spec.name==name,'Actual original interpreter spec')
        module=sys.modules.get(name);values=vars(module)if type(module)is types.ModuleType else None
        if values is not None and values.get('__spec__')is not None and values['__spec__'].name==name:
            spec=values['__spec__'];preloaded.append(name)
            module_name=values['__name__'];package=values.get('__package__');file=values.get('__file__')
        else:
            module_name=name;package=name if spec.submodule_search_locations is not None else name.rpartition('.')[0]
            file=None if spec.loader_state is None else vars(spec.loader_state).get('filename')
        modules[name]=dict(spec=policy.spec_description(spec),module_name=module_name,
            module_package=package,module_file=file,
            code_sha256=None if builtin else policy.code_sha256(_imp.get_frozen_object(name)))
    for name,module in tuple(sys.modules.items()):
        if type(module)is not types.ModuleType:continue
        spec=vars(module).get('__spec__')
        if spec is not None and spec.origin in('built-in','frozen')and name!=spec.name:
            aliases.append(dict(registry_name=name,spec_name=spec.name,module_name=vars(module).get('__name__')))
    maps=Path('/proc/self/maps').read_text();rows=parse_maps(maps);paths={};kernel=[]
    for row in rows:
        path=row['path']
        if path.startswith('/')and('x'in row['permissions']or re.search(r'\.so(?:\.|$)',Path(path).name)):
            require(not path.endswith(' (deleted)'),'No deleted observer native mapping')
            p=Path(path);require(p.resolve()==p,'Canonical mapped native candidate')
            info=p.stat();require(stat.S_ISREG(info.st_mode),'Ordinary mapped native candidate')
            require([os.major(info.st_dev),os.minor(info.st_dev)]==row['device']and info.st_ino==row['inode'],
                    'Observed native path matches mapped inode')
            paths[path]=dict(path=path,fullstat=fullstat(info),bytes=info.st_size,payload_sha256=None,payload_read=False)
        elif 'x'in row['permissions']:
            require(path in('[vdso]','[vsyscall]'),'Original kernel executable mapping');kernel.append(row)
    runtime=policy.runtime_description();exe=policy.running_executable()
    require(exe['path']==runtime['resolved_executable']and exe['path']in paths,'Actual mapped original interpreter')
    bindings,providers=policy.provider_bindings()
    del bindings
    require('site'not in sys.modules and not {'torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules),
            'No site or SDK/model execution in observation')
    return dict(schema='nemotron4b_interpreter_manifest_candidate_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phase=phase,original_plan=dict(path=str(PLAN),sha256=PLAN_SHA),
        manifest=dict(runtime=runtime,executable=exe,providers=providers,modules=modules),
        module_counts=dict(builtin=sum(row['spec']['origin']=='built-in'for row in modules.values()),
                           frozen=sum(row['spec']['origin']=='frozen'for row in modules.values())),
        frozen_name_discovery=dict(complete_enumerator=complete,method=method,names=frozen),
        preloaded_canonical_interpreter_names=preloaded,preloaded_alias_boundaries=aliases,
        enumerated_but_unavailable=unavailable,native_stat_only_candidates=paths,observed_kernel_exec=kernel,
        current_maps_are_observer_state_not_future_worker_maps=True,
        native_candidate_payload_bytes=sum(row['bytes']for row in paths.values()),
        explicit_native_payload_file_bytes_read=0,frozen_code_definitions_inspected_in_memory=True,
        site_module_body_executed=False,site_main_executed=False,installed_startup_hooks_executed=False,
        actual_interpreter_policy_instantiated=False,actual_interpreter_authenticated=False,
        actual_model_or_SDK_executed=False,complete_runtime_dependency_closure_proven=False,
        runtime_admitted=False,new_speedup_claim=False)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in('native-meta','nxd-meta'),'One original phase')
    print(json.dumps(describe(sys.argv[1]),sort_keys=True))
