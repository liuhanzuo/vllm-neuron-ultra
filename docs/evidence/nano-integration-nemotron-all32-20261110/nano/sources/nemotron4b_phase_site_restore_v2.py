"""Restore separately reviewed phase startup after a real parent gate.

Unlike helper restore v1, preserves four independently recorded CPython prefix
strings and explicit directory aliases, and supports authenticated frozen site.
No model/native admission, interpreter authentication or dependency closure is
provided by this module; actual origin_context/authenticate_site are mandatory.
"""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
import importlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import time

from nemotron4b_guarded_helper_entry_v2 import BoundedText
from nemotron4b_helper_dependency_catalog_v1 import FIELDS,source as metadata_source
from nemotron4b_pinned_source_reader_v1 import source as python_source
from nemotron4b_process_handshake_v1 import require

PREFIXES=('prefix','exec_prefix','base_prefix','base_exec_prefix')
USED=False


def prefixes():return {name:getattr(sys,name)for name in PREFIXES}
def identity(info):return {name:getattr(info,name)for name in FIELDS}


def paths(policy):
    aliases=policy['directory_aliases']
    require(type(aliases)is dict and len(aliases)<=16,'Exact bounded directory aliases')
    for path,row in aliases.items():
        p=Path(path)
        require(type(row)is dict and set(row)=={'link_text','target','fullstat'}
                and p.is_absolute()and p.is_symlink()and identity(p.lstat())==row['fullstat']
                and stat.S_ISLNK(row['fullstat']['st_mode'])and os.readlink(p)==row['link_text']
                and str(p.resolve())==row['target']and Path(row['target']).is_dir(),
                'Original explicitly recorded directory alias unchanged')
    declared=[*policy['pre_path'],*policy['post_path'],*policy['pre_prefixes'].values(),
              *policy['post_prefixes'].values(),policy['script'],policy['venv_config'],policy['site_file']]
    for value in declared:
        p=Path(value);require(p.is_absolute()and '..'not in p.parts,'Exact original absolute startup path')
        for part in(p,*p.parents):
            if part.is_symlink():require(str(part)in aliases,'No unreviewed startup path alias')


def startup_inputs(policy):
    require(type(policy)is dict and set(policy)=={'script','pre_argv','site_file','site_origin','venv_config',
            'site_packages','files','pth_files','pre_path','post_path','pre_prefixes','post_prefixes',
            'directory_aliases'},'Exact phase startup policy fields')
    require(all(type(policy[k])is dict and set(policy[k])==set(PREFIXES)for k in('pre_prefixes','post_prefixes')),
            'All four exact pre/post prefixes')
    require(type(policy['pre_argv'])is list and policy['pre_argv']and policy['pre_argv'][0]==policy['script'],
            'Actual phase startup argv')
    require(policy['site_origin']in('frozen',policy['site_file']),'Original frozen or source site origin')
    paths(policy)
    directory=Path(policy['site_packages'])
    require(directory.is_dir()and directory.resolve()==directory,'Original canonical phase site packages')
    actual=sorted(str(p)for p in directory.glob('*.pth'))
    require(actual==policy['pth_files'],'Complete original phase pth set')
    require({policy['site_file'],policy['venv_config'],*actual}<=set(policy['files']),
            'Original site/config/allpth sources required')
    verified={}
    for path,row in policy['files'].items():
        if path.endswith('.py'):raw,current=python_source(path,row)
        else:raw,current=metadata_source(path,row['sha256'])
        require(len(raw)==row['bytes']and current['fullstat']==row['fullstat'],
                'Complete original phase startup source identity')
        verified[path]=dict(row)
    return dict(files=verified,pth_files=actual,directory_aliases=policy['directory_aliases'])


@contextmanager
def restore(policy,*,guard,origin_context,authenticate_site,provenance):
    global USED
    require(not USED,'One phase startup restoration per process');USED=True
    require(all(callable(x)for x in(guard,origin_context,authenticate_site))and type(provenance)is dict
            and not provenance,'Actual resource/origin/site callbacks and fresh record')
    provenance.update(schema='nemotron4b_phase_site_restore_v2',complete=False,
                      actual_SDK_runtime_qualified=False,runtime_admitted=False,
                      no_site_flag_equivalence_proven=False,new_speedup_claim=False)
    require(sys.flags.no_site==1 and sys.flags.no_user_site==1 and sys.dont_write_bytecode
            and sys.flags.optimize==0 and __debug__,'Original gated -S -B no-user-site runtime')
    require(sys.argv==policy['pre_argv']and sys.path==policy['pre_path']
            and prefixes()==policy['pre_prefixes'],'Actual exact phase argv/path/four prefixes')
    require('site'not in sys.modules and not {'torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules),
            'Phase startup precedes SDK/model imports')
    guard();before=startup_inputs(policy);provenance['inputs_before']=before
    stdout,stderr=BoundedText(4096),BoundedText(4096)
    try:
        with origin_context()as imports:
            require(type(imports)is dict,'Authenticated actual phase import context')
            guard();spec=importlib.util.find_spec('site')
            require(spec is not None and spec.origin==policy['site_origin'],'Original phase site provider')
            # Frozen interpreter code must be authenticated by its actual
            # interpreter policy before import, never by disk source alone.
            authenticate_site(spec,None)
            provenance['restore_start_ns']=time.monotonic_ns()
            try:
                with redirect_stdout(stdout),redirect_stderr(stderr):
                    site=importlib.import_module('site')
                    require(site.__file__==policy['site_file']and site.__spec__.origin==policy['site_origin'],
                            'Actual original site module file and origin')
                    authenticate_site(site.__spec__,site)
                    site.main()
            finally:
                provenance['startup_stdout']=b''.join(stdout.parts).decode('utf-8',errors='replace')
                provenance['startup_stderr']=b''.join(stderr.parts).decode('utf-8',errors='replace')
            require(stdout.bytes()==stderr.bytes()==b'','Silent complete phase startup; swallowed errors fail')
            guard();require(sys.path==policy['post_path']and prefixes()==policy['post_prefixes'],
                            'Exact restored phase path and all four prefixes')
            require(site.ENABLE_USER_SITE is False and 'usercustomize'not in sys.modules,
                    'Original user-site exclusion retained')
            require(startup_inputs(policy)==before,'Original startup sources/aliases retained')
            provenance.update(startup_complete=True,startup_complete_ns=time.monotonic_ns(),
                              actual_path=list(sys.path),actual_prefixes=prefixes(),
                              site_origin=site.__spec__.origin,imports=imports)
            yield provenance
            guard();require(startup_inputs(policy)==before and sys.path==policy['post_path']
                            and prefixes()==policy['post_prefixes']and site.ENABLE_USER_SITE is False,
                            'Phase startup state retained after original work')
        guard();provenance['complete']=True
    except BaseException as error:
        provenance['failure']=dict(type=type(error).__name__,message=str(error)[:2048])
        raise
