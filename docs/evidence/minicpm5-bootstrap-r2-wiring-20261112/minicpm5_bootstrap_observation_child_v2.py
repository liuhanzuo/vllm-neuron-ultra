"""Exact literal discovery child. Invoked only with frozen helper source and ACK.

Never invokes either formal source_entry or creates a native authority.
Only sys/posix/time are imported before the earliest full map snapshot.
"""
import sys
import posix
import time


def validate_arguments(runtime,started):
    require=helpers['require'];digest=helpers['digest']
    args=runtime.argv
    require(len(args)==10 and args[0]=='-c' and args[1]=='--mini-bootstrap-observe','Exact discovery child argv')
    role=args[2];deadline=float(args[8])
    require(role in ('sdk310','outer312') and started<deadline<=started+900,'One inherited finite deadline')
    require(runtime.flags.isolated and runtime.flags.no_site and runtime.flags.ignore_environment
        and runtime.flags.no_user_site and runtime.flags.optimize==0 and runtime.dont_write_bytecode and __debug__
        and runtime.pycache_prefix=='/dev/null','Original isolated source-only Python flags')
    require(len(runtime.orig_argv)==17 and runtime.orig_argv[0]==runtime.executable
        and runtime.orig_argv[1:7]==['-I','-S','-B','-X','pycache_prefix=/dev/null','-c']
        and runtime.orig_argv[8:]==args[1:] and digest(runtime.orig_argv[7].encode())==args[9],
        'Actual entire literal command authenticated')
    return role,deadline


def validate_admission(args,original,admission,deadline,parent_pid):
    require=helpers['require']
    command=list(admission['command']);require(len(command)==17 and command[13]=='admission:self',
        'No circular or fabricated admission hash')
    command[13]=args[6]
    require(admission['schema']=='minicpm5_bootstrap_observation_admission_v1'
      and admission['role']==args[2] and admission['plan']==dict(path=args[3],sha256=args[4])
      and admission['deadline']==deadline==admission['started_monotonic']+900
      and admission['literal_sha256']==args[9] and command==original
      and parent_pid==admission['parent_owner']['pid']
      and admission['accepted_native_authority_created'] is False,
      'Exact actual admission, original parent and unchanged900s deadline')


def main():
    require=helpers['require'];digest=helpers['digest'];args=sys.argv
    role,deadline=validate_arguments(sys,time.monotonic())
    types_tracker=types_helpers['TypesSourceObservation'](sys) if role=='outer312' else None
    def time_check():require(time.monotonic()<deadline,'Original inherited deadline expired')
    def maps():
        time_check();fd=posix.open('/proc/self/maps',posix.O_RDONLY|posix.O_CLOEXEC);raw=bytearray()
        try:
            while True:
                time_check();part=posix.read(fd,min(65536,1048577-len(raw)))
                if not part:break
                raw.extend(part);require(len(raw)<=1048576,'Full maps fit one-file budget; never truncate')
        finally:posix.close(fd)
        return bytes(raw)
    earliest=maps()

    class DiscoveryEntry(helpers['Entry']):
        allow_ctypes=False
        def native_import_allowed(self,name):
            return self.allow_ctypes and (name=='_ctypes' or name=='ctypes' or name.startswith('ctypes.'))
    entry=DiscoveryEntry();entry.deadline=deadline;entry.role=role
    entry.prefix_identity=entry.prefix();entry.state='prefix_verified'
    external=sys.modules['_frozen_importlib_external']
    entry.install(external)
    # Parent owns the only pipe writer; owner is recorded before this byte.
    fd=int(args[7]);require(posix.read(fd,1)==b'R','Exact live-owner parent ACK');posix.close(fd)
    if types_tracker is not None:types_tracker.install_before_json(entry,external,helpers['SOURCE_TOTAL_CAP'])
    import json
    if types_tracker is not None:types_tracker.after_json()
    def read_json(path,sha):return json.loads(entry.read_source(path,sha,8*2**20)[0])
    plan=read_json(args[3],args[4]);entry.plan={'bootstrap_sources':plan['source_sha256']}
    admission=read_json(args[5],args[6])
    validate_admission(args,sys.orig_argv,admission,deadline,posix.getppid())
    table=read_json(plan['candidate_inputs']['path'],plan['candidate_inputs']['sha256'])
    require(sys.executable==table[role]['python'] and posix.readlink('/proc/self/exe')==table[role]['executable']
      and tuple(sys.version_info[:2])==((3,10) if role=='sdk310' else (3,12)), 'Exact original role interpreter')
    content=read_json(admission['content']['path'],admission['content']['sha256'])
    require(content['current_candidate_bytes_authenticated'] is True
      and content['accepted_native_authority_created'] is False,'Saved finite content observation, not authority')
    guard=entry.check
    out=admission['output_root']
    require(out==plan['output_root']+'/'+role,'Only owned role output')
    def write(name,raw,failed=False):
        time_check()
        if not failed:guard()
        require(name in ('earliest.maps','pre-ctypes.maps','post-ctypes.maps','failed-bootstrap.maps','observation.json','failure.json')
            and type(raw) is bytes and len(raw)<=2097152,'Finite original one-file output')
        path=out+'/'+name;fd=posix.open(path,posix.O_WRONLY|posix.O_CREAT|posix.O_EXCL|posix.O_NOFOLLOW,0o600)
        try:
            at=0
            while at<len(raw):
                time_check();written=posix.write(fd,raw[at:]);require(written>0,'Complete output write');at+=written
            posix.fsync(fd)
        finally:posix.close(fd)
        directory=posix.open(out,posix.O_RDONLY|posix.O_DIRECTORY|posix.O_NOFOLLOW)
        try:posix.fsync(directory)
        finally:posix.close(directory)
        return dict(path=path,bytes=len(raw),sha256=digest(raw,time_check if failed else entry.check))
    refs={'earliest':write('earliest.maps',earliest)}
    try:
        # Wrap the existing source finder for exact nonlocal ctypes/struct sources.
        # Unknown stdlib extensions are allowed to be observed, never authenticated here.
        original_finder=entry.finder;pure=table[role]['pure_source_candidates']
        class PureLoader(external.SourceFileLoader):
            def get_code(self,name):
                raw,identity=entry.read_source(self.path,pure[self.path]['sha256'])
                entry.source_bytes+=len(raw)
                require(entry.source_bytes<=helpers['SOURCE_TOTAL_CAP'],'Whole executed source byte cap')
                entry.sources[self.path]=dict(sha256=pure[self.path]['sha256'],bytes=len(raw),identity=identity)
                return compile(raw,self.path,'exec',dont_inherit=True,optimize=0)
        class DiscoveryFinder:
            def find_spec(self,name,path=None,target=None):
                found=original_finder.find_spec(name,path,target)
                if found is not None:return found
                if name=='_ctypes' and role=='outer312':
                    require('_ctypes' in sys.builtin_module_names,'Original outer builtin ctypes provider');return None
                spec=external.PathFinder.find_spec(name,path,target)
                if name=='_ctypes' and role=='sdk310':
                    require(spec is not None and spec.origin==table[role]['ctypes_extension']
                        and type(spec.loader) is external.ExtensionFileLoader,'Exact SDK ctypes extension selected')
                if spec is not None and spec.origin in pure:
                    require(type(spec.loader) is external.SourceFileLoader,'Exact ordinary pure-source provider')
                    spec.loader=PureLoader(name,spec.origin);return spec
                if name=='ctypes' or name.startswith('ctypes.'):
                    raise ValueError('ctypes pure source outside exact saved origin set')
                return None
        wrapped=DiscoveryFinder();sys.meta_path[0]=wrapped;entry.finder=wrapped
        # Only source directories; -S stays effective and no .pth/sitecustomize runs.
        sys.path[:0]=[helpers['ROOT'],helpers['ROOT']+'/research']
        from research import minicpm5_bootstrap_observation_v2 as core
        review=read_json(admission['review']['path'],admission['review']['sha256'])
        core.validate(plan,review,admission['plan'])
        import importlib
        require('ctypes' not in sys.modules and '_ctypes' not in sys.modules,'No premature ctypes import')
        importlib.import_module(core.ENTRIES[role])
        require('ctypes' not in sys.modules and '_ctypes' not in sys.modules,'Actual formal inert bootstrap does not load ctypes')
        from research import minicpm5_sdk_mapping_native_closure_v2 as closure
        from research import descriptor_serial_lock_proof_v1 as lockproof
        import minicpm5_layer0_execution_protocol_v4 as proc
        def guard():
            entry.check();require(proc.proc(admission['parent_owner']['pid'])==admission['parent_owner'], 'Same actual original parent')
            lockproof.prove(admission['parent_owner'],admission['lock_identity'],admission['retained_fd'],
               proc=proc.proc,open_binary=lambda path:open(path,'rb'),lstat=posix.lstat,stat=posix.stat,readlink=posix.readlink)
        guard()
        import resource
        require(resource.getrlimit(resource.RLIMIT_AS)==(2147483648,)*2
          and resource.getrlimit(resource.RLIMIT_CPU)==(600,605)
          and resource.getrlimit(resource.RLIMIT_FSIZE)==(2097152,)*2
          and posix.sched_getaffinity(0)=={6} and posix.getpriority(posix.PRIO_PROCESS,0)>=19,
          'Exact inherited original AS/CPU/FSIZE/affinity/nice limits')
        before=maps()
        refs['pre_ctypes']=write('pre-ctypes.maps',before)
        entry.allow_ctypes=True
        import ctypes
        require(ctypes.__file__ in pure and isinstance(ctypes.__loader__,PureLoader),'Actual pinned ctypes pure source')
        libc=ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libc.so.6',use_errno=True)
        guard();after=maps();refs['post_ctypes']=write('post-ctypes.maps',after)
        selected={name:dict(origin=getattr(getattr(sys.modules.get(name),'__spec__',None),'origin',None),
             loader=type(getattr(getattr(sys.modules.get(name),'__spec__',None),'loader',None)).__name__,
             builtin=name in sys.builtin_module_names) for name in ('_ctypes','_struct','_hashlib','_blake2','zlib')}
        pure_receipts={path:dict(observed_module_names=sorted(name for name,module in tuple(sys.modules.items())
            if getattr(module,'__file__',None)==path),pinned_source_receipt_present=path in entry.sources)
            for path in pure}
        from pathlib import Path
        summaries={name:core.summarize_maps(raw.decode('ascii'),role,table,content,closure,lambda path:str(Path(path).resolve()))
            for name,raw in [('earliest',earliest),('pre_ctypes',before),('post_ctypes',after)]}
        ffi=content['files'][table[role]['ffi_candidate']]
        types_observation=types_tracker.finish(pure[types_helpers['PIN']['path']]) if types_tracker is not None else None
        source_gaps=core.source_execution_gaps(role,types_observation,entry.sources)
        value=dict(schema='minicpm5_bootstrap_role_observation_v2',role=role,complete=True,plan=admission['plan'],
            admission=dict(path=args[5],sha256=args[6]),parent_owner=admission['parent_owner'],pid=posix.getpid(),
            executable=posix.readlink('/proc/self/exe'),argv=list(sys.orig_argv),deadline=deadline,
            formal_bootstrap_entry=core.ENTRIES[role],executed_sources=entry.sources,
            maps=refs,map_summaries=summaries,selected_stdlib_providers=selected,
            known_pure_source_observations=pure_receipts,types_source_observation=types_observation,
            source_execution_provenance_gaps=source_gaps,
            libc_requested=libc._name,ffi_definitions=ffi.get('ffi'),ffi_failure=ffi.get('ffi_failure'),
            ffi_definition_scope='known_candidate_content_only_not_actual_provider_selection',
            missing_content_origins=any(v['gaps'] for v in summaries.values()),
            accepted_native_authority_created=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        write('observation.json',core.encoded(value));guard()
    except BaseException as error:
        # The original entry marks failed imports permanently. Preserve partial
        # raw evidence without attempting more imports or pretending pre-ctypes
        # bootstrap completed. The absolute deadline still bounds these writes.
        try:
            refs['failed_bootstrap']=write('failed-bootstrap.maps',maps(),failed=True)
            value=dict(error_type=type(error).__name__,error=str(error)[:4096],maps=refs,
                executed_sources=entry.sources,accepted_native_authority_created=False,
                full_model_qualified=False,new_speedup_claim=False)
            write('failure.json',(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(),failed=True)
        except BaseException as reporting:
            posix.write(2,('Observation failure evidence unavailable: '+repr(reporting)[:1024]+'\n').encode())
        raise


if __name__=='__main__':raise RuntimeError('Only the root-reviewed exact literal/parent ACK may invoke this child')
