def original_code_fixture(native_raises=False, mutation=None, commit_early=True):
    """Real original Python code, with explicitly inert native/model objects."""
    permit,initial,controller,witness=sample();store={};current=[copy.deepcopy(initial)]
    old_life,old_observe=runtime.lifecycle,runtime.observe
    fixture=types.SimpleNamespace()
    fixture.p=types.SimpleNamespace(directory=lambda ignored:root);fixture.CONTROL=control
    fixture.read=lambda path:json.loads(life.read_bytes(path))
    def record(path,value):
        raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        result=atomic_writer(path,raw)
        transactions.append(dict(name=Path(path).name,end=time.monotonic(),sha256=result['sha256']))
    fixture.record=record;fixture.monitor=lambda ignored:None
    runtime.lifecycle=lambda:fixture;runtime.observe=lambda ignored:copy.deepcopy(current[0])
    class Context:
        kind='server';spawned=True
        def __init__(self):
            self.permit=permit;self.row=permit['run'];self.plan={'source_sha256':dict(hook.PINS)}
            self.title_witness=None;runtime.init_context(self);self.polls=0;self.native_calls=0;self.callback_lock=threading.RLock();self.external_writer=external_writer
        def source_bytes(self,path,wanted):
            raw=path.read_bytes();check('fixed source input '+path.name,hashlib.sha256(raw).hexdigest()==wanted);return raw
        def actual_environment(self):return {}
        def poll(self):
            self.polls+=1;guard_work()
            if commit_early:runtime.acknowledge(permit,controller)
            runtime.worker_check(self)
            if not commit_early:runtime.acknowledge(permit,controller)
        write=actual_context_write
        record=protocol.RuntimeContext.record
        record_engine_title_before=actual_title_write
        def begin_engine_title(self,owner,value):return runtime.begin(self,owner,value)
        def complete_engine_title(self,owner,token,result):return runtime.complete(self,owner,token,result)
        def fail_engine_title(self,owner,token,error_type):return runtime.fail(self,owner,token,error_type)
    context=Context();modules={}
    core=module(hook.MODULES[hook.CORE],hook.CORE);utils=module(hook.MODULES[hook.UTILS],hook.UTILS);envs=module(hook.MODULES[hook.ENVS],hook.ENVS)
    modules.update({core.__name__:core,utils.__name__:utils,envs.__name__:envs})
    envs.VLLM_PROCESS_NAME_PREFIX='VLLM';utils.envs=envs
    package=types.SimpleNamespace(__version__='1.3.7');native=types.SimpleNamespace()
    class NativeError(RuntimeError):pass
    def setproctitle(name):
        context.native_calls+=1
        check('actual original Python setter calls inert native once with title',name==t.TITLE.decode())
        current[0]=changed(initial)
        if native_raises:raise NativeError('inert original native failure')
    package.setproctitle=native.setproctitle=setproctitle;utils.setproctitle=package
    observer=hook.EngineCoreTitleHook(context,modules);context.title_witness=observer
    # Only genuine native provenance is replaced, explicitly and locally.
    # All core/alias/default/caller/line/DP Python binding code remains real.
    def inert_bind_native():
        observer.package,observer.native,observer.native_setter=package,native,setproctitle
    observer.bind_native=inert_bind_native
    setter=types.FunctionType(observer.setter_code,vars(utils),argdefs=('', 'VLLM'))
    utils.set_process_title=core.set_process_title=setter
    engine=types.FunctionType(observer.core_code,vars(core));engine.__kwdefaults__={'dp_rank':0,'local_dp_rank':0}
    class Engine:
        run_engine_core=staticmethod(engine)
        def __init__(self,*args,**kwargs):pass
        def run_busy_loop(self):check('engine body reached only after commit',observer.committed)
        def shutdown(self):pass
    core.EngineCoreProc=Engine;core.maybe_register_config_serialize_by_value=lambda:None
    core.signal=types.SimpleNamespace(SIGTERM=15,SIGINT=2,signal=lambda *args:None)
    core.decorate_logs=lambda:None;core.logger=types.SimpleNamespace(debug=lambda *a:None,exception=lambda *a:None)
    import smollm3_observer_invariance_protocol_v14 as actual_protocol
    actual_ray=actual_protocol.environment_policy.ray
    check("actual production dispatcher is reviewed Ray module",actual_ray is ray)
    owner=actual_ray.RayImportWitness.__new__(actual_ray.RayImportWitness)
    owner.context=context;owner.busy=False;owner.after={'fixture_committed':True};owner.before=None
    owner.installed_trace=None
    config=types.SimpleNamespace(parallel_config=types.SimpleNamespace(data_parallel_size=1))
    if mutation:mutation(observer,core,utils,envs,config)
    failure=None
    try:
        check('no pre-existing trace owner in source check',sys.gettrace()is None)
        owner.install()
        try:engine(vllm_config=config)
        except BaseException as exc:failure=exc
        check('single Ray owner retained or CPython disabled it on callback failure',
              observer.trace_owner is owner.installed_trace and
              (sys.gettrace()is owner.installed_trace or failure is not None and sys.gettrace()is None))
    finally:
        sys.settrace(None);runtime.lifecycle, runtime.observe=old_life,old_observe
    return dict(observer=observer,context=context,store=store,failure=failure,native_error=NativeError,
                initial=initial,after=current[0],owner=owner,core=core)
