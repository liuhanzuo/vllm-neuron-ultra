"""Focused inert checks: no native packages, payloads, or subprocess launches."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import minicpm5_frame17_mlp_worker_v2 as w
import minicpm5_frame17_mlp_replay_v2 as replay
import minicpm5_frame17_mlp_runner_v2 as runner

checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (RuntimeError,ValueError,AssertionError):checks.append(name)
    else:raise AssertionError('Invalid condition accepted: '+name)


def run():
    check('import remains inert',not any(n in sys.modules for n in ('torch','numpy','torch_neuronx','neuronxcc')))
    binary=b'\x00\x80\xff\x00'
    w.same_bytes(binary,bytes(bytearray(binary)),'Actual binary equality')
    check('NUL/non-UTF8 raw equality accepted without JSON',True)
    rejects('one changed raw byte refused',lambda:w.same_bytes(binary,b'\x00\x80\xfe\x00','mismatch'))
    rejects('raw truncation refused',lambda:w.same_bytes(binary,binary[:-1],'length'))
    rejects('text cannot substitute raw bytes',lambda:w.same_bytes(binary,'\x00\x80\xff\x00','type'))
    rejects('capture role set uses direct equality before any payload IO',lambda:replay.run({},Path('/inert'),{},lambda:None))
    for n in ('worker','replay','runner'):
        path=w.ROOT/f'minicpm5_frame17_mlp_{n}_v2.py'
        tree=ast.parse(path.read_text());check(n+' syntax',bool(tree.body))
        topimports=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom))]
        check(n+' top-level native import absent',all(not any(a.name.split('.')[0]in ('torch','numpy')for a in x.names)for x in topimports))
    for role,path in w.ACTIVATIONS.items():
        text=Path(path).read_text();check(role+' actual SiLUActivation body/mapping',w.activation_source(text))
        rejects(role+' incorrect nn.SiLU expectation rejected',lambda:w.activation_source(text.replace('"silu": SiLUActivation','"silu": nn.SiLU')))
        rejects(role+' changed SiLU math rejected',lambda:w.activation_source(text.replace('return nn.functional.silu(input)','return input')))
        rejects(role+' changed kernel decorator rejected',lambda:w.activation_source(text.replace('@use_kernel_forward_from_hub("SiLU")','@use_kernel_forward_from_hub("Other")')))
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)and n.name=='SiLUActivation')
        nn=types.SimpleNamespace(Module=object)
        namespace=dict(nn=nn,Tensor=object,__name__='inert_activations',use_kernel_forward_from_hub=lambda name:lambda cls:cls)
        exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec',dont_inherit=True),namespace)
        cls=namespace['SiLUActivation'];act=cls();torch=types.SimpleNamespace(nn=nn)
        check(role+' actual undecorated fallback code accepted',w.activation_dispatch(torch,act,cls,text,path)['runtime_code_exact'])
        act.forward=types.MethodType(lambda self,input:input,act)
        rejects(role+' instance dispatch override refused',lambda:w.activation_dispatch(torch,act,cls,text,path))
        del act.forward
        original=cls.forward;cls.forward=lambda self,input:input
        rejects(role+' class kernel dispatch refused',lambda:w.activation_dispatch(torch,act,cls,text,path))
        cls.forward=original
        rejects(role+' alternate nn global refused',lambda:w.activation_dispatch(types.SimpleNamespace(nn=object()),act,cls,text,path))
    for shape,stride in [([1,28,2048],[57344,2048,1]),([1,128,2048],[1,1,128]),([1,128,3072],[393216,3072,1])]:
        check('actual dense layout '+str(stride),w.checked_layout(shape,stride,0)==shape[1]*shape[2])
    for shape,stride,offset in [([1,1,2048],[2048,2048,1],0),([1,128,2048],[1,1,1],0),
                              ([1,128,2048],[1,1,128],1),([1,128,2048],[1,0,128],0),
                              ([1,128,2048],[1,1,129],0),([2,128,2048],[262144,2048,1],0)]:
        rejects('reject wrong/overlap/partial layout '+str((shape,stride,offset)),lambda:w.checked_layout(shape,stride,offset))
    route=dict(past_key_value=None,get_kv_per_layer=False,update_kv_per_layer=False,is_for_context_encoding=True,idx=0,
               windowed_context_encoding_window_idx=-1)
    w.cpu_route(route);check('ordinary no-history prefill accepted',True)
    for k,v in [('past_key_value',object()),('get_kv_per_layer',True),('update_kv_per_layer',True),
                ('is_for_context_encoding',False),('idx',1),('windowed_context_encoding_window_idx',1)]:
        bad=dict(route);bad[k]=v;rejects('reject route '+k,lambda:w.cpu_route(bad))
    check('down-only sufficient label remains local',replay.classify(True,False,True).endswith('not_final_KL_causality'))
    check('front-half difference refuses unique attribution',replay.classify(False,False,False).startswith('front_half'))
    rejects('same exact product contradictory output refused',lambda:replay.classify(True,False,False))
    # Exercise exact full-shape control rejection independently from Tensor code.
    original=w.metadata;w.metadata=lambda torch,value:value
    try:
        good={n:dict(shape=w.expected_shapes(True)[n],sha256='a'*64)for n in w.FIELDS}
        check('all full controls accepted',len(replay.exact_controls(None,good,copy.deepcopy(good),'inert'))==5)
        for name in w.FIELDS:
            bad=copy.deepcopy(good);bad[name]['sha256']='b'*64
            rejects('full control changed '+name,lambda:replay.exact_controls(None,good,bad,'inert'))
        altered=copy.deepcopy(good);altered['input']['stride']=[1,1,28]
        rejects('full control layout changed',lambda:replay.exact_controls(None,good,altered,'inert'))
    finally:w.metadata=original
    # Hook ownership and restoration, using entirely inert module objects.
    handles=[]
    class Handle:
        def __init__(self):self.removed=False;handles.append(self)
        def remove(self):self.removed=True
    class Module:
        def register_forward_hook(self,*a,**kw):return Handle()
        def register_forward_pre_hook(self,*a,**kw):return Handle()
    class KV:
        def get_cache(self,*a):return 'original'
        def update_cache(self,*a):return 'original'
    class Pred:
        def __init__(self,*a):self.closed=False
        def close(self):self.closed=True
    mlp=Module();mlp.gate_proj=Module();mlp.up_proj=Module();mlp.down_proj=Module()
    layer=Module();layer.mlp=mlp
    core=types.SimpleNamespace(layers=[layer,Module()],norm=Module(),kv_mgr=KV(),
        neuron_config=types.SimpleNamespace(is_prefix_caching=False,windowed_context_encoding_size=None))
    saved=w.PredictionHooks;w.PredictionHooks=Pred
    try:
        observer=w.Observers(None,core,False,Path('/inert'),lambda:None)
        rejects('actual cache lookup forbidden',lambda:core.kv_mgr.get_cache())
        rejects('actual cache update forbidden',lambda:core.kv_mgr.update_cache())
        observer.route(None,(),route);check('one prefix route observed',observer.routes==1)
        rejects('second prefix rejected',lambda:observer.route(None,(),route))
        rejects('incomplete capture cannot stop successfully',lambda:observer.stop(None,(),None))
        observer.values={n:None for n in w.FIELDS}
        try:observer.stop(None,(),None)
        except w.PrefixDone as done:check('only private terminal object',done is observer.done)
        else:raise AssertionError('Prefix did not terminate')
        observer.close();check('all installed hooks removed',all(h.removed for h in handles))
        check('both KV methods restored',core.kv_mgr.get_cache()==core.kv_mgr.update_cache()=='original' and not core.kv_mgr.__dict__)
        # Cleanup on partially failed observer installation restores earlier hooks.
        core.neuron_config.is_prefix_caching=True;start=len(handles)
        rejects('bad prefix mode setup rolls back',lambda:w.Observers(None,core,False,Path('/inert'),lambda:None))
        check('partial setup hooks removed',all(h.removed for h in handles[start:]))
    finally:w.PredictionHooks=saved
    # Cleanup keeps the primary exception while still executing later actions.
    visited=[]
    def failed():visited.append('failure');raise ValueError('secondary')
    try:
        try:raise RuntimeError('primary')
        finally:w.phase.base.cleanup_actions([failed,lambda:visited.append('last')],'inert')
    except RuntimeError as e:check('cleanup retains primary and tries all',str(e)=='primary' and visited==['failure','last'])
    check('capture 16GiB',runner.props('cpu')['MemoryMax']==str(16*2**30))
    check('replay resident 4GiB no address-space trap',runner.props('replay')['MemoryMax']==str(4*2**30) and 'LimitAS'not in runner.props('replay'))
    check('replay one CPU',runner.props('replay')['AllowedCPUs']=='6')
    spec=dict(output_root=str(w.ROOT/'minicpm5-frame17-mlp-r2-20261107-v2'),units={'native':'trn1-mini-f17-mlp-native-inert'},master_port=29724)
    ref=dict(path='/inert/admission.json',sha256='a'*64);cmd=runner.service_command(spec,ref,'a'*32,'native')
    props=runner.props('native');value={k:v for k,v in props.items()if not k.endswith('Sec')}
    value.update(LoadState='loaded',InvocationID='a'*32,ExecStart='argv[]='+' '.join(cmd[cmd.index('/usr/bin/env'):])+' ;',
                 ExecStopPost='',RuntimeMaxUSec='600s',TimeoutStopUSec='15s')
    runner.service_identity(spec,ref,'a'*32,'native',value);check('exact service resources accepted',True)
    process=dict(cgroup='0::'+runner.p.GROUP_PREFIX+spec['units']['native']+'.service',
                 argv=runner.arguments('phase',ref,'native'))
    check('final Python exec acknowledged',runner.phase_exec_ready(process,spec,ref,'a'*32,'native'))
    transient=dict(process,argv=cmd[cmd.index('/usr/bin/env'):])
    check('exact env transition waits without ACK',runner.phase_exec_ready(transient,spec,ref,'a'*32,'native')is False)
    check('missing process waits without ACK',runner.phase_exec_ready(None,spec,ref,'a'*32,'native')is False)
    rejects('unrelated exec never acknowledged',lambda:runner.phase_exec_ready(dict(process,argv=['unrelated']),spec,ref,'a'*32,'native'))
    rejects('wrong phase cgroup rejected',lambda:runner.phase_exec_ready(dict(process,cgroup='wrong'),spec,ref,'a'*32,'native'))
    for key,bad in [('MemoryMax','infinity'),('MemorySwapMax','infinity'),('ExecStopPost','unexpected'),('ExecStart','other')]:
        changed=dict(value);changed[key]=bad
        rejects('reject actual service drift '+key,lambda:runner.service_identity(spec,ref,'a'*32,'native',changed))
    check('no native imports after all controls',not any(n in sys.modules for n in ('torch','numpy','torch_neuronx','neuronxcc')))
    return dict(schema='minicpm5_frame17_mlp_source_controls_v2',passed=True,checks=len(checks),controls=checks,
                native_imports=0,raw_payload_reads=0,model_launches=0,
                source_sha256={str(w.ROOT/f'minicpm5_frame17_mlp_{n}_v2.py'):hashlib.sha256((w.ROOT/f'minicpm5_frame17_mlp_{n}_v2.py').read_bytes()).hexdigest()for n in ('worker','replay','runner')})


if __name__=='__main__':
    result=run()
    if len(sys.argv)==2:
        path=Path(sys.argv[1]);w.once(path,result)
    print(json.dumps(dict(passed=True,checks=result['checks'],native_imports=0,model_launches=0)))
