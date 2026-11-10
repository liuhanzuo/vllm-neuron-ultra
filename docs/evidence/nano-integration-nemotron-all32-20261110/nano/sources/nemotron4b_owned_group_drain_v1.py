"""Owned service cleanup for the future Nano4B native/meta root launcher.

No launch/admission/lock release. The root must own the serial lock, persist
its exact nonce-bearing service command, and supply the operation guard before
using LiveProbe. Source tests use generated cgroup trees and a fake service.
Successful child exits never substitute for an empty whole cgroup tree.
"""
import copy
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time

from nemotron4b_process_handshake_v1 import require

GROUP_PREFIX='/user.slice/user-1000.slice/user@1000.service/app.slice/'
UNIT_PREFIX='trn1-latest-verify-nemotron4b-native-prerequisite-v3-'
KEYS=('LoadState','ActiveState','SubState','MainPID','ControlPID','Job',
      'InvocationID','ControlGroup','ExecStart','ExecMainPID',
      'ExecMainCode','ExecMainStatus','Result')


def supervisor_summary(result,nonce):
    """Adapt the two existing supervisor schemas without promoting failures.

    Raw stdout/stderr stay in result for the caller to persist separately.
    Successful summaries require actual direct-child exit/capture fields;
    complete original records still need independent final root verification.
    """
    require(type(result)is dict and result.get('nonce')==nonce,'Actual matching supervisor result')
    schema=result.get('schema')
    if schema=='nemotron4b_gated_child_supervision_v1':
        passed=result['successful'];rows=[result]
        if passed:
            require(result['handshake']is not None and result['failure']is None,
                    'Successful gated child requires captured handshake')
    elif schema=='nemotron4b_gated_rank_pair_v1':
        passed=result['direct_children_successful']
        require(set(result['ranks'])=={'rank-0','rank-1'},'Both actual rank outcomes')
        rows=list(result['ranks'].values())
        if passed:
            require(result['pair_capture']is not None and result['pair_capture_receipt']is not None
                    and result['failure']is None and all(r['grant']is not None
                    and r['first_actual_capture']==r['second_actual_capture']
                    and r['first_actual_capture']is not None for r in rows),
                    'Successful pair requires original capture and grants')
    else:raise ValueError('Only original gated child/pair result schemas')
    require(type(passed)is bool and type(result['cleanup_errors'])is list,'Typed supervisor outcome')
    for row in rows:
        require(type(row['output'])is dict and set(row['output'])=={'stdout','stderr'}
                and all(type(raw)is bytes and len(raw)<=2**20+1 for raw in row['output'].values()),
                'Actual bounded captured output including one over-cap sentinel byte')
    if passed:
        require(not result['cleanup_errors']and all(r['child_reaped']is True
                and r['actual_exit_observed']is True and type(r['actual_exit_code'])is int
                and r['actual_exit_code']==0 and not r['cleanup_signals']for r in rows),
                'Successful direct children exited zero without forced cleanup')
    return dict(successful=passed,supervisor_schema=schema,nonce=nonce,
                direct_children=len(rows),actual_exit_codes=[r['actual_exit_code']for r in rows])


class PendingDrain(RuntimeError):
    """Authenticated workload still present, or terminal samples unstable."""


def bounded(path,cap=16384):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        chunks=[];size=0
        while True:
            value=os.read(fd,min(4096,cap+1-size))
            if not value:break
            chunks.append(value);size+=len(value)
            require(size<=cap,'Bounded complete kernel metadata')
        return b''.join(chunks).decode()
    finally:os.close(fd)


def process(pid):
    require(type(pid)is int and pid>1,'Explicit recorded PID')
    base=Path('/proc')/str(pid)
    try:
        before=bounded(base/'stat',8192).rsplit(') ',1)[1].split()
        group=bounded(base/'cgroup',4096).strip()
        after=bounded(base/'stat',8192).rsplit(') ',1)[1].split()
    except FileNotFoundError:return None
    require(before[19]==after[19]and group.startswith('0::/')and '\n'not in group,
            'Stable process start and unified cgroup')
    # Zombies still count as present until the actual owner reaps them.
    return dict(pid=pid,start_ticks=before[19],cgroup=group,state=after[0])


def inspect_group(group,*,root=Path('/sys/fs/cgroup')):
    require(group in [GROUP_PREFIX+UNIT_PREFIX+p+'.service'for p in('native-meta','nxd-meta')],
            'Exact bounded Nano4B service group')
    root=Path(root);base=root/group.lstrip('/')
    require(root.is_absolute()and root.resolve()==root
            and not any(p.is_symlink()for p in(base,*base.parents)),
            'Canonical original cgroup tree')
    if not base.exists():return dict(group=group,absent=True,empty=True,rows=[])
    pending=[base];rows=[];total=0
    while pending:
        directory=pending.pop()
        require(len(rows)<128,'Bounded complete cgroup subtree')
        info=directory.lstat();require(stat.S_ISDIR(info.st_mode),'Ordinary cgroup directory')
        words=bounded(directory/'cgroup.procs').split()
        require(all(re.fullmatch('[1-9][0-9]*',x)for x in words),'Actual cgroup process list')
        pids=sorted({int(x)for x in words});total+=len(pids)
        require(total<=4096,'Bounded process membership')
        events={}
        for line in bounded(directory/'cgroup.events').splitlines():
            parts=line.split();require(len(parts)==2 and parts[0]not in events,'Complete unique cgroup events')
            events[parts[0]]=parts[1]
        require(events.get('populated')in('0','1'),'Kernel subtree population field required')
        children=[]
        with os.scandir(directory)as entries:
            for entry in entries:
                require(not entry.is_symlink(),'No cgroup subtree symlinks')
                if entry.is_dir(follow_symlinks=False):children.append(Path(entry.path))
        after=directory.lstat()
        require((info.st_dev,info.st_ino)==(after.st_dev,after.st_ino),'Stable cgroup directory identity')
        rows.append(dict(path=str(directory.relative_to(base)),device=info.st_dev,
                         inode=info.st_ino,pids=pids,populated=events['populated']))
        pending.extend(sorted(children,reverse=True))
    return dict(group=group,absent=False,empty=all(not r['pids']and r['populated']=='0'for r in rows),rows=rows)


def terminal(value):
    return (value['MainPID']==value['ControlPID']=='0'and not value['Job']
            and (value['ActiveState']in('inactive','failed')or
                 (value['ActiveState'],value['SubState'])==('active','exited')))


class LiveProbe:
    """Actual operations require a fresh caller lock/ownership guard each time.

    Root exclusivity prevents cooperating controllers from replacing a unit
    between query and operation; systemctl does not offer atomic invocation-ID
    conditional stop. The caller must preserve the fixed fresh namespace.
    """
    def __init__(self,operation_guard):
        require(callable(operation_guard),'Actual operation/serial guard required')
        self.guard=operation_guard

    def command(self,args):
        self.guard()
        output={'stdout':bytearray(),'stderr':bytearray()}
        child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,close_fds=True)
        deadline=time.monotonic()+10
        try:
            with selectors.DefaultSelector()as selector:
                for name in output:
                    stream=getattr(child,name);os.set_blocking(stream.fileno(),False)
                    selector.register(stream,selectors.EVENT_READ,name)
                while selector.get_map():
                    self.guard();left=deadline-time.monotonic()
                    require(left>0,'Bounded service operation time')
                    for key,_ in selector.select(min(.05,left)):
                        value=os.read(key.fileobj.fileno(),min(4096,65537-len(output[key.data])))
                        if not value:selector.unregister(key.fileobj)
                        else:
                            output[key.data].extend(value)
                            require(len(output[key.data])<=65536,'Bounded systemd response')
                code=child.wait(timeout=max(.001,deadline-time.monotonic()))
            require(code==0,'Actual systemd operation failed: '+output['stderr'].decode(errors='replace')[:1024])
        finally:
            if child.poll()is None:child.kill()
            child.wait(timeout=2)
            child.stdout.close();child.stderr.close()
        self.guard()
        return output['stdout'].decode()

    def properties(self,unit,resource_keys):
        keys=','.join((*KEYS,*resource_keys))
        text=self.command(['systemctl','--user','show','--all',unit,'--property='+keys])
        rows=[line.split('=',1)for line in text.splitlines()if '='in line]
        require(len(rows)==len({x[0]for x in rows}),'Unique service property fields')
        return dict(rows)

    def stop(self,unit):
        self.command(['systemctl','--user','stop','--no-block',unit])

    def kill(self,unit):
        self.command(['systemctl','--user','kill','--kill-whom=all','--signal=SIGKILL',unit])

    def group(self,name):
        self.guard();return inspect_group(name)

    def process(self,pid):
        self.guard();return process(pid)

    @staticmethod
    def clock():return time.monotonic()

    @staticmethod
    def wait(seconds):time.sleep(seconds)


class OwnedPhase:
    def __init__(self,*,phase,nonce,exec_argv,resources,probe,record):
        require(phase in('native-meta','nxd-meta'),'Original two native/meta phases')
        require(type(nonce)is str and re.fullmatch('[0-9a-f]{32}',nonce),'Exact attempt nonce')
        require(type(exec_argv)is list and exec_argv[:2]==['/usr/bin/env','-i']
                and exec_argv.count('NEMOTRON4B_NATIVE_PREREQUISITE_ATTEMPT='+nonce)==1
                and all(type(x)is str and x and not any(c.isspace()for c in x)for x in exec_argv),
                'Exact unambiguous clean nonce-bearing command')
        require(type(resources)is dict and resources and all(type(k)is str and type(v)is str
                and re.fullmatch('[A-Za-z][A-Za-z0-9]*',k)for k,v in resources.items()),
                'Exact reviewed resource property table')
        require(not set(resources)&set(KEYS)and callable(record),'Disjoint resource keys and durable record callback')
        self.unit=UNIT_PREFIX+phase;self.group_name=GROUP_PREFIX+self.unit+'.service'
        self.nonce=nonce;self.argv=tuple(exec_argv);self.resources=copy.deepcopy(resources)
        self.probe,self.record=probe,record
        self.invocation=None;self.owners=[];self.actions=[];self.samples=[];self.main_owner=None

    def properties(self):
        value=self.probe.properties(self.unit,tuple(self.resources))
        require(type(value)is dict and set(KEYS)<=set(value),'Complete service identity query')
        return value

    def authenticate(self,value):
        require(value['LoadState']=='loaded','Owned service identity unavailable')
        invocation=value['InvocationID']
        require(re.fullmatch('[0-9a-f]{32}',invocation),'Actual service invocation ID')
        match=re.search(r'argv\[\]=(.*?) ; ignore_errors=',value['ExecStart'])
        require(match is not None and match.group(1)==' '.join(self.argv),'Owned service command/nonce differs')
        require(value['ControlGroup']in('',self.group_name)
                and(terminal(value)or value['ControlGroup']==self.group_name),'Owned service group differs')
        require(self.invocation is None or self.invocation==invocation,'Owned invocation was replaced')
        if self.invocation is None:
            # Persist before the first potentially mutating cleanup operation.
            self.record(dict(kind='owned_invocation',unit=self.unit,invocation_id=invocation,
                             group=self.group_name,exec_argv=list(self.argv),nonce=self.nonce))
            self.invocation=invocation
        return value

    def running_guard(self):
        value=self.authenticate(self.properties())
        for key,wanted in self.resources.items():require(value.get(key)==wanted,'Actual resource mismatch: '+key)
        return value

    def register_owner(self,owner):
        require(type(owner)is dict and type(owner.get('pid'))is int and owner['pid']>1
                and type(owner.get('start_ticks'))is str and owner['start_ticks'].isdigit()
                and owner.get('cgroup')=='0::'+self.group_name,'Exact captured owned process')
        service=self.authenticate(self.properties())
        current=self.probe.process(owner['pid'])
        require(current is not None and all(current[k]==owner[k]for k in('pid','start_ticks','cgroup')),
                'Actually live captured process before registration')
        require(not any((x['pid'],x['start_ticks'])==(owner['pid'],owner['start_ticks'])for x in self.owners),
                'Unique original process capture')
        self.record(dict(kind='owned_process',owner=copy.deepcopy(owner),invocation_id=self.invocation))
        self.owners.append(copy.deepcopy(owner))
        if service['MainPID']==str(owner['pid']):
            require(self.main_owner is None,'One original service main capture')
            self.main_owner=copy.deepcopy(owner)

    def sample(self):
        value=self.properties()
        if value['LoadState']=='not-found':
            require(not value['InvocationID']and not value['ControlGroup']and terminal(value),
                    'Absent service has no live work')
        else:self.authenticate(value)
        gone=[]
        for owner in self.owners:
            actual=self.probe.process(owner['pid'])
            if actual is not None and actual['start_ticks']==owner['start_ticks']:
                raise PendingDrain('Recorded process still present: '+str(owner['pid']))
            gone.append(dict(pid=owner['pid'],start_ticks=owner['start_ticks'],gone=True))
        tree=self.probe.group(self.group_name)
        require(tree['group']==self.group_name,'Exact inspected owned tree')
        if not terminal(value)or not tree['empty']:raise PendingDrain('Service or whole subtree still populated')
        # Ignore volatile accounting/resource fields while retaining actual exit.
        return dict(properties={k:value[k]for k in KEYS},tree=tree,owners=gone)

    def prove_drained(self):
        first=self.sample();self.probe.wait(.25);second=self.sample()
        if first!=second:raise PendingDrain('Two terminal/drain samples differ')
        self.samples=[first,second]
        return dict(owned_group_drained=True,samples=copy.deepcopy(self.samples),
                    invocation_id=self.invocation,actions=copy.deepcopy(self.actions))

    def act(self,action):
        # Resource failures do not remove ownership or prevent cleanup.
        self.authenticate(self.properties())
        event=dict(kind='cleanup_intent',action=action,unit=self.unit,invocation_id=self.invocation)
        self.record(event)
        result=dict(event,completed=False)
        self.actions.append(result)
        getattr(self.probe,action)(self.unit)
        result['completed']=True

    def cleanup(self,*,seconds=30):
        # Polling budget; each actual systemctl operation separately has a
        # ten-second cap. Root must also enforce its full cleanup wall budget.
        require(type(seconds)in(int,float)and 0<seconds<=60,'Bounded whole-group polling budget')
        deadline=self.probe.clock()+seconds
        errors=[]
        def complete(result):
            result['operation_errors']=list(errors)
            return result
        try:return self.prove_drained()
        except PendingDrain:pass
        for action in('stop','kill'):
            try:self.act(action)
            except (OSError,subprocess.SubprocessError)as error:
                errors.append(repr(error))
            except ValueError as error:
                # A command may race with normal exit. Accept only a fresh,
                # complete drain; changed ownership never permits another action.
                errors.append(repr(error))
                try:return complete(self.prove_drained())
                except PendingDrain:raise
            try:return complete(self.prove_drained())
            except PendingDrain:pass
        while self.probe.clock()<deadline:
            self.probe.wait(min(.25,max(0,deadline-self.probe.clock())))
            try:
                return complete(self.prove_drained())
            except PendingDrain:pass
        raise PendingDrain('Authenticated owned group did not fully drain before deadline')

    def finish(self,child_result):
        """Join existing child/pair result with whole-group cleanup evidence.

        Returned success is only lifecycle success; real scientific receipts,
        full predecessor admission and parent serial lock release are separate.
        Persist failure before returning; cleanup never promotes a failed run.
        """
        require(type(child_result)is dict and type(child_result.get('successful'))is bool,
                'Actual child/pair outcome required')
        outcome=dict(schema='nemotron4b_owned_phase_terminal_v1',unit=self.unit,
                     child_result=copy.deepcopy(child_result),successful=False,
                     owned_group_drained=False,runtime_admitted=False,serial_lock_release_authorized=False,
                     full_model_qualified=False,new_speedup_claim=False)
        try:
            outcome['drain']=self.cleanup();outcome['owned_group_drained']=True
            props=outcome['drain']['samples'][-1]['properties']
            service_success=(props['LoadState']=='loaded'and props['Result']=='success'
                             and props['ExecMainCode']=='1'and props['ExecMainStatus']=='0')
            outcome['cleanup_actions_required']=bool(self.actions)
            outcome['successful']=(child_result['successful']and service_success
                                   and self.main_owner is not None
                                   and props['ExecMainPID']==str(self.main_owner['pid'])
                                   and not self.actions and not outcome['drain'].get('operation_errors'))
        except BaseException as error:
            outcome['cleanup_failure']=dict(type=type(error).__name__,message=str(error)[:2048])
            self.record(copy.deepcopy(outcome))
            raise
        self.record(copy.deepcopy(outcome))
        return outcome

    def finish_supervisor(self,result,*,persist_result):
        """Concrete adapter for child supervisor v1 and rank-pair v1 receipts.

        Persist the complete original outcome, including binary output, before
        recording the combined terminal result. This never releases a lock.
        """
        try:
            require(callable(persist_result),'Durable original supervisor output writer')
            summary=supervisor_summary(result,self.nonce)
            reference=persist_result(result)
            require(type(reference)is dict and set(reference)=={'path','sha256'}
                    and type(reference['path'])is str and reference['path'].startswith('/')
                    and type(reference['sha256'])is str and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
                    'Original supervisor receipt must be durably referenced')
            summary['original_supervisor_result']=copy.deepcopy(reference)
        except BaseException as error:
            # Receipt/adapter failure must not strand a still-owned service.
            self.finish(dict(successful=False,supervisor_adapter_failure=dict(
                type=type(error).__name__,message=str(error)[:2048])))
            raise
        return self.finish(summary)
