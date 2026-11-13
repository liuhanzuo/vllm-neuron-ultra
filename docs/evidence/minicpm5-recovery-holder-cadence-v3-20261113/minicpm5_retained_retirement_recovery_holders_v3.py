"""Read-only privileged all-UID holder scan for the one fixed retirement map.

No unlink, signal, service control, lock acquisition or namespace mutation.
The exact caller's retained original lock is proved before and after both scans.
"""
import argparse
import os
from pathlib import Path
import sys
import json
import signal

Q=Path(__file__).resolve().parent
for directory in (Q,Q.parent):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_retirement_contract_v2 as c
import minicpm5_retained_retirement_recovery_scope_v3 as scope
import minicpm5_retained_retirement_recovery_v3 as recovery


def inspect(plan,owner,proc_root=Path('/proc')):
    """Fail closed on permission/read errors; only exited-process races pass."""
    keys={(r[k]['device'],r[k]['inode'])for r in plan['entries']for k in ('original','keeper')}
    parents={str(Path(r[k]['realpath']).parent)for r in plan['entries']for k in ('original','keeper')}
    target_root='/mnt/nvme/minicpm5-history-crossover-v1'
    rows=list(proc_root.iterdir());c.require(len(rows)<=8192,'Finite complete process namespace')
    counts=dict(processes=0,fd_entries=0,mapped_lines=0,exited_races=0);matches=[]
    for proc in sorted(rows):
        if not proc.name.isdigit():continue
        try:
            before=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            if int(proc.name)==owner['pid']:
                c.require(before==str(owner['start_ticks']),'Exact intended remover only may hold transaction descriptors')
                continue
            counts['processes']+=1
            with(proc/'cmdline').open('rb')as stream:argv=stream.read((1<<20)+1)
            c.require(len(argv)<=1<<20,'Bounded process argv')
            if target_root.encode()in argv:matches.append(dict(pid=int(proc.name),kind='argv'))
            fds=list((proc/'fd').iterdir());c.require(len(fds)<=65536,'Bounded complete fd directory')
            for fd in fds:
                try:
                    s=fd.stat();link=os.readlink(fd);counts['fd_entries']+=1
                    if(s.st_dev,s.st_ino)in keys or link in parents or link==target_root:
                        matches.append(dict(pid=int(proc.name),kind='fd',fd=fd.name))
                except(FileNotFoundError,ProcessLookupError):pass
            with(proc/'maps').open('r')as stream:maps=stream.read((16<<20)+1)
            c.require(len(maps)<=16<<20,'Bounded complete maps')
            for line in maps.splitlines():
                fields=line.split(None,5)
                if len(fields)<5:continue
                counts['mapped_lines']+=1;major,minor=fields[3].split(':')
                if(os.makedev(int(major,16),int(minor,16)),int(fields[4]))in keys:
                    matches.append(dict(pid=int(proc.name),kind='map'))
            c.require((proc/'stat').read_text().rsplit(')',1)[1].split()[19]==before,'PID identity changed during complete holder scan')
        except(FileNotFoundError,ProcessLookupError):counts['exited_races']+=1
    c.require(not matches,'A separate process holds or names the fixed retirement namespace: '+repr(matches[:16]))
    return dict(complete=True,all_uid_processes_checked=True,zero_other_argv_fd_map_holders=True,
        exact_remover_exclusion=dict(pid=owner['pid'],start_ticks=str(owner['start_ticks'])),counts=counts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--permit',required=True);parser.add_argument('--permit-sha256',required=True)
    parser.add_argument('--scope-nonce',required=True)
    args=parser.parse_args(argv);c.require(os.geteuid()==0 and sys.dont_write_bytecode and __debug__,'Privileged read-only -B inspector')
    sys.addaudithook(scope.forbid_helper_children)
    signal.signal(signal.SIGALRM,scope.deadline_expired);signal.setitimer(signal.ITIMER_REAL,scope.DEADLINE)
    import qwen35_sampler_runtime_io_v12 as io
    import minicpm5_retained_raw_transport_v1 as transport
    permit_ref=dict(path=args.permit,sha256=args.permit_sha256);permit=io.read(permit_ref)
    print(json.dumps(scope.start_ack(io,permit_ref,permit,args.scope_nonce,os.getpid())),flush=True)
    spec=io.read(permit['spec'],64<<20);c.validate(spec)
    recovery.validate_helper_permit(io,permit_ref,permit,spec)
    plan=io.read(c.PLAN,16<<20);transport.validate_plan(plan)
    before=io.identity(permit_ref['path']);samples=[]
    for _ in range(2):
        io.descriptor(permit['owner'],permit['retained_fd']);samples.append(inspect(plan,permit['owner']))
        c.require(io.identity(permit_ref['path'])==before,'Exact holder permit unchanged')
    io.descriptor(permit['owner'],permit['retained_fd'])
    print(json.dumps(dict(schema='minicpm5_retained_retirement_all_uid_holders_v2',permit=permit_ref,
        complete=True,lock_identity=c.LOCK,samples=samples,read_only=True)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
