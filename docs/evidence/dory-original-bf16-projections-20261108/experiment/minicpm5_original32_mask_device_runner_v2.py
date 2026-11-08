"""Finite original32 device runner v2; includes the failed r1 predecessor."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

import minicpm5_original88_decode_mask_analysis_v1 as masks

p, five, post, same, ROOT = masks.p, masks.five, masks.post, masks.same, masks.ROOT
SELF = str(Path(__file__).resolve())
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
WORKER = str(ROOT/'minicpm5_original32_mask_device_worker_v1.py')
CONTEXT = str(ROOT/'minicpm5_original32_device_context_v1.py')
CAP = 128 * 2**20
PROCESS_KEYS = ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')
FINAL_INPUT_PLAN = masks.ref('minicpm5-original88-python-trace-inputs-20261107-v1.json', '17849dbee01a34f232a72432919fb7d042d7052797fd1d25290382703ab6c196')
SCIENCE_INPUT_PLAN = masks.ref('minicpm5-layer0-runtime-inputs-20261107-v1.json', 'b8d84596f9fa6b54c0fa599af6ad976f051e47544a2d8c76027136efe2254800')
MASK_EXIT = masks.ref('minicpm5-original88-decode-mask-root-exit-r1-20261107-v1.json', '580dcdff25da4a194d5c797303a076f8faf0d64ae86e5f3e301a5f2355684f21')
MASK_REVIEW = masks.ref('minicpm5-original88-decode-mask-independent-metadata-review-20261107-v1.json', '2eaee60f4513053af1358e154ac67ad3e004e4b173416661275824e986bb2eb7')
CALL4_EXIT = masks.ref('minicpm5-call4-mask-root-exit-r1-20261107-v1.json', 'd11c23e3f000425ba736a549f5a8580dabe2d37ceada6c2b840cb6114aabf5ae')
CALL4_REVIEW = masks.ref('minicpm5-call4-mask-independent-result-review-20261107-v1.json', '3dc4440c061000a0fea5b6567cdddec16f28932b60d0c129616a22ce7d1719ca')
FAILED_R1_EXIT = masks.ref('minicpm5-original32-mask-device-root-exit-r1-20261107-v1.json', 'f3be18933039c2150007b5ab7dde3bd15b188d8512b979ba6481f09f7ae71615')
PROPS = dict(MemoryMax=str(26*2**30), MemoryHigh=str(24*2**30), MemorySwapMax='0', TasksMax='128',
             AllowedCPUs='6-7', CPUAffinity='6-7', Nice='19', KillMode='control-group', RemainAfterExit='yes',
             SendSIGKILL='yes', WorkingDirectory=str(ROOT), LimitFSIZE=str(16*2**20),
             RuntimeMaxSec='3600', TimeoutStopSec='15')


def reference(path):
    return dict(path=str(path), sha256=p.sha(path))


def process(value):
    return post.process({k:value[k] for k in PROCESS_KEYS})


def check_plan_extension(plan, original):
    allowed = {'schema', 'output_root', 'source_sha256', 'inputs', 'interpreters', 'device_artifact', 'original_plan'}
    same({k:v for k,v in plan.items() if k not in allowed},
         {k:v for k,v in original.items() if k not in allowed}, 'Original scientific and helper fields retained')
    p.require(plan['schema'] == 'minicpm5_original32_device_runtime_inputs_v1', 'New device context schema')
    for name in ('source_sha256', 'inputs'):
        p.require(all(plan[name].get(k) == v for k,v in original[name].items()), 'No inherited '+name+' entry replaced')
    same(plan['interpreters'], dict(original['interpreters'], device=PYTHON), 'Only device interpreter added')
    same(plan['original_plan'], SCIENCE_INPUT_PLAN, 'Pinned original scientific plan')
    p.require(isinstance(plan['device_artifact'], dict) and plan['device_artifact'], 'Bound compiled artifact declaration')


def read_bundle(spec_ref, review_ref):
    spec = p.read_ref(spec_ref); review = p.read_ref(review_ref)
    p.require(spec['schema'] == 'minicpm5_original32_mask_device_execution_spec_v1', 'Finite original32 device spec')
    p.require(review['reviewer'] == '/root' and review['passed'] is True and not review['blockers'], 'Root source approval')
    same(review['spec'], spec_ref, 'Reviewed exact execution spec')
    p.require(spec['worker'] == WORKER and spec['runner'] == SELF, 'Finite executable paths')
    for path, sha in spec['source_sha256'].items():
        p.require(p.sha(path) == sha, 'Reviewed source changed: '+path)
    for path in (SELF, WORKER, CONTEXT):
        p.require(spec['source_sha256'].get(path) == p.sha(path), 'Entrypoint/context pinned')
    plan = p.read_ref(spec['plan'])
    same(plan['source_sha256'], spec['source_sha256'], 'Plan source closure')
    same(plan['output_root'], spec['output_root'], 'Fresh output binding')
    check_plan_extension(plan, p.read_ref(FINAL_INPUT_PLAN))
    root = Path(spec['output_root'])
    p.require(root.parent == ROOT and re.fullmatch(r'minicpm5-original32-strict-prior-device-[a-z0-9-]{1,100}', root.name), 'Fresh output scope')
    p.require(spec['output_cap_bytes'] == CAP and spec['root_floor_bytes'] == 28*2**30, 'Original floor/finite output cap')
    same(spec['resources'], PROPS, 'Fixed single-device resource profile')
    p.require(re.fullmatch(r'trn1-mini-original32-mask-[a-z0-9-]{1,80}', spec['unit']), 'Named finite device job')
    return spec, plan


def usage(root, floor):
    total = 0; count = 0
    for entry in Path(root).rglob('*'):
        info = entry.lstat(); p.require(not stat.S_ISLNK(info.st_mode), 'Output symlink')
        if stat.S_ISDIR(info.st_mode): continue
        p.require(stat.S_ISREG(info.st_mode) and info.st_uid == 1000 and info.st_nlink == 1, 'Regular owned output')
        total += max(info.st_size, info.st_blocks*512); count += 1
    p.require(total <= CAP and count <= 128, 'Finite output tree cap')
    fs = os.statvfs(ROOT); p.require(fs.f_bavail*fs.f_frsize >= floor, 'Original root floor')
    return dict(allocated_bytes=total, files=count)


def env(attempt):
    return dict(PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',
                HOME='/home/ubuntu', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
                MINI_LAYER0_ATTEMPT=attempt, HF_HOME='/mnt/nvme/hf', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                NEURON_RT_VISIBLE_CORES='0-1', NEURON_COMPILE_CACHE_URL='/mnt/nvme/neuron-cache',
                TOKENIZERS_PARALLELISM='false')


def argv(admission_ref):
    return [PYTHON, '-B', SELF, '--mode', 'worker', '--admission', admission_ref['path'],
            '--admission-sha256', admission_ref['sha256']]


def service_command(spec, admission_ref, attempt):
    return ['systemd-run', '--user', '--quiet', '--unit='+spec['unit'], '--service-type=exec',
            *[item for key,value in PROPS.items() for item in ('-p', key+'='+value)],
            '-p', 'StandardOutput=append:'+str(Path(spec['output_root'])/'runtime.log'), '-p', 'StandardError=inherit',
            '/usr/bin/env', '-i', *[k+'='+v for k,v in env(attempt).items()], *argv(admission_ref)]


def validate_service_identity(spec, admission_ref, attempt, value, owner=None):
    p.require(value['LoadState'] == 'loaded' and p.invocation(value['InvocationID']), 'Identifiable current service')
    command = service_command(spec, admission_ref, attempt)
    same(re.findall(r'argv\[\]=(.*?)\s*;', value['ExecStart']),
         [' '.join(command[command.index('/usr/bin/env'):])], 'Exact device service command')
    if owner is not None:
        p.require(value['InvocationID'] == owner['invocation_id'] and value['ExecMainPID'] == str(owner['pid']), 'Same service owner')
    return value


def validate_service(spec, admission_ref, attempt, value, owner=None):
    for key, expected in PROPS.items():
        if key in ('RuntimeMaxSec', 'TimeoutStopSec'):
            p.require(p.low.duration_seconds(value[key.replace('Sec', 'USec')]) == int(expected), 'Service duration '+key)
        else: p.require(value[key] == expected, 'Actual service property '+key)
    validate_service_identity(spec, admission_ref, attempt, value, owner)
    p.require(value['ExecStopPost'] == '', 'No extra service cleanup')
    return value


def call4_predecessors():
    ended = p.read_ref(CALL4_EXIT); review = p.read_ref(CALL4_REVIEW)
    p.require(ended['reviewer'] == '/root' and ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Actual call4 root exit0')
    p.require(review['passed'] is True and not review['blockers'], 'Independent call4 result accepted')
    same(review['root_actual_exit_ref'], CALL4_EXIT, 'Reviewed exact call4 exit')
    live = p.read_ref(ended['root_live']); outputs = {key:p.read_ref(ref) for key,ref in ended['outputs'].items()}
    p.require(live['actual_identity_verified'] is True and live['reviewer'] == '/root', 'Observed call4 identity')
    same(live['owner'], ended['owner'], 'Call4 live/exit owner')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Call4 actual tool handle')
    same(ended['actual_execution_handle'], dict(kind='tool_session', session_id=11091), 'Fixed completed call4 session')
    outer = outputs['outer-result.json']; service = outputs['service-exit.json']; children = outputs['children-exits.json']; coordinator = outputs['coordinator-result.json']
    same(outer['owner'], ended['owner'], 'Call4 outer owner'); same(service['observed_by_outer'], ended['owner'], 'Call4 exit observer')
    p.require(outer['complete'] is True and service['successful'] is True, 'Call4 outer/service completed')
    same([service['properties'][key] for key in ('Result','ExecMainCode','ExecMainStatus')], ['success','1','0'], 'Call4 actual service exit0')
    p.require(p.terminal(service['properties']), 'Call4 terminal service record')
    same(service['properties']['InvocationID'], service['owner']['invocation_id'], 'Call4 invocation')
    same(service['properties']['ExecMainPID'], str(service['owner']['pid']), 'Call4 service PID')
    p.require(children['complete'] is children['successful'] is coordinator['complete'] is True, 'Call4 complete children')
    same(children['exit_codes'], [0,0], 'Call4 actual rank exits'); same(coordinator['rank_exits'], [0,0], 'Call4 coordinator rank exits')
    same(children['owners'], coordinator['rank_owners'], 'Call4 rank owner records')
    owners = [process(ended['owner']), process(service['owner'])]
    for rank in (0,1):
        result = outputs['rank'+str(rank)+'-result.json']
        same(result, coordinator['results'][rank], 'Call4 rank result')
        same(result['process'], children['owners'][rank], 'Call4 actual rank identity')
        p.require(result['complete'] is True and result['rank'] == rank and result['both_rank_control_gate_passed'] is True, 'Call4 completed control gate')
        p.require(result['process']['ppid'] == service['owner']['pid'] and result['process']['cgroup'] == service['owner']['cgroup'], 'Call4 direct rank owner')
        owners.append(process(result['process']))
    group = p.GROUP_PREFIX+service['owner']['unit']+'.service'
    p.require(ended['outer_coordinator_ranks_gone'] is True, 'Root observed call4 drain')
    for record in (ended['service_group_after_exit'], service['group']):
        p.require(record['group'] == group and record['absent'] is True, 'Call4 recorded group drain')
    return owners, dict(group=group, owner=service['owner'])


def drain_call4(prior):
    value = p.properties(prior['owner']['unit'])
    if value['LoadState'] != 'not-found':
        p.require(value['LoadState'] == 'loaded' and p.terminal(value), 'Prior call4 service actually terminal')
        same(value['InvocationID'], prior['owner']['invocation_id'], 'Prior call4 invocation unchanged')
        same(value['ExecMainPID'], str(prior['owner']['pid']), 'Prior call4 service PID unchanged')
    return dict(properties=value, group=p.empty_group(prior['group']))


def failed_r1_predecessors():
    ended = p.read_ref(FAILED_R1_EXIT); live = p.read_ref(ended['root_live'])
    p.require(ended['reviewer'] == '/root' and ended['actual_exit_observed'] is True and
              type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 1, 'Actual failed r1 exit1')
    p.require(live['reviewer'] == '/root' and live['actual_identity_verified'] is True, 'Actual failed r1 live identity')
    same(ended['owner'], live['owner'], 'Failed r1 live/exit owner')
    same(ended['actual_execution_handle'], dict(kind='tool_session', session_id=29622), 'Failed r1 actual tool session')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Failed r1 live/exit handle')
    admission = p.read_ref(ended['outputs']['admission.json']); service = p.read_ref(ended['outputs']['service-exit.json'])
    owner = p.read_ref(ended['outputs']['device-owner.json']); failure = p.read_ref(ended['outputs']['device-failure.json'])
    model_failure = p.read_ref(ended['outputs']['device/failure.json'])
    same(service['owner'], owner, 'Failed r1 service owner')
    same(service['observed_by_outer'], ended['owner'], 'Failed r1 independent service exit observer')
    same(admission['outer_owner'], ended['owner'], 'Failed r1 admission owner')
    same(failure['process'], process(owner), 'Failed r1 worker failure owner')
    p.require(service['successful'] is False and p.terminal(service['properties']), 'Failed r1 actual service terminal')
    same([service['properties'][key] for key in ('Result','ExecMainCode','ExecMainStatus')],
         ['exit-code','1','1'], 'Failed r1 actual service exit1')
    same(service['properties']['InvocationID'], owner['invocation_id'], 'Failed r1 invocation')
    same(service['properties']['ExecMainPID'], str(owner['pid']), 'Failed r1 service PID')
    same([failure['error_type'], failure['error'], model_failure['error_type'], model_failure['error']],
         ['RuntimeError','Input not admitted','RuntimeError','Input not admitted'], 'Retained input-admission failure')
    p.require(failure['self_exit_claimed'] is model_failure['self_exit_claimed'] is False, 'No self-certified failed exit')
    p.require(ended['completed_call_reports'] == ended['completed_arms'] == 0 and model_failure['completed_arms'] == [],
              'Failed r1 has no complete trajectory')
    p.require(ended['outer_and_device_gone'] is True, 'Root observed failed r1 process drain')
    group = p.GROUP_PREFIX+owner['unit']+'.service'
    for record in (ended['service_group_after_exit'], service['group']):
        p.require(record['group'] == group and record['absent'] is True, 'Failed r1 recorded group drain')
    return [process(ended['owner']), process(owner)], dict(group=group, owner=owner)


def drain_failed_r1(prior):
    value = p.properties(prior['owner']['unit'])
    if value['LoadState'] != 'not-found':
        p.require(value['LoadState'] == 'loaded' and p.terminal(value), 'Failed r1 service actually terminal')
        same(value['InvocationID'], prior['owner']['invocation_id'], 'Failed r1 invocation unchanged')
        same(value['ExecMainPID'], str(prior['owner']['pid']), 'Failed r1 service PID unchanged')
    return dict(properties=value, group=p.empty_group(prior['group']))


def check_prerequisites(spec):
    bound = five.load_finalized(spec['finalization'], spec['catalog'], spec['finalizer_exit'])
    owners, _ = masks.prerequisites(bound)
    ended = p.read_ref(MASK_EXIT); independent = p.read_ref(MASK_REVIEW)
    p.require(independent['passed'] is True and ended['reviewer'] == '/root' and ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Actual decode-mask reviewed exit')
    live = p.read_ref(ended['root_live']); result = p.read_ref(ended['outputs']['report'])
    same(live['owner'], ended['owner'], 'Decode-mask live/exit'); same(result['execution_owner'], ended['owner'], 'Decode-mask result owner')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'Decode-mask actual session')
    p.require(result['summary']['cpu_rows'] == result['summary']['current_also_allowed_in_prior_rows'] == 96, 'Actual96 mask finding')
    owners.append(process(ended['owner']))
    check_plan_extension(p.read_ref(spec['plan']), bound['plan'])
    more, prior = call4_predecessors(); failed_owners, failed_prior = failed_r1_predecessors()
    prior['failed_r1'] = failed_prior
    return bound, owners+more+failed_owners, prior


def wait_file(path, guard, seconds=30):
    end = time.monotonic()+seconds
    while not Path(path).exists():
        guard(); p.require(time.monotonic() < end, 'Parent identity acknowledgement timeout'); time.sleep(.05)
    return p.read(path)


def child_admission(admission_ref):
    admission = p.read_ref(admission_ref); spec, plan = read_bundle(admission['spec'], admission['review'])
    p.require(admission['schema'] == 'minicpm5_original32_mask_device_admission_v1', 'New device admission')
    same(admission['plan'], spec['plan'], 'Current run plan binding')
    p.require(admission['predecessors_checked'] is True, 'Parent checked all predecessors')
    p.lock_proof(admission['outer_owner'], admission['lock_identity'])
    return admission, spec, plan


def worker(admission_ref):
    a, spec, plan = child_admission(admission_ref); root = Path(spec['output_root'])
    lease = lambda:p.lock_proof(a['outer_owner'], a['lock_identity'])
    observed = wait_file(root/'device-owner.json', lease); actual = process(observed)
    same(actual, p.proc(os.getpid()), 'Parent observed actual device worker')
    same(observed['unit'], spec['unit'], 'Current device service')
    p.require(p.invocation(observed['invocation_id']), 'Parent observed device invocation')
    permit = dict(schema='minicpm5_original32_device_runtime_permit_v1', plan_sha256=p.digest(plan),
                  role='device', rank=-1, arm='device_trajectory', attempt_id=a['attempt_id'],
                  serial_exclusive=True, source_review_accepted=True, all_predecessors_drained=True,
                  uid=1000, **actual, python=PYTHON, environment=env(a['attempt_id']), output_root=spec['output_root'])
    def guard():
        lease(); p.clear_window(); same(p.proc(os.getpid()), actual, 'Device worker identity remains fixed')
        return permit
    try:
        from minicpm5_original32_device_context_v1 import DeviceContext
        context = DeviceContext(plan, p.digest(plan), permit, guard); context.verify_sources()
        import minicpm5_original32_mask_device_worker_v1 as implementation
        result = implementation.run(context, root/'device')
        p.require(result['complete'] is True, 'Finite device trajectory complete')
        p.once(root/'device-result.json', dict(result, process=actual, admission=admission_ref, self_exit_claimed=False))
    except BaseException as error:
        p.once(root/'device-failure.json', dict(process=actual, error_type=type(error).__name__, error=str(error), self_exit_claimed=False))
        raise


def outer(spec_ref, review_ref):
    spec, plan = read_bundle(spec_ref, review_ref); bound, owners, prior = check_prerequisites(spec)
    root = Path(spec['output_root']); p.require(not root.exists(), 'Fresh device output directory'); p.user_bus()
    fd, owner, lock = post.external.acquire_lock()
    launch_attempted = False; drained = False; service_owner = None; terminal = None; ar = None; a = None; failure = None
    def interrupted(signum, frame): raise InterruptedError('Outer signal '+str(signum))
    handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        p.clear_window(); p.active_heavy(); p.gone(bound['finalizer_owner']); p.predecessor_drain(bound['catalog'])
        for previous in owners: p.gone(previous)
        prior_drain = drain_call4(prior)
        failed_r1_drain = drain_failed_r1(prior['failed_r1'])
        p.require(p.properties(spec['unit'])['LoadState'] == 'not-found', 'Fresh device service name')
        root.mkdir(mode=0o700); usage(root, spec['root_floor_bytes'])
        a = dict(schema='minicpm5_original32_mask_device_admission_v1', spec=spec_ref, review=review_ref,
                 plan=spec['plan'], outer_owner=owner, lock_identity=lock, attempt_id=os.urandom(16).hex(),
                 predecessors_checked=True, later_owners=owners, call4_exit=CALL4_EXIT, call4_review=CALL4_REVIEW,
                 call4_fresh_group_drain=prior_drain, failed_r1_exit=FAILED_R1_EXIT,
                 failed_r1_fresh_group_drain=failed_r1_drain)
        p.once(root/'admission.json', a); ar = reference(root/'admission.json')
        launch_attempted = True
        p.command(service_command(spec, ar, a['attempt_id']), timeout=30)
        deadline = time.monotonic()+30
        while service_owner is None:
            value = validate_service(spec, ar, a['attempt_id'], p.properties(spec['unit']))
            pid = int(value['MainPID']); actual = p.proc(pid) if pid else None
            if actual:
                same(actual['argv'], argv(ar), 'Actual device worker argv')
                same(actual['cgroup'], '0::'+p.GROUP_PREFIX+spec['unit']+'.service', 'Dedicated device group')
                service_owner = dict(actual, unit=spec['unit'], invocation_id=value['InvocationID'])
                p.once(root/'device-owner.json', service_owner)
            else:
                p.require(not p.terminal(value) and time.monotonic() < deadline, 'No live device worker'); time.sleep(.05)
        deadline = time.monotonic()+3630
        while True:
            p.lock_proof(owner, lock); p.clear_window(); usage(root, spec['root_floor_bytes']); p.active_heavy((spec['unit'],))
            value = validate_service(spec, ar, a['attempt_id'], p.properties(spec['unit']), service_owner)
            if p.terminal(value): terminal = value; break
            p.require(time.monotonic() < deadline, 'Bounded device deadline'); time.sleep(1)
        p.gone(service_owner); drain = p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service'); drained = True
        succeeded = terminal['Result'] == 'success' and terminal['ExecMainCode'] == '1' and terminal['ExecMainStatus'] == '0'
        p.once(root/'service-exit.json', dict(owner=service_owner, properties=terminal, group=drain,
                                             observed_by_outer=owner, successful=succeeded))
        p.require(succeeded, 'Actual device job failed')
        result = p.read(root/'device-result.json'); p.require(result['complete'] is True, 'Complete finite device result')
        same(result['process'], process(service_owner), 'Device result actual identity'); same(result['admission'], ar, 'Device result admission')
        p.require(result['self_exit_claimed'] is False, 'Worker does not certify its own exit')
        p.once(root/'outer-result.json', dict(complete=True, owner=owner, service_exit=reference(root/'service-exit.json'),
                                             device_result=reference(root/'device-result.json'), usage=usage(root, spec['root_floor_bytes']),
                                             self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False))
        return reference(root/'outer-result.json')
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error), owner=owner, self_exit_claimed=False)
        if root.exists(): p.once(root/'outer-failure.json', failure)
        raise
    finally:
        try:
            if launch_attempted and not drained:
                value = p.properties(spec['unit'])
                if value['LoadState'] != 'not-found':
                    validate_service_identity(spec, ar, a['attempt_id'], value, service_owner)
                    p.command(['systemctl', '--user', 'stop', spec['unit']], timeout=30)
                    group = p.empty_group(p.GROUP_PREFIX+spec['unit']+'.service'); after = p.properties(spec['unit'])
                    same(after['InvocationID'], value['InvocationID'], 'Cleanup retained same invocation')
                    p.require(p.terminal(after), 'Cleanup actual terminal')
                    p.once(root/'cleanup.json', dict(before=value, after=after, group=group, owner=service_owner,
                                                     actual_stop_completed=True, outer_failure=failure))
        finally:
            post.external.release_lock(fd)
            for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(); cli.add_argument('--mode', choices=['outer', 'worker'], required=True)
    for key in ('spec', 'review', 'admission'):
        cli.add_argument('--'+key); cli.add_argument('--'+key+'-sha256')
    args = cli.parse_args(); p.require(sys.dont_write_bytecode and __debug__, 'python -B required')
    refs = {k:dict(path=getattr(args,k), sha256=getattr(args,k+'_sha256')) for k in ('spec','review','admission')}
    if args.mode == 'outer': print(json.dumps(outer(refs['spec'], refs['review']), indent=2))
    else: worker(refs['admission'])
