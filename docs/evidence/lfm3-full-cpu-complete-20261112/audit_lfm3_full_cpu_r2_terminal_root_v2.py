"""Audit saved r2 results after root has observed the actual tool exit.

Uses both original scientific and independent metadata validators; no native
model, SDK, simulator, compiler, or full serial-runner imports are permitted.
"""
import datetime
import fcntl
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
OUT = Path('/mnt/nvme/lfm3-nki05-full-cpu-r2-20261112')
DENIED = {'ctypes', '_ctypes', 'torch', 'torch_neuronx', 'torch_xla', 'numpy',
          'nki', 'neuronxcc', 'neuronx_distributed', 'neuronx_distributed_inference'}


class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in DENIED:
            raise RuntimeError('Native import forbidden in saved-JSON audit: ' + fullname)
        return None


def validate_retired_service(current, ended, cleanup, worker, terminal):
    """Bind historical exit to the exact invocation, then require actual retirement."""
    def require(value, message):
        if not value:
            raise RuntimeError(message)
    unit = 'trn1-lfm3-nki05-full-cpu-r2-20261112'
    invocation = 'ea961c63d568400c803def561cf7b6cd'
    require(worker['unit'] == unit and worker['invocation_id'] == invocation,
            'Original worker service identity')
    saved = ended['properties']
    require(ended['successful'] is True and ended['owner'] == worker
            and saved['LoadState'] == 'loaded' and saved['Id'] == unit + '.service'
            and saved['InvocationID'] == invocation and saved['ExecMainPID'] == str(worker['pid'])
            and saved['Result'] == 'success' and saved['ExecMainCode'] == '1'
            and saved['ExecMainStatus'] == '0' and terminal(saved),
            'Historical actual exit zero remains bound to the original invocation')
    samples = cleanup['samples']
    require(cleanup['physically_drained'] is True and cleanup['owner'] == worker
            and cleanup['self_exit_claimed'] is False and len(samples) == 2
            and samples[0] == samples[1], 'Original stable final cleanup receipt')
    for sample in samples:
        require(sample['group']['group'] == worker['cgroup'].removeprefix('0::')
                and sample['group']['absent'] is True, 'Original owned cgroup absent')
    for props in [current] + [sample['properties'] for sample in samples]:
        require(props['Id'] == unit + '.service' and props['LoadState'] == 'not-found'
                and props['InvocationID'] == '' and props['ExecMainPID'] == '0'
                and props['MainPID'] == props['ControlPID'] == '0'
                and props['ActiveState'] == 'inactive' and props['SubState'] == 'dead'
                and props['Job'] == '' and props['ControlGroup'] == '' and terminal(props),
                'Exact service unloaded with no job, PID, invocation or cgroup')


def main():
    if sys.flags.optimize:
        raise RuntimeError('Optimized Python is forbidden for this audit')
    actual_exit, destination = map(Path, sys.argv[1:])
    assert actual_exit.parent == destination.parent == Q and not destination.exists()
    assert not DENIED.intersection(sys.modules)
    sys.meta_path.insert(0, NativeBlocker())
    sys.path.insert(0, str(ROOT))
    import minicpm5_layer0_execution_protocol_v4 as p
    import lfm3_nki05_full_cpu_leaf_v3 as science
    import lfm3_nki05_full_cpu_metadata_v1 as metadata

    def ref(path):
        return dict(path=str(path), sha256=hashlib.sha256(p.text(path)).hexdigest())

    host = SimpleNamespace(p=p, SELF=metadata.CPU_SELF, reference=ref)
    binding = dict(spec=metadata.CPU_SPEC, admission=metadata.CPU_ADMISSION,
                   actual_exit=ref(actual_exit))
    for key, name in dict(result='diagnostic-result.json', worker_result='worker-result.json',
                          service_exit='service-exit.json', outer_result='outer-result.json',
                          cleanup='cleanup-drained.json').items():
        binding[key] = ref(OUT / name)
    metadata.validate_binding(binding)
    result = p.read(binding['result']['path'], binding['result']['sha256'])
    assert science.audit_full_success(host, result) is True
    audit = metadata.audit(host, binding)
    assert len(audit['children']) == 874
    for owner in audit['owners'] + audit['children']:
        p.gone(owner)
    worker = audit['owners'][1]
    group = p.empty_group(worker['cgroup'].removeprefix('0::'))
    properties = p.properties(audit['unit'])
    ended = p.read(binding['service_exit']['path'], binding['service_exit']['sha256'])
    cleanup = p.read(binding['cleanup']['path'], binding['cleanup']['sha256'])
    validate_retired_service(properties, ended, cleanup, worker, p.terminal)
    second = dict(properties=p.properties(audit['unit']),
                  group=p.empty_group(worker['cgroup'].removeprefix('0::')))
    assert second == dict(properties=properties, group=group)
    live_matches = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            owner = p.proc(int(entry.name))
            if owner and metadata.CPU_SELF in owner['argv']:
                live_matches.append(owner)
    assert live_matches == []
    lock = ROOT / 'serial-enrollment.lock'
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == (66306, 8294421)
        assert (lock.stat().st_dev, lock.stat().st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    raw = Path('/proc/locks').read_text()
    target_rows = []
    for line in raw.splitlines():
        for token in line.split():
            parts = token.split(':')
            if len(parts) == 3:
                try:
                    ident = (os.makedev(int(parts[0], 16), int(parts[1], 16)), int(parts[2]))
                except ValueError:
                    continue
                if ident == (66306, 8294421):
                    target_rows.append(line)
    assert target_rows == []
    for owner in audit['owners'] + audit['children']:
        p.gone(owner)
    assert ref(actual_exit) == binding['actual_exit']
    loaded = sorted(name for name in sys.modules if name.split('.')[0] in DENIED)
    assert loaded == []
    audit.update(schema='lfm3_full_cpu_r2_terminal_root_audit_v2', reviewer='/root',
                 observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 original_science_audit_full_success=True, independent_metadata_audit=True,
                 all874_child_ACK_PID_start_argv_parent_exits_checked=True,
                 all876_recorded_processes_gone=True, live_runner_matches=live_matches,
                 physical_drain=dict(passed=True, worker_group=group, properties=properties,
                     samples=[dict(properties=properties, group=group), second],
                     saved_service_exit_properties=ended['properties'],
                     service_exit=binding['service_exit'], cleanup=binding['cleanup'],
                     service_retirement_verified=True),
                 serial_lock=dict(device=66306, inode=8294421, acquired_nonblocking=acquired,
                                  released=True, remaining_proc_lock_rows=target_rows),
                 model_result_reads_saved_JSON_only=True, native_import_blocker_installed=True,
                 loaded_native_names=loaded, audit_source=ref(__file__),
                 original_science_source=ref(science.__file__),
                 independent_metadata_source=ref(metadata.__file__))
    with destination.open('x') as stream:
        json.dump(audit, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, actual_tool=audit['actual_tool'], children=874,
                          cpu_cases=3716, cpu_full_passes=3, audit=ref(destination),
                          device_behavior_proven=False, full_model_qualified=False,
                          new_speedup_claim=False)))


if __name__ == '__main__':
    main()
