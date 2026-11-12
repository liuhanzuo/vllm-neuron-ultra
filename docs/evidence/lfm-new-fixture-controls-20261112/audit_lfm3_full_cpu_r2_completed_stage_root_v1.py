"""Read completed r2 JSON with the unchanged scientific validators; no runtime import."""
import datetime
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
OUT = Path('/mnt/nvme/lfm3-nki05-full-cpu-r2-20261112')
DENIED = ('ctypes', '_ctypes', 'torch', 'torch_neuronx', 'torch_xla', 'numpy',
          'nki', 'neuronxcc', 'neuronx_distributed', 'neuronx_distributed_inference')


class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in DENIED:
            raise RuntimeError('Root saved-JSON audit blocks native import: ' + fullname)
        return None


def reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    stage = sys.argv[1]
    assert stage in ('new_fixture', 'candidate')
    destination = Path(sys.argv[2])
    assert destination.parent == ROOT / 'research' and not destination.exists()
    assert not any(name.split('.')[0] in DENIED for name in sys.modules)
    sys.meta_path.insert(0, NativeBlocker())
    sys.path.insert(0, str(ROOT))
    import lfm3_nki05_full_cpu_leaf_v3 as science
    import minicpm5_layer0_execution_protocol_v4 as protocol
    from research import descriptor_serial_lock_proof_v1 as descriptor

    host = SimpleNamespace(p=protocol, SELF=str(ROOT / 'run_lfm3_nki05_full_cpu_serial_v3.py'),
                           reference=reference)
    admission_ref = reference(OUT / 'admission.json')
    assert admission_ref['sha256'] == 'bcc175f2aeb21fdfb0fd4c0fa585336ef6b7996c3666b6596ffc4a78f7b026c5'
    admission = protocol.read(admission_ref['path'], admission_ref['sha256'])
    parent = protocol.read(OUT / 'worker-owner.json')
    values = science.matrices()
    summary = science.stage_complete(host, OUT, stage, admission_ref)
    plans = [row for row in science.schedule() if row['stage'] == stage]
    assert [ref['path'] for ref in summary['phases']] == [str(OUT / stage / (plan['phase'] + '-outcomes.json')) for plan in plans]
    rows = []
    cases = 0
    for plan, phase_ref in zip(plans, summary['phases']):
        phase = plan['phase']
        result = protocol.read(phase_ref['path'], phase_ref['sha256'])
        assert result['phase'] == phase
        outcomes = result['outcomes']
        assert [(row['begin'], row['end']) for row in outcomes] == [tuple(bound) for bound in plan['bounds']]
        for outcome in outcomes:
            begin, end = outcome['begin'], outcome['end']
            path = science.result_path(OUT, stage, phase, begin, end)
            assert outcome['stage'] == stage and outcome['phase'] == phase
            assert outcome['returncode'] == 0 and outcome['passed'] is True
            assert outcome['error'] is None and outcome['secondary_errors'] == []
            assert outcome['report'] == reference(path)
            report = protocol.read(path, outcome['report']['sha256'])
            science.validate_report(report, stage, phase, begin, end, values[phase][begin:end], admission_ref)
            old_path = science.result_path(OUT, 'original_fixture', phase, begin, end)
            old = protocol.read(old_path)
            assert report['checks'] == old['checks']
            launch = science.launch_path(OUT, stage, phase, begin, end)
            ack = protocol.read(launch)
            assert ack['pid'] == report['pid'] == outcome['pid']
            assert ack['start_ticks'] == report['start_ticks'] == outcome['start_ticks']
            assert ack['argv'] == science.command(host, admission_ref, stage, phase, begin, end)
            assert ack['parent'] == dict(parent, cgroup=parent['cgroup'].removeprefix('0::'))
            protocol.gone(ack)
            rows.append(dict(phase=phase, begin=begin, end=end, report=reference(path),
                             original_report=reference(old_path), launch_owner=reference(launch),
                             actual_child=dict(pid=ack['pid'], start_ticks=ack['start_ticks']),
                             actual_returncode=0, recorded_child_gone=True,
                             original_validator_passed=True, all_original_fixture_descriptors_identical=True))
            cases += end - begin
    assert len(rows) == 291 and cases == 3716
    lock = descriptor.prove(admission['outer_owner'], admission['lock_identity'], admission['retained_fd'],
                            proc=protocol.proc, open_binary=lambda path: open(path, 'rb'),
                            lstat=os.lstat, stat=os.stat, readlink=os.readlink)
    assert lock['passed'] is True
    loaded = sorted(name for name in sys.modules if name.split('.')[0] in DENIED)
    assert loaded == []
    sources = []
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).suffix == '.py' and Path(name).resolve().is_relative_to(ROOT):
            sources.append(reference(Path(name).resolve()))
    record = dict(schema='lfm3_full_cpu_r2_completed_stage_root_audit_v1',
                  observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), reviewer='/root',
                  stage=stage, admission=admission_ref, stage_summary=reference(OUT / (stage + '-complete.json')),
                  scientific_validator=reference(Path(science.__file__)), audit_source=reference(__file__),
                  source_sha256=sources, batches=len(rows), cases=cases, passed=True, rows=rows,
                  all_actual_child_exits_zero=True, all291_recorded_children_gone=True,
                  all_saved_ACK_argv_PID_start_parent_bindings_match=True,
                  all_original_scientific_report_validators_passed=True,
                  all_four_original_fixture_hash_descriptors_identical=True,
                  root_read_saved_JSON_only=True, native_import_blocker_installed=True,
                  loaded_native_names=loaded, original_serial_lock_still_held=True,
                  original_lock_observation=lock, full_CPU_gate_complete=False,
                  full_model_qualified=False, new_speedup_claim=False)
    with destination.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(dict(stage=stage, batches=len(rows), cases=cases, passed=True, audit=reference(destination))))


if __name__ == '__main__':
    main()
