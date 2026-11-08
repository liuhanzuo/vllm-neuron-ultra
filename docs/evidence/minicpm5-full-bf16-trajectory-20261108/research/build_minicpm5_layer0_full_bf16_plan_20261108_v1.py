"""Build a fresh finite plan/spec after source reviews; does not launch anything."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(path=str(path), sha256=digest(path))


def main():
    if not sys.dont_write_bytecode:
        raise RuntimeError('Use Python -B')
    sys.path.insert(0, str(ROOT))
    runner = importlib.import_module('minicpm5_original32_layer0_full_bf16_cpu_runner_v1')
    worker = importlib.import_module('minicpm5_original32_layer0_full_bf16_cpu_worker_v1')
    prior = runner.p.read_ref(runner.PREVIOUS_PLAN)
    previous_spec = runner.p.read_ref(runner.PREVIOUS_SPEC)
    plan = copy.deepcopy(prior)
    output = Path('/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-r1-20261108-v1')
    if output.exists():
        raise RuntimeError('Fresh experiment namespace required')
    plan['output_root'] = str(output)
    for path in (Path(runner.__file__), Path(worker.__file__), Q/'minicpm5_full_bf16_gather_v1.py'):
        if str(path) in plan['source_sha256']:
            raise RuntimeError('New entrypoint must not replace an inherited source')
        plan['source_sha256'][str(path)] = digest(path)
    refs = [runner.PREVIOUS_PLAN, runner.PREVIOUS_EXIT, runner.PREVIOUS_SUMMARY, runner.DESIGN, runner.GATHER_TERMINAL, runner.GATHER_EXIT]
    ended = runner.p.read_ref(runner.PREVIOUS_EXIT)
    for rank in (0, 1):
        rank_ref = ended['outputs'][f'rank{rank}_result']
        refs.append(rank_ref)
        result = runner.p.read_ref(rank_ref)
        arm = next(row for row in result['arm_reports'] if row['arm'] == 'strict_cpu_s0')
        refs.append(dict(path=arm['report'], sha256=arm['sha256']))
    anchors = runner.strict_anchor_refs()
    refs += [dict(path=row['path'], sha256=row['sha256']) for row in anchors]
    added = {}
    for reference in refs:
        path = Path(reference['path'])
        if path.suffix != '.json' or path.stat().st_size > 8 * 2**20:
            raise RuntimeError('Only bounded immutable JSON metadata may be added')
        if digest(path) != reference['sha256']:
            raise RuntimeError('Changed metadata input')
        if str(path) in plan['inputs']:
            if plan['inputs'][str(path)]['sha256'] != reference['sha256']:
                raise RuntimeError('Inherited input changed')
            continue
        row = dict(sha256=reference['sha256'], identity=runner.identity(path))
        plan['inputs'][str(path)] = row
        added[str(path)] = row
    plan['layer0_full_bf16_trajectory'] = dict(
        schema='minicpm5_original32_layer0_full_bf16_cpu_plan_v1', arms=runner.ARMS,
        predecessor_plan=runner.PREVIOUS_PLAN, predecessor_actual_exit=runner.PREVIOUS_EXIT,
        predecessor_summary=runner.PREVIOUS_SUMMARY, strict_anchor_reports=anchors,
        design=runner.DESIGN, per_rank_output_cap_bytes=runner.RANK_CAP,
        total_output_cap_bytes=runner.CAP, selected_tap_names=runner.SELECTED_TAPS)

    reads = set()

    def admitted_read(path, sha):
        entry = plan['inputs'].get(str(path))
        if entry is None or entry['sha256'] != sha or runner.identity(path) != entry['identity']:
            raise RuntimeError('Unadmitted/changed metadata read: ' + str(path))
        reads.add(str(path))
        return runner.p.read_ref(dict(path=str(path), sha256=sha))

    worker.validate_plan(plan, admitted_read)
    runner.check_plan_extension(plan, prior, added)
    spec = copy.deepcopy(previous_spec)
    spec.update(schema='minicpm5_original32_layer0_full_bf16_cpu_execution_spec_v1',
        runner=str(Path(runner.__file__)), worker=str(Path(worker.__file__)),
        output_root=str(output), source_sha256=plan['source_sha256'],
        output_cap_bytes=runner.CAP, root_floor_bytes=28 * 2**30, output_floor_bytes=8*2**30,
        resources=runner.PROPS, unit='trn1-mini-original32-layer0-full-bf16-r1-20261108-v1',
        master_port=29653, added_inputs=added,
        model_qualification_claim=False, automatic_full_gates_launch=False)
    plan_path = Q / 'minicpm5-layer0-full-bf16-cpu-input-plan-20261108-v1.json'
    spec_path = Q / 'minicpm5-layer0-full-bf16-cpu-execution-spec-20261108-v1.json'
    check_path = Q / 'minicpm5-layer0-full-bf16-cpu-plan-build-check-20261108-v1.json'
    if any(path.exists() for path in (plan_path, spec_path, check_path)):
        raise RuntimeError('Immutable plan/spec/check namespace already used')
    spec['plan'] = write(plan_path, plan)
    spec_ref = write(spec_path, spec)
    write(check_path, dict(schema='minicpm5_layer0_full_bf16_plan_build_check_v1',
        plan=spec['plan'], spec=spec_ref, source_count=len(plan['source_sha256']),
        input_count=len(plan['inputs']), added_input_count=len(added),
        worker_plan_reads_admitted=sorted(reads), worker_validator_passed=True,
        runner_extension_validator_passed=True, model_started=False))
    print(json.dumps(dict(plan=spec['plan'], spec=spec_ref,
                         sources=len(plan['source_sha256']), inputs=len(plan['inputs']),
                         added_inputs=len(added)), indent=2))


if __name__ == '__main__':
    main()
