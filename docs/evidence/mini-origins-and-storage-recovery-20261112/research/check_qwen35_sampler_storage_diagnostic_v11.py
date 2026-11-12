"""Source/mock checks only. No model, services, filesystem probes or native imports."""
import ast
import copy
import io as stdio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qwen35_sampler_runtime_contract_v10 as old_contract
import qwen35_sampler_runtime_contract_v11 as contract
import qwen35_sampler_runtime_io_v10 as old_io
import qwen35_sampler_runtime_io_v11 as new_io
import prepare_qwen35_sampler_v11_candidate_v1 as prep


PREFIX = 'actual storage floor/reservation failed'
USAGE = dict(classes={}, allocated_bytes=0, entries=0)
ROOT = str(contract.ROOT)
NVME = '/mnt/nvme'


def call(module, *, root_free, nvme_free, reserve=False, allocated=0,
         memory=None, storage_error=None):
    usage = dict(USAGE, allocated_bytes=allocated)
    calls = []
    memory = contract.LIMITS['memory_available_floor_bytes'] if memory is None else memory

    def statvfs(path):
        calls.append(('statvfs', str(path)))
        return SimpleNamespace(f_bavail={ROOT: root_free, NVME: nvme_free}[str(path)],
                               f_frsize=1, f_fsid={ROOT: 66306, NVME: 66304}[str(path)])

    def storage(spec):
        calls.append(('storage', spec))
        if storage_error:
            raise storage_error
        return copy.deepcopy(usage)

    def open_memory(path, *args, **kwargs):
        assert str(path) == '/proc/meminfo'
        calls.append(('memory', str(path)))
        return stdio.StringIO('MemAvailable: '+str(memory//1024)+' kB\n')

    with patch.object(module, 'storage', storage), patch.object(module.os, 'statvfs', statvfs), \
         patch.object(Path, 'open', open_memory), \
         patch.object(module.os, 'stat', side_effect=AssertionError('no additional device/path stat')):
        try:
            value = module.resources({'fixture': True}, reserve=reserve)
        except ValueError as error:
            return {'error': str(error)}, calls
        return value, calls


class StorageDiagnostic(unittest.TestCase):
    def defaults(self, **kw):
        return dict(root_free=contract.LIMITS['root_floor_bytes'],
                    nvme_free=contract.LIMITS['nvme_floor_bytes']+contract.LIMITS['total_allocated_output_bytes'], **kw)

    def failure(self, **kw):
        value, calls = call(new_io, **kw)
        self.assertTrue(value['error'].startswith(PREFIX+': '))
        return json.loads(value['error'].split(': ', 1)[1]), calls

    def test_root_below_floor_identifies_same_first_sample_and_stops(self):
        args=self.defaults(); args['root_free']-=1
        row, calls=self.failure(**args)
        self.assertEqual(row, dict(path=ROOT, device=dict(kind='statvfs_fsid', value=66306),
            available_bytes=args['root_free'], floor_bytes=contract.LIMITS['root_floor_bytes'],
            remaining_reservation_bytes=0, required_bytes=contract.LIMITS['root_floor_bytes'],
            shortfall_bytes=1, failed_constraints=['floor'], reserve=False))
        self.assertEqual([r for r in calls if r[0]=='statvfs'], [('statvfs', ROOT)])
        self.assertFalse(any(r[0]=='memory' for r in calls))

    def test_root_floor_equal_success_and_return_shape_unchanged(self):
        args=self.defaults()
        old, old_calls=call(old_io, **args); new, new_calls=call(new_io, **args)
        self.assertEqual(old, new); self.assertEqual(old_calls, new_calls)
        self.assertEqual(set(new), {'storage', 'available_disk_bytes', 'available_memory_bytes'})

    def test_nvme_below_floor_identifies_second_sample(self):
        args=self.defaults(); args['nvme_free']=contract.LIMITS['nvme_floor_bytes']-1
        row, calls=self.failure(**args)
        self.assertEqual(row['path'], NVME)
        self.assertEqual(row['device'], dict(kind='statvfs_fsid', value=66304))
        self.assertEqual(row['failed_constraints'], ['floor'])
        self.assertEqual(row['remaining_reservation_bytes'], 0)
        self.assertEqual([r for r in calls if r[0]=='statvfs'], [('statvfs', ROOT), ('statvfs', NVME)])

    def test_nvme_reservation_required_minus_one_and_exact_boundary(self):
        args=self.defaults(reserve=True, allocated=4096); args['nvme_free']-=4097
        row, _=self.failure(**args)
        self.assertEqual(row['remaining_reservation_bytes'], contract.LIMITS['total_allocated_output_bytes']-4096)
        self.assertEqual(row['failed_constraints'], ['reservation'])
        self.assertEqual(row['shortfall_bytes'], 1)
        args['nvme_free']+=1
        self.assertNotIn('error', call(new_io, **args)[0])

    def test_both_floor_and_reserved_space_failures_reported(self):
        args=self.defaults(reserve=True); args['nvme_free']=contract.LIMITS['nvme_floor_bytes']-17
        row, _=self.failure(**args)
        self.assertEqual(row['failed_constraints'], ['floor', 'reservation'])
        self.assertEqual(row['shortfall_bytes'], contract.LIMITS['total_allocated_output_bytes']+17)

    def test_root_never_reserves_output_even_when_reserve_true(self):
        args=self.defaults(reserve=True); args['root_free']-=1
        row, _=self.failure(**args)
        self.assertEqual(row['remaining_reservation_bytes'], 0)
        self.assertEqual(row['required_bytes'], contract.LIMITS['root_floor_bytes'])

    def test_no_reservation_when_reserve_false_or_output_budget_consumed(self):
        for reserve, allocated in [(False, 0), (True, contract.LIMITS['total_allocated_output_bytes']),
                                   (True, contract.LIMITS['total_allocated_output_bytes']+1)]:
            args=self.defaults(reserve=reserve, allocated=allocated)
            args['nvme_free']=contract.LIMITS['nvme_floor_bytes']
            with self.subTest(reserve=reserve, allocated=allocated):
                self.assertNotIn('error', call(new_io, **args)[0])

    def test_memory_floor_failure_unchanged_after_two_disk_samples(self):
        args=self.defaults(memory=contract.LIMITS['memory_available_floor_bytes']-1024)
        self.assertEqual(call(old_io, **args), call(new_io, **args))
        self.assertEqual(call(new_io, **args)[0]['error'], 'MemAvailable floor failed')

    def test_storage_failure_preserved_before_any_disk_sample(self):
        args=self.defaults(storage_error=ValueError('per-class allocated output limit exceeded'))
        self.assertEqual(call(old_io, **args), call(new_io, **args))
        self.assertEqual(len(call(new_io, **args)[1]), 1)

    def test_byte_arithmetic_uses_same_bavail_and_frsize(self):
        args=self.defaults(); samples=[]
        original=SimpleNamespace(f_bavail=1000, f_frsize=4096, f_fsid=77)
        with patch.object(new_io, 'storage', return_value=USAGE), \
             patch.object(new_io.os, 'statvfs', side_effect=lambda p: samples.append(p) or original):
            with self.assertRaises(ValueError) as ctx:
                new_io.resources({'fixture': True})
        row=json.loads(str(ctx.exception).split(': ', 1)[1])
        self.assertEqual(row['available_bytes'], 4096000)
        self.assertEqual(row['device']['value'], 77)
        self.assertEqual(samples, [contract.ROOT])

    def test_old_new_decisions_and_observation_order_match_600_boundary_cases(self):
        count=0
        root_floor=contract.LIMITS['root_floor_bytes']; floor=contract.LIMITS['nvme_floor_bytes']
        cap=contract.LIMITS['total_allocated_output_bytes']
        for reserve in (False, True):
            for allocated in (0, 1, cap//2, cap-1, cap, cap+1):
                remaining=max(0, cap-allocated) if reserve else 0
                for root_delta in (-1, 0, 1, 4096, -4096):
                    for nvme_free in (floor-1, floor, floor+1, floor+remaining-1, floor+remaining,
                                      floor+remaining+1, floor-4096, floor+4096, floor+cap, floor+cap+1):
                        kwargs=dict(root_free=root_floor+root_delta, nvme_free=nvme_free,
                                    reserve=reserve, allocated=allocated)
                        old, oc=call(old_io, **kwargs); new, nc=call(new_io, **kwargs)
                        self.assertEqual(oc, nc)
                        self.assertEqual('error' in old, 'error' in new)
                        if 'error' in old:
                            self.assertTrue(new['error'].startswith(old['error']))
                        else:
                            self.assertEqual(old, new)
                        count+=1
        self.assertEqual(count, 600)

    def test_only_runtime_resources_body_changes_after_version_normalization(self):
        for name in prep.NAMES:
            old_name=name.replace('_v11.py', '_v10.py')
            old=(contract.Q/old_name).read_text()
            new=(contract.Q/name).read_text().replace('_v11', '_v10')
            if name.endswith('_io_v11.py'):
                ot=ast.parse(old); nt=ast.parse(new)
                olddefs={node.name:ast.dump(node, include_attributes=False) for node in ot.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                newdefs={node.name:ast.dump(node, include_attributes=False) for node in nt.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                self.assertEqual(set(olddefs), set(newdefs))
                self.assertEqual([k for k in olddefs if olddefs[k]!=newdefs[k]], ['resources'])
                ot.body=[n for n in ot.body if not isinstance(n, ast.FunctionDef) or n.name!='resources']
                nt.body=[n for n in nt.body if not isinstance(n, ast.FunctionDef) or n.name!='resources']
                self.assertEqual(ast.dump(ot), ast.dump(nt))
            else:
                self.assertEqual(old, new)
        self.assertEqual(old_contract.LIMITS, contract.LIMITS)

    def test_all_short_ipc_paths_remain_exactly_the_v10_paths(self):
        for run in ('r8-v10-20261112', 'r9-v11-20261112', 'a'*23):
            spec=dict(run_name=run, output_root='/mnt/nvme/neuron-work/qwen35-sampler-pair-'+run)
            self.assertEqual(old_contract.ipc_paths(spec), contract.ipc_paths(spec))


class Preparation(unittest.TestCase):
    def fixture(self):
        future=dict(run_name='r9-v11-20261112', output_root='/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112',
            running_handoff_path=str(contract.Q/'v11-mock-running.json'), overlay_path=str(contract.Q/'v11-mock-overlay.json'),
            spec_path=str(contract.Q/'v11-mock-spec.json'), review_candidate_path=str(contract.Q/'v11-mock-review.json'),
            successor_sources=dict(path=str(contract.Q/'v9-mock-manifest.json'), sha256='9'*64))
        oldref=dict(path=str(contract.Q/'v10-mock-manifest.json'), sha256='a'*64)
        ref=dict(path=str(contract.Q/'v11-mock-manifest.json'), sha256='b'*64)
        pins={str(contract.Q/'retained-v9.py'):'c'*64, str(contract.Q/'retained-v10.py'):'d'*64}
        manifest=dict(schema='qwen35_sampler_storage_diagnostic_v11_sources_v1', previous_source_manifest=oldref,
            source_files={name:dict(path=str(contract.Q/name), sha256='e'*64) for name in prep.NAMES}, limits=copy.deepcopy(contract.LIMITS))
        saved=dict(value=manifest, identity=dict(realpath=ref['path'], device=1, inode=1, size=100, mtime_ns=1, ctime_ns=1))
        spec=dict(run_name=future['run_name'], output_root=future['output_root'], source_sha256=copy.deepcopy(pins), input_extension={})
        overlay=dict(source_extension=copy.deepcopy(pins), input_extension={})
        review=dict(passed=False, launch_authorized=False)
        result=dict(artifacts={future['spec_path']:spec, future['overlay_path']:overlay, future['review_candidate_path']:review}, evidence={})
        return future, oldref, ref, saved, result, pins

    def run_fixture(self, future, oldref, ref, saved, result):
        reads=[]
        with patch.object(prep.previous, 'prepare', return_value=result) as old:
            value=prep.prepare(future, oldref, ref, lambda r: reads.append(r) or saved, lambda:None)
        old.assert_called_once()
        self.assertEqual(old.call_args.args[:2], (future, oldref))
        self.assertEqual(reads, [ref])
        return value

    def test_old_pins_retained_six_added_and_candidate_stays_unapproved(self):
        future, oldref, ref, saved, result, pins=self.fixture()
        value=self.run_fixture(future, oldref, ref, saved, result)
        spec=value['artifacts'][future['spec_path']]; review=value['artifacts'][future['review_candidate_path']]
        self.assertTrue(all(spec['source_sha256'][p]==s for p,s in pins.items()))
        self.assertEqual(len(spec['source_sha256']), len(pins)+6)
        self.assertFalse(review['passed']); self.assertFalse(review['launch_authorized'])
        self.assertEqual(spec['input_extension'], value['artifacts'][future['overlay_path']]['input_extension'])

    def test_changed_limit_rejected_before_previous_preparation(self):
        future, oldref, ref, saved, result, _=self.fixture()
        saved['value']['limits']['root_floor_bytes']-=1
        with patch.object(prep.previous, 'prepare', side_effect=AssertionError('must not prepare')):
            with self.assertRaisesRegex(ValueError, 'unchanged limits'):
                prep.prepare(future, oldref, ref, lambda r:saved, lambda:None)

    def test_wrong_previous_manifest_rejected(self):
        future, oldref, ref, saved, result, _=self.fixture()
        saved['value']['previous_source_manifest']=dict(oldref, sha256='f'*64)
        with self.assertRaisesRegex(ValueError, 'Exact additive'):
            prep.prepare(future, oldref, ref, lambda r:saved, lambda:None)

    def test_bad_source_identity_rejected(self):
        future, oldref, ref, saved, result, _=self.fixture()
        saved['identity']['realpath']='/tmp/wrong.json'
        with self.assertRaisesRegex(ValueError, 'Stable full'):
            prep.prepare(future, oldref, ref, lambda r:saved, lambda:None)

    def test_source_pin_conflict_rejected(self):
        future, oldref, ref, saved, result, _=self.fixture()
        result['artifacts'][future['spec_path']]['source_sha256'][str(contract.Q/prep.NAMES[0])]='f'*64
        with patch.object(prep.previous, 'prepare', return_value=result), self.assertRaisesRegex(ValueError, 'conflict'):
            prep.prepare(future, oldref, ref, lambda r:saved, lambda:None)

    def test_missing_runtime_file_rejected(self):
        future, oldref, ref, saved, result, _=self.fixture()
        del saved['value']['source_files'][prep.NAMES[0]]
        with self.assertRaisesRegex(ValueError, 'Exact additive'):
            prep.prepare(future, oldref, ref, lambda r:saved, lambda:None)

    def test_input_output_collision_rejected_before_reader(self):
        future, oldref, ref, saved, result, _=self.fixture()
        future['spec_path']=ref['path']
        with self.assertRaisesRegex(ValueError, 'distinct'):
            prep.prepare(future, oldref, ref, lambda r: (_ for _ in ()).throw(AssertionError('must not read')), lambda:None)


if __name__ == '__main__':
    unittest.main()
