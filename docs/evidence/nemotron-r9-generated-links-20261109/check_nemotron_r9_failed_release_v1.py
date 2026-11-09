"""Exclusive source/retained-file check for failed-only r9 recovery."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(Q), str(R)]
from nemotron9b_package_controller_v18 import acquire, release
import nemotron9b_package_protocol_v18 as p
import nemotron9b_r4_terminal_metadata_v2 as t
from dory_after_nemotron_r9_v1 import SPEC, REVIEW, CONTROL
from inspect_nemotron_r9_failed_v1 import save
import nemotron9b_compiler_dynamic_links_v2 as links
import nemotron9b_package_storage_inventory_candidate_v6 as inventory


def main():
    p.low.user_bus()
    fd, owner, lock = acquire()
    out = Q/'nemotron9b-r9-link-source-check-20261109-v1'
    try:
        p.require(not out.exists(), 'Fresh source check evidence')
        out.mkdir()
        save(out/'intent.json', dict(owner=owner, lock=lock, source=t.ref(__file__),
            native_started=False, actual_self_exit_claimed=False))
        receipts = []
        for record, chunk in (
            (Q/'nemotron9b-r9-failed-inspection-20261109-v1/intent.json', 'c9ae0b'),
            (Q/'nemotron9b-r9-recovery-source-inspection-20261109-v1.json', 'ecc1e4')):
            value = t.load(t.ref(record))
            p.low.gone(value['owner'])
            receipts.append(dict(intent=t.ref(record), owner=value['owner'], reviewer='/root',
                actual_external_exit_observed=True, session_id=None, exit_code=0, tool_chunk=chunk,
                synchronous_tool_exit=True))
        save(Q/'nemotron9b-r9-recovery-inspections-actual-exits-20261109-v1.json', receipts)
        spec = t.load(SPEC)
        pre = t.load(t.ref(CONTROL/'external-pre-stop.json'))
        for stage in ('controller', *p.STAGES):
            props = p.validate_after(spec, stage, SPEC, REVIEW, pre['rows'][stage], p.low.properties(spec['units'][stage]))
            p.require(p.low.terminal(props), 'Only after actual terminal')
            p.low.empty_group(p.group(spec, stage))
        p.low.active_heavy()
        import check_nemotron9b_compiler_dynamic_links_v2 as tests
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        p.require(result.wasSuccessful(), 'All new strict-path checks pass')
        observed = t.load(t.ref(Q/'nemotron9b-r9-failed-inspection-20261109-v1/compiler-links.json'))
        root = Path(spec['work_root'])/'compile-work'
        accepted = []
        for row in observed['rejected']:
            p.low.lock_proof(owner, lock)
            accepted.append(links.link_record(row['path'], root, hash_content=False))
        policy = t.load(spec['compiler_asset_policy'])
        work = Path(spec['work_root'])
        used = inventory.inventory(work, ipc_root=work/'temporary/tmp', compiler_root=root, compiler_assets=policy)
        p.require(used <= spec['disk_budget']['nvme_cap'], 'All original storage caps retained')
        sources = [Q/n for n in ('nemotron9b_compiler_dynamic_links_v2.py', 'nemotron9b_compiler_asset_links_v5.py',
            'nemotron9b_package_storage_inventory_candidate_v6.py', 'check_nemotron9b_compiler_dynamic_links_v2.py',
            'check_nemotron_r9_failed_release_v1.py', 'release_nemotron_r9_failed_v1.py')]
        sources += [R/n for n in ('nemotron9b_compiler_dynamic_links_v1.py','nemotron9b_compiler_asset_links_v4.py',
                                  'nemotron9b_package_storage_inventory_candidate_v5.py')]
        pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        for name, digest in pins.items():
            if name in spec['source_sha256']:
                p.require(digest == spec['source_sha256'][name], 'Frozen original source unchanged')
        save(out/'result.json', dict(complete=True, owner=owner, tests_run=result.testsRun,
            real_rejected_links=len(observed['rejected']), all_real_rejected_names_now_strictly_recognized=True,
            accepted_real_metadata=accepted, nvme_used=used, source_sha256=pins,
            original_spec=SPEC, original_review=REVIEW, actual_self_exit_claimed=False,
            original_failure_preserved=True, original_generated_numeric_payload_bytes_hashed=0,
            native_started=False, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(dict(owner=owner, result=str(out/'result.json'), tests_run=result.testsRun,
                              rejected=len(observed['rejected']), nvme_used=used)), flush=True)
    except BaseException as error:
        if out.exists():save(out/'failure.json', dict(owner=owner, error=repr(error)))
        raise
    finally:
        release(fd)


if __name__ == '__main__':main()
