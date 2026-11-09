"""Release only the already failed r9 audit with reviewed filename extension.

No frozen source is edited. Source validation and all original budget/drain
checks remain; only this failed recovery process uses the new strict inventory.
The durable pre-stop snapshot forces external.audit to remain unsuccessful.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(Q), str(R)]
import nemotron9b_package_protocol_v18 as p
import nemotron9b_package_external_v18 as external
from nemotron9b_package_controller_v18 import acquire, release
import nemotron9b_package_storage_inventory_candidate_v6 as inventory
import nemotron9b_r4_terminal_metadata_v2 as t
from dory_after_nemotron_r9_v1 import SPEC, REVIEW, CONTROL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true', required=True)
    parser.add_argument('--release-failed', action='store_true', required=True)
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256', 'extension-review', 'extension-review-sha256'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    s = dict(path=a.spec, sha256=a.spec_sha256)
    r = dict(path=a.review, sha256=a.review_sha256)
    e = dict(path=a.extension_review, sha256=a.extension_review_sha256)
    p.require(s == SPEC and r == REVIEW, 'Only original failed r9')
    extension = t.load(e)
    p.require(extension['reviewer'] == '/root' and extension['passed'] is True
            and extension['failure_release_only'] is True and extension['full_model_qualified'] is False
            and extension['original_spec'] == SPEC and extension['original_review'] == REVIEW,
            'Exact independently reviewed failed-only recovery')
    for name, digest in extension['source_sha256'].items():
        p.require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, 'Pinned recovery source')
    p.require(str(Path(__file__).resolve()) in extension['source_sha256']
            and str(Path(inventory.__file__).resolve()) in extension['source_sha256'], 'Explicit recovery closure')
    spec = t.load(SPEC)
    p.low.user_bus()
    fd, owner, lock = acquire()
    try:
        intent = p.read(CONTROL/'external-launch-intent.json')
        exited = t.load(extension['original_actual_exit'])
        p.require(t.validate_actual_exit(exited, dict(owner=intent['external_owner']), SPEC, REVIEW, 69950) == 1,
                  'Actual failed external tool exit')
        p.low.gone(intent['external_owner'])
        checks = t.load(extension['source_checks'])
        check_exit = t.load(extension['source_check_exit'])
        p.require(checks['complete'] is True and checks['source_sha256'] == extension['source_sha256']
                and check_exit['owner'] == checks['owner'] and check_exit['exit_code'] == 0
                and check_exit['actual_external_exit_observed'] is True, 'Real completed source check')
        p.low.gone(checks['owner'])
        failed = p.read(CONTROL/'failed-terminal.json')
        p.require(failed['passed'] is False and failed['all_started_groups_physically_drained'] is True
                and (CONTROL/'external-pre-stop.json').exists() and not (CONTROL/'terminal.json').exists()
                and not (CONTROL/'root-release.json').exists(), 'Original failure and durable pre-stop; no success possible')
        for stage in ('controller', *p.STAGES):
            p.require(p.low.terminal(p.low.properties(spec['units'][stage])), 'Already terminal original service')
            p.low.empty_group(p.group(spec, stage))
        p.once(CONTROL/'failed-release-extension-v1-intent.json', dict(owner=owner, lock=lock,
            extension_review=e, original_actual_exit=extension['original_actual_exit'],
            original_failed_terminal=p.ref(CONTROL/'failed-terminal.json'),
            original_pre_stop=p.ref(CONTROL/'external-pre-stop.json'),
            dependency_substitution='protocol_v18.output_inventory = reviewed candidate_v6 only in failed recovery',
            frozen_sources_unchanged=True, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(dict(owner=owner, extension_review=e)), flush=True)
    finally:
        release(fd)
    original_bundle = p.bundle
    def reviewed_bundle(spec_ref, review_ref):
        result = original_bundle(spec_ref, review_ref)
        p.output_inventory = inventory
        return result
    p.bundle = reviewed_bundle
    result = external.audit(SPEC, REVIEW, release_failed=True)
    p.require(result['failed'] is True and result['successful'] is False
            and result['model_qualified'] is False and result['explicit_failed_release'] is True,
            'Release original failure only; never qualify or restart')
    p.once(CONTROL/'failed-release-extension-v1-result.json', dict(owner=owner, extension_review=e,
        release=p.ref(CONTROL/'root-release.json'), original_failure_preserved=True,
        full_model_qualified=False, new_speedup_claim=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':main()
