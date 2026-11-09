"""Persist concrete failed-only recovery source review under serial lock."""
import hashlib
import json
from pathlib import Path
import sys
Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(Q), str(R)]
from nemotron9b_package_controller_v18 import acquire, release
import nemotron9b_package_protocol_v18 as p
import nemotron9b_r4_terminal_metadata_v2 as t
from dory_after_nemotron_r9_v1 import SPEC, REVIEW
from inspect_nemotron_r9_failed_v1 import save


def main():
    fd, owner, lock = acquire()
    try:
        save(Q/'nemotron9b-r9-release-review-intent-20261109-v1.json', dict(owner=owner, lock=lock,
            source=t.ref(__file__), native_started=False, actual_self_exit_claimed=False))
        result_ref = t.ref(Q/'nemotron9b-r9-link-source-check-20261109-v1/result.json')
        checks = t.load(result_ref)
        p.require(checks['complete'] is True and checks['tests_run'] == 14 and checks['real_rejected_links'] == 32,
                  'All actual link source checks completed')
        p.low.gone(checks['owner'])
        receipt = save(Q/'nemotron9b-r9-link-source-check-actual-exit-20261109-v1.json', dict(
            reviewer='/root', owner=checks['owner'], result=result_ref, actual_external_exit_observed=True,
            session_id=None, synchronous_tool_exit=True, exit_code=0, tool_chunk='b63b91'))
        spec = t.load(SPEC)
        p.require(set(p.required_sources()) <= set(spec['source_sha256']), 'Frozen source discovery unchanged')
        for name, digest in checks['source_sha256'].items():
            p.require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, 'Exact checked extension source')
            if name in spec['source_sha256']:
                p.require(spec['source_sha256'][name] == digest, 'No frozen original source changed')
        review = dict(schema='nemotron_r9_failed_release_source_review_v1', reviewer='/root', passed=True,
            failure_release_only=True, full_model_qualified=False, new_speedup_claim=False,
            original_spec=SPEC, original_review=REVIEW,
            original_actual_exit=t.ref(Q/'nemotron9b-r9-launch-actual-exit-20261109-v1.json'),
            source_checks=result_ref, source_check_exit=receipt, source_sha256=checks['source_sha256'],
            checks=['Exact duplicated tiled_pf_transpose filename with matching repeated subgraph and bounded numeric suffixes.',
                    'Original no-follow directory handles, same-unit derived target, owned regular target, stability and size checks retained.',
                    'Original installed assets and numeric/debug routing retained. No original module globals or source file mutated.',
                    'Complete actual output inventory succeeds with original caps;32 rejected links verified as metadata, not full numerical validation.',
                    'Failed-only wrapper requires actual69950/exit1, original failed terminal and durable pre-stop; external audit cannot promote success.',
                    'Original frozen bundle/source and physical drain checks remain. No model/compiler restart.'],
            blockers=[], owner=owner, actual_self_exit_claimed=False)
        ref = save(Q/'nemotron9b-r9-failed-release-source-review-20261109-v1.json', review)
        print(json.dumps(dict(owner=owner, review=ref)), flush=True)
    finally:
        release(fd)


if __name__ == '__main__':main()
