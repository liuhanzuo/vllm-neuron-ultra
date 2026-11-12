"""Compose one unreviewed snapshot plan from the exact idle124-row checkpoint.

Metadata/source-only. No runtime probes, locks, payloads, or automatic launch.
"""
import argparse
import copy
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import prepare_minicpm5_sdk310_deb_plan_v4 as previous
import minicpm5_sdk310_signed_index_snapshot_v1 as snapshot
c=snapshot.c
MANIFEST=dict(path=str(Q/'minicpm5-sdk310-signed-index-snapshot-source-manifest-20261112-v1.json'),sha256='291891b5fc69c5decf263f7436066898e1796a0d868445bc42e55b3f4b12e7c7')
OLD_ACCEPTANCE=dict(path=str(Q/'minicpm5-sdk310-deb-retirement-root-source-review-20261112-v2.json'),sha256='71e2597078f69eed223c544a6d18a99f761ad7c2b25926494682f8a3d39b5872')
PREVIOUS_BUILDER=dict(path=str(Q/'prepare_minicpm5_sdk310_deb_plan_v4.py'),sha256='fb9131fc64f690cdf12b871af2c42adf23a4dcb103f10219d6ace92b374f8ca4')


def build(read,source_review_ref,output_root):
    previous.reference(source_review_ref)
    manifest=read(MANIFEST);accepted=read(source_review_ref);old=read(snapshot.OLD_MANIFEST)
    c.require(accepted['reviewer']=='/root' and accepted['passed'] is True and accepted['blockers']==[]
              and previous.same_ref(accepted['manifest'],MANIFEST) and accepted['source_sha256']==manifest['source_sha256'],
              'Exact root source acceptance of all100 snapshot sources')
    c.require(manifest['complete_retained_source_count']==len(manifest['source_sha256'])==100
              and manifest['previous_source_count']==98 and manifest['previous_manifest']==snapshot.OLD_MANIFEST
              and all(manifest['source_sha256'].get(p)==h for p,h in old['source_sha256'].items())
              and set(manifest['source_sha256'])-set(old['source_sha256'])=={str(Q/name)for name in snapshot.NEW_NAMES},
              'Complete original98 pins plus exactly two independent snapshot sources')
    c.require(manifest['budget']==c.INVENTORY and manifest['source_assessment']==snapshot.ASSESSMENT
              and manifest['catalog']==snapshot.CATALOG,'Original budget and exact snapshot/catalog binding')
    root=Path(output_root)
    c.require(root.parent==Path('/mnt/nvme') and root.name.startswith('dory-serial-inventory-minicpm5-sdk310-index-snapshot-'), 'Fresh separate immutable snapshot namespace')
    catalog=read(snapshot.CATALOG);c.require(len(catalog['runs'])==124,'Exact complete124 history')
    predecessor=catalog['runs'][-1]['terminal'];receipt=read(predecessor)
    c.require(receipt['actual_tool']==dict(kind='tool_session',session_id=33030,terminal_chunk='f3d394',exit_code=1)
              and receipt['owner_gone'] is True and receipt['output_directory_created'] is False
              and receipt['child_started'] is False and receipt['package_download_started'] is False,'Genuine drained preoutput failure is immediate predecessor')
    plan=previous.build(read,snapshot.CATALOG,old['actual_scope'],predecessor,'download',output_root,
                        retirement_acceptance_ref=OLD_ACCEPTANCE)
    for key in ('archive_inputs','replay_from','producer_actual_exit','producer_recorder_actual_exit'):plan.pop(key,None)
    plan.update(work_kind=snapshot.WORK_KIND,archive_mode='snapshot',snapshot_assessment=snapshot.ASSESSMENT,
                source_sha256=copy.deepcopy(manifest['source_sha256']),source_manifest=MANIFEST,
                source_only_acceptance=copy.deepcopy(source_review_ref),
                snapshot_manifest_output=str(root/'snapshot-manifest.json'),snapshot_plaintext_directory=str(root/'indexes'),
                snapshot_prior_builder=PREVIOUS_BUILDER,snapshot_original_inputs=manifest['original_inputs'],
                root_plan_review_required=True,fresh_physical_checks_required=True,runtime_admitted=False)
    c.require(len(plan['predecessor_owners'])==2811 and len(plan['predecessor_services'])==1455,
              'Complete retained historical observations including both failed Mini attempts and all874 r2 children')
    return plan


def main():
    parser=argparse.ArgumentParser()
    for name in ('source-review','source-review-sha256','output-root','destination'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();c.require(sys.dont_write_bytecode and __debug__,'Builder -B and assertions')
    c.require(c.reference(PREVIOUS_BUILDER['path'],2*c.MIB)[0]['sha256']==PREVIOUS_BUILDER['sha256'],'Frozen prior complete-history composition')
    plan=build(c.load,dict(path=a.source_review,sha256=a.source_review_sha256),a.output_root)
    c.verify_sources(plan['source_sha256'])
    destination=Path(a.destination)
    c.require(destination.parent==Q and destination.suffix=='.json' and not destination.name.startswith('continuation-handoff-'),'New plan only; no handoff')
    plan['preparation_builder']=c.reference(Path(__file__).resolve(),2*c.MIB)[0]
    print(c.save(destination,plan))


if __name__=='__main__':main()
