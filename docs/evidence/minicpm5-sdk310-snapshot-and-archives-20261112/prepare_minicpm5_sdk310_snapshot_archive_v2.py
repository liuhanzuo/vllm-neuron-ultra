"""Source-only candidate composition; no default refs, writes or runtime launch.

Call in a separate preparation process. This helper is not a runtime dependency
and must never be imported into the 106-source serial namespace.
"""
import copy
import prepare_minicpm5_sdk310_deb_plan_v4 as old
import minicpm5_sdk310_snapshot_read_binding_v2 as binding
from minicpm5_sdk310_snapshot_read_binding_v2 import c


def build(read, *, snapshot_binding, status_observation, reader_source_manifest, reader_source_acceptance, **original):
    """Retain the complete old builder/history, then add only reviewed read refs.

    original contains the exact keyword arguments accepted by old.build:
    catalog_ref, scope_ref, predecessor_ref, mode, output_root, replay_from,
    producer_exit_ref, producer_recorder_exit_ref, retirement_acceptance_ref.
    Every future actual reference is supplied explicitly by root.
    """
    plan=old.build(read,**original)
    binding.metadata_ref(reader_source_manifest);binding.metadata_ref(reader_source_acceptance)
    manifest,review=read(reader_source_manifest),read(reader_source_acceptance)
    c.require(review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
              and binding.same_ref(review['manifest'],reader_source_manifest)
              and review['metadata_source_count_limit']==106, 'Independent exact source acceptance of the narrow reader')
    c.require(manifest['complete_retained_source_count']==len(manifest['source_sha256'])==106
              and manifest['archive_stage_budget']==c.INVENTORY
              and manifest['scientific_constraints']==binding.CONSTRAINTS,
              '106 source entries only; original archive/scientific limits retained')
    c.require(all(manifest['source_sha256'].get(path)==sha for path,sha in plan['source_sha256'].items()),
              'No frozen historical source pin may be removed or refreshed')
    plan.update(source_sha256=copy.deepcopy(manifest['source_sha256']),
                signed_index_snapshot=copy.deepcopy(snapshot_binding),dpkg_status_observation=copy.deepcopy(status_observation),metadata_source_count_limit=106,
                snapshot_reader_source_manifest=copy.deepcopy(reader_source_manifest),
                snapshot_reader_source_acceptance=copy.deepcopy(reader_source_acceptance))
    # Saved JSON ancestry only. No raw snapshot or compressed file is read here.
    n=read(binding.NEXT)
    binding.snapshot_chain(plan,read,n)
    gap=read(n['source_gap_report'])
    original=[r for r in gap['metadata_inputs'] if r['path']==binding.OLD_STATUS['path']]
    c.require(len(original)==1, 'Single exact original status input')
    binding.status_metadata(plan,read,n,gap,original[0])
    return plan


if __name__=='__main__':
    raise RuntimeError('Source-only API: explicit real completion refs and independent root review required')
