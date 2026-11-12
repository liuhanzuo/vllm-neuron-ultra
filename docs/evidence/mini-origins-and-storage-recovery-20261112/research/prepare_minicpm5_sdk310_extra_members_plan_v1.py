"""Compose an unadmitted supplemental plan from the latest complete idle catalog.

Uses the existing archive planner for full history and retirement provenance.
No archive/current-binary read, native import, launch, lock or plan acceptance.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import minicpm5_sdk310_extra_members_binding_v1 as binding
import minicpm5_sdk310_extra_members_contract_v1 as c
import prepare_minicpm5_sdk310_deb_plan_v4 as previous


def build(read, catalog_ref, predecessor_ref, source_manifest_ref, source_acceptance_ref, output_root):
    for ref in (catalog_ref, predecessor_ref, source_manifest_ref, source_acceptance_ref):
        previous.reference(ref)
    old = read(binding.BASE)
    catalog = read(catalog_ref)
    handoff = read(catalog['current_handoff'])
    c.require(handoff['running'] is False and handoff['active_serial'] is None
              and handoff['live_tool_sessions'] == [] and handoff['active_heavy_units'] == [],
              'An actual ended idle handoff is required; no running snapshot accepted')
    manifest = binding.validate_sources_manifest(read(source_manifest_ref),read)
    accepted = read(source_acceptance_ref)
    c.require(accepted['reviewer'] == '/root' and accepted['passed'] is True and accepted['blockers'] == []
              and accepted['manifest'] == source_manifest_ref and accepted['source_sha256'] == manifest['source_sha256'],
              'Separate root source acceptance required before plan composition')
    prior_catalog=read(old['complete_current_serial_catalog'])
    c.require(catalog['runs'][:len(prior_catalog['runs'])] == prior_catalog['runs'],
              'Every previously completed archive history row is retained unchanged')
    # Reuse the reviewed planner's full history checks, all 874 LFM children,
    # exact retired-service bindings, and latest actual predecessor matching.
    # Its eight-package validation applies only to the old eight-package input.
    def legacy_ref(name):return {key:binding.helper.REFS[name][key] for key in ('path','sha256')}
    value=previous.build(read,catalog_ref,old['r2_predecessor_scope'],predecessor_ref,
        'replay',output_root,legacy_ref('download'),legacy_ref('download_exit'),
        old['producer_recorder_actual_exit'],old['source_only_acceptance'])
    for key in ('archive_inputs','archive_mode','replay_from'):
        del value[key]
    value.update(work_kind=binding.WORK_KIND,source_sha256=copy.deepcopy(manifest['source_sha256']),
        extra_member_metadata=copy.deepcopy(binding.helper.REFS),
        supplemental_source_manifest=copy.deepcopy(source_manifest_ref),
        supplemental_source_acceptance=copy.deepcopy(source_acceptance_ref),
        metadata_source_count_limit=112,original_eight_package_plan=copy.deepcopy(binding.BASE),
        fresh_physical_checks_required=True,root_plan_review_required=True,runtime_admitted=False)
    c.require(Path(output_root).name.startswith('dory-serial-inventory-minicpm5-sdk310-extra-members-'),
              'Fresh supplemental-only result namespace')
    return value


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','predecessor-exit','source-manifest','source-acceptance'):
        cli.add_argument('--'+name,required=True);cli.add_argument('--'+name+'-sha256',required=True)
    cli.add_argument('--output-root',required=True);cli.add_argument('--destination',required=True)
    args=cli.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B and assertions')
    def ref(name):return dict(path=getattr(args,name),sha256=getattr(args,name+'_sha256'))
    value=build(c.load,ref('catalog'),ref('predecessor_exit'),ref('source_manifest'),ref('source_acceptance'),args.output_root)
    c.verify_sources(value['source_sha256'])
    value['preparation_builder']=c.reference(Path(__file__).resolve(),2*c.MIB)[0]
    path=Path(args.destination)
    c.require(path.parent==c.Q and path.suffix=='.json'
              and path.name.startswith('minicpm5-sdk310-extra-members-') and path.name.endswith('-plan.json'),
              'Write one new unreviewed plan only; never a handoff, catalog or actual exit')
    print(json.dumps(c.save(path,value),sort_keys=True))


if __name__=='__main__':main()
