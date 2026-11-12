"""Compose one unreviewed discovery plan from explicit completed history refs.

No physical checks, payload/native reads, admission, authority or automatic run.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q.parent),str(Q)]
from research import minicpm5_bootstrap_observation_v1 as core
import prepare_minicpm5_sdk310_deb_plan_v4 as previous


def build(read,*,catalog_ref,scope_ref,predecessor_ref,retirement_acceptance_ref,
          source_manifest_ref,source_acceptance_ref,output_root):
    # Reuse complete actual history and all three precise retirement exceptions.
    # The returned archive-shaped plan is only a source template; no archive
    # action occurs, and archive-specific fields are removed below.
    value=previous.build(read,catalog_ref,scope_ref,predecessor_ref,'download',output_root,
                         retirement_acceptance_ref=retirement_acceptance_ref)
    manifest=read(source_manifest_ref);accepted=read(source_acceptance_ref)
    retained=read(core.RETAINED_MANIFEST)
    core.require(accepted['reviewer']=='/root' and accepted['passed'] is True and accepted['blockers']==[]
      and core.ref2(accepted['manifest'])==core.ref2(source_manifest_ref),'Exact independent observation source acceptance')
    pins=manifest['source_sha256']
    core.require(manifest['schema']=='minicpm5_bootstrap_observation_source_manifest_v1'
      and 1<=len(pins)<=core.SOURCE_CAP and manifest['limits']==core.LIMITS and manifest['artifact_limits']==core.CAPS
      and manifest['candidate_inputs']==core.CANDIDATES
      and manifest['retained_manifest']==core.RETAINED_MANIFEST
      and len(retained['source_sha256'])==106
      and all(pins.get(path)==sha for path,sha in retained['source_sha256'].items())
      and all(pins.get(path)==sha for path,sha in value['source_sha256'].items()),
      'Every inherited source including the latest106 retained; explicit observation code only')
    core.candidates(read)
    for key in ('archive_inputs','archive_mode','replay_from','producer_actual_exit','producer_recorder_actual_exit'):
        value.pop(key,None)
    value.update(schema='minicpm5_bootstrap_observation_plan_v1',work_kind='minicpm5_dual_python_bootstrap_discovery_v1',
      source_sha256=copy.deepcopy(pins),budget=copy.deepcopy(core.BUDGET),limits=copy.deepcopy(core.LIMITS),
      artifact_limits=copy.deepcopy(core.CAPS),role_order=list(core.ROLES),candidate_inputs=copy.deepcopy(core.CANDIDATES),
      observation_source_manifest=copy.deepcopy(source_manifest_ref),observation_source_acceptance=copy.deepcopy(source_acceptance_ref),
      observation_only=True,accepted_native_authority_created=False,runtime_admitted=False)
    return value


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    names=('catalog','scope','predecessor','retirement-acceptance','source-manifest','source-acceptance')
    for name in names:cli.add_argument('--'+name,required=True);cli.add_argument('--'+name+'-sha256',required=True)
    cli.add_argument('--output-root',required=True);cli.add_argument('--destination',required=True)
    a=cli.parse_args(argv)
    def ref(name):return dict(path=getattr(a,name.replace('-','_')),sha256=getattr(a,name.replace('-','_')+'_sha256'))
    value=build(previous.c.load,catalog_ref=ref('catalog'),scope_ref=ref('scope'),predecessor_ref=ref('predecessor'),
       retirement_acceptance_ref=ref('retirement-acceptance'),source_manifest_ref=ref('source-manifest'),
       source_acceptance_ref=ref('source-acceptance'),output_root=a.output_root)
    core.verify_sources(value['source_sha256'],previous.c.reference)
    destination=Path(a.destination)
    core.require(destination.parent==Q and destination.suffix=='.json'
       and not destination.name.startswith('continuation-handoff-'),'Only a new unreviewed plan, no handoff')
    print(json.dumps(previous.c.save(destination,value,8*2**20)),flush=True)


if __name__=='__main__':main()
