"""Small native overlay with complete catalog; no launcher, writes or native imports.

Resolve only after authenticating the real completed recovery. Base scientific
plan remains immutable; resolved plan is memory-only and its digest is bound by
future native permit/admission. Caller separately authenticates the small overlay.
"""
import copy
from pathlib import Path
import re
from research import minicpm5_layer1_operator_recovered_completion_v2 as completion
from research import minicpm5_layer1_operator_native_manifest_v1 as manifest
from research import minicpm5_layer1_operator_native_output_catalog_v2 as output_contract
from minicpm5_layer0_runtime_auth_v1 import digest
require=completion.require
SCHEMA='minicpm5_layer1_operator_native_small_overlay_v2'


def output_catalog(contract):
    # Derive the complete finite contract from the original native manifest.
    # Catalog completeness does not imply an implemented/admitted runtime.
    return output_contract.catalog(contract)


def resolve(overlay,recovered,read,guard):
    require(callable(read)and callable(guard),'Explicit authenticated metadata reader/guard');guard()
    require(set(overlay)=={'schema','base_plan','sdk_operator_recovery','current_serial_catalog',
        'native_manifest','source_extension','input_extension','output_root'} and overlay['schema']==SCHEMA,
        'Exact small native overlay, never a replacement large scientific plan')
    require(recovered['metadata_complete']is True and recovered['runtime_admitted']is False
        and recovered['original_model_exit_code']==1 and recovered['binding']==overlay['sdk_operator_recovery']==completion.BINDING
        and overlay['base_plan']==recovered['base_plan_ref'],'Actually completed recovered data, immutable original model exit1')
    for key in ('base_plan','native_manifest','current_serial_catalog'):completion.sources.reference(overlay[key])
    path=Path(overlay['output_root'])
    require(path.parent==Path('/mnt/nvme') and re.fullmatch('minicpm5-layer1-operator-native-[a-z0-9-]{1,64}',path.name),
        'Fresh native-only NVMe namespace supplied by reviewed future outer')
    base=recovered['base_plan'];plan=copy.deepcopy(base);pins=dict(recovered['source_sha256'])
    require(all(pins.get(path)==sha for path,sha in base['source_sha256'].items()),'Every original source retained unchanged')
    for path,sha in overlay['source_extension'].items():
        require(type(path)is str and Path(path).is_absolute() and str(Path(path))==path and '..'not in Path(path).parts
            and path not in pins and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),
            'Only new explicit native/current-serial source pins, never replacements')
        pins[path]=sha
    inputs=copy.deepcopy(base['inputs'])
    for path,row in overlay['input_extension'].items():
        require(type(path)is str and Path(path).is_absolute() and str(Path(path))==path and '..'not in Path(path).parts
            and path not in inputs and type(row)is dict and set(row)=={'sha256','identity'}
            and type(row['sha256'])is str and re.fullmatch('[a-f0-9]{64}',row['sha256']) and type(row['identity'])is dict,
            'Only new exact input identities, all original inputs preserved')
        inputs[path]=copy.deepcopy(row)
    guard();contract=read(overlay['native_manifest'])
    require(contract==manifest.build(base,recovered['native']['native']),'Complete original native24-frame manifest rebuilt from saved native metadata')
    catalog=output_catalog(contract)
    plan.update(source_sha256=pins,inputs=inputs,output_root=overlay['output_root'],
        native_operator_manifest=contract,sdk_operator_recovery=copy.deepcopy(completion.BINDING),
        serial_predecessors=overlay['current_serial_catalog'],runtime_output_manifest=catalog)
    # Keep the old success binding for SDK batch/layout; it is not the failed operator run.
    require(plan['layer1_operator_boundary']==completion.latest.sdk.BINDING,'Original successfully completed batch/layout boundary unchanged')
    view=dict(latest_completion=base['latest_completion'],serial_predecessors=overlay['current_serial_catalog'],source_sha256=pins)
    serial=completion.latest.serial_catalog(view,read,guard)
    rows={row['terminal']['path']:row for row in serial['runs']}
    require(all(rows.get(row['terminal']['path'])==row for row in recovered['historical']['later_serial']['runs']),
        'Every exact recovery-historical serial record preserved')
    ended=recovered['actual_exit'];recorded=recovered['recorder_terminal']
    for ref,owner,tool in [(completion.BINDING['actual_exit'],ended['owner'],ended['actual_tool']),
                          (completion.BINDING['recorder_terminal'],recorded['owner'],recorded['actual_tool'])]:
        row=rows.get(ref['path'])
        require(row is not None and row['terminal']==ref and row['tool']==tool and owner in row['owners'],
            'Current catalog explicitly includes successful recovery and independent recorder actual exits/owners')
    owners=copy.deepcopy(recovered['done']['operator_sdk']['owners'])+copy.deepcopy(serial['owners'])
    require(all(owner in owners for owner in recovered['owners']),'Latest scope preserves every exact previous owner')
    units=set(recovered['complete_known_units'])|set(serial['units'])
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+completion.physical.GROUP_PREFIX):
            units.add(completion.physical.group_unit(group[3:]))
    scope=dict(recovered['done'],owners=owners,later_serial=serial,complete_known_units=sorted(units))
    guard();return dict(plan=plan,plan_sha256=digest(plan),overlay=copy.deepcopy(overlay),
        native_bundle=recovered['native'],current_scope=scope,current_serial_view=view,
        native_runtime_output_catalog=catalog,native_runtime_output_catalog_sha256=digest(catalog),
        metadata_complete=True,physical_checks_performed=False,full_runtime_catalog_pending=False,
        runtime_integration_pending=True,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
