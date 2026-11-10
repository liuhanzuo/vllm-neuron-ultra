"""Root-space diagnostics and a stronger fresh-launch check, with no I/O."""
import json

ROOT_FLOOR=28*2**30
NVME_FLOOR=30*2**30
ROOT_START_MARGIN=512*2**20
POLICY=dict(root_floor_bytes=ROOT_FLOOR,nvme_floor_bytes=NVME_FLOOR,
    fresh_root_headroom_bytes=ROOT_START_MARGIN,unchanged_metadata_cap_bytes=2**30,
    unchanged_work_cap_bytes=18*2**30,record_actual_floor_failure_bytes=True)


def require_floor(available,required,filesystem,context):
    if type(available)is not int or type(required)is not int or available<0 or required<0:
        raise RuntimeError('Invalid storage byte count')
    if available<required:
        raise RuntimeError('Storage floor: '+json.dumps(dict(filesystem=filesystem,context=context,
            available_bytes=available,required_bytes=required,deficit_bytes=required-available),sort_keys=True))


def check(values,metadata_used,work_used,reserve,fresh):
    if not (type(metadata_used)is int and 0<=metadata_used<=2**30
            and type(work_used)is int and 0<=work_used<=18*2**30):
        raise RuntimeError('Original combined output caps')
    if type(reserve)is not bool or type(fresh)is not bool:
        raise RuntimeError('Explicit reservation/admission state')
    root_required=ROOT_FLOOR+(ROOT_START_MARGIN if fresh else 0)
    remaining=(2**30-metadata_used)+(18*2**30-work_used)if reserve else 0
    require_floor(values['root'],root_required,'root','fresh-admission'if fresh else'runtime')
    require_floor(values['nvme'],NVME_FLOOR+remaining,'nvme','full-remaining-reservation'if reserve else'runtime')
    return dict(root_required_bytes=root_required,root_floor_margin_bytes=values['root']-ROOT_FLOOR,
        nvme_required_bytes=NVME_FLOOR+remaining,remaining_nvme_reservation=remaining,
        fresh_root_margin_is_admission_only=True)
