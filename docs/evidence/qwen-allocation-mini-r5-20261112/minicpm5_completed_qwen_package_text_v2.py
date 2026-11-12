"""Exact two completed slot-preparation text pins, additive to frozen v1.

Shared by preparation and actual PV authentication. No IO on import and no
generic /tmp permission. Readers retain their original metadata-size limits.
"""
from pathlib import Path
from research import minicpm5_completed_qwen_package_text_v1 as previous

ROOT,require=previous.ROOT,previous.require
Q=ROOT/'research'
SLOT_CATALOG=dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v104.json'),
    sha256='9bc186b643d554423a35390389cc28ea245c6fc383c02e41c37528731cffb4a5')
SLOT_TEXT={
    '/tmp/trn1-prepare-qwen-slot-special-r1-20261112.py':'22de1c3228633a17906ea0cd8e626702aedd8c0e155bf5812708934a57bed0ab',
    '/tmp/trn1-prepare-qwen-slot-special-r2-20261112.py':'4a6aa6e4daf82a5e91291b57ef3e6e14b6b44c3da4224d61b3fd57010c26afe7'}
SLOT_ENDED=(
    dict(index=151,path=next(iter(SLOT_TEXT)),terminal=dict(
        path=str(Q/'qwen35-slot-special-value-r1-root-actual-exit-20261112-v1.json'),
        sha256='97053ac5e360db70663026b2e91d53cf2d41cfea004ad478af6616917bc7c26c'),
        tool=dict(kind='tool_session',session_id=31827,terminal_chunk='436e73',exit_code=1),calls=0),
    dict(index=152,path=list(SLOT_TEXT)[1],terminal=dict(
        path=str(Q/'qwen35-slot-special-value-r2-root-actual-exit-20261112-v1.json'),
        sha256='24536a19e7fcef6a8d13e17ab1f10749916b2e4ca057076a729f94b4904e09fd'),
        tool=dict(kind='tool_session',session_id=29233,terminal_chunk='bc15e6',exit_code=0),calls=6))


def authenticate_completed_slots(catalog_ref,read,guard):
    previous.codec.reference(catalog_ref);guard()
    historical=read(SLOT_CATALOG)
    current=historical if catalog_ref==SLOT_CATALOG else read(catalog_ref)
    require(type(historical['runs']) is list and len(historical['runs'])==153
        and type(current['runs']) is list and len(current['runs'])>=153
        and current['runs'][:153]==historical['runs'],'Every exact v104 row retained')
    for proof in SLOT_ENDED:
        row=historical['runs'][proof['index']]
        require(row['terminal']==proof['terminal'] and row['tool']==proof['tool']
            and row['source_sha256'].get(proof['path'])==SLOT_TEXT[proof['path']],
            'Exact completed slot tool and preparation source pin')
        ended=read(proof['terminal'])
        require(ended['schema']=='qwen35_slot_special_value_root_actual_exit_v1'
            and ended['reviewer']=='/root' and ended['actual_tool']==row['tool']
            and ended['owners']==row['owners'] and ended['units']==row['units']
            and ended['actual_tool_exit_observed'] is True and ended['actual_outer_exit_observed'] is True
            and ended['physically_drained'] is True and ended['owners_gone'] is True
            and ended['original_lock_reacquired_and_released'] is True
            and ended['lock_identity']=={'device':66306,'inode':8294421}
            and ended['runtime_calls']==proof['calls'] and ended['full_model_qualified'] is False
            and ended['qualification_matrix_completed'] is False and ended['new_speedup_claim'] is False,
            'Actual completed slot owner/tool/drain evidence; no model qualification')
        guard()
    return dict(SLOT_TEXT)


def extension_shape(extension,*,completed_lfm_text,catalog_ref,read,guard=lambda:None):
    require(type(extension) is dict and 0<len(extension)<=2048,'Original bounded additive extension')
    selected={path:sha for path,sha in extension.items() if path in SLOT_TEXT}
    ordinary={path:sha for path,sha in extension.items() if path not in SLOT_TEXT}
    # Every other source, including unknown /tmp paths, retains exact v1 rules.
    previous.extension_shape(ordinary,completed_lfm_text=completed_lfm_text,
        catalog_ref=catalog_ref,read=read,guard=guard)
    if selected:
        require(selected==SLOT_TEXT,'Both exact completed slot preparation sources, no subset or changed SHA')
        require(authenticate_completed_slots(catalog_ref,read,guard)==selected,
                'Shared preparation and runtime completed-slot source binding')
    return extension
