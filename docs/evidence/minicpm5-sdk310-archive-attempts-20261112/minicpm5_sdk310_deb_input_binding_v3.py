"""Mini archive v3 adds exactly two independently evidenced historical retirements."""
from minicpm5_sdk310_deb_input_binding_v2 import *
import minicpm5_sdk310_deb_input_binding_v2 as previous
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement

SUCCESSOR_NAMES = ('minicpm5_sdk310_retired_lfm_r2_v2.py', 'minicpm5_sdk310_deb_input_binding_v3.py',
                   'run_minicpm5_sdk310_deb_members_root_v3.py', 'minicpm5_sdk310_deb_members_worker_v3.py',
                   'record_minicpm5_sdk310_deb_stage_exit_v3.py')


def validate(plan, review, plan_ref):
    previous.validate(plan, review, plan_ref)
    for name in SUCCESSOR_NAMES:c.require(str(c.Q/name) in plan['source_sha256'],'All exact retirement successor sources reviewed')
    retirement.validate(plan,review)
    return plan
