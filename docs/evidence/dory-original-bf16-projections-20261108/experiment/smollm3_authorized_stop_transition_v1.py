"""Pure classification of one systemd property sample during authorized cleanup.

No process, filesystem, service, lock or native calls occur here. ``l`` supplies
existing pure lifecycle validators; the controller owns observations and IO.
Missing units carry physical-drain evidence only, never an exit-success claim.
"""
import copy
import re


class AuthorizedStopAbsent(RuntimeError):
    """Only a fully bound durable stop intent permits this typed transition."""
    def __init__(self, observation):
        super().__init__('Bound authorized stop transitioned to missing unit; physical evidence only')
        self.observation = copy.deepcopy(observation)


def authenticate_loaded_sample(l, intent, value, claim=None):
    """The predicates in lifecycle_v18.authenticate_service, without resampling."""
    l.require(value['LoadState'] == 'loaded' and value['Id'] == intent['unit'] + '.service',
              'Loaded exact service')
    l.require(l.installed(value) == ' '.join(intent['execargv']), 'Installed argv differs')
    l.require(re.fullmatch('[0-9a-f]{32}', value['InvocationID']) and
              value['InvocationID'] != '0' * 32, 'Actual invocation ID')
    l.require(claim is None or value['InvocationID'] == claim, 'Invocation replaced')
    l.require(not value['ControlGroup'] or
              value['ControlGroup'] == l.GROUP_PREFIX + intent['unit'] + '.service',
              'Unexpected cgroup')
    l.resource_check(value, intent['kind'])
    return value


def retained_stop_authorization(l, state, record):
    """Check the complete saved authorization against actual retained identity.

The record proves authorization, not that systemctl stop ran or succeeded.
The original owner's successful exit is deliberately not inferred here.
"""
    intent, owner, binding = state['intent'], state.get('owner'), state.get('binding')
    l.require(type(record) is dict and binding is not None and state.get('claim') is not None,
              'Bound durable stop intent required for missing owned service')
    l.require(record.get('authorized_stop') is True and
              record.get('successful_client_exit') is False and
              record.get('server_normal_shutdown') is (intent['kind'] == 'server'),
              'Exact stop authorization flags')
    l.require(record.get('intent') == intent and record.get('owner') == owner and
              record.get('binding') == binding and record.get('claim') == state['claim'],
              'Stop intent/owner/invocation/binding mismatch')
    l.require(binding['invocation_id'] == state['claim'], 'Stop state claim differs from binding')
    value = record.get('properties')
    l.require(type(value) is dict, 'Retained stop authorization properties required')
    authenticate_loaded_sample(l, intent, value, state['claim'])
    l.verify_binding(intent, value, binding)
    if owner is not None:
        l.require(owner['unit'] == intent['unit'] and owner['argv'] == intent['argv'] and
                  owner['cgroup'] == '0::' + l.GROUP_PREFIX + intent['unit'] + '.service',
                  'Retained stop owner command/cgroup mismatch')
        l.verify_retained_binding(intent, owner, binding)
    return dict(authorized_stop=True, authorization_only=True,
                original_owner=copy.deepcopy(owner), binding=copy.deepcopy(binding),
                claim=state['claim'], successful_exit_proven=False)


def authorized_absent_sample(l, state, value, record, origin):
    """Only genuine not-found properties plus exact stop proof may transition."""
    intent = state['intent']
    l.require(value['LoadState'] == 'not-found' and value['Id'] == intent['unit'] + '.service',
              'Missing exact stopped service required')
    # systemd's not-found Result=success is a default, not an observed exit.
    l.require(l.terminal(value) and value['ActiveState'] == 'inactive' and value['SubState'] == 'dead'
              and value['InvocationID'] == '' and value['ExecMainPID'] == '0'
              and value['ExecMainCode'] == '0' and value['ExecMainStatus'] == '0'
              and value['ControlGroup'] == '', 'Incomplete or foreign not-found observation')
    authorization = retained_stop_authorization(l, state, record)
    return dict(schema='smollm3_authorized_stop_absence_v1', origin=origin,
                properties=copy.deepcopy(value), authorization=authorization,
                physical_only=True, success_proven=False)
