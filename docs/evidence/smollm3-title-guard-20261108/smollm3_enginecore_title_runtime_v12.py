# Source-only candidate; not connected to an admitted runtime.
"""Dedicated transaction glue for the finite EngineCore title successor.

No I/O on import. Every future ordinary or in-flight poll retains the complete
existing lifecycle guard; only the same producer's witnessed argv transition is
handled differently. The original immutable permit is never rewritten.
"""
import copy
from pathlib import Path
import sys
import threading
import time

import smollm3_enginecore_title_transition_v1 as title

require = title.require
STAGES = ('before', 'before-ack', 'after', 'commit')


def names(pid):
    return {stage: 'title-' + stage + '-' + str(pid) + '.json' for stage in STAGES}


def lifecycle():
    import smollm3_observer_invariance_lifecycle_v23 as life
    return life


def retained(permit):
    life = lifecycle(); root = life.p.directory(permit['run']['run_id']); paths = names(permit['pid'])
    if (root / ('title-failure-' + str(permit['pid']) + '.json')).exists():
        raise RuntimeError('failed title transition cannot authenticate a producer')
    exists = {stage: (root / name).exists() for stage, name in paths.items()}
    if not any(exists.values()): return None
    require(exists['before'], 'title receipt without before intent')
    if not exists['before-ack']:
        require(not exists['after'] and not exists['commit'], 'after/commit cannot precede before ACK')
        return {'intent_only': life.read(root / paths['before'])}
    if exists['commit']:
        require(exists['after'], 'commit cannot precede after')
        return {stage.replace('-', '_'): life.read(root / paths[stage]) for stage in STAGES}
    value = {stage.replace('-', '_'): life.read(root / paths[stage]) for stage in ('before', 'before-ack')}
    if exists['after']: value['uncommitted_after'] = life.read(root / paths['after'])
    return value


def observe(permit):
    return title.observe_process(permit['pid'], permit['spawn_observation']['exe']['target'])


def check_producer(permit, *, pending=False, transaction=None):
    actual = observe(permit); initial = permit['spawn_observation']; receipts = retained(permit)
    if receipts is not None and 'intent_only' in receipts:
        title.validate_before(permit, receipts['intent_only'])
        require(receipts['intent_only']['before'] == initial, 'title intent original admission observation')
        return title.check_live(permit, initial, actual)
    if receipts is not None and 'uncommitted_after' in receipts:
        title.validate_after(permit, receipts['before'], receipts['before_ack'], receipts['uncommitted_after'])
        receipts = {k: receipts[k] for k in ('before', 'before_ack')}
    if transaction is not None:
        require(type(transaction) is _InFlight and transaction.permit == permit, 'one private in-flight producer capability')
        transaction.check()
        if receipts is not None and 'commit' in receipts:
            require(receipts['before'] == transaction.before and receipts['before_ack'] == transaction.ack
                    and receipts['after'] == transaction.after, 'root committed this exact still-returning transaction')
            return title.check_live(permit, initial, actual, receipts)
        require(receipts is not None and set(receipts) == {'before', 'before_ack'}, 'actual root before ACK required in-flight')
        require(receipts == {'before': transaction.before, 'before_ack': transaction.ack}, 'same actual acknowledged attempt')
        title.validate_after(permit, transaction.before, transaction.ack, transaction.after)
        require(actual == transaction.after['after'], 'actual post-call observation retained in-flight')
        value = title.check_live(permit, initial, actual, receipts, pending=True, now=time.monotonic())
        value.update(state='inflight', authenticated=True, work_authorized=False)
        return value
    if not pending and receipts is not None and set(receipts) == {'before', 'before_ack'} and actual == initial:
        title.validate_before_ack(permit, receipts['before'], receipts['before_ack'])
        require(receipts['before']['before'] == initial, 'original title wait belongs to this admission')
        value = title.check_live(permit, initial, actual)
        value.update(pending_title_intent=title.digest(receipts['before']), work_authorized=False)
        return value
    return title.check_live(permit, initial, actual, receipts, pending=pending, now=time.monotonic())


class _InFlight:
    """Only complete_engine_title creates and holds this stack-bound capability."""
    def __init__(self, context, hook, token, before, ack, after):
        self.context = context; self.hook = hook; self.token = token; self.permit = context.permit
        self.before = before; self.ack = ack; self.after = after; self.thread = threading.get_ident()

    def check(self):
        require(threading.get_ident() == self.thread and self.context.title_witness is self.hook
                and self.context._title_token is self.token and self.context._title_inflight is self
                and self.hook.called and not self.hook.committed and not self.hook.failed, 'one actual in-flight title wrapper')
        frame = sys._getframe(1); found = False
        try:
            for _ in range(40):
                if frame is None: break
                if frame.f_code is self.hook.wrapper.__code__ and frame.f_locals.get('token') is self.token:
                    found = True; break
                frame = frame.f_back
        finally:
            del frame
        require(found, 'in-flight capability only while original setter wrapper is on stack')


def inflight(context):
    if context is None: return None
    value = getattr(context, '_title_inflight', None)
    if value is not None: value.check()
    return value


def worker_check(context):
    value = check_producer(context.permit, transaction=inflight(context))
    require(value['authenticated'] and value['state'] in ('original', 'committed', 'inflight'), 'ordinary producer authentication')
    return value


def init_context(context):
    context.title_witness = None
    context._title_token = None
    context._title_before = context._title_ack = context._title_inflight = None


def install_hook(context):
    from smollm3_enginecore_title_hook_v3 import EngineCoreTitleHook, PINS
    context.poll()
    require(context.kind == 'server' and context.spawned and context.title_witness is None, 'one spawned title hook')
    context.title_witness = EngineCoreTitleHook(context)
    context.record('title-hook-' + str(context.permit['pid']) + '.json',
        dict(schema='smollm3_enginecore_title_hook_v3', spawn_identity=title.identity(context.permit),
             permit_sha256=title.digest(context.permit), source_sha256=PINS, single_ray_trace_owner=True,
             same_in_both_modes=True, actual_engine_entry_observed=False))


def begin(context, hook, witness):
    require(context.title_witness is hook and hook.called and not hook.committed and context._title_token is None,
            'one exact called title hook transaction')
    context.poll()
    # Intent does not exist until the ordinary pre-publication poll completes.
    # The dedicated writer retains both full polls and all durable-write checks.
    before = context.record_engine_title_before(hook, witness)
    paths = names(context.permit['pid']); life = lifecycle(); root = life.p.directory(context.row['run_id'])
    while not (root / paths['before-ack']).exists():
        context.poll(); require(time.monotonic() < before['deadline_monotonic'], 'bounded original-title before ACK wait')
        time.sleep(.02)
    ack = life.read(root / paths['before-ack']); title.validate_before_ack(context.permit, before, ack)
    token = object()
    context._title_token = token; context._title_before = before; context._title_ack = ack
    return token


def complete(context, hook, token, original_result):
    require(context.title_witness is hook and context._title_token is token and token is not None
            and context._title_inflight is None and original_result is None, 'same completed original setter transaction')
    before, ack = context._title_before, context._title_ack
    after = title.make_after(context.permit, before, ack, observe(context.permit), time.monotonic(), True)
    context._title_inflight = _InFlight(context, hook, token, before, ack, after)
    context._title_inflight.check()
    paths = names(context.permit['pid']); life = lifecycle(); root = life.p.directory(context.row['run_id'])
    # The ordinary writer and both ordinary full polls remain in force.
    context.record(paths['after'], after)
    while not (root / paths['commit']).exists():
        context.poll(); require(time.monotonic() < before['deadline_monotonic'], 'bounded exact in-flight commit wait')
        time.sleep(.02)
    committed = life.read(root / paths['commit'])
    title.validate_complete(context.permit, before, ack, after, committed)
    require(observe(context.permit) == after['after'], 'actual title retained through root commit')
    context._title_inflight = None
    context.poll()


def fail(context, hook, token, error_type):
    require(context.title_witness is hook and context._title_token is token, 'failure belongs to exact current attempt')
    # Match the existing best-effort wrapper-failure policy. This writes only a
    # failure receipt, never a permit/ACK or authenticated transition success.
    life = lifecycle(); actual = observe(context.permit)
    initial = context.permit['spawn_observation']
    require(initial['exe'] == actual['exe'] and all(initial['identity'][k] == actual['identity'][k] for k in title.STABLE),
            'failure still belongs to original producer')
    life.monitor(context.row['run_id'])
    life.record(life.p.directory(context.row['run_id']) / ('title-failure-' + str(context.permit['pid']) + '.json'),
                dict(schema='smollm3_enginecore_title_failure_v1', before_sha256=title.digest(context._title_before),
                     error_type=error_type, observation=actual, transition_completed=False))


def no_prior_terminal_failure(permit):
    """Bounded retained failure/terminal evidence forbids any new ACK/commit.

    Final audit independently still rejects failures published after a check;
    separate immutable files are not claimed to form one atomic transaction.
    """
    life = lifecycle(); root = life.p.directory(permit['run']['run_id'])
    for pattern in ('title-failure-*.json', 'wrapper-failure-*.json', 'spawn-failure-*.json', '*-failure.json'):
        require(not list(root.glob(pattern)), 'earlier run failure forbids title ACK/commit')
    for name in ('server-stop-intent.json', 'server-terminal.json', 'client-stop-intent.json', 'client-terminal.json'):
        require(not (root/name).exists(), 'earlier run stop/terminal forbids title ACK/commit')
    for name in ('supervisor-terminal.json', 'external-terminal.json', 'external-recovery-required.json'):
        require(not (life.CONTROL/name).exists(), 'earlier matrix terminal forbids title ACK/commit')


def acknowledge(permit, controller):
    """Called only inside the original controller's fully guarded ACK loop."""
    life = lifecycle(); root = life.p.directory(permit['run']['run_id']); paths = names(permit['pid'])
    no_prior_terminal_failure(permit)
    if not (root / paths['before']).exists():
        return check_producer(permit)
    before = life.read(root / paths['before']); title.validate_before(permit, before)
    require(before['before'] == permit['spawn_observation'], 'controller original admitted full observation')
    if not (root / paths['before-ack']).exists():
        ack = title.before_ack(permit, before, controller, observe(permit), time.monotonic())
        no_prior_terminal_failure(permit)
        life.record(root / paths['before-ack'], ack)
    else:
        ack = life.read(root / paths['before-ack']); title.validate_before_ack(permit, before, ack)
    if not (root / paths['after']).exists():
        return check_producer(permit, pending=True)
    after = life.read(root / paths['after']); title.validate_after(permit, before, ack, after)
    if not (root / paths['commit']).exists():
        committed = title.commit_ack(permit, before, ack, after, controller, observe(permit), time.monotonic())
        no_prior_terminal_failure(permit)
        life.record(root / paths['commit'], committed)
    return check_producer(permit)


def spawn_parent(context):
    require(inflight(context) is None, 'no child dispatch during title transaction')
    checked = check_producer(context.permit)
    require(checked['authenticated'], 'spawn parent cannot be a pending title')
    receipts = retained(context.permit)
    require(receipts is None or set(receipts) == {'before', 'before_ack', 'after', 'commit'}, 'only committed parent title may spawn')
    return dict(initial=context.permit['spawn_observation'], live=checked['live_observation'], receipts=receipts)


def validate_spawn_parent(permit, value):
    require(type(value) is dict and set(value) == {'initial', 'live', 'receipts'} and value['initial'] == permit['spawn_observation'],
            'retained exact parent title observation')
    checked = title.check_live(permit, value['initial'], value['live'], value['receipts'])
    require(checked['authenticated'] and checked['state'] in ('original', 'committed'), 'actual original or committed spawn parent')
    return checked


def audit(permit):
    life = lifecycle(); root = life.p.directory(permit['run']['run_id']); paths = names(permit['pid'])
    receipts = retained(permit)
    if receipts is None: return dict(changed=False, references=[])
    require(set(receipts) == {'before', 'before_ack', 'after', 'commit'}, 'all title operations completed before qualification')
    closure = title.validate_complete(permit, receipts['before'], receipts['before_ack'], receipts['after'], receipts['commit'])
    require(receipts['before']['before'] == permit['spawn_observation'], 'final original spawn observation')
    hook = life.read(root / ('title-hook-' + str(permit['pid']) + '.json'))
    witness = receipts['before']['witness']
    from smollm3_enginecore_title_hook_v3 import CORE, UTILS, ENVS, PINS, EXTENSION, EXTENSION_IDENTITY
    require(hook['source_sha256'] == PINS and witness['core_source_sha256'] == PINS[CORE]
            and witness['utils_source_sha256'] == PINS[UTILS] and witness['envs_source_sha256'] == PINS[ENVS]
            and witness['native_binding']['extension'] == EXTENSION
            and witness['native_binding']['file_identity'] == EXTENSION_IDENTITY,
            'postdrain actual title callable/native receipt matches exact source hook')
    return dict(changed=True, closure=closure,
                references=[dict(path=str(root / paths[stage]), sha256=life.digest_bytes(life.read_bytes(root / paths[stage]))) for stage in STAGES])
