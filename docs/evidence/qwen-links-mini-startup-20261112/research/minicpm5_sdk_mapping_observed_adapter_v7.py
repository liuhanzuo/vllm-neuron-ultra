"""Lazy observed SDK lifecycle factory; no inherited-history import on import.

The admitted original outer explicitly supplies its already-loaded diagnostic
module. This factory performs no native load, process/lock action or admission.
Its returned adapter coordinates actual gated child and wait ownership only
when called later under the original authenticated lease and absolute deadline.
"""
import ast
import copy
import inspect
from pathlib import Path
import signal
import time
import types
from research import minicpm5_sdk_mapping_observer_backend_v6 as backend
from research import minicpm5_sdk_mapping_observer_logic_v3 as logic

require = logic.require
FILES = backend.ORIGINAL_NAMES | backend.OBSERVER_NAMES


def make_adapter_class(original):
    """Use the original diagnostic only after its full runtime prerequisites.

    No default import: tests supply selected original AST definitions and mocks.
    Future actual binding must pin this factory and the supplied original module.
    """
    r, serial, require = original.r, original.serial, original.require
    ObserverStore = backend.make_store_class(original.original)


    def callback_source(callback, pins):
        """Future root-approved callback implementations belong to the full map."""
        function = getattr(callback, '__func__', callback)
        require(type(function) is types.FunctionType, 'Explicit Python native verifier implementation')
        path = str(Path(function.__code__.co_filename).resolve())
        require(path in pins and path.endswith('.py'), 'Native verifier source in complete authenticated map')
        source = r.sources.text(path, pins[path])
        require(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function.__name__
                    and node.lineno == function.__code__.co_firstlineno
                    for node in ast.walk(ast.parse(source))), 'Exact reviewed verifier source definition')
        require(not inspect.iscoroutinefunction(function), 'Synchronous authority callback required')
        return path


    class ObservedSingleSDKAdapter(original.SingleSDKAdapter):
        """One gated original child, then exactly one ptrace/waitpid owner.

    Use inside the original outer try/finally. That outer must keep its descriptor
    until drain_owned returns physical proof, and keep retrying on uncertainty.
    Never use the old SingleSDKAdapter.finish or the old diagnostic CLI with this
    adapter: an observer failure with a zero child exit must still fail the run.
    """
        def __init__(self, authenticated, admission_ref, admission, lease, guard,
                     metadata, store, *, outer_deadline, authority, authenticate, verify_loaded_native):
            super().__init__(authenticated, admission_ref, admission, lease, guard, metadata, store)
            require(type(store) is ObserverStore and store.guard is guard
                    and guard.lease is lease, 'Original shared lease/guard and six-file Store')
            require(authority is not None and callable(authenticate) and callable(verify_loaded_native),
                    'No implicit or fabricated observer authority')
            require(type(outer_deadline) in (int, float) and 0 < outer_deadline < float('inf'),
                    'Original outer absolute monotonic deadline required')
            self.outer_deadline = outer_deadline
            self.authority = copy.deepcopy(authority)
            self.authenticate_native = authenticate
            self.verify_loaded_native = verify_loaded_native
            self.native_ports = self.observer = self.observed = None
            self.gate = None
            self.prerequisites_checked = False

        def observer_guard(self):
            self.guard()
            require(time.monotonic() < self.outer_deadline,
                    'Original outer absolute deadline expired; no observer deadline reset')

        def authenticate_complete_prerequisites(self):
            require(not self.prerequisites_checked, 'Once-only complete outer prerequisites')
            super().authenticate_complete_prerequisites()
            require(serial.exact(original.prior.LIMITS, logic.LIMITS)
                    and serial.exact(serial.CAPS, logic.CAPS), 'Original 2GiB/900s and all output budgets')
            # Query the original outer alarm without cancelling or restarting it.
            # Its earlier prerequisite time remains charged to the same 900s.
            remaining, interval = signal.getitimer(signal.ITIMER_REAL)
            require(0 < remaining <= logic.LIMITS['wall_seconds'] and interval == 0,
                    'Original finite outer alarm still active, never a fresh phase deadline')
            require(0 < self.outer_deadline-time.monotonic() <= logic.LIMITS['wall_seconds'],
                    'Original absolute outer deadline, including prerequisite time')
            pins = self.authenticated['context']['pins_plan']['source_sha256']
            for module in (original, backend, logic):
                path = str(Path(module.__file__).resolve())
                require(path in pins, 'Every observer implementation is in complete inherited source map')
                r.sources.text(path, pins[path])
            path = str(Path(__file__).resolve())
            require(path in pins, 'Observed lifecycle source explicitly reviewed')
            r.sources.text(path, pins[path])
            callback_source(self.authenticate_native, pins)
            callback_source(self.verify_loaded_native, pins)
            self.prerequisites_checked = True

        def start(self):
            """Original spawn/identity/gate construction; gate remains unwritten."""
            require(self.prerequisites_checked and self.state == 'new' and self.child is None,
                    'Authenticated once-only SDK child, no retry/resume')
            self.state = 'starting'
            a = self.authenticated
            worker_path = a['spec']['worker']
            argv = serial.phase_argv(worker_path, self.admission_ref, 1)
            self.argv = argv
            self.guard(force=True)
            self.observer_guard()
            r.sources.text(worker_path, a['context']['pins_plan']['source_sha256'][worker_path])
            env = r.p.environment({}, 'cpu_control', self.lease.admission['attempt_id'])
            self.child = original.subprocess.Popen(argv, env=env, cwd=original.ROOT,
                stdin=original.subprocess.PIPE, close_fds=True, start_new_session=True)
            self.child_started = True
            deadline = time.monotonic()+10
            while True:
                self.observer_guard()
                current = r.p.proc(self.child.pid)
                if current is not None and current['argv'] == argv:
                    r.life.process(current, argv, self.lease.admission['outer_owner']['pid'],
                                   self.lease.admission['outer_owner']['cgroup'])
                    require(str(Path('/proc/%d/exe' % current['pid']).resolve()) ==
                            str(Path(serial.python_for(1)).resolve()), 'Exact original SDK executable')
                    self.owner = current
                    break
                # This poll is strictly before native factory/wait ownership transfer.
                require(self.child.poll() is None and time.monotonic() < deadline,
                        'Bounded original child exec/acknowledgement')
                time.sleep(.01)
            self.store.json('phase-1-owner.json', current)
            payload = dict(schema=original.RELEASE_SCHEMA, admission=self.admission_ref,
                phase=1, owner=current, source_plan={key: a['context']['pins_plan'][key]
                    for key in ('source_contract', 'source_sha256')}, own_gate=None)
            self.gate = serial.encoded(payload)
            require(0 < len(self.gate) <= original.PIPE_CAP, 'Whole original source release packet')
            require(r.p.proc(current['pid']) == current, 'Exact acknowledged child before observing')
            self.state = 'gated'
            return copy.deepcopy(current), argv

        def _finish_observed(self, result, *, cleanup):
            require(self.native_ports is not None and self.observer is not None
                    and self.observer.ports is self.native_ports
                    and result.get('physically_drained') is True,
                    'Actual complete observed drain before returning wait ownership')
            terminal = result.get('sdk_terminal')
            require(type(terminal) is dict and terminal.get('parent_observed_wait') is True
                    and type(terminal.get('returncode')) is int
                    and terminal['returncode'] == self.child.returncode,
                    'Keep actual child result independently of observation status')
            self.observed = copy.deepcopy(result)
            if self.native_ports.wait_lock_owned:
                self.native_ports.finish_wait_handoff(result)
            # The original helper performs fresh gone/group checks and persists the
            # actual terminal. It performs no Popen poll/wait and never releases lock.
            return self._record_drained(terminal['returncode'], cleanup=cleanup)

        def finish(self, owner, argv):
            require(self.state == 'gated' and owner == self.owner and argv == self.argv
                    and self.child is not None and self.native_ports is None,
                    'Exact gated SDK child, no previous wait ownership transfer')
            # Factory preparation owns no wait lock. Persist both references
            # before the separate, alarm-safe, nonblocking ownership transition.
            self.observer = logic.ObservedSDK(self.child, owner, self.gate, self.authority,
                                            self.observer_guard, self.store, None)
            self.native_ports = backend.make_ports(self.child, owner, self.authority, self.observer_guard,
                authenticate=self.authenticate_native, verify_loaded_native=self.verify_loaded_native,
                inspect_owner=r.p.proc, owned_group_members=r.group_members)
            self.observer.ports = self.native_ports
            self.native_ports.begin_wait_handoff()
            self.state = 'observing'
            result = self.observer.run()
            self.observed = copy.deepcopy(result)
            if result['physically_drained'] is not True:
                self.state = 'cleanup_pending'
                raise logic.ObservationError('Observed SDK cleanup pending; retain original lock and handles')
            return self._finish_observed(result, cleanup=result['failure'] is not None)

        def drain_owned(self):
            if self.native_ports is None:
                # Preparation never owns the wait lock, including an alarm
                # after factory return but before this object's assignment.
                return super().drain_owned()
            require(not self.native_ports.wait_transition_uncertain,
                    'Uncertain wait transition: retain original lock and exact handles')
            if not self.native_ports.wait_handoff_started:
                require(not self.native_ports.wait_lock_owned,
                        'No ownership before explicit handoff')
                return super().drain_owned()
            if self.state != 'drained':
                require(self.observer is not None and self.observer.ports in (None, self.native_ports),
                        'Retain exact observer and Popen through cleanup')
                # Strong ports references precede handoff. Keep exact binding
                # through alarms delivered after the ownership flag commit.
                self.observer.ports = self.native_ports
                # Continue only the existing observer's cleanup. Never reset its
                # counters/task registry or replace it with Popen-based cleanup.
                if not self.observer._drained():
                    self.observer._cleanup()
                result = self.observer._result(self.observer._drained())
                require(result['physically_drained'] is True,
                        'Observed owned tasks not drained; original outer must retain lock')
                self._finish_observed(result, cleanup=True)
            return dict(physically_drained=True, no_services_started=True,
                        exit_receipt_saved=self.receipt_saved, child_started=self.child_started,
                        observer_wait_handoff_complete=not self.native_ports.wait_lock_owned)

        def result(self):
            """Read the final actual import summary only after observed physical drain."""
            require(self.state == 'drained' and self.terminal is not None and self.observed is not None,
                    'No result before actual child/observer drain')
            observation_ok = (self.observed['observation_complete'] is True
                and self.observed['trace_complete'] is True and self.observed['trace_saved'] is True
                and self.observed['exit_receipt_saved'] is True and self.observed['failure'] is None)
            result = dict(terminal=copy.deepcopy(self.terminal), observer=copy.deepcopy(self.observed),
                diagnostic_passed=False, full_model_qualified=False, new_speedup_claim=False)
            if self.terminal['returncode'] != 0 or not observation_ok or not self.receipt_saved:
                result['exit_code'] = result_exit_code(result)
                return result
            summary, summary_ref = self.store.load('phase-1-summary.json')
            require(summary['schema'] == original.SUMMARY_SCHEMA
                    and summary['diagnostic_passed'] is True and summary['stage'] == 'complete'
                    and summary['observation_and_error_payload_complete'] is True,
                    'Successful original SDK import summary required')
            require(set(self.store.inventory()['names']) == FILES,
                    'Exactly six accounted diagnostic/observer artifacts')
            self.metadata.unchanged()
            r.verify_loaded(self.authenticated['context']['pins_plan'], self.guard)
            self.guard(force=True)
            result.update(diagnostic_passed=True, summary=original.original.short(summary_ref), exit_code=0)
            return result

    return types.SimpleNamespace(ObservedSingleSDKAdapter=ObservedSingleSDKAdapter,
        ObserverStore=ObserverStore, callback_source=callback_source, original=original,
        backend=backend, logic=logic, serial=serial, r=r, signal=signal, time=time,
        result_exit_code=result_exit_code, FILES=FILES)

def result_exit_code(result):
    """Do not reuse the old CLI's child-only exit policy for an observer run."""
    code = result['terminal']['returncode']
    require(type(code) is int, 'Actual child exit code')
    if result.get('diagnostic_passed') is True:
        require(code == 0, 'No passing diagnosis with failed SDK child')
        return 0
    return code if 0 < code <= 255 else 1
