"""Concrete -c SDK start under original observed adapter and original cleanup.

No imported history, native load, child launch, or lock operation at import.
The frozen original observer retains all finish/wait/drain semantics. The only
changed lifecycle method is start(), which records and checks the real new argv.
"""
import copy
import sys
from pathlib import Path
import time
from research import minicpm5_sdk_mapping_observed_adapter_v7 as observed
from research import minicpm5_sdk_mapping_nocache_launch_v4 as launch
from research import minicpm5_formal_literal_codec_v2 as codec


def make_adapter_class(original):
    base=observed.make_adapter_class(original)
    r,serial,require=original.r,original.serial,original.require

    class SourceEntrySDKAdapter(base.ObservedSingleSDKAdapter):
        def __init__(self,*args,sdk_entry_bytes,sdk_entry_ref,**kwargs):
            super().__init__(*args,**kwargs)
            self.sdk_entry_bytes=sdk_entry_bytes
            self.sdk_entry_ref=copy.deepcopy(sdk_entry_ref)

        def start(self):
            require(self.prerequisites_checked and self.state=='new' and self.child is None,
                    'Single fully authenticated original SDK child')
            self.state='starting'
            a=self.authenticated
            plan=self.admission['source_entry_plan']
            require(plan['python']==serial.python_for(1)
                and self.admission['source_entry_binding']['original_worker']==a['spec']['worker']
                and a['context']['pins_plan']['source_sha256'].get(plan['entry_path'])==plan['bootstrap_sources'][plan['entry_path']],
                'New actual source entry is additive to complete original worker/source contract')
            argv=launch.admitted_sdk(self.sdk_entry_bytes,self.sdk_entry_ref,self.admission,
                self.admission_ref,self.outer_deadline)
            self.argv=argv
            self.store.before_spawn(self.admission, self.admission_ref, argv)
            self.guard(force=True);self.observer_guard()
            env=r.p.environment({},'cpu_control',self.lease.admission['attempt_id'])
            self.child=original.subprocess.Popen(argv,env=env,cwd=original.ROOT,
                stdin=original.subprocess.PIPE,close_fds=True,start_new_session=True)
            self.child_started=True
            deadline=min(self.outer_deadline,time.monotonic()+10)
            while True:
                self.observer_guard()
                current=r.p.proc(self.child.pid)
                if current is not None and current['argv']==argv:
                    r.life.process(current,argv,self.lease.admission['outer_owner']['pid'],
                        self.lease.admission['outer_owner']['cgroup'])
                    require(str(Path('/proc/%d/exe'%current['pid']).resolve())==plan['executable'],
                            'Actual SDK310 executable matches separately reviewed worker entry')
                    self.owner=current
                    break
                require(self.child.poll() is None and time.monotonic()<deadline,
                        'Original bounded actual child exec/ACK')
                time.sleep(.01)
            self.store.json('phase-1-owner.json',current)
            payload=dict(schema=original.RELEASE_SCHEMA,admission=self.admission_ref,phase=1,
                owner=current,source_plan={key:a['context']['pins_plan'][key]
                    for key in ('source_contract','source_sha256')},own_gate=None)
            self.gate=serial.encoded(payload)
            require(0<len(self.gate)<=original.PIPE_CAP,'Original whole inherited release packet')
            require(r.p.proc(current['pid'])==current,'Actual acknowledged child remains gated')
            self.state='gated'
            return copy.deepcopy(current),argv


        def _persist_terminal(self):
            # Identical actual drain/once-only persistence policy. Only the
            # bounded stderr representation changes after a failed write.
            require(self.terminal is not None and self.state == 'drained', 'Physical drain before terminal persistence')
            if self.receipt_attempted:
                return
            self.receipt_attempted = True
            try:
                self.store.json('phase-1-exit.json', self.terminal)
                self.receipt_saved = True
            except BaseException as error:
                message = dict(event='sdk_import_diagnostic_exit_receipt_write_failed',
                    actual_parent_observed_terminal=self.terminal, exit_receipt_saved=False,
                    reporting_error_type=type(error).__name__, diagnostic_passed=False)
                try:
                    raw = codec.encode('exit_fallback', message, self.sdk_entry_bytes.decode('utf-8'))
                    require(len(raw) <= original.SUMMARY_CAP, 'Original complete failed-terminal16KiB envelope')
                    print(raw.decode(), end='', file=sys.stderr, flush=True)
                except BaseException:
                    # A second reporting failure never replaces the original
                    # durable-write failure. No truncated report is emitted.
                    raise error
                raise

    base.SourceEntrySDKAdapter=SourceEntrySDKAdapter
    return base
