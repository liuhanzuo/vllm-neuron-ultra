"""Native source-phase integration; original receipt and identity checks remain.

Only text authentication uses the sealed phase. The original context still
deep-copies the plan, and all original metadata/native identity checks run on
plain objects after successful phase exit. Any authentication failure latches.
"""
import copy
from pathlib import Path
from minicpm5_layer1_operator_native_runtime_context_v1 import RuntimeContext as PreviousContext, sources, identity
from research.minicpm5_native_source_phase_seal_v3 import SourcePhase

require = sources.require
_COMPLETE = object()


class RuntimeContext(PreviousContext):
    def __init__(self, plan, plan_sha256, permit, external_guard, *, contract, resolved, lease, ports):
        self._source_phase_args = contract, resolved, lease, ports
        self._source_verification_started = False
        self._source_failure = None
        super().__init__(plan, plan_sha256, permit, external_guard)

    def _deny_source_failure(self, *args, **kwargs):
        raise self._source_failure

    def verify_sources(self):
        try:
            require(not self._source_verification_started, 'Source authentication cannot be retried or repeated')
            self._source_verification_started = True
            completed = self._verify_sources_once()
            require(completed is _COMPLETE and self._sources_checked is False,
                    'All source, admission, receipt and native identity checks reached normal completion')
            self._sources_checked = True
        except BaseException as error:
            if self._source_failure is None:
                self._source_failure = error
            self._sources_checked = False
            self.guard = self._deny_source_failure
            self.external_guard = self._deny_source_failure
            raise self._source_failure

    def _verify_sources_once(self):
        self.guard('native');require(self.permit['role']=='native' and self.permit['rank']==-1,'Original native-only role')
        require(not self._sources_checked,'One complete source authentication before model work')
        def read(ref):
            self.guard()
            require(self.plan['source_sha256'].get(ref['path'])==ref['sha256'],
                'Every classifier metadata reference source-pinned')
            return sources.load(ref)
        partition=sources.classify(self.plan['source_sha256'],self.plan['source_contract'],read)
        contract,resolved,lease,ports=self._source_phase_args
        with SourcePhase(contract,resolved,self,partition,lease,ports,sources) as phase:
            phase.verify_text_sources()
        root=Path(self.plan['output_root']);runtime={}
        def receipt(ref,name):
            self.guard();sources.reference(ref)
            require(ref['path']==str(root/name),'Exact admitted runtime metadata file')
            before=identity(ref['path']);require(before['size']<sources.metadata.LIMIT,'Bounded original runtime receipt')
            value=sources.load(ref);require(identity(ref['path'])==before,'Stable complete runtime metadata read')
            runtime[ref['path']]=before;return value
        permit=self.permit
        a=receipt(permit['runtime_admission'],'admission.json')
        require(a['schema']=='minicpm5_layer1_operator_native_admission_v1'
            and a['source_contract']==self.plan['source_contract']
            and a['overlay']==permit['runtime_plan'] and a['resolved_plan_sha256']==self.plan_sha256
            and a['sdk_operator_recovery']==self.plan['sdk_operator_recovery']==permit['sdk_operator_recovery']
            and a['outer_owner']==permit['outer_owner']
            and a['lock_identity']==permit['lock_identity'] and a['attempt_id']==permit['attempt_id']
            and a['latest_completion']==self.plan['latest_completion']
            and a['completed_sdk_layout']==self.plan['layer1_operator_boundary']
            and a['serial_predecessors']==self.plan['serial_predecessors']
            and a['source_authentication']==permit['source_authentication']
            and a['runtime_admitted']is a['predecessors_checked']is True,
            'Native source authentication and resolved overlay digest bound by original guarded admission')
        authenticated=receipt(a['source_authentication'],'source-authentication.json')
        sources.validate_native_receipt(authenticated,partition,a['outer_owner'],a['lock_identity'],a['attempt_id'])
        sources.check_native_identities(authenticated,partition,self.guard)
        self._source_partition=partition;self._inherited_native_receipt=copy.deepcopy(authenticated)
        self._source_runtime_identities=runtime
        self.guard()
        return _COMPLETE
