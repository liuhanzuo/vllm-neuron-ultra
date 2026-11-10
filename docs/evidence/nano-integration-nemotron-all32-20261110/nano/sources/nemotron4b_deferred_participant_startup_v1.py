"""Build actual phase origin policy only after the participant's parent gate.

The caller supplies complete reviewed manifests and a concrete origin builder;
this factory cannot authenticate omitted origins or create serial admission.
The same factory fits native_worker_execution_v4 and guarded_helper_entry_v3.
"""
from contextlib import contextmanager
import copy
import sys

from nemotron4b_participant_invocation_v1 import ParticipantBinding
from nemotron4b_process_handshake_v1 import require


class DeferredParticipantStartup:
    def __init__(self,binding,template,*,build_origin):
        require(type(binding)is ParticipantBinding and callable(build_origin),
                'Exact bound participant and explicit origin builder')
        binding.startup_policy(template)
        self.binding=binding;self.template=copy.deepcopy(template)
        self.builder=build_origin;self.used=False;self.origin=None;self.provenance=None

    @contextmanager
    def __call__(self):
        require(not self.used,'One deferred startup per participant');self.used=True
        binding=self.binding
        require(binding.release is not None,'Actual parent grant before building startup origins')
        binding.guard();binding.attempt.caches[binding.role].check_runtime()
        require(sys.argv==binding.argv[5:],'Original bound wrapper argv before origin construction')
        self.origin=self.builder(binding)
        start=binding.startup_context(self.origin,self.template)
        with start()as provenance:
            self.provenance=provenance
            yield provenance
            require(sys.argv==binding.argv[5:],'Original bound wrapper argv before restored startup exit')
        require(type(provenance)is dict and provenance.get('complete')is True,
                'Actual deferred phase startup completed')
        binding.guard()
