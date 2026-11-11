"""Normal-exit lifetime fix for the frozen source-phase seal v1.

The sealing, dynamic checks and failure latch are inherited unchanged. A normal
exit releases temporary snapshots and bound closures without waiting for GC.
"""
from research.minicpm5_native_source_phase_seal_v1 import SourcePhase as OriginalPhase


class SourcePhase(OriginalPhase):
    def __exit__(self, kind, error, traceback):
        result = super().__exit__(kind, error, traceback)
        if kind is None and self.failure is None:
            # super restored the original context API after every exit check.
            # Remove self -> bound method/closure -> self cycles and all full
            # plan/history snapshots before the next model/weight operation.
            self.__dict__.clear()
            self.started = self.closed = self.authenticated = True
            self.failure = None
        return result
