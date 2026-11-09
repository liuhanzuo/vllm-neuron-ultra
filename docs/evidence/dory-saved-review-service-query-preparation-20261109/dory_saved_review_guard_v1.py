"""Prepared guard integration; preserves each original check and its cadence.

Startup uses the original service query. Only after source authentication may
an executor attach the separately authenticated private connection. This module
does not acquire the serial lock, admit a reviewer, or alter another process.
"""
from dory_service_query_auth_v1 import require


class CompleteReviewGuard:
    def __init__(self, *, low, owner, lock, budget, started, initial_nvme,
                 free, memory_free, clock):
        self.low, self.owner, self.lock = low, owner, lock
        self.budget = dict(budget)
        self.started, self.initial_nvme = started, initial_nvme
        self.free, self.memory_free, self.clock = free, memory_free, clock
        self.query = None
        self.closed = self.poisoned = False
        self.original_service_checks = self.persistent_service_checks = 0
        self.successful_guards = 0

    def attach_authenticated_query(self, query):
        require(not self.closed and not self.poisoned and self.query is None,
                'Exactly one service connection after original startup guards')
        require(self.successful_guards > 0 and self.original_service_checks > 0,
                'Original startup guard must run before transport replacement')
        require(query.startup_file_count == 41 and query.startup_bytes == 4326816
                and query.requests == 0 and not query.closed and not query.poisoned,
                'Only fresh fully authenticated prepared transport')
        query.audit()
        self.query = query

    def __call__(self):
        require(not self.closed and not self.poisoned, 'Failed or closed guard cannot resume')
        try:
            self.low.lock_proof(self.owner, self.lock)
            self.low.clear_window()
            if self.query is None:
                self.original_service_checks += 1
                self.low.active_heavy()
            else:
                self.persistent_service_checks += 1
                self.query.check()
            budget = self.budget
            require(self.clock() - self.started < budget['wall_seconds']
                    and self.free('/') >= budget['root_floor']
                    and self.free('/mnt/nvme') >= max(
                        budget['nvme_floor'], self.initial_nvme - budget['nvme_new_bytes'])
                    and self.memory_free() >= budget['memory_available_floor_bytes'],
                    'Full original resource bounds')
            self.successful_guards += 1
        except BaseException:
            self.poisoned = True
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            if self.query is not None:
                try:
                    self.query.close()
                except BaseException:
                    self.poisoned = True
                    raise

    def accounting(self):
        if self.query is not None:
            require(self.query.requests == self.persistent_service_checks,
                    'Every persistent guard made exactly one fresh service request')
        return dict(original_service_checks=self.original_service_checks,
                    persistent_service_checks=self.persistent_service_checks,
                    successful_guards=self.successful_guards,
                    closed=self.closed, poisoned=self.poisoned,
                    poll_results_cached=False, guard_frequency_reduced=False)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires an independently reviewed executor')
