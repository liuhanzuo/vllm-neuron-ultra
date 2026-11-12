"""Frozen native backend and Store with one pure budget calculation shared.

Only reserve_runtime changes. Fresh actual admission readback, identity checks,
guard checkpoints, writes, observers, native authority and waits are inherited.
No current binding, IO, native operation, process or lock action occurs here.
"""
from research import minicpm5_sdk_mapping_observer_backend_v4 as previous
from research import minicpm5_formal_literal_budget_v1 as prepared

make_ports = previous.make_ports
NativePorts = previous.NativePorts
ORIGINAL_NAMES, OBSERVER_NAMES = previous.ORIGINAL_NAMES, previous.OBSERVER_NAMES


def make_store_class(original):
    base = previous.make_store_class(original)
    require = previous.require
    logic = previous.literal_backend.logic

    class PreparedBudgetStore(base):
        def reserve_runtime(self, value, reference, argv):
            require(self.runtime_reservation is None, 'One complete pre-spawn reservation')
            inventory = self.inventory()
            require(inventory['names'] == [], 'Reserve all original runtime artifacts before admission write')
            raw, expected = self.prospective_admission(value)
            require(reference == expected, 'Reservation binds actual proposed wire reference')
            budget = prepared.combined_budget(raw, argv, self.literal, str(self.root))
            require(self.logical_metadata + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    <= original.serial.CAPS['metadata_bytes'], 'Original complete metadata reservation')
            require(self.allocated + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    + 6 * 65536 <= original.serial.CAPS['allocated_bytes'], 'Original complete allocation reservation')
            self.runtime_reservation = budget
            self.reserved_admission = dict(reference)
            self.reserved_argv = list(argv)
            return dict(budget)

    return PreparedBudgetStore
