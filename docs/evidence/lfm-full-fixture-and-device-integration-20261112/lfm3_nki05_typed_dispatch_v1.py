"""Explicit uninstalled production dispatch; one-row prefill stays original."""


def make_writer(capacity,original_store_slots,typed_writer):
    assert capacity in (4,8,16) and callable(original_store_slots) and callable(typed_writer)
    def writer(cache,updates,slots):
        assert cache.shape[0]==capacity and 1<=updates.shape[0]<=capacity
        if updates.shape[0]==1:
            return original_store_slots(cache,updates,slots)
        return typed_writer(cache,updates,slots)
    return writer
