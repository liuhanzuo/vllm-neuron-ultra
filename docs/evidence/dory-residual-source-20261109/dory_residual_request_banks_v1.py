"""Prepared 56-history Dory request ownership, with no numerical imports.

This is a diagnostic CPU replay component, not a production scheduler. Native
admission, fixed model identity, original event replay, raw-byte checks and
independent review remain the caller's responsibility.
"""
from dataclasses import dataclass
from types import MappingProxyType

from dory_residual_visit_capture_v1 import require


CACHE_IDS = (tuple(range(0, 16, 2))
             + tuple(range(16, 56, 2))
             + tuple(range(88, 128, 2))
             + tuple(range(56, 72, 2)))
MAX_POSITION = 262144


@dataclass(frozen=True)
class Handle:
    slot: int
    epoch: int
    request: str

    def __post_init__(self):
        require(type(self.slot) is int and self.slot >= 0,
                "Nonnegative integer request slot")
        require(type(self.epoch) is int and self.epoch > 0,
                "Positive integer request epoch")
        require(isinstance(self.request, str) and bool(self.request),
                "Nonempty request identity")


@dataclass
class _Bank:
    handle: Handle
    alive: bool
    length: int
    histories: dict


class FailureCaptureError(RuntimeError):
    def __init__(self, original, retention):
        super().__init__("Original forward failed and partial-evidence capture failed")
        self.original, self.retention = original, retention


class ResidualRequestBanks:
    def __init__(self, caches, extract_id, validate_tensor, storage_identity):
        """Callbacks must be authenticated by the future native source bundle.

        validate_tensor(tensor, length) must enforce BF16 [4,length,256] and
        returns None on success. storage_identity uses actual tensor data_ptr.
        No tensor data is copied or converted by this ownership component.
        """
        pairs = [(extract_id(cache), cache) for cache in caches]
        require(len(pairs) == 56 and all(type(key) is int for key, _ in pairs)
                and len({key for key, _ in pairs}) == 56
                and set(key for key, _ in pairs) == set(CACHE_IDS)
                and len({id(cache) for _, cache in pairs}) == 56,
                "Exactly 56 distinct original per-loop caches")
        require(all(cache.keys is None and cache.values is None for _, cache in pairs),
                "Original caches empty before replay")
        self.caches = dict(pairs)
        self.validate_tensor, self.storage_identity = validate_tensor, storage_identity
        self._banks, self._epochs = {}, {}
        self._pending = None

    def _idle(self):
        require(self._pending is None, "Request lifecycle cannot change during a forward")

    def _owned(self, handle):
        require(isinstance(handle, Handle), "Explicit request handle required")
        bank = self._banks.get(handle.slot)
        require(bank is not None and bank.alive and bank.handle == handle,
                "Stale, invalidated or foreign request handle")
        return bank

    def clear_bound(self):
        self._idle()
        self._clear_bound()

    def _clear_bound(self):
        for cache in self.caches.values():
            cache.keys = cache.values = None

    def start(self, slot, epoch, request):
        self._idle()
        handle = Handle(slot, epoch, request)
        require(epoch == self._epochs.get(slot, 0) + 1,
                "Request start must advance the original epoch exactly once")
        old = self._banks.get(slot)
        if old is not None:
            old.alive = False
            old.histories.clear()
        self._clear_bound()
        self._banks[slot] = _Bank(handle, True, 0, {})
        self._epochs[slot] = epoch
        return handle

    def release(self, handle):
        self._idle()
        bank = self._owned(handle)
        bank.alive = False
        bank.histories.clear()
        del self._banks[handle.slot]
        self._clear_bound()

    def histories(self, handle):
        self._idle()
        return MappingProxyType(self._owned(handle).histories)

    def inventory(self, fingerprint, exclude=None):
        """The caller chooses actual byte hashes, never storage pointers, here."""
        self._idle()
        return {str(slot): dict(epoch=bank.handle.epoch, request=bank.handle.request,
                               alive=bank.alive, length=bank.length,
                               histories={f"{key}-{j}": fingerprint(tensor)
                                          for key in CACHE_IDS if key in bank.histories
                                          for j, tensor in enumerate(bank.histories[key])})
                for slot, bank in sorted(self._banks.items()) if slot != exclude}

    def _begin(self, handle, start, end):
        self._idle()
        bank = self._owned(handle)
        require(type(start) is int and type(end) is int
                and start == bank.length and start < end <= MAX_POSITION,
                "Original contiguous positions inside context limit")
        require(set(bank.histories) == (set(CACHE_IDS) if start else set()),
                "Complete original histories or empty fresh request")
        # All rejection checks precede rebinding or changes to any request.
        for key, cache in self.caches.items():
            cache.keys, cache.values = bank.histories.get(key, (None, None))
        self._pending = (handle, start, end)

    def _commit(self, handle):
        require(self._pending is not None and self._pending[0] == handle,
                "Only the current forward may commit")
        bank = self._owned(handle)
        end = self._pending[2]
        histories = {}
        identities = []
        for key in CACHE_IDS:
            cache = self.caches[key]
            pair = (cache.keys, cache.values)
            for tensor in pair:
                require(tensor is not None, "Every original cache updated")
                require(self.validate_tensor(tensor, end) is None,
                        "Tensor validator must return None after checking")
                identity = self.storage_identity(tensor)
                require(type(identity) is int and identity > 0,
                        "Actual nonzero tensor storage identity")
                identities.append(identity)
            histories[key] = pair
        require(len(set(identities)) == 112, "Independent 112 K/V tensor storages")
        other = {self.storage_identity(tensor)
                 for slot, other_bank in self._banks.items() if slot != handle.slot
                 for pair in other_bank.histories.values() for tensor in pair}
        require(set(identities).isdisjoint(other), "Request histories must not alias other slots")
        bank.histories = histories
        bank.length = end
        self._pending = None

    def _abort(self, handle):
        require(self._pending is not None and self._pending[0] == handle,
                "Only the current failed request may abort")
        bank = self._owned(handle)
        bank.alive = False
        bank.histories.clear()
        self._clear_bound()
        self._pending = None

    def execute(self, handle, start, end, forward, capture_failure):
        """Run original backbone AND output head in forward, then commit.

        capture_failure(original_error, bound_caches) runs before cleanup so a
        worker can save partial tensors and loop observations. Its own failure
        cannot skip invalidation or turn a failed native call into a success.
        """
        self._begin(handle, start, end)
        try:
            result = forward()
            self._commit(handle)
            return result
        except BaseException as original:
            retention_error = None
            try:
                capture_failure(original, MappingProxyType(self.caches))
            except BaseException as retention:
                retention_error = retention
            finally:
                self._abort(handle)
            if retention_error is not None:
                raise FailureCaptureError(original, retention_error) from original
            raise
