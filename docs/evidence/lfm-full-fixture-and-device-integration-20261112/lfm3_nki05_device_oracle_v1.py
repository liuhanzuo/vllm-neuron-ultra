"""Original package Torch sequential oracle, imported only in admitted device leaves."""


def store_slots(cache,updates,slots):
    import torch
    indices=torch.arange(cache.shape[0],device=cache.device)
    for row in range(updates.shape[0]):
        selected=indices==slots[row]
        cache=torch.where(selected.reshape((-1,)+(1,)*(cache.ndim-1)),updates[row:row+1],cache)
    return cache
