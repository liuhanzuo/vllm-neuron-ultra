"""Typed trace wrapper; native Torch is imported only during an admitted call."""


def apply_typed_writer(cache,updates,slots,torch_api,device_kernel):
    assert cache.dtype in (torch_api.bfloat16,torch_api.float32) and updates.dtype==cache.dtype
    assert slots.dtype in (torch_api.int32,torch_api.int64)
    assert cache.ndim>=2 and updates.ndim==cache.ndim and slots.ndim==1
    capacity,rows=cache.shape[0],updates.shape[0]
    assert capacity in (4,8,16) and 1<=rows<=capacity
    assert updates.shape[1:]==cache.shape[1:] and slots.shape==(rows,)
    assert cache.device==updates.device==slots.device
    assert cache.is_contiguous() and updates.is_contiguous() and slots.is_contiguous()
    width=1
    for size in cache.shape[1:]:
        assert size>0
        width*=size
    assert width%128==0
    # Exact original normalization before narrowing: huge int64 must stay inactive.
    valid=(slots>=0)&(slots<capacity)
    normalized=torch_api.where(valid,slots,torch_api.full_like(slots,-1)).to(torch_api.int32)
    result=device_kernel(cache.reshape(capacity,width),updates.reshape(rows,width),normalized)
    assert result.dtype==cache.dtype and result.shape==(capacity,width)
    return result.reshape(cache.shape)


def make_typed_writer(device_kernel,admit):
    assert callable(device_kernel) and callable(admit)
    def writer(cache,updates,slots):
        assert admit() is True,'Owned phase must remain admitted before Torch input'
        import torch
        result=apply_typed_writer(cache,updates,slots,torch,device_kernel)
        assert admit() is True,'Owned phase must remain admitted after Torch output'
        return result
    return writer
