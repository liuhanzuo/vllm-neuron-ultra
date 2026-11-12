"""Proposed diagnostic-only BF16 identity DMA; import only inside owned leaf.

No native execution has qualified this source. The future leaf passes the
exact conv cache reshaped to (384,128), then restores (4,3072,4).
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit(platform_target='trn1')
def qwen35_conv_identity_dma(cache):
    assert cache.shape == (384, 128)
    assert cache.dtype == nl.bfloat16
    output = nl.ndarray((384, 128), dtype=cache.dtype, buffer=nl.shared_hbm)
    ip = nl.arange(128)[:, None]
    jf = nl.arange(128)[None, :]
    for tile in nl.static_range(3):
        value = nl.load(cache[tile * 128 + ip, jf])
        nl.store(output[tile * 128 + ip, jf], value=value)
    return output
