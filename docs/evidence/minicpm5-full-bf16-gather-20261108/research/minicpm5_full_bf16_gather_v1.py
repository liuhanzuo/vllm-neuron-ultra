"""Original BF16 TP2 input gather plus one full projection, CPU qualification only.

The caller owns the unchanged full weight, process group and lifetime. No global
patching, casting, output reduction, or model loading takes place here.
"""
def full_bf16_projection(torch, local_input, full_weight, group):
    dist = torch.distributed
    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise ValueError('Actual two-rank process group required')
    ranks = dist.get_process_group_ranks(group)
    if ranks != [0, 1] or dist.get_rank(group) != dist.get_rank():
        raise ValueError('Original rank/head order [0,1] required')
    if (local_input.dtype != torch.bfloat16 or full_weight.dtype != torch.bfloat16
            or local_input.device.type != 'cpu' or full_weight.device.type != 'cpu'
            or local_input.ndim != 3 or local_input.shape[-1] != 1024
            or list(full_weight.shape) != [2048, 2048]):
        raise ValueError('Original BF16 input and complete weight required')
    # Collective completion may mutate its buffers outside inference mode.
    with torch.inference_mode(False):
        parts = [torch.empty_like(local_input, memory_format=torch.contiguous_format) for _ in range(2)]
    dist.all_gather(parts, local_input.contiguous(), group=group)
    gathered = torch.cat(parts, dim=-1).contiguous()
    output = torch.nn.functional.linear(gathered, full_weight)
    if output.dtype != torch.bfloat16 or output.shape != gathered.shape:
        raise ValueError('BF16 complete output required')
    return output, gathered
