"""YaRN math matching the installed vLLM interpretation of DeepSeek's config.

Frequency ramp follows vLLM's Apache-2.0 rotary_embedding/yarn_scaling_rope.py.
attn_factor multiplies the default YaRN magnitude; it is not HF's final
attention_factor. Original DeepSeek 0528 Qwen3-8B therefore has magnitude1.
"""
import math
import torch
from torch import nn


class DeepSeekQwen3RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope = config.rope_scaling
        assert rope['rope_type'] == 'yarn'
        assert not set(rope)-{'rope_type','type','factor','original_max_position_embeddings','attn_factor','rope_theta'}
        self.dim = config.head_dim
        self.base = config.rope_theta
        self.factor = rope['factor']
        self.original_length = rope['original_max_position_embeddings']
        self.magnitude = (1.0 if self.factor<=1 else 1+.1*math.log(self.factor))*rope.get('attn_factor',1.)
        self.register_buffer('inv_freq',None,persistent=False)

    def get_inv_freqs(self, device):
        frequencies = self.base**(torch.arange(0,self.dim,2,dtype=torch.float32,device=device)/self.dim)
        extrapolated = 1.0/frequencies
        interpolated = 1.0/(self.factor*frequencies)
        def correction(rotations):
            return self.dim*math.log(self.original_length/(rotations*2*math.pi))/(2*math.log(self.base))
        low,high = max(math.floor(correction(32)),0),min(math.ceil(correction(1)),self.dim-1)
        if low==high: high+=.001
        mask = 1-((torch.arange(self.dim//2,dtype=torch.float32,device=device)-low)/(high-low)).clamp(0,1)
        return interpolated*(1-mask)+extrapolated*mask

    def _apply(self, fn, recurse=True):
        original = self.inv_freq
        super()._apply(fn,recurse=recurse)
        if original is not None: self.inv_freq=original.to(device=self.inv_freq.device,dtype=torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq is None: self.inv_freq=self.get_inv_freqs(x.device)
        inv=self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,1)
        phases=(inv@position_ids[:,None,:].float()).transpose(1,2)
        phases=torch.cat([phases,phases],dim=-1)
        return (phases.cos()*self.magnitude).to(x.dtype),(phases.sin()*self.magnitude).to(x.dtype)
