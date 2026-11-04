# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded top-k sampling for Neuron's host CPU sampler.

Only the per-instance top-k/top-p operation is replaced. Logit processors,
temperature, penalties, greedy sampling, and raw logprobs remain in vLLM.
The compact distribution consumes fewer random values than the reference;
calls with explicit request generators therefore retain the reference path.
"""

import os

import torch
import torch.nn as nn
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

_CANDIDATE_CAP = 128
_MAX_TOP_K = 64
_CDF_MARGIN = 1e-4


def _shortlist(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Return compact token IDs/scores, or (None, None) for fallback rows.

    Keep all ties at the kth value. Fall back when the shortlist cannot hold
    them, or when floating-point reduction order or tied-token sort order
    could change the nucleus boundary. Negative infinity is allowed for
    masked-out tokens, but the maximum and kth values must be finite.
    """
    width = logits.shape[1]
    count = min(_CANDIDATE_CAP + 1, width)
    values, indices = logits.topk(count, dim=-1, sorted=True)
    result = []
    for row in range(logits.shape[0]):
        threshold = values[row, int(k[row]) - 1]
        if not (torch.isfinite(values[row, 0])
                and torch.isfinite(threshold)):
            result.append((None, None))
            continue
        if width > _CANDIDATE_CAP and values[row, -1] >= threshold:
            result.append((None, None))
            continue

        eligible = values[row] >= threshold
        scores = values[row, eligible].flip(0)
        token_ids = indices[row, eligible].flip(0)
        if p is not None:
            cutoff = 1 - p[row]
            cdf = scores.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
            if torch.any((cdf - cutoff).abs() <= _CDF_MARGIN):
                result.append((None, None))
                continue
            keep = cdf > cutoff
            keep[-1] = True
            first = int(torch.nonzero(keep, as_tuple=False)[0, 0])
            if first and scores[first] == scores[first - 1]:
                result.append((None, None))
                continue
            scores = scores[keep]
            token_ids = token_ids[keep]
        result.append((token_ids, scores))
    return result


class BoundedTopKTopPSampler(nn.Module):
    """Wrap the original operation, retaining it for unsupported calls/rows.

    Qualification is limited to CPU float32 logits, integer top-k in [1, 64],
    and optional CPU float32 top-p in (0, 1]. Explicit generators and processed
    logits/logprobs always use the entire original call, preserving its output
    and RNG behavior. Unseeded draws need not match the original sequence.
    """

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.reference = reference

    def _supported(self, logits: torch.Tensor, k: torch.Tensor | None,
                   p: torch.Tensor | None) -> bool:
        if (logits.device.type != "cpu" or logits.dtype != torch.float32
                or logits.layout != torch.strided or logits.ndim != 2
                or logits.shape[0] == 0 or logits.shape[1] == 0
                or getattr(self.reference, "logprobs_mode", None)
                not in ("raw_logits", "raw_logprobs")):
            return False
        if (k is None or k.device.type != "cpu"
                or k.dtype not in (torch.int32, torch.int64)
                or k.layout != torch.strided or k.shape != logits.shape[:1]):
            return False
        if not bool(torch.all((k >= 1)
                              & (k <= min(_MAX_TOP_K, logits.shape[1])))):
            return False
        if p is not None:
            if (p.device.type != "cpu" or p.dtype != torch.float32
                    or p.layout != torch.strided
                    or p.shape != logits.shape[:1]):
                return False
            if not bool(torch.all((p > 0) & (p <= 1))):
                return False
        return True

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if generators or not self._supported(logits, k, p):
            return self.reference(logits, generators, k, p)

        # _supported established that k is a nonempty CPU integer vector.
        assert k is not None
        selected = []
        for row, (token_ids, scores) in enumerate(_shortlist(logits, k, p)):
            if token_ids is None:
                sampled, _ = self.reference(
                    logits[row:row + 1], {}, k[row:row + 1],
                    None if p is None else p[row:row + 1])
                selected.append(sampled[0])
            else:
                assert scores is not None
                probs = scores.softmax(dim=-1, dtype=torch.float32).unsqueeze(0)
                sampled = random_sample(probs, {})[0]
                selected.append(token_ids[sampled])
        return torch.stack(selected), None


def maybe_enable_cpu_sampler(sampler: nn.Module) -> None:
    """Enable only this sampler instance when explicitly requested."""
    if os.environ.get("NEURON_CPU_SAMPLER_FAST", "0") != "1":
        return
    if not isinstance(sampler.topk_topp_sampler, BoundedTopKTopPSampler):
        sampler.topk_topp_sampler = BoundedTopKTopPSampler(
            sampler.topk_topp_sampler)
