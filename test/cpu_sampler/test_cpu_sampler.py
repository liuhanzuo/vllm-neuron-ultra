# SPDX-License-Identifier: Apache-2.0
"""Real torch/vLLM tests, outside test/unit's SDK mocks.

These tests use vLLM's native CPU reference without invoking torch.compile or
loading a Neuron model. Run only in the serialized validation slot on Trn1.
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import (TopKTopPSampler,
                                                apply_top_k_top_p)
from vllm.v1.sample.sampler import Sampler

from vllm_neuron.worker import cpu_sampler
from vllm_neuron.worker.cpu_sampler import (BoundedTopKTopPSampler, _shortlist,
                                           maybe_enable_cpu_sampler)


def make_reference(mode="raw_logprobs"):
    reference = TopKTopPSampler(mode)
    # Match Neuron's native dispatch without CPU torch.compile warm-up.
    reference.forward = reference.forward_native
    return reference


def assert_compact_distribution(monkeypatch, logits, k, p):
    """Inspect probabilities passed to random_sample, before its mutation."""
    expected = apply_top_k_top_p(logits.clone(), k.clone(),
                                None if p is None else p.clone())
    compact_probs = []

    def capture(probs, generators):
        assert not generators
        compact_probs.append(probs[0].clone())
        return probs.argmax(dim=-1)

    monkeypatch.setattr(cpu_sampler, "random_sample", capture)
    actual, processed = BoundedTopKTopPSampler(make_reference())(
        logits.clone(), {}, k.clone(), None if p is None else p.clone())
    assert processed is None
    candidates = _shortlist(logits, k, p)
    capture_index = 0
    for row, (token_ids, _) in enumerate(candidates):
        assert torch.isfinite(expected[row, actual[row]])
        if token_ids is None:
            continue
        # Scatter the ACTUAL COMPACT probabilities. Computing softmax after
        # scattering logits would hide compact/full reduction-order errors.
        probabilities = compact_probs[capture_index]
        capture_index += 1
        mapped = torch.zeros_like(logits[row])
        mapped[token_ids] = probabilities
        support = torch.zeros_like(logits[row], dtype=torch.bool)
        support[token_ids] = True
        assert torch.equal(support, torch.isfinite(expected[row]))
        torch.testing.assert_close(mapped, expected[row].softmax(-1),
                                   rtol=0, atol=1e-6)
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.),
                                   rtol=0, atol=1e-6)
    assert capture_index == len(compact_probs)
    return capture_index


@pytest.mark.parametrize("width", [8, 256, 128000])
@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("p_value", [None, .9, 1.])
def test_actual_compact_probabilities(monkeypatch, width, rounded, p_value):
    rng = torch.Generator().manual_seed(7321)
    logits = torch.randn((4, width), generator=rng) * 2
    if rounded:
        logits = logits.bfloat16().float()
    k = torch.tensor([1, min(4, width), min(20, width), min(64, width)])
    p = None if p_value is None else torch.full((4,), p_value)
    assert assert_compact_distribution(monkeypatch, logits, k, p) > 0


def test_masked_tokens_and_noncontiguous_logits(monkeypatch):
    rng = torch.Generator().manual_seed(10)
    logits = torch.randn((256, 2), generator=rng).t()
    logits[:, 50:] = -torch.inf
    assert not logits.is_contiguous()
    assert assert_compact_distribution(
        monkeypatch, logits, torch.tensor([4, 20], dtype=torch.int32),
        torch.tensor([.9, .95])) > 0


def test_kth_ties_are_retained(monkeypatch):
    logits = torch.full((1, 256), -10.)
    logits[0, :3] = 2.
    assert assert_compact_distribution(
        monkeypatch, logits, torch.tensor([1]), None) == 1
    ids, _ = _shortlist(logits, torch.tensor([1]), None)[0]
    assert set(ids.tolist()) == {0, 1, 2}


@pytest.mark.parametrize("case", ["many_ties", "nucleus_tie", "cdf_boundary",
                                  "nan", "positive_inf", "negative_inf"])
def test_ambiguous_or_nonfinite_rows_use_reference(case):
    logits = torch.full((1, 256), -20.)
    k, p = torch.tensor([4]), torch.tensor([.3])
    if case == "many_ties":
        logits.zero_()
    elif case == "nucleus_tie":
        logits[0, :4] = torch.tensor([-1., -1., 0., 0.])
    elif case == "cdf_boundary":
        logits[0, :2] = 0.
        k, p = torch.tensor([2]), torch.tensor([.5])
    else:
        logits[0, 0] = {"nan": float("nan"), "positive_inf": float("inf"),
                        "negative_inf": -float("inf")}[case]
        if case == "negative_inf":
            logits.fill_(-torch.inf)
    assert _shortlist(logits, k, p) == [(None, None)]
    reference = make_reference()
    sentinel = (torch.tensor([7]), None)
    reference.forward = Mock(return_value=sentinel)
    result = BoundedTopKTopPSampler(reference)(logits, {}, k, p)
    assert torch.equal(result[0], sentinel[0])
    reference.forward.assert_called_once()


def test_mixed_rows_fall_back_individually(monkeypatch):
    logits = torch.stack([torch.linspace(-4, 4, 256), torch.zeros(256),
                          torch.linspace(4, -4, 256)])
    k, p = torch.tensor([20, 40, 1]), torch.tensor([.9, .9, 1.])
    reference = make_reference()
    original = reference.forward
    reference.forward = Mock(wraps=original)
    captured = []

    def capture(probs, generators):
        captured.append(probs.shape)
        return probs.argmax(dim=-1)

    monkeypatch.setattr(cpu_sampler, "random_sample", capture)
    result, _ = BoundedTopKTopPSampler(reference)(logits, {}, k, p)
    assert result.shape == (3,)
    assert len(captured) == 2
    reference.forward.assert_called_once()
    args = reference.forward.call_args.args
    assert torch.equal(args[0], logits[1:2])
    assert torch.equal(args[2], k[1:2])
    assert torch.equal(args[3], p[1:2])


@pytest.mark.parametrize("case", [
    "k_none", "k_float", "k_bool", "k_scalar", "k_matrix", "k_short",
    "k_zero", "k_large", "k_meta", "p_double", "p_integer", "p_scalar",
    "p_matrix", "p_short", "p_zero", "p_large", "p_nan", "p_inf", "p_meta",
    "logits_half", "logits_rank3", "logits_meta", "empty_batch", "empty_vocab",
])
def test_unsupported_calls_preserve_reference_arguments(case):
    logits = torch.zeros(2, 256)
    k, p = torch.tensor([20, 40]), torch.tensor([.9, .95])
    changes = {
        "k_none": (None, p), "k_float": (k.float(), p),
        "k_bool": (k.bool(), p), "k_scalar": (k[0], p),
        "k_matrix": (k[:, None], p), "k_short": (k[:1], p),
        "k_zero": (torch.tensor([0, 40]), p),
        "k_large": (torch.tensor([65, 40]), p),
        "k_meta": (k.to("meta"), p), "p_double": (k, p.double()),
        "p_integer": (k, p.long()), "p_scalar": (k, p[0]),
        "p_matrix": (k, p[:, None]), "p_short": (k, p[:1]),
        "p_zero": (k, torch.tensor([0., .9])),
        "p_large": (k, torch.tensor([1.1, .9])),
        "p_nan": (k, torch.tensor([float("nan"), .9])),
        "p_inf": (k, torch.tensor([float("inf"), .9])),
        "p_meta": (k, p.to("meta")),
    }
    if case in changes:
        k, p = changes[case]
    elif case == "logits_half":
        logits = logits.half()
    elif case == "logits_rank3":
        logits = logits[:, :, None]
    elif case == "logits_meta":
        logits = logits.to("meta")
    elif case == "empty_batch":
        logits, k, p = logits[:0], k[:0], p[:0]
    elif case == "empty_vocab":
        logits = logits[:, :0]
    reference = make_reference()
    sentinel = (torch.tensor([9]), object())
    reference.forward = Mock(return_value=sentinel)
    generators = {}
    result = BoundedTopKTopPSampler(reference)(logits, generators, k, p)
    assert result is sentinel
    reference.forward.assert_called_once()
    assert all(actual is expected for actual, expected in zip(
        reference.forward.call_args.args, (logits, generators, k, p)))


@pytest.mark.parametrize("mode", ["processed_logits", "processed_logprobs"])
def test_processed_outputs_and_rng_are_forwarded(mode):
    logits = torch.linspace(-4, 4, 512).reshape(2, 256)
    k, p = torch.tensor([20, 40]), torch.tensor([.9, .95])
    reference = make_reference(mode)
    wrapped = BoundedTopKTopPSampler(reference)
    torch.manual_seed(32)
    expected = reference(logits.clone(), {}, k.clone(), p.clone())
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(32)
    actual = wrapped(logits.clone(), {}, k.clone(), p.clone())
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize("seeded_rows", [[0], [0, 1]])
def test_explicit_generators_preserve_whole_call_and_rng(seeded_rows):
    logits = torch.linspace(-4, 4, 512).reshape(2, 256)
    k, p = torch.tensor([20, 40]), torch.tensor([.9, .95])
    reference = make_reference()
    wrapped = BoundedTopKTopPSampler(reference)

    def run(operation):
        generators = {row: torch.Generator().manual_seed(32 + row)
                      for row in seeded_rows}
        torch.manual_seed(91)
        result = operation(logits.clone(), generators, k.clone(), p.clone())
        states = [generators[row].get_state() for row in seeded_rows]
        return result, states, torch.random.get_rng_state()

    expected, expected_states, expected_global = run(reference)
    actual, actual_states, actual_global = run(wrapped)
    assert torch.equal(actual[0], expected[0])
    assert all(torch.equal(a, b) for a, b in zip(actual_states, expected_states))
    assert torch.equal(actual_global, expected_global)


def test_enabling_is_opt_in_instance_local_and_idempotent(monkeypatch):
    monkeypatch.delenv("NEURON_CPU_SAMPLER_FAST", raising=False)
    sampler, other = Sampler(), Sampler()
    original, other_original = sampler.topk_topp_sampler, other.topk_topp_sampler
    maybe_enable_cpu_sampler(sampler)
    assert sampler.topk_topp_sampler is original
    monkeypatch.setenv("NEURON_CPU_SAMPLER_FAST", "1")
    maybe_enable_cpu_sampler(sampler)
    wrapped = sampler.topk_topp_sampler
    assert isinstance(wrapped, BoundedTopKTopPSampler)
    assert wrapped.reference is original
    assert other.topk_topp_sampler is other_original
    maybe_enable_cpu_sampler(sampler)
    assert sampler.topk_topp_sampler is wrapped


@pytest.mark.skipif(not os.environ.get("NEURON_SAMPLER_CAPTURE_PATHS"),
                    reason="Optional real-logit captures were not supplied")
def test_real_logit_captures_compact_probabilities(monkeypatch):
    """Replay frozen captures when provided by the qualification runner."""
    fast_rows = 0
    for filename in os.environ["NEURON_SAMPLER_CAPTURE_PATHS"].split(os.pathsep):
        frames = torch.load(Path(filename), map_location="cpu", weights_only=True)
        assert frames
        for batch in (1, 4):
            for start in range(0, len(frames), batch):
                logits = torch.stack([frame["logits"].flatten()
                                      for frame in frames[start:start + batch]])
                logits = logits.float() / .7
                for k_value, p_value in ((1, 1.), (20, .95), (40, .9), (64, .5)):
                    fast_rows += assert_compact_distribution(
                        monkeypatch, logits,
                        torch.full((logits.shape[0],), k_value, dtype=torch.long),
                        torch.full((logits.shape[0],), p_value))
    assert fast_rows > 0
