# SPDX-License-Identifier: Apache-2.0
"""Explicit serial build/load entry for future qualification, never serving.

The input JSON is the complete NeuronConfig keyword dictionary. Admission pins
BF16, TP2, B4, six context/decode buckets and context4096 before any compilation.
No execution of this entry is part of the source delivery.
"""
import argparse
from .identity import read_json, selected_variant


def build(model_path, neuron_configuration):
    selected_variant()
    from .registration import install
    install()
    from .modeling import NeuronQwen35ForCausalLM
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from .artifacts import load_or_compile
    cls = NeuronQwen35ForCausalLM
    config = cls.get_config_cls()(cls.get_neuron_config_cls()(**neuron_configuration),
                                 load_config=load_pretrained_config(model_path))

    class Owner:
        def _load_compiled_model(self, path, model_class, kwargs):
            self.model = model_class(path)
            self.model.load(path)

    owner = Owner()
    load_or_compile(owner, model_path, cls, config, {})
    return owner.model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model"); parser.add_argument("neuron_config_json")
    args = parser.parse_args()
    build(args.model, read_json(args.neuron_config_json))


if __name__ == "__main__":
    main()
