# SPDX-License-Identifier: Apache-2.0
"""Exact pinned SDK function for AST-only mocked execution; no SDK imports."""
from __future__ import annotations

def load_pretrained_config(
    model_path_or_name: Optional[Union[str, os.PathLike]] = None,
    hf_config: Optional[PretrainedConfig] = None,
):
    """Return a load_config hook for InferenceConfig that loads the config from a PretrainedConfig."""

    def load_config(self: InferenceConfig):
        if (model_path_or_name is None and hf_config is None) or (
            model_path_or_name is not None and hf_config is not None
        ):
            raise ValueError('Please provide only one of "model_path_or_name" or "hf_config"')

        if model_path_or_name is not None:
            config: PretrainedConfig = AutoConfig.from_pretrained(model_path_or_name)
        else:
            config: PretrainedConfig = hf_config
        config_dict = config.to_dict()

        # Fix transformers_version (config.to_dict() sets it to current transformers version).
        if config.transformers_version is not None:
            config_dict["transformers_version"] = config.transformers_version

        # Set torch_dtype in NeuronConfig.
        hf_dtype = config_dict.get("dtype", config_dict.get("torch_dtype", None))
        if hf_dtype is not None:
            if self.neuron_config is not None and not self.neuron_config.overrides_torch_dtype:
                # Update neuron_config's torch_dtype if not overriden by the user.
                self.neuron_config.torch_dtype = hf_dtype
                if isinstance(self.neuron_config.torch_dtype, str):
                    self.neuron_config.torch_dtype = to_torch_dtype(self.neuron_config.torch_dtype)
            config_dict.pop("dtype", None)
            config_dict.pop("torch_dtype", None)

        # Convert nested configs to namespaces.
        for k, v in config_dict.items():
            if isinstance(getattr(config, k), PretrainedConfig):
                config_dict[k] = SimpleNamespace(**v)
                if config.transformers_version is not None:
                    config_dict[k].transformers_version = config.transformers_version

        self.__dict__.update(config_dict)
        if hasattr(config, "attribute_map"):
            self.attribute_map = config.attribute_map

    return load_config
