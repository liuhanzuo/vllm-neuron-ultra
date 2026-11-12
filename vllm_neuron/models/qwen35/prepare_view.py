# SPDX-License-Identifier: Apache-2.0
"""Build a new pinned text configuration view; checkpoint/tokenizer bytes stay unchanged."""
import argparse
import copy
import json
import os
from pathlib import Path
from .identity import (SOURCE_FILES, CONFIG_VIEW_SHA256, VIEW_MANIFEST,
                       digest_file, read_json, require, verify_files, view_manifest)
from .provenance import ORIGINAL_CONFIG


def compatible_model_config(config):
    require(config == ORIGINAL_CONFIG, "Qwen3.5 source config differs from pinned revision")
    result = copy.deepcopy(config["text_config"])
    result.update(model_type="qwen35_neuron_text", architectures=["Qwen35NeuronTextForCausalLM"],
                  torch_dtype="bfloat16", pad_token_id=248044, bos_token_id=None,
                  rope_theta=result["rope_parameters"]["rope_theta"])
    return result


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), "Fresh text view destination required")
    require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
            "Ordinary existing destination parent required")
    verify_files(source, SOURCE_FILES)
    config = compatible_model_config(read_json(source / "config.json"))
    destination.mkdir(exist_ok=False)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    for name in SOURCE_FILES.keys() - {"config.json"}:
        (destination / name).symlink_to(source / name)
    require(digest_file(destination / "config.json") == CONFIG_VIEW_SHA256, "Exact text view transform required")
    verify_files(destination, {k: v for k, v in SOURCE_FILES.items() if k != "config.json"})
    manifest = view_manifest()
    with (destination / VIEW_MANIFEST).open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source"); parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
