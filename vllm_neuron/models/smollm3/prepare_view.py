# SPDX-License-Identifier: Apache-2.0
"""Create a new pinned SmolLM3 config/tokenizer view; never alter the source."""
import argparse
import json
from pathlib import Path

from .identity import (MODEL_ID, REVISION, SCHEMA, SOURCE_FILES, VIEW_FILES,
                       VIEW_MANIFEST, require, read_json, verify_files)


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(),
            "Destination must not exist; existing model views are never replaced")
    # These streaming hashes include original checkpoint bytes. Run preparation
    # as a serialized build task on a shared benchmark machine.
    verify_files(source, SOURCE_FILES)
    config = read_json(source / "config.json")
    config.update(model_type="smollm3_neuron_text",
                  architectures=["SmolLM3NeuronTextForCausalLM"], head_dim=128)
    tokenizer = read_json(source / "tokenizer_config.json")
    tokenizer["clean_up_tokenization_spaces"] = False
    destination.mkdir(parents=False, exist_ok=False)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    (destination / "tokenizer_config.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False, indent=2))
    for name in SOURCE_FILES.keys() - {"config.json", "tokenizer_config.json"}:
        (destination / name).symlink_to(source / name)
    verify_files(destination, VIEW_FILES)
    # The manifest is the completion marker. Failure preserves the partial
    # directory for diagnosis and never turns it into an admitted model view.
    with (destination / VIEW_MANIFEST).open("x") as stream:
        json.dump({"schema": SCHEMA, "model": MODEL_ID,
                   "revision": REVISION, "files": VIEW_FILES}, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
