# SPDX-License-Identifier: Apache-2.0
"""Create a new LFM3 TF4 tokenizer view; never alter a source or existing view."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from .identity import (MODEL_ID, REVISION, VIEW_SCHEMA, SOURCE_FILES, TOKENIZER_VIEW_SHA256,
                       VIEW_MANIFEST, digest_file, read_json, require,
                       tokenizer_fixture_identity, verify_files)


def compatible_tokenizer_config(config):
    require(config.get("tokenizer_class") == "TokenizersBackend", "Unexpected original tokenizer class")
    require(config.get("extra_special_tokens") == [], "Unexpected original extra special tokens")
    result = dict(config)
    standard = {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}
    named = {key: value for key, value in config.items()
             if key.endswith("_token") and key not in standard and isinstance(value, str)}
    result["tokenizer_class"] = "PreTrainedTokenizerFast"
    result["extra_special_tokens"] = named
    return result


def validate_tokenizer(tokenizer, fixtures):
    require(fixtures["revision"] == REVISION, "Tokenizer fixture revision differs")
    require(len(fixtures["hf_prompts"]) == 6 and len(fixtures["quality_inputs"]) == 96,
            "Require all six HF and all 96 quality inputs")
    for row in fixtures["hf_prompts"] + fixtures["quality_inputs"]:
        actual = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        require(actual == row["input_ids"], "Token IDs changed for fixture " + str(row["id"]))
        if "max_tokens" in row:
            require(len(actual) + row["max_tokens"] <= 2048, "Quality fixture exceeds original context")
    require(sorted(tokenizer.all_special_ids) == sorted(fixtures["all_special_ids"]),
            "Special-token IDs changed")
    for row in fixtures["decode_checks"]:
        require(tokenizer.decode(row["ids"], skip_special_tokens=row["skip_special_tokens"]) == row["text"],
                "Token strings/decoding changed")
    return {"hf_prompts": 6, "quality_inputs": 96}


def load_tokenizer(path):
    require(importlib.metadata.version("transformers") == "4.57.6", "Use the pinned Transformers 4.57.6")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), "Destination must not exist")
    require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
            "Create an ordinary destination parent explicitly")
    # Full streaming source verification belongs to an explicit serialized
    # preparation, never a source-only check or timed request.
    verify_files(source, SOURCE_FILES)
    config = compatible_tokenizer_config(read_json(source / "tokenizer_config.json"))
    fixtures = read_json(Path(__file__).with_name("tokenizer_fixtures.json"))
    destination.mkdir(parents=False, exist_ok=False)
    (destination / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    for name in SOURCE_FILES.keys() - {"tokenizer_config.json"}:
        (destination / name).symlink_to(source / name)
    require(digest_file(destination / "tokenizer_config.json") == TOKENIZER_VIEW_SHA256,
            "Tokenizer view differs from the recorded compatibility transform")
    checks = validate_tokenizer(load_tokenizer(destination), fixtures)
    verify_files(destination, {name: row for name, row in SOURCE_FILES.items() if name != "tokenizer_config.json"})
    # Publish completion only after all inputs match. Leave failed partial
    # directories untouched for diagnosis; never repair/replace automatically.
    with (destination / VIEW_MANIFEST).open("x") as stream:
        json.dump({"schema": VIEW_SCHEMA, "model": MODEL_ID, "revision": REVISION,
                   "source_files": SOURCE_FILES, "tokenizer_config_sha256": TOKENIZER_VIEW_SHA256,
                   "tokenizer_fixtures_sha256": tokenizer_fixture_identity(), "tokenizer_checks": checks}, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
