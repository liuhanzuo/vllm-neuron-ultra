# SPDX-License-Identifier: Apache-2.0
"""Explicit pinned model/tokenizer view; never modifies original files."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from .identity import (MODEL_ID, REVISION, SOURCE_FILES, CONFIG_VIEW_SHA256,
    TOKENIZER_VIEW_SHA256, VIEW_MANIFEST, digest_file, read_json, require,
    verify_files, view_manifest)
from .provenance import ORIGINAL_CONFIG


def compatible_model_config(config):
    require(config == ORIGINAL_CONFIG, "Original config differs from pinned revision")
    result = dict(config)
    result.pop("auto_map")
    result.update(model_type="nemotron9b_neuron_text", architectures=["Nemotron9BNeuronTextForCausalLM"],
                  time_step_limit=[0.0, float("inf")])
    return result


def compatible_tokenizer_config(config):
    require(config.get("tokenizer_class") == "PreTrainedTokenizer", "Unexpected original tokenizer class")
    result = dict(config)
    result["tokenizer_class"] = "PreTrainedTokenizerFast"
    return result


def validate_tokenizer(tokenizer, fixtures):
    require(fixtures["revision"] == REVISION and len(fixtures["fixtures"]) == 96 and len(fixtures["plain"]) == 22,
            "Complete original tokenizer fixtures required")
    for row in fixtures["fixtures"]:
        actual = tokenizer.apply_chat_template([{"role":"user", "content":row["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        require(actual == row["input_ids"], "Original chat token IDs changed: " + row["id"])
        require(len(actual) + row["output_budget"] <= 4096, "Original quality context overflow")
    for row in fixtures["plain"]:
        ids = tokenizer.encode(row["text"], add_special_tokens=row["add_special_tokens"])
        require(ids == row["input_ids"], "Original raw token IDs changed")
        require(tokenizer.decode(ids, skip_special_tokens=False) == row["decode"] and
                tokenizer.decode(ids, skip_special_tokens=True) == row["decode_skip_special"], "Original decoding changed")
    require(sorted(tokenizer.all_special_ids) == fixtures["special_token_ids"], "Special-token identity changed")
    vocabulary = hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    require(vocabulary == fixtures["vocabulary_sha256"], "Original full vocabulary changed")
    chat=lambda text,**kwargs:tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,**kwargs)
    default=chat("What is 2+2?")
    require(default == chat("What is 2+2?",enable_thinking=False) and default.endswith("<think>\n"), "Original default thinking changed")
    require(chat("What is 2+2? /no_think").endswith("<think></think>"), "Original no_think control changed")
    return {"quality_inputs":96,"raw_checks":22,"thinking_controls":2}


def load_tokenizer(path):
    require(importlib.metadata.version("transformers") == "4.57.6", "Pinned Transformers4.57.6 required")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path),trust_remote_code=False,local_files_only=True)


def prepare(source,destination):
    source=Path(source).resolve();destination=Path(destination).absolute()
    require(not destination.exists() and not destination.is_symlink(), "Fresh destination required")
    require(destination.parent.is_dir() and destination.parent.resolve()==destination.parent, "Ordinary destination parent required")
    verify_files(source,SOURCE_FILES)
    config=compatible_model_config(read_json(source/"config.json"))
    tokenizer=compatible_tokenizer_config(read_json(source/"tokenizer_config.json"))
    fixtures=read_json(Path(__file__).with_name("tokenizer_fixtures.json"))
    destination.mkdir(exist_ok=False)
    (destination/"config.json").write_text(json.dumps(config,indent=2))
    (destination/"tokenizer_config.json").write_text(json.dumps(tokenizer,ensure_ascii=False,indent=2))
    for name in SOURCE_FILES.keys()-{"config.json","tokenizer_config.json"}:
        (destination/name).symlink_to(source/name)
    require(digest_file(destination/"config.json")==CONFIG_VIEW_SHA256 and
            digest_file(destination/"tokenizer_config.json")==TOKENIZER_VIEW_SHA256, "Exact compatibility transform required")
    checks=validate_tokenizer(load_tokenizer(destination),fixtures)
    verify_files(destination,{n:v for n,v in SOURCE_FILES.items()if n not in ("config.json","tokenizer_config.json")})
    manifest=view_manifest();require(manifest["tokenizer_checks"]==checks,"Tokenizer check scope")
    # Failed preparation retains its partial directory; publish only after validation.
    with (destination/VIEW_MANIFEST).open("x") as f:
        json.dump(manifest,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source");parser.add_argument("destination")
    args=parser.parse_args();prepare(args.source,args.destination)


if __name__=="__main__":main()
