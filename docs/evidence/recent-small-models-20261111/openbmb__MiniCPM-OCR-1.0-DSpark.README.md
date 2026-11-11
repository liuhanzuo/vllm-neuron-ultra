---
license: apache-2.0
language:
  - en
  - zh
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - minicpm-ocr
  - dspark
  - speculative-decoding
  - draft-model
  - ocr
  - multimodal
base_model: openbmb/MiniCPM-OCR-1.0
base_model_relation: finetune
---

<div align="center">

<p align="center">
  <a href="https://github.com/OpenBMB/MiniCPM"><img src="https://raw.githubusercontent.com/OpenBMB/MiniCPM-OCR/main/assets/minicpm-ocr/minicpm-logo.jpg" width="300" alt="MiniCPM logo"></a>
</p>

<p align="center">
  <b>Usage Guide</b>
</p>

<table align="center" width="100%">
  <tr>
    <td align="center" width="25%" valign="top">
      <b>🚀 Free API</b><br/>
      <sub>Try it online for free, then call the API from your own code.</sub><br/>
      Playground (<a href="https://platform.modelbest.cn/console/playground/document-parse">CN</a> /
      <a href="https://platform.modelbest.ai/console/playground/document-parse">Global</a>)<br/>
      API docs (<a href="https://platform.modelbest.cn/console/docs/api/document-parse">CN</a> /
      <a href="https://platform.modelbest.ai/console/docs/api/document-parse">Global</a>)
    </td>
    <td align="center" width="25%" valign="top">
      <b>🤖 Agent Skill</b><br/>
      <sub>One prompt: the agent installs it and parses your first document.</sub><br/>
      <a href="https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/SKILL_en.md">SKILL.md</a>
    </td>
    <td align="center" width="25%" valign="top">
      <b>💻 Self-hosting</b><br/>
      <sub>vLLM / SGLang / llama.cpp / Ollama, plus an official Docker image. Data stays on your side.</sub><br/>
      <a href="https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/minicpm_ocr_tutorial_en.md">Deployment guide</a>
    </td>
    <td align="center" width="25%" valign="top">
      <b>📱 Mobile App</b><br/>
      <sub>On-device and offline. Document parsing is iOS-only for now.</sub><br/>
      <a href="https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/README.md#11-ios-demo">iOS Demo</a>
    </td>
  </tr>
</table>

<p align="center">
  Demo (<a href="https://platform.modelbest.cn/console/playground/document-parse">CN</a> /
  <a href="https://platform.modelbest.ai/console/playground/document-parse">Global</a>) &nbsp;·&nbsp;
  <a href="https://github.com/OpenBMB/MiniCPM-OCR">GitHub</a> &nbsp;·&nbsp;
  <a href="https://github.com/OpenBMB/MiniCPM-OCR#using-minicpm-ocr">SDK usage</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark">DSpark</a> &nbsp;·&nbsp;
  <a href="https://discord.gg/JE8KQqtdn">Discord</a>
  &nbsp;|&nbsp; <a href="https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark/blob/main/README_zh.md">中文</a>
</p>

</div>

# MiniCPM-OCR 1.0 DSpark

> **This repository hosts the DSpark draft (speculative decoding) head for
> [MiniCPM-OCR 1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0).** It is a draft model, not a standalone OCR model —
> it must be paired with the matching main checkpoint.

**DSpark** speeds up decoding with a small draft model that proposes tokens the main model then
verifies in one pass. With this head, MiniCPM-OCR 1.0 improves **end-to-end processing efficiency
by 55%**; throughput reaches about **1.49×** PaddleOCR-VL-1.6 and **3.62×** MinerU2.5 Pro.

## Specification

| Property | Value |
| :--- | :--- |
| Target model | MiniCPM-OCR 1.0 (1.3B) |
| Draft parameters | 296,837,889 |
| Draft layers | 5 |
| Draft tokens per forward pass | 15 |
| Target hidden-state layers | `[3, 7, 11, 15, 19]` |
| Checkpoint precision | BF16 |
| Input vocabulary | 248,100 |
| Draft output vocabulary | 114,688 |

The output LM head and the Markov output rows are cropped to 114,688 tokens. The checkpoint keeps
its `d2t` offset mapping and the full input vocabulary; `local-to-target.json` holds the
corresponding absolute target token IDs.

> **Note on `d2t`**: in this checkpoint the tensor is *delta*-encoded
> (`target_id - draft_id`). It must be converted to absolute IDs (`d2t[i] + i`) before use — the
> GGUF build already does this, and `local-to-target.json` is the ground truth to verify against.

## Parsing pipeline

MiniCPM-OCR 1.0 parses full documents as a **two-stage pipeline** orchestrated by the [`minicpm-ocr` SDK](https://pypi.org/project/minicpm-ocr/). The two stages use separate models, run in separate environments, and may run on different machines:

- **Stage 1 — Layout detection (SDK environment).** The PP-DocLayoutV3 layout model detects document regions — text, tables, formulas, charts, seals — and determines reading order; the SDK crops each region. It runs locally inside the SDK environment and requires PaddlePaddle / PaddleX.
- **Stage 2 — Region recognition (inference service).** The SDK sends each cropped region over HTTP to a MiniCPM-OCR 1.0 inference service — served by vLLM, SGLang, llama.cpp, or Ollama — with a prompt matching the region type, then assembles the recognized content into the final Markdown / JSON in reading order.

<p align="center"><img src="assets/parsing_pipeline.svg" alt="Parsing pipeline: in the minicpm-ocr SDK environment, the PP-DocLayoutV3 layout model detects and crops document regions; each cropped region is sent over HTTP to a separate recognition service running MiniCPM-OCR 1.0 (vLLM, SGLang, llama.cpp, or Ollama); the SDK then assembles the recognized text, tables, formulas, charts, and seals into Markdown and JSON in reading order." width="90%"></p>

See the [model guide](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/minicpm_ocr_1dot0_en.md) for more on architecture, parsing workflow, deployment options, and benchmark breakdowns.

## Usage with vLLM

```bash
vllm serve openbmb/MiniCPM-OCR-1.0 \
  --served-model-name MiniCPM-OCR \
  --trust-remote-code \
  --max-model-len 40960 \
  --max-num-batched-tokens 40960 \
  --max-num-seqs 1 \
  --trust-request-mm-kwargs \
  --mm-processor-kwargs '{"downsample_mode":"4x","max_slice_nums":9}' \
  --speculative-config '{"method":"dspark",
                         "model":"openbmb/MiniCPM-OCR-1.0-DSpark",
                         "num_speculative_tokens":15,
                         "enable_adaptive_verification":false}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

Selection and tuning: [DSpark acceleration](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/minicpm_ocr_tutorial_en.md#3-start-the-service).


## Related checkpoints

| Format | Hugging Face | ModelScope |
| :--- | :--- | :--- |
| BF16 (original) | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0) |
| GGUF (llama.cpp) | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-gguf) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-gguf) |
| AWQ W4A16 | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-AWQ) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-AWQ) |
| GPTQ W4A16 | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-GPTQ) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-GPTQ) |
| DSpark draft (BF16) **(this repo)** | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-DSpark) |
| DSpark draft (GGUF) | [🤗](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark-gguf) | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0-DSpark-gguf) |

DSpark requires a **matching main and draft model from the same release**; do not pair this draft
with a different main checkpoint.

## FAQ

Installation, model downloads, GPU memory, input formats, and unexpected outputs:
[FAQ](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/faq_en.md).

## License

Model weights, code, and documentation are released under the
[Apache License 2.0](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/LICENSE).
