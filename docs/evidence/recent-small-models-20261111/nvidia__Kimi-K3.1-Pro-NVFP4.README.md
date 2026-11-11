---
pipeline_tag: image-text-to-text
base_model:
- moonshotai/Kimi-K3.1-Pro
license: other
license_name: nvidia-open-model-agreement
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- Kimi-K3.1-Pro
- quantized
- NVFP4
- nvfp4
---

# Model Overview

## Description:
The NVIDIA Kimi-K3.1-Pro NVFP4 model is the quantized version of Moonshot AI's Kimi-K3.1-Pro model, which is an auto-regressive language model that uses an optimized transformer architecture. Kimi-K3.1-Pro is a natively multimodal Mixture-of-Experts model post-trained from Kimi K3, with 2.8T total parameters, 104B activated parameters, and a 1,048,576-token context window. For more information, please check [here](https://huggingface.co/moonshotai/Kimi-K3.1-Pro). The NVIDIA Kimi-K3.1-Pro NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial or non-commercial use. <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(Kimi-K3.1-Pro) Model Card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) from Moonshot AI.

## License/Terms of Use:
**GOVERNING DOWNLOAD TERMS:** Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The base model Kimi-K3.1-Pro is governed by the [Kimi-K3.1-Pro License](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE). **Kimi-K3.1-Pro**.

## Deployment Geography:
Global <br>

## Use Case:
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

## Release Date:
Huggingface **11/10/2026** via https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4 <br>

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformer <br>
**Network Architecture:**  Multimodal Mixture-of-Experts (MoE) <br>
**Number of Model Parameters:** 2.8T in total and 104B activated <br>
**This model was developed based on [Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro).** <br>

## Input:
**Input Type(s):** Text, Image <br>
**Input Format(s):** String, Red, Green, Blue (RGB) <br>
**Input Parameters:** Text: One-Dimensional (1D); Image: Two-Dimensional (2D) <br>
**Other Properties Related to Input:** Context length up to 1,048,576 tokens. Images are encoded by MoonViT-V2. <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** 1D (One-Dimensional): Sequences <br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** <br>
* **vLLM** <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Supported Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model version is NVFP4 1.0 version and is quantized with nvidia-modelopt **v0.48.0** <br>

## Calibration Dataset:
**Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) <br>
**Data Collection Method by dataset:** Automated. <br>
**Labeling Method by dataset:** Automated. <br>
**Properties:** The cnn_dailymail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The Nemotron-Post-Training-Dataset-v2 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

## Training Dataset:
**Data Modality:** Undisclosed <br>
**Data Collection Method by dataset:** Undisclosed <br>
**Labeling Method by dataset:** Undisclosed <br>
**Properties:** Undisclosed

## Testing Dataset:
**Data Collection Method by dataset:** Not Applicable <br>
**Labeling Method by dataset:** Not Applicable <br>
**Properties:** Not Applicable. NVIDIA did not perform a separate testing-dataset pass for this quantized checkpoint. <br>

## Evaluation Dataset:
**Datasets:** GPQA Diamond, SciCode, Tau2-bench Telecom, MMMU-Pro, AA-LCR, and Terminal-Bench 2.1 <br>
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected <br>
**Labeling Method by dataset:** Hybrid: Manually-Labeled, Automated <br>
**Properties:** We evaluated the model on text-based reasoning and multimodal benchmarks: GPQA Diamond contains 448 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry;  MMMU Pro is the more challenging version of the Massive Multi-discipline Multimodal Understanding benchmark, measuring college-level multimodal reasoning across diverse disciplines with expanded answer choices and a vision-only input setting. <br>

## Inference:
**Acceleration Engine:** **vLLM** <br>
**Test Hardware:** **NVIDIA Blackwell GB300** <br>

## Post Training Quantization
This model was obtained by quantizing the weights and activations of Kimi-K3.1-Pro to NVFP4 data type, ready for inference with **vLLM**. The source checkpoint stores MXFP4 weights. The routed MoE expert weights are converted from MXFP4 to NVFP4 and the input_scales are set to 1.0. The shared experts and the KDA attention are quantized to MXFP8 and the activation scales are learnt on the fly. The MLA layers are kept in BF16 for the best accuracy recovery.
## Usage

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), you can start the docker `vllm/vllm-openai:kimi-k31-cu134` and run the sample command below:

```sh
vllm serve nvidia/Kimi-K3.1-Pro-NVFP4 --port 8000 --tensor-parallel-size 8 --max-model-len 1048576 --max-num-seqs 256 --load-format fastsafetensors --trust-remote-code --reasoning-parser kimi_k3 --enable-auto-tool-choice --tool-call-parser kimi_k3
```

## Evaluation
<table>
  <tr>
    <td><strong>Precision</strong></td>
    <td><strong>GPQA Diamond</strong></td>
    <td><strong>AA-LCR</strong></td>
    <td><strong>IFBench</strong></td>
    <td><strong>MMMU Pro</strong></td>
    <td><strong>SciCode</strong></td>
    <td><strong>TB2.1</strong></td>
  </tr>
  <tr>
    <td>MXFP4</td>
    <td><strong>93.97</strong></td>
    <td><strong>81.69</strong></td>
    <td><strong>73.40</strong></td>
    <td><strong>81.68</strong></td>
    <td><strong>59.52</strong></td>
    <td><strong>88.20</strong></td>
  </tr>
  <tr>
    <td>NVFP4</td>
    <td><strong>93.84</strong></td>
    <td><strong>82.06</strong></td>
    <td><strong>73.40</strong></td>
    <td><strong>81.79</strong></td>
    <td><strong>59.47</strong></td>
    <td><strong>87.78</strong></td>
  </tr>
</table>

> Baseline: [Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), evaluated with Temperature 1.0, top-p 0.95

### KVV Evaluation
We also evaluate our NVFP4 checkpoint on the [Kimi Vendor Verifier (KVV)](https://github.com/MoonshotAI/Kimi-Vendor-Verifier#k31-pro-evaluation-results) which verifies whether a vendor deployment meets Kimi model accuracy and API feature requirements. We use the same settings as provided in the repository by KVV for the thinking mode, repeats, temperature, harness, etc.
<table>
  <tr>
    <td><strong>Provdier</strong></td>
    <td><strong>OCRBench</strong></td>
    <td><strong>MMMU-Pro</strong></td>
    <td><strong>ToolCall</strong></td>
    <td><strong>BEAM (1M)</strong></td>
    <td><strong>DeepSWE</strong></td>
  </tr>
  <tr>
    <td>Moonshot</td>
    <td><strong>87.20</strong></td>
    <td><strong>84.10</strong></td>
    <td><strong>97.20</strong></td>
    <td><strong>39.20</strong></td>
    <td><strong>75.50</strong></td>
  </tr>
  <tr>
    <td>Nvidia NVFP4</td>
    <td><strong>86.70</strong></td>
    <td><strong>84.10</strong></td>
    <td><strong>97.40</strong></td>
    <td><strong>38.70</strong></td>
    <td><strong>77.40</strong></td>
  </tr>
</table>

Both the accuracy benchmarks and the KVV evaluations use the main KV/ indexer cache in fp8, and the SSM state is in fp32.


## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
