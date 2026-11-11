---
license: other
license_name: nvidia-open-model-license
license_link: https://huggingface.co/nvidia/Metatron-v1/blob/main/LICENSE.pdf
language:
- en
library_name: protomotions
tags:
- robotics
- reinforcement-learning
- humanoid
- onnx
- metatron
---

# Metatron v1

## Choose an artifact

| Package | Use | Documentation |
| --- | --- | --- |
| [G1/](G1/) | Native PyTorch G1 inference: checkpoint, resolved config, reconstruction recipe and legal files | [G1 model card](G1/MODEL_CARD.md), [download and run](G1/README.md) |
| `demo/model.onnx` | Single-robot browser scene/recovery pipeline | Browser export card below |
| `racer/network.onnx` | Six-robot batched G1 Racer network | Browser export card below |

The native checkpoint is 459,455,983 bytes (SHA256
`e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a`).
It retains the posterior and normalization statistics; the deterministic browser
graphs export the prior/decoder path. Interfaces are not interchangeable.
The native recipe masks scene inputs; see its card before re-exporting.

`config.json` is a machine-readable artifact index with file sizes and SHA256
digests, consumed by ProtoMotions' pinned native downloader. Model files are
hosted here, not served by GitHub Pages. The existing browser paths are unchanged.

## License

Model license: NVIDIA Open Model License. The [full agreement](LICENSE.pdf) and
[attribution notice](NOTICE.txt) govern model artifacts, including native weights
and serialized configurations. Source code remains Apache-2.0. Dataset and asset
rights are separate; this repository does not redistribute training motion clips.

- Deployment Geography: Global.
- Use Case: Researchers and developers running interactive humanoid control in a
  browser; the separate native G1 package also supports simulator and supervised
  real-G1 deployment workflows.
- Release: Metatron v1, hosted in this repository and integrated with
  [ProtoMotions](https://github.com/NVlabs/ProtoMotions). Pin a Hub revision; no
  public ProtoMotions release tag is asserted here.

# Browser export model card

## Overview

ONNX inference exports of the unified Metatron G1 scene/recovery policy used by the
Metatron browser demo and G1 Racer. Both files use the same learned policy
weights, with different runtime interfaces. They are **not interchangeable**.
The current exports are the epoch-6410 SAMP-capable unified student. Walking,
scene interaction and recovery share one policy, with no separate recovery adapter.

| Field | Value |
| --- | --- |
| Robot | Unitree G1, 29 controlled joints |
| Model | MaskedMimic-style transformer student with deterministic prior and decoder |
| Inference | Deterministic learned prior; FP32 |
| Format | ONNX |
| Integration | ONNX Runtime Web with WebGPU or WebAssembly |

## Intended Use

Interactive humanoid control, locomotion and recovery evaluation with the
matching browser applications. The applications provide physics, observations,
task goals and control logic; the ONNX files alone are not complete demos.

Metatron supports deployment workflows beyond the browser, including other
simulators and real G1 robots. These particular browser exports require their
matching observation and control interfaces: compatibility is not a safety
certification or a claim that these exact files have been validated on hardware.

## Artifacts

| File | Interface | Size |
| --- | --- | --- |
| `demo/model.onnx` | Single-robot pipeline, including observation processing and action conversion | 403,375,789 bytes |
| `racer/network.onnx` | Six-robot batched network; requires G1 Racer's separate observation processing | 403,036,698 bytes |

SHA256 digests:

- `demo/model.onnx`: `7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4`
- `racer/network.onnx`: `130a22718a34c0da488e2157553b40efedfefcf6ca6cb6b97cfdd67f8dfbeaf3`

The Racer export changes graph organization and batching, not learned weights.
Neither artifact is a resumable training checkpoint. Runtime code, robot assets,
motion clips and application configuration are distributed separately.

## Inputs and Outputs

The demo pipeline consumes current G1 state, eight historical state samples,
paired incoming raw actions, nine-entry raw-action history, masked future
pose/style/velocity goals and scene-box context (39 inputs). There is no
`recovery_enabled` input. It returns joint PD targets (`action`),
PD gains (`stiffness`, `damping`) and `raw_action` for history feedback.
Recovery targets remain live runtime inputs, not baked goals. Applications use
staged local goals with independent per-robot recovery state.

G1 Racer computes observations outside the neural graph and evaluates all six
robots in one batch. Its network exposes `action` and `raw_action`; controller
parameters come from the paired runtime. Do not feed raw demo inputs directly
to this network or change its batch size without rebuilding and validating it.

Preserve the paired application's joint/body ordering, quaternion convention,
history ordering, normalization, target masks, control rate and action mapping.
Use the matching runtime manifest and ONNX tensor signatures for exact layouts.

## Runtime Support and Validation

The browser integrations use MuJoCo WASM for simulation and ONNX Runtime Web for
inference. WebGPU is recommended on capable devices; CPU/WASM is a fallback and
does not guarantee real-time six-robot operation. GPU drivers, browsers and
hardware affect startup time and throughput.

The six-robot ONNX and compiled-observation paths were checked against the
single-robot demo export on 60 distinct/history-perturbed test states, with
maximum absolute action error 5.55e-6. Recovery height, recovery timing, action
history and scene inputs were independently varied and affected the outputs.
These are numerical export checks, not task-success or real-robot benchmarks.
Racer's paired application also runs action-parity fixtures at startup.

For reproducibility, pin downloads to a repository commit rather than `main`.
Public browser delivery requires anonymous access to the files; never embed a
Hugging Face access token in client-side code.

## Limitations and Safety

- Recovery, balance and tracking can fail, especially outside the evaluated
  states, terrain and command distribution.
- Game launch assists, low-gravity zones and respawns are application mechanics,
  not capabilities guaranteed by the learned policy or physical robot.
- The model is not a collision-avoidance system or a safety controller.
- Hardware use requires independent validation, appropriate joint/torque limits,
  supervised testing, and an emergency-stop mechanism.
- These exports are not a complete training-reproduction package. No training
  dataset or training checkpoint is included.

## License and Asset Rights

Model terms: [NVIDIA Open Model License Agreement](LICENSE.pdf), with the
required attribution in [NOTICE.txt](NOTICE.txt). The complete agreement
governs; this card does not replace or extend it.

Model licensing does not grant rights to motion datasets, recovery reference
clips, robot meshes, game assets or third-party runtime software. None of those
assets is included in this model repository. Obtain them through their
respective distributions and follow their separate terms. Permission to include
a clip inside a demo or game does not imply permission to redistribute that
clip as a standalone dataset.

These ONNX exports are publicly available for anonymous download. Their
availability does not change the separate distribution terms or release status
of the associated application bundles, motion clips and other assets.

## References

- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- [ProtoMotions documentation](https://nvlabs.github.io/ProtoMotions/)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
