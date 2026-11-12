# MiniCPM5：libcusparseLt 映射失败与 2 GiB AS 的证据边界

当前 `/home/ubuntu/venvs/vllm_neuron` Python3.10 环境安装的是 CUDA build：源码常量分别为 torch **2.8.0+cu128**（CUDA12.8）、torchvision **0.23.0+cu128**、torchaudio **2.8.0+cu128**。这次只读 version.py/METADATA/RECORD，没有导入包。故障记录使用同一环境路径。

失败发生在 `torch/__init__.py:416` 的 `from torch._C import *`，错误为 `libcusparseLt.so.0: failed to map segment from shared object`。尚未运行模型。原异常的 `errno_present/errno_obtained/errno_inferred` 都是 false，根因没有被证实；不能写成已证实的 ENOMEM、物理内存不足或 Neuron 模型错误。

| R1 已保存观测 | import 前 | 失败后 |
|---|---:|---:|
| RLIMIT_AS | 2 GiB | 2 GiB |
| VmSize | 155.45 MiB | 354.14 MiB |
| VmPeak | 307.97 MiB | 1,659.80 MiB |
| VmRSS | 146.78 MiB | 196.68 MiB |
| Threads | 1 | 1 |

RLIMIT_AS 限制进程虚拟地址空间，RSS 是另一指标。2,048 MiB 减去已记录成功 VmPeak 约剩 388.20 MiB；这是与历史峰值的算术差，不能当成失败瞬间的可分配空间。失败的 mmap 可能不计入 VmPeak，而 loader 回滚会移除已经成功映射的内容，因此失败后的 VmSize 也不能重建失败瞬间。

已安装 RECORD 记载 libcusparseLt.so.0 文件为 451,980,513 B（约431.04 MiB），libtorch_cuda.so 为 1,015,364,433 B（约968.33 MiB），libtorch_cpu.so 为430,926,857 B（约410.96 MiB）。这些是包记录中的文件大小，本次没有读取相应 ELF 或验证当前 payload。文件大小不等于 PT_LOAD 内存跨度，更不等于同一时刻成功映射的内存总和。CUDA 依赖与 AS 压力形成值得检查的候选解释，但目前没有 syscall errno、mmap 长度/地址或该库 PT_LOAD 明细作因果证明。

R1 输出只有 admission、owner、summary、exit 四文件，没有保存完整 maps 或 mmap 轨迹。本次在该输出以及 run root/research 顶层相关 JSON/log/txt 中未找到这次故障的每段映射证据；这不构成全盘不存在任何相关资料的证明。

## 原环境依赖与可验证的后继路线

| 已安装包 | 元数据约束 |
|---|---|
| torch-neuronx 2.8.0.2.12.22436+0f1dac25 | torch==2.8.*、torch-xla==2.8.*、libneuronxla>=2.2,<2.3、protobuf>3.19、numpy、psutil |
| neuronx-distributed 0.17.26814+4b18de63 | torch-neuronx、torch-xla、safetensors、tenacity、setuptools>=70.1 |
| NxDI 0.8.16251+f3ca5575 | torch_neuronx>=2.5、neuronx_distributed、transformers==4.57.*、torchvision 等 |
| vllm-neuron 0.2.2+lts | vllm==0.11.0、torch-neuronx>=2.7.0、neuronx-cc>=2.0.0a0、torchvision |
| vllm 0.11.0 | torch==2.8.0、torchaudio==2.8.0、torchvision==0.23.0 |

当前 torch wheel 本身声明 nvidia-cusparselt-cu12==0.7.1 以及其他 CUDA12 依赖。上面 Neuron/vLLM 包的版本约束没有显式指定 +cu128。可以准备独立环境候选，保持 Python3.10/对应 Neuron 版本，验证 torch2.8.0+cpu、torchvision0.23.0+cpu、torchaudio2.8.0+cpu 的完整官方来源及二进制兼容性，以去掉这条文本推理诊断中可能不必要的 CUDA 导入负担。版本元数据可匹配不代表 torch-xla/torch-neuronx/libneuronxla ABI 已验证。

候选必须另建完整 source/native identity、spec/review 和新 baseline 标签，不可直接替换旧环境继续复用历史凭据。先在相同 2 GiB 与其余原限额下完成 import-only 诊断，再过原整模型数值、冷缓存、长解码、槽位复用和性能门槛。模型/revision、分词器、BF16 等精度、TP/布局与科学阈值都不因此改变。此处仅提出路线，没有下载、安装、改环境或执行模型。

本地当前 upstream 源码属于不同版本：NxDI setup.py 已要求 torch_neuronx==2.9.*，vllm-neuron 当前 setup guide 描述 Neuron2.31+、Python3.11+、Trn2/Trn3 与 libtorch-neuronx-lite 路线。不能把这些新主线说明直接当作现有0.2.2 LTS/torch2.8 栈的替换依据，也不能把版本差异误报为 repo 缺陷。
